import os
from dotenv import load_dotenv

# Carga las variables de entorno definidas en .env (solo en desarrollo)
load_dotenv()

import logging
import json
import re
import requests
import random
import string
import datetime
import unicodedata
import difflib            # para detección aproximada de palabras
from PIL import Image, UnidentifiedImageError
import imagehash

from telegram import (
    Update,
    ReplyKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardRemove,
    InputMediaPhoto,
)
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from openai import AsyncOpenAI

# Inicializa logging en DEBUG para capturar errores
logging.basicConfig(level=logging.DEBUG)

# ——— Variables de entorno ——————————————————————————————————————————————
TOKEN                  = os.environ["TOKEN_TELEGRAM"]
OPENAI_API_KEY         = os.environ["OPENAI_API_KEY"]
NOMBRE_NEGOCIO         = os.environ.get("NOMBRE_NEGOCIO", "Tienda Tenis")
URL_SHEETS_INVENTARIO  = os.environ["URL_SHEETS_INVENTARIO"]
URL_SHEETS_PEDIDOS     = os.environ["URL_SHEETS_PEDIDOS"]
CARPETA_IMAGENES       = os.environ["CARPETA_IMAGENES"]
EMAIL_DEVOLUCIONES     = os.environ["EMAIL_DEVOLUCIONES"]
EMAIL_JEFE             = os.environ["EMAIL_JEFE"]
# Si usas SMTP para envíos de correo, también:
SMTP_SERVER            = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT              = int(os.environ.get("SMTP_PORT", 587))
EMAIL_REMITENTE        = os.environ.get("EMAIL_REMITENTE")
EMAIL_PASSWORD         = os.environ.get("EMAIL_PASSWORD")


# ——— CONSTANTES DE MENSAJES —————————————————————————————————————————
WELCOME_TEXT = (
    f"¡Bienvenido a {NOMBRE_NEGOCIO}!\n\n"
    "¿Qué te gustaría hacer hoy?\n"
    "– ¿Tienes alguna marca en mente?\n"
    "– ¿Puedes enviarme la foto del que quieres ordenar?\n"
    "– ¿Te gustaría ver el catálogo para que te antojes?\n"
    "– ¿Te gustaría rastrear tu pedido?\n"
    "– ¿Te gustaría realizar un cambio?\n"
    "Cuéntame sin ningún problema 😀"
)
CLIP_INSTRUCTIONS = (
    "¡Claro! Para enviarme una imagen, pulsa el ícono de clip (📎) en la barra de chat, "
    "selecciona “Galería” o “Archivo” y elige la foto que quieras enviar. Cuando la subas, "
    "la revisaré y te ayudo con lo que necesites."
)
CATALOG_LINK    = "https://shoopsneakers.com/categoria-producto/hombre/"
CATALOG_MESSAGE = f"Aquí tienes el catálogo: {CATALOG_LINK}"

# ——— SETUP ————————————————————————————————————————————————————————
logging.basicConfig(level=logging.INFO)
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# ——— EMAIL STUBS ————————————————————————————————————————————————————
def enviar_correo(dest, subj, body):
    logging.info(f"[EMAIL STUB] To: {dest}\nSubject: {subj}\n{body}")

def enviar_correo_con_adjunto(dest, subj, body, adj):
    logging.info(f"[EMAIL STUB] To: {dest}\nSubject: {subj}\n{body}\n[Adj: {adj}]")

# ——— GOOGLE SHEETS UTILS ————————————————————————————————————————
# ——— GOOGLE SHEETS UTILS ————————————————————————————————————————
# ——— GOOGLE SHEETS UTILS ————————————————————————————————————————
def registrar_orden(data: dict):
    # Mapear a snake_case para doPost(e)
    payload = {
        "numero_venta": data.get("Número Venta", ""),
        "fecha_venta":  data.get("Fecha Venta", ""),
        "cliente":      data.get("Cliente", ""),
        "telefono":     data.get("Teléfono", ""),
        "producto":     data.get("Producto", ""),
        "color":        data.get("Color", ""),
        "talla":        data.get("Talla", ""),
        "correo":       data.get("Correo", ""),
        "pago":         data.get("Pago", ""),
        "estado":       data.get("Estado", "")
    }
    logging.info(f"[SHEETS] Payload JSON que envío:\n{payload}")
    try:
        resp = requests.post(URL_SHEETS_PEDIDOS, json=payload)
        logging.info(f"[SHEETS] HTTP {resp.status_code} — Body: {resp.text}")
    except Exception as e:
        logging.error(f"[SHEETS] Error al hacer POST: {e}")

def generate_sale_id() -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    rnd = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
    return f"VEN-{ts}-{rnd}"

# ——— CACHE DE HASHES —————————————————————————————————————————————
EXTENSIONES_IMAGEN = ('.jpg', '.jpeg', '.png')
PRECALC_HASHES: dict[str, imagehash.ImageHash] = {}

def precargar_hashes():
    total = 0
    for fn in os.listdir(CARPETA_IMAGENES):
        if fn.lower().endswith(EXTENSIONES_IMAGEN):
            try:
                PRECALC_HASHES[fn] = imagehash.phash(Image.open(os.path.join(CARPETA_IMAGENES, fn)))
                total += 1
            except:
                pass
    logging.info(f"Hashes precargados: {total}")

precargar_hashes()

PHASH_THRESHOLD = 15
AHASH_THRESHOLD  = 12

def compare_with_local_images(path: str) -> str | None:
    try:
        img_up = Image.open(path)
    except Exception as e:
        logging.error(f"No pude leer la imagen subida: {e}")
        return None
    ph_up = imagehash.phash(img_up)
    ah_up = imagehash.average_hash(img_up)
    for fn, ph_local in PRECALC_HASHES.items():
        local_path = os.path.join(CARPETA_IMAGENES, fn)
        try:
            img_loc = Image.open(local_path)
        except:
            continue
        ah_loc = imagehash.average_hash(img_loc)
        if abs(ph_up - ph_local) <= PHASH_THRESHOLD and abs(ah_up - ah_loc) <= AHASH_THRESHOLD:
            return os.path.splitext(fn)[0]
    return None

# ——— UTILIDADES —————————————————————————————————————————————————
estado_usuario: dict[int, dict] = {}
inventario_cache = None

def normalize(text: str) -> str:
    t = unicodedata.normalize('NFKD', text.strip().lower())
    return ''.join(ch for ch in t if not unicodedata.combining(ch))

def reset_estado(cid: int):
    estado_usuario[cid] = {
        "fase": "inicio",
        "marca": None, "modelo": None, "color": None, "talla": None,
        "nombre": None, "correo": None, "telefono": None,
        "ciudad": None, "provincia": None, "direccion": None,
        "referencia": None, "resumen": None, "sale_id": None
    }

def menu_botones(opts: list[str]):
    return ReplyKeyboardMarkup([[KeyboardButton(o)] for o in opts], resize_keyboard=True)

def obtener_inventario() -> list[dict]:
    global inventario_cache
    if inventario_cache is None:
        try:
            inventario_cache = requests.get(URL_SHEETS_INVENTARIO).json()
        except:
            inventario_cache = []
    return inventario_cache

def disponible(item: dict) -> bool:
    return normalize(item.get("stock","")) == "si"

def obtener_marcas_unicas(inv: list[dict]) -> list[str]:
    return sorted({i.get("marca","").strip() for i in inv if disponible(i)})

def obtener_modelos_por_marca(inv: list[dict], marca: str) -> list[str]:
    return sorted({i.get("modelo","").strip()
                   for i in inv
                   if normalize(i.get("marca","")) == normalize(marca) and disponible(i)})

def obtener_colores_por_modelo(inv: list[dict], marca: str, modelo: str) -> list[str]:
    return sorted({i.get("color","").strip()
                   for i in inv
                   if normalize(i.get("marca","")) == normalize(marca)
                   and normalize(i.get("modelo","")) == normalize(modelo)
                   and disponible(i)})

def obtener_tallas_por_color(inv: list[dict], marca: str, modelo: str, color: str) -> list[str]:
    return sorted({str(i.get("talla","")).strip()
                   for i in inv
                   if normalize(i.get("marca","")) == normalize(marca)
                   and normalize(i.get("modelo","")) == normalize(modelo)
                   and normalize(i.get("color","")) == normalize(color)
                   and disponible(i)})

# ——— HANDLERS —————————————————————————————————————————————————

async def saludo_bienvenida(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        WELCOME_TEXT,
        reply_markup=menu_botones([
            "Hacer pedido", "Enviar imagen", "Ver catálogo", "Rastrear pedido", "Realizar cambio"
        ])
    )

async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    cid = update.effective_chat.id
    reset_estado(cid)
    await saludo_bienvenida(update, ctx)
    estado_usuario[cid]["fase"] = "esperando_comando"

async def responder(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    cid = update.effective_chat.id
    if cid not in estado_usuario:
        reset_estado(cid)
    est = estado_usuario[cid]
    inv = obtener_inventario()
    txt_raw = update.message.text or ""
    txt = normalize(txt_raw)

    # Fase inicio
    if est["fase"] == "inicio":
        await saludo_bienvenida(update, ctx)
        est["fase"] = "esperando_comando"
        return

    # Esperando comando principal
    if est["fase"] == "esperando_comando":
        if txt in ("menu", "inicio"):
            reset_estado(cid)
            await saludo_bienvenida(update, ctx)
            return
        if "rastrear" in txt:
            est["fase"] = "esperando_numero_rastreo"
            await update.message.reply_text("Perfecto, envíame el número de venta.", reply_markup=ReplyKeyboardRemove())
            return
        if any(k in txt for k in ("cambio", "reembol", "devolucion")):
            est["fase"] = "esperando_numero_devolucion"
            await update.message.reply_text("Envíame el número de venta para realizar la devolución del pedido.", reply_markup=ReplyKeyboardRemove())
            return
        if re.search(r"\b(catálogo|catalogo)\b", txt):
            reset_estado(cid)
            await update.message.reply_text(CATALOG_MESSAGE, reply_markup=menu_botones([
                "Hacer pedido", "Enviar imagen", "Ver catálogo", "Rastrear pedido", "Realizar cambio"
            ]))
            return
        if "imagen" in txt or "foto" in txt:
            est["fase"] = "esperando_imagen"
            await update.message.reply_text(CLIP_INSTRUCTIONS, reply_markup=ReplyKeyboardRemove())
            return
    # dentro de responder(), en el nivel de la función (4 espacios)
    if est["fase"] == "esperando_comando":
        # … tus otras comprobaciones aquí (8 espacios) …
        if "imagen" in txt or "foto" in txt:
            est["fase"] = "esperando_imagen"
            await update.message.reply_text(CLIP_INSTRUCTIONS, reply_markup=ReplyKeyboardRemove())
            return

        # detección de marca por tokens (8 espacios)
    if est["fase"] == "esperando_comando":
        # … tus otras comprobaciones …

        #     Detección aproximada de marca con difflib ——————————————————
        palabras_usuario = txt.split()  # tokens del mensaje del usuario
        for m in obtener_marcas_unicas(inv):
            tokens_marca = normalize(m).split()  # tokens de la marca
            for tok in tokens_marca:
                # busca coincidencias cercanas con cutoff=0.6
                match = difflib.get_close_matches(tok, palabras_usuario, n=1, cutoff=0.6)
                if match:
                    est["marca"] = m
                    est["fase"]  = "esperando_modelo"
                    await update.message.reply_text(
                        f"¡Genial! Veo que buscas {m}. ¿Qué modelo de {m} te interesa?",
                        reply_markup=menu_botones(obtener_modelos_por_marca(inv, m))
                    )
                    return


        # fallback de comando (8 espacios)
        await update.message.reply_text(
            "No entendí tu elección. …"
        )
        return
        await update.message.reply_text("No entendí, elige del menú o escribe 'Ver catálogo', 'Enviar imagen', 'Rastrear pedido' o una marca.")
        return

    # Rastrear pedido
    if est["fase"] == "esperando_numero_rastreo":
        await update.message.reply_text("Aquí tu guía para rastrear pedido: https://www.instagram.com/juanp_ocampo/")
        reset_estado(cid)
        return

    # Realizar cambio / devolución
    if est["fase"] == "esperando_numero_devolucion":
        est["referencia"] = txt_raw.strip()
        est["fase"] = "esperando_motivo_devolucion"
        await update.message.reply_text("¿Por qué solicitas la devolución? Por talla, defecto, etc.")
        return
    if est["fase"] == "esperando_motivo_devolucion":
        mot = txt_raw.strip()
        enviar_correo(EMAIL_DEVOLUCIONES, f"Devolución {NOMBRE_NEGOCIO}", f"Venta: {est['referencia']}\nMotivo: {mot}")
        await update.message.reply_text("Solicitud enviada. Te contactamos pronto.")
        reset_estado(cid)
        return

    # Envío de imagen para reconocer
    if est["fase"] == "esperando_imagen" and update.message.photo:
        f = await update.message.photo[-1].get_file()
        tmp = os.path.join("temp", f"{cid}.jpg")
        os.makedirs("temp", exist_ok=True)
        await f.download_to_drive(tmp)
        ref = compare_with_local_images(tmp)
        os.remove(tmp)
        if ref:
            marca, modelo, *rest = ref.split('_')
            color = rest[0] if rest else None
            est.update({"marca": marca, "modelo": modelo, "color": color, "fase": "imagen_detectada"})
            await update.message.reply_text(f"Es {marca} {modelo} color {color}. ¿Continuar? (SI/NO)",
                                            reply_markup=menu_botones(["SI", "NO"]))
        else:
            reset_estado(cid)
            await update.message.reply_text("No reconocí. Empieza de nuevo con /start.")
        return

    # Confirmación imagen detectada
    if est["fase"] == "imagen_detectada":
        if txt in ("si", "s"):
            est["fase"] = "esperando_talla"
            await update.message.reply_text("¿Qué talla deseas?",
                                            reply_markup=menu_botones(obtener_tallas_por_color(inv, est["marca"], est["modelo"], est["color"])))
        else:
            await update.message.reply_text("Cancelado. /start para reiniciar.")
            reset_estado(cid)
        return

    # Selección de modelo/color/talla
    if est["fase"] == "esperando_modelo":
        modelos = obtener_modelos_por_marca(inv, est["marca"])
        if txt in map(normalize, modelos):
            est["modelo"] = next(m for m in modelos if normalize(m)==txt)
            est["fase"] = "esperando_color"
            await update.message.reply_text("¿Qué color prefieres?",
                                            reply_markup=menu_botones(obtener_colores_por_modelo(inv, est["marca"], est["modelo"])))
        else:
            await update.message.reply_text("Elige un modelo válido.", reply_markup=menu_botones(modelos))
        return

    if est["fase"] == "esperando_color":
        colores = obtener_colores_por_modelo(inv, est["marca"], est["modelo"])
        if txt in map(normalize, colores):
            est["color"] = next(c for c in colores if normalize(c)==txt)
            est["fase"] = "esperando_talla"
            await update.message.reply_text("¿Qué talla deseas?",
                                            reply_markup=menu_botones(obtener_tallas_por_color(inv, est["marca"], est["modelo"], est["color"])))
        else:
            await update.message.reply_text("Elige un color válido.", reply_markup=menu_botones(colores))
        return

    if est["fase"] == "esperando_talla":
        tallas = obtener_tallas_por_color(inv, est["marca"], est["modelo"], est["color"])
        if txt in map(normalize, tallas):
            est["talla"] = next(t for t in tallas if normalize(t)==txt)
            est["fase"] = "esperando_nombre"
            await update.message.reply_text("¿Cuál es tu nombre?")
        else:
            await update.message.reply_text("Elige una talla válida.", reply_markup=menu_botones(tallas))
        return

    # Recopilación de datos y pago
    if est["fase"] == "esperando_nombre":
        est["nombre"] = txt_raw
        est["fase"] = "esperando_correo"
        await update.message.reply_text("¿Tu correo electrónico? 📧")
        return
    if est["fase"] == "esperando_correo":
        if re.match(r"[^@]+@[^@]+\.[^@]+", txt_raw):
            est["correo"] = txt_raw
            est["fase"] = "esperando_telefono"
            await update.message.reply_text("¿Tu teléfono? 📱")
        else:
            await update.message.reply_text("Correo inválido.")
        return
    if est["fase"] == "esperando_telefono":
        if re.match(r"^\+?\d{7,15}$", txt_raw):
            est["telefono"] = txt_raw
            est["fase"] = "esperando_ciudad"
            await update.message.reply_text("¿Ciudad?")
        else:
            await update.message.reply_text("Teléfono inválido.")
        return
    if est["fase"] == "esperando_ciudad":
        est["ciudad"] = txt_raw
        est["fase"] = "esperando_provincia"
        await update.message.reply_text("¿Provincia/departamento?")
        return
    if est["fase"] == "esperando_provincia":
        est["provincia"] = txt_raw
        est["fase"] = "esperando_direccion"
        await update.message.reply_text("¿Dirección de envío?")
        return
    if est["fase"] == "esperando_direccion":
        est["direccion"] = txt_raw
        precio = next((i["precio"] for i in inv
                       if normalize(i["marca"])==normalize(est["marca"])
                       and normalize(i["modelo"])==normalize(est["modelo"])
                       and normalize(i["color"])==normalize(est["color"])), "N/A")
        sale_id = generate_sale_id()
        est["sale_id"] = sale_id
        resumen = {
            "Número Venta": sale_id,
            "Fecha Venta": datetime.datetime.now().isoformat(),
            "Cliente": est["nombre"],
            "Teléfono": est["telefono"],
            "Producto": f"{est['marca']} {est['modelo']}",
            "Color": est["color"],
            "Talla": est["talla"],
            "Correo": est["correo"],
            "Pago": None,
            "Estado": "PENDIENTE"
        }
        est["resumen"] = resumen
        text_res = (
            f"✅ Pedido: {sale_id}\n"
            f"👤 {est['nombre']}  📧 {est['correo']}  📲 {est['telefono']}\n"
            f"🏠 {est['direccion']}, {est['ciudad']}, {est['provincia']}\n"
            f"👟 {est['marca']} {est['modelo']} color {est['color']} talla {est['talla']}\n"
            f"💰 {precio}\n\n"
            "Elige método de pago: TRANSFERENCIA, QR o CONTRA ENTREGA"
        )
        est["fase"] = "esperando_pago"
        await update.message.reply_text(text_res, reply_markup=menu_botones(["TRANSFERENCIA","QR","CONTRA ENTREGA"]))
        return

    # Manejo pago
    if est["fase"] == "esperando_pago":
        opt = normalize(txt_raw.replace(" ", ""))
        resumen = est["resumen"]
        if opt == "transferencia":
            est["fase"] = "esperando_comprobante"
            resumen["Pago"] = "Transferencia"
            await update.message.reply_text("Envía la foto de tu comprobante.")
        elif opt == "qr":
            est["fase"] = "esperando_comprobante"
            resumen["Pago"] = "QR"
            await update.message.reply_text("Escanea el QR y envía la foto.")
        elif opt=="contraentrega":
            resumen["Pago"] = "Contra entrega"
            registrar_orden(resumen)
            await update.message.reply_text("Pedido registrado para contra entrega. ¡Gracias!")
            reset_estado(cid)
        else:
            await update.message.reply_text("Método inválido.")
        return

    # Recibo comprobante
    if est["fase"] == "esperando_comprobante" and update.message.photo:
        f = await update.message.photo[-1].get_file()
        tmp = os.path.join("temp", f"{cid}_proof.jpg")
        os.makedirs("temp", exist_ok=True)
        await f.download_to_drive(tmp)
        resumen = est["resumen"]
        registrar_orden(resumen)
        enviar_correo(est["correo"], f"Pago recibido {resumen['Número Venta']}", json.dumps(resumen, indent=2))
        enviar_correo_con_adjunto(EMAIL_JEFE, f"Comprobante {resumen['Número Venta']}", json.dumps(resumen, indent=2), tmp)
        os.remove(tmp)
        await update.message.reply_text("✅ ¡Pago registrado! Tu pedido está en proceso.")
        reset_estado(cid)
        return

    # Fallback
    await update.message.reply_text("No entendí. Usa /start para reiniciar.")

def main():
    # 1) Construye la aplicación
    app = ApplicationBuilder().token(TOKEN).build()

    # 2) Registra tus handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT | filters.PHOTO, responder))

    # 3) Manejador de errores para capturar excepciones internas
    async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
        logging.error("❌ Excepción procesando update:", exc_info=context.error)
    app.add_error_handler(error_handler)

    # 4) Lee variables de entorno para ngrok
    port = int(os.environ.get("PORT", 8443))
    ngrok_host = os.environ.get("NGROK_HOSTNAME")
    if not ngrok_host:
        logging.error("❌ Debes exportar NGROK_HOSTNAME antes de arrancar")
        return

    # 5) Configura y arranca el webhook
    logging.info(f"🚀 Configurando webhook_url en: https://{ngrok_host}/{TOKEN}")
    app.run_webhook(
        listen="0.0.0.0",
        port=port,
        url_path=TOKEN,
        webhook_url=f"https://{ngrok_host}/{TOKEN}"
    )

if __name__ == "__main__":
    main()
