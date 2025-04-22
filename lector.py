import os
import io
import logging
from dotenv import load_dotenv
load_dotenv()

import imagehash
from PIL import Image
from openai import AsyncOpenAI
import json
import re
import requests
import random
import string
import datetime
import unicodedata
import difflib

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from telegram import (
    Update,
    ReplyKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardRemove,
)
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# Carga la credencial desde el secret de Render
creds_info = json.loads(os.environ["GOOGLE_CREDS_JSON"])
creds = service_account.Credentials.from_service_account_info(
    creds_info,
    scopes=["https://www.googleapis.com/auth/drive.readonly"],
)
drive_service = build("drive", "v3", credentials=creds)

DRIVE_FOLDER_ID = os.environ["DRIVE_FOLDER_ID"]

PHASH_THRESHOLD = 15
AHASH_THRESHOLD = 12

def precargar_hashes_from_drive(folder_id: str) -> dict[str, list[tuple[imagehash.ImageHash, imagehash.ImageHash]]]:
    """
    Descarga todas las imágenes de la carpeta de Drive, extrae SKU=modelo_color,
    calcula phash/ahash y agrupa por SKU.
    """
    model_hashes: dict[str, list[tuple[imagehash.ImageHash, imagehash.ImageHash]]] = {}
    page_token = None

    while True:
        resp = drive_service.files().list(
            q=f"'{folder_id}' in parents and mimeType contains 'image/'",
            spaces="drive",
            fields="nextPageToken, files(id, name)",
            pageToken=page_token
        ).execute()

        for f in resp.get("files", []):
            fid, name = f["id"], f["name"]
            # 1) Construye el SKU usando las 3 primeras partes del nombre (sin extensión)
            base = os.path.splitext(name)[0]       # ej: "DS_277_TATATA_1"
            parts = base.split('_')                # ["DS","277","TATATA","1"]
            sku = parts[0]
            if len(parts) > 1:
                sku += "_" + parts[1]              # e.g. "DS_277"
            if len(parts) > 2:
                sku += "_" + parts[2]              # e.g. "DS_277_TATATA"

            # 2) Descarga la imagen
            request = drive_service.files().get_media(fileId=fid)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            fh.seek(0)

            # 3) Calcula hashes y agrúpalos bajo ese SKU
            try:
                img = Image.open(fh)
                ph = imagehash.phash(img)
                ah = imagehash.average_hash(img)
                model_hashes.setdefault(sku, []).append((ph, ah))
            except Exception as e:
                logging.error(f"No pude procesar {name}: {e}")

        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    logging.info(f"▶ Precargados hashes para {len(model_hashes)} SKUs")
    return model_hashes

MODEL_HASHES = precargar_hashes_from_drive(DRIVE_FOLDER_ID)

def identify_model_from_stream(path: str) -> str | None:
    try:
        img_up = Image.open(path)
    except Exception as e:
        logging.error(f"No pude leer la imagen subida: {e}")
        return None

    ph_up = imagehash.phash(img_up)
    ah_up = imagehash.average_hash(img_up)

    for model, refs in MODEL_HASHES.items():
        for ph_ref, ah_ref in refs:
            if abs(ph_up - ph_ref) <= PHASH_THRESHOLD and abs(ah_up - ah_ref) <= AHASH_THRESHOLD:
                return model
    return None

# ——— VARIABLES DE ENTORNO ——————————————————————————————————————————————
TOKEN                 = os.environ["TOKEN_TELEGRAM"]
OPENAI_API_KEY        = os.environ["OPENAI_API_KEY"]
NOMBRE_NEGOCIO        = os.environ.get("NOMBRE_NEGOCIO", "Tienda Tenis")
URL_SHEETS_INVENTARIO = os.environ["URL_SHEETS_INVENTARIO"]
URL_SHEETS_PEDIDOS    = os.environ["URL_SHEETS_PEDIDOS"]
EMAIL_DEVOLUCIONES    = os.environ["EMAIL_DEVOLUCIONES"]
EMAIL_JEFE            = os.environ["EMAIL_JEFE"]
SMTP_SERVER           = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT             = int(os.environ.get("SMTP_PORT", 587))
EMAIL_REMITENTE       = os.environ.get("EMAIL_REMITENTE")
EMAIL_PASSWORD        = os.environ.get("EMAIL_PASSWORD")

WELCOME_TEXT = (
    f"¡Bienvenido a {NOMBRE_NEGOCIO}!\n\n"
    "¿Qué te gustaría hacer hoy?\n"
    "– ¿Tienes alguna marca en mente?\n"
    "– ¿Puedes enviarme la foto del que quieres ordenar?\n"
    "– ¿Te gustaría ver el catálogo?\n"
    "– ¿Te gustaría rastrear tu pedido?\n"
    "– ¿Te gustaría realizar un cambio?\n"
    "Cuéntame sin ningún problema 😀"
)
CLIP_INSTRUCTIONS = (
    "Para enviarme una imagen, pulsa el ícono de clip (📎), "
    "selecciona “Galería” o “Archivo” y elige la foto."
)
CATALOG_LINK    = "https://shoopsneakers.com/categoria-producto/hombre/"
CATALOG_MESSAGE = f"Aquí tienes el catálogo: {CATALOG_LINK}"

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

def enviar_correo(dest, subj, body):
    logging.info(f"[EMAIL STUB] To: {dest}\nSubject: {subj}\n{body}")

def enviar_correo_con_adjunto(dest, subj, body, adj):
    logging.info(f"[EMAIL STUB] To: {dest}\nSubject: {subj}\n{body}\n[Adj: {adj}]")

# ——— UTILIDADES DE INVENTARIO —————————————————————————————————————————
estado_usuario: dict[int, dict] = {}
inventario_cache = None

def normalize(text) -> str:
    # Asegura que text sea cadena y no None
    s = "" if text is None else str(text)
    t = unicodedata.normalize('NFKD', s.strip().lower())
    return "".join(ch for ch in t if not unicodedata.combining(ch))

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
                   if normalize(i.get("marca"))==normalize(marca)
                   and normalize(i.get("modelo"))==normalize(modelo)
                   and normalize(i.get("color"))==normalize(color)
                   and disponible(i)})

def registrar_orden(data: dict):
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

# ——— HANDLERS —————————————————————————————————————————————————————

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

    # Esperando comando
    if est["fase"] == "inicio":
        await saludo_bienvenida(update, ctx)
        est["fase"] = "esperando_comando"
        return

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

        # ——— Detección aproximada de marca (esta es la clave) ——————————————————
        palabras_usuario = txt.split()
        for m in obtener_marcas_unicas(inv):
            tokens_marca = normalize(m).split()
            for tok in tokens_marca:
                match = difflib.get_close_matches(tok, palabras_usuario, n=1, cutoff=0.6)
                if match:
                    est["marca"] = m
                    est["fase"] = "esperando_modelo"
                    await update.message.reply_text(
                        f"¡Genial! Veo que buscas {m}. ¿Qué modelo de {m} te interesa?",
                        reply_markup=menu_botones(obtener_modelos_por_marca(inv, m))
                    )
                    return

        # ——— Fallback si no detectó nada ————————————————————————————————
        await update.message.reply_text("No entendí tu elección. Usa /start para volver al menú.")
        return

    # Rastrear pedido
    if est["fase"] == "esperando_numero_rastreo":
        await update.message.reply_text("Guía para rastrear: https://www.instagram.com/juanp_ocampo/")
        reset_estado(cid)
        return

    # Devolución
    if est["fase"] == "esperando_numero_devolucion":
        est["referencia"] = txt_raw.strip()
        est["fase"] = "esperando_motivo_devolucion"
        await update.message.reply_text("Motivo de devolución:")
        return
    if est["fase"] == "esperando_motivo_devolucion":
        enviar_correo(EMAIL_DEVOLUCIONES, f"Devolución {NOMBRE_NEGOCIO}", f"Venta: {est['referencia']}\nMotivo: {txt_raw}")
        await update.message.reply_text("Solicitud enviada.")
        reset_estado(cid)
        return

    # Imagen enviada
    # Imagen enviada
    if est["fase"] == "esperando_imagen" and update.message.photo:
        # 1) Descarga la foto a disco
        f = await update.message.photo[-1].get_file()
        tmp = os.path.join("temp", f"{cid}.jpg")
        os.makedirs("temp", exist_ok=True)
        await f.download_to_drive(tmp)

        # 2) Identifica modelo desde Drive
        ref = identify_model_from_stream(tmp)
        os.remove(tmp)

        # 3) Desempaqueta directamente MARCA_MODELO_COLOR
        if ref:
            # ref viene como "DS_277_TATATA"
            marca, modelo, color = ref.split('_')
            est.update({
                "marca": marca,
                "modelo": modelo,
                "color": color,
                "fase": "imagen_detectada"
            })
            await update.message.reply_text(
                f"La imagen coincide con {marca} {modelo} color {color}. ¿Continuamos? (SI/NO)",
                reply_markup=menu_botones(["SI", "NO"])
            )
        else:
            reset_estado(cid)
            await update.message.reply_text("No reconocí el modelo. /start para reiniciar.")
        return

    # Confirmación imagen
    if est["fase"] == "imagen_detectada":
        if txt in ("si", "s"):
            est["fase"] = "esperando_talla"
            await update.message.reply_text(
                "¿Qué talla deseas?",
                reply_markup=menu_botones(obtener_tallas_por_color(inv, est["marca"], est["modelo"], est["color"]))
            )
        else:
            await update.message.reply_text("Cancelado. /start para reiniciar.")
            reset_estado(cid)
        return

    # Selección manual de modelo/color/talla
    if est["fase"] == "esperando_modelo":
        modelos = obtener_modelos_por_marca(inv, est["marca"])
        if txt in map(normalize, modelos):
            est["modelo"] = next(m for m in modelos if normalize(m)==txt)
            est["fase"] = "esperando_color"
            await update.message.reply_text(
                "¿Color?",
                reply_markup=menu_botones(obtener_colores_por_modelo(inv, est["marca"], est["modelo"]))
            )
        else:
            await update.message.reply_text("Elige un modelo válido.", reply_markup=menu_botones(modelos))
        return

    if est["fase"] == "esperando_color":
        colores = obtener_colores_por_modelo(inv, est["marca"], est["modelo"])
        if txt in map(normalize, colores):
            est["color"] = next(c for c in colores if normalize(c)==txt)
            est["fase"] = "esperando_talla"
            await update.message.reply_text(
                "¿Talla?",
                reply_markup=menu_botones(obtener_tallas_por_color(inv, est["marca"], est["modelo"], est["color"]))
            )
        else:
            await update.message.reply_text("Elige un color válido.", reply_markup=menu_botones(colores))
        return

    # Datos del usuario y pago
    if est["fase"] == "esperando_talla":
        tallas = obtener_tallas_por_color(inv, est["marca"], est["modelo"], est["color"])
        if txt in map(normalize, tallas):
            est["talla"] = next(t for t in tallas if normalize(t)==txt)
            est["fase"] = "esperando_nombre"
            await update.message.reply_text("¿Tu nombre?")
        else:
            await update.message.reply_text("Elige una talla válida.", reply_markup=menu_botones(tallas))
        return

    if est["fase"] == "esperando_nombre":
        est["nombre"] = txt_raw
        est["fase"] = "esperando_correo"
        await update.message.reply_text("¿Tu correo? 📧")
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
        elif opt == "contraentrega":
            resumen["Pago"] = "Contra entrega"
            registrar_orden(resumen)
            await update.message.reply_text("Pedido registrado para contra entrega. ¡Gracias!")
            reset_estado(cid)
        else:
            await update.message.reply_text("Método inválido.")
        return

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
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT | filters.PHOTO, responder))

    async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
        logging.error("❌ Excepción procesando update:", exc_info=context.error)
    app.add_error_handler(error_handler)

    port = int(os.environ.get("PORT", 8443))
    ngrok_host = os.environ.get("NGROK_HOSTNAME")
    if not ngrok_host:
        logging.error("❌ Debes exportar NGROK_HOSTNAME antes de arrancar")
        return

    app.run_webhook(
        listen="0.0.0.0",
        port=port,
        url_path=TOKEN,
        webhook_url=f"https://{ngrok_host}/{TOKEN}"
    )

if __name__ == "__main__":
    main()
