import os
import io
import logging
from dotenv import load_dotenv
load_dotenv()
import imagehash
from types import SimpleNamespace
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
import asyncio

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from telegram import (
    Update,
    ReplyKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardRemove,
    InputMediaPhoto,
)
from telegram.constants import ChatAction  # ✅ Importación correcta en versiones 20+
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from telegram.constants import ChatAction  # ✅ Importación correcta en versiones 20+
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)


creds_info = json.loads(os.environ["GOOGLE_CREDS_JSON"])
creds = service_account.Credentials.from_service_account_info(
    creds_info,
    scopes=["https://www.googleapis.com/auth/drive.readonly"],
)
drive_service = build("drive", "v3", credentials=creds)

DRIVE_FOLDER_ID = os.environ["DRIVE_FOLDER_ID"]
def precargar_imagenes_drive(service, root_id):
    """
    Recorre recursivamente la carpeta raíz de Drive (root_id),
    descarga cada imagen, calcula su hash perceptual
    y devuelve un dict {hash: (marca, modelo, color)}.
    """
    imagenes = []

    def _walk(current_id, ruta):
        query = f"'{current_id}' in parents and trashed=false"
        page_token = None
        while True:
            resp = service.files().list(
                q=query,
                fields="nextPageToken, files(id,name,mimeType)",
                pageToken=page_token
            ).execute()

            for f in resp.get("files", []):
                if f["mimeType"] == "application/vnd.google-apps.folder":
                    _walk(f["id"], ruta + [f["name"]])
                elif f["mimeType"].startswith("image/"):
                    imagenes.append((f["id"], ruta, f["name"]))
            page_token = resp.get("nextPageToken")
            if page_token is None:
                break

    _walk(root_id, [])  # inicia el recorrido

    cache = {}
    for file_id, ruta, filename in imagenes:
        try:
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
            res = requests.get(url)
            res.raise_for_status()
            img = Image.open(io.BytesIO(res.content))
            h = str(imagehash.phash(img))
            marca, modelo, color = (ruta + ["", "", ""])[:3]
            cache[h] = (marca, modelo, color)
        except Exception as e:
            print(f"⚠️  No se pudo procesar {filename}: {e}")

    print(f"✅ Hashes precargados: {len(cache)} imágenes")
    return cache


PHASH_THRESHOLD = 20
AHASH_THRESHOLD = 18

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
            base = os.path.splitext(name)[0]
            parts = base.split('_')
            sku = parts[0]
            if len(parts) > 1:
                sku += "_" + parts[1]
            if len(parts) > 2:
                sku += "_" + parts[2]

            request = drive_service.files().get_media(fileId=fid)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            fh.seek(0)

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

MODEL_HASHES = precargar_imagenes_drive(drive_service, DRIVE_FOLDER_ID)

def identify_model_from_stream(path: str) -> str | None:
    """
    Abre la imagen subida, calcula su hash y busca directamente
    en MODEL_HASHES cuál es el modelo (marca_modelo_color).
    """
    try:
        img_up = Image.open(path)
    except Exception as e:
        logging.error(f"No pude leer la imagen subida: {e}")
        return None

    # ─── Aquí calculas y buscas el hash ───
    img_hash = str(imagehash.phash(img_up))
    modelo = next(
        (m for m, hashes in MODEL_HASHES.items() if img_hash in hashes),
        None
    )
    # ───────────────────────────────────────

    return modelo

# ——— VARIABLES DE ENTORNO ——————————————————————————————————————————————
OPENAI_API_KEY        = os.environ["OPENAI_API_KEY"]
NOMBRE_NEGOCIO        = os.environ.get("NOMBRE_NEGOCIO", "X100🔥👟")
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
    "– ¿Tienes alguna referencia en mente?\n"
    "– ¿Puedes enviarme la foto del pedido?\n"
    "– ¿Te gustaría ver el catálogo?\n"
    "– ¿Te gustaría rastrear tu pedido?\n"
    "– ¿Te gustaría realizar un cambio?\n"
    "Cuéntame sin ningún problema 😀"
)
CLIP_INSTRUCTIONS = (
    "Para enviarme una imagen, pulsa el ícono de clip (📎), "
    "selecciona “Galería” o “Archivo” y elige la foto."
)
CATALOG_LINK    = "https://wa.me/c/573007607245🔝"
CATALOG_MESSAGE = f"Aquí tienes el catálogo: {CATALOG_LINK}"

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

def enviar_correo(dest, subj, body):
    logging.info(f"[EMAIL STUB] To: {dest}\nSubject: {subj}\n{body}")

def enviar_correo_con_adjunto(dest, subj, body, adj):
    logging.info(f"[EMAIL STUB] To: {dest}\nSubject: {subj}\n{body}\n[Adj: {adj}]")

# ——— UTILIDADES DE INVENTARIO —————————————————————————————————————————
estado_usuario: dict[int, dict] = {}
inventario_cache = None

def menciona_imagen(texto: str) -> bool:
    texto = normalize(texto)
    claves = [
        "foto", "imagen", "pantallazo", "screenshot", "captura",
        "tengo una foto", "te paso la imagen", "imagen del modelo",
        "screen", "de instagram", "vi en insta", "historia de insta",
        "la tengo guardada", "foto del tenis", "foto del zapato"
    ]
    return any(palabra in texto for palabra in claves)

def normalize(text) -> str:
    s = "" if text is None else str(text)
    t = unicodedata.normalize('NFKD', s.strip().lower())
    return "".join(ch for ch in t if not unicodedata.combining(ch))

CONVERSION_TALLAS = {
    "usa": {
        "6": "38", "6.5": "38.5", "7": "39", "7.5": "39.5",
        "8": "40", "8.5": "40.5", "9": "41", "9.5": "41.5",
        "10": "42", "10.5": "43", "11": "44", "11.5": "44.5",
        "12": "45", "12.5": "45.5", "13": "46"
    },
    "euro": {
        "38": "38", "38.5": "38.5", "39": "39", "39.5": "39.5",
        "40": "40", "40.5": "40.5", "41": "41", "41.5": "41.5",
        "42": "42", "43": "43", "44": "44", "44.5": "44.5",
        "45": "45", "45.5": "45.5", "46": "46"
    },
    "colombia": {
        "38": "38", "38.5": "38.5", "39": "39", "39.5": "39.5",
        "40": "40", "40.5": "40.5", "41": "41", "41.5": "41.5",
        "42": "42", "43": "43", "44": "44", "44.5": "44.5",
        "45": "45", "45.5": "45.5", "46": "46"
    }
}
def detectar_talla(texto_usuario: str, tallas_disponibles: list[str]) -> str | None:
    texto = normalize(texto_usuario)

    # Reemplazos comunes para tallas con medio punto
    reemplazos = {
        "seis y medio": "6.5", "7 y medio": "7.5", "ocho y medio": "8.5", "nueve y medio": "9.5",
        "diez y medio": "10.5", "once y medio": "11.5", "doce y medio": "12.5",

        "6 y 1/2": "6.5", "7 y 1/2": "7.5", "8 y 1/2": "8.5", "9 y 1/2": "9.5",
        "10 y 1/2": "10.5", "11 y 1/2": "11.5", "12 y 1/2": "12.5",

        "seis punto cinco": "6.5", "siete punto cinco": "7.5", "ocho punto cinco": "8.5",
        "nueve punto cinco": "9.5", "diez punto cinco": "10.5", "once punto cinco": "11.5", "doce punto cinco": "12.5",

        "6.5": "6.5", "7.5": "7.5", "8.5": "8.5", "9.5": "9.5", "10.5": "10.5",
        "11.5": "11.5", "12.5": "12.5", "13": "13"
    }

    for k, v in reemplazos.items():
        if k in texto:
            texto = texto.replace(k, v)

    # Sistema
    if "usa" in texto:
        sistema = "usa"
    elif "euro" in texto or "europea" in texto:
        sistema = "euro"
    elif "colomb" in texto:
        sistema = "colombia"
    else:
        sistema = None

    # Extrae números incluyendo decimales (6.5, 10.5, etc)
    numeros = re.findall(r"\d+(?:\.\d+)?", texto)

    if not numeros:
        return None

    for num in numeros:
        if sistema:
            talla_estandar = CONVERSION_TALLAS.get(sistema, {}).get(num)
            if talla_estandar and talla_estandar in tallas_disponibles:
                return talla_estandar
        else:
            if num in tallas_disponibles:
                return num

    return None

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

#  TRANSCRIPCIÓN DE AUDIO (WHISPER)
# ───────────────────────────────────────────────────────────────
TEMP_AUDIO_DIR = "temp_audio"
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

async def transcribe_audio(file_path: str) -> str | None:
    """
    Envía el .ogg a Whisper-1 (idioma ES) y devuelve la transcripción.
    Si falla, devuelve None.
    """
    try:
        with open(file_path, "rb") as f:
            rsp = await client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="es",               # fuerza español
                response_format="text",      # devuelve str plano
                prompt="Español Colombia, jerga: parce, mano, ñero, buenos días, buenas, hola"
            )
        if isinstance(rsp, str) and rsp.strip():
            return rsp.strip()
    except Exception as e:
        logging.error(f"❌ Whisper error: {e}")
    return None

# ——— Función para mostrar imágenes de modelo desde Drive ——————————————————
async def mostrar_imagenes_modelo(cid, ctx, marca, tipo_modelo):
    sku = f"{marca.replace(' ','_')}_{tipo_modelo}"
    resp = drive_service.files().list(
        q=(
            f"'{DRIVE_FOLDER_ID}' in parents "
            f"and name contains '{sku}' "
            "and mimeType contains 'image/'"
        ),
        spaces="drive",
        fields="files(id, name)"
    ).execute()
    files = resp.get("files", [])
    if not files:
        await ctx.bot.send_message(cid, "Lo siento, no encontré imágenes de ese modelo.")
        return

    media = []
    for f in files[:5]:
        url = f"https://drive.google.com/uc?id={f['id']}"
        caption = f["name"].split("_",2)[-1]
        media.append(InputMediaPhoto(media=url, caption=caption))

    await ctx.bot.send_chat_action(cid, ChatAction.UPLOAD_PHOTO)
    await ctx.bot.send_media_group(chat_id=cid, media=media)

    est = estado_usuario[cid]
    est["fase"] = "esperando_color"
    est["modelo"] = tipo_modelo
    await ctx.bot.send_message(
        chat_id=cid,
        text="¿Qué color te gustaría?",
        reply_markup=menu_botones(
            obtener_colores_por_modelo(obtener_inventario(), est["marca"], tipo_modelo)
        )
    )

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

#  HOJA DE PEDIDOS
# ───────────────────────────────────────────────────────────────
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
    cid = update.effective_chat.id            # ← se define de primeras

    # 1) Primer contacto: saludo y se queda en esperando_comando
    if cid not in estado_usuario:
        reset_estado(cid)
        await update.message.reply_text(
            WELCOME_TEXT,
            reply_markup=menu_botones([
                "Hacer pedido", "Enviar imagen", "Ver catálogo",
                "Rastrear pedido", "Realizar cambio"
            ])
        )
        return                                 # ← esperamos el siguiente mensaje

    est = estado_usuario[cid]                 # ← ya existe
    inv = obtener_inventario()                # ← cache de inventario

    # 2) Procesamos audio o texto plano ───────────────────────────
    txt_raw = ""
    if update.message.voice or update.message.audio:
        fobj = update.message.voice or update.message.audio
        tg_file = await fobj.get_file()
        local_path = os.path.join(TEMP_AUDIO_DIR, f"{cid}_{tg_file.file_id}.ogg")
        await tg_file.download_to_drive(local_path)
        txt_raw = await transcribe_audio(local_path)
        os.remove(local_path)
        if not txt_raw:
            await update.message.reply_text(
                "Ese audio se escucha muy mal 😕. ¿Podrías enviarlo de nuevo o escribir tu mensaje?",
                reply_markup=ReplyKeyboardRemove()
            )
            return
    else:
        txt_raw = update.message.text or ""

    txt = normalize(txt_raw)

    # 3) Reinicio explícito (/start, inicio, etc.) ────────────────
    if txt in ("reset", "reiniciar", "empezar", "volver", "/start", "menu", "inicio"):
        reset_estado(cid)
        await update.message.reply_text(
            WELCOME_TEXT,
            reply_markup=menu_botones([
                "Hacer pedido", "Enviar imagen", "Ver catálogo",
                "Rastrear pedido", "Realizar cambio"
            ])
        )
        return

    # 4) Intención global de enviar imagen (en cualquier fase) ────
    if menciona_imagen(txt):
        if est["fase"] != "esperando_imagen":
            est["fase"] = "esperando_imagen"
            await update.message.reply_text(CLIP_INSTRUCTIONS, reply_markup=ReplyKeyboardRemove())
        return

    # ── NUEVO: Detectar voz/audio y transcribir ─────────────────────
    txt_raw = ""
    if update.message.voice or update.message.audio:
        fobj = update.message.voice or update.message.audio
        tg_file = await fobj.get_file()
        local_path = os.path.join(TEMP_AUDIO_DIR, f"{cid}_{tg_file.file_id}.ogg")
        await tg_file.download_to_drive(local_path)

        txt_raw = await transcribe_audio(local_path)
        os.remove(local_path)  # limpia temp

        if not txt_raw:
            await update.message.reply_text(
                "Ese audio se escucha muy mal 😕. ¿Podrías enviarlo de nuevo o escribir tu mensaje?",
                reply_markup=ReplyKeyboardRemove()
            )
            return
    else:
        # Mensaje de texto normal
        txt_raw = update.message.text or ""

    txt = normalize(txt_raw)
#   👉 Si el usuario menciona que tiene imagen, sin importar la fase
    # ── Detección global de intención de imagen ───────────────
    if menciona_imagen(txt):
        # Solo cambia de fase si aún no estaba esperando la foto
        if est["fase"] != "esperando_imagen":
            est["fase"] = "esperando_imagen"
            await update.message.reply_text(
                CLIP_INSTRUCTIONS,
                reply_markup=ReplyKeyboardRemove()
            )
        return

    # ——— Palabras clave para reiniciar en cualquier fase ———
    if txt in ("reset", "reiniciar", "empezar", "volver", "/start", "menu", "inicio"):
        reset_estado(cid)
        await saludo_bienvenida(update, ctx)
        # ⬇️ dejamos al usuario ya en “esperando_comando” ⬇️
        estado_usuario[cid]["fase"] = "esperando_comando"
        # ⚠️ NO hacemos return: el mismo mensaje seguirá fluyendo

    # ——— Fase inicial (solo ocurre si algo cambió la fase a "inicio") ———
    if est["fase"] == "inicio":
        await saludo_bienvenida(update, ctx)
        est["fase"] = "esperando_comando"
        # ⚠️ NO hacemos return: procesamos el mensaje que llegó
        # (así “quiero unos gabbana” se analiza inmediatamente)

    # ——— Esperando comando principal ———
    if est["fase"] == "esperando_comando":
        if "rastrear" in txt:
            est["fase"] = "esperando_numero_rastreo"
            await update.message.reply_text(
                "Perfecto, envíame el número de venta.",
                reply_markup=ReplyKeyboardRemove()
            )
            return

        if any(k in txt for k in ("cambio", "reembol", "devolucion")):
            est["fase"] = "esperando_numero_devolucion"
            await update.message.reply_text(
                "Envíame el número de venta para realizar la devolución del pedido.",
                reply_markup=ReplyKeyboardRemove()
            )
            return

        if re.search(r"\b(catálogo|catalogo)\b", txt):
            reset_estado(cid)
            await update.message.reply_text(
                CATALOG_MESSAGE,
                reply_markup=menu_botones([
                    "Hacer pedido", "Enviar imagen", "Ver catálogo",
                    "Rastrear pedido", "Realizar cambio"
                ])
            )
            return


        # ——— Detección de marca ———
        marcas = obtener_marcas_unicas(inv)
        logging.debug(f"[Chat {cid}] txt_norm={txt!r} | marcas={marcas}")

        elegido = next((m for m in marcas if any(tok in txt for tok in normalize(m).split())), None)
        if not elegido:
            palabras_usuario = txt.split()
            for m in marcas:
                for tok in normalize(m).split():
                    if difflib.get_close_matches(tok, palabras_usuario, n=1, cutoff=0.6):
                        elegido = m
                        break
                if elegido:
                    break

        if elegido:
            logging.info(f"Marca detectada: {elegido}")
            est["marca"] = elegido
            est["fase"] = "esperando_modelo"
            await update.message.reply_text(
                f"¡Genial! Veo que buscas {elegido}. ¿Qué modelo de {elegido} te interesa?",
                reply_markup=menu_botones(obtener_modelos_por_marca(inv, elegido))
            )
            return
            

        # ——— Fallback ———
        logging.debug("No detectó ninguna marca en esperando_comando")
        await update.message.reply_text(
            "No entendí tu elección. Usa /start para volver al menú."
        )
        return

    # ——— Rastrear pedido ———
    if est["fase"] == "esperando_numero_rastreo":
        await update.message.reply_text("Guía para rastrear: https://www.instagram.com/juanp_ocampo/")
        reset_estado(cid)
        return

    # ——— Devolución ———
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
                "¿Que color desea?",
                reply_markup=menu_botones(obtener_colores_por_modelo(inv, est["marca"], est["modelo"]))
            )
        else:
            await update.message.reply_text("Elige un modelo válido.", reply_markup=menu_botones(modelos))
        return

    if est["fase"] == "esperando_color":
        colores = obtener_colores_por_modelo(inv, est["marca"], est["modelo"])
        if txt in map(normalize, colores):
            est["color"] = next(c for c in colores if normalize(c) == txt)
            est["fase"] = "esperando_talla"
            tallas = obtener_tallas_por_color(inv, est["marca"], est["modelo"], est["color"])
            await update.message.reply_text(
                f"Las tallas disponibles para {est['modelo']} color {est['color']} son: {', '.join(tallas)}"
            )
            await update.message.reply_text(
                "¿Qué talla deseas?",
                reply_markup=menu_botones(tallas)
            )
        else:
            await update.message.reply_text(
                f"Los colores disponibles para {est['modelo']} son:\n" +
                "\n".join(f"- {c}" for c in colores)
            )
            await update.message.reply_text(
                "¿Cuál color te interesa?",
                reply_markup=menu_botones(colores)
            )
        return

    # Datos del usuario y pago
    if est["fase"] == "esperando_talla":
        tallas = obtener_tallas_por_color(inv, est["marca"], est["modelo"], est["color"])
        talla_detectada = detectar_talla(txt_raw, tallas)
        
        if talla_detectada:
            est["talla"] = talla_detectada
            est["fase"] = "esperando_nombre"
            await update.message.reply_text("¿Tu nombre?")
        else:
            await update.message.reply_text(
                f"Las tallas disponibles para {est['modelo']} color {est['color']} son: {', '.join(tallas)}"
            )
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

   
# --------------------------------------------------------------------
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
import nest_asyncio
import asyncio
import logging
import os
import re
from types import SimpleNamespace

nest_asyncio.apply()

# 1. Instancia FastAPI
api = FastAPI(title="AYA Bot – WhatsApp")

# 2. Conversión de número WhatsApp (ej. 573001234567@c.us → 573001234567)
def wa_chat_id(wa_from: str) -> str:
    return re.sub(r"\D", "", wa_from)

# 3. Simula un mensaje de WhatsApp dentro del flujo `responder`
async def procesar_wa(cid: str, body: str) -> dict:
    class DummyMsg(SimpleNamespace):
        async def reply_text(self, text, **kw): self._ctx.resp.append(text)

    dummy_msg = DummyMsg(text=body, photo=None, voice=None, audio=None)
    dummy_update = SimpleNamespace(
        message=dummy_msg,
        effective_chat=SimpleNamespace(id=cid)
    )

    class DummyCtx(SimpleNamespace):
        async def bot_send(self, chat_id, text, **kw): self.resp.append(text)

    ctx = DummyCtx(resp=[], bot=SimpleNamespace(
        send_message=lambda chat_id, text, **kw: asyncio.create_task(ctx.bot_send(chat_id, text))
    ))
    dummy_msg._ctx = ctx

    await responder(dummy_update, ctx)
    return {"type": "text", "text": ctx.resp[-1] if ctx.resp else "No entendí 🥲"}

# 4. Webhook para WhatsApp (usado por Venom)
# --------------------------------------------------------------------
# 4. Webhook para WhatsApp (usado por Venom)
#     – Maneja texto (body) y también imágenes (type == "image")
# --------------------------------------------------------------------
# ---------------------------------------------------------------------
#  webhook /venom  – recibe mensajes de Venom (WhatsApp)
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
#  webhook /venom  – recibe mensajes de Venom (WhatsApp)
# ---------------------------------------------------------------------
@api.post("/venom")
async def venom_webhook(req: Request):
    try:
        # -----------------------------------------------------------------
        # 1. Leer JSON entrante
        # -----------------------------------------------------------------
        data = await req.json()

        import pprint, base64, datetime, os, logging
        pp = pprint.PrettyPrinter(indent=2)
        pp.pprint(data)             # ← DEBUG opcional

        cid      = wa_chat_id(data.get("from", ""))
        body     = data.get("body", "") or ""
        mtype    = data.get("type", "")
        mimetype = (data.get("mimetype") or "").lower()   # 🔧 asegura str

        logging.info(f"📩 Mensaje recibido — CID: {cid} — Tipo: {mtype}")

        # -----------------------------------------------------------------
        # 2. Si es IMAGEN en base64
        # -----------------------------------------------------------------
        if mtype == "image" or mimetype.startswith("image"):     # 🔧 usa mimetype
            try:
                img_data = base64.b64decode(body)
                logging.info("📥 Imagen decodificada correctamente desde base64.")
            except Exception as e:
                logging.error(f"❌ No se pudo decodificar la imagen: {e}")
                return JSONResponse({"type": "text",
                                     "text": "No pude procesar la imagen 😕"})

            # Guardar archivo temporal
            os.makedirs("temp", exist_ok=True)
            local_path = f"temp/{cid}_{int(datetime.datetime.now().timestamp())}.jpg"
            with open(local_path, "wb") as f:
                f.write(img_data)
            logging.info(f"💾 Imagen guardada en: {local_path}")

            # Calcular hash y buscar coincidencia
            try:
                logging.info("🧠 Enviando a identify_model_from_stream…")
                ref = identify_model_from_stream(local_path)
                logging.info(f"🔍 Resultado del hash: {ref}")
            except Exception as e:
                logging.error(f"❌ Error al procesar la imagen: {e}")
                return JSONResponse({"type": "text",
                                     "text": "Error al procesar la imagen 😕"})
            finally:
                try:
                    os.remove(local_path)
                    logging.info(f"🧹 Eliminado {local_path}")
                except:
                    pass

            # Coincidencia
            if ref:
                try:
                    marca, modelo, color = ref.split('_', 2)
                except ValueError:
                    marca, modelo, color = "Marca", "Modelo", "Color"

                estado_usuario.setdefault(cid, reset_estado(cid))
                est = estado_usuario[cid]
                est.update({"fase":   "imagen_detectada",
                            "marca":  marca,
                            "modelo": modelo,
                            "color":  color})

                return JSONResponse({"type": "text",
                                     "text": f"La imagen coincide con {marca} {modelo} color {color}. "
                                             "¿Deseas continuar tu compra? (SI/NO)"})

            # Sin coincidencia
            reset_estado(cid)
            return JSONResponse({"type": "text",
                                 "text": "No reconocí el modelo. "
                                         "Puedes intentar con otra imagen o escribir /start para reiniciar."})

        # -----------------------------------------------------------------
        # 3. Si NO es imagen → procesar texto
        # -----------------------------------------------------------------
        reply = await procesar_wa(cid, body)
        return JSONResponse(reply)

    # ---------------------------------------------------------------------
    # 4. Error global del webhook
    # ---------------------------------------------------------------------
    except Exception:
        logging.exception("🔥 Error en /venom")
        return JSONResponse({"type": "text",
                             "text": "Error interno en el bot. Intenta de nuevo."},
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


# -------------------------------------------------------------------------
# 5. Arranque del servidor
if __name__ == "__main__":
    import os
    import uvicorn

    # Leer la variable que Render inyecta
    port = int(os.environ.get("PORT", 8000))  
    # OJO: si tú localmente usas otro puerto, cámbialo aquí, pero en Render
    # siempre tendrás PORT definido y será un número distinto.

    uvicorn.run(
        "lector:app",        # módulo:app
        host="0.0.0.0",      # escucha en todas las interfaces
        port=port            # ¡úsalo aquí!
    )


