# ——— Librerías estándar de Python ———
import os
import io
import base64
import logging
import json
import re
import requests
import random
import string
import datetime
import unicodedata
import difflib
import asyncio
from types import SimpleNamespace
from collections import defaultdict

# ——— Librerías externas ———
from dotenv import load_dotenv
from PIL import Image
import imagehash
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
import nest_asyncio
import openai

# Google Cloud
from google.cloud import vision
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# Telegram
from telegram import (
    Update,
    ReplyKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardRemove,
    InputMediaPhoto,
)
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# Inicializa dotenv
load_dotenv()

# FastAPI instance
api = FastAPI()


creds_info = json.loads(os.environ["GOOGLE_CREDS_JSON"])
creds = service_account.Credentials.from_service_account_info(
    creds_info,
    scopes=["https://www.googleapis.com/auth/drive.readonly"],
)
drive_service = build("drive", "v3", credentials=creds)
# Cliente OCR de Google Cloud Vision
vision_client = vision.ImageAnnotatorClient(credentials=creds)

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

for h, ref in MODEL_HASHES.items():
    print(f"HASH precargado: {h} → {ref}")

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
    "Si tienes una foto puedes enviarla\n"
    "Si tienes numero de referencia enviamelo\n"
    "Puedes enviarme la foto del pedido\n"
    "Te gustaria ver unos videos de nuestras referencias👟?\n"
    "Cuéntame sin ningún problema 😀"
)
CLIP_INSTRUCTIONS = (
    "Para enviarme una imagen, pulsa el ícono de clip (📎), "
    "selecciona “Galería” o “Archivo” y elige la foto."
)
CATALOG_LINK    = "https://wa.me/c/573007607245🔝"
CATALOG_MESSAGE = f"Aquí tienes el catálogo: {CATALOG_LINK}"

def enviar_correo(dest, subj, body):
    logging.info(f"[EMAIL STUB] To: {dest}\nSubject: {subj}\n{body}")

def enviar_correo_con_adjunto(dest, subj, body, adj):
    logging.info(f"[EMAIL STUB] To: {dest}\nSubject: {subj}\n{body}\n[Adj: {adj}]")

def extraer_texto_comprobante(path_local: str) -> str:
    """
    Usa Google Cloud Vision OCR para extraer texto de una imagen local.
    """
    try:
        logging.info(f"[OCR] Procesando imagen: {path_local}")

        # 1. Cargar credenciales
        credentials = service_account.Credentials.from_service_account_info(
            json.loads(os.environ["GOOGLE_CREDS_JSON"])
        )
        client = vision.ImageAnnotatorClient(credentials=credentials)

        # 2. Leer imagen
        with io.open(path_local, "rb") as image_file:
            content = image_file.read()

        if not content:
            logging.warning("[OCR] Imagen está vacía o no se pudo leer.")
            return ""

        image = vision.Image(content=content)

        # 3. Ejecutar OCR
        response = client.text_detection(image=image)

        # 4. Capturar errores de API
        if response.error.message:
            logging.error(f"[OCR ERROR] Google Vision respondió con error: {response.error.message}")
            return ""

        texts = response.text_annotations

        if texts:
            texto_detectado = texts[0].description
            logging.info("[OCR RESULTADO] Texto completo detectado:")
            for i, linea in enumerate(texto_detectado.splitlines()):
                logging.info(f"[OCR LINEA {i}] → {repr(linea)}")
            return texto_detectado
        else:
            logging.warning("[OCR] No se detectó texto en la imagen.")
            return ""

    except Exception as e:
        logging.exception(f"❌ Error extrayendo texto del comprobante: {e}")
        return ""

def es_comprobante_valido(texto: str) -> bool:
    # 🔍 Mostrar texto crudo completo
    logging.info("[OCR DEBUG] Texto detectado en comprobante:\n" + texto)

    # 🔍 Mostrar línea por línea con representación exacta
    for i, linea in enumerate(texto.splitlines()):
        logging.info(f"[OCR LINEA {i}] → {repr(linea)}")

    # 🔄 Normalizar texto (quitar tildes, pasar a minúsculas, quitar signos)
    texto_normalizado = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("utf-8")
    texto_normalizado = texto_normalizado.lower()
    texto_normalizado = re.sub(r"[^\w\s]", "", texto_normalizado)

    logging.info("[OCR DEBUG] Texto normalizado:\n" + texto_normalizado)

    # 🔑 Frases clave válidas
    claves = [
        "pago exitoso",
        "transferencia exitosa",
        "comprobante",
        "recibo",
        "pago aprobado",
        "transferencia realizada"
    ]

    for clave in claves:
        if clave in texto_normalizado:
            logging.info(f"[OCR DEBUG] Coincidencia encontrada: '{clave}'")
            return True

    logging.warning("[OCR DEBUG] No se encontró ninguna clave válida en el texto extraído.")
    return False
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

def obtener_colores_por_modelo(inv: list[dict], modelo: str) -> list[str]:
    return sorted({i.get("color", "").strip()
                   for i in inv
                   if normalize(i.get("modelo", "")) == normalize(modelo)
                   and disponible(i)})

def obtener_tallas_por_color(inv: list[dict], modelo: str, color: str) -> list[str]:
    return sorted({
        str(i.get("talla", "")).strip()
        for i in inv
        if normalize(i.get("modelo", "")) == normalize(modelo)
        and normalize(i.get("color", "")) == normalize(color)
        and disponible(i)
    })

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
                response_format="text",      # devuelve texto plano
                prompt="Español Colombia, jerga: parce, mano, ñero, buenos días, buenas, hola"
            )
        if isinstance(rsp, str) and rsp.strip():
            return rsp.strip()
    except Exception as e:
        logging.error(f"❌ Whisper error: {e}")
    return None

# ───────────────────────────────────────────────────────────────

# 🔥 Enviar video de referencia
async def enviar_video_referencia(cid, ctx, referencia):
    try:
        videos = {
            "ds 277": "ID_VIDEO_DS277",
            "277": "ID_VIDEO_DS277",
            "ds 288": "ID_VIDEO_DS288",
            "288": "ID_VIDEO_DS288",
            "ds 299": "ID_VIDEO_DS299",
            "299": "ID_VIDEO_DS299",
        }

        video_id = videos.get(referencia.lower())

        if video_id:
            video_url = f"https://drive.google.com/uc?id={video_id}"
            await ctx.bot.send_chat_action(chat_id=cid, action=ChatAction.UPLOAD_VIDEO)
            await ctx.bot.send_video(
                chat_id=cid,
                video=video_url,
                caption=f"🎬 Video de referencia {referencia.upper()}.\n¿Deseas continuar tu compra? (SI/NO)"
            )
        else:
            await ctx.bot.send_message(
                chat_id=cid,
                text="😕 No tengo un video específico para esa referencia."
            )
    except Exception as e:
        logging.error(f"Error enviando video: {e}")
        await ctx.bot.send_message(
            chat_id=cid,
            text="⚠️ Ocurrió un error al intentar enviar el video. Intenta de nuevo."
        )

# ───────────────────────────────────────────────────────────────

# 🔥 Manejar preguntas frecuentes (FAQ)
async def manejar_pqrs(update, ctx) -> bool:
    txt = normalize(update.message.text or "")

    faq_respuestas = {
        "garantia": "🛡️ Todos nuestros productos tienen *60 días de garantía* por defectos de fábrica.",
        "garantía": "🛡️ Todos nuestros productos tienen *60 días de garantía* por defectos de fábrica.",
        "envio": "🚚 Hacemos envíos a toda Colombia en máximo *2 días hábiles*.",
        "envío": "🚚 Hacemos envíos a toda Colombia en máximo *2 días hábiles*.",
        "demora": "⏳ El envío normalmente tarda *2 días hábiles*. ¡Te llega rápido!",
        "contraentrega": "💵 Tenemos *pago contra entrega* con anticipo de $35.000.",
        "pago": "💳 Puedes pagar por transferencia bancaria, QR o contra entrega.",
        "original": "✅ Sí, son *originales colombianos* de alta calidad.",
        "ubicacion": "📍 Estamos en *Bucaramanga, Santander* y enviamos a todo el país.",
        "ubicación": "📍 Estamos en *Bucaramanga, Santander* y enviamos a todo el país.",
        "talla": "📏 Nuestra horma es *normal*. La talla que usas normalmente te quedará perfecta.",
        "descuento": "🎉 Si compras 2 pares te damos *10% de descuento* adicional."
    }

    for palabra, respuesta in faq_respuestas.items():
        if palabra in txt:
            await ctx.bot.send_message(
                chat_id=update.effective_chat.id,
                text=respuesta,
                parse_mode="Markdown"
            )
            return True

    return False

# 🔥 Manejar imagen enviada por el usuario
async def manejar_imagen(update, ctx):
    cid = update.effective_chat.id
    est = estado_usuario.setdefault(cid, reset_estado(cid))

    # Descargar la imagen temporalmente
    f = await update.message.photo[-1].get_file()
    tmp_path = os.path.join("temp", f"{cid}.jpg")
    os.makedirs("temp", exist_ok=True)
    await f.download_to_drive(tmp_path)

    # Buscar el modelo usando hashing
    ref = identify_model_from_stream(tmp_path)
    os.remove(tmp_path)

    if ref:
        try:
            marca, modelo, color = ref.split('_')
        except Exception as e:
            logging.error(f"Error al desempaquetar referencia: {e}")
            marca, modelo, color = "Desconocido", ref, ""

        est.update({
            "marca": marca,
            "modelo": modelo,
            "color": color,
            "fase": "imagen_detectada"
        })

        await ctx.bot.send_message(
            chat_id=cid,
            text=f"📸 La imagen coincide con:\n"
                 f"*Marca:* {marca}\n"
                 f"*Modelo:* {modelo}\n"
                 f"*Color:* {color}\n\n"
                 "¿Deseas continuar tu compra con este modelo? (SI/NO)",
            parse_mode="Markdown",
            reply_markup=menu_botones(["SI", "NO"])
        )
    else:
        reset_estado(cid)
        await ctx.bot.send_message(
            chat_id=cid,
            text="😔 No pude reconocer el modelo de la imagen. ¿Quieres intentar otra vez?",
            parse_mode="Markdown",
            reply_markup=menu_botones(["Enviar otra imagen", "Ver catálogo"])
        )

# ───────────────────────────────────────────────────────────────

# 🔥 Mostrar imágenes del modelo detectado
async def mostrar_imagenes_modelo(cid, ctx, marca, tipo_modelo):
    sku = f"{marca.replace(' ', '_')}_{tipo_modelo}"
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
        caption = f["name"].split("_", 2)[-1]
        media.append(InputMediaPhoto(media=url, caption=caption))

    await ctx.bot.send_chat_action(cid, action=ChatAction.UPLOAD_PHOTO)
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

# ───────────────────────────────────────────────────────────────

# 🔥 Registrar la orden en Google Sheets
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

# ───────────────────────────────────────────────────────────────

# 🔥 Generar ID único para una venta
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


# --------------------------------------------------------------------------------------------------

async def responder(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    cid = update.effective_chat.id

    # 1) Primer contacto: saludo
    if cid not in estado_usuario:
        reset_estado(cid)
        await update.message.reply_text(
            WELCOME_TEXT,
            reply_markup=menu_botones([
                "Hacer pedido", "Enviar imagen", "Ver catálogo",
                "Rastrear pedido", "Realizar cambio"
            ])
        )
        return

    # 2) Estado actual e inventario
    est = estado_usuario[cid]
    inv = obtener_inventario()

    # 3) Captura y normaliza texto
    txt_raw = update.message.text or ""
    txt = normalize(txt_raw)

    # ✅ DEBUG real
    print("🧠 FASE:", est.get("fase"))
    print("🧠 TEXTO:", txt_raw, "|", repr(txt_raw))
    print("🧠 ESTADO:", est)

    # 4) Reinicio explícito si escribe /start o similares
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

# ─────────── Preguntas frecuentes (FAQ) ───────────
    if est.get("fase") not in ("esperando_pago", "esperando_comprobante"):

        # FAQ 1: ¿Cuánto demora el envío?
        if any(frase in txt for frase in (
            "cuanto demora", "cuánto demora", "cuanto tarda", "cuánto tarda",
            "cuanto se demora", "cuánto se demora", "en cuanto llega", "en cuánto llega",
            "me llega rapido", "llegan rapido"
        )):
            await ctx.bot.send_message(
                chat_id=cid,
                text=(
                    "🚚 El tiempo de entrega depende de la ciudad de destino, "
                    "pero generalmente tarda *2 días hábiles* en llegar.\n\n"
                    "Si lo necesitas para *mañana mismo*, podemos enviarlo al terminal de transporte. "
                    "En ese caso aplica *pago anticipado* (no contra entrega)."
                ),
                parse_mode="Markdown"
            )
            return

        # FAQ 2: ¿Tienen pago contra entrega?
        if any(frase in txt for frase in (
            "pago contra entrega", "pago contraentrega", "contraentrega", "contra entrega",
            "pagan al recibir", "puedo pagar al recibir", "tienen contra entrega"
        )):
            await ctx.bot.send_message(
                chat_id=cid,
                text=(
                    "📦 ¡Claro que sí! Tenemos *pago contra entrega*.\n\n"
                    "Pedimos un *anticipo de $35 000* que cubre el envío. "
                    "Ese valor se descuenta del precio total cuando recibes el pedido."
                ),
                parse_mode="Markdown"
            )
            return

        # FAQ 3: ¿Tienen garantía?
        if any(frase in txt for frase in (
            "tienen garantia", "tienen garantía", "hay garantía", "hay garantia",
            "garantía", "garantia", "tienen garantia de fabrica"
        )):
            await ctx.bot.send_message(
                chat_id=cid,
                text=(
                    "🛡️ Todos nuestros productos tienen *garantía de 60 días* "
                    "por defectos de fábrica o problemas de pegado.\n\n"
                    "Cualquier inconveniente, estamos para ayudarte."
                ),
                parse_mode="Markdown"
            )
            return


    # ... puedes dejar tus demás FAQs igual que están ...

    # FAQ 4: ¿Cómo sé que no me van a robar?
    if any(frase in txt for frase in (
        "no me van a robar", "me van a robar", "es seguro",
        "como se que es seguro", "no es estafa", "es confiable",
        "me estafan", "roban por internet", "es real", "desconfío",
        "no me da confianza", "no confío", "dudas"
    )):
        video_url = "https://tudominio.com/videos/video_confianza.mp4"
        await ctx.bot.send_message(
            chat_id=cid,
            text=(
                "🤝 Entendemos tu preocupación. "
                "Te compartimos este video para que veas que somos una tienda real y seria."
            ),
            parse_mode="Markdown"
        )
        await ctx.bot.send_chat_action(chat_id=cid, action=ChatAction.UPLOAD_VIDEO)
        await ctx.bot.send_video(
            chat_id=cid,
            video=video_url,
            caption="¡Estamos aquí para ayudarte en lo que necesites! 👟✨"
        )
        return

    # FAQ 5: ¿Dónde están ubicados?
    if any(frase in txt for frase in (
        "donde estan ubicados", "donde queda", "ubicacion", "ubicación",
        "direccion", "dirección", "donde estan", "donde es la tienda",
        "estan ubicados", "ubicados en donde", "en que ciudad estan", "en que parte estan"
    )):
        await ctx.bot.send_message(
            chat_id=cid,
            text=(
                "📍 Estamos en *Bucaramanga, Santander*.\n\n"
                "🏡 *Barrio San Miguel, Calle 52 #16-74*\n\n"
                "🚚 ¡Enviamos a todo Colombia con Servientrega!\n\n"
                "Ubicación Google Maps: https://maps.google.com/?q=7.109500,-73.121597"
            ),
            parse_mode="Markdown"
        )
        return

    # FAQ 6: ¿Son nacionales o importados?
    if any(frase in txt for frase in (
        "son nacionales", "son importados", "es nacional o importado",
        "nacionales o importados", "hecho en colombia", "fabricados en colombia",
        "son de aqui", "es de colombia", "fabricacion colombiana"
    )):
        await ctx.bot.send_message(
            chat_id=cid,
            text=(
                "🇨🇴 Nuestra marca es *100 % colombiana* y las zapatillas "
                "se elaboran con orgullo en *Bucaramanga* por artesanos locales."
            ),
            parse_mode="Markdown"
        )
        return

    # FAQ 7: ¿Son originales?
    if any(frase in txt for frase in (
        "son originales", "es original", "originales",
        "es copia", "son copia", "son replica", "réplica", "imitacion"
    )):
        await ctx.bot.send_message(
            chat_id=cid,
            text="✅ ¡Claro! Son *originales*. Somos *X100*, marca 100 % colombiana reconocida por su calidad y diseño.",
            parse_mode="Markdown"
        )
        return

    # FAQ 8: ¿De qué calidad son?
    if any(frase in txt for frase in (
        "que calidad son", "de que calidad son", "son buena calidad", "son de buena calidad",
        "son de mala calidad", "que calidad manejan", "que calidad tienen", "calidad de las zapatillas"
    )):
        await ctx.bot.send_message(
            chat_id=cid,
            text=(
                "✨ Nuestras zapatillas están elaboradas con *materiales de alta calidad*.\n\n"
                "Cada par se fabrica cuidadosamente para asegurar *calidad AAA* 👟🔝, "
                "garantizando comodidad, durabilidad y excelente acabado."
            ),
            parse_mode="Markdown"
        )
        return

    # FAQ 9: ¿Hay descuento si compro 2 pares?
    if any(frase in txt for frase in (
        "si compro 2 pares", "dos pares descuento", "descuento por 2 pares",
        "descuento por dos pares", "me descuentan si compro dos", "descuento si compro dos",
        "hay descuento por dos", "promocion dos pares", "descuento en 2 pares"
    )):
        await ctx.bot.send_message(
            chat_id=cid,
            text=(
                "🎉 ¡Sí! Si compras *2 pares* te damos un *10% de descuento adicional* sobre el total.\n\n"
                "¡Aprovecha para estrenar más y pagar menos! 🔥👟👟"
            ),
            parse_mode="Markdown"
        )
        return

    # FAQ 10: ¿Manejan precios para mayoristas?
    if any(frase in txt for frase in (
        "precio mayorista", "precios para mayoristas", "mayorista", "quiero vender",
        "puedo venderlos", "descuento para revender", "revender", "comprar para vender",
        "manejan precios para mayoristas", "mayoreo", "venta al por mayor"
    )):
        await ctx.bot.send_message(
            chat_id=cid,
            text=(
                "🛍️ ¡Claro! Manejamos *precios para mayoristas* en pedidos de *6 pares en adelante*, "
                "sin importar tallas ni referencias.\n\n"
                "Condición: vender mínimo al mismo precio que nosotros para cuidar el mercado."
            ),
            parse_mode="Markdown"
        )
        return

    # FAQ 11: ¿Las tallas son normales o grandes?
    if any(frase in txt for frase in (
        "las tallas son normales", "horma normal", "talla normal",
        "horma grande", "horma pequeña", "tallas grandes", "tallas pequeñas",
        "las tallas son grandes", "las tallas son pequeñas", "como son las tallas"
    )):
        await ctx.bot.send_message(
            chat_id=cid,
            text=(
                "👟 Nuestra horma es *normal*. Si calzas talla *40* nacional, te queda bien la *40* de nosotros.\n\n"
                "Para mayor seguridad, puedes enviarnos una foto de la *etiqueta interna* de tus tenis actuales 📏✨."
            ),
            parse_mode="Markdown"
        )
        return

    # FAQ 12: ¿Cuál es la talla más grande que manejan?
    if any(frase in txt for frase in (
        "talla mas grande", "talla más grande", "cual es la talla mas grande",
        "hasta que talla llegan", "mayor talla", "talla maxima", "talla máxima"
    )):
        await ctx.bot.send_message(
            chat_id=cid,
            text=(
                "📏 La talla más grande que manejamos es:\n\n"
                "• *45 Nacional* 🇨🇴\n"
                "• *47 Europeo* 🇪🇺\n\n"
                "¡También tenemos opciones para pies grandes! 👟✨"
            ),
            parse_mode="Markdown"
        )
        return

    # 📷 Si el usuario envía una foto (detectamos modelo automáticamente)
    if update.message.photo:
        f = await update.message.photo[-1].get_file()
        tmp = os.path.join("temp", f"{cid}.jpg")
        os.makedirs("temp", exist_ok=True)
        await f.download_to_drive(tmp)

        ref = identify_model_from_stream(tmp)
        os.remove(tmp)

        if ref:
            marca, modelo, color = ref.split('_')
            est.update({
                "marca": marca,
                "modelo": modelo,
                "color": color,
                "fase": "imagen_detectada"
            })
            await ctx.bot.send_message(
                chat_id=cid,
                text=f"La imagen coincide con {marca} {modelo} color {color}. ¿Continuamos? (SI/NO)",
                reply_markup=menu_botones(["SI", "NO"]),
            )
        else:
            reset_estado(cid)
            await ctx.bot.send_message(
                chat_id=cid,
                text="😕 No reconocí el modelo. Escribe /start para reiniciar.",
                parse_mode="Markdown"
            )
        return

    # 📷 Confirmación si la imagen detectada fue correcta
    if est.get("fase") == "imagen_detectada":
        if any(frase in txt for frase in ("si", "sí", "s", "claro", "claro que sí", "quiero comprar", "continuar", "vamos")):
            est["fase"] = "esperando_talla"
            tallas = obtener_tallas_por_color(inv, est["modelo"], est["color"])
            if isinstance(tallas, (int, float, str)):
                tallas = [str(tallas)]
            await ctx.bot.send_message(
                chat_id=cid,
                text="¡Perfecto! 🎯 ¿Qué talla deseas?",
                reply_markup=menu_botones(tallas),
            )
            return
        else:
            await ctx.bot.send_message(
                chat_id=cid,
                text="Cancelado. /start para reiniciar o cuéntame si quieres ver otra referencia. 📋",
                parse_mode="Markdown"
            )
            reset_estado(cid)
            return

    # 🛒 Flujo manual si está buscando modelo
    if est.get("fase") == "esperando_modelo":
        modelos = obtener_modelos_por_marca(inv, est["marca"])
        if txt in map(normalize, modelos):
            est["modelo"] = next(m for m in modelos if normalize(m) == txt)
            est["fase"] = "esperando_color"
            colores = obtener_colores_por_modelo(inv, est["modelo"])
            await ctx.bot.send_message(
                chat_id=cid,
                text="¿Qué color deseas?",
                reply_markup=menu_botones(colores),
            )
        else:
            await ctx.bot.send_message(
                chat_id=cid,
                text="Elige un modelo válido.",
                reply_markup=menu_botones(modelos),
            )
        return

    # 🎨 Elegir color del modelo
    if est.get("fase") == "esperando_color":
        colores = obtener_colores_por_modelo(inv, est["modelo"])
        if isinstance(colores, (int, float, str)):
            colores = [str(colores)]

        # Normalizar entrada y colores
        colores_normalizados = {normalize(c): c for c in colores}
        entrada_normalizada = normalize(txt)

        # Buscar coincidencia cercana
        import difflib
        coincidencias = difflib.get_close_matches(entrada_normalizada, colores_normalizados.keys(), n=1, cutoff=0.6)

        if coincidencias:
            color_seleccionado = colores_normalizados[coincidencias[0]]
            est["color"] = color_seleccionado
            est["fase"] = "esperando_talla"

            tallas = obtener_tallas_por_color(inv, est["modelo"], est["color"])
            if isinstance(tallas, (int, float, str)):
                tallas = [str(tallas)]

            await ctx.bot.send_message(
                chat_id=cid,
                text=(
                    f"Perfecto 🎯 ¿Qué talla deseas para el modelo *{est['modelo']}* color *{est['color']}*?\n\n"
                    f"👉 Tallas disponibles: {', '.join(tallas)}"
                ),
                parse_mode="Markdown"
            )
        else:
            colores_str = "\n".join(f"- {c}" for c in colores)
            await ctx.bot.send_message(
                chat_id=cid,
                text=(
                    f"⚠️ No entendí ese color.\n\n"
                    f"🎨 Los colores disponibles para *{est['modelo']}* son:\n\n"
                    f"{colores_str}\n\n"
                    "¿Cuál color te interesa?"
                ),
                parse_mode="Markdown"
            )
        return

    # 👟 Elegir talla
    if est.get("fase") == "esperando_talla":
        tallas = obtener_tallas_por_color(inv, est["modelo"], est["color"])
        
        if isinstance(tallas, (int, float, str)):
            tallas = [str(tallas)]

        talla_detectada = detectar_talla(txt_raw, tallas)

        if talla_detectada:
            est["talla"] = talla_detectada
            est["fase"] = "esperando_nombre"
            await ctx.bot.send_message(
                chat_id=cid,
                text="¿Tu nombre completo? 👤",
                parse_mode="Markdown"
            )
        else:
            await ctx.bot.send_message(
                chat_id=cid,
                text=f"⚠️ Las tallas disponibles para {est['modelo']} color {est['color']} son:\n{', '.join(tallas)}",
                parse_mode="Markdown",
                reply_markup=menu_botones(tallas),
            )
        return

    # ✏️ Nombre del cliente
    if est.get("fase") == "esperando_nombre":
        est["nombre"] = txt_raw
        est["fase"] = "esperando_correo"
        await ctx.bot.send_message(
            chat_id=cid,
            text="¿Cuál es tu correo electrónico? 📧",
            parse_mode="Markdown"
        )
        return

    # 📧 Correo del cliente
    if est.get("fase") == "esperando_correo":
        if re.match(r"[^@]+@[^@]+\.[^@]+", txt_raw):
            est["correo"] = txt_raw
            est["fase"] = "esperando_telefono"
            await ctx.bot.send_message(
                chat_id=cid,
                text="¿Tu número de teléfono? 📱",
                parse_mode="Markdown"
            )
        else:
            await ctx.bot.send_message(
                chat_id=cid,
                text="⚠️ Correo inválido. Intenta de nuevo.",
                parse_mode="Markdown"
            )
        return


    # 📞 Teléfono del cliente
    if est.get("fase") == "esperando_telefono":
        if re.match(r"^\+?\d{7,15}$", txt_raw):
            est["telefono"] = txt_raw
            est["fase"] = "esperando_ciudad"
            await ctx.bot.send_message(
                chat_id=cid,
                text="¿En qué ciudad estás? 🏙️",
                parse_mode="Markdown"
            )
        else:
            await ctx.bot.send_message(
                chat_id=cid,
                text="⚠️ Teléfono inválido. Intenta de nuevo.",
                parse_mode="Markdown"
            )
        return

    # 🌆 Ciudad del cliente
    if est.get("fase") == "esperando_ciudad":
        est["ciudad"] = txt_raw
        est["fase"] = "esperando_provincia"
        await ctx.bot.send_message(
            chat_id=cid,
            text="¿En qué departamento o provincia estás? 🏞️",
            parse_mode="Markdown"
        )
        return

    # 🏞️ Provincia del cliente
    if est.get("fase") == "esperando_provincia":
        est["provincia"] = txt_raw
        est["fase"] = "esperando_direccion"
        await ctx.bot.send_message(
            chat_id=cid,
            text="¿Dirección exacta de envío? 🏡",
            parse_mode="Markdown"
        )
        return

    # ------------------------------------------------------------------------
    # 🏡 Dirección de envío
    if est.get("fase") == "esperando_direccion":
        est["direccion"] = txt_raw.strip()

        precio = next(
            (
                i["precio"] for i in inv
                if normalize(i["marca"]) == normalize(est["marca"])
                and normalize(i["modelo"]) == normalize(est["modelo"])
                and normalize(i["color"]) == normalize(est["color"])
            ),
            None
        )
        if precio is None:
            await ctx.bot.send_message(
                chat_id=cid,
                text="No pude obtener el precio, intenta de nuevo."
            )
            return

        est["precio_total"] = int(precio)
        sale_id = generate_sale_id()
        est["sale_id"] = sale_id
        est["resumen"] = {
            "Número Venta": sale_id,
            "Fecha Venta": datetime.datetime.now().isoformat(),
            "Cliente": est["nombre"],
            "Teléfono": est["telefono"],
            "Producto": est["modelo"],
            "Color": est["color"],
            "Talla": est["talla"],
            "Correo": est["correo"],
            "Pago": None,
            "Estado": "PENDIENTE"
        }

        msg = (
            f"✅ Pedido: {sale_id}\n"
            f"Nombre: {est['nombre']}\n"
            f"Correo: {est['correo']}\n"
            f"Celular: {est['telefono']}\n"
            f"Dirección: {est['direccion']}, {est['ciudad']}, {est['provincia']}\n"
            f"Producto: {est['modelo']} color {est['color']} talla {est['talla']}\n"
            f"Valor a pagar: {precio:,} COP\n\n"
            "¿Cómo deseas hacer el pago?\n"
            "• Contraentrega: adelanta 35 000 COP para el envío (se descuenta del total).\n"
            "• Transferencia inmediata: paga completo hoy y obtén 5 % de descuento.\n\n"
            "Escribe tu método de pago: Transferencia o Contraentrega."
        )
        await ctx.bot.send_message(chat_id=cid, text=msg)
        est["fase"] = "esperando_pago"
        estado_usuario[cid] = est
        return

    # 💳 Método de pago
    if est.get("fase") == "esperando_pago":
        print("🧪 ENTRÓ AL BLOQUE DE PAGO ✅")

        opciones = {
            "transferencia": "transferencia",
            "transf": "transferencia",
            "trans": "transferencia",
            "pago inmediato": "transferencia",
            "qr": "transferencia",
            "contraentrega": "contraentrega",
            "contra entrega": "contraentrega",
            "contra": "contraentrega",
            "contrapago": "contraentrega"
        }

        txt_norm = normalize(txt_raw).lower().strip()
        op_detectada = next((v for k, v in opciones.items() if k in txt_norm), None)

        print("🧪 opción detectada:", op_detectada)

        if not op_detectada:
            print("❌ Opción inválida detectada")
            await ctx.bot.send_message(chat_id=cid, text="⚠️ Opción no válida. Escribe Transferencia o Contraentrega.")
            return

        resumen = est.get("resumen")
        precio_original = est.get("precio_total")

        if not resumen or not precio_original:
            print("❌ ERROR: resumen o precio_total vacíos")
            await ctx.bot.send_message(chat_id=cid, text="❌ Hubo un problema. Escribe *hola* para reiniciar.")
            reset_estado(cid)
            estado_usuario.pop(cid, None)
            return

        precio_original = int(precio_original)

        if op_detectada == "transferencia":
            est["fase"] = "esperando_comprobante"
            resumen["Pago"] = "Transferencia"
            descuento = round(precio_original * 0.05)
            valor_final = precio_original - descuento
            resumen["Descuento"] = f"-{descuento} COP"
            resumen["Valor Final"] = valor_final
            estado_usuario[cid] = est

            msg = (
                "🟢 Elegiste TRANSFERENCIA.\n"
                f"💰 Valor original: {precio_original} COP\n"
                f"🎉 Descuento 5 %: -{descuento} COP\n"
                f"✅ Total a pagar: {valor_final} COP\n\n"
                "💳 Cuentas disponibles:\n"
                "- Bancolombia 30300002233 (X100 SAS)\n"
                "- Nequi 3177171171\n"
                "- Daviplata 3004141021\n\n"
                "📸 Envía la foto del comprobante aquí."
            )

            print("🧪 MENSAJE A ENVIAR:\n", msg)
            await ctx.bot.send_message(chat_id=cid, text=msg)
            print("✅ MENSAJE ENVIADO (transferencia)")
            return

        else:
            est["fase"] = "esperando_comprobante"
            resumen["Pago"] = "Contra entrega"
            resumen["Valor Anticipo"] = 35000
            estado_usuario[cid] = est

            msg = (
                "🟡 Elegiste CONTRAENTREGA.\n"
                "Debes adelantar 35 000 COP para el envío (se descuenta del total).\n\n"
                "💳 Cuentas disponibles:\n"
                "- Bancolombia 30300002233 (X100 SAS)\n"
                "- Nequi 3177171171\n"
                "- Daviplata 3004141021\n\n"
                "📸 Envía la foto del comprobante aquí."
            )

            print("💬 Enviando mensaje:\n", msg)
            await ctx.bot.send_message(chat_id=cid, text=msg)
            return

    # ------------------------------------------------------------------------
    # 📸 Recibir comprobante de pago
    # ------------------------------------------------------------------------
    if est.get("fase") == "esperando_comprobante" and update.message.photo:
        f = await update.message.photo[-1].get_file()
        tmp = os.path.join("temp", f"{cid}_proof.jpg")
        os.makedirs("temp", exist_ok=True)
        await f.download_to_drive(tmp)

        # OCR con Google Cloud Vision
        with io.open(tmp, "rb") as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        response = vision_client.text_detection(image=image)
        textos_detectados = response.text_annotations

        texto_extraido = textos_detectados[0].description if textos_detectados else ""
        print("🧾 TEXTO EXTRAÍDO:\n", texto_extraido)

        # Verificación básica del comprobante
        if not es_comprobante_valido(texto_extraido):
            await ctx.bot.send_message(
                chat_id=cid,
                text="⚠️ El comprobante no parece válido. Asegúrate de que sea legible y que diga 'Pago exitoso' o 'Transferencia realizada'."
            )
            os.remove(tmp)
            return

        # Si es válido, continuar flujo
        resumen = est["resumen"]
        registrar_orden(resumen)

        enviar_correo(
            est["correo"],
            f"Pago recibido {resumen['Número Venta']}",
            json.dumps(resumen, indent=2)
        )
        enviar_correo_con_adjunto(
            EMAIL_JEFE,
            f"Comprobante {resumen['Número Venta']}",
            json.dumps(resumen, indent=2),
            tmp
        )

        os.remove(tmp)

        await ctx.bot.send_message(
            chat_id=cid,
            text="✅ ¡Pago registrado exitosamente! Tu pedido está en proceso. 🚚"
        )

        reset_estado(cid)
        estado_usuario.pop(cid, None)
        return


    # 🚚 Rastrear pedido
    if est.get("fase") == "esperando_numero_rastreo":
        await ctx.bot.send_message(
            chat_id=cid,
            text="📦 Puedes rastrear tu pedido aquí:\nhttps://www.instagram.com/juanp_ocampo/",
            parse_mode="Markdown"
        )
        reset_estado(cid)
        return

    # 🔄 Solicitud de devolución
    if est.get("fase") == "esperando_numero_devolucion":
        est["referencia"] = txt_raw.strip()
        est["fase"] = "esperando_motivo_devolucion"
        await ctx.bot.send_message(
            chat_id=cid,
            text="📝 ¿Cuál es el motivo de la devolución?",
            parse_mode="Markdown"
        )
        return

    if est.get("fase") == "esperando_motivo_devolucion":
        enviar_correo(
            EMAIL_DEVOLUCIONES,
            f"Solicitud de Devolución {NOMBRE_NEGOCIO}",
            f"Venta: {est['referencia']}\nMotivo: {txt_raw}"
        )
        await ctx.bot.send_message(
            chat_id=cid,
            text="✅ Solicitud de devolución enviada exitosamente.",
            parse_mode="Markdown"
        )
        reset_estado(cid)
        return

    # 🎙️ Procesar audio
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
        txt = normalize(txt_raw)

    # 🖼️ Intención global de imagen
    if menciona_imagen(txt):
        if est.get("fase") != "esperando_imagen":
            est["fase"] = "esperando_imagen"
            await update.message.reply_text(CLIP_INSTRUCTIONS, reply_markup=ReplyKeyboardRemove())
        return

    # 💬 Manejar precio por referencia
    if await manejar_precio(update, ctx, inv):
        return

    # 🛒 Confirmación para continuar con compra después de ver precios
    if est.get("fase") == "confirmar_compra":
        if txt in ("si", "sí", "si quiero comprar", "sí quiero comprar", "quiero comprar", "comprar", "dale", "SI", "De una", "claro"):
            modelo = est.get("modelo_confirmado")
            color_confirmado = est.get("color_confirmado")

            if not modelo:
                await ctx.bot.send_message(
                    chat_id=cid,
                    text="❌ No encontré el modelo que seleccionaste. Vuelve a escribirlo o envíame una imagen.",
                    parse_mode="Markdown"
                )
                est["fase"] = "inicio"
                estado_usuario[cid] = est
                return

            est["modelo"] = modelo

            colores = obtener_colores_por_modelo(inv, modelo)
            if isinstance(colores, (int, float, str)):
                colores = [str(colores)]

            if len(colores) == 1:
                est["color"] = colores[0]
                est["fase"] = "esperando_talla"
                tallas = obtener_tallas_por_color(inv, modelo, colores[0])
                if isinstance(tallas, (int, float, str)):
                    tallas = [str(tallas)]

                await ctx.bot.send_message(
                    chat_id=cid,
                    text=f"Perfecto 🎯 ¿Qué talla deseas para el modelo *{modelo}* color *{colores[0]}*?\n👉 Tallas disponibles: {', '.join(tallas)}",
                    parse_mode="Markdown"
                )
            else:
                est["fase"] = "esperando_color"
                colores_str = "\n".join(f"- {c}" for c in colores)
                await ctx.bot.send_message(
                    chat_id=cid,
                    text=f"🎨 El modelo *{modelo}* está disponible en varios colores:\n\n{colores_str}\n\n¿Cuál color te interesa?",
                    parse_mode="Markdown"
                )

            estado_usuario[cid] = est
            return

        else:
            await ctx.bot.send_message(
                chat_id=cid,
                text="No hay problema. Si deseas, puedes ver nuestro catálogo completo 📋.\n👉 https://wa.me/c/573246666652",
                parse_mode="Markdown"
            )
            est["fase"] = "inicio"
            estado_usuario[cid] = est
            return


    # 🔄 Si el usuario dice “sí” antes de confirmar compra (por fuera del flujo)
    if "sí" in txt_raw or "claro" in txt_raw or "dale" in txt_raw or "quiero" in txt_raw:
        modelo = est.get("modelo_confirmado")
        if not modelo:
            await ctx.bot.send_message(
                chat_id=cid,
                text="❌ No encontré el modelo que seleccionaste. Vuelve a escribirlo o envíame una imagen.",
                parse_mode="Markdown"
            )
            est["fase"] = "inicio"
            return

        colores = obtener_colores_por_modelo(inventario, modelo)

        # Caso único color
        if len(colores) == 1:
            est["modelo"] = modelo
            est["color"] = colores[0]
            est["fase"] = "esperando_talla"

            tallas = obtener_tallas_por_color(inventario, modelo, colores[0])
            await ctx.bot.send_message(
                chat_id=cid,
                text=f"Perfecto 👌 ¿Qué talla deseas para el modelo *{modelo}* color *{colores[0]}*? 👟📏",
                parse_mode="Markdown",
                reply_markup=menu_botones(tallas),
            )
            return

        # Caso múltiples colores
        est["modelo"] = modelo
        est["fase"] = "esperando_color"

        await ctx.bot.send_message(
            chat_id=cid,
            text=f"🎨 ¿Qué color deseas para el modelo *{modelo}*?",
            parse_mode="Markdown",
            reply_markup=menu_botones(colores),
        )
        return


    # 🔒 Video de confianza si desconfía
    if any(frase in txt for frase in (
        "no me fio", "no confio", "es seguro", "como se que no me roban",
        "como se que no me estafan", "desconfio", "no creo", "estafa", "miedo a comprar"
    )):
        VIDEO_DRIVE_ID = "TU_ID_DEL_VIDEO_DE_CONFIANZA"
        video_url = f"https://drive.google.com/uc?id={VIDEO_DRIVE_ID}"

        await ctx.bot.send_chat_action(chat_id=cid, action=ChatAction.UPLOAD_VIDEO)
        await ctx.bot.send_video(
            chat_id=cid,
            video=video_url,
            caption=(
                "🔒 Entendemos perfectamente tu preocupación. "
                "Aquí te dejamos un video corto donde nuestros clientes reales comparten su experiencia. "
                "Somos una empresa seria y segura, ¡puedes confiar en nosotros! 😊👍"
            )
        )
        return

    # 🎬 Si pide videos normales
    if any(frase in txt for frase in ("videos", "quiero videos", "ver videos")):
        await ctx.bot.send_message(
            chat_id=cid,
            text=(
                "🎬 ¡Claro! Aquí tienes videos de nuestras referencias más populares:\n\n"
                "• DS 277: https://drive.google.com/file/d/1W7nMJ4RRYUvr9LiPDe5p_U6Mg_azyHLN/view?usp=drive_link\n"
                "• DS 288: https://youtu.be/ID_DEL_VIDEO_288\n"
                "• DS 299: https://youtu.be/ID_DEL_VIDEO_299\n\n"
                "¿Cuál te gustaría ver?"
            ),
            reply_markup=menu_botones(["DS 277", "DS 288", "DS 299"]),
            parse_mode="Markdown"
        )
        est["fase"] = "esperando_video_referencia"
        return

    # 🎬 Esperar selección de video
    if est.get("fase") == "esperando_video_referencia":
        await enviar_video_referencia(cid, ctx, txt)
        est["fase"] = "inicio"
        return

    if await manejar_precio(update, ctx, inv):
        return

    if await manejar_catalogo(update, ctx):
        return

    # 🖼️ Procesar imagen subida si estaba esperando
    if est.get("fase") == "esperando_imagen" and update.message.photo:
        f = await update.message.photo[-1].get_file()
        tmp = os.path.join("temp", f"{cid}.jpg")
        os.makedirs("temp", exist_ok=True)
        await f.download_to_drive(tmp)

        ref_detectada = identify_model_from_stream(tmp)
        os.remove(tmp)

        if ref_detectada:
            marca, modelo, color = ref_detectada.split('_')
            est.update({
                "marca": marca,
                "modelo": modelo,
                "color": color,
                "fase": "imagen_detectada"
            })
            await ctx.bot.send_message(
                chat_id=cid,
                text=f"La imagen coincide con {marca} {modelo} color {color}. ¿Continuamos? (SI/NO)",
                reply_markup=menu_botones(["SI", "NO"]),
            )
        else:
            reset_estado(cid)
            await ctx.bot.send_message(
                chat_id=cid,
                text="😕 No reconocí el modelo en la imagen. ¿Puedes intentar otra imagen o escribir /start?",
                parse_mode="Markdown"
            )
        return

    # 🛍️ Detectar marca escrita
    marcas = obtener_marcas_unicas(inv)
    elegida = next((m for m in marcas if any(t in txt for t in normalize(m).split())), None)

    if not elegida:
        tokens = txt.split()
        for m in marcas:
            for tok in normalize(m).split():
                if difflib.get_close_matches(tok, tokens, n=1, cutoff=0.6):
                    elegida = m
                    break
            if elegida:
                break

    if elegida:
        est["marca"] = elegida
        est["fase"] = "esperando_modelo"
        await ctx.bot.send_message(
            chat_id=cid,
            text=f"¡Genial! Veo que buscas {elegida}. ¿Qué modelo de {elegida} te interesa?",
            reply_markup=menu_botones(obtener_modelos_por_marca(inv, elegida)),
        )
        return




     # ——— Hasta aquí llega todo el flujo normal del bot ———

    # 🔥 Fallback inteligente CORREGIDO con 4 espacios

    # 1) Detectar palabras típicas primero (antes que IA)
    palabras_clave_flujo = [
        "catalogo", "catálogo", "ver catálogo", "ver catalogo",
        "imagen", "foto", "enviar imagen", "ver tallas",
        "quiero comprar", "hacer pedido", "comprar", "zapatos", "tenis",
        "pago", "contraentrega", "garantía", "garantia",
        "demora", "envío", "envio"
    ]

    if any(palabra in txt for palabra in palabras_clave_flujo):
        await ctx.bot.send_message(
            chat_id=cid,
            text="📋 Parece que quieres hacer un pedido o consultar el catálogo. Usa las opciones disponibles 😉",
            reply_markup=menu_botones(["Hacer pedido", "Ver catálogo", "Enviar imagen"])
        )
        return

    # 2) NO usar IA si estamos en una fase crítica (proteger cierre de venta)
    fases_criticas = [
        "esperando_talla", "esperando_color", "esperando_nombre", "esperando_correo",
        "esperando_telefono", "esperando_ciudad", "esperando_provincia", "esperando_direccion",
        "esperando_pago", "esperando_comprobante", "imagen_detectada",
        "esperando_video_referencia", "esperando_numero_rastreo",
        "esperando_numero_devolucion", "esperando_motivo_devolucion"
    ]

    if est.get("fase") in fases_criticas:
        await ctx.bot.send_message(
            chat_id=cid,
            text="✏️ Por favor completa primero el proceso en el que estás. ¿Te ayudo a terminarlo?",
            reply_markup=menu_botones(["Volver al menú"])
        )
        return

    # 3) Ahora sí, usar IA si no entendimos nada
    respuesta_fallback = await responder_con_openai(txt_raw)
    if respuesta_fallback:
        await ctx.bot.send_message(
            chat_id=cid,
            text=respuesta_fallback,
            reply_markup=menu_botones(["Hacer pedido", "Ver catálogo", "Enviar imagen"])
        )
    else:
        await ctx.bot.send_message(
            chat_id=cid,
            text="😅 No logré entender tu solicitud. ¿Quieres ver el catálogo o realizar un pedido?",
            reply_markup=menu_botones(["Hacer pedido", "Ver catálogo"])
        )
    return

# Función para manejar la solicitud de precio por referencia
PALABRAS_PRECIO = ['precio', 'vale', 'cuesta', 'valor', 'coste', 'precios', 'cuánto']

async def manejar_precio(update, ctx, inventario):
    import logging
    cid = update.effective_chat.id
    mensaje = (update.message.text or "").lower()
    txt = normalize(mensaje)
    logging.debug(f"[manejar_precio] Mensaje recibido: {mensaje}")

    # Detectar referencia de 3 o 4 dígitos
    m_ref = re.search(r"(?:referencia|modelo)?\s*(\d{3,4})", txt)
    if not m_ref:
        logging.debug("[manejar_precio] No se detectó referencia en el mensaje.")
        return False

    referencia = m_ref.group(1)
    logging.debug(f"[manejar_precio] Referencia detectada: {referencia}")

    # Buscar productos que coincidan
    productos = [
        item for item in inventario
        if referencia in normalize(item.get("modelo", "")) and disponible(item)
    ]
    logging.debug(f"[manejar_precio] Productos encontrados con stock: {len(productos)}")

    if productos:
        from collections import defaultdict
        agrupados = defaultdict(set)

        for item in productos:
            try:
                precio_raw = str(item.get("precio", "0")).replace(".", "").replace("COP", "").strip()
                precio_formateado = f"{int(precio_raw):,}COP"
            except Exception as e:
                logging.error(f"[manejar_precio] Error formateando precio: {e}")
                precio_formateado = "No disponible"

            try:
                key = (
                    item.get("modelo", "desconocido"),
                    item.get("color", "varios colores"),
                    precio_formateado
                )
                agrupados[key].add(str(item.get("talla", "")))
            except Exception as e:
                logging.error(f"[manejar_precio] Error agrupando tallas: {e}")
        
        respuesta_final = ""
        primer_producto = productos[0]

        for (modelo, color, precio), tallas in agrupados.items():
            try:
                if not isinstance(tallas, (set, list, tuple)):
                    tallas = [str(tallas)]

                tallas_ordenadas = sorted(tallas, key=lambda t: int(t) if t.isdigit() else t)
                tallas_str = ", ".join(tallas_ordenadas)

                respuesta_final += (
                    f"👟 *{modelo}* ({color})\n"
                    f"💲 Precio: *{precio}*\n"
                    f"Tallas disponibles: {tallas_str}\n\n"
                )
            except Exception as e:
                logging.error(f"[manejar_precio] Error formateando tallas: {e}")

        # ✅ Guardar estado de forma segura
        est = estado_usuario.get(cid, {})
        est["fase"] = "confirmar_compra"
        est["modelo_confirmado"] = primer_producto["modelo"]
        est["color_confirmado"] = primer_producto["color"]
        est["marca"] = primer_producto.get("marca", "sin marca")
        estado_usuario[cid] = est  # ✅ GUARDA el estado correctamente

        logging.debug(f"[manejar_precio] Guardado modelo: {primer_producto['modelo']}, color: {primer_producto['color']}")

        await ctx.bot.send_message(
            chat_id=cid,
            text=(
                f"Veo que estás interesado en nuestra referencia *{referencia}*:\n\n"
                f"{respuesta_final}"
                "¿Te gustaría proseguir con la compra?\n\n"
                "👉 Escribe: *sí quiero comprar* o *no, gracias*"
            ),
            parse_mode="Markdown"
        )
        return True

    else:
        logging.debug("[manejar_precio] No se encontraron productos con esa referencia.")
        await ctx.bot.send_message(
            chat_id=cid,
            text=(
                f"😕 No encontré la referencia {referencia}. "
                "¿Quieres revisar el catálogo?\n\n"
                "👉 Opciones: Ver catálogo / Volver al menú"
            ),
            parse_mode="Markdown"
        )
        return True





   
# --------------------------------------------------------------------

nest_asyncio.apply()

# 1. Instancia FastAPI
api = FastAPI(title="AYA Bot – WhatsApp")

# 2. Conversión de número WhatsApp
def wa_chat_id(wa_from: str) -> str:
    return re.sub(r"\D", "", wa_from)

async def responder_con_openai(mensaje_usuario):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    try:
        respuesta = await openai.ChatCompletion.acreate(
            model="gpt-4-1106-preview",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres un asesor de ventas de la tienda de zapatos deportivos 'X100🔥👟'. "
                        "Solo vendemos nuestra propia marca *X100* (no manejamos marcas como Skechers, Adidas, Nike, etc.). "
                        "Nuestros productos son 100% colombianos 🇨🇴 y hechos en Bucaramanga.\n\n"
                        "Tu objetivo principal es:\n"
                        "- Ayudar al cliente a consultar el catálogo 📋\n"
                        "- Siempre que puedas pedir la referencia del teni\n"
                        "- Pedir que envíe una imagen del zapato que busca 📸\n"
                        "Siempre que puedas, invita amablemente al cliente a enviarte el número de referencia o una imagen para agilizar el pedido.\n"
                        "Si el cliente pregunta por marcas externas, responde cálidamente explicando que solo manejamos X100.\n\n"
                        "Cuando no entiendas muy bien la intención, ofrece opciones como:\n"
                        "- '¿Te gustaría ver nuestro catálogo? 📋'\n"
                        "- '¿Me puedes enviar la referencia del modelo que te interesa? 📋✨'\n"
                        "- '¿Quieres enviarme una imagen para ayudarte mejor? 📸'\n\n"
                        "Responde de forma CÁLIDA, POSITIVA, BREVE (máximo 2 o 3 líneas), usando emojis amistosos 🎯👟🚀✨.\n"
                        "Actúa como un asesor de ventas que siempre busca ayudar al cliente y CERRAR la compra de manera rápida, amigable y eficiente."
                    )
                },
                {
                    "role": "user",
                    "content": mensaje_usuario
                }
            ],
            temperature=0.5,
            max_tokens=300
        )
        return respuesta['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error al consultar OpenAI: {e}")
        return "Disculpa, estamos teniendo un inconveniente en este momento. ¿Puedes intentar de nuevo más tarde?"


# 4. Procesar mensaje de WhatsApp
async def procesar_wa(cid: str, body: str) -> dict:
    cid = str(cid)  # 🔐 Asegura que el ID sea string SIEMPRE
    texto = body.lower() if body else ""
    txt = texto if texto else ""

    class DummyCtx(SimpleNamespace):
        async def bot_send(self, chat_id, text, **kw): self.resp.append(text)
        async def bot_send_chat_action(self, chat_id, action, **kw): pass
        async def bot_send_video(self, chat_id, video, caption=None, **kw): self.resp.append(f"[VIDEO] {caption or ' '}]")

    # 👇 Define ctx primero
    ctx = DummyCtx(resp=[])

    # 👇 Ahora sí define ctx.bot correctamente
    ctx.bot = SimpleNamespace(
        send_message=ctx.bot_send,
        send_chat_action=ctx.bot_send_chat_action,
        send_video=ctx.bot_send_video
    )

    class DummyMsg(SimpleNamespace):
        def __init__(self, text, ctx, photo=None, voice=None, audio=None):
            self.text = text
            self.photo = photo
            self.voice = voice
            self.audio = audio
            self._ctx = ctx

        async def reply_text(self, text, **kw):
            self._ctx.resp.append(text)

    dummy_msg = DummyMsg(text=body, ctx=ctx, photo=None, voice=None, audio=None)
    dummy_update = SimpleNamespace(
        message=dummy_msg,
        effective_chat=SimpleNamespace(id=cid)
    )

    # 🧠 Revisa si el estado no existe o está vacío
    if cid not in estado_usuario or not estado_usuario[cid].get("fase"):
        reset_estado(cid)
        estado_usuario[cid] = {"fase": "inicio"}

    try:
        await responder(dummy_update, ctx)

        if ctx.resp:
            print(f"[DEBUG] BOT respondió correctamente: {ctx.resp}")
            return {"type": "text", "text": "\n".join(ctx.resp)}
        else:
            est = estado_usuario.get(cid, {})
            if est.get("fase") in ("esperando_pago", "esperando_comprobante"):
                print("[DEBUG] Fase crítica: el bot no respondió pero no se usará IA.")
                return {"type": "text", "text": "💬 Estoy esperando que confirmes tu método de pago o me envíes el comprobante. 📸"}

            print(f"[DEBUG] BOT no respondió nada, se usará IA para el mensaje: {body}")
            respuesta_ia = await responder_con_openai(body)
            return {"type": "text", "text": respuesta_ia}

    except Exception as e:
        print(f"🔥 Error interno en procesar_wa(): {e}")
        print(f"[DEBUG] Usando IA como fallback por error de bot en mensaje: {body}")
        respuesta_ia = await responder_con_openai(body)
        return {"type": "text", "text": respuesta_ia}
@api.post("/venom")
async def venom_webhook(req: Request):
    try:
        # 1️⃣ Leer JSON
        data = await req.json()
        cid      = wa_chat_id(data.get("from", ""))
        body     = data.get("body", "") or ""
        mtype    = (data.get("type") or "").lower()
        mimetype = (data.get("mimetype") or "").lower()

        logging.info(f"📩 Mensaje recibido — CID: {cid} — Tipo: {mtype} — MIME: {mimetype}")

        # 2️⃣ Procesar texto
        if mtype == "chat":
            fase_actual = estado_usuario.get(cid, {}).get("fase", "")
            logging.info(f"💬 Texto recibido en fase: {fase_actual or 'NO DEFINIDA'}")
            reply = await procesar_wa(cid, body)
            return JSONResponse(reply)

        # 3️⃣ Procesar imagen
        elif mtype == "image" or mimetype.startswith("image"):
            try:
                logging.info("🖼️ Imagen recibida. Decodificando base64...")
                b64_str = body.split(",", 1)[1] if "," in body else body
                img_bytes = base64.b64decode(b64_str + "===")
                os.makedirs("temp", exist_ok=True)
                path_local = f"temp/{cid}_img.jpg"
                with open(path_local, "wb") as f:
                    f.write(img_bytes)
                logging.info(f"✅ Imagen guardada temporalmente en {path_local}")
            except Exception as e:
                logging.error(f"❌ Error al guardar imagen base64: {e}")
                return JSONResponse({"type": "text", "text": "❌ No pude leer la imagen 😕"})

            est = estado_usuario.get(cid, {})
            fase = est.get("fase", "")
            logging.info(f"🔍 Fase actual del usuario {cid}: {fase or 'NO DEFINIDA'}")

            # 4️⃣ Si espera comprobante
            if fase == "esperando_comprobante":
                logging.info("🧾 Fase: esperando_comprobante — Ejecutando OCR")
                texto = extraer_texto_comprobante(path_local)
                logging.info(f"📃 Texto OCR:\n{texto[:300]}")

                if es_comprobante_valido(texto):
                    logging.info("✅ OCR válido. Se registra orden.")
                    resumen = est.get("resumen", {})
                    registrar_orden(resumen)

                    enviar_correo(
                        est["correo"],
                        f"Pago recibido {resumen.get('Número Venta')}",
                        json.dumps(resumen, indent=2)
                    )
                    enviar_correo_con_adjunto(
                        EMAIL_JEFE,
                        f"Comprobante {resumen.get('Número Venta')}",
                        json.dumps(resumen, indent=2),
                        path_local
                    )
                    os.remove(path_local)
                    reset_estado(cid)
                    return JSONResponse({"type": "text", "text": "✅ Comprobante verificado. Tu pedido está en proceso. 🚚"})
                else:
                    logging.warning("⚠️ OCR no válido. No se reconoció 'Pago exitoso'.")
                    os.remove(path_local)
                    return JSONResponse({"type": "text", "text": "⚠️ No pude verificar el comprobante. Asegúrate que diga 'Pago exitoso' o 'Transferencia exitosa'."})

            # 5️⃣ Si no está esperando comprobante
            logging.info("📸 Fase no es 'esperando_comprobante'. Intentando reconocimiento por hash.")
            try:
                img = Image.open(path_local)
                h_in = str(imagehash.phash(img))
                ref = MODEL_HASHES.get(h_in)
                logging.info(f"🧠 Hash de imagen: {h_in} → Resultado: {ref}")
            except Exception as e:
                logging.error(f"❌ Error al calcular hash de imagen: {e}")
                return JSONResponse({"type": "text", "text": "😕 No pude procesar la imagen."})

            if ref:
                marca, modelo, color = ref
                logging.info(f"🎯 Imagen reconocida como {marca} {modelo} {color}")
                estado_usuario.setdefault(cid, reset_estado(cid))
                estado_usuario[cid].update(
                    fase="imagen_detectada", marca=marca, modelo=modelo, color=color
                )
                os.remove(path_local)
                return JSONResponse({
                    "type": "text",
                    "text": f"La imagen coincide con {marca} {modelo} color {color}. ¿Deseas continuar tu compra? (SI/NO)"
                })
            else:
                logging.warning("🚫 Imagen no reconocida. Se reinicia estado.")
                os.remove(path_local)
                reset_estado(cid)
                return JSONResponse({"type": "text", "text": "😕 No reconocí el modelo. Puedes intentarlo con otra imagen."})

        # 6️⃣ Mensaje no manejado
        logging.warning(f"🤷‍♂️ Tipo de mensaje no manejado: {mtype}")
        return JSONResponse({
            "type": "text",
            "text": f"⚠️ Tipo de mensaje no manejado: {mtype}"
        })

    except Exception as e:
        logging.exception("🔥 Error general en venom_webhook")
        return JSONResponse(
            {"type": "text", "text": "⚠️ Error interno procesando el mensaje."},
            status_code=500
        )
# -------------------------------------------------------------------------
# 5. Arranque del servidor
# -------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn, os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(api, host="0.0.0.0", port=port)


