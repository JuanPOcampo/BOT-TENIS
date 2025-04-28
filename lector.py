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

# ——— Librerías externas ———
from dotenv import load_dotenv
from PIL import Image
import imagehash
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
import nest_asyncio
import openai
# Google Drive
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

    est = estado_usuario[cid]                  # ← ya existe
    inv = obtener_inventario()                 # ← cache de inventario

    if update.message.photo:                   # ← ¡Ahora sí bien!
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
            await ctx.bot.send_message(
                chat_id=cid,
                text=f"La imagen coincide con {marca} {modelo} color {color}. ¿Continuamos? (SI/NO)",
                reply_markup=menu_botones(["SI", "NO"]),
            )
        else:
            reset_estado(cid)
            await ctx.bot.send_message(
                chat_id=cid,
                text="No reconocí el modelo. /start para reiniciar.",
                parse_mode="Markdown"
            )
        return


     # Confirmación imagen
    if est["fase"] == "imagen_detectada":
        if txt in ("si", "s"):
            est["fase"] = "esperando_talla"
            await ctx.bot.send_message(
                chat_id=cid,
                text="¿Qué talla deseas?",
                reply_markup=menu_botones(
                    obtener_tallas_por_color(inv, est["marca"], est["modelo"], est["color"])
                ),
            )
        else:
            await ctx.bot.send_message(
                chat_id=cid,
                text="Cancelado. /start para reiniciar.",
                parse_mode="Markdown"
            )
            reset_estado(cid)
        return

    # Selección manual de modelo/color/talla
    if est["fase"] == "esperando_modelo":
        modelos = obtener_modelos_por_marca(inv, est["marca"])
        if txt in map(normalize, modelos):
            est["modelo"] = next(m for m in modelos if normalize(m) == txt)
            est["fase"] = "esperando_color"
            await ctx.bot.send_message(
                chat_id=cid,
                text="¿Qué color deseas?",
                reply_markup=menu_botones(
                    obtener_colores_por_modelo(inv, est["marca"], est["modelo"])
                ),
            )
        else:
            await ctx.bot.send_message(
                chat_id=cid,
                text="Elige un modelo válido.",
                reply_markup=menu_botones(modelos),
            )
        return

    if est["fase"] == "esperando_color":
        colores = obtener_colores_por_modelo(inv, est["marca"], est["modelo"])
        if txt in map(normalize, colores):
            est["color"] = next(c for c in colores if normalize(c) == txt)
            est["fase"] = "esperando_talla"
            tallas = obtener_tallas_por_color(inv, est["marca"], est["modelo"], est["color"])
            await ctx.bot.send_message(
                chat_id=cid,
                text=f"Las tallas disponibles para {est['modelo']} color {est['color']} son: {', '.join(tallas)}",
                parse_mode="Markdown"
            )
            await ctx.bot.send_message(
                chat_id=cid,
                text="¿Qué talla deseas?",
                reply_markup=menu_botones(tallas),
            )
        else:
            await ctx.bot.send_message(
                chat_id=cid,
                text=f"Los colores disponibles para {est['modelo']} son:\n" +
                     "\n".join(f"- {c}" for c in colores),
                parse_mode="Markdown"
            )
            await ctx.bot.send_message(
                chat_id=cid,
                text="¿Cuál color te interesa?",
                reply_markup=menu_botones(colores),
            )
        return

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

    # 5) Manejar precio por referencia ────────────────────────────
    if await manejar_precio(update, ctx, inv):
        return

# Función para manejar la solicitud de precio por referencia
PALABRAS_PRECIO = ['precio', 'vale', 'cuesta', 'valor', 'coste', 'precios', 'cuánto']

async def manejar_precio(update, ctx, inventario):
    cid = update.effective_chat.id
    mensaje = (update.message.text or "").lower()

    # Busca números en el mensaje (referencias numéricas)
    numeros = re.findall(r"\b\d+\b", mensaje)

    if numeros:
        referencia = numeros[0]  # toma solo la primera referencia detectada
        referencia_normalizada = normalize(referencia)

        # Busca el modelo en inventario
        modelos_encontrados = [
            item for item in inventario
            if referencia_normalizada in normalize(item.get("modelo", "")) and disponible(item)
        ]

        if modelos_encontrados:
            respuestas = []
            for modelo in modelos_encontrados:
                marca = modelo.get('marca', 'desconocida')
                modelo_nombre = modelo.get('modelo', 'desconocido')
                color = modelo.get('color', 'varios colores')
                precio = modelo.get('precio', 'No disponible')
                respuestas.append(f"✅ {marca} {modelo_nombre} ({color}) cuesta {precio}.")

            respuesta_final = "\n".join(respuestas)
            await ctx.bot.send_message(
                chat_id=cid,
                text=respuesta_final,
                reply_markup=menu_botones(["Hacer pedido", "Ver catálogo", "Enviar imagen"]),
                parse_mode="Markdown"
            )
            return True
        else:
            await ctx.bot.send_message(
                chat_id=cid,
                text=f"No encontré productos con la referencia '{referencia}'. ¿Quieres revisar el catálogo?",
                reply_markup=menu_botones(["Ver catálogo", "Volver al menú"]),
                parse_mode="Markdown"
            )
            return True
    return False



    # 🔥 NUEVO: Envío de video automático cuando el cliente muestra desconfianza
    if any(frase in txt for frase in (
        "no me fio", "no confio", "es seguro", "como se que no me roban",
        "como se que no me estafan", "desconfio", "no creo", "estafa", "miedo a comprar"
    )):
        VIDEO_DRIVE_ID = "TU_ID_DEL_VIDEO_DE_CONFIANZA"  # ← recuerda poner el ID real
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

# 🔥 Función fallback inteligente
async def fallback_inteligente(txt, update):
    respuesta_ia = await consultar_ia_fallback(txt)
    if respuesta_ia:
        await ctx.bot.send_message(
    chat_id=update.effective_chat.id,
    text=respuesta_ia,
            reply_markup=menu_botones(["Ver catálogo", "Hacer pedido", "Enviar imagen"],
)
)
    else:
        await ctx.bot.send_message(
    chat_id=update.effective_chat.id,
    text="😔 No logré entenderte bien. ¿Quieres volver al inicio?",
            reply_markup=menu_botones(["Volver al menú"],
    parse_mode="Markdown"
)
)

    # 🔥 Validar si realmente se extrajo texto
    if not txt_raw:
        await ctx.bot.send_message(
            chat_id=cid,
            text="❗ No recibí un mensaje válido. ¿Puedes enviarme texto o un audio claro?",
            reply_markup=ReplyKeyboardRemove(),
            parse_mode="Markdown"
        )
        return

    txt = normalize(txt_raw)

    # 🔥 1) Reinicio explícito primero
    if txt in ("reset", "reiniciar", "empezar", "volver", "/start", "menu", "inicio"):
        reset_estado(cid)
        await saludo_bienvenida(update, ctx)
        return

    # 🔥 2) Si estaba esperando que elija un video
    if est.get("fase") == "esperando_video_referencia":
        await enviar_video_referencia(cid, ctx, txt)
        est["fase"] = "inicio"
        return

    # 🔥 3) Ofrecer videos si los pidió
    if any(frase in txt for frase in (
        "videos", "tienen videos", "muéstrame videos", "videos de referencia",
        "ver videos", "quiero videos", "puedes mandarme videos"
    )):
        await ctx.bot.send_message(
            chat_id=cid,
            text="🎬 ¡Claro! Tengo videos de nuestras referencias más populares:\n\n"
                 "• DS 277\n"
                 "• DS 288\n"
                 "• DS 299\n\n"
                 "Dime cuál te gustaría ver o escribe directamente la referencia.",
            reply_markup=menu_botones(["DS 277", "DS 288", "DS 299"]),
            parse_mode="Markdown"
        )
        est["fase"] = "esperando_video_referencia"
        return


    # 🔥 4) Detección automática de referencia (modelo o referencia de 3-4 dígitos)
    m_ref = re.search(r"(?:referencia|modelo)?\s*(\d{3,4})", txt)
    if m_ref:
        referencia = m_ref.group(1)
        inventario = obtener_inventario()

        productos = [
            it for it in inventario
            if referencia in normalize(it.get("modelo", ""))
            and disponible(it)
        ]

        if productos:
            agrupados = {}
            for it in productos:
                key = (it['modelo'], it['color'], it['precio'])
                agrupados.setdefault(key, []).append(str(it.get('talla', "")))

            respuesta = "💰 Encontré esto para ti:\n\n"
            for (modelo, color, precio), tallas in agrupados.items():
                tallas_ordenadas = sorted(tallas, key=lambda x: int(x) if x.isdigit() else x)
                respuesta += (
                    f"👟 {modelo} ({color}) — *{precio}*\n"
                    f"Tallas disponibles: {', '.join(tallas_ordenadas)}\n\n"
                )

            respuesta += "¿Quieres pedir alguno de estos modelos?"
            await ctx.bot.send_message(
    chat_id=update.effective_chat.id,
    text=respuesta,
                reply_markup=menu_botones(["Sí, quiero pedir", "Volver al menú"],
)
)
            est["fase"] = "inicio"
            return
        else:
            await ctx.bot.send_message(
    chat_id=update.effective_chat.id,
    text=f"😕 No encontré la referencia {referencia}. ¿Quieres intentar con otra?",
                reply_markup=menu_botones(["Volver al menú"],
)
)
            return

    # 🔥 5) Procesar imágenes (cuando el usuario menciona "imagen")
    if menciona_imagen(txt):
        if est.get("fase") != "esperando_imagen":
            est["fase"] = "esperando_imagen"
            await ctx.bot.send_message(
                chat_id=cid,
                text=CLIP_INSTRUCTIONS,
                reply_markup=ReplyKeyboardRemove(),
                parse_mode="Markdown"
            )
        return

    # 🔥 6) Identificar modelo a partir de la foto recibida
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

    # 🔥 7) Solicitud explícita de precios usando helper manejar_precio
    if await manejar_precio(update, ctx, inventario):
        return

    # 🔥 8) Detección de marca escrita por el usuario
    marcas = obtener_marcas_unicas(inventario)
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
            reply_markup=menu_botones(obtener_modelos_por_marca(inventario, elegida)),
        )
        return


    # ——— Rastrear pedido ———
    if est.get("fase") == "esperando_numero_rastreo":
        await ctx.bot.send_message(
            chat_id=cid,
            text="Guía para rastrear: https://www.instagram.com/juanp_ocampo/",
            parse_mode="Markdown"
        )
        reset_estado(cid)
        return


    # ——— Devolución ———
    if est.get("fase") == "esperando_numero_devolucion":
        est["referencia"] = txt_raw.strip()
        est["fase"] = "esperando_motivo_devolucion"
        await ctx.bot.send_message(
            chat_id=cid,
            text="Motivo de devolución:",
            parse_mode="Markdown"
        )
        return

    if est.get("fase") == "esperando_motivo_devolucion":
        enviar_correo(
            EMAIL_DEVOLUCIONES,
            f"Devolución {NOMBRE_NEGOCIO}",
            f"Venta: {est['referencia']}\nMotivo: {txt_raw}"
        )
        await ctx.bot.send_message(
            chat_id=cid,
            text="Solicitud enviada exitosamente. ✅",
            parse_mode="Markdown"
        )
        reset_estado(cid)
        return

    # Datos del usuario y pago
    if est["fase"] == "esperando_talla":
        tallas = obtener_tallas_por_color(inv, est["marca"], est["modelo"], est["color"])
        talla_detectada = detectar_talla(txt_raw, tallas)
        
        if talla_detectada:
            est["talla"] = talla_detectada
            est["fase"] = "esperando_nombre"
            await ctx.bot.send_message(
                chat_id=cid,
                text="¿Tu nombre?",
                parse_mode="Markdown"
            )
        else:
            await ctx.bot.send_message(
                chat_id=cid,
                text=f"Las tallas disponibles para {est['modelo']} color {est['color']} son: {', '.join(tallas)}",
                parse_mode="Markdown"
            )
            await ctx.bot.send_message(
                chat_id=cid,
                text="Elige una talla válida.",
                reply_markup=menu_botones(tallas),
            )
        return

    if est["fase"] == "esperando_nombre":
        est["nombre"] = txt_raw
        est["fase"] = "esperando_correo"
        await ctx.bot.send_message(
            chat_id=cid,
            text="¿Tu correo? 📧",
            parse_mode="Markdown"
        )
        return

    if est["fase"] == "esperando_correo":
        if re.match(r"[^@]+@[^@]+\.[^@]+", txt_raw):
            est["correo"] = txt_raw
            est["fase"] = "esperando_telefono"
            await ctx.bot.send_message(
                chat_id=cid,
                text="¿Tu teléfono? 📱",
                parse_mode="Markdown"
            )
        else:
            await ctx.bot.send_message(
                chat_id=cid,
                text="Correo inválido. 📧 Intenta de nuevo.",
                parse_mode="Markdown"
            )
        return

    if est["fase"] == "esperando_telefono":
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
                text="Teléfono inválido. 📱 Intenta de nuevo.",
                parse_mode="Markdown"
            )
        return

    if est["fase"] == "esperando_ciudad":
        est["ciudad"] = txt_raw
        est["fase"] = "esperando_provincia"
        await ctx.bot.send_message(
            chat_id=cid,
            text="¿Provincia o departamento? 🏞️",
            parse_mode="Markdown"
        )
        return

    if est["fase"] == "esperando_provincia":
        est["provincia"] = txt_raw
        est["fase"] = "esperando_direccion"
        await ctx.bot.send_message(
            chat_id=cid,
            text="¿Dirección de envío? 🏡",
            parse_mode="Markdown"
        )
        return

    if est["fase"] == "esperando_direccion":
        est["direccion"] = txt_raw
        precio = next((i["precio"] for i in inv
                       if normalize(i["marca"]) == normalize(est["marca"])
                       and normalize(i["modelo"]) == normalize(est["modelo"])
                       and normalize(i["color"]) == normalize(est["color"])), "N/A")
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
            "Elige método de pago:"
        )
        await ctx.bot.send_message(
            chat_id=cid,
            text=text_res,
            reply_markup=menu_botones(["TRANSFERENCIA", "QR", "CONTRA ENTREGA"]),
        )
        est["fase"] = "esperando_pago"
        return

    if est["fase"] == "esperando_pago":
        opt = normalize(txt_raw.replace(" ", ""))
        resumen = est["resumen"]
        if opt == "transferencia":
            est["fase"] = "esperando_comprobante"
            resumen["Pago"] = "Transferencia"
            await ctx.bot.send_message(
                chat_id=cid,
                text="Envía la foto de tu comprobante. 📸",
                parse_mode="Markdown"
            )
        elif opt == "qr":
            est["fase"] = "esperando_comprobante"
            resumen["Pago"] = "QR"
            await ctx.bot.send_message(
                chat_id=cid,
                text="Escanea el QR y envía la foto. 📸",
                parse_mode="Markdown"
            )
        elif opt == "contraentrega":
            resumen["Pago"] = "Contra entrega"
            registrar_orden(resumen)
            await ctx.bot.send_message(
                chat_id=cid,
                text="✅ Pedido registrado para *Contra entrega*. ¡Gracias!",
                parse_mode="Markdown"
            )
            reset_estado(cid)
        else:
            await ctx.bot.send_message(
                chat_id=cid,
                text="Método de pago inválido. Elige uno correcto.",
                parse_mode="Markdown"
            )
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

        await ctx.bot.send_message(
            chat_id=cid,
            text="✅ ¡Pago registrado exitosamente! Tu pedido está en proceso. 🚚",
            parse_mode="Markdown"
        )
        reset_estado(cid)
        return

    # ——— Hasta aquí llega todo el flujo normal del bot ———

    # 🔥 1) Fallback inteligente rápido
    respuesta_fallback = await consultar_ia_fallback(txt_raw)
    if respuesta_fallback:
        await ctx.bot.send_message(
            chat_id=cid,
            text=respuesta_fallback,
            reply_markup=menu_botones(["Hacer pedido", "Ver catálogo", "Enviar imagen"])
        )
        return

    # 🔥 2) Detección de palabras clave típicas para guiar al menú
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
            text="📋 Parece que quieres hacer un pedido o consultar el catálogo. Por favor usa las opciones disponibles para continuar. 😉",
            reply_markup=menu_botones(["Hacer pedido", "Ver catálogo", "Enviar imagen"])
        )
        return

    # 🔥 3) Fallback final: usar IA real GPT-4 si no coincidió nada antes
    respuesta = await responder_con_openai(txt_raw)
    await ctx.bot.send_message(
        chat_id=cid,
        text=respuesta,
        reply_markup=menu_botones(["Hacer pedido", "Ver catálogo", "Enviar imagen"])
    )
    return


   
# --------------------------------------------------------------------

nest_asyncio.apply()

# 1. Instancia FastAPI
api = FastAPI(title="AYA Bot – WhatsApp")

# 2. Conversión de número WhatsApp
def wa_chat_id(wa_from: str) -> str:
    return re.sub(r"\D", "", wa_from)

async def responder_con_openai(mensaje_usuario):
    openai.api_key = os.getenv("OPENAI_API_KEY")  # o fija la clave aquí si prefieres

    try:
        respuesta = await openai.ChatCompletion.acreate(
            model="gpt-4-1106-preview",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres un asesor de ventas experto en una tienda de zapatos. "
                        "Responde de forma AMABLE, DIRECTA y en MENOS DE 3 LÍNEAS. "
                        "No repitas demasiado la pregunta del cliente. "
                        "Sé breve y concreto. "
                        "Cuando puedas, invita al cliente a elegir talla o color. "
                        "Habla como un vendedor amigable que quiere cerrar la venta rápido."
                    )
                },
                {"role": "user", "content": mensaje_usuario}
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
    texto = body.lower() if body else ""  # Si no hay texto, asigna vacío a 'texto'
    palabras_genericas = [
        "hola", "buenas", "gracias", "catálogo", "ver catálogo", 
        "hacer pedido", "enviar imagen", "rastrear pedido", "realizar cambio"
    ]

    # Asegurándonos de que 'txt' siempre tenga algo
    txt = texto if texto else ""  # Si texto es nulo o vacío, asigna vacío a 'txt'

    # 1) Preguntas frecuentes (FAQ)
    if any(frase in txt for frase in (
        "cuánto cuesta", "precio", "garantía", "en qué ciudad están", 
        "pago contra entrega", "envío rápido"
    )):
        if "precio" in txt:
            await ctx.bot.send_message(
                chat_id=cid,
                text="💰 El precio depende del modelo y la talla. ¿Qué modelo te interesa?",
                reply_markup=menu_botones(["Ver catálogo", "Hacer pedido"])
            )
            return

        if "garantía" in txt:
            await ctx.bot.send_message(
                chat_id=cid,
                text="🛡️ Todos nuestros productos tienen una garantía de 60 días por defectos de fabricación.",
                reply_markup=menu_botones(["Hacer pedido", "Ver catálogo"])
            )
            return

        if "envío rápido" in txt:
            await ctx.bot.send_message(
                chat_id=cid,
                text="🚚 El tiempo de entrega es generalmente de 2 días hábiles, pero podemos enviarlo más rápido si es necesario.",
                reply_markup=menu_botones(["Hacer pedido", "Ver catálogo"])
            )
            return

    # 🔥 Pregunta 1: ¿cuánto se demora el envío?
    if any(frase in txt for frase in (
        "cuanto demora", "cuánto demora", "cuanto tarda", "cuánto tarda",
        "cuanto se demora", "cuánto se demora", "en cuanto llega", "en cuánto llega",
        "me llega rapido", "llegan rapido"
    )):
        await ctx.bot.send_message(
            chat_id=cid,
            text=(
                "🚚 El tiempo de entrega depende de la ciudad de destino, "
                "pero generalmente tardan *2 días hábiles* en llegar a la puerta de tu casa.\n\n"
                "Si los necesitas para *mañana mismo*, podemos enviarlos para reclamar en el "
                "terminal de transporte de tu ciudad y mañana, después de las 2 p. m., ya los puedes recoger. "
                "Para esta modalidad de envío solo aplica el *pago anticipado* porque las empresas de buses "
                "no tienen convenio de pago contra entrega."
            ),
            parse_mode="Markdown"
        )
        return

    # 🔥 Pregunta 2: ¿tienen pago contra entrega?
    if any(frase in txt for frase in (
        "pago contra entrega", "contraentrega", "contra entrega",
        "pagan al recibir", "puedo pagar al recibir", "tienen contra entrega"
    )):
        await ctx.bot.send_message(
            chat_id=cid,
            text=(
                "📦 Por supuesto que tenemos *pago contra entrega*.\n\n"
                "Para esta modalidad pedimos un *anticipo de $35 000*, que cubre el flete. "
                "Ese valor se descuenta del precio total.\n\n"
                "Ejemplo: si las zapatillas cuestan *$219 900* y haces el anticipo, "
                "cuando lleguen solo le pagas a la transportadora *$184 900*.\n\n"
                "Solicitamos el anticipo porque, en el pasado, algunos pedidos no se recogían "
                "y la transportadora nos devolvía las zapatillas, generando pérdidas de envío."
            ),
            parse_mode="Markdown"
        )
        return

    # 🔥 Pregunta 3: ¿tienen garantía?
    if any(frase in txt for frase in (
        "tienen garantia", "tienen garantía", "hay garantía", "hay garantia",
        "garantía", "garantia", "tienen garantia de fabrica"
    )):
        await ctx.bot.send_message(
            chat_id=cid,
            text=(
                "🛡️ ¡Claro que sí! Todos nuestros productos tienen *garantía de 60 días* "
                "por *pegue* o *defectos de fábrica*.\n\n"
                "Si llegas a tener algún inconveniente, contáctanos y te ayudaremos a solucionarlo."
            ),
            parse_mode="Markdown"
        )
        return


    # 🔥 Pregunta 4: ¿cómo sé que no me van a robar?
    if any(frase in txt for frase in (
        "no me van a robar", "me van a robar", "es seguro",
        "como se que es seguro", "no es estafa", "es confiable",
        "me estafan", "roban por internet", "es real", "desconfío",
        "no me da confianza", "no confío", "dudas"
    )):
        await ctx.bot.send_message(
            chat_id=cid,
            text=(
                "🤝 Entendemos que la confianza es muy importante. "
                "Te compartimos este breve video para que conozcas más sobre nuestra empresa y puedas comprar tranquilo:"
            ),
            parse_mode="Markdown"
        )
        video_url = "https://tudominio.com/videos/video_confianza.mp4"
        await ctx.bot.send_chat_action(chat_id=cid, action=ChatAction.UPLOAD_VIDEO)
        await ctx.bot.send_video(
            chat_id=cid,
            video=video_url,
            caption="¡Estamos aquí para ayudarte en lo que necesites! 👟✨"
        )
        return

    # 🔥 Pregunta 5: ¿dónde están ubicados?
    if any(frase in txt for frase in (
        "donde estan ubicados", "donde queda", "ubicacion", "ubicación",
        "direccion", "dirección", "donde estan", "donde es la tienda",
        "estan ubicados", "ubicados en donde", "en que ciudad estan", "en que parte estan"
    )):
        await ctx.bot.send_message(
            chat_id=cid,
            text=(
                "📍 Estamos en *Bucaramanga, Santander* 💛💚.\n\n"
                "🏡 *Barrio San Miguel, CLL 52 # 16-74* 🔥💋\n\n"
                "🚚 Enviamos a *todo el país* con *Servientrega*.\n\n"
                "Google Maps → https://maps.google.com/?q=7.109500,-73.121597"
            ),
            parse_mode="Markdown"
        )
        return

    # 🔥 Pregunta 6: ¿son nacionales o importados?
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


    # 🔥 Pregunta 7: ¿son originales?
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

    # 🔥 Pregunta 8: ¿de qué calidad son?
    if any(frase in txt for frase in (
        "que calidad son", "de que calidad son", "son buena calidad", "son de buena calidad",
        "son de mala calidad", "que calidad manejan", "que calidad tienen", "calidad de las zapatillas"
    )):
        await ctx.bot.send_message(
            chat_id=cid,
            text="✨ Nuestras zapatillas están elaboradas con *materiales de alta calidad*.\n\n"
                 "Cada par es fabricado cuidadosamente para asegurar que sean de *calidad AAA* 👟🔝, "
                 "garantizando comodidad, durabilidad y un excelente acabado.",
            parse_mode="Markdown"
        )
        return

    # 🔥 Pregunta 9: ¿hay descuento si compro 2 pares?
    if any(frase in txt for frase in (
        "si compro 2 pares", "dos pares descuento", "descuento por 2 pares",
        "descuento por dos pares", "me descuentan si compro dos", "descuento si compro dos",
        "hay descuento por dos", "promocion dos pares", "descuento en 2 pares"
    )):
        await ctx.bot.send_message(
            chat_id=cid,
            text="🎉 ¡Claro que sí! Si compras *2 pares* te damos un *10% de descuento adicional* sobre el total.\n\n"
                 "Aprovecha esta oportunidad para estrenar más y pagar menos 🔥👟👟.",
            parse_mode="Markdown"
        )
        return

    # 🔥 Pregunta 10: ¿manejan precios para mayoristas?
    if any(frase in txt for frase in (
        "precio mayorista", "precios para mayoristas", "mayorista", "quiero vender",
        "puedo venderlos", "descuento para revender", "revender", "comprar para vender",
        "manejan precios para mayoristas", "mayoreo", "venta al por mayor"
    )):
        await ctx.bot.send_message(
            chat_id=cid,
            text="🛍️ ¡Claro que sí! Manejamos *precios para mayoristas* en pedidos de *6 pares en adelante*, "
                 "sin importar talla ni referencia.\n\n"
                 "La única condición es que debes venderlos *mínimo al mismo precio* que nosotros para "
                 "cuidar el mercado y asegurar que tengas un buen margen de utilidad.",
            parse_mode="Markdown"
        )
        return

    # 🔥 Pregunta 11: ¿las tallas son normales o grandes?
    if any(frase in txt for frase in (
        "las tallas son normales", "horma normal", "talla normal",
        "horma grande", "horma pequeña", "tallas grandes", "tallas pequeñas",
        "las tallas son grandes", "las tallas son pequeñas", "como son las tallas"
    )):
        await ctx.bot.send_message(
            chat_id=cid,
            text="👟 Nuestra horma es *normal*. Si calzas talla *40* nacional, te queda bien la *40* de nosotros.\n\n"
                 "Para mayor seguridad, te recomendamos enviarnos una foto de la *etiqueta interna* de tus tenis actuales "
                 "(en la lengüeta o costado interno), donde aparece el equivalente en *centímetros*. "
                 "Así podemos confirmar mejor la talla correcta para ti. 📏✨",
            parse_mode="Markdown"
        )
        return

    # 🔥 Pregunta 12: ¿cuál es la talla más grande que manejan?
    if any(frase in txt for frase in (
        "talla mas grande", "talla más grande", "cual es la talla mas grande",
        "hasta que talla llegan", "mayor talla", "talla maxima", "talla máxima"
    )):
        await ctx.bot.send_message(
            chat_id=cid,
            text="📏 La talla más grande que manejamos es:\n\n"
                 "• *45 Nacional* 🇨🇴\n"
                 "• *47 Europeo* 🇪🇺\n\n"
                 "Así que si usas tallas grandes, también tenemos opciones para ti 👟✨.",
            parse_mode="Markdown"
        )
        return

    # 2) Procesamiento de imagen
    if menciona_imagen(txt):
        if est["fase"] != "esperando_imagen":
            est["fase"] = "esperando_imagen"
            await update.message.reply_text(CLIP_INSTRUCTIONS, reply_markup=ReplyKeyboardRemove())
        return

    # 3) Reinicio explícito primero
    if txt in ("reset", "reiniciar", "empezar", "volver", "/start", "menu", "inicio"):
        reset_estado(cid)
        await saludo_bienvenida(update, ctx)
        return

    # 4) Continuar con otros flujos del bot
    # Lógica existente para referencias, imágenes, precios, etc.
    if await manejar_precio(update, ctx, inventario):
        return

    # Fallback de IA si no se encuentra una coincidencia
    respuesta_fallback = await consultar_ia_fallback(txt_raw)
    if respuesta_fallback:
        await ctx.bot.send_message(
            chat_id=cid,
            text=respuesta_fallback,
            reply_markup=menu_botones(["Hacer pedido", "Ver catálogo", "Enviar imagen"]),
        )
    else:
        await ctx.bot.send_message(
            chat_id=cid,
            text="😅 No logré entender tu solicitud. ¿Quieres volver al menú?",
            reply_markup=menu_botones(["Volver al menú"]),
        )
    return
# 4. Webhook para WhatsApp (usado por Venom)
# ---------- VENOM WEBHOOK ----------
@api.post("/venom")
async def venom_webhook(req: Request):
    try:
        # 1️⃣ Leer JSON
        data = await req.json()
        cid      = wa_chat_id(data.get("from", ""))
        body     = data.get("body", "") or ""
        mtype    = (data.get("type") or "").lower()
        mimetype = (data.get("mimetype") or "").lower()

        logging.info(f"📩 Mensaje recibido — CID: {cid} — Tipo: {mtype}")

        # 2️⃣ Si es imagen en base64
        if mtype == "image" or mimetype.startswith("image"):
            try:
                b64_str = body.split(",", 1)[1] if "," in body else body
                img_bytes = base64.b64decode(b64_str + "===")
                img = Image.open(io.BytesIO(img_bytes))
                img.load()  # ← fuerza carga completa
                logging.info("✅ Imagen decodificada y cargada")
            except Exception as e:
                logging.error(f"❌ No pude leer la imagen: {e}")
                return JSONResponse(
                    {"type": "text", "text": "No pude leer la imagen 😕"},
                    status_code=status.HTTP_400_BAD_REQUEST
                )

            # 3️⃣ Calcular hash
            h_in = str(imagehash.phash(img))
            ref = MODEL_HASHES.get(h_in)
            logging.info(f"🔍 Hash {h_in} → {ref}")

            if ref:
                marca, modelo, color = ref
                estado_usuario.setdefault(cid, reset_estado(cid))
                estado_usuario[cid].update(
                    fase="imagen_detectada", marca=marca, modelo=modelo, color=color
                )
                return JSONResponse({
                    "type": "text",
                    "text": f"La imagen coincide con {marca} {modelo} color {color}. ¿Deseas continuar tu compra? (SI/NO)"
                })
            else:
                reset_estado(cid)
                return JSONResponse({
                    "type": "text",
                })

        # 4️⃣ Si NO es imagen, procesa como texto normal
        reply = await procesar_wa(cid, body)
        return JSONResponse(reply)

    except Exception as e:
        logging.exception("🔥 Error en /venom")
        return JSONResponse(
            {"type": "text", "text": "Error interno en el bot. Intenta de nuevo."},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
# -------------------------------------------------------------------------
# 5. Arranque del servidor
# -------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn, os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(api, host="0.0.0.0", port=port)


