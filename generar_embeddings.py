import os
import io
import json
import logging
from PIL import Image
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

import torch
from transformers import CLIPProcessor, CLIPModel
from torch.nn.functional import normalize

# Configuración
logging.basicConfig(level=logging.INFO)
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
FOLDER_ID = '1OXHjSG82RO9KGkNIZIRVusFpFhZlujQE'
CACHE_DIR = "/var/data/drive"
EMBEDDINGS_PATH = "/var/data/embeddings.json"

# Credenciales
creds_info = json.loads(os.environ["GOOGLE_CREDS_JSON"])
creds = service_account.Credentials.from_service_account_info(creds_info, scopes=SCOPES)

# CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def generar_embedding(image: Image.Image):
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
        emb = normalize(emb, dim=-1)
    return emb[0].numpy().tolist()

def cargar_servicio_drive():
    return build('drive', 'v3', credentials=creds)

def listar_carpetas(service, folder_id):
    resultados = service.files().list(
        q=f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed = false",
        fields="files(id, name)").execute()
    return resultados.get('files', [])

def listar_imagenes(service, folder_id):
    resultados = service.files().list(
        q=f"'{folder_id}' in parents and mimeType contains 'image/' and trashed = false",
        fields="files(id, name)").execute()
    return resultados.get('files', [])

def descargar_imagen(service, file_id, destino):
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    with open(destino, "wb") as f:
        f.write(fh.read())

def main():
    if os.path.exists(EMBEDDINGS_PATH):
        logging.info("✅ embeddings.json ya existe. No se regenera.")
        return

    service = cargar_servicio_drive()
    carpetas = listar_carpetas(service, FOLDER_ID)
    logging.info(f"📦 Se encontraron {len(carpetas)} carpetas de modelos.")

    os.makedirs(CACHE_DIR, exist_ok=True)
    embeddings = {}

    for carpeta in carpetas:
        modelo = carpeta["name"]
        carpeta_id = carpeta["id"]
        logging.info(f"📁 Modelo: {modelo}")

        modelo_dir = os.path.join(CACHE_DIR, modelo)
        os.makedirs(modelo_dir, exist_ok=True)

        imagenes = listar_imagenes(service, carpeta_id)
        modelo_embeddings = []

        for img in imagenes:
            path_local = os.path.join(modelo_dir, img["name"])
            try:
                if not os.path.exists(path_local):
                    logging.info(f"⬇️ Descargando imagen: {img['name']}")
                    descargar_imagen(service, img["id"], path_local)
                else:
                    logging.info(f"✅ Ya existe: {img['name']}")

                imagen = Image.open(path_local).convert("RGB")
                emb = generar_embedding(imagen)
                modelo_embeddings.append(emb)

            except Exception as e:
                logging.warning(f"⚠️ Error con {img['name']}: {e}")

        if modelo_embeddings:
            embeddings[modelo] = modelo_embeddings

    os.makedirs("/var/data", exist_ok=True)
    with open(EMBEDDINGS_PATH, "w") as f:
        json.dump(embeddings, f)

    logging.info(f"🎉 Archivo embeddings.json creado con éxito en {EMBEDDINGS_PATH}")

# Este archivo solo se ejecuta a mano, desde shell
# No se ejecuta automáticamente desde Render
