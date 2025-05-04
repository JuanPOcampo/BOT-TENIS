import os
import io
import json
import logging
from PIL import Image
import numpy as np

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account

from transformers import CLIPProcessor, CLIPModel

# 📌 Carga las credenciales desde la variable de entorno
creds_info = json.loads(os.environ["GOOGLE_CREDS_JSON"])
creds = service_account.Credentials.from_service_account_info(
    creds_info,
    scopes=["https://www.googleapis.com/auth/drive.readonly"]
)

# 📁 ID de la carpeta principal en Google Drive
CARPETA_MODELOS_ID = os.environ["DRIVE_FOLDER_ID"]

# 🔗 Inicializa el cliente de Google Drive
drive_service = build("drive", "v3", credentials=creds)

# 🧠 Modelo CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 🧬 Generar embedding desde imagen PIL
def generar_embedding(image: Image.Image):
    inputs = clip_processor(images=image, return_tensors="pt")
    outputs = clip_model.get_image_features(**inputs)
    return outputs[0].detach().numpy()[0].tolist()

# 🖼️ Descargar imagen de Drive
def descargar_imagen(file_id):
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    return Image.open(fh).convert("RGB")

# 🔁 Recorre Drive y genera embeddings
def generar_embeddings():
    print("🚀 Recorriendo carpetas de modelos en Google Drive...")
    embeddings = {}

    carpetas = drive_service.files().list(
        q=f"'{CARPETA_MODELOS_ID}' in parents and mimeType='application/vnd.google-apps.folder'",
        fields="files(id, name)"
    ).execute().get("files", [])

    for carpeta in carpetas:
        modelo = carpeta["name"]
        carpeta_id = carpeta["id"]
        print(f"📁 Modelo: {modelo}")

        imagenes = drive_service.files().list(
            q=f"'{carpeta_id}' in parents and mimeType contains 'image/'",
            fields="files(id, name)"
        ).execute().get("files", [])

        embeddings[modelo] = []

        for img in imagenes:
            try:
                print(f"   ⬇️ Procesando imagen: {img['name']}")
                image_pil = descargar_imagen(img["id"])
                embedding = generar_embedding(image_pil)
                embeddings[modelo].append(embedding)
            except Exception as e:
                print(f"   ⚠️ Error con imagen {img['name']}: {e}")

    print("✅ Embeddings generados. Guardando archivo JSON...")

    # 🔐 No crees la carpeta, ya existe: solo guarda
    with open("/var/data/embeddings.json", "w") as f:
        json.dump(embeddings, f)

    print("🎉 Archivo embeddings.json creado con éxito.")

# ▶️ Ejecutar script
if __name__ == "__main__":
    generar_embeddings()
