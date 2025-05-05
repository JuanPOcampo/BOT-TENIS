# BotVentasIA

Este proyecto contiene dos partes:

1. `bot_backend/`: Bot con IA que atiende clientes por WhatsApp, procesa imágenes, audios y registra pedidos.
2. `embeddings_generator/`: Script que recorre las carpetas de modelos en Google Drive, genera embeddings con CLIP y los guarda en `embeddings.json`.

## 🔄 Flujo de uso

1. Subes nuevas imágenes a Google Drive (una carpeta por modelo).
2. Ejecutas `generar_embeddings_clip.py` para actualizar el archivo `embeddings.json`.
3. Subes ese archivo al backend del bot para que reconozca los nuevos modelos.

