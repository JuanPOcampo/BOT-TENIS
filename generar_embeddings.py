import json
import torch
import numpy as np
import logging

def cargar_embeddings(path="embeddings.json"):
    try:
        with open(path, "r") as f:
            data = json.load(f)
        logging.info(f"[CLIP] Embeddings cargados: {len(data)} modelos")
        return data
    except Exception as e:
        logging.error(f"[CLIP] ❌ Error cargando embeddings: {e}")
        return {}

def comparar_clip(embedding_usuario, embeddings_data):
    try:
        emb_usuario = torch.tensor(embedding_usuario, dtype=torch.float32)
        emb_usuario = emb_usuario / emb_usuario.norm()

        mejor_modelo = None
        mejor_similitud = -1

        for modelo, vectores in embeddings_data.items():
            if not vectores:
                continue

            for idx, emb_ref in enumerate(vectores):
                try:
                    emb_tensor = torch.tensor(emb_ref, dtype=torch.float32)
                    emb_tensor = emb_tensor / emb_tensor.norm()

                    similitud = torch.dot(emb_usuario, emb_tensor).item()
                    logging.debug(f"[CLIP] {modelo} [{idx}] → Sim: {similitud:.4f}")

                    if similitud > mejor_similitud:
                        mejor_similitud = similitud
                        mejor_modelo = modelo
                except Exception as err:
                    logging.warning(f"[CLIP] ⚠️ Error comparando con {modelo}[{idx}]: {err}")

        if mejor_modelo:
            logging.info(f"[CLIP] ✅ Mejor match: {mejor_modelo} ({mejor_similitud:.4f})")
            return mejor_modelo, mejor_similitud
        else:
            logging.warning("[CLIP] 😕 No se encontró coincidencia")
            return None, 0.0

    except Exception as e:
        logging.error(f"[CLIP] ❌ Error en comparación: {e}")
        return None, 0.0
