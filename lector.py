@api.post("/venom")
async def venom_webhook(req: Request):
    try:
        # 1️⃣ Leer JSON
        data = await req.json()
        cid = wa_chat_id(data.get("from", ""))
        body = data.get("body", "") or ""
        mtype = (data.get("type") or "").lower()
        mimetype = (data.get("mimetype") or "").lower()

        logging.info(f"📩 Mensaje recibido — CID: {cid} — Tipo: {mtype} — MIME: {mimetype}")

        # 2️⃣ Si es imagen en base64
        if mtype == "image" or mimetype.startswith("image"):
            try:
                b64_str = body.split(",", 1)[1] if "," in body else body
                img_bytes = base64.b64decode(b64_str + "===")
                img = Image.open(io.BytesIO(img_bytes))
                img.load()  # fuerza carga completa
                logging.info(f"✅ Imagen decodificada correctamente. Tamaño: {img.size}")
            except Exception as e:
                logging.error(f"❌ No pude leer la imagen: {e}")
                return JSONResponse({"type": "text", "text": "❌ No pude leer la imagen 😕"})

            # 🧠 Obtener estado del usuario
            est = estado_usuario.get(cid, {})
            fase = est.get("fase", "")
            logging.info(f"🔍 Fase actual del usuario {cid}: {fase or 'NO DEFINIDA'}")

            # 3️⃣ Si está esperando comprobante → OCR
            if fase == "esperando_comprobante":
                try:
                    temp_path = f"temp/{cid}_proof.jpg"
                    os.makedirs("temp", exist_ok=True)
                    with open(temp_path, "wb") as f:
                        f.write(img_bytes)

                    texto = extraer_texto_comprobante(temp_path)
                    logging.info(f"[OCR] Texto extraído:\n{texto[:500]}")

                    if es_comprobante_valido(texto):
                        logging.info("✅ Comprobante válido por OCR")
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
                            temp_path
                        )
                        os.remove(temp_path)
                        reset_estado(cid)
                        return JSONResponse({
                            "type": "text",
                            "text": "✅ Comprobante verificado. Tu pedido está en proceso. 🚚"
                        })
                    else:
                        logging.warning("⚠️ OCR no válido. Texto no contiene 'pago exitoso'")
                        os.remove(temp_path)
                        return JSONResponse({
                            "type": "text",
                            "text": "⚠️ No pude verificar el comprobante. Asegúrate que diga 'Pago exitoso'."
                        })
                except Exception as e:
                    logging.error(f"❌ Error al procesar comprobante: {e}")
                    return JSONResponse({"type": "text", "text": "❌ No pude procesar el comprobante. Intenta con otra imagen."})

            # 4️⃣ Si NO es comprobante → Detectar modelo por hash
            try:
                h_in = str(imagehash.phash(img))
                ref = MODEL_HASHES.get(h_in)
                logging.info(f"🔍 Hash calculado: {h_in} → {ref}")

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
                    logging.warning("❌ No se detectó ninguna coincidencia de modelo para esta imagen.")
                    reset_estado(cid)
                    return JSONResponse({
                        "type": "text",
                        "text": "❌ No reconocí el modelo. Puedes intentar con otra imagen clara."
                    })

            except Exception as e:
                logging.error(f"❌ Error al identificar modelo: {e}")
                return JSONResponse({"type": "text", "text": "❌ Ocurrió un error al intentar detectar el modelo."})

        # 5️⃣ Si es texto u otro tipo
        elif mtype == "chat":
            fase_actual = estado_usuario.get(cid, {}).get("fase", "")
            logging.info(f"💬 Texto recibido en fase: {fase_actual or 'NO DEFINIDA'}")
            reply = await procesar_wa(cid, body)
            return JSONResponse(reply)

        # 6️⃣ Tipo no manejado
        else:
            logging.warning(f"🤷‍♂️ Tipo de mensaje no manejado: {mtype}")
            return JSONResponse({
                "type": "text",
                "text": f"⚠️ Tipo de mensaje no manejado: {mtype}"
            })

    except Exception as e:
        logging.exception("🔥 Error general en venom_webhook")
        return JSONResponse(
            {"type": "text", "text": "⚠️ Error interno procesando el mensaje."},
            status_code=200
        )
