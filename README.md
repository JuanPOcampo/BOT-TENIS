import json
import re
import logging
import requests
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

# CONFIGURACIÓN
TOKEN = '8171144664:AAEWKcLeXr_XT1qz1uySc-tHmr3G1b6pq-0'
NOMBRE_NEGOCIO = 'Agenda de Citas Juanpa'
# URL del Google Sheets para guardar las citas (asegúrate de usar el enlace que termina en /exec)
SHEETS_URL = 'https://script.google.com/macros/s/AKfycbxQ65lqrjxnVOSpC6tgM8IAPTBubqNPcsOBA7hv9GwcY5_lfLe9JpgM1fjRuSS82GO5jA/exec'

logging.basicConfig(level=logging.INFO)

# Diccionario para almacenar el estado de cada usuario (por chat_id)
estado_usuario = {}

def reset_estado(chat_id):
    estado_usuario[chat_id] = {
        "fase": "inicio",
        "servicio": None,
        "fecha": None,
        "hora": None,
        "nombre": None,
        "correo": None,
        "telefono": None,
        "observaciones": ""
    }

def menu_botones(lista_opciones):
    botones = [[KeyboardButton(op)] for op in lista_opciones]
    return ReplyKeyboardMarkup(botones, resize_keyboard=True)

def es_correo_valido(correo):
    return re.match(r"[^@]+@[^@]+\.[^@]+", correo)

def es_telefono_valido(numero):
    return re.match(r"^\+?\d{7,15}$", numero)

def guardar_cita(estado):
    """
    Envía los datos de la cita al Google Sheets usando una petición GET.
    """
    params = {
        "nombre": estado["nombre"],
        "correo": estado["correo"],
        "telefono": estado["telefono"],
        "servicio": estado["servicio"],
        "fecha": estado["fecha"],
        "hora": estado["hora"],
        "observaciones": estado["observaciones"]
    }
    try:
        response = requests.get(SHEETS_URL, params=params)
        logging.info("Cita guardada: " + response.text)
    except Exception as e:
        logging.error("Error al guardar la cita: " + str(e))

async def saludo_bienvenida(update: Update, context: ContextTypes.DEFAULT_TYPE):
    mensaje = (
        f"¡Bienvenido a {NOMBRE_NEGOCIO}!\n"
        "¿Qué te gustaría hacer hoy?"
    )
    await update.message.reply_text(mensaje, reply_markup=menu_botones(["Agendar cita", "Ver servicios"]))

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    reset_estado(chat_id)
    await saludo_bienvenida(update, context)

async def responder(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id

    if chat_id not in estado_usuario:
        reset_estado(chat_id)
    estado = estado_usuario[chat_id]

    if update.message.photo:
        return

    if not update.message.text:
        return

    mensaje = update.message.text.strip()

    # Saludo inicial
    saludos = ["hola", "buenas", "hi", "hello", "saludos"]
    if any(s in mensaje.lower() for s in saludos):
        return await start(update, context)

    # Opción para ver servicios
    if mensaje == "Ver servicios":
        servicios = "Nuestros servicios incluyen: Consulta, Revisión, Asesoría."
        return await update.message.reply_text(servicios)

    # Iniciar flujo para agendar cita
    if mensaje == "Agendar cita":
        estado["fase"] = "esperando_servicio"
        opciones_servicio = ["Consulta", "Revisión", "Asesoría"]
        return await update.message.reply_text("¿Qué servicio deseas?", reply_markup=menu_botones(opciones_servicio))

    # Flujo de agendamiento
    if estado["fase"] == "esperando_servicio":
        estado["servicio"] = mensaje
        estado["fase"] = "esperando_fecha"
        return await update.message.reply_text("¿Para qué día deseas la cita? (Ej: 2025-04-15)")

    if estado["fase"] == "esperando_fecha":
        estado["fecha"] = mensaje
        estado["fase"] = "esperando_hora"
        return await update.message.reply_text("¿A qué hora te gustaría? (Ej: 15:30)")

    if estado["fase"] == "esperando_hora":
        estado["hora"] = mensaje
        estado["fase"] = "esperando_nombre"
        return await update.message.reply_text("¿Cuál es tu nombre completo?")

    if estado["fase"] == "esperando_nombre":
        estado["nombre"] = mensaje
        estado["fase"] = "esperando_correo"
        return await update.message.reply_text("¿Cuál es tu correo electrónico?")

    if estado["fase"] == "esperando_correo":
        if not es_correo_valido(mensaje):
            return await update.message.reply_text("El correo no es válido. Por favor, ingresa un correo correcto.")
        estado["correo"] = mensaje
        estado["fase"] = "esperando_telefono"
        return await update.message.reply_text("¿Cuál es tu número de teléfono? (Incluye el código de país)")

    if estado["fase"] == "esperando_telefono":
        if not es_telefono_valido(mensaje):
            return await update.message.reply_text("El número no es válido. Por favor, ingresa un número correcto.")
        estado["telefono"] = mensaje
        estado["fase"] = "esperando_observaciones"
        return await update.message.reply_text("¿Alguna observación adicional? Si no, puedes escribir 'No'.")

    if estado["fase"] == "esperando_observaciones":
        estado["observaciones"] = mensaje if mensaje.lower() != "no" else ""
        resumen = (
            f"✅ Cita agendada:\n"
            f"Servicio: {estado['servicio']}\n"
            f"Fecha: {estado['fecha']}\n"
            f"Hora: {estado['hora']}\n"
            f"Nombre: {estado['nombre']}\n"
            f"Correo: {estado['correo']}\n"
            f"Teléfono: {estado['telefono']}\n"
            f"Observaciones: {estado['observaciones'] if estado['observaciones'] else 'Ninguna'}\n"
        )
        await update.message.reply_text(resumen)
        await update.message.reply_text("¡Tu cita ha sido agendada! Recibirás una confirmación próximamente.")
        guardar_cita(estado)
        reset_estado(chat_id)
        return

    return await saludo_bienvenida(update, context)

def main():
    application = ApplicationBuilder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT, responder))
    application.run_polling()

if __name__ == "__main__":
    main()
