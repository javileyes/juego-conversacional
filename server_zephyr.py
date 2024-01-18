from flask import Flask, request, jsonify
from transformers import MarianMTModel, MarianTokenizer
from threading import Lock
import torch

import time
# Asegúrate de importar tu modelo Whisper correctamente
# from tu_paquete import modelWhisper

import whisper
modelWhisper = whisper.load_model('medium')


model_name = 'Helsinki-NLP/opus-mt-es-en'  # Modelo para traducir de español a inglés
modelo_traductor = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)


def translate_text_to_english(text):
    print("Traduciendo texto:", text)
    tokens = tokenizer(text, return_tensors='pt', padding=True)
    translated = modelo_traductor.generate(**tokens)
    decoded = []
    for t in translated:
        decoded.append(tokenizer.decode(t, skip_special_tokens=True))
    
    return decoded[0]


modelo = "zypher"

if modelo == "mistral":
    from modelo_mistral_base import generate_long_chat, load_model
elif modelo == "zypher":
    from modelo_Zypher_beta import generate_long_chat, load_model
else:
    print("modelo no encontrado")
    exit()

ai = "assistant"
user = "user"

contexto = """

"""

system_prompt = """
You are a kind and helpful assistan bot. You are here to help the user to find the best answer to his question.
"""

saludo = "Hello, I am ready to receive and process your input."

idioma = "en"

import sys

# Verifica si el comando tenía flag -s o --short
if "-s" in sys.argv or "--short" in sys.argv:
    short_answer = True

# Si encuentra el flag -es cambia el idioma a español
if "-es" in sys.argv:
    idioma = "es"

# Filtra los argumentos para eliminar los flags
args = [arg for arg in sys.argv[1:] if arg not in ["-s", "--short", "-es"]]

# Asigna los valores a system_prompt y saludo basándose en los argumentos restantes
if len(args) > 0:
    system_prompt = args[0]
if len(args) > 1:
    saludo = args[1]

if modelo == "mistral":
    historico = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>assistant\n{saludo}<|im_end|>\n"
elif modelo == "zypher":    
    historico = f"<|system|>{system_prompt}</s>\n<|assistant|>\n{saludo}</s>\n"


# load model
load_model(user=user, ai=ai)

print(f"{ai}:", saludo)

# Crea un bloqueo para proteger el código contra la concurrencia a la hora de transcribir
transcribe_lock = Lock()

# Crea un bloqueo para proteger el código contra la concurrencia a la hora de traducir
translate_lock = Lock()

# Crea un bloqueo para proteger el código contra la concurrencia a la hora de generar texto
generate_lock = Lock()

app = Flask(__name__)


@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    global historico
    global user
    global ai
    # global iteracion

    traduccion = ""

    # Comprueba si el archivo fue enviado
    if 'file' not in request.files:
        return jsonify(error="No file part"), 400

    file = request.files['file']

    # Comprueba si el usuario no seleccionó un archivo
    if file.filename == '':
        return jsonify(error="No selected file"), 400

    # Genera un nombre de archivo único utilizando una marca de tiempo
    timestamp = int(time.time() * 1000)  # Marca de tiempo en milisegundos
    mp3_filepath = f"received_audio_{timestamp}.mp3"
    file.save(mp3_filepath)

    # Transcribe el archivo MP3 (Asegúrate de tener el modelo cargado correctamente)
    # Transcribe el archivo MP3 dentro de una sección crítica protegida por un bloqueo
    with transcribe_lock:
        # transcripcion = modelWhisper.transcribe(mp3_filepath, fp16=False)
        # transcipción lenguaje inglés
        transcripcion = modelWhisper.transcribe(mp3_filepath, fp16=False, language=idioma)
        transcripcion = transcripcion["text"]
        print("transcripción:", transcripcion)

    # si el idioma es español, traduce la transcripción al inglés
    if idioma == "es":
        with translate_lock:
            traduccion = translate_text_to_english(transcripcion)
        print("traducción:", traduccion)
        entrada = traduccion
    else:
        entrada = transcripcion
    # prompt = f"{historico}\n{user}:{entrada}\n{ai}:"
    # print("prompt:", prompt)


    with generate_lock:
        historico, output = generate_long_chat(historico, ai, user, input_text=entrada, max_additional_tokens=2048, short_answer=short_answer, streaming=False, printing=False)
        print("output:", output)
        # print("historico:", historico)


    return jsonify(entrada=transcripcion, entrada_traducida=traduccion, respuesta=output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5500, threaded=True)    
