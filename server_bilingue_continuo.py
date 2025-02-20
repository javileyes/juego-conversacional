#port forwarding##ConversationalS	192.168.1.138	TCP	5500	Enable		5500	all#########################################
HISTORICO_LOG = "historico.log"

def printf(filename, text, add_newline=True):
    """
    Añade una cadena de texto a un archivo especificado. Si el archivo no existe, se crea.
    Opcionalmente, añade un salto de línea al final del texto dependiendo del parámetro add_newline.
    
    Parámetros:
    filename (str): El nombre del archivo al cual se desea añadir el texto.
    text (str): El texto que se desea añadir al archivo.
    add_newline (bool): Indica si se debe añadir un salto de línea al final del texto (por defecto es True).
    """
    try:
        with open(filename, 'a') as file:
            if add_newline:
                file.write(text + "\n")
            else:
                file.write(text)
    except IOError as e:
        print(f"Ocurrió un error al abrir o escribir en el archivo: {e}")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")

#########################################
####    MarianMT
#########################################
from transformers import MarianMTModel, MarianTokenizer

# Modelo para traducir de español a inglés
model_name_es_en = 'Helsinki-NLP/opus-mt-es-en'
modelo_traductor = MarianMTModel.from_pretrained(model_name_es_en)
tokenizer_es_en = MarianTokenizer.from_pretrained(model_name_es_en)

def translate_text_to_english(text):
    print("Traduciendo texto a inglés:", text)
    tokens = tokenizer_es_en(text, return_tensors='pt', padding=True)
    translated = modelo_traductor.generate(**tokens)
    decoded = []
    for t in translated:
        decoded.append(tokenizer_es_en.decode(t, skip_special_tokens=True))
    return decoded[0]

# Modelo para traducir de inglés a español
model_name_en_es = 'Helsinki-NLP/opus-mt-tc-big-en-es'
model_big_en_es = MarianMTModel.from_pretrained(model_name_en_es)
tokenizer_en_es = MarianTokenizer.from_pretrained(model_name_en_es)

def translate_text_to_spanish(text):
    print("Traduciendo texto a español:", text)
    tokens = tokenizer_en_es(text, return_tensors='pt', padding=True)
    translated = model_big_en_es.generate(**tokens)
    decoded = []
    for t in translated:
        decoded.append(tokenizer_en_es.decode(t, skip_special_tokens=True))
    return decoded[0]
        

import whisper
modelWhisper = whisper.load_model('turbo')

LOGGING = False
from threading import Lock
import re

def encontrar_coincidencia(texto, cadena_busqueda="</s>"):
    indice = texto.find(cadena_busqueda)
    if indice != -1:
        return texto[:indice + len(cadena_busqueda)]
    else:
        return ""

def ajustar_contexto(texto, max_longitud=15000, secuencia="[SYSTEM_PROMPT]", system_end="[/SYSTEM_PROMPT]"):
    system_prompt = encontrar_coincidencia(texto, system_end)
    if len(texto) > max_longitud:
        indice_secuencia = 0
        while True:
            indice_secuencia = texto.find(secuencia, indice_secuencia + 1)
            if indice_secuencia == -1 or len(system_prompt) + len(texto) - indice_secuencia <= max_longitud:
                break
        if indice_secuencia != -1:
            return system_prompt + texto[indice_secuencia:]
        else:
            return system_prompt + texto[-max_longitud + len(system_prompt):]
    else:
        return system_prompt + texto

generate_lock = Lock()


import threading

class EstadoGeneracion:
    def __init__(self):
        self.parts = [""] * 100
        self.top = -1
        self.generando = False
        self.primer_audio = ""
        self.historico = ""

estado_generacion = {}
estado_generacion['anonimo'] = EstadoGeneracion()

import re


#########################################
#### NUEVA FUNCIÓN: Segmentación del texto transcrito
#########################################
def segment_transcribed_text(userID, text, min_length=34):
    """
    Segmenta el texto recibido (ya traducido por Whisper) en partes según la puntuación,
    de forma similar a como se segmentaba el texto generado por el LLM.
    Además, genera el primer audio usando el sintetizador de voz, solo si es la primera parte.
    Ahora, en lugar de reiniciar el índice del buffer a 0, se añaden al final de la cola.
    """
    if userID not in estado_generacion:
         estado_generacion[userID] = EstadoGeneracion()
    # No reiniciamos el buffer; se agrega a continuación de lo ya existente
    indiceParte = estado_generacion[userID].top + 1  # Comenzamos desde el siguiente índice disponible
    parte_actual = ""
    for char in text:
        parte_actual += char
        if char in ",;:.?!" and len(parte_actual) > min_length:
            if estado_generacion[userID].top == -1:
                # Genera el primer audio solo si no se ha generado aún
                estado_generacion[userID].primer_audio = voz_sintetica_english(parte_actual)
            estado_generacion[userID].parts[indiceParte] = parte_actual
            estado_generacion[userID].top = indiceParte
            indiceParte += 1
            parte_actual = ""
    if len(parte_actual) > 1:
        if estado_generacion[userID].top == -1:
            estado_generacion[userID].primer_audio = voz_sintetica_english(parte_actual)
        estado_generacion[userID].parts[indiceParte] = parte_actual
        estado_generacion[userID].top = indiceParte

#########################################
####    SERVER
#########################################
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import threading
import os
import torch
from pydub import AudioSegment
import pandas as pd
import random
import time

# if LOGGING:
#     print("El modelo es:", model)

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

if "-es" in sys.argv:
    idioma = "es"

args = [arg for arg in sys.argv[1:] if arg not in ["-s", "--short", "-es"]]

if len(args) > 0:
    system_prompt = args[0]
if len(args) > 1:
    saludo = args[1]

historico = f"[SYSTEM_PROMPT]{system_prompt}[/SYSTEM_PROMPT][INST]Espero un saludo[/INST]{saludo}</s>"

if LOGGING:
    print(f"{ai}:", saludo)

transcribe_lock = Lock()

app = Flask(__name__)
CORS(app)
output = ""

def eliminar_archivos_temp(nombre_inicio='temp_synth_audio'):
    archivos = os.listdir('.')
    archivos_temp = [archivo for archivo in archivos if archivo.startswith(nombre_inicio)]
    for archivo in archivos_temp:
        try:
            os.remove(archivo)
            print(f"Archivo eliminado: {archivo}")
        except Exception as e:
            print(f"No se pudo eliminar el archivo {archivo}. Razón: {e}")

@app.route('/inicio', methods=['POST'])
def print_strings():
    eliminar_archivos_temp("received_audio")
    eliminar_archivos_temp()
    data = request.json
    system_prompt = data.get('system_prompt')
    saludo = data.get('saludo')
    userID = data.get('userID')
    printf(HISTORICO_LOG, f"System prompt: {system_prompt}")
    printf(HISTORICO_LOG, f"userID: {userID}")
    if not userID:
        return jsonify(error="No se proporcionó userID"), 400
    userID = int(userID)
    def elegir_personaje_aleatorio():
        df = pd.read_csv('Personajes_ficcion.csv')
        return random.choice(df.iloc[:, 0].tolist())
    if "#personaje" in system_prompt:
        personaje_aleatorio = elegir_personaje_aleatorio()
        system_prompt = system_prompt.replace("#personaje", personaje_aleatorio)
    if "#personaje" in saludo:
        saludo = saludo.replace("#personaje", personaje_aleatorio)
    if userID not in estado_generacion:
        estado_generacion[userID] = EstadoGeneracion()
    estado_generacion[userID].historico = f"[SYSTEM_PROMPT]{system_prompt}[/SYSTEM_PROMPT][INST]Espero un saludo[/INST]{saludo}"
    # pre_warm_chat(estado_generacion[userID].historico + "</s>")
    return jsonify({"message": saludo, "historico": estado_generacion[userID].historico, "userID": userID}), 200

@app.route('/get-translations-file', methods=['GET'])
def get_translations():
    return send_from_directory(directory='.', path='translations.csv', as_attachment=True)

import csv
import shutil
@app.route('/save-translations-file', methods=['POST'])
def save_translations():
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    try:
        shutil.copy('translations.csv', 'translations.csv.bak')
        with open('translations.csv', mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, delimiter='#')
            for row in data:
                writer.writerow(row)
        return jsonify({'message': 'File successfully saved'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/all_conversation', methods=['GET'])
def all_conversation():
    filepath = 'conversacion.mp3'
    if not os.path.exists(filepath):
        return jsonify(error="Archivo de conversación no encontrado"), 404
    with open(filepath, 'rb') as audio_file:
        audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')
    return jsonify(audio_base64=audio_base64)

import subprocess

def convert_ogg_to_mp3(source_ogg_path, target_mp3_path):
    command = ['ffmpeg', '-y' ,'-i', source_ogg_path, target_mp3_path]
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode != 0:
        print(f"Error al convertir {source_ogg_path} a {target_mp3_path}")
        print("Salida de error de ffmpeg:")
        print(process.stderr.decode())

def convert_wav_to_mp3(source_wav_path, target_mp3_path):
    command = ['ffmpeg', '-i', source_wav_path, target_mp3_path]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

lockAddAudio = threading.Lock()



@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """
    Este endpoint ahora:
      1. Recibe el archivo de audio.
      2. Utiliza Whisper para transcribir automáticamente (detectando el idioma).
      3. Si se detecta español, traduce la transcripción a inglés.
         Si se detecta inglés, traduce la transcripción a español.
      4. Segmenta el texto traducido para la síntesis de voz.
    """
    global estado_generacion, idioma

    if LOGGING:
        print("Transcribiendo audio y aplicando traducción según idioma detectado...")
    global user, ai

    if 'userID' not in request.form:
        return jsonify(error="No se proporcionó userID"), 400
    userID = int(request.form['userID'])

    if userID not in estado_generacion:
        estado_generacion[userID] = EstadoGeneracion()

    if 'file' not in request.files:
        return jsonify(error="No se proporcionó file"), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify(error="No selected file"), 400

    timestamp = int(time.time() * 1000)
    ogg_filepath = f"received_audio_{timestamp}.ogg"
    file.save(ogg_filepath)

    start_transcribe_time = time.time()
    with transcribe_lock:
        # Se elimina el parámetro de idioma para que Whisper detecte automáticamente
        result = modelWhisper.transcribe(ogg_filepath, fp16=False)
        transcripcion = result["text"]
        detected_lang = result.get("language", None)
    end_transcribe_time = time.time()

    if LOGGING:
        print("Transcripción original:", transcripcion)
        print("Idioma detectado:", detected_lang)

    # Si el idioma detectado es español, se traduce a inglés.
    # Si es inglés, se traduce a español.
    traduccion = ""
    if detected_lang == "es":
        idioma = "en"
        traduccion = translate_text_to_english(transcripcion)
    elif detected_lang == "en":
        idioma = "es"
        traduccion = translate_text_to_spanish(transcripcion)

    if LOGGING:
        print("Transcripción final (traducida):", traduccion)

    # Se segmenta el texto traducido para la síntesis de voz.
    segment_transcribed_text(userID, traduccion)

    # Se añade el audio a la conversación de forma asíncrona
    # add_audio_to_conversation_async(ogg_filepath)

    return jsonify(entrada=transcripcion, prompt="", entrada_traducida=traduccion)

@app.route('/only_transcribe', methods=['POST'])
def only_transcribe_audio():
    if LOGGING:
        print("Transcribiendo audio...")
    if 'file' not in request.files:
        print("No file part")
        return jsonify(error="No file part"), 400
    file = request.files['file']
    if file.filename == '':
        print("No selected file")
        return jsonify(error="No selected file"), 400
    if LOGGING:
        print("creando fichero ogg audio antes de transcripción...")
    timestamp = int(time.time() * 1000)
    ogg_filepath = f"received_audio_{timestamp}.ogg"
    file.save(ogg_filepath)
    start_transcribe_time = time.time()
    with transcribe_lock:
        if LOGGING:
            print("después del transcribe lock")
        transcripcion = modelWhisper.transcribe(ogg_filepath, fp16=False, language=idioma)["text"]
    end_transcribe_time = time.time()
    if LOGGING:
        print(f"Transcripción completada en {end_transcribe_time - start_transcribe_time} segundos")
        print("transcripción:", transcripcion)
    # add_audio_to_conversation_async(ogg_filepath)
    return jsonify(entrada=transcripcion, entrada_traducida="")

@app.route('/get_next_part', methods=['GET'])
def get_next_part():
    global estado_generacion
    if LOGGING:
        print("LAS CLAVES de usuario y sus TOP:")
        for clave in estado_generacion.keys():
            print(clave, "TOP:", estado_generacion[clave].top)
    userID = request.args.get('userID', default=0, type=int)
    index = request.args.get('index', default=0, type=int)
    if LOGGING:
        print(f"userID:{userID} partes: {estado_generacion[userID].parts}, generando: {estado_generacion[userID].generando}, index: {index}, estado_generacion[userID].top: {estado_generacion[userID].top}")
    while True:
        if index is not None and index >= 0 and index <= estado_generacion[userID].top:
            part = estado_generacion[userID].parts[index]
            estado_generacion[userID].parts[index] = ""
            if LOGGING:
                print("con index:", index, "estado_generacion[userID].top:", estado_generacion[userID].top)
                print(f"Enviando parte: {part}")
            return jsonify(output=part)
        elif estado_generacion[userID].generando:
            if LOGGING:
                print("Esperando a que se generen más partes...")
            time.sleep(0.1)
        else:
            if LOGGING:
                print("No hay más partes para enviar", "index:", index, "estado_generacion[userID].top:", estado_generacion[userID].top)
            return jsonify(output="")

@app.route('/texto', methods=['POST'])
def process_text():
    global user, ai, estado_generacion, idioma
    texto = request.json.get('texto') if request.is_json else request.form.get('texto')
    userID = int(request.json.get('userID') if request.is_json else request.form.get('userID'))
    if userID not in estado_generacion:
        estado_generacion[userID] = EstadoGeneracion()
    prompt = f"[INST]{texto}[/INST]"
    printf(HISTORICO_LOG, f"EN SERVICIO TEXTO, USER:{estado_generacion[userID].historico}")
    if LOGGING:
        print("Texto recibido:", texto)
        print("Prompt generado:", prompt)
    estado_generacion[userID].historico += prompt
    printf(HISTORICO_LOG, f"SUPERHISTORICO:{estado_generacion[userID].historico}")
    if LOGGING:
        print("Histórico después de actualizar:", estado_generacion[userID].historico)
    #ver si el texto está en inglés o en español con librería langdetect
    import langdetect
    try:
        lang = langdetect.detect(texto)
        if lang == "es":
            idioma = "en"
        elif lang == "en":
            idioma = "es"
    except:
        idioma = "en"
    if LOGGING:
        print("Idioma detectado:", idioma)
    traduccion = ""
    if idioma == "en":
        print("idioma español")
        traduccion= translate_text_to_english(texto)
    elif idioma == "es":
        print("idioma ingles")
        traduccion= translate_text_to_spanish(texto)
    
    segment_transcribed_text(userID, traduccion)
    return jsonify(entrada=texto, prompt=prompt, entrada_traducida=traduccion)

if idioma == "en":
    print("idioma ingles")
elif idioma == "es":
    print("idioma español")

import base64
@app.route('/audio', methods=['POST'])
def generate_audio():
    texto = request.json.get('texto')
    if LOGGING:
        print('TEXTO!!!!!!!!:', texto)
    if not texto:
        return jsonify(error="No se proporcionó texto"), 400
    audio_base64 = voz_sintetica_english(texto)
    return jsonify(audio_base64=audio_base64)

import base64
@app.route('/primer_audio', methods=['GET'])
def primer_audio():
    userID = request.args.get('userID', default=0, type=int)
    while estado_generacion[userID].primer_audio == "wait":
        time.sleep(0.1)
        if LOGGING:
            print("esperando primer audio")
    audio_base64 = estado_generacion[userID].primer_audio
    return jsonify(audio_base64=audio_base64)

import soundfile as sf
import base64
import numpy as np
import tempfile

def add_comma_after_punctuation(text: str) -> str:
    punctuation_marks = ['-','\n', '*','\\']
    for mark in punctuation_marks:
        text = text.replace(mark, ',')
    return text

from kokoro import KPipeline
pipeline_en = KPipeline(lang_code='a')
pipeline_es = KPipeline(lang_code='e')

def voz_sintetica_english(texto):
    global pipeline_en, pipeline_es, idioma   
    texto = add_comma_after_punctuation(texto)
    voice = 'af_heart'
    
    if idioma == "en":
        pipeline = pipeline_en
    elif idioma == "es":
        pipeline = pipeline_es
    else:
        pipeline = pipeline_en
    
    generator = pipeline(texto, voice=voice, speed=1, split_pattern=r'\n+')
    audios = []
    for idx, (gs, ps, audio) in enumerate(generator):
        audios.append(audio)
    if audios:
        combined_audio = np.concatenate(audios, axis=0)
    else:
        return ""
    sample_rate = 24000
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_file_path = temp_file.name
    temp_file.close()
    sf.write(temp_file_path, combined_audio, sample_rate)
    # add_audio_to_conversation_async(temp_file_path, convert_to_mp3=True)
    with open(temp_file_path, 'rb') as f:
        audio_base64 = base64.b64encode(f.read()).decode('utf-8')
    return audio_base64

def print_routes(app):
    print("Endpoints disponibles:")
    for rule in app.url_map.iter_rules():
        methods = ','.join(sorted(rule.methods))
        print(f"{rule.endpoint}: {rule.rule} [{methods}]")

if __name__ == '__main__':
    print_routes(app)
    app.run(host='0.0.0.0', port=5500, threaded=True)