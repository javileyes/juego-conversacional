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
        # Abrir el archivo en modo de añadir ('append'). El modo 'a' asegura que si el archivo no existe, se crea.
        with open(filename, 'a') as file:
            if add_newline:
                file.write(text + "\n")  # Añade el texto y un salto de línea al final
            else:
                file.write(text)  # Añade solo el texto sin salto de línea
    except IOError as e:
        # Manejar posibles errores de entrada/salida, como permisos insuficientes
        print(f"Ocurrió un error al abrir o escribir en el archivo: {e}")
    except Exception as e:
        # Manejar cualquier otro tipo de errores
        print(f"Ocurrió un error inesperado: {e}")
        
#########################################
####    MODEL IN LLAMA_CPP
#########################################
# %%writefile cargar_llama_cpp.py
# from ctransformers import AutoModelForCausalLM, AutoTokenizer

import os
# import accelerate

import subprocess

# Definir las variables de entorno y las rutas
BASE_FOLDER = "./"
REPO = "QuantFactory"
TYPE_MODEL = "Meta-Llama-3-8B-Instruct-GGUF"
MODEL = "Meta-Llama-3-8B-Instruct.Q8_0.gguf"
MODEL_PATH = os.path.join(BASE_FOLDER, MODEL)
CONTEXT_LENGTH = 8192

# Crear el directorio base si no existe
if not os.path.exists(BASE_FOLDER):
    os.mkdir(BASE_FOLDER)
    print("Creado directorio base:", BASE_FOLDER)

# Descargar el modelo si no existe
if not os.path.exists(MODEL_PATH):
    url = f"https://huggingface.co/{REPO}/{TYPE_MODEL}/resolve/main/{MODEL}?download=true"
    cmd = f'curl -L "{url}" -o "{MODEL_PATH}"'
    print("Descargando:", MODEL)
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        print("Descarga completa.")
    except subprocess.CalledProcessError as e:
        print("Error al descargar el archivo:", e)

# %cd {BASE_FOLDER}

from llama_cpp import Llama

model=None
# eos_token_id=None

# Función para cargar el modelo si aún no está cargado
def load_model():
    global model
    # global tokenizer
    if model is None:  # Verifica si model está vacío o no parece ser un modelo válido
        print("Cargando modelo...")
        # model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", device_map='auto', load_in_8bit=True, trust_remote_code=True)
        # model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", device_map='auto', torch_dtype="auto", trust_remote_code=True)
        enable_gpu = True  # offload LLM layers to the GPU (must fit in memory)

        model = Llama(
            model_path=MODEL_PATH,
            n_gpu_layers=-1 if enable_gpu else 0,
            n_ctx=CONTEXT_LENGTH,
            # verbose=False,
        )
        model.verbose=False

        print("Modelo cargado.")
    else:
        print("Modelo ya estaba cargado.")


load_model()

import whisper
modelWhisper = whisper.load_model('turbo')


from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface

# Carga el modelo y la configuración
models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
    "facebook/fastspeech2-en-ljspeech",
    arg_overrides={"vocoder": "hifigan", "fp16": False}
)

# Asegúrate de que models es una lista
if not isinstance(models, list):
    models = [models]

modelT2S = models[0]
# modelT2S = modelT2S.to('cuda:0')

TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)

# Aquí, asumimos que task.build_generator puede manejar correctamente el objeto cfg y model
generator = task.build_generator(models, cfg)
#MODELO LO CARGAMOS A PARTE PORQUE TARDA EN CARGARSE

#########################################
####    CHATBOT
#########################################
# %%writefile modelo_llama3.py
# from cargar_llama_cpp import model
LOGGING = False
from threading import Lock

# global model
def encontrar_coincidencia(texto, cadena_busqueda="<|eot_id|>"):
    """
    Esta función busca la primera aparición de una cadena de búsqueda en un texto dado y devuelve el substring
    desde el principio del texto hasta el final de esta coincidencia (incluida).
    
    Parámetros:
    texto (str): El texto en el que se buscará la cadena.
    cadena_busqueda (str): La cadena de caracteres que se buscará en el texto.
    
    Retorna:
    str: El substring desde el inicio hasta el final de la primera coincidencia de la cadena buscada,
    incluyendo la coincidencia. Si no se encuentra ninguna coincidencia, devuelve una cadena vacía.
    """
    # Buscar la posición de la primera coincidencia de la cadena en el texto
    indice = texto.find(cadena_busqueda)
    
    if indice != -1:
        # Devolver el substring desde el inicio hasta el final de la coincidencia
        return texto[:indice + len(cadena_busqueda)]
    else:
        # Devolver una cadena vacía si no hay coincidencia
        return ""


# VENTANA DESLIZANTE
def ajustar_contexto(texto, max_longitud=15000, secuencia="<|start_header_id|>", system_end="<|eot_id|>"):
    system_prompt = encontrar_coincidencia(texto, system_end)
    # Comprobar si la longitud del texto es mayor que el máximo permitido
    if len(texto) > max_longitud:
        indice_secuencia = 0

        while True:
            # Buscar la secuencia de ajuste
            indice_secuencia = texto.find(secuencia, indice_secuencia + 1)

            # Si la secuencia no se encuentra o el texto restante es menor que la longitud máxima
            if indice_secuencia == -1 or len(system_prompt) + len(texto) - indice_secuencia <= max_longitud:
                break

        # Si encontramos una secuencia válida
        if indice_secuencia != -1:
            return system_prompt + texto[indice_secuencia:]

        else:
            # Si no se encuentra ninguna secuencia adecuada, tomar los últimos max_longitud caracteres
            return system_prompt + texto[-max_longitud + len(system_prompt):]
    else:
        return system_prompt + texto



generate_lock = Lock()

def pre_warm_chat(historico, max_additional_tokens=100, stop=["</s>","user:"], short_answer=True, streaming=False, printing=False):
 
    # if short_answer:
    #     # añade como stop el salto de linea
    #     stop.append("\n")

    outputs = ""

    with generate_lock:
        response=model(prompt=historico, max_tokens=max_additional_tokens, temperature=0, top_p=1,
                    top_k=0, repeat_penalty=1,
                    stream=True)


        respuesta = ""

        for chunk in response:
            trozo = chunk['choices'][0]['text']
            # trozo.replace("\n", "")
            # trozo.replace("<|EOT|>", "")
            # for caracter in trozo:
            #     cadena_con_codigos += f"{caracter}({ord(caracter)}) "
            respuesta += trozo
            print(trozo, end="", flush=True)
            # linea += trozo

            # if len(linea)>35:
                # print(linea, end="", flush=True)  # Impresión en consola

                # linea = ""


        outputs = historico + respuesta
        return historico, outputs



import threading

class EstadoGeneracion:
    def __init__(self):
        # lista de 100 partes del texto
        self.parts = [""]*100
        self.top = -1
        self.generando = False
        self.primer_audio = ""
        # self.lock = threading.Lock()


estado_generacion = {}
estado_generacion['anonimo'] = EstadoGeneracion()

# generate_lock = Lock()

def generate_in_file_parts(userID, historico, ai, user, input_text, max_additional_tokens=2000, short_answer=False, streaming=True, printing=True):
    global estado_generacion

    printf(HISTORICO_LOG, f"generando para USER:{userID}\nhistorico:{historico}\ninput_text:{input_text}")
    # global generate_lock
    if userID not in estado_generacion:
        estado_generacion[userID] = EstadoGeneracion()

    estado_generacion[userID].generando = True
    with generate_lock:
        estado_generacion[userID].top = -1
        print(f"Empezamos a generar ponemos el TOP a {estado_generacion[userID].top} para USER:{userID}!!:", input_text)
        indiceParte = 0
        # estado_generacion[userID].generando = True
        print(f"generando={estado_generacion[userID].generando}; Generando respuesta para USER:{userID}:", input_text)
        # estado_generacion.parts = []  # lista de partes del texto
        parte_actual = ""  # añade la primera parte

        if short_answer:
            # añade como stop el salto de linea
            stop.append("\n")


        prompt = f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        final_prompt = historico + "\n" + prompt


        model_inputs = final_prompt

        outputs = ""
        print(f"{ai}:", end="")


        # outputs = ""
        colchon = (CONTEXT_LENGTH - max_additional_tokens)*3
        print("Longitud:",len(final_prompt), "Colchon:", colchon)
        if len(final_prompt)> colchon: #cuenta la vieja cada token son 3 caracteres (como poco)
            print("Ajustando contexto!!!")
            final_prompt = ajustar_contexto(final_prompt, max_longitud=colchon)
            print(final_prompt)
            #contexto ajustado imprimir los primeros 500 caracteres
            # print(final_prompt[:500])
            #imprimir los 500 últimos caracteres
            # print(final_prompt[-500:])

        response=model(prompt=final_prompt, max_tokens=max_additional_tokens, temperature=0, top_p=1,
                      top_k=0, repeat_penalty=1,
                      stream=True)


        respuesta = ""
        estado_generacion[userID].primer_audio = "wait"

        for chunk in response:
            trozo = chunk['choices'][0]['text']
            # trozo.replace("\n", "")
            # trozo.replace("<|EOT|>", "")
            # for caracter in trozo:
            #     cadena_con_codigos += f"{caracter}({ord(caracter)}) "
            respuesta += trozo
            print(trozo, end="", flush=True)

            outputs += trozo
            parte_actual += trozo
            if trozo in ",;:.?!" and len(parte_actual)>44 or trozo in "." and len(parte_actual)>1:

                if indiceParte == 0: #creamos primer audio rápido para rápida respuesta
                    estado_generacion[userID].primer_audio = voz_sintetica_english(parte_actual, "true")
                    
                estado_generacion[userID].parts[indiceParte] = parte_actual
                estado_generacion[userID].top = indiceParte
                if LOGGING:
                    print(f"trozo generado para USER: {userID}:", parte_actual)
                    print("se ha generado para entrada de indiceParte (ahora TOP tb vale esto):", indiceParte)
                indiceParte += 1                
                # print("se incrementa indiceParte (pero TOP aun no) a:", indiceParte)
                parte_actual = ""


        if len(parte_actual)>1:
            estado_generacion[userID].parts[indiceParte] = parte_actual
            estado_generacion[userID].top = indiceParte

        all_text = model_inputs + outputs + "<|eot_id|>"
        estado_generacion[userID].generando = False
        if LOGGING:
            print(f"generando={estado_generacion[userID].generando}; Respuesta Terminada. El total generado para {user}:", outputs)

        return all_text, outputs


#########################################
####    SERVER
#########################################
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from threading import Lock
import threading
import os
import torch
from pydub import AudioSegment
import pandas as pd
import random

import time


# import whisper
# modelWhisper = whisper.load_model('medium')



# parts = []  # lista de partes del texto
# generando = False
# global model



# from modelo_llama3 import generate_in_file_parts, pre_warm_chat
if LOGGING:
    print("El modelo es:", model)

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
else:
    short_answer = False

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


historico = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{saludo}<|eot_id|>"


# load model
# load_model()

if LOGGING:
    print(f"{ai}:", saludo)

# Crea un bloqueo para proteger el código contra la concurrencia a la hora de transcribir
transcribe_lock = Lock()


# generate_lock = Lock()

app = Flask(__name__)
# app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024  # 30 MB

#import logging
#logging.basicConfig(level=logging.DEBUG)

CORS(app)

output = ""


@app.route('/alive')
def alive():
    return jsonify(True)



def eliminar_archivos_temp(nombre_inicio='temp_synth_audio'):
    # Obtener una lista de todos los archivos en el directorio actual
    archivos = os.listdir('.')
    
    # Filtrar archivos que comienzan con 'temp_sync'
    archivos_temp = [archivo for archivo in archivos if archivo.startswith(nombre_inicio)]
    
    # Iterar sobre la lista de archivos y eliminarlos
    for archivo in archivos_temp:
        try:
            os.remove(archivo)
            print(f"Archivo eliminado: {archivo}")
        except Exception as e:
            print(f"No se pudo eliminar el archivo {archivo}. Razón: {e}")

@app.route('/inicio', methods=['POST'])
def print_strings():
    # global modelo
    # global historico
    eliminar_archivos_temp("received_audio")
    eliminar_archivos_temp()
    historico = ""

    # Lee el archivo CSV y selecciona un personaje de ficción al azar
    def elegir_personaje_aleatorio():
        df = pd.read_csv('Personajes_ficcion.csv')
        return random.choice(df.iloc[:, 0].tolist())

    # Obtiene los datos del cuerpo de la solicitud
    data = request.json

    # Extrae los strings del objeto JSON
    system_prompt = data.get('system_prompt')
    saludo = data.get('saludo')

    # Preprocesamiento para reemplazar "#personaje" con un personaje aleatorio
    if "#personaje" in system_prompt:
        personaje_aleatorio = elegir_personaje_aleatorio()
        system_prompt = system_prompt.replace("#personaje", personaje_aleatorio)

    # Preprocesamiento para reemplazar "#personaje" con un personaje aleatorio en el saludo
    if "#personaje" in saludo:
        saludo = saludo.replace("#personaje", personaje_aleatorio)

    # Imprime los strings en el log del servidor
    if LOGGING:
        print("INICIALIZANDO CONVERSACIÓN")
    conversation_file = 'conversacion.mp3'
    # si existe el archivo de conversación, lo elimina
    if os.path.exists(conversation_file):
        os.remove(conversation_file)

    if LOGGING:
        print(f"system: {system_prompt}, saludo: {saludo}")

    # if modelo == "mistral":
    #     historico = f"system\n{system_prompt}\nassistant\n{saludo}\n"
    # elif modelo == "zypher":
    #     historico = f"{system_prompt}</s>\n\n{saludo}</s>\n"
    historico = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{saludo}"#<|eot_id|>"

    pre_warm_chat(historico + "<|eot_id|>")

    # Retorna una respuesta para indicar que se recibieron y procesaron los datos
    return jsonify({"message": saludo, "historico": historico}), 200


@app.route('/get-translations-file', methods=['GET'])
def get_translations():
    return send_from_directory(directory='.', path='translations.csv', as_attachment=True)

import csv
import shutil
@app.route('/save-translations-file', methods=['POST'])
def save_translations():
    data = request.json  # Asume que el cliente envía los datos como JSON
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        # Hace una copia de seguridad del archivo translations.csv antes de modificarlo
        shutil.copy('translations.csv', 'translations.csv.bak')

        # Abre el archivo translations.csv para escribir y actualiza con los datos recibidos
        with open('translations.csv', mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, delimiter='#')
            for row in data:
                writer.writerow(row)

        return jsonify({'message': 'File successfully saved'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/all_conversation', methods=['GET'])
def all_conversation():
    # Asegúrate de que el path al archivo sea correcto para tu estructura de proyecto
    filepath = 'conversacion.mp3'
    if not os.path.exists(filepath):
        return jsonify(error="Archivo de conversación no encontrado"), 404

    # Leer el archivo y convertirlo a base64
    with open(filepath, 'rb') as audio_file:
        audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')

    return jsonify(audio_base64=audio_base64)



import subprocess

def convert_ogg_to_mp3(source_ogg_path, target_mp3_path):
    """
    Utiliza ffmpeg para convertir un archivo .ogg a .mp3.
    """
    command = ['ffmpeg', '-y' ,'-i', source_ogg_path, target_mp3_path]
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Si el comando falla, imprime la salida de error
    if process.returncode != 0:
        print(f"Error al convertir {source_ogg_path} a {target_mp3_path}")
        print("Salida de error de ffmpeg:")
        print(process.stderr.decode())




def convert_wav_to_mp3(source_wav_path, target_mp3_path):
    command = ['ffmpeg', '-i', source_wav_path, target_mp3_path]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


lockAddAudio = threading.Lock()
def add_audio_to_conversation_async(source_path, convert_to_mp3=False):
   
    def task():
        with lockAddAudio:
            if convert_to_mp3:
                # Convertir de WAV a MP3 si es necesario
                temp_mp3_path = source_path.replace('.wav', '.mp3')
                convert_wav_to_mp3(source_path, temp_mp3_path)
                final_path = temp_mp3_path
            else:
                final_path = source_path

            # Añadir al archivo de conversación
            sound = AudioSegment.from_file(final_path)
            conversation_file = 'conversacion.mp3'
            if os.path.exists(conversation_file):
                conversation_audio = AudioSegment.from_mp3(conversation_file)
                combined_audio = conversation_audio + sound
        
                    
            else:
                combined_audio = sound
            combined_audio.export(conversation_file, format='mp3')

            # Limpiar archivos temporales
            os.remove(source_path)
            if convert_to_mp3:
                os.remove(temp_mp3_path)


    thread = threading.Thread(target=task)
    thread.start()



def generate_chat_background(userID, entrada, phistorico, ai, user, short_answer):
    # global output  # Indicar que se utilizará la variable global 'output'
    if LOGGING:
        print("OJOOOOOOOO!!!!!!  generate_chat_background USERID:", userID, "entrada:", entrada)
    start_generation_time = time.time()
    # Ejecutar la generación de chat en un hilo aparte
    historico_local, output_local = generate_in_file_parts(userID, phistorico, ai, user, input_text=entrada, max_additional_tokens=2048, short_answer=short_answer, streaming=True, printing=False)
    end_generation_time = time.time()
    generation_duration = end_generation_time - start_generation_time
    if LOGGING:
        print(f"Generación completada en {generation_duration} segundos")

    # Actualizar las variables globales con los resultados obtenidos
    # global historico
    # historico = historico_local
    # output = output_local

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if LOGGING:
        print("Transcribiendo audio...")
    global user
    global ai


    if 'userID' not in request.form:
        return jsonify(error="No se proporcionó userID"), 400
    userID = int(request.form['userID'])

    if 'historico' not in request.form:
        return jsonify(error="No se proporcionó historico"), 400
    historico = request.form['historico']  # Asumiendo que se envía como JSON y necesitará ser parseado en Python

    # Extraer el archivo
    if 'file' not in request.files:
        return jsonify(error="No se proporcionó file"), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify(error="No selected file"), 400

    timestamp = int(time.time() * 1000)
    ogg_filepath = f"received_audio_{timestamp}.ogg"
    file.save(ogg_filepath)

    start_transcribe_time = time.time()
    if LOGGING:
        print("antes del transcribe lock, userID:", userID)
    with transcribe_lock:
        if LOGGING:
            print("después del transcribe lock, userID:", userID)
        transcripcion = modelWhisper.transcribe(ogg_filepath, fp16=False, language=idioma)
        transcripcion = transcripcion["text"]
    end_transcribe_time = time.time()
    transcribe_duration = end_transcribe_time - start_transcribe_time
    if LOGGING:
        print(f"Transcripción completada en {transcribe_duration} segundos")
        print("transcripción:", transcripcion)

    # Iniciar la generación de chat en un hilo aparte
    thread = threading.Thread(target=generate_chat_background, args=(userID, transcripcion, historico, ai, user, short_answer))
    thread.start()

    # Inicia el proceso de adición del audio .ogg en segundo plano, considerando su conversión a .mp3
    add_audio_to_conversation_async(ogg_filepath)

    # La respuesta ya no incluirá 'output' porque se generará en segundo plano
    prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{transcripcion}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    historico += prompt    
    return jsonify(entrada=transcripcion, prompt=prompt, entrada_traducida="")
    # return jsonify(entrada=transcripcion, historico=historico, entrada_traducida="")





@app.route('/only_transcribe', methods=['POST'])
def only_transcribe_audio():
    if LOGGING:
        print("Transcribiendo audio...")
    # global historico
    # global user
    # global ai

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
    if LOGGING:
        print("antes del transcribe lock")
    with transcribe_lock:
        if LOGGING:
            print("después del transcribe lock")
        transcripcion = modelWhisper.transcribe(ogg_filepath, fp16=False, language=idioma)
        transcripcion = transcripcion["text"]
    end_transcribe_time = time.time()
    transcribe_duration = end_transcribe_time - start_transcribe_time
    if LOGGING:
        print(f"Transcripción completada en {transcribe_duration} segundos")
        print("transcripción:", transcripcion)

    # Iniciar la generación de chat en un hilo aparte
    #thread = threading.Thread(target=generate_chat_background, args=(transcripcion, historico, ai, user, short_answer))
    #thread.start()

    # Inicia el proceso de adición del audio .ogg en segundo plano, considerando su conversión a .mp3
    add_audio_to_conversation_async(ogg_filepath)

    # La respuesta ya no incluirá 'output' porque se generará en segundo plano
    return jsonify(entrada=transcripcion, entrada_traducida="")



@app.route('/get_next_part', methods=['GET'])
def get_next_part():
    global estado_generacion
    if LOGGING:
        print("LAS CLAVES de usuario y sus TOP:")
        for clave in estado_generacion.keys():
            print(clave," TOP:", estado_generacion[clave].top)

    userID = request.args.get('userID', default=0, type=int)
    # Obtener el índice de la solicitud. Si no se proporciona, por defecto es None
    index = request.args.get('index', default=0, type=int)

    if LOGGING:
        print(f"userID:{userID} partes: {estado_generacion[userID].parts}, generando: {estado_generacion[userID].generando}, index: {index}, estado_generacion[userID].top: {estado_generacion[userID].top}")

    while True:
        # if estado_generacion.parts:
            # Verificar si el índice es válido
        if index is not None and index >= 0 and index <= estado_generacion[userID].top:

            part = estado_generacion[userID].parts[index]
            estado_generacion[userID].parts[index] = ""  # Elimina el elemento en el índice dado
            if LOGGING:
                print("con index:", index, "estado_generacion[userID].top:", estado_generacion[userID].top)    
                print(f"Enviando parte: {part}")
            return jsonify(output=part)
            # else:
            #     print("Índice inválido o fuera de límites")
            #     return jsonify(error="Índice inválido o fuera de límites"), 400
        elif estado_generacion[userID].generando:
            if LOGGING:
                print("Esperando a que se generen más partes...")
            time.sleep(0.1)  # Espera 0.1 segundos antes de volver a verificar
        else:
            if LOGGING:
                print("No hay más partes para enviar", "index:", index, "estado_generacion[userID].top:", estado_generacion[userID].top)
            return jsonify(output="") # Si 'generando' es False y 'parts' está vacía, devuelve una cadena vacía



@app.route('/texto', methods=['POST'])
def process_text():
    # global historico
    global user
    global ai

    # Recibe el texto directamente del cuerpo de la solicitud
    data = request.json
    if not data or 'texto' not in data:
        return jsonify(error="No se proporcionó texto"), 400

    texto = data['texto']

    if 'historico' not in data:
        return jsonify(error="No se proporcionó historico"), 400

    historico = data['historico']

    if 'userID' not in data:
        return jsonify(error="No se proporcionó userID"), 400

    userID = data['userID']

    if LOGGING:
        print("HISTORICO!!!:", historico)

    # Utiliza la variable 'idioma' declarada globalmente
    global idioma


    # Generación de respuesta basada en el texto proporcionado
    thread = threading.Thread(target=generate_chat_background, args=(userID, texto, historico, ai, user, short_answer))
    thread.start()


    # si el idioma es español, traduce la respuesta al español

    prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{texto}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    historico += prompt
    return jsonify(entrada=texto, historico=historico, entrada_traducida="")



import torch  # Importamos PyTorch para poder usar la función `to()`

# Función para mover recursivamente todos los tensores en una estructura anidada a un dispositivo
def move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(item, device) for item in obj]
    else:
        return obj


# PREPARAMOS INSTANCIAS FAIRSEQ

if idioma == "en":
    print("idioma ingles") # ESTO HAY QUE CAMBIARLO 
    # from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
    # from fairseq.models.text_to_speech.hub_interface import TTSHubInterface

    # # Carga el modelo y la configuración
    # models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
    #     "facebook/fastspeech2-en-ljspeech",
    #     arg_overrides={"vocoder": "hifigan", "fp16": False}
    # )

    # # Asegúrate de que models es una lista
    # if not isinstance(models, list):
    #     models = [models]

    # modelT2S = models[0]
    # modelT2S = modelT2S.to('cuda:0')

    # TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)

    # # Aquí, asumimos que task.build_generator puede manejar correctamente el objeto cfg y model
    # generator = task.build_generator(models, cfg)



elif idioma == "es":
    # El modelo no entiende de números aritméticos. Esta función los convierte a palabras.
    import re
    from num2words import num2words

    def number_to_words(num_str):
        try:
            num = int(num_str)
            return num2words(num, lang='es')
        except ValueError:
            return "Por favor, introduzca un número válido."

    def process_numbers_in_line(line):
        def replace_with_words(match):
            return number_to_words(match.group())

        return re.sub(r'\b\d+\b', replace_with_words, line)

    # Ejemplo de uso
    line = "Tengo 3 manzanas y 15 naranjas, sumando un total de 18 frutas."
    new_line = process_numbers_in_line(line)
    print(new_line)
    # Salida: "Tengo tres manzanas y quince naranjas, sumando un total de dieciocho frutas."


    # Diccionario con las traducciones

    def process_abrev(line):
        translations = {
        'Dr': 'doctor',
        'Sr': 'señor',
        'Sra': 'señora',
        # Añade más traducciones aquí
    }
        for abbr, full in translations.items():
            line = line.replace(f'{abbr}.', full)
            line = line.replace(f'{abbr} ', f'{full} ')
        return line

    def otras_traducciones(line):

        translations = {
        '-': ',',
        '—': ',',
        '%': ' por ciento '
        # Añade más traducciones aquí
        }

        for old, new in translations.items():
            line = line.replace(old, new)
        return line


    def preprocesado_al_modelo(line):
        line_with_numbers = process_numbers_in_line(line)
        line_with_both = process_abrev(line_with_numbers)
        line_with_all = otras_traducciones(line_with_both)
        return line_with_all

    import os
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    print(os.environ['TF_ENABLE_ONEDNN_OPTS'])


    from pydub import AudioSegment, silence

    def quitar_silencios(input_filepath, output_filepath, min_silence_len=1500, new_silence_len=750, silence_thresh=-60):
        """
        Elimina silencios largos de un archivo de audio.

        Parámetros:
        - input_filepath: ruta al archivo de audio de entrada (MP3).
        - output_filepath: ruta al archivo de audio de salida (MP3).
        - min_silence_len: duración mínima del silencio a eliminar (en milisegundos).
        - new_silence_len: duración de los nuevos segmentos de silencio (en milisegundos).
        - silence_thresh: umbral de silencio (en dB).
        """

        # Cargar el archivo de audio
        audio_segment = AudioSegment.from_wav(input_filepath)

        # Encuentra los segmentos de audio separados por silencios
        segments = silence.split_on_silence(audio_segment, min_silence_len=min_silence_len, silence_thresh=silence_thresh)

        # Crear un nuevo segmento de audio con silencios ajustados
        new_audio_segment = AudioSegment.empty()
        silence_chunk = AudioSegment.silent(duration=new_silence_len)  # Chunk de silencio de la nueva duración

        # Añade cada segmento de audio al nuevo audio, intercalando con los nuevos segmentos de silencio
        for segment in segments:
            new_audio_segment += segment + silence_chunk

        # Removemos el último chunk de silencio añadido
        new_audio_segment = new_audio_segment[:-new_silence_len]

        # Guarda el nuevo archivo de audio
        new_audio_segment.export(output_filepath, format="wav")


    # Cargamos el modelo generador fairseq
    from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
    from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
    import IPython.display as ipd





    # Cargamos el modelo y la configuración desde el modelo preentrenado de Hugging Face
    models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
        "facebook/tts_transformer-es-css10",
        arg_overrides={"vocoder": "hifigan", "fp16": False}
    )
    modelT2S = models[0]

    # Movemos el modelo al dispositivo GPU
    # modelT2S = modelT2S.to('cuda:0')

    # Actualizamos la configuración con los datos del task
    TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)

    # Creamos el generador
    generator = task.build_generator([modelT2S], cfg)


    import torchaudio

    import re

    def dividir_texto_con_minimo_palabras(texto, min_palabras=8):
        partes = re.split(r'([.,;:?!])', texto)
        partes_filtradas = [parte.strip() for parte in partes if parte.strip()]
        partes_combinadas = []
        parte_actual = ''

        for parte in partes_filtradas:
            if parte in '.,;:?!':
                parte_actual += parte
                if len(parte_actual.split()) >= min_palabras:
                    partes_combinadas.append(parte_actual)
                    parte_actual = ''
                else:
                    parte_actual += ' '
            else:
                parte_actual += parte + ' '

        if len(parte_actual.strip()) > 10:
            partes_combinadas.append(parte_actual.strip())

        return partes_combinadas

    def combinar_audios(audios_temporales):
        audio_combinado = "audio_combinado.wav"
        # Cargar el primer archivo de audio para inicializar la concatenación
        wav_total, rate = torchaudio.load(audios_temporales[0])

        # Iterar sobre los archivos restantes y concatenarlos
        for archivo in audios_temporales[1:]:
            wav, _ = torchaudio.load(archivo)
            wav_total = torch.cat((wav_total, wav), 1)

        # Guardar el audio combinado en un archivo final
        torchaudio.save(audio_combinado, wav_total, rate)

        return audio_combinado

    def voz_sintetica_spanish(text):
        text = preprocesado_al_modelo(text)

        lista_dividida = dividir_texto_con_minimo_palabras(text)

        audios_temporales = []

        for parte in lista_dividida:
            # Preparamos los datos de entrada para el modelo
            sample = TTSHubInterface.get_model_input(task, parte)

            # Movemos los datos al dispositivo GPU
            # sample = move_to_device(sample, 'cuda:0')

            # Realizamos la predicción
            wav, rate = TTSHubInterface.get_prediction(task, modelT2S, generator, sample)

            if len(wav.shape) == 1:
                wav = wav.unsqueeze(0)

            # temp_file_name = "Temporal.wav"
            temp_file_name = f"temporal_{parte[:10]}.wav"
            torchaudio.save(temp_file_name, wav.to('cpu'), rate)
            audios_temporales.append(temp_file_name)

            combinado = combinar_audios(audios_temporales)
            sin_silencios = "sin_silencios.wav"
            # quitamos silencios
            quitar_silencios(combinado, sin_silencios, min_silence_len=1500, new_silence_len=750, silence_thresh=-60)


        with open(sin_silencios, "rb") as audio_file:
            audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')

        return audio_base64



import base64
@app.route('/audio', methods=['POST'])
def generate_audio():
    texto = request.json.get('texto')
    pausa = request.json.get('pausa')
    if LOGGING:
        print('PAUSA!!!!!!!!:', pausa)
        print('TEXTO!!!!!!!!:', texto)
    if not texto:
        return jsonify(error="No se proporcionó texto"), 400

    if idioma == "en":
        audio_base64 = voz_sintetica_english(texto, pausa)
        return jsonify(audio_base64=audio_base64)
    elif idioma == "es":
        audio_base64 = voz_sintetica_spanish(texto)
        return jsonify(audio_base64=audio_base64)


import base64
@app.route('/primer_audio', methods=['GET'])
def primer_audio():
    # if LOGGING:
    

    userID = request.args.get('userID', default=0, type=int)
   
    if idioma == "en":
        while estado_generacion[userID].primer_audio == "wait":
            time.sleep(0.1)
            if LOGGING:
                print("esperando primer audio")
        audio_base64 = estado_generacion[userID].primer_audio
        # print('RECUPERANDO PRIMER AUDIO!!!!!!!!:', audio_base64)
        return jsonify(audio_base64=audio_base64)
    elif idioma == "es":
        audio_base64 = voz_sintetica_spanish(texto)
        return jsonify(audio_base64=audio_base64)


import io
import soundfile as sf


def add_comma_after_punctuation(text: str) -> str:
    # Lista de caracteres después de los cuales se debe agregar una coma
    punctuation_marks = ['.', '!', '?', '(', ')', ':', '\n']

    # Recorre cada marca de puntuación y añade una coma después de cada ocurrencia
    for mark in punctuation_marks:
        text = text.replace(mark, mark + ',...,')

    return text

# Ejemplo de uso de la función
#example_text = "Hello! How are you? I hope you're doing well. Let's meet tomorrow."
#modified_text = add_comma_after_punctuation(example_text)
#print(modified_text)

import io
import base64
import soundfile as sf
import os
import threading
from pydub import AudioSegment
import subprocess


def add_silence_to_audio(audio_path, duration_ms=3000):
    """Añade un segmento de silencio al final de un archivo de audio."""
    # Carga el audio
    sound = AudioSegment.from_wav(audio_path)
    # Genera el silencio
    silence = AudioSegment.silent(duration=duration_ms)
    # Concatena el audio con el silencio
    combined = sound + silence
    # Guarda el nuevo archivo
    combined.export(audio_path, format='wav')


def voz_sintetica_english(texto, pausa="true"):
    if pausa == "true":
        texto = add_comma_after_punctuation(texto)
    # Preparamos los datos de entrada para el modelo
    sample = TTSHubInterface.get_model_input(task, texto)

    # Movemos los datos al dispositivo GPU
    # sample = move_to_device(sample, 'cuda:0')

    # Realizamos la predicción
    wav, rate = TTSHubInterface.get_prediction(task, modelT2S, generator, sample)


        # Convertimos el tensor wav a un buffer de audio en memoria y luego a un archivo temporal
    temp_wav_path = f"temp_synth_audio_{int(time.time() * 1000)}.wav"
    with io.BytesIO() as audio_buffer:
        sf.write(audio_buffer, wav.cpu().numpy(), rate, format='WAV')
        audio_buffer.seek(0)  # Regresamos al inicio del buffer para leerlo
        # Guardar en un archivo temporal
        with open(temp_wav_path, 'wb') as f:
            f.write(audio_buffer.read())

  

    # if len(texto) <= 20:
    #     # Añade silencio al final del archivo de audio
    #     add_silence_to_audio(temp_wav_path, 1000)  # Añade 1 segundo de silencio para que no de problemas en audios cortos

    if len(texto) <= 30:
        print("Añadiendo 700 milisegundos de silencio")
        add_silence_to_audio(temp_wav_path, 700)

    elif len(texto) <= 44:
        print("Añadiendo 0.5 segundos de silencio")
        add_silence_to_audio(temp_wav_path, 500)

    # Añadir el audio al archivo de conversación en segundo plano
    add_audio_to_conversation_async(temp_wav_path, convert_to_mp3=True)  # Asegúrate de implementar la conversión dentro de esta función si es necesario


    # Convertir el buffer a base64 para retornar
    with open(temp_wav_path, 'rb') as f:
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
