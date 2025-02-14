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
REPO = "unsloth"
TYPE_MODEL = "Mistral-Small-24B-Instruct-2501-GGUF"
# MODEL = "Mistral-Small-24B-Instruct-2501.Q8_0.gguf"
MODEL = "Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf"
# MODEL = "Mistral-Small-24B-Instruct-2501-Q6_K.gguf"
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

print("intentamos cargar el modelo")
load_model()

import whisper
modelWhisper = whisper.load_model('turbo')


LOGGING = False
from threading import Lock

# global model
def encontrar_coincidencia(texto, cadena_busqueda="</s>"):
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
def ajustar_contexto(texto, max_longitud=15000, secuencia="[SYSTEM_PROMPT]", system_end="[/SYSTEM_PROMPT]"):
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

def pre_warm_chat(phistorico, max_additional_tokens=0):
 
    # if short_answer:
    #     # añade como stop el salto de linea
    #     stop.append("\n")

    outputs = ""

    with generate_lock:
        response=model(prompt=phistorico, max_tokens=max_additional_tokens, temperature=0, top_p=1,
                    top_k=0, repeat_penalty=1,
                    )


        respuesta = ""
        
        for chunk in response:
            return phistorico, outputs
        #     trozo = chunk['choices'][0]['text']
        #     # trozo.replace("\n", "")
        #     # trozo.replace("<|EOT|>", "")
        #     # for caracter in trozo:
        #     #     cadena_con_codigos += f"{caracter}({ord(caracter)}) "
        #     respuesta += trozo
        #     print(trozo, end="", flush=True)
        #     # linea += trozo

        #     # if len(linea)>35:
        #         # print(linea, end="", flush=True)  # Impresión en consola

        #         # linea = ""


        outputs = phistorico + respuesta
        return phistorico, outputs



import threading

class EstadoGeneracion:
    def __init__(self):
        # lista de 100 partes del texto
        self.parts = [""]*100
        self.top = -1
        self.generando = False
        self.primer_audio = ""
        self.historico = ""
        # self.lock = threading.Lock()


estado_generacion = {}
estado_generacion['anonimo'] = EstadoGeneracion()

# generate_lock = Lock()
import re
def generate_in_file_parts(userID, phistorico, ai, user, input_text, max_additional_tokens=2000):
    global estado_generacion

    printf(HISTORICO_LOG, f"generando para USER:{userID}\nhistorico:{phistorico}\ninput_text:{input_text}")
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




        prompt = f"</s>[INST]{input_text}[/INST]"

        final_prompt = estado_generacion[userID].historico + prompt


        model_inputs = final_prompt

        outputs = ""
        print(f"{ai}:", end="")


        colchon = (CONTEXT_LENGTH - max_additional_tokens)*3
        print("Longitud:",len(final_prompt), "Colchon:", colchon)
        if len(final_prompt)> colchon: #cuenta la vieja cada token son 3 caracteres (como poco)
            print("Ajustando contexto!!!")
            final_prompt = ajustar_contexto(final_prompt, max_longitud=colchon)
            print(final_prompt)


        response=model(prompt=final_prompt, max_tokens=max_additional_tokens, temperature=0, top_p=1,
                      top_k=0, repeat_penalty=1,
                      stream=True)


        respuesta = ""
        estado_generacion[userID].primer_audio = "wait"

        for chunk in response:
            trozo = chunk['choices'][0]['text']
   
            respuesta += trozo
            print(trozo, end="", flush=True)

            outputs += trozo
            parte_actual += trozo

            minimo_caracteres = 34
         
            if trozo in ",;:.?!" and len(parte_actual)> minimo_caracteres or trozo in "." and len(parte_actual)>1:
                parte_actual = re.sub(r'\\{2,}', r'\\', parte_actual)
                parte_actual = parte_actual.replace("\\\"", "\"")
                parte_actual = parte_actual.replace("\\n", "\n")

                if indiceParte == 0: #creamos primer audio rápido para rápida respuesta
                    estado_generacion[userID].primer_audio = voz_sintetica_english(parte_actual)
                    
                estado_generacion[userID].parts[indiceParte] = parte_actual
                estado_generacion[userID].top = indiceParte
                if LOGGING:
                    print(f"trozo generado para USER: {userID}:", parte_actual)
                    print("se ha generado para entrada de indiceParte (ahora TOP tb vale esto):", indiceParte)
                indiceParte += 1                
                # print("se incrementa indiceParte (pero TOP aun no) a:", indiceParte)
                parte_actual = ""


        if len(parte_actual)>1:
            parte_actual = re.sub(r'\\{2,}', r'\\', parte_actual)
            parte_actual = parte_actual.replace("\\\"", "\"")
            parte_actual = parte_actual.replace("\\n", "\n")

            # printf(HISTORICO_LOG, f"trozo ÚLTIMO generado para USER: {userID}:", parte_actual)
            if indiceParte == 0: #creamos primer audio rápido para rápida respuesta
                estado_generacion[userID].primer_audio = voz_sintetica_english(parte_actual)
            estado_generacion[userID].parts[indiceParte] = parte_actual
            estado_generacion[userID].top = indiceParte
            indiceParte += 1
            parte_actual = ""

            if LOGGING:
                print(f"trozo ÚLTIMO generado para USER: {userID}:", parte_actual)
                print("se ha generado para entrada de indiceParte (ahora TOP tb vale esto):", indiceParte)

        all_text = model_inputs + outputs + "</s>"
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


historico = f"[SYSTEM_PROMPT]{system_prompt}[/SYSTEM_PROMPT][INST]Espero un saludo[/INST]{saludo}</s>"


if LOGGING:
    print(f"{ai}:", saludo)

# Crea un bloqueo para proteger el código contra la concurrencia a la hora de transcribir
transcribe_lock = Lock()


app = Flask(__name__)


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
    # global system_prompt
    # Elimina archivos temporales
    eliminar_archivos_temp("received_audio")
    eliminar_archivos_temp()

    # Obtiene los datos enviados (incluyendo system_prompt, saludo y userID)
    data = request.json
    system_prompt = data.get('system_prompt')
    saludo = data.get('saludo')
    userID = data.get('userID')
    printf(HISTORICO_LOG, f"System prompt: {system_prompt}")
    printf(HISTORICO_LOG, f"userID: {userID}")



    if not userID:
        return jsonify(error="No se proporcionó userID"), 400
    userID = int(userID)

    # Si hay tokens especiales, reemplaza #personaje con uno aleatorio
    def elegir_personaje_aleatorio():
        df = pd.read_csv('Personajes_ficcion.csv')
        return random.choice(df.iloc[:, 0].tolist())

    if "#personaje" in system_prompt:
        personaje_aleatorio = elegir_personaje_aleatorio()
        system_prompt = system_prompt.replace("#personaje", personaje_aleatorio)
    if "#personaje" in saludo:
        saludo = saludo.replace("#personaje", personaje_aleatorio)

    
    # Asegura que exista un estado para este userID y lo inicializa con el historial
    if userID not in estado_generacion:
        estado_generacion[userID] = EstadoGeneracion()
    estado_generacion[userID].historico = f"[SYSTEM_PROMPT]{system_prompt}[/SYSTEM_PROMPT][INST]Espero un saludo[/INST]{saludo}"

    pre_warm_chat(estado_generacion[userID].historico + "</s>")

    # Retorna el saludo y el historial junto al userID
    return jsonify({"message": saludo, "historico": estado_generacion[userID].historico, "userID": userID}), 200


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



def generate_chat_background(userID, entrada, phistorico, ai, user):
    global estado_generacion
    # global output  # Indicar que se utilizará la variable global 'output'
    if LOGGING:
        print("OJOOOOOOOO!!!!!!  generate_chat_background USERID:", userID, "entrada:", entrada)
    start_generation_time = time.time()
    # Ejecutar la generación de chat en un hilo aparte
    estado_generacion[userID].historico, output_local = generate_in_file_parts(userID, phistorico, ai, user, input_text=entrada, max_additional_tokens=2048)
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
    global estado_generacion

    if LOGGING:
        print("Transcribiendo audio...")
    global user, ai

    # Extraer el userID del formulario o JSON
    if 'userID' not in request.form:
        return jsonify(error="No se proporcionó userID"), 400
    userID = int(request.form['userID'])

    # Asegurarse de tener un estado para este usuario (historial persistente)
    if userID not in estado_generacion:
        estado_generacion[userID] = EstadoGeneracion()

    # En lugar de recibir el historico desde el cliente, se usa el almacenado en el servidor

    # Extraer el archivo de audio
    if 'file' not in request.files:
        return jsonify(error="No se proporcionó file"), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify(error="No selected file"), 400

    timestamp = int(time.time() * 1000)
    ogg_filepath = f"received_audio_{timestamp}.ogg"
    file.save(ogg_filepath)

    # Transcripción del audio utilizando Whisper (con bloqueo para evitar concurrencia)
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

    # Preparar el prompt y actualizar el historial en el servidor
    prompt = f"[INST]{transcripcion}[/INST]"
    estado_generacion[userID].historico += prompt

    # Iniciar la generación del chat en un hilo aparte usando el historial actualizado
    thread = threading.Thread(target=generate_chat_background, args=(userID, transcripcion, estado_generacion[userID].historico, ai, user))
    thread.start()

    # Iniciar la adición del audio (con posible conversión a MP3) de forma asíncrona
    add_audio_to_conversation_async(ogg_filepath)

    return jsonify(entrada=transcripcion, prompt=prompt, entrada_traducida="")




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
    #thread = threading.Thread(target=generate_chat_background, args=(transcripcion, historico, ai, user))
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
    global user, ai, estado_generacion

    texto = request.json.get('texto') if request.is_json else request.form.get('texto')
    userID = int(request.json.get('userID') if request.is_json else request.form.get('userID'))

    if userID not in estado_generacion:
        estado_generacion[userID] = EstadoGeneracion()

    # Definir prompt inmediatamente
    prompt = f"[INST]{texto}[/INST]"
    # Recuperar y actualizar el historial almacenado
    printf(HISTORICO_LOG, f"EN SERVICIO TEXTO, USER:{estado_generacion[userID].historico}")

    if LOGGING:
        print("Texto recibido:", texto)
        # print("Histórico antes de actualizar:", historico)
        print("Prompt generado:", prompt)

    estado_generacion[userID].historico += prompt
    printf(HISTORICO_LOG, f"SUPERHISTORICO:{estado_generacion[userID].historico}")

    if LOGGING:
        print("Histórico después de actualizar:", estado_generacion[userID].historico)

    thread = threading.Thread(target=generate_chat_background, args=(userID, texto, estado_generacion[userID].historico, ai, user))
    thread.start()

    return jsonify(entrada=texto, prompt=prompt, entrada_traducida="")





if idioma == "en":
    print("idioma ingles")
elif idioma == "es":
    print("idioma español")



import base64
@app.route('/audio', methods=['POST'])
def generate_audio():
    texto = request.json.get('texto')
    # pausa = request.json.get('pausa')
    if LOGGING:
        # print('PAUSA!!!!!!!!:', pausa)
        print('TEXTO!!!!!!!!:', texto)
    if not texto:
        return jsonify(error="No se proporcionó texto"), 400

    # if idioma == "en":
    audio_base64 = voz_sintetica_english(texto)
    return jsonify(audio_base64=audio_base64)
    # elif idioma == "es":
    #     audio_base64 = voz_sintetica_spanish(texto)
    #     return jsonify(audio_base64=audio_base64)


import base64
@app.route('/primer_audio', methods=['GET'])
def primer_audio():
    # if LOGGING:
    

    userID = request.args.get('userID', default=0, type=int)
   
    # if idioma == "en":
    while estado_generacion[userID].primer_audio == "wait":
        time.sleep(0.1)
        if LOGGING:
            print("esperando primer audio")
    audio_base64 = estado_generacion[userID].primer_audio
        # print('RECUPERANDO PRIMER AUDIO!!!!!!!!:', audio_base64)
        # return jsonify(audio_base64=audio_base64)
    # elif idioma == "es":
        # audio_base64 = voz_sintetica_spanish(texto)
        
        
    return jsonify(audio_base64=audio_base64)


import io
import soundfile as sf


def add_comma_after_punctuation(text: str) -> str:
    # Lista de caracteres después de los cuales se debe agregar una coma
    punctuation_marks = ['-','\n', '*','\\']

    # Recorre cada marca de puntuación y añade una coma después de cada ocurrencia
    for mark in punctuation_marks:
        text = text.replace(mark, ',')

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


from kokoro import KPipeline  # Asegúrate de tener instalado el paquete kokoro
if idioma == "en":
    pipeline = KPipeline(lang_code='a')  # 🇺🇸 'a' => American English, 🇬🇧 'b' => British English 🇪🇸 'e' => Spanish es
elif idioma == "es":
    pipeline = KPipeline(lang_code='e')
else:
    pipeline = KPipeline(lang_code='a')

def voz_sintetica_english(texto):
    """
    Sintetiza voz en inglés utilizando el modelo Kokoro.

    Esta función:
      - Opcionalmente añade pausas (usando la función add_comma_after_punctuation).
      - Procesa el texto y obtiene segmentos de audio.
      - Une los segmentos en un solo audio.
      - Guarda el audio en un archivo temporal, lo envía al proceso de adición a la conversación y lo retorna codificado en base64.
    """
    import numpy as np
    import tempfile
    import base64
    import soundfile as sf
    
    global pipeline    

    # Si se desea una pausa, añade comas tras la puntuación (puedes ajustar esta lógica según necesites)
    # if pausa == "true":
    texto = add_comma_after_punctuation(texto)

    # Inicializa el pipeline de Kokoro.
    # Nota: Revisa que 'lang_code' y 'voice' sean los adecuados para la voz en inglés que deseas usar.
    
    voice = 'af_heart'  # Puedes cambiar la voz a otra disponible

    # Genera los segmentos de audio; en este ejemplo se usa split_pattern para separar por saltos de línea.
    generator = pipeline(texto, voice=voice, speed=1, split_pattern=r'\n+')

    # Acumula los segmentos generados en una lista
    audios = []
    for idx, (gs, ps, audio) in enumerate(generator):
        audios.append(audio)

    if audios:
        # Une todos los segmentos en un único array
        combined_audio = np.concatenate(audios, axis=0)
    else:
        # Si no se generó audio, se retorna cadena vacía
        return ""

    sample_rate = 24000  # Tasa de muestreo usada en el ejemplo de Kokoro

    # Crea un archivo temporal para guardar el audio generado
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_file_path = temp_file.name
    temp_file.close()
    sf.write(temp_file_path, combined_audio, sample_rate)

    # Llama a la función que agrega el audio a la conversación de forma asíncrona
    add_audio_to_conversation_async(temp_file_path, convert_to_mp3=True)

    # Lee el archivo temporal y lo codifica en base64 para retornarlo
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