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


import whisper
modelWhisper = whisper.load_model('medium')


modelo = "zypher"

# parts = []  # lista de partes del texto
# generando = False

if modelo == "mistral":
    from modelo_mistral_base import generate_in_file_parts, load_model
elif modelo == "zypher":
    from modelo_Zypher_beta import generate_in_file_parts, pre_warm_chat, load_model, estado_generacion
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

if modelo == "mistral":
    historico = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>assistant\n{saludo}<|im_end|>\n"
elif modelo == "zypher":
    historico = f"<|system|>{system_prompt}</s>\n<|assistant|>\n{saludo}</s>\n"


# load model
load_model(user=user, ai=ai)

print(f"{ai}:", saludo)

# Crea un bloqueo para proteger el código contra la concurrencia a la hora de transcribir
transcribe_lock = Lock()


# generate_lock = Lock()

app = Flask(__name__)
# app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024  # 30 MB

CORS(app)

output = ""

@app.route('/inicio', methods=['POST'])
def print_strings():
    global modelo
    global historico

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
    print("INICIALIZANDO CONVERSACIÓN")
    conversation_file = 'conversacion.mp3'
    # si existe el archivo de conversación, lo elimina
    if os.path.exists(conversation_file):
        os.remove(conversation_file)
        
    print(f"system: {system_prompt}, saludo: {saludo}")

    if modelo == "mistral":
        historico = f"system\n{system_prompt}\nassistant\n{saludo}\n"
    elif modelo == "zypher":
        historico = f"{system_prompt}</s>\n\n{saludo}</s>\n"

    pre_warm_chat(historico)

    # Retorna una respuesta para indicar que se recibieron y procesaron los datos
    return jsonify({"message": saludo}), 200


@app.route('/get-translations-file', methods=['GET'])
def get_translations():
    return send_from_directory(directory='.', path='translations.csv', as_attachment=True)



import csv
import shutil
# hay algo en la pila de protocolos que impide transferencia de archivos grandes (se guardarán archivos a local) TODO revisar
@app.route('/save-translations-file', methods=['POST'] )
def save_translations():
    print("Guardando archivo de traducciones...", request, "fin de request")
    try:
        data = request.json  # Asume que el cliente envía los datos como JSON
    except Exception as e:
        print("Error al leer los datos del cuerpo de la solicitud")
        return jsonify({'error': str(e)}), 500
    #imprime longitud de data
    print("Longitud de data:", len(data))
    if not data:
        print("No data provided in /save-translations-file")
        return jsonify({'error': 'No data provided'}), 400
    
    try:
        # Hace una copia de seguridad del archivo translations.csv antes de modificarlo
        print("Haciendo copia de seguridad del archivo translations.csv")
        shutil.copy('translations.csv', 'translations.csv.bak')
        print("Copia de seguridad del archivo translations.csv creada con éxito")

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


def add_audio_to_conversation_async(source_path, convert_to_mp3=False):
    def task():
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



def generate_chat_background(entrada, phistorico, ai, user, short_answer):
    global output  # Indicar que se utilizará la variable global 'output'
    start_generation_time = time.time()
    # Ejecutar la generación de chat en un hilo aparte
    historico_local, output_local = generate_in_file_parts(phistorico, ai, user, input_text=entrada, max_additional_tokens=2048, short_answer=short_answer, streaming=True, printing=False)
    end_generation_time = time.time()
    generation_duration = end_generation_time - start_generation_time
    print(f"Generación completada en {generation_duration} segundos")
    
    # Actualizar las variables globales con los resultados obtenidos
    global historico
    historico = historico_local
    output = output_local

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    print("Transcribiendo audio...")
    global historico
    global user
    global ai

    if 'file' not in request.files:
        return jsonify(error="No file part"), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify(error="No selected file"), 400

    timestamp = int(time.time() * 1000)
    ogg_filepath = f"received_audio_{timestamp}.ogg"
    file.save(ogg_filepath)

    start_transcribe_time = time.time()
    with transcribe_lock:
        transcripcion = modelWhisper.transcribe(ogg_filepath, fp16=False, language=idioma)
        transcripcion = transcripcion["text"]
    end_transcribe_time = time.time()
    transcribe_duration = end_transcribe_time - start_transcribe_time
    print(f"Transcripción completada en {transcribe_duration} segundos")

    print("transcripción:", transcripcion)

    # Iniciar la generación de chat en un hilo aparte
    thread = threading.Thread(target=generate_chat_background, args=(transcripcion, historico, ai, user, short_answer))
    thread.start()

    # Inicia el proceso de adición del audio .ogg en segundo plano, considerando su conversión a .mp3
    add_audio_to_conversation_async(ogg_filepath)

    # La respuesta ya no incluirá 'output' porque se generará en segundo plano
    return jsonify(entrada=transcripcion, entrada_traducida="")


@app.route('/only_transcribe', methods=['POST'])
def only_transcribe_audio():
    print("Transcribiendo audio...")
    global historico
    global user
    global ai

    if 'file' not in request.files:
        return jsonify(error="No file part"), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify(error="No selected file"), 400

    timestamp = int(time.time() * 1000)
    ogg_filepath = f"received_audio_{timestamp}.ogg"
    file.save(ogg_filepath)

    start_transcribe_time = time.time()
    with transcribe_lock:
        transcripcion = modelWhisper.transcribe(ogg_filepath, fp16=False, language=idioma)
        transcripcion = transcripcion["text"]
    end_transcribe_time = time.time()
    transcribe_duration = end_transcribe_time - start_transcribe_time
    print(f"Transcripción completada en {transcribe_duration} segundos")

    print("transcripción:", transcripcion)

    # Iniciar la generación de chat en un hilo aparte
    # thread = threading.Thread(target=generate_chat_background, args=(transcripcion, historico, ai, user, short_answer))
    # thread.start()

    # Inicia el proceso de adición del audio .ogg en segundo plano, considerando su conversión a .mp3
    add_audio_to_conversation_async(ogg_filepath)

    # La respuesta ya no incluirá 'output' porque se generará en segundo plano
    return jsonify(entrada=transcripcion, entrada_traducida="")



# from modelo_Zypher_beta import estado_generacion
# estado_lock = threading.Lock()

@app.route('/get_next_part', methods=['GET'])
def get_next_part():
    global estado_generacion

    # Obtener el índice de la solicitud. Si no se proporciona, por defecto es None
    index = request.args.get('index', default=None, type=int)

    print(f"partes: {estado_generacion.parts}, generando: {estado_generacion.generando}, index: {index}, estado_generacion.top: {estado_generacion.top}")

    while True:
        # if estado_generacion.parts:
            # Verificar si el índice es válido
        if index is not None and index >= 0 and index <= estado_generacion.top:
            part = estado_generacion.parts[index]
            estado_generacion.parts[index] = ""  # Elimina el elemento en el índice dado
            # with estado_generacion.lock:  # Asegurarse de que el acceso a 'parts' es seguro
            # part = ""
            # contador = 0
            # while part == "" and contador < 100:
            #     parte = estado_generacion.parts[index]
            #     if parte != "":
                #     part = parte
                #     estado_generacion.parts[index] = ""  # Elimina el elemento en el índice dado
                # else:
                #     time.sleep(0.1)
                #     contador += 1
            print(f"Enviando parte: {part}")
            return jsonify(output=part)
            # else:
            #     print("Índice inválido o fuera de límites")
            #     return jsonify(error="Índice inválido o fuera de límites"), 400
        elif estado_generacion.generando:
            print("Esperando a que se generen más partes...")
            time.sleep(0.1)  # Espera 0.1 segundos antes de volver a verificar
        else:
            print("No hay más partes para enviar", "index:", index, "estado_generacion.top:", estado_generacion.top)
            return jsonify(output="") # Si 'generando' es False y 'parts' está vacía, devuelve una cadena vacía



@app.route('/texto', methods=['POST'])
def process_text():
    global historico
    global user
    global ai

    # Recibe el texto directamente del cuerpo de la solicitud
    data = request.json
    if not data or 'texto' not in data:
        return jsonify(error="No se proporcionó texto"), 400

    texto = data['texto']

    # Utiliza la variable 'idioma' declarada globalmente
    global idioma

    entrada = texto

    # Generación de respuesta basada en el texto proporcionado
    thread = threading.Thread(target=generate_chat_background, args=(entrada, historico, ai, user, short_answer))
    thread.start()


    # si el idioma es español, traduce la respuesta al español


    return jsonify(entrada=texto, entrada_traducida="")



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


# PREPARAMOS INSTANCIAS DE OpenVoice

if idioma == "en":

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

    model = models[0]
    model = model.to('cuda:0')

    TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)

    # Aquí, asumimos que task.build_generator puede manejar correctamente el objeto cfg y model
    generator = task.build_generator(models, cfg)



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
    model = models[0]

    # Movemos el modelo al dispositivo GPU
    model = model.to('cuda:0')

    # Actualizamos la configuración con los datos del task
    TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)

    # Creamos el generador
    generator = task.build_generator([model], cfg)


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
            sample = move_to_device(sample, 'cuda:0')

            # Realizamos la predicción
            wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)

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
    texto = request.json.get('texto', '')

    if not texto:
        return jsonify(error="No se proporcionó texto"), 400

    if idioma == "en":
        audio_base64 = voz_sintetica_english(texto)
        return jsonify(audio_base64=audio_base64)
    elif idioma == "es":
        audio_base64 = voz_sintetica_spanish(texto)
        return jsonify(audio_base64=audio_base64)

import base64
import io
import soundfile as sf


def add_comma_after_punctuation(text: str) -> str:
    # Lista de caracteres después de los cuales se debe agregar una coma
    punctuation_marks = ['.', '!', '?', '(', ')', ':']
    
    # Recorre cada marca de puntuación y añade una coma después de cada ocurrencia
    for mark in punctuation_marks:
        text = text.replace(mark, mark + ',')
    
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



def voz_sintetica_english(texto):
    texto = add_comma_after_punctuation(texto) #preprocesamos para mejora del modelo
    # Preparamos los datos de entrada para el modelo
    sample = TTSHubInterface.get_model_input(task, texto)

    # Movemos los datos al dispositivo GPU
    sample = move_to_device(sample, 'cuda:0')

    # Realizamos la predicción
    wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)


        # Convertimos el tensor wav a un buffer de audio en memoria y luego a un archivo temporal
    temp_wav_path = f"temp_synth_audio_{int(time.time() * 1000)}.wav"
    with io.BytesIO() as audio_buffer:
        sf.write(audio_buffer, wav.cpu().numpy(), rate, format='WAV')
        audio_buffer.seek(0)  # Regresamos al inicio del buffer para leerlo
        # Guardar en un archivo temporal
        with open(temp_wav_path, 'wb') as f:
            f.write(audio_buffer.read())

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
