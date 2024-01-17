from flask import Flask, request, jsonify
from threading import Lock
import torch

import time
# Asegúrate de importar tu modelo Whisper correctamente
# from tu_paquete import modelWhisper

import whisper
modelWhisper = whisper.load_model('medium')


# carga modelo PHI-1.5
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_default_device('cuda')

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", trust_remote_code=True, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True, torch_dtype="auto")

max_additional_tokens = 50  # Establece tu límite para el número de tokens adicionales aquí

def find_nth_occurrence(string, substring, n):
    start = 0
    for _ in range(n):
        start = string.find(substring, start) + 1
        if start == 0:
            return -1
    return start - 1


app = Flask(__name__)

# Crea un bloqueo para proteger el código contra la concurrencia a la hora de transcribir
transcribe_lock = Lock()

# Crea un bloqueo para proteger el código contra la concurrencia a la hora de generar texto
generate_lock = Lock()

# ai = "Alice:"
# user = "Bob:"

# historico = f"Contexto, {ai} is a high school teacher, she like play tennis and watch television. This is the firs time {ai} and {user} meet.\n"

iteracion = 0

ai = "King Arthur"
user = "Bob"

historico = f"Context: {user}, a commoner, talk alone with {ai}.\n"

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    global historico
    global user
    global ai
    global iteracion

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
        transcripcion = modelWhisper.transcribe(mp3_filepath, fp16=False, language="en")
        transcripcion = transcripcion["text"]

    prompt = f"{historico}\n{user}:{transcripcion}\n{ai}:"
    print("prompt:", prompt)


    # Asegúrate de que tanto el modelo como los inputs están en el dispositivo GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with generate_lock:
        inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}


        input_length = inputs["input_ids"].size(1)  # Obtén el número de tokens en la entrada

        max_additional_tokens = 50  # Establece tu límite para el número de tokens adicionales aquí
        # Generar texto normalmente sin usar early_stopping para la cadena
        outputs = model.generate(**inputs, max_length=input_length + max_additional_tokens)

        # Decodificar el texto generado
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("output_generated_text:", generated_text)

        # Encuentra la n-ésima ocurrencia del nombre que tenga ai
        n = iteracion + 2  # Cambia este valor para encontrar una ocurrencia diferente
        print("n:", n)
        start_index = find_nth_occurrence(generated_text, f"{ai}:", n-1) + len(f"{ai}:")
        stop_index = find_nth_occurrence(generated_text, f"{user}:", n)
        # si no genera respuesta ficticia de usuario cortamos en el primer salto de linea
        if stop_index == -1:
            stop_index =  generated_text.find("\n", start_index) - 1

        # Si encontramos la n-ésima ocurrencia, cortamos el texto en ese punto
        if stop_index != -1:
            historico = generated_text[:stop_index]
            generated_text = generated_text[start_index:stop_index]

        print("generated_text:", generated_text)
        iteracion += 1
    # # Devuelve la transcripción
    return jsonify(transcripcion=generated_text)
    # Devuelve la transcripción
    # return jsonify(transcripcion={"text": transcripcion})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5500, threaded=True)
