from prompt_toolkit import prompt
from prompt_toolkit.key_binding import KeyBindings

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

import sys

# Verifica si el comando tenía flag -s o --short
if "-s" in sys.argv or "--short" in sys.argv:
    short_answer = True

traducir = False
if "-es" in sys.argv:
    traducir = True

# Filtra los argumentos para eliminar los flags
args = [arg for arg in sys.argv[1:] if arg not in ["-s", "--short", "-es"]]

# Asigna los valores a system_prompt y saludo basándose en los argumentos restantes
if len(args) > 0:
    system_prompt = args[0]
    # print("SYSTEM PROMPT!!:", system_prompt)
if len(args) > 1:
    saludo = args[1]
    # print("SALUDO!!:", saludo)


if traducir:
    from transformers import MarianMTModel, MarianTokenizer
    
    print("Cargando modelos de traducción...")

    model_name = 'Helsinki-NLP/opus-mt-es-en'  # Modelo para traducir de español a inglés
    modelo_traductor = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)


    def translate_text_to_english(text):
        # print("Traduciendo texto:", text)
        tokens = tokenizer(text, return_tensors='pt', padding=True)
        translated = modelo_traductor.generate(**tokens)
        decoded = []
        for t in translated:
            decoded.append(tokenizer.decode(t, skip_special_tokens=True))
        
        return decoded[0]
    

    model_name = 'Helsinki-NLP/opus-mt-tc-big-en-es'  # Modelo para traducir de inglés a español
    model_big_en_es = MarianMTModel.from_pretrained(model_name)
    tokenizer_big_en_es = MarianTokenizer.from_pretrained(model_name)

    def translate_text_to_spanish(text):
        # with translate_es_lock:
        tokenizer = tokenizer_big_en_es
        model = model_big_en_es

        tokens = tokenizer(text, return_tensors='pt', padding=True)
        translated = model.generate(**tokens)
        # decoded = tokenizer.decode(translation[0], skip_special_tokens=True)
        decoded = []
        for t in translated:
            # decoded = tokenizer.decode(t, skip_special_tokens=True)
            decoded.append(tokenizer.decode(t, skip_special_tokens=True))
        
        return decoded[0]
    


if modelo == "mistral":
    historico = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>assistant\n{saludo}<|im_end|>\n"
elif modelo == "zypher":    
    historico = f"<|system|>{system_prompt}</s>\n<|assistant|>\n{saludo}</s>\n"


# load model
load_model(user=user, ai=ai)
print("\n################################################\n")
print(f"{ai}:", saludo)

if traducir:
    saludo = translate_text_to_spanish(saludo)
    print("\n################################################\n")
    print(f"{ai}:", saludo)


# def wrap_text(text, width=90): #preserve_newlines
#     # Split the input text into lines based on newline characters
#     lines = text.split('\n')

#     # Wrap each line individually
#     wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

#     # Join the wrapped lines back together using newline characters
#     wrapped_text = '\n'.join(wrapped_lines)

#     return wrapped_text



# Crear vinculaciones de teclas personalizadas.
kb = KeyBindings()

# capturar shirft+enter

@kb.add('c-a')
def _(event):
    " Insertar un salto de línea cuando se presiona Control+Enter. "
    event.current_buffer.insert_text('\n')

def input_prompt(mode="ACTION"):
    try:
        if mode == "ACTION":
            user_input = prompt(f'{user}> ', multiline=False, key_bindings=kb)
        elif mode == "DIALOGUE":
            user_input = prompt(f'{user}: ', multiline=False, key_bindings=kb)
        return user_input
    except KeyboardInterrupt:
        # En caso de Ctrl-C, devuelve un string vacío.
        return ''


# def print_wrapped(text, width=80):
#     words = text.split()
#     line = ''

#     for word in words:
#         if len(line) + len(word) + 1 > width:
#             print(line)
#             line = word
#         else:
#             if line:
#                 line += ' '
#             line += word
#     print(line)


while True:
    # read input
    input_text = input_prompt()
    if input_text == "/exit": break
    if input_text == "/historico": 
        print(historico)
        continue
    if input_text == "/clear":
        historico = ""
        continue
    # generate response
    if traducir:
        input_text = translate_text_to_english(input_text)
        print("\n############## TRADUCCIÓN: ####################\n")
        print(input_text)
    historico, respuesta = generate_long_chat(historico, ai, user, input_text=input_text, max_additional_tokens=2048, short_answer=short_answer)
    
    if traducir:
        respuesta = translate_text_to_spanish(respuesta)
        print("\n############## TRADUCCIÓN: ####################\n")
        print(respuesta)
    # print historico
    print(f"\n################################################\n")