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

# Filtra los argumentos para eliminar los flags
args = [arg for arg in sys.argv[1:] if arg not in ["-s", "--short"]]

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


def wrap_text(text, width=90): #preserve_newlines
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text



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


def print_wrapped(text, width=80):
    words = text.split()
    line = ''

    for word in words:
        if len(line) + len(word) + 1 > width:
            print(line)
            line = word
        else:
            if line:
                line += ' '
            line += word
    print(line)


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
    historico = generate_long_chat(historico, ai, user, input_text=input_text, max_additional_tokens=2048, short_answer=short_answer)
    # print response
    # print(salida)
    print(f"\n################################################\n")