from prompt_toolkit import prompt
from prompt_toolkit.key_binding import KeyBindings

from modelo_Zypher_beta import generate_long_chat


ai = "assistant"
user = "user"

contexto = """

"""

system_prompt = """
You are a kind and helpful assistan bot. You are here to help the user to find the best answer to his question.
"""

saludo = "Hi, I am a kind and helpful assistant bot. I am here to help you to find the best answer to your question."

historico = f"<|system|>{system_prompt}</s>\n{ai}\n{saludo}</s>\n"



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
    historico = generate_long_chat(historico, ai, user, input_text=input_text,max_additional_tokens=2048)
    # print response
    # print(salida)
    print(f"\n################################################\n")