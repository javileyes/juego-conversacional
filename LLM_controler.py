from prompt_toolkit import prompt
from prompt_toolkit.key_binding import KeyBindings

from modelo_Zypher_beta import generate_long_chat, load_model
# from modelo_mistral_base import generate_long_chat, load_model

#sumarización de conversaciones
"""
Summarize the text, leaving only the personal and objective data of the 'player' such as name, job, intentions, preferences. Ignore the data of the other interlocutor. Important, The structure will be like this:
The name of 'Player' is...
The job of 'Player' is...
The goal of 'Player' is...
'Player' likes...
etc 
"""

ai = "assistant"
user = "user"

contexto = """
You are Charles_Moreau, you already know your interlocutor (player).
The name of your interlocutor is Javier.
The job of your interlocutor is police inspector.
The goal of your interlocutor is unknown in this conversation.
Your interlocutor likes blue.
Your interlocutor does not like honey with almonds.
Important, If your interlocutor says something that contradicts all of this, let them know and demand clarification.
Important, You know your interlocutor from today. You just talked to him a moment ago, Greet him lightly.
"""

# conversacion = """
# "player> Hi!
# Charles_Moreau:Hi Javier! What's up?
# player: Very well!
# Charles_Moreau:Are you a police inspector?
# player: no. I am teacher.
# Charles_Moreau:You are lying to me, Javier is a police inspector!
# player: yes you right.
# Charles_Moreau:What's your favorite color, Javier?
# player: red.
# Charles_Moreau:Do you like honey with almonds, Javier?
# player: I love it.
# Charles_Moreau:That's great! I have a key for you. It's the key to my house."
# """

system_prompt = f"""
If in the conversation your interlocutor said something that contradicts his previous context, write "inconsistency:" and succinctly note the contradiction. If there is no contradiction, write "pass.".
Context:
{contexto}
"""

import sys
# si el comando se ejecuta con parámetro (texto entre comillas) entonces se usa ese texto como system_prompt
if len(sys.argv) > 1:
    system_prompt = sys.argv[1]

saludo = "Hello, I am ready to receive the conversation and check its consistency."


historico = f"<|system|>{system_prompt}</s>\n|assistant|\n{saludo}</s>\n"


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


def comprueba_consistencia_player(texto):
    global contexto
    global historico
    # print("texto:", texto)
    # print("contexto:", contexto)
    # print("historico:", historico)
    # print("##################")
    # print("##################")
    # print("##################")
    # print("##################")
    #


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
    historico, _ = generate_long_chat(historico, ai, user, input_text=input_text, max_additional_tokens=2048)
    # print response
    # print(salida)
    print(f"\n################################################\n")