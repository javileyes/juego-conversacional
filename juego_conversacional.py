from neo4j import GraphDatabase
import textwrap
import importar_contexto

from prompt_toolkit import prompt
# from prompt_toolkit.input.defaults import create_input
from prompt_toolkit.key_binding import KeyBindings
# from prompt_toolkit.filters import Condition

from ctransformers import AutoModelForCausalLM

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
model = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-OpenOrca-GGUF", model_file="mistral-7b-openorca.Q8_0.gguf", model_type="mistral", gpu_layers=50, context_length=4000)



model.PAD_TOKEN_ID=3200
# configure EOS TOKEN model to 13
# model.EOS_TOKEN_ID=13

iteracion = 0

ai = "peter"
user = "player"
ai = "Charles_Moreau"

######################################################################3

contexto = ""


system_prompt = """
You are a role player of a game that consists of answering questions about your context in the game. As the user asks the questions, You will try to faithfully answer the questions according to your context.
"""

# saludo = "Hi, do you like to play?"

inicio = "ok, let's start!\n"


# juego de interrogatorio (IA defensiva)
historico = f"{contexto}<|im_start|>{user}\nI want us to play a roleplay where I am a police officer who interrogate you about why you were at a crime scene. The scene is a pub where there were a few people at the bar: 1 couple and a group of 4 friends and you who were alone. A murderer entered and killed the couple with two shots, then ran away and got on his motorcycle that was parked at the door and fled. The police arrived 10 minutes later and that is where the interrogation begins. I am a police inspector and I want information about what happened and also to verify that you are not a suspect.You are Peter, a medical student who had gone down to the bar to rest after a long day of studying.\n{ai}\n{inicio}"

# crea una class Actor, tiene un atributo "contexto", otro string llamado "sytem_prompt", y otro string llamado "historico".
class Actor:
    def __init__(self, name, historico):
        self.name = name
        # historico son todos los contextos más las conversaciones
        self.historico = historico

actor = Actor(f"{ai}", importar_contexto.preparar_contexto(f"{ai}"))
actor.historico = actor.historico.format(ai=ai, user=user)
historico = actor.historico

def wrap_text(text, width=90): #preserve_newlines
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text


# def find_nth_occurrence(string, substring, n):
#     start = 0
#     for _ in range(n):
#         start = string.find(substring, start) + 1
#         if start == 0:
#             return -1
#     return start - 1

# Encuentra la última aparición de una subcadena en una cadena
def find_last_occurrence(string, substring):
    return string.rfind(substring)



def generate_chat(n, ai, user, input_text, system_prompt="",max_additional_tokens=64):
    global historico

    if system_prompt != "":
        system_prompt = f"""<|im_start|> system\n{system_prompt}<|im_end|>"""
    else:
        system_prompt = ""
        
    prompt = f"{user}:{input_text}\n{ai}:"
    final_prompt = system_prompt + historico + prompt

    # si actor.name es igual a f"{ai}" entonces final_prompt = actor.historico + prompt
    if actor.name == f"{ai}":
        final_prompt = historico + prompt
        # print("actor.historico:", actor.historico)

    # print("final_prompt:", final_prompt)
    # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    inputs = final_prompt

    # input_length = inputs["input_ids"].size(1)  # Obtén el número de tokens en la entrada
    # print("input_length:", input_length)
    # max_length = input_length + max_additional_tokens  # Calcula la longitud máxima de la secuencia de salida

    indice = find_last_occurrence(inputs, f"{user}:")
    if indice == -1:
        # print("no se encontro el indice")
        indice = len(inputs)

    # print("indice:", indice)
    # model_inputs = inputs.to(device)
    model_inputs = inputs
    # model.to(device)

    # outputs = model(model_inputs,
    #                          max_new_tokens=max_additional_tokens,
    #                          temperature=0.1
    #                          )
    outputs = ""
    # frases_cortas = True
    contador = 0
    print(f"{ai}:", end="")
    for text in model(model_inputs, stream=True):

        print(text, end="", flush=True)
        outputs += text
        if text=="\n" or contador > max_additional_tokens and text in ".?!":
            break

    print("")
    text = model_inputs + outputs


    # print("outputs:", outputs)

    historico_index = indice
    # print("historico_index:", historico_index)

    inicio_salida_index = text.find(f"{ai}:", indice)
 
    # fin_salida_index igual al siguiente salto de linea
    fin_salida_index = text.find('\n', inicio_salida_index)

    if fin_salida_index == -1:
        fin_salida_index = len(text)

    # print(f"historico_index:{historico_index}")
    # print(f"{inicio_salida_index},{fin_salida_index}")

    salida = text[inicio_salida_index:fin_salida_index]

    historico_add = text[historico_index:fin_salida_index] + '\n'
    # print("historico_add:", historico_add)
    historico += historico_add
    

    # print("nuevo texto:", text)
    wrapped_text = wrap_text(salida)
    return wrapped_text

##########################################################





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


def underscores_to_spaces(text):
    return text.replace('_', ' ')

def spaces_to_underscores(text):
    return text.replace(' ', '_')



class Game:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.location = 'Jardin'
        self.known_passageways = {}

    def close(self):
        self.driver.close()


    def get_description(self, location):
        with self.driver.session() as session:
            result = session.execute_read(self._get_description, location)
            return result

    def _get_description(self, tx, location):
        print("buscar en:", location)
        location = spaces_to_underscores(location)
        query = (
            "MATCH (h:Habitaculo {referencia: $location}) "
            "RETURN h.descripcion AS descripcion"
        )
        result = tx.run(query, location=location)
        record = result.single()
        if record is None:
            print(f"Error: No se pudo encontrar el habitáculo '{location}'. Asegúrate de que el nombre esté escrito correctamente y exista en la base de datos.")
            return ""
        return record["descripcion"]


    def get_passageways(self, location):
        with self.driver.session() as session:
            result = session.execute_read(self._get_passageways, location)
            return result

    def _get_passageways(self, tx, location):
        query = (
            "MATCH (h:Habitaculo {referencia: $location})-[r:CONECTA_A]-(dest:Habitaculo) "
            "MATCH (p:Pasarela) "
            "WHERE r.por = p.referencia "
            "RETURN p.nombre AS nombre, p.descripcion AS descripcion, p.referencia AS referencia, dest.referencia AS destino"
        )
        result = tx.run(query, location=location)
        return [(record["nombre"], record["descripcion"], record["referencia"], record["destino"]) for record in result]


    def get_habitaculos(self, location):
        with self.driver.session() as session:
            result = session.execute_read(self._get_habitaculos, location)
            return result

    def _get_habitaculos(self, tx, location):
        query = (
            "MATCH (h:Habitaculo {referencia: $location})-[r:CONECTA_A]-(dest:Habitaculo) "
            "RETURN dest.referencia AS destino"
        )
        result = tx.run(query, location=location)
        return [(record["destino"]) for record in result]



    def update_description(self, passageway, new_description):
        with self.driver.session() as session:
            session.write_transaction(self._update_description, passageway, new_description)

    def _update_description(self, tx, passageway, new_description):
        query = (
            "MATCH (p:Pasarela {nombre: $passageway}) "
            "SET p.referencia = $new_description"
        )
        tx.run(query, passageway=passageway, new_description=new_description)

    def play(self):
        global iteracion
        print("Wellcome to the Mistery of Morreay Family!.")
        self.show_location(self.location)
        mode = "ACTION"
        while True:
            command = input_prompt(mode)#input("> ").strip()
            if command.startswith("/"): #es un comando
                if mode == "DIALOGUE":
                    print("hasta luego")
                    mode = "ACTION"
                if command.startswith("/cross "):
                    destination = command.split("/cross ", 1)[1]
                    self.cross(destination)
                elif command == "/look":
                    self.show_location(self.location)
                elif command == "/historico":
                    print(historico)                    
                elif command.lower() in "/help":
                    print("Comandos disponibles:")
                    print("/cross <pasarela> - Cruza por una pasarela.")
                    print("/go <habitáculo> - Ir a un habitáculo.")
                    print("/look - Mira alrededor.")
                    print("/help - Muestra esta ayuda.")
                    print("/exit - Salir del juego.")
                elif command.startswith("/go "):
                    destination = command.split("/go ", 1)[1]
                    self.go_to(destination)
                elif command.lower() in "/exit":
                    print("thanks for play.")
                    break
                else:
                    print("Comando no reconocido.")
            else: #es un diálogo
                if mode == "ACTION":
                    print("dialogo")
                    mode = "DIALOGUE"
                iteracion += 1
                salida = generate_chat(iteracion, ai, user, input_text=command,
                                        system_prompt=system_prompt,
                                        max_additional_tokens=16)
                # print response
                # print(salida)

    def show_location(self, location):
        description = self.get_description(location)
        passageways = self.get_passageways(location)

        print_wrapped(f"Estás en {location}. {description}", 120)
        print("--------------------------------------------------------")
        print("Salidas:")
        for name, des, ref, _ in passageways:
            print(f"{underscores_to_spaces(ref)}: {des}")

    def cross(self, passageway_quick_desc):
        # Encuentra la conexión basada en la descripción rápida de la pasarela
        connection = self.find_connection_by_quick_desc(self.location, passageway_quick_desc)
        if connection:
            new_location = connection
            self.location = new_location
            self.show_location(new_location)
        else:
            print("No se puede cruzar por ahí. Intenta algo diferente.")

    def go_to(self, passageway_quick_desc):
        # Encuentra la conexión basada en el nombre del habitáculo destino
        connection = self.find_habitaculo_by_quick_desc(self.location, passageway_quick_desc)
        if connection:
            new_location = connection
            self.location = new_location
            self.show_location(new_location)
        else:
            print("No se puede ir ahí directamente. Intenta cruzar por una pasarela.")

    def find_connection_by_quick_desc(self, location, quick_desc):
        # quick_desc = spaces_to_underscores(quick_desc)
        passageways = self.get_passageways(location)
        for name, des, ref, dest in passageways:
            if spaces_to_underscores(quick_desc).lower() in ref.lower():
                return dest
        return None

    def find_habitaculo_by_quick_desc(self, location, quick_desc):
        hab = self.get_habitaculos(location)
        for ref in hab:
            if spaces_to_underscores(quick_desc).lower() in ref.lower():
                return ref
        return None


if __name__ == "__main__":
    game = Game("bolt://localhost:7687", "neo4j", "gitano666666")
    try:
        game.play()
    finally:
        game.close()