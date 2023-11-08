from neo4j import GraphDatabase
import textwrap
import importar_contexto
import load_node4j

from prompt_toolkit import prompt
# from prompt_toolkit.input.defaults import create_input
from prompt_toolkit.key_binding import KeyBindings


load_node4j.check_neo_connection()

import modelo_Zypher_beta as zypher

# iteracion = 0

user = "player"
ai = "Charles_Moreau"

zypher.load_model(user)

######################################################################3

contexto = ""


system_prompt = """
You are a role player of a game that consists of answering questions about your context in the game. As the user asks the questions, You will try to faithfully answer the questions according to your context.
"""

# saludo = "Hi, do you like to play?"

inicio = "ok, let's start!\n"



# crea una class Actor, tiene un atributo "contexto", otro string llamado "sytem_prompt", y otro string llamado "historico".
class Actor:
    def __init__(self, name, historico):
        self.name = name
        # historico son todos los contextos más las conversaciones
        self.historico = historico

actor = Actor(f"{ai}", importar_contexto.preparar_contexto_zhyper(f"{ai}"))
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
        # global iteracion
        global historico
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
                # iteracion += 1
                # salida = generate_chat(iteracion, ai, user, input_text=command,
                #                         system_prompt=system_prompt,
                #                         max_additional_tokens=64)
                historico = zypher.generate_chat(historico, ai, user, input_text=command,                                    
                                        max_additional_tokens=16, stop=["</s>"])                
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