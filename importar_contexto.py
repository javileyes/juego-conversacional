import csv
import glob
import os

def read_player_file(personaje):
    try:
        # Construir el nombre del archivo basándose en el parámetro personaje
        filename = f"./contexts/{personaje}/{personaje}_player.txt"
        
        # Abrir el archivo y leer su contenido
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
            
        # Devolver el contenido del archivo
        return content
    
    except FileNotFoundError:
        return "El archivo no se encontró."
    
    except Exception as e:
        return f"Ocurrió un error al leer el archivo: {str(e)}"
    

def read_system_file(personaje):
    try:
        # Construir el nombre del archivo basándose en el parámetro personaje
        filename = f"./contexts/{personaje}/System_{personaje}.txt"
        
        # Abrir el archivo y leer su contenido
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
            
        # Devolver el contenido del archivo
        return content
    
    except FileNotFoundError:
        return "El archivo no se encontró."
    
    except Exception as e:
        return f"Ocurrió un error al leer el archivo: {str(e)}"

# # Llamada a la función con un ejemplo de nombre de personaje
# text = read_system_file('Charles_Moreau')
# print(text)

def read_system_file_2_character(personaje1, personaje2):
    try:
        # Construir el nombre del archivo basándose en el parámetro personaje
        filename = f"./contexts/{personaje1}__{personaje2}/System_{personaje1}__{personaje2}.txt"
        
        # Abrir el archivo y leer su contenido
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
            
        # Devolver el contenido del archivo
        return content
    
    except FileNotFoundError:
        return "El archivo no se encontró."
    
    except Exception as e:
        return f"Ocurrió un error al leer el archivo: {str(e)}"
    
    

def read_context_file(personaje):
    try:
        # Construir el nombre del archivo basándose en el parámetro personaje
        filename = f"./contexts/{personaje}/{personaje}.txt"
        
        # Abrir el archivo y leer su contenido
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
            
        # Devolver el contenido del archivo
        return content
    
    except FileNotFoundError:
        return "El archivo no se encontró."
    
    except Exception as e:
        return f"Ocurrió un error al leer el archivo: {str(e)}"

# # Llamada a la función con un ejemplo de nombre de personaje
# text = read_system_file('Charles_Moreau')
# print(text)

def read_character_file(personaje):
    try:
        # Construir el nombre del archivo basándose en el parámetro personaje
        filename = f"./contexts/{personaje}/Character_{personaje}.txt"
        
        # Abrir el archivo y leer su contenido
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
            
        # Devolver el contenido del archivo
        return content
    
    except FileNotFoundError:
        return "El archivo no se encontró."
    
    except Exception as e:
        return f"Ocurrió un error al leer el archivo: {str(e)}"
    

def csv_to_text(filename):
    # Abrir el archivo CSV
    with open(filename, newline='', encoding='utf-8') as csvfile:
        # Crear un objeto reader de CSV con el delimitador específico
        csv_reader = csv.reader(csvfile, delimiter='$')
        
        # Unir las filas y columnas del CSV con '\n'
        text = '</s>\n'.join(['</s>\n'.join(row) for row in csv_reader])
        
        return text


def csv_forbidden_to_text(filename):
    # Abrir el archivo CSV
    with open(filename, newline='', encoding='utf-8') as csvfile:
        # Crear un objeto reader de CSV con el delimitador específico
        csv_reader = csv.reader(csvfile, delimiter='\n')
        
        text = ""
        # Unir las filas y columnas del CSV con '\n'
        for row in csv_reader:
            linea = "".join(row)
            text += "{ai}:" + f"{linea}" + "\n"
        
        return text



def leer_ejemplos(personaje):
    # Inicializar una variable para almacenar el texto acumulado
    all_text = ""
    
    # Crear el patrón para buscar los archivos (p.ej., "personaje1.csv", "personaje2.csv", ...)
    pattern = f"./contexts/{personaje}/{personaje}*.csv"
    
    # Ordenar los archivos para procesarlos en orden numérico
    files = sorted(glob.glob(pattern), key=lambda x: int(x.split(personaje)[2].replace('.csv', '')))
    
    # Recorrer todos los archivos que coinciden con el patrón
    for i, filename in enumerate(files):
        # Verificar si el archivo es un archivo CSV válido
        if os.path.isfile(filename) and filename.endswith('.csv'):
            # Llamar a la función csv_to_text y acumular el texto
            all_text += f'#example of conversation {i+1}:\n' + csv_to_text(filename) + f"#end of example {i+1}\n"
            
    return all_text   


# # Llamada a la función y guardar el resultado en una variable
# text_result = leer_ejemplos('Charles_Moreau')

# # Imprimir el texto resultante
# print(text_result)


def preparar_contexto(personaje):

    system = read_system_file(personaje)
    system = f"<|im_start|>system\n{system}\n<|im_end|>"
    context = read_context_file(personaje)
    context = "<|im_start|>{user}\n" + context + "\n<|im_end|>\n" + "<|im_start|>{ai}\n" + "ok, I will strictly follow this context" + "\n<|im_end|>"
    ejemplos = leer_ejemplos(personaje)
    ejemplos = "<|im_start|>{user}\nI am going to list some examples but it is VERY IMPORTANT that you do not repeat them but rather create similar examples:\n" + ejemplos + "<|im_end|>\n" + "<|im_start|>{ai}\n" + "ok, I understand what type of dialogue I can have and under no circumstances will I repeat any of the examples." + "\n<|im_end|>"
    all_text = system + '\n' + context + '\n'# + ejemplos

    return all_text


def preparar_contexto_zhyper(personaje):

    #cosas igual al contendio del fichero ./contexts/player/player.txt
    player = read_player_file('player')

    player_knoledge = read_player_file(personaje) #conocimiento que el caracter tiene del player
    player_knoledge = player_knoledge.format(player=player)
    character_name = personaje
    character = read_character_file(personaje)
    system = read_system_file(personaje)
    system = f"<|system|>\n{system.format(character=character, player=player)}</s>"
    context = read_context_file(personaje)
    context = "<|user|>\n" + context + "Have you understood who you are?" + "</s>\n<|assistant|>\n" + "ok, I will strictly follow this context" + "</s>"
    ejemplos = leer_ejemplos(personaje)
    ejemplos = "<|user|>\nI am going to list some examples do you use them and it is very important that you can create similar examples:\n" + ejemplos + "</s>\n" + "<|assistant|>\n" + "ok, I understand what type of dialogue I can have." + "</s>\n"
    # prohibido = csv_forbidden_to_text(f"./contexts/Forbidden_{personaje}.csv")
    # prohibido = "<|user|>\nI am going to list some examples that you should not under any circumstances say:\n" + prohibido + "</s>\n" + "<|assistant|>\n" + "ok, I understand the kind of things I shouldn't say." + "</s>\n"
    # all_text = system + '\n' + context + '\n' + ejemplos + '\n' + prohibido
    all_text = system + '\n' + context + '\n' + ejemplos

    return all_text
    

def preparar_contexto_zhyper_2_character(personaje1, personaje2):
    character_name1 = personaje1
    character_name2 = personaje2
    character1 = read_character_file(personaje1)
    character2 = read_character_file(personaje2)
    system1 = read_system_file_2_character(personaje1, personaje2)


# Llamada a la función y guardar el resultado en una variable
# text_result = preparar_contexto('Charles_Moreau')

# Imprimir el texto resultante
# print(text_result)