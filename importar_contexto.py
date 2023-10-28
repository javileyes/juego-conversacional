import csv
import glob
import os

def read_system_file(personaje):
    try:
        # Construir el nombre del archivo basándose en el parámetro personaje
        filename = f"./contexts/System_{personaje}.txt"
        
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

def read_context_file(personaje):
    try:
        # Construir el nombre del archivo basándose en el parámetro personaje
        filename = f"./contexts/{personaje}.txt"
        
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


def csv_to_text(filename):
    # Abrir el archivo CSV
    with open(filename, newline='', encoding='utf-8') as csvfile:
        # Crear un objeto reader de CSV con el delimitador específico
        csv_reader = csv.reader(csvfile, delimiter='$')
        
        # Unir las filas y columnas del CSV con '\n'
        text = '\n'.join(['\n'.join(row) for row in csv_reader])
        
        return text



def leer_ejemplos(personaje):
    # Inicializar una variable para almacenar el texto acumulado
    all_text = ""
    
    # Crear el patrón para buscar los archivos (p.ej., "personaje1.csv", "personaje2.csv", ...)
    pattern = f"./contexts/{personaje}*.csv"
    
    # Ordenar los archivos para procesarlos en orden numérico
    files = sorted(glob.glob(pattern), key=lambda x: int(x.split(personaje)[1].replace('.csv', '')))
    
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
    system = f"<|im_start|> system\n{system}\n<|im_end|>"
    context = read_context_file(personaje)
    context = "<|im_start|>{user}\n" + context + "\n<|im_end|>\n" + "<|im_start|>{ai}\n" + "ok, I will strictly follow this context" + "\n<|im_end|>"
    ejemplos = leer_ejemplos(personaje)
    ejemplos = "<|im_start|>{user}\n" + ejemplos + "<|im_end|>\n" + "<|im_start|>{ai}\n" + "ok, I understand what type of dialogue I can have." + "\n<|im_end|>"
    all_text = system + '\n' + context + '\n'# + ejemplos

    return all_text
    

# Llamada a la función y guardar el resultado en una variable
text_result = preparar_contexto('Charles_Moreau')

# Imprimir el texto resultante
print(text_result)