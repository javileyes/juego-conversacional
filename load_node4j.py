import socket
import time
import os
import subprocess


def check_neo_connection():

    path = os.getenv('PATH')
    # check if ./neo4j/bin is inside path
    if './neo4j/bin' in path:
        print('Neo4j is in Path')
    else:
        # add neo4j to path
        new_path = './neo4j/bin:' + path
        # export PATH
        os.environ['PATH'] = new_path


    process = subprocess.run(["pm2", "list"], capture_output=True, text=True)

    if "neo4j" in process.stdout:
        print("El demonio neo4j está activo.")
    else:
        print("Levantando Neo4j...")


        # execute in python: "pm2 start "neo4j console"  --output log.txt --error error.txt"
        # Define el comando y los argumentos en una lista
        # command = ["pm2", "start", "neo4j console", "--output", "log.txt", "--error", "error.txt"]
        # command = ["pm2", "start", "./neo4j/bin/neo4j", "console", "--output", "log.txt", "--error", "error.txt"]
        command = "pm2 start 'neo4j console' --output log.txt --error error.txt"

        # Ejecuta el comando
        process = subprocess.run(command, text=True, shell=True)
        # mostrar pid del proceso process
        # print("PID:", process.pid)
        # Imprime la salida estándar y la salida de error estándar
        # print("STDOUT:", process.stdout)
        # print("STDERR:", process.stderr)
        # !pm2 start "neo4j console"  --output log.txt --error error.txt


        process = subprocess.run(["pm2", "list"], capture_output=True, text=True)

        if "neo4j" in process.stdout:
            print("El proceso Neo4j está activo.")
        else:
            print("El proceso no está activo.")


    print ("checking Neo4j connection...")
    # Crea un socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    while True:
        try:
            # Intenta conectarse al servicio Neo4j en localhost en el puerto 7474.
            result = sock.connect_ex(('localhost', 7474))
            
            # Si el resultado es 0, significa que la conexión fue exitosa.
            if result == 0:
                print("Neo4j is ready to use")
                break
            else:
                # Espera 10 segundos antes de intentar nuevamente si la conexión no fue exitosa.
                time.sleep(10)
        except socket.error as e:
            # Imprime la excepción (opcional, puedes comentar esta línea si no la necesitas).
            print(e)
            # Espera 10 segundos antes de intentar nuevamente después de una excepción.
            time.sleep(3)
        finally:
            # Cierra el socket.
            sock.close()
            # Crea un nuevo socket para el próximo intento.
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        