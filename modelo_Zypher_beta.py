from ctransformers import AutoModelForCausalLM, AutoTokenizer
from threading import Lock

model=None
eos_token_id=None

def load_model(user="user", ai="assistant"):
    global model
    global eos_token_id
    # shift first leter to upper case for user
    User = user[0].upper() + user[1:]
  
    # modelo NOTUS fintuneado de mistral con dataset de chats curados que supera a zephyr.
    # model = AutoModelForCausalLM.from_pretrained("TheBloke/notus-7B-v1-GGUF", model_file="notus-7b-v1.Q5_K_M.gguf", model_type="mistral", gpu_layers=50, context_length=8000)

    model = AutoModelForCausalLM.from_pretrained("TheBloke/zephyr-7B-beta-GGUF", model_file="zephyr-7b-beta.Q8_0.gguf", model_type="mistral", gpu_layers=50, context_length=8000)

    eos_token_id = model.eos_token_id
    print("eos_token_id:", eos_token_id)
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained("argilla/notus-7b-v1-lora")
    # print("token eos:", tokenizer.decode(eos_token_id))


def generate_chat(historico, ai, user, input_text, max_additional_tokens=64, stop=["</s>"]):
    global model

    # if stop is None:
    #     stop = [eos_token_id]
   

    User = user
    # prompt = f"{user}:{input_text}</s>\n{ai}:"
    prompt = f"{user}:{input_text}\n{ai}:"

  
    final_prompt = historico + "\n" + prompt
 

    # input_length = inputs["input_ids"].size(1)  # Obtén el número de tokens en la entrada
    # print("input_length:", input_length)
    # max_length = input_length + max_additional_tokens  # Calcula la longitud máxima de la secuencia de salida

    model_inputs = final_prompt
 
    outputs = ""
    # frases_cortas = True
    warning = False
    contador = 0
    print(f"{ai}:", end="")
    for text in model(model_inputs, stream=True, temperature=0.1, stop=stop):
        contador += 1
        # si warning y text contiene ":"

        # if warning and text.lower()==user.lower(): 
            # break
        outputs += text
        if text in ".?!": warning = True  
        if warning and (":" in text or "#" in text or "\n\n" in outputs):     
            # print("MONDONGO!!")
            print("\r\033[K", end="")            
            # elimina del texto la ultima linea
            outputs = outputs.rsplit('\n', 1)[0]
            # imprime retorno de linea para eliminar lo escrito
            # print("\r", end="")
            # outputs = outputs.rsplit('\n', 1)[0]
            break
        # if "{user}:" in outputs then delete from outputs and break
        # if f"{user}:" in outputs: 
        #     outputs = outputs.replace(f"{user}:", "")
        #     break
        # if f"{User}:" in outputs: 
        #     outputs = outputs.replace(f"{User}:", "")
        #     break
        
        print(text, end="", flush=True)
        
        if contador > max_additional_tokens and text in ".?!":
            break

    print("")
    # si output termina por "</s" se elimina:
    if outputs.endswith("</s"):
        outputs = outputs[:-3]
    text = model_inputs + outputs + "</s>"

    

    return text


def generate_long_chat(historico, ai, user, input_text, max_additional_tokens=2000, stop=["</s>","user:"], short_answer=False, streaming=True, printing=True):
    if short_answer:
        # añade como stop el salto de linea
        stop.append("\n")

    # if stop is None:
    #     stop = [eos_token_id]

    prompt = f"{user}:{input_text}</s>\n{ai}:"
  
    final_prompt = historico + "\n" + prompt
 
    # inputs = final_prompt

    # input_length = inputs["input_ids"].size(1)  # Obtén el número de tokens en la entrada
    # print("input_length:", input_length)
    # max_length = input_length + max_additional_tokens  # Calcula la longitud máxima de la secuencia de salida

    model_inputs = final_prompt
 
    outputs = ""
    # frases_cortas = True
    warning = False
    # contador = 0
    print(f"{ai}:", end="")
    for text in model(model_inputs, stream=streaming, max_new_tokens= max_additional_tokens, stop=stop):
        # contador += 1
        # if text.lower()==user.lower(): 
        #     break
       
        # if text in ".?!": warning = True
        if printing:    
            print(text, end="", flush=True)

        outputs += text
        # if text=="\n" or contador > max_additional_tokens and text in ".?!":
        #     break


    if outputs.endswith("</s"):
        outputs = outputs[:-3]

    all_text = model_inputs + outputs + "</s>"

    return all_text, outputs


import threading

class EstadoGeneracion:
    def __init__(self):
        self.parts = []
        self.generando = False
        self.lock = threading.Lock()


estado_generacion = EstadoGeneracion()

generate_lock = Lock()

def generate_in_file_parts(historico, ai, user, input_text, max_additional_tokens=2000, stop=["</s>","user:"], short_answer=False, streaming=True, printing=True):
    global estado_generacion

    # global generate_lock

    with generate_lock:
        estado_generacion.generando = True
        print(f"generando={estado_generacion.generando}; Generando respuesta para {user}:", input_text)
        contador = 0 # indice de la parte
        estado_generacion.parts = []  # lista de partes del texto
        parte_actual = ""  # añade la primera parte

        if short_answer:
            # añade como stop el salto de linea
            stop.append("\n")


        prompt = f"{user}:{input_text}</s>\n{ai}:"
    
        final_prompt = historico + "\n" + prompt
    

        model_inputs = final_prompt
    
        outputs = ""
        print(f"{ai}:", end="")
        for text in model(model_inputs, stream=streaming, max_new_tokens= max_additional_tokens, stop=stop):
            if printing:    
                print(text, end="", flush=True)

            outputs += text
            parte_actual += text
            if text in ",;:.?!" and len(parte_actual)>15:
                contador += 1
                estado_generacion.parts.append(parte_actual)
                parte_actual = ""
            

        if outputs.endswith("</s"):
            outputs = outputs[:-3]
            parte_actual = parte_actual[:-3]

        if parte_actual:
            estado_generacion.parts.append(parte_actual)

        all_text = model_inputs + outputs + "</s>"
        estado_generacion.generando = False
        print(f"generando={estado_generacion.generando}; Respuesta generada para {user}:", outputs)

        return all_text, outputs