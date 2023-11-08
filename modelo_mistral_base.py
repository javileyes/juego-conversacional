from ctransformers import AutoModelForCausalLM

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
# model = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-OpenOrca-GGUF", model_file="mistral-7b-openorca.Q8_0.gguf", model_type="mistral", gpu_layers=50, context_length=4000)

# model.PAD_TOKEN_ID=3200

model=None
historico = ""

def load_model(user="user", ai="assistant"):
    global model
    # shift first leter to upper case for user
    User = user[0].upper() + user[1:]
    # Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
    model = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-OpenOrca-GGUF", model_file="mistral-7b-openorca.Q8_0.gguf", model_type="mistral", gpu_layers=50, context_length=4000)
    model.PAD_TOKEN_ID=3200



def wrap_text(text, width=90): #preserve_newlines
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text


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
    # if actor.name == f"{ai}":
    #     final_prompt = historico + prompt
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


    # Otra forma de hacerlo en streaming y parando cuando se encuentra un salto de linea o cuando se alcanza el maximo de tokens y se encuentra un punto, signo de interrogacion o exclamacion.
    outputs = ""
    # frases_cortas = True
    warning = False
    contador = 0
    print(f"{ai}:", end="")
    for text in model(model_inputs, stream=True):
        if warning and text==user: 
            break
       
        if text in ".?!": warning = True        
        print(text, end="", flush=True)
        outputs += text
        if text=="\n" or contador > max_additional_tokens and text in ".?!":
            break

    print("")
    text = model_inputs + outputs


    # print("outputs:", outputs)

    # historico_index = indice
    # print("historico_index:", historico_index)

    # inicio_salida_index = text.find(f"{ai}:", indice)
 
    # fin_salida_index igual al siguiente salto de linea
    # fin_salida_index = text.find('\n', inicio_salida_index)

    # if fin_salida_index == -1:
    #     fin_salida_index = len(text)

    # print(f"historico_index:{historico_index}")
    # print(f"{inicio_salida_index},{fin_salida_index}")

    # salida = text[inicio_salida_index:fin_salida_index]

    # historico_add = text[historico_index:fin_salida_index] + '\n'
    # print("historico_add:", historico_add)
    # historico += historico_add
    historico = text

    # print("nuevo texto:", text)
    # wrapped_text = wrap_text(salida)
    # wrapped_text = wrap_text(outputs)
    return historico




def generate_long_chat(n, ai, user, input_text, system_prompt="",max_additional_tokens=2000):
    global historico
    global model


    if system_prompt != "":
        system_prompt = f"""<|im_start|> system\n{system_prompt}<|im_end|>"""
    else:
        system_prompt = ""
        
    prompt = f"{user}:{input_text}\n{ai}:"
    final_prompt = system_prompt + historico + prompt

    # si actor.name es igual a f"{ai}" entonces final_prompt = actor.historico + prompt
    # if actor.name == f"{ai}":
    #     final_prompt = historico + prompt
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


    # Otra forma de hacerlo en streaming y parando cuando se encuentra un salto de linea o cuando se alcanza el maximo de tokens y se encuentra un punto, signo de interrogacion o exclamacion.
    outputs = ""
    # frases_cortas = True
    warning = False
    contador = 0
    print(f"{ai}:", end="")
    for text in model(model_inputs, stream=True, max_new_tokens= max_additional_tokens):
        # if warning and text==user: 
        #     break
       
        # if text in ".?!": warning = True        
        print(text, end="", flush=True)
        outputs += text
        # if text=="\n" or contador > max_additional_tokens and text in ".?!":
        #     break

    print("")
    text = model_inputs + outputs


    # print("outputs:", outputs)

    # historico_index = indice
    # print("historico_index:", historico_index)

    # inicio_salida_index = text.find(f"{ai}:", indice)
 
    # fin_salida_index igual al siguiente salto de linea
    # fin_salida_index = text.find('\n', inicio_salida_index)

    # if fin_salida_index == -1:
    #     fin_salida_index = len(text)

    # print(f"historico_index:{historico_index}")
    # print(f"{inicio_salida_index},{fin_salida_index}")

    # salida = text[inicio_salida_index:fin_salida_index]

    # historico_add = text[historico_index:fin_salida_index] + '\n'
    # print("historico_add:", historico_add)
    # historico += historico_add
    historico = text

    # print("nuevo texto:", text)
    # wrapped_text = wrap_text(salida)
    # wrapped_text = wrap_text(outputs)
    return historico