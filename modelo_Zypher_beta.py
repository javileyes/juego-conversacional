from ctransformers import AutoModelForCausalLM

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
model = AutoModelForCausalLM.from_pretrained("TheBloke/zephyr-7B-beta-GGUF", model_file="zephyr-7b-beta.Q8_0.gguf", model_type="mistral", gpu_layers=50, context_length=4000)

# print(model("AI is going to"))

def generate_chat(historico, ai, user, input_text, max_additional_tokens=64):
        
    prompt = f"{user}:{input_text}\n{ai}:"
  
    final_prompt = historico + prompt
 
    inputs = final_prompt

    # input_length = inputs["input_ids"].size(1)  # Obtén el número de tokens en la entrada
    # print("input_length:", input_length)
    # max_length = input_length + max_additional_tokens  # Calcula la longitud máxima de la secuencia de salida

    model_inputs = inputs
 
    outputs = ""
    # frases_cortas = True
    warning = False
    contador = 0
    print(f"{ai}:", end="")
    for text in model(model_inputs, stream=True):
        contador += 1
        if warning and text.lower()==user.lower(): 
            break
       
        if text in ".?!": warning = True        
        print(text, end="", flush=True)
        outputs += text
        if text=="\n" or contador > max_additional_tokens and text in ".?!":
            break

    print("")
    text = model_inputs + outputs

    return text