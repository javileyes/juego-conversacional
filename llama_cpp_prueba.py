from llama_cpp import Llama

enable_gpu = True  # offload LLM layers to the GPU (must fit in memory)

llm = Llama(
    model_path="./mistral-7b-instruct-v0.2.Q5_K_M.gguf",
    n_gpu_layers=32 if enable_gpu else 0,
    n_ctx=2048,
    verbose=False,
)


prompt = "¿Puedes escribirme una SPARQL query completa (usando Select, filter, service) para obtener el nombre de los países que tienen una población mayor a 100 millones de habitantes?"

prompt_template=f"<s>[INST] {prompt} [/INST]"

response=llm(prompt=prompt_template, max_tokens=1024, temperature=0.5, top_p=0.95,
                  repeat_penalty=1.2, top_k=150,
                  echo=True)

response