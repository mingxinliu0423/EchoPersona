#!/usr/bin/env python3
from llama_cpp import Llama

MODEL_PATH = "/home/lmx/EchoPersona/models/L3.1-MOE-13.7B/moe13b-q4ks.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=131072,
    n_gpu_layers=999,    
    n_batch=512,         
    n_threads=12,        
    verbose=False         
)

messages = [
    {"role": "system", "content": "You are TARS./no_think"}
]

while True:
    user_in = input("You:").strip()
    if user_in.lower() in {"exit", "quit", "bye"}:
        break
    messages.append({"role": "user", "content": user_in})

    answer = []
    in_think = False
    buf = []

    for chunk in llm.create_chat_completion(
        messages=messages,
        max_tokens=384,
        temperature=0.8,
        top_p=0.95,
        stream=True,
    ):
        delta = chunk["choices"][0]["delta"].get("content", "")
        if not delta:
            continue

        if "<think>" in delta:
            in_think = True
        if not in_think:
            print(delta, end="", flush=True)
            buf.append(delta)
        if "</think>" in delta:
            in_think = False

    print()
    messages.append({"role": "assistant", "content": "".join(buf)})

