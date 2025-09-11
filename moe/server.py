import os, threading
from typing import Generator, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import gradio as gr
from llama_cpp import Llama

# ---- env ----
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
MODEL_PATH      = os.getenv("MODEL_PATH", "").strip()
HOST            = os.getenv("HOST", "0.0.0.0")
PORT            = int(os.getenv("PORT", "7860"))
N_CTX           = int(os.getenv("N_CTX", "4096"))
N_GPU_LAYERS    = int(os.getenv("N_GPU_LAYERS", "-1"))
N_BATCH         = int(os.getenv("N_BATCH", "512"))
CONCURRENCY     = int(os.getenv("CONCURRENCY", "2"))
TEMPERATURE_DEF = float(os.getenv("TEMPERATURE", "0.8"))
TOP_P_DEF       = float(os.getenv("TOP_P", "0.9"))

assert MODEL_PATH and os.path.exists(MODEL_PATH), f"MODEL_PATH 不存在：{MODEL_PATH}"

# ---- load model ----
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=N_CTX,
    n_gpu_layers=N_GPU_LAYERS,
    n_batch=N_BATCH,
    verbose=False,
)

# ---- warmup ----
def _warmup():
    try:
        llm.create_completion("Hello", max_tokens=1, temperature=0.0, top_p=1.0)
    except Exception as e:
        print("[warmup] error:", e)
threading.Thread(target=_warmup, daemon=True).start()

# ---- infer helpers ----
def infer_stream(prompt: str, max_tokens: int, temperature: float, top_p: float) -> Generator[str, None, None]:
    for out in llm.create_completion(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=True,
    ):
        yield out["choices"][0]["text"]

def infer_once(prompt: str, max_tokens: int, temperature: float, top_p: float) -> str:
    out = llm.create_completion(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=False,
    )
    return "".join(choice["text"] for choice in out["choices"])

# ---- FastAPI ----
class CompletionReq(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: bool = True

api = FastAPI()

@api.get("/healthz")
def healthz():
    return {"ok": True, "model_path": MODEL_PATH, "n_ctx": N_CTX}

@api.post("/v1/completions")
def completions(req: CompletionReq):
    temperature = req.temperature if req.temperature is not None else TEMPERATURE_DEF
    top_p = req.top_p if req.top_p is not None else TOP_P_DEF
    if req.stream:
        return StreamingResponse(
            infer_stream(req.prompt, req.max_tokens, temperature, top_p),
            media_type="text/event-stream",
        )
    text = infer_once(req.prompt, req.max_tokens, temperature, top_p)
    return JSONResponse({"text": text})

# ---- Gradio ----
def gr_infer(prompt, max_tokens, temperature, top_p):
    buf = ""
    for tok in infer_stream(prompt, int(max_tokens), float(temperature), float(top_p)):
        buf += tok
        yield buf

with gr.Blocks(title="LLM Server") as demo:
    gr.Markdown("### LLM Streaming Demo (llama.cpp + FastAPI + Gradio)")
    inp  = gr.Textbox(label="Prompt", value="Say hello in 10 words.")
    mtk  = gr.Slider(1, 1024, value=256, step=1, label="max_tokens")
    temp = gr.Slider(0.0, 1.5, value=TEMPERATURE_DEF, step=0.05, label="temperature")
    topp = gr.Slider(0.1, 1.0, value=TOP_P_DEF, step=0.05, label="top_p")
    out  = gr.Textbox(label="Output")
    btn  = gr.Button("Generate")
    btn.click(fn=gr_infer, inputs=[inp, mtk, temp, topp], outputs=[out])

# ---- Queue 兼容 Gradio 3/4/5 ----
# 旧版：queue(concurrency_count=..., max_size=...)
# 新版(5.x)：queue(max_size=..., default_concurrency_limit=...)
try:
    demo = demo.queue(concurrency_count=CONCURRENCY, max_size=32)
except TypeError:
    try:
        demo = demo.queue(max_size=32, default_concurrency_limit=CONCURRENCY)
    except TypeError:
        demo = demo.queue(max_size=32)

# uvicorn 会在此变量上找 ASGI app
app = gr.mount_gradio_app(api, demo, path="/")
