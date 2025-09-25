import os, threading, sys
from typing import Generator, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import gradio as gr

try:
    from llama_cpp import Llama
except ImportError:
    print("ERROR: llama-cpp-python not installed. Run: pip install llama-cpp-python")
    sys.exit(1)

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

if not MODEL_PATH or not os.path.exists(MODEL_PATH):
    print(f"ERROR: MODEL_PATH not found: {MODEL_PATH}")
    print("Check your .env file and ensure the GGUF model exists")
    sys.exit(1)

print(f"Loading model: {MODEL_PATH}")
print(f"Context: {N_CTX}, GPU layers: {N_GPU_LAYERS}")

try:
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        n_gpu_layers=N_GPU_LAYERS,
        n_batch=N_BATCH,
        verbose=False,
    )
    print("Model loaded successfully")
except Exception as e:
    print(f"Failed to load model: {e}")
    sys.exit(1)

def _warmup():
    try:
        llm.create_completion("Hello", max_tokens=1, temperature=0.0, top_p=1.0)
        print("Model warmed up")
    except Exception as e:
        print(f"Warmup failed: {e}")

threading.Thread(target=_warmup, daemon=True).start()
def infer_stream(prompt: str, max_tokens: int, temperature: float, top_p: float) -> Generator[str, None, None]:
    try:
        for out in llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
        ):
            yield out["choices"][0]["text"]
    except Exception as e:
        yield f"Error: {e}"

def infer_once(prompt: str, max_tokens: int, temperature: float, top_p: float) -> str:
    try:
        out = llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=False,
        )
        return "".join(choice["text"] for choice in out["choices"])
    except Exception as e:
        return f"Error: {e}"
class CompletionReq(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: bool = True

api = FastAPI(title="LLM API Server", version="1.0.0")

@api.get("/healthz")
def healthz():
    return {
        "ok": True, 
        "model_path": MODEL_PATH, 
        "n_ctx": N_CTX,
        "status": "ready"
    }

@api.post("/v1/completions")
def completions(req: CompletionReq):
    try:
        temperature = req.temperature if req.temperature is not None else TEMPERATURE_DEF
        top_p = req.top_p if req.top_p is not None else TOP_P_DEF
        
        if req.stream:
            return StreamingResponse(
                infer_stream(req.prompt, req.max_tokens, temperature, top_p),
                media_type="text/event-stream",
            )
        
        text = infer_once(req.prompt, req.max_tokens, temperature, top_p)
        return JSONResponse({"text": text})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
def gr_infer(prompt, max_tokens, temperature, top_p):
    if not prompt.strip():
        yield "Please enter a prompt!"
        return
        
    buf = ""
    try:
        for tok in infer_stream(prompt, int(max_tokens), float(temperature), float(top_p)):
            buf += tok
            yield buf
    except Exception as e:
        yield f"Error: {e}"

with gr.Blocks(title="LLM Server", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 L3.1-MOE-13.7B Server")
    gr.Markdown("### Production LLM Server with llama.cpp + FastAPI + Gradio")
    
    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(
                label="Prompt", 
                value="Explain artificial intelligence in simple terms.",
                lines=3,
                placeholder="Enter your prompt here..."
            )
            with gr.Row():
                mtk = gr.Slider(1, 1024, value=256, step=1, label="Max Tokens")
                temp = gr.Slider(0.0, 1.5, value=TEMPERATURE_DEF, step=0.05, label="Temperature")
                topp = gr.Slider(0.1, 1.0, value=TOP_P_DEF, step=0.05, label="Top P")
            btn = gr.Button("🚀 Generate", variant="primary", size="lg")
        
        with gr.Column():
            out = gr.Textbox(label="Response", lines=12, max_lines=20, show_copy_button=True)
    
    btn.click(fn=gr_infer, inputs=[inp, mtk, temp, topp], outputs=[out], concurrency_limit=CONCURRENCY)

try:
    demo = demo.queue(max_size=32)
except Exception as e:
    print(f"Queue setup warning: {e}")
    demo = demo

if __name__ == "__main__":
    import uvicorn
    import threading
    
    # Start FastAPI server in background thread
    def start_api():
        print(f"📡 Starting FastAPI on {HOST}:{PORT+1}")
        print(f"� API: http://localhost:{PORT+1}/v1/completions")
        print(f"❤️ Health: http://localhost:{PORT+1}/healthz")
        uvicorn.run(api, host=HOST, port=PORT+1, log_level="warning")
    
    api_thread = threading.Thread(target=start_api, daemon=True)
    api_thread.start()
    
    # Start Gradio server on main thread
    print(f"🚀 Starting Gradio UI on {HOST}:{PORT}")
    print(f"🌐 Web UI: http://localhost:{PORT}")
    
    demo.launch(
        server_name=HOST,
        server_port=PORT,
        share=False,
        show_error=True,
        quiet=False
    )
