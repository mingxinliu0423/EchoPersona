# EchoPersona - Dual Model Chat System

A dual-model chat system featuring both PyTorch and GGUF model serving for comparative AI responses.

## Models

### 1. DeepSeek-V2-Lite-Chat (Development)
- **Format**: SafeTensors (PyTorch)
- **Location**: `models/DeepSeek-V2-Lite-Chat/`
- **Purpose**: Full-precision development and experimentation
- **Interface**: Jupyter notebook with Transformers

### 2. L3.1-MOE-13.7B (Production)
- **Format**: GGUF (Quantized)
- **Location**: `models/L3.1-MOE-13.7B/moe13b-q4ks.gguf`
- **Purpose**: Efficient production serving
- **Interface**: FastAPI + Gradio web interface

## Quick Start

### 1. Start the MOE Server
```bash
cd moe/
python server.py
```
Access web interface at: http://localhost:7860

### 2. Use Jupyter Notebook
```bash
jupyter notebook dual_model_chat.ipynb
```

## Features

- **Dual Model Comparison**: Side-by-side responses from both models
- **Sample Data Testing**: 1000 Alpaca samples for evaluation
- **Interactive Chat**: Compare model responses in real-time
- **Web Interface**: Gradio-based chat for the GGUF model
- **API Access**: RESTful API for programmatic access

## File Structure

```
EchoPersona/
├── dual_model_chat.ipynb      # Main notebook for both models
├── alpaca_sample.jsonl        # Sample data (1000 entries)
├── models/                    # Model files
│   ├── DeepSeek-V2-Lite-Chat/ # PyTorch model (~29GB)
│   └── L3.1-MOE-13.7B/        # GGUF model (~8GB)
├── moe/                       # Production server
│   ├── server.py             # FastAPI + Gradio server
│   ├── .env                  # Configuration
│   └── environment.yml       # Conda environment
└── LLaMA-Factory/            # Training framework
```

## API Endpoints

- `GET /healthz` - Server health check
- `POST /v1/completions` - Text completion
- `GET /` - Gradio web interface

## Configuration

Edit `moe/.env` to configure:
- `MODEL_PATH`: Path to GGUF model
- `N_GPU_LAYERS`: GPU acceleration (-1 for all)
- `N_CTX`: Context window size
- `CONCURRENCY`: Max concurrent requests

## Memory Usage

- **DeepSeek (Full)**: ~29GB VRAM
- **L3.1-MOE (GGUF)**: ~8GB VRAM
- **Both models**: Requires 40GB+ VRAM

## Performance Comparison

| Model | Format | Size | Speed | Quality | Use Case |
|-------|--------|------|-------|---------|----------|
| DeepSeek | PyTorch | 29GB | Slower | High | Development |
| L3.1-MOE | GGUF | 8GB | Faster | Good | Production |

## Dependencies

See `moe/environment.yml` for conda environment setup.

Key packages:
- `transformers` (DeepSeek model)
- `llama-cpp-python` (GGUF model)
- `fastapi` (API server)
- `gradio` (Web interface)
- `torch` (PyTorch backend)