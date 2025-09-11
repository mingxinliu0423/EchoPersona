#!/usr/bin/env bash
set -euo pipefail

# Usage: bash init_moe_layout.sh <MODEL_PATH>
MODEL_PATH_ARG="${1:-models/L3.1-MOE-13.7B/moe13b-q4ks.gguf}"

# project root = current dir
ROOT="$(pwd)"
MOE_DIR="$ROOT/moe"
DOCS_DIR="$MOE_DIR/docs"

mkdir -p "$MOE_DIR" "$DOCS_DIR"

# 1) environment.yml (conda)
cat > "$MOE_DIR/environment.yml" <<'YML'
name: llmserve
channels: [defaults, conda-forge, nvidia, pytorch]
dependencies:
  - python=3.10
  - pip
  - pip:
      - "llama-cpp-python[cuda]"
      - fastapi
      - uvicorn
      - gradio
      - pydantic
      - python-dotenv
      - huggingface_hub
      - hf-transfer
YML

# 2) .env (runtime config)
cat > "$MOE_DIR/.env" <<ENV
# ---- Runtime config (edit as needed) ----
MODEL_PATH=$MODEL_PATH_ARG
HOST=0.0.0.0
PORT=7860
N_CTX=4096
N_GPU_LAYERS=-1
N_BATCH=512
CONCURRENCY=2
TEMPERATURE=0.8
TOP_P=0.9
# Hugging Face cache (optional)
HF_HOME=\$HOME/.cache/huggingface
HF_HUB_ENABLE_HF_TRANSFER=1
ENV

# 3) server.py (占位模板：后续再填实现)
cat > "$MOE_DIR/server.py" <<'PY'
"""
占位文件：服务入口（Gradio + FastAPI）。
你后面说“可以写代码了”时，再把实现粘过来。
"""
if __name__ == "__main__":
    print("server.py placeholder. Fill implementation later.")
PY

# 4) systemd 单元（本地模板）
cat > "$MOE_DIR/llm-serve.service" <<'UNIT'
# 拿到服务器后放到 /etc/systemd/system/ 再启用
[Unit]
Description=LLM Inference Server
After=network.target

[Service]
Type=simple
User=%i
WorkingDirectory=/srv/llm
EnvironmentFile=/srv/llm/.env
ExecStart=/bin/bash -lc 'conda activate llmserve && uvicorn server:app --host ${HOST} --port ${PORT}'
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
UNIT

# 5) Nginx 反代（可选）
cat > "$MOE_DIR/nginx.conf" <<'NG'
# 放到 /etc/nginx/conf.d/ 并改域名/证书后 reload
server {
  listen 80;
  server_name _;

  location / {
    proxy_pass http://127.0.0.1:7860;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_read_timeout 3600s;
    proxy_send_timeout 3600s;
  }
}
NG

# 6) 运行/验证脚本
cat > "$MOE_DIR/start.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
source ~/.bashrc || true
conda activate llmserve
export $(grep -v '^#' .env | xargs -d '\n') || true
python -c "import sys; print('Using .env MODEL_PATH=', '${MODEL_PATH}')" || true
uvicorn server:app --host "${HOST:-0.0.0.0}" --port "${PORT:-7860}"
SH
chmod +x "$MOE_DIR/start.sh"

cat > "$MOE_DIR/healthcheck.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
HOST="${1:-127.0.0.1}"
PORT="${2:-7860}"
set -x
curl -s "http://${HOST}:${PORT}/healthz" || true
SH
chmod +x "$MOE_DIR/healthcheck.sh"

# 7) 文档模板
cat > "$DOCS_DIR/MODEL_MANIFEST.md" <<MAN
# MODEL_MANIFEST

- Model file: \`$MODEL_PATH_ARG\`
- Format: GGUF
- Quantization: q4_k_s (示例)
- Source repo: (填写)
- Commit / Date: (填写)
- Target GPU / VRAM: (填写)
- Notes: (如需 tokenizer 兼容性/rope 设置等)
MAN

cat > "$DOCS_DIR/SMOKE.md" <<'MD'
# SMOKE / 验收步骤（示例）
1. conda env: `conda env create -f environment.yml && conda activate llmserve`
2. 填好 `.env` 的 `MODEL_PATH`
3. 启动：`bash start.sh`
4. 健康检查：`bash healthcheck.sh <ip> 7860`
5. REST 调用（示例）：
curl -s -X POST http://<ip>:7860/v1/completions
-H "Content-Type: application/json"
-d '{"prompt":"hello","max_tokens":16,"stream":false}'
6. 浏览器打开 `http://<ip>:7860/` 看 Gradio（实现后可用）
MD

cat > "$DOCS_DIR/LOGGING.md" <<'MD'
# LOGGING / 排错笔记（示例）
- 运行日志：`journalctl -u llm-serve -f`（上到服务器后）
- Nginx 日志：`/var/log/nginx/access.log`, `/var/log/nginx/error.log`
- 常见问题：
- OOM：调小 N_BATCH / N_CTX / 并发；或换更小量化
- 首 token 慢：确认已预热、模型在 NVMe、进程常驻
MD

echo "✅ Created templates under: $MOE_DIR"
echo "👉 Next:"
echo "   1) 打开 $MOE_DIR/.env 把 MODEL_PATH 改成你的实际 GGUF 路径"
echo "   2) conda env:  conda env create -f $MOE_DIR/environment.yml && conda activate llmserve"
echo "   3) 等你说“可以写代码了”，我再把 server.py 的实现填上"