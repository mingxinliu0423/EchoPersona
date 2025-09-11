#!/usr/bin/env bash
set -euo pipefail

# 让脚本里能用 conda activate（不依赖是否 init 过）
if command -v conda >/dev/null 2>&1; then
  # shell hook：不同机器路径可能不同，用 conda info --base 获取更稳
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1090
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
  echo "conda not found in PATH"; exit 1
fi

conda activate llmserve

# 加载 .env（支持空行/注释；避免 xargs 对空格/特殊字符问题）
set -a
# shellcheck disable=SC1091
source ./.env
set +a

python -c "print('Using .env MODEL_PATH =', '${MODEL_PATH}')" || true
exec uvicorn server:app --host "${HOST:-0.0.0.0}" --port "${PORT:-7860}"
