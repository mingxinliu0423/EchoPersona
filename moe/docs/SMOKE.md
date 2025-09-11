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
