# LOGGING / 排错笔记（示例）
- 运行日志：`journalctl -u llm-serve -f`（上到服务器后）
- Nginx 日志：`/var/log/nginx/access.log`, `/var/log/nginx/error.log`
- 常见问题：
- OOM：调小 N_BATCH / N_CTX / 并发；或换更小量化
- 首 token 慢：确认已预热、模型在 NVMe、进程常驻
