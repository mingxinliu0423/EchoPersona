#!/usr/bin/env bash
set -euo pipefail
HOST="${1:-127.0.0.1}"
PORT="${2:-7860}"
set -x
curl -s "http://${HOST}:${PORT}/healthz" || true
