#!/usr/bin/env bash
# Start the API. Override the port with PORT=... (default 8000).
set -e
cd "$(dirname "$0")/.."

PORT="${PORT:-8000}"
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
