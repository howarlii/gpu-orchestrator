#!/usr/bin/env bash
# Launch the orchestrator. Override PORT/HOST via env.
cd "$(dirname "$0")"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8800}"
exec .venv/bin/python -m uvicorn server:app --host "$HOST" --port "$PORT" "$@"
