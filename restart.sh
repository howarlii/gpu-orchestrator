#!/usr/bin/env bash
# Restart (or start/stop/status) the orchestrator server.
#
# Running tasks are detached children (launched with setsid), so they SURVIVE a
# server restart and are re-adopted by PID when the server comes back up — it is
# safe to restart while tasks are running.
#
# Usage:
#   ./restart.sh            # stop (if running) then start  [default]
#   ./restart.sh start
#   ./restart.sh stop
#   ./restart.sh status
#
# Env overrides (match run.sh):  HOST=0.0.0.0  PORT=8800  LOG=server.out
set -u
cd "$(dirname "$0")"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8800}"
LOG="${LOG:-server.out}"
PYBIN=".venv/bin/python"; [[ -x "$PYBIN" ]] || PYBIN=python3

# Match THIS orchestrator's uvicorn process, scoped by port so we never kill an
# unrelated server / another project's instance on the same host.
PATTERN="uvicorn server:app .*--port ${PORT}\b"

_pids() { pgrep -f "$PATTERN" 2>/dev/null || true; }

stop() {
  local pids; pids="$(_pids)"
  if [[ -z "$pids" ]]; then
    echo "stop: no server running on port ${PORT}"
    return 0
  fi
  echo "stop: sending SIGTERM to pid(s): $pids"
  kill -TERM $pids 2>/dev/null || true
  for _ in $(seq 1 40); do          # wait up to ~12s for a clean exit
    [[ -z "$(_pids)" ]] && { echo "stop: stopped cleanly"; return 0; }
    sleep 0.3
  done
  echo "stop: still alive, sending SIGKILL"
  kill -KILL $(_pids) 2>/dev/null || true
  sleep 0.5
}

start() {
  if [[ -n "$(_pids)" ]]; then
    echo "start: already running on port ${PORT} (pid: $(_pids))"
    return 0
  fi
  echo "start: launching server on ${HOST}:${PORT} (log: ${LOG})"
  { echo; echo "===== restart $(date '+%Y-%m-%d %H:%M:%S') ====="; } >>"$LOG"
  # setsid + </dev/null detaches the server so it survives this shell/terminal.
  setsid "$PYBIN" -m uvicorn server:app --host "$HOST" --port "$PORT" \
    >>"$LOG" 2>&1 </dev/null &
  sleep 1.5
  local pid; pid="$(_pids)"
  if [[ -n "$pid" ]]; then
    echo "start: up (pid: $pid)  ->  http://localhost:${PORT}"
  else
    echo "start: FAILED — last log lines:"; tail -n 25 "$LOG"; exit 1
  fi
}

case "${1:-restart}" in
  stop)        stop ;;
  start)       start ;;
  status)      pgrep -af "$PATTERN" || echo "not running on port ${PORT}" ;;
  restart|"")  stop; start ;;
  *) echo "usage: $0 [restart|start|stop|status]"; exit 2 ;;
esac
