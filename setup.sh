#!/usr/bin/env bash
# One-time setup: create venv, install deps, vendor frontend libs.
set -e
cd "$(dirname "$0")"

PY=python3
if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.12 .venv 2>/dev/null || uv venv .venv
  uv pip install --python .venv/bin/python -r requirements.txt
else
  $PY -m venv .venv
  .venv/bin/python -m pip install -U pip
  .venv/bin/python -m pip install -r requirements.txt
fi

mkdir -p static/vendor
dl(){ curl -fsSL "$1" -o "$2" && echo "  got $2" || echo "  FAILED $2 ($1)"; }
echo "Downloading frontend vendor files..."
dl https://cdn.jsdelivr.net/npm/uplot@1.6.30/dist/uPlot.iife.min.js static/vendor/uPlot.iife.min.js
dl https://cdn.jsdelivr.net/npm/uplot@1.6.30/dist/uPlot.min.css     static/vendor/uPlot.min.css
dl https://cdn.jsdelivr.net/npm/alpinejs@3.14.1/dist/cdn.min.js     static/vendor/alpine.min.js

echo "Setup done. Run:  ./run.sh   then open http://localhost:8800"
