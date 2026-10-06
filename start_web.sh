#!/usr/bin/env bash
cd "$(dirname "$0")"
[ -f .venv/bin/activate ] && . .venv/bin/activate
echo "Open http://localhost:8000/ui/ in your browser"
python -m uvicorn web_server:app --port 8000
