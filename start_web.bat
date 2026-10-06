@echo off
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
echo Open http://localhost:8000/ui/ in your browser
python -m uvicorn web_server:app --port 8000
