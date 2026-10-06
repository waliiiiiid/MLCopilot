"""Web gateway for the ML Copilot web app. Nothing in backend/ is modified.

Run from the project root:   python -m uvicorn web_server:app --port 8000
Then open:                   http://localhost:8000/ui/

It imports your FastAPI `app` (/, /upload_dataset, /chat, /ml-pipeline stay exactly
as they are) and adds what Streamlit used to do in-process: running step_1 / step_2,
listing generated images, and serving the UI and the outputs folder.
"""
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)  # backend uses cwd-relative dataset/ and outputs/

from fastapi import HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.main import app
from backend.agents.workflows.first_graph import step_1
from backend.agents.workflows.second_graph import step_2

IMAGES = ('.png', '.jpg', '.jpeg', '.webp')


class RunRequest(BaseModel):
    filename: str


def _run(graph, filename: str, tag: str):
    path = os.path.join('dataset', os.path.basename(filename))
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail='Dataset not found. Upload it first.')
    try:
        r = graph.invoke({
            'code': '', 'execution_return_code': '', 'execution_stderr': '',
            'execution_stdout': '', 'execution_success': '', 'file_path': path,
            'plan': '', 'profile': '',
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    os.makedirs('outputs', exist_ok=True)
    Path('outputs', f'{tag}_code.py').write_text(r['code'], encoding='utf-8')
    Path('outputs', f'{tag}_plan.txt').write_text(r['plan'], encoding='utf-8')
    return {
        'code': r['code'], 'plan': r['plan'], 'profile': r.get('profile', ''),
        'ok': r.get('execution_success') is True,
        'stderr': r.get('execution_stderr', ''),
    }


@app.post('/run_visuals')
def run_visuals(req: RunRequest):
    return _run(step_1, req.filename, 'visual')


@app.post('/run_cleaning')
def run_cleaning(req: RunRequest):
    return _run(step_2, req.filename, 'cleaning')


@app.get('/images')
def images(sub: str = ''):
    folder = os.path.join('outputs', 'evaluation') if sub == 'evaluation' else 'outputs'
    if not os.path.isdir(folder):
        return {'images': []}
    return {'images': sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(IMAGES) and os.path.isfile(os.path.join(folder, f)))}


os.makedirs('outputs', exist_ok=True)
app.mount('/outputs', StaticFiles(directory='outputs', check_dir=False), name='outputs')
app.mount('/ui', StaticFiles(directory=os.path.join('frontend', 'web'), html=True), name='ui')
