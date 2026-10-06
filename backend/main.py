import os
import shutil
from backend.agents.workflows.first_graph import step_1
from backend.agents.workflows.second_graph import step_2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.agents.chat import chatbot
app=FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])
from backend.agents.workflows.ml import pipeline

@app.get('/')
async def home():

    folder = "outputs"

    if os.path.exists(folder):
        shutil.rmtree(folder)

    os.makedirs(folder)

    if os.path.exists('dataset'):
        shutil.rmtree('dataset')

    os.makedirs('dataset')

    if os.path.exists('models'):
        shutil.rmtree('models')
    os.makedirs('models')

  
    return{
        'message':'project folders initialized'
    }


@app.post("/upload_dataset")
async def upload_csv(file: UploadFile = File(...)):

    # Validate extension
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only .CSV files are allowed"
        )

    # Make sure the directory exists
    os.makedirs("dataset", exist_ok=True)

    # Create file path
    file_path = os.path.join("dataset", file.filename)

    # Read the uploaded file
    contents = await file.read()

    # Save it
    with open(file_path, "wb") as f:
        f.write(contents)

    return {
        "message": "CSV Uploaded",
        "filename": file.filename
    }

class ChatRequest(BaseModel):
    prompt:str
    raw_profile: str
    cleaning_plan: str
    visual_plan: str
    cleaning_code:str
    visual_code:str


@app.post("/chat")
def chat(request: ChatRequest):
    cleaned_path='outputs/cleaned_dataset.csv'
    prompt = request.prompt
    raw_profile = request.raw_profile
    cleaning_plan = request.cleaning_plan
    visual_plan = request.visual_plan
    cleaning_code=request.cleaning_code
    visual_code=request.visual_code

        # send these to your agent here
    result = chatbot.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"""
    User question:
    {request.prompt}

    Raw dataset profile:
    {request.raw_profile}

    Cleaning plan:
    {request.cleaning_plan}

    Visualization plan:
    {request.visual_plan}
    cleaning code:
    {cleaning_code}
    visualization code:
    {visual_code}
cleaned dataset path:{cleaned_path}
    """
                }
            ]
        },
        config={
            "configurable": {
                "thread_id": "user-123"
            }
        }
    )

    return {
            "reply": result['messages'][-1].content
        }

class path(BaseModel):
    dataset_path:str


class FilePath(BaseModel):
    dataset_path: str


@app.post("/ml-pipeline")
async def ml_pipeline(file_path: FilePath):

    response = pipeline.invoke({
        "base_model": "",
        "dataset_path": file_path.dataset_path,
        "metric": "",
        "metric_value": "",
        "pre_process": "",
        "r2": "",
        "report": "",
        "target_column": "",
        "task_type": "",
        "test_path": "",
        "test_shape": "",
        "train_path": "",
        "train_shape": "",
    })

    return {
        "model": response["base_model"],
        "metric": response["metric"],
        "metric_value": response["metric_value"],
        "pre_process": response["pre_process"],
        "task_type": response["task_type"],
        "train_shape": response["train_shape"],
        "test_shape": response["test_shape"],
        "target_column": response["target_column"],
        "report": response["report"],
        "r2": response["r2"],
    }





# # ---------------------------------------------------------------------------
# # Additional endpoints used by the HTML frontend (existing endpoints above are
# # unchanged). These wrap the exact same step_1 / step_2 calls the Streamlit
# # app used to make in-process.
# # ---------------------------------------------------------------------------

# class PipelineRequest(BaseModel):
#     filename: str


# def _initial_state(dataset_path: str):
#     return {
#         'code': '',
#         'execution_return_code': '',
#         'execution_stderr': '',
#         'execution_stdout': '',
#         'execution_success': '',
#         'file_path': dataset_path,
#         'plan': '',
#         'profile': ''
#     }


# def _write(path: str, content: str):
#     with open(path, 'w', encoding='utf-8') as f:
#         f.write(content)


# @app.post("/run_visuals")
# def run_visuals(request: PipelineRequest):
#     dataset_path = os.path.join('dataset', os.path.basename(request.filename))
#     if not os.path.exists(dataset_path):
#         raise HTTPException(status_code=404, detail="Dataset not found. Upload it first.")
#     try:
#         graphs = step_1.invoke(_initial_state(dataset_path))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"could not build the visuals {str(e)}")

#     os.makedirs('outputs', exist_ok=True)
#     _write('outputs/visual_code.py', graphs['code'])
#     _write('outputs/visual_plan.txt', graphs['plan'])
#     return {
#         'visual_code': graphs['code'],
#         'visual_plan': graphs['plan'],
#         'raw_profile': graphs['profile'],
#     }


# @app.post("/run_cleaning")
# def run_cleaning(request: PipelineRequest):
#     dataset_path = os.path.join('dataset', os.path.basename(request.filename))
#     if not os.path.exists(dataset_path):
#         raise HTTPException(status_code=404, detail="Dataset not found. Upload it first.")
#     try:
#         cleaned = step_2.invoke(_initial_state(dataset_path))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Could not Clean The Dataset {str(e)}")

#     os.makedirs('outputs', exist_ok=True)
#     _write('outputs/cleaning_code.py', cleaned['code'])
#     _write('outputs/cleaning_plan.txt', cleaned['plan'])
#     return {
#         'cleaning_code': cleaned['code'],
#         'cleaning_plan': cleaned['plan'],
#     }


# @app.get("/images")
# def list_images():
#     folder = 'outputs'
#     if not os.path.isdir(folder):
#         return {'images': []}
#     images = sorted(
#         f for f in os.listdir(folder)
#         if os.path.isfile(os.path.join(folder, f))
#         and f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
#     )
#     return {'images': images}


# @app.get("/agent_files")
# def agent_files():
#     def read(name):
#         path = os.path.join('outputs', name)
#         if os.path.exists(path):
#             with open(path, 'r', encoding='utf-8') as f:
#                 return f.read()
#         return ''
#     return {
#         'visual_plan': read('visual_plan.txt'),
#         'cleaning_plan': read('cleaning_plan.txt'),
#         'visual_code': read('visual_code.py'),
#         'cleaning_code': read('cleaning_code.py'),
#     }


# # Serve generated figures and the HTML frontend from the same origin (no CORS).
# os.makedirs('outputs', exist_ok=True)
# app.mount('/outputs', StaticFiles(directory='outputs', check_dir=False), name='outputs')
# app.mount('/ui', StaticFiles(directory=os.path.join('frontend', 'web'), html=True), name='ui')
