from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from dotenv import load_dotenv
from backend.agents.tools.profile_dataset import profile_dataset
from backend.agents.workflows.ml_state import ml_state
import pandas as pd
from sklearn.metrics import classification_report
import joblib
from sklearn.metrics import r2_score
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

key=os.environ.get('GOOGLE_API_KEY_2')
#model=ChatGroq(model='openai/gpt-oss-120b',temperature=0.2,max_tokens=2000)
#model=ChatGroq(model='openai/gpt-oss-20b',temperature=0.2,max_tokens=2000)
#model=ChatGroq(model='openai/gpt-oss-safeguard-20b',temperature=0.2,max_tokens=2000)

model=ChatGoogleGenerativeAI(model='gemini-3.8-flash', google_api_key=key)

# def evaluate_model(state:ml_state):
#     df=pd.read_csv('outputs/test_processed.csv')
#     model_path=f'models/{state['base_model']}.joblib'
#     x=df.drop(columns=[state['target_column']])
#     y=df[state['target_column']]
#     model=joblib.load(model_path)
#     y_pred=model.predict(x)
#     if state['task_type'].lower()=='classification':
#         report=classification_report(y,y_pred)
#     if state['task_type'].lower()=='regression':
#         r2=r2_score(y,y_pred)

#     return{
#         'report':report,
#         'r2':r2
#     }


# def evaluate_model(state: ml_state):
#     df = pd.read_csv("outputs/test_processed.csv")

#     model_path = f"models/{state['base_model']}.joblib"

#     x = df.drop(columns=[state["target_column"]])
#     y = df[state["target_column"]]

#     trained_model = joblib.load(model_path)
#     y_pred = trained_model.predict(x)
    

#     report = ""
#     r2 = 0.0

#     if state["task_type"].lower() == "classification":
#         y_proba = trained_model.predict_proba(x)[:, 1]
#         report = classification_report(y, y_pred)

#     elif state["task_type"].lower() == "regression":
#         r2 = r2_score(y, y_pred)

#     return {
#         "report": report,
#         "r2": r2,
#         'y_proba': y_proba,
#         'y_pred':y_pred,
#         'y_real':y
#     }

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, r2_score

from backend.agents.workflows.ml_state import ml_state

BASE_DIR = Path(__file__).resolve().parents[2]

def evaluate_model(state: ml_state):
    #df = pd.read_csv(BASE_DIR / "outputs" / "test_processed.csv")
    df = pd.read_csv("outputs/test_processed.csv")
    model_path = BASE_DIR / "models" / f"{state['base_model']}.joblib"


    target = state["target_column"]
    x = df.drop(columns=[target])
    y = df[target]

    trained_model = joblib.load(model_path)

    y_pred = trained_model.predict(x).tolist()

    result = {
        "report": "",
        "r2": 0.0,
        "y_proba": None,
        "y_pred": y_pred,
        "y_real": y.tolist(),
    }

    task = state["task_type"].lower()

    if task == "classification":
        result["report"] = classification_report(y, y_pred)
        result["y_proba"] = trained_model.predict_proba(x)[:, 1].tolist()
    elif task == "regression":
        result["r2"] = r2_score(y, y_pred)
    else:
        raise ValueError(f"Unknown task_type: {state['task_type']}")

    return result