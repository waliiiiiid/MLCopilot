from langchain_groq import ChatGroq
from langchain.agents import create_agent
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from backend.agents.tools.docker_execution import docker_executor
from backend.agents.workflows.ml_state import ml_state
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

def splitter(state:ml_state):
    path=str(state['dataset_path'])
    df=pd.read_csv(path)
    rows=df.shape[0]
    train_split=int(0.8*rows)
  
    train_ds=df.iloc[:train_split]
    test_ds=df.iloc[train_split:]

    train_ds.to_csv(os.path.join('outputs', "train.csv"), index=False)
    test_ds.to_csv(os.path.join('outputs', "test.csv"), index=False)
    print('data splitted')
    return{
        'train_path':'outputs/train.csv',
        'test_path':'outputs/test.csv',
        'train_shape':train_ds.shape,
        'test_shape':test_ds.shape
    }