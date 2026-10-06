from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from dotenv import load_dotenv
from backend.agents.tools.profile_dataset import profile_dataset
from backend.agents.tools.docker_execution import docker_executor
from backend.agents.workflows.ml_state import ml_state
from backend.agents.tools.optimization import optimize_model
from pydantic import BaseModel
from langchain.agents.structured_output import ToolStrategy
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

key=os.environ.get('GOOGLE_API_KEY_2')
#model=ChatGroq(model='openai/gpt-oss-120b',temperature=0.2,max_tokens=2000)
#model=ChatGroq(model='openai/gpt-oss-20b',temperature=0.2,max_tokens=2000)
#model=ChatGroq(model='openai/gpt-oss-safeguard-20b',temperature=0.2,max_tokens=2000)

model=ChatGoogleGenerativeAI(model='gemini-3.8-flash', google_api_key=key)

model_trainer = create_agent(
    model=model,
    tools=[profile_dataset, optimize_model],
    system_prompt="""
You are a machine learning engineer. The dataset is already cleaned and ready to train.
You receive a model name, target column, task type and file path.

Steps:
1. Call profile_dataset(file_path).
2. Call optimize_model with a small parameter grid (at most ~10 combinations)
   and a scoring metric that fits the task (classification: 'accuracy' or 'f1_weighted';
   regression: 'r2').
3. If the tool returns an error, fix your arguments and retry.
4. Reply with the best parameters, cv_score, test_score and model_path exactly as
   the tool returned them. Never invent numbers.
""",
)



def train_model(state: ml_state):
    response = model_trainer.invoke({
        "messages": [
            {
                "role": "user",
                "content": (
                    f"train dataset path:'outputs/train_processed.csv',"
                    f"model={state['base_model']},"
                    f"target column:{state['target_column']},"
                    f"task:{state['task_type']}"
                )
            }
        ]
    })



    

    # return {
    #     "train_report": response["messages"][-1].content
    # }

# response = model_trainer.invoke({
#         "messages": [
#             {
#                 "role": "user",
#                 "content": (
#                     "train dataset path:'outputs/train_processed.csv',"
#                     "model=LogisticRegression,"
#                     "target column:species,"
#                     "task:classification,"
#                 )
#             }
#         ]
#     })

# print(response['messages'][-1].content)