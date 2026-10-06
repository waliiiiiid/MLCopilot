from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from dotenv import load_dotenv
import os
from backend.agents.tools.profile_dataset import profile_dataset
from backend.agents.tools.docker_execution import docker_executor
from backend.agents.workflows.ml_state import ml_state
from backend.agents.tools.model_selection import select_model
import json

from langchain_groq import ChatGroq
from dotenv import load_dotenv

import os

load_dotenv()

key=os.environ.get('GOOGLE_API_KEY_2')
#model=ChatGroq(model='openai/gpt-oss-120b',temperature=0.2,max_tokens=2000)
#model=ChatGroq(model='openai/gpt-oss-20b',temperature=0.2,max_tokens=2000)
#model=ChatGroq(model='openai/gpt-oss-safeguard-20b',temperature=0.2,max_tokens=2000)

model=ChatGoogleGenerativeAI(model='gemini-3.8-flash', google_api_key=key)

model_selector=create_agent(
    model=model,
    tools=[profile_dataset,select_model],
    system_prompt = """
You are the Model Selection agent in a multi-agent machine learning pipeline.
Context:
- The user message contains two lines:
    Line 1: the file path of the preprocessed CSV dataset.
    Line 2: "Target column: <name>", the target column provided by the user.
- Pass the file path (line 1 only) as the file_path argument to every tool call.
- Use the provided target column exactly as given. Do not infer, change, or
  second-guess it.
- Your output is consumed by a downstream agent, not a human, so it must
  follow the exact output format below.

Tools:
- profile_dataset(file_path): returns shape, per-column dtype, unique counts,
  sample values, and value counts for low-cardinality columns.
- select_model(file_path, task, target_column, metric): runs 5-fold
  cross-validation on candidate models and returns the best model and all scores.
Workflow:
1. Call profile_dataset first.
2. Verify that the provided target column exists in the dataset profile.
   - If it does not exist, return the error output described below.
3. Determine the task type from the target's profile:
   - "classification": categorical, boolean, or few distinct integer values
     representing classes.
   - "regression": continuous numeric values with many distinct values.
4. Choose the metric:
   - Balanced classification: "accuracy".
   - Imbalanced classification (smallest class under ~20% of rows): "f1_macro".
   - Regression: "r2".
5. Call select_model once with task, target_column, and metric.
6. Take the best_model and its mean CV score from the tool result and return the
   final output.

Output format (return exactly this JSON, with no extra text or markdown):
{
  "target_column": "<the provided target column>",
  "task_type": "classification" | "regression",
  "base_model": "<best model name from select_model>",
  "metric": "<metric name>",
  "metric_value": <mean CV score of the best model, as a number>
}

If you cannot complete the task (no dataset, target column missing from the
dataset, or all models failed), return exactly:
{
  "error": "<short reason>"
}
Rules:
- Do not use markdown of any kind. No code fences, no backticks, no bold, no headers, no bullet points.
- Return only the raw JSON object, starting with { and ending with }. Nothing before it and nothing after it.
- Report only values returned by the tools. Never invent or round scores yourself.
- Do not call select_model more than once unless the first call returned an error
  that you can fix (e.g. wrong column name).
- Do not include explanations, rankings, or commentary in the output.
"""
)

# def model_selection_node(state:ml_state):
#     response=model_selector.invoke({
#         'messages':[
#             {'role':'user','content':"file path:'outputs/train_processed.csv'\nTarget column: "+state['target_column']}
#         ]
#     })
#     result=json.loads(response['messages'][-1].content)
#     print(response['messages'][-1].content)
#     return{
#         'base_model':result['base_model'],
#         'metric':result['metric'],
#         'metric_value':result['metric_value'],
#         'task_type':result['task_type'],
#         'target_column':result['target_column']
#     }
  


def model_selection_node(state:ml_state):
    response=model_selector.invoke({
        'messages':[
            {'role':'user','content':"file path:'outputs/train_processed.csv'\nTarget column: "+state['target_column']}
        ]
    })
    result=json.loads(response['messages'][-1].text)
    print(response['messages'][-1].text)
    return{
        'base_model':result['base_model'],
        'metric':result['metric'],
        'metric_value':result['metric_value'],
        'task_type':result['task_type'],
        'target_column':result['target_column']
    }
  

# response=model_selector.invoke({
#     'messages':[
#         {'role':'user','content':'outputs/train_processed.csv'}
#     ]
# })

# result=json.loads(response['messages'][-1].content)

# print(result['base_model'])
# print(result['metric']) 
# print(result['metric_value'])
# print(result['task_type']) 
# print(result['target_column']) 
