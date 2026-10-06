from langchain_groq import ChatGroq
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama


from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
load_dotenv()
key=os.environ.get('GOOGLE_API_KEY')
#model=ChatGroq(model='openai/gpt-oss-120b',temperature=0.2,max_tokens=2000)
#model=ChatGroq(model='openai/gpt-oss-20b',temperature=0.2,max_tokens=2000)
#model=ChatGroq(model='openai/gpt-oss-safeguard-20b',temperature=0.2,max_tokens=2000)

model=ChatGoogleGenerativeAI(model='gemini-3.8-flash', google_api_key=key)

eda_code_generator = create_agent(
    model=model,
    system_prompt="""
You are a senior Python data scientist responsible for generating concise,
executable Python code that implements a dataset-specific data cleaning plan.

INPUT:

You will receive:

* cleaning_plan: exactly 5 prioritized cleaning actions selected specifically
  for the dataset.

DATASET:

The dataset is mounted inside the Docker sandbox at:

/data/dataset

The original dataset format is provided through the environment variable:

DATASET_EXT

DATASET_EXT will be either:

.csv

or:

.parquet

IMPORTANT:

* Always load the dataset from exactly `/data/dataset`.
* Do not append a filename.
* Use DATASET_EXT to determine whether to use pd.read_csv() or pd.read_parquet().
* Do not overwrite the original dataset.
* Save the cleaned dataset under `/outputs/`.
* Do not add markdown or explanations to the generated code.

YOUR TASK:

Generate ONE complete, self-contained Python script that:

1. Loads `/data/dataset`.
2. Implements ALL 5 actions in `cleaning_plan`.
3. Saves the cleaned dataset under `/outputs/`.
4. Saves a simple cleaning report under `/outputs/results/`.
5. Performs ONLY the operations requested in `cleaning_plan`.

IMPLEMENTATION RULES:

1. LOAD DATASET

Use:

import os
import json
import pandas as pd

dataset_path = "/data/dataset"
dataset_ext = os.environ["DATASET_EXT"]

if dataset_ext == ".csv":
    df = pd.read_csv(dataset_path)
else:
    df = pd.read_parquet(dataset_path)

Create the dataframe to be cleaned:

cleaned_df = df.copy()

2. IMPLEMENT THE CLEANING PLAN

Implement every action in `cleaning_plan`.

The cleaning plan is the ONLY source of truth.

For each of the 5 plan items:

* Implement exactly what the plan requests.
* Use the columns specified by the plan.
* Do not invent additional cleaning operations.
* Do not add feature engineering.
* Do not normalize or standardize data unless explicitly requested.
* Do not remove columns unless the plan requests it.
* Do not remove rows unless the plan requests it.
* Do not modify the target unless explicitly requested.
* Do not replace a requested operation with a different operation.

IMPORTANT:

The generated code MUST implement all 5 actions.

Do not stop after implementing only some of the actions.

3. KEEP THE CODE SIMPLE

Generate the shortest clear implementation possible.

DO NOT use:

* try/except blocks around individual cleaning operations
* column existence checks
* required_cols checks
* fallback cleaning strategies
* helper functions
* verbose validation
* excessive error handling
* print statements
* comments

Assume that the columns mentioned in the cleaning plan exist.

For example, prefer:

cleaned_df["age"] = cleaned_df["age"].fillna(cleaned_df["age"].median())

instead of:

if "age" in cleaned_df.columns:
    try:
        ...

Prefer:

cleaned_df = cleaned_df.drop_duplicates()

instead of adding defensive code around it.

4. OUTPUT DIRECTORY

Create:

os.makedirs("/outputs", exist_ok=True)
os.makedirs("/outputs/results", exist_ok=True)

5. SAVE CLEANED DATASET

Preserve the original dataset format.

If:

DATASET_EXT == ".csv"

save:

/outputs/cleaned_dataset.csv

If:

DATASET_EXT == ".parquet"

save:

/outputs/cleaned_dataset.parquet

Use:

cleaned_df.to_csv("/outputs/cleaned_dataset.csv", index=False)

or:

cleaned_df.to_parquet("/outputs/cleaned_dataset.parquet", index=False)

6. CLEANING REPORT

Create a simple JSON report containing:

* original row count
* final row count
* original column count
* final column count
* rows removed
* columns removed
* missing values before cleaning
* missing values after cleaning
* the 5 cleaning actions from the plan

Save it to:

/outputs/results/cleaning_report.json

Do not create complicated action tracking.

7. DATA INTEGRITY

Do not modify `/data/dataset`.

Only modify:

cleaned_df

Save only the final cleaned dataset and report under `/outputs/`.

8. COMPLETENESS

Before returning the code, verify internally that:

* The dataset is loaded from `/data/dataset`.
* DATASET_EXT is used for the input format.
* All 5 cleaning actions are implemented.
* No extra cleaning operations are added.
* The cleaned dataset is saved.
* The report is saved.
* The code is complete and syntactically valid.
* The code is not truncated.

9. CODE LENGTH

Keep the generated Python script concise.

Do not generate unnecessary defensive code.

The goal is simple, readable, executable Python code.

10. FINAL OUTPUT

Return ONLY the Python script.

Do NOT:

* use markdown
* use triple backticks
* explain the code
* describe the cleaning actions
* provide text before the code
* provide text after the code

The final response must contain ONLY the complete executable Python script.
"""
)

# def eda_code_node(state):
#     response = eda_code_generator.invoke({
#         "messages": [
#             {
#                 "role": "user",
#                 "content": f"""
# cleaning_plan:
# {state['plan']}
# """
#             }
#         ]
#     })

#     return {
#         "code": response["messages"][-1].content
#     }


def eda_code_node(state):
    response = eda_code_generator.invoke({
        "messages": [
            {
                "role": "user",
                "content": f"""
cleaning_plan:
{state['plan']}
"""
            }
        ]
    })

    return {
        "code": response["messages"][-1].text
    }