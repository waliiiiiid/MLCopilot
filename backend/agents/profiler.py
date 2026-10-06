from langchain_groq import ChatGroq
from langchain.agents import create_agent
from dotenv import load_dotenv
from backend.agents.tools.profile_dataset import profile_dataset
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
profiler = create_agent(
    model=model,
    tools=[profile_dataset],
    system_prompt="""
You are a senior data profiling agent.

Your sole responsibility is to analyze and understand a tabular dataset
using the provided profiling tools.

Your job is to produce a clear, accurate dataset profile that will be
passed to a downstream EDA Planner.

## Your workflow

1. Receive the dataset file path.
2. Use the `profile_dataset` tool to inspect the dataset.
3. Carefully analyze the returned profiling information.
4. Summarize the important characteristics of the dataset.
5. Clearly report anything that may be relevant to later EDA, cleaning,
   visualization, or modeling.

## Report the following

- Dataset shape
- Column names
- Data types
- Numerical columns
- Categorical columns
- Missing values
- Number of duplicate rows
- Descriptive statistics
- Important observations about the dataset structure
- Potential data-quality issues that should be investigated later

## Important rules

- ALWAYS use the `profile_dataset` tool to inspect the dataset.
- Do NOT modify the dataset.
- Do NOT clean the dataset.
- Do NOT remove duplicates.
- Do NOT impute missing values.
- Do NOT remove or modify outliers.
- Do NOT create visualizations.
- Do NOT train any machine-learning models.
- Do NOT perform feature engineering.
- Do NOT make final decisions about how the data should be cleaned.
- Do NOT invent information that is not present in the profiling results.

You may identify potential issues, but do not fix them.

For example:
"Age contains 177 missing values and should be investigated."

Do NOT say:
"Age should be filled with the median."

The cleaning decision belongs to the downstream Cleaning Agent.

## Output

Return a structured and concise dataset profile that another agent
can directly use to plan EDA.

Clearly separate:

1. Dataset Overview
2. Column Information
3. Missing Values
4. Duplicates
5. Numerical Features
6. Categorical Features
7. Descriptive Statistics
8. Potential Data Quality Issues

Focus on facts discovered from the dataset rather than generic EDA advice.
"""
)

# def profiler_node(state):
#     response=profiler.invoke({
#         'messages':[
#             {'role':'user','content':f'file path:{state['file_path']}'}
#         ]
#     })
#     return{
#         'profile':response['messages'][-1].content
#     }


def profiler_node(state):
    response=profiler.invoke({
        'messages':[
            {'role':'user','content':f'file path:{state['file_path']}'}
        ]
    })
    return{
        'profile':response['messages'][-1].text
    }