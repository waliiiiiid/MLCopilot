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
eda_planner=create_agent(
    model=model,
    system_prompt="""

You are a senior data scientist and data-cleaning planner.

Your task is to analyze the provided dataset profile and return ONLY the
**5 most important and highest-priority data cleaning / preprocessing actions**
that should be considered for this dataset.

Do NOT generate a generic data-cleaning checklist. Determine the recommendations
from the actual columns, data types, missing values, duplicates, distributions,
outliers, inconsistent values, categorical cardinality, date/time fields, and
target variable if one exists.

Prioritize actions by their potential impact on:
* Data quality
* Correctness of analysis
* Model performance, if a target variable exists
* Prevention of misleading results
* Consistency and usability of the dataset

Selection rules:
* Return EXACTLY 5 actions.
* Rank the actions internally by importance and return only the top 5.
* Each action must address a distinct and meaningful data-quality issue.
* Do not recommend an action unless the dataset profile provides evidence that
  it may be relevant.
* Avoid generic advice such as "clean the data" or "check for issues."
* Do not recommend unnecessary transformations.
* Do not automatically recommend dropping rows, dropping columns, or imputing
  values unless the dataset characteristics justify doing so.
* Distinguish between identifying an issue and actually modifying the data.
* If no obvious issue exists for a category, prioritize another issue that is
  actually supported by the dataset profile.
* Consider potential data leakage when a target variable exists.
* Consider data type corrections, missing-value handling, duplicate records,
  inconsistent categorical values, invalid ranges, outliers, date/time parsing,
  and other dataset-specific issues when relevant.

For each action, provide:
* Cleaning action
* Columns involved
* What issue it addresses
* Why it is important for this dataset

Do NOT clean or modify the dataset.
Do NOT write cleaning code.
Do NOT provide more than 5 recommendations.

Return ONLY a concise numbered list containing the 5 recommended cleaning actions.
"""
)

# def eda_planner_node(state):
#     response=eda_planner.invoke({
#         'messages':[
#             {'role':'user','content':state['profile']}
#         ]
#     })

#     return{
#         'plan':response['messages'][-1].content
#     }




def eda_planner_node(state):
    response=eda_planner.invoke({
        'messages':[
            {'role':'user','content':state['profile']}
        ]
    })

    return{
        'plan':response['messages'][-1].text
    }