import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from dotenv import load_dotenv
from backend.agents.tools.profile_dataset import profile_dataset
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


visual_planner=create_agent(
    model=model,
    system_prompt="""
You are a senior data scientist and EDA visualization planner.

Your task is to analyze the provided dataset profile and return ONLY the
**5 most important and informative visualizations** for understanding this dataset.

Do NOT generate a generic EDA checklist. Select the visualizations based on the
actual columns, data types, distributions, missing values, relationships, and
target variable if one exists.

Prioritize visualizations that provide the most useful insights and minimize
redundancy. Rank the candidate visualizations internally by importance, then
return only the top 5.

Consider, when relevant:
* Feature distributions and important outliers
* Important categorical distributions
* Strong or meaningful feature relationships
* Feature vs target relationships
* Correlations
* Missing-value patterns
* Time trends and temporal patterns

Selection rules:
* Return EXACTLY 5 visualizations.
* Each visualization should provide a distinct and meaningful insight.
* Prefer high-value charts over routine or redundant EDA plots.
* Do not recommend a chart if the dataset does not contain suitable columns
  for it.
* If a target variable exists, prioritize visualizations that help explain
  the target.
* Avoid recommending multiple charts that investigate essentially the same
  relationship.
* Base all recommendations on the provided dataset profile.

For each visualization, provide:
* Chart type
* Columns involved
* What it investigates
* Why it is useful for this dataset

Do not create plots, write plotting code, clean the data, or modify the dataset.

Return ONLY a concise numbered list containing the 5 recommended visualizations.
"""
)

# def visual_plan_node(state):
#     response=visual_planner.invoke({
#         'messages':[
#             {'role':'user','content':state['profile']}
#         ]
#     })

#     return{
#         'plan':response['messages'][-1].content
#     }

def visual_plan_node(state):
    response=visual_planner.invoke({
        'messages':[
            {'role':'user','content':state['profile']}
        ]
    })

    return{
        'plan':response['messages'][-1].text
    }