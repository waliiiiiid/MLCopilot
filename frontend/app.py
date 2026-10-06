import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
from backend.agents.workflows.first_graph import step_1
from backend.agents.workflows.second_graph import step_2
import requests
import pandas as pd

API_URL = "http://localhost:8000/"


st.title("MLCopilot")
st.set_page_config(
    page_icon='🤖',
    page_title='ML Copilot'
)

response=requests.get(f'{API_URL}')

file = st.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)


if file is not None:
    with st.spinner('Reading File'):
        try:
            files = {
                "file": (
                    file.name,
                    file.getvalue(),
                    "text/csv"
                )
            }

            response = requests.post(
                f"{API_URL}upload_dataset",
                files=files
            )

            if response.status_code == 200:
                result = response.json()

                st.success(result["message"])
                st.write(file.name)
                dataset_path=f'dataset/{file.name}'
                st.session_state.dataset_path=dataset_path
                df=pd.read_csv(f'dataset/{file.name}')
                st.dataframe(
                    df,
                    use_container_width=True,
                    height=200                 
                )

            else:
                st.error(f"Something went wrong: {response.text}")

        except Exception as e:
            st.error(f"Error: {str(e)}")

if st.button('Start',use_container_width=True):
    with st.spinner("Analyzing your dataset and generating visualizations..."):
        try:
            graphs=step_1.invoke({
                'code':'',
                'execution_return_code':'',
                'execution_stderr':'',
                'execution_stdout':'',
                'execution_success':'',
                'file_path':dataset_path,
                'plan':'',
                'profile':''
            })
            visual_code=graphs['code']
            visual_plan=graphs['plan']
            raw_profile=graphs['profile']
            st.session_state.visual_code = visual_code
            st.session_state.visual_plan = visual_plan
            st.session_state.raw_profile = raw_profile
            with open('outputs/visual_code.py','w',encoding="utf-8")as f:
                f.write(visual_code)
            with open('outputs/visual_plan.txt','w',encoding="utf-8")as f:
                f.write(visual_plan)
            st.success('Visuals Created Successfully!')
        except Exception as e:
            st.error(f'could not build the visuals {str(e)}')
    with st.spinner('Cleaning Data...'):
        try:
            cleaned_data=step_2.invoke({
                'code':'',
                'execution_return_code':'',
                'execution_stderr':'',
                'execution_stdout':'',
                'execution_success':'',
                'file_path':dataset_path,
                'plan':'',
                'profile':''
            })
            cleaning_code=cleaned_data['code']
            cleaning_plan=cleaned_data['plan']
            st.session_state.cleaning_code = cleaning_code
            st.session_state.cleaning_plan = cleaning_plan
            with open('outputs/cleaning_code.py','w',encoding="utf-8")as f:
                f.write(cleaning_code)
            with open('outputs/cleaning_plan.txt','w',encoding="utf-8")as f:
                f.write(cleaning_plan)
            st.success('Data Cleaning was Done Successfully!')
            
            st.switch_page('pages/page2.py')
            
        except Exception as e:
            st.error(f'Could not Clean The Dataset{str(e)}')







# # folder = "outputs"
# # for filename in os.listdir(folder):
# #     file_path = os.path.join(folder, filename)

# #     if os.path.isfile(file_path) and filename.lower().endswith(
# #         (".png", ".jpg", ".jpeg", ".webp")
# #     ):
# #         st.image(file_path, caption=filename)



