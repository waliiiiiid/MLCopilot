import streamlit as st
import requests
from backend.agents.workflows.ml import pipeline
API = "http://localhost:8000/"
from pathlib import Path


path=st.session_state.get('dataset_path')
st.write(path)

if st.button('start'):
    response = pipeline.invoke({
        'base_model': '',
        'dataset_path': path,
        'metric': '',
        'metric_value': '',
        'pre_process': '',
        'r2': '',
        'report': '',
        'target_column': '',
        'task_type': '',
        'test_path': '',
        'test_shape': '',
        'train_path': '',
        'train_shape': '',
        'y_pred': '',
        'y_proba': '',
        'y_real': ''
    })

    st.success("### Pipeline Finished Successfully")

    st.write("**base_model:**", response.get('base_model'))
    #st.write("**dataset_path:**", response.get('dataset_path'))
    st.write("**metric:**", response.get('metric'))
    st.write("**metric_value:**", response.get('metric_value'))
    st.write("**pre_process:**", response.get('pre_process'))
    st.write("**r2:**", response.get('r2'))
    st.write("**report:**", response.get('report'))
    st.write("**target_column:**", response.get('target_column'))
    st.write("**task_type:**", response.get('task_type'))
    st.write("**test_shape:**", response.get('test_shape'))
    st.write("**train_shape:**", response.get('train_shape'))

image_folder = Path("outputs/evaluation")
images = []
for extension in ["*.png", "*.jpg", "*.jpeg", "*.webp"]:
    images.extend(image_folder.glob(extension))

# Sort by filename
images = sorted(images)

if not images:
    st.info("No images found in the outputs folder.")
else:
    for image_path in images:
        st.subheader(image_path.name)
        st.image(str(image_path), use_container_width=True)