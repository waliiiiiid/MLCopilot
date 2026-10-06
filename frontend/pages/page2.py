import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import requests
from backend.agents.workflows.first_graph import step_1
from backend.agents.workflows.second_graph import step_2

st.set_page_config(
    page_icon='🤖',
    page_title='ML Copilot'
)

st.title("MLCopilot")

# -----------------------------------------
# Chatbot API config (change when your endpoint is ready)
# -----------------------------------------
CHAT_API_URL = "http://localhost:8000/chat"
CHAT_TIMEOUT = 120  # seconds, agent + docker execution can be slow
raw_profile = st.session_state.get("raw_profile", "")
visual_plan = st.session_state.get("visual_plan", "")
cleaning_plan = st.session_state.get("cleaning_plan", "")
visual_code = st.session_state.get("visual_code", "")
cleaning_code = st.session_state.get("cleaning_code", "")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

col1, col2 = st.columns(2)

with col1:
    st.subheader('EDA', text_alignment='center')
    folder = "outputs"

    images = []

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)

        if os.path.isfile(file_path) and filename.lower().endswith(
            (".png", ".jpg", ".jpeg", ".webp")
        ):
            images.append(file_path)

    # Sort images so the order is consistent
    images.sort()

    # Initialize image index
    if "image_index" not in st.session_state:
        st.session_state.image_index = 0

    if images:
        # Guard against the index going out of range if images change
        st.session_state.image_index %= len(images)

        current_image = images[st.session_state.image_index]

        st.image(
            current_image,
            caption=os.path.basename(current_image),
            use_container_width=True
        )

        if st.button("Next →", use_container_width=True):
            st.session_state.image_index = (
                st.session_state.image_index + 1
            ) % len(images)

            st.rerun()

    else:
        st.info("No images found.")

with col2:
    st.subheader('ASK AI', text_alignment='center')

    # Scrollable message area
    chat_box = st.container(height=300, border=True)

    with chat_box:
        if not st.session_state.messages:
            st.caption("How Can I Help You?")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    prompt = st.chat_input("Ask the ML Copilot...")

    if prompt:
        # 🔧 CHANGED: store ONLY the prompt in history.
        # Before, the whole profile/plans/code was saved as the "user message",
        # so it was printed in the chat bubble on every rerun (and had a
        # duplicated cleaning_plan and no visual_code).
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        payload = {
            "prompt": prompt,
            "raw_profile": raw_profile,
            "cleaning_plan": cleaning_plan,
            "visual_plan": visual_plan,
            "cleaning_code": cleaning_code,   # ➕ NEW: required by your FastAPI ChatRequest
            "visual_code": visual_code,       # ➕ NEW: required by your FastAPI ChatRequest
        }
        # ↑ Missing these two fields caused FastAPI to return 422,
        #   so status_code was never 200 -> "Agent is Unavailable!" every time.

        with chat_box:
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # ➕ NEW: try/except so a stopped backend or timeout
                    # doesn't crash the whole Streamlit page
                    try:
                        response = requests.post(
                            CHAT_API_URL,
                            json=payload,
                            timeout=CHAT_TIMEOUT   # ➕ NEW: uses your CHAT_TIMEOUT (was unused)
                        )
                        if response.status_code == 200:
                            reply = response.json()['reply']
                        else:
                            # 🔧 CHANGED: show the status code so errors are debuggable
                            reply = f'Agent is Unavailable! (status {response.status_code})'
                    except requests.exceptions.RequestException:
                        reply = 'Agent is Unavailable! Is the backend running?'
                st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})
        # Rerun so newly saved images from the agent show up in the EDA column
        st.rerun()

    if st.session_state.messages:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

if st.button(
    "View Agent EDA Steps",
    use_container_width=True
):

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Visualization Steps")

        with open("outputs/visual_plan.txt", "r", encoding="utf-8") as f:
            visual_plan = f.read()

        st.text_area("Visualization Plan", value=visual_plan, height=400)

    with col2:
        st.subheader("🧹 Cleaning Steps")

        with open("outputs/cleaning_plan.txt", "r", encoding="utf-8") as f:
            cleaning_plan = f.read()

        st.text_area("Cleaning Plan", value=cleaning_plan, height=400)

if st.button(
    "View Agent Generated Codes",
    use_container_width=True
):

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Visualization Code")

        with open("outputs/visual_code.py", "r", encoding="utf-8") as f:
            visual_code = f.read()

        st.code(visual_code, language='python')

    with col2:
        st.subheader("🧹 Cleaning Code")

        with open("outputs/cleaning_code.py", "r", encoding="utf-8") as f:
            cleaning_code = f.read()

        st.code(cleaning_code, language='python')

if st.button('Start ML Model Building and Training', use_container_width=True):
    st.switch_page('pages/ml.py')
    