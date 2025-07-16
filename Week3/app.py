import streamlit as st
from pipeline import CodeGenPipeline

st.set_page_config(page_title="CodeGenBot", page_icon="🤖")
st.title("💻 CodeGenBot")

# Initialize pipeline only once (cache in session state)
if "pipeline" not in st.session_state:
    st.session_state.pipeline = CodeGenPipeline("hf://datasets/openai/openai_humaneval/openai_humaneval/test-00000-of-00001.parquet")

# Memory for chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        st.chat_message("assistant").code(msg["content"], language="python")
    else:
        st.chat_message(msg["role"]).write(msg["content"])

# User input
user_input = st.chat_input("Ask CodeGenBot to generate Python code...")

if user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Generate code using your pipeline
    with st.spinner("Generating code..."):
        try:
            code_output = st.session_state.pipeline.generate_code_from_prompt(user_input)
        except Exception as e:
            code_output = f"Error: {e}"

    # Save assistant's reply and rerun to display it immediately
    st.session_state.messages.append({"role": "assistant", "content": code_output})
    st.rerun()