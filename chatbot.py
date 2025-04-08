import streamlit as st
from uuid import uuid4
from module import * 

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'processing' not in st.session_state:
    st.session_state.processing = False

st.title("Inventory Optimization Chatbot")

for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(message["content"])

def handle_response(prompt):
    st.session_state.processing = True

    with st.spinner("Generating response..."):
        response = get_response(prompt)
        if response:
            with st.chat_message("assistant"):
                st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            response = "Sorry, I couldn't generate a response. Try again later."
            with st.chat_message("assistant"):
                st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    st.session_state.processing = False

if not st.session_state.processing:
    prompt = st.chat_input("Ask me anything related to stock management")
else:
    prompt = None 

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    handle_response(prompt)
