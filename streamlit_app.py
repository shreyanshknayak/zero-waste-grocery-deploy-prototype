# streamlit_app.py
import streamlit as st
import requests

st.set_page_config(page_title="Zero-Waste Grocery Helper", layout="wide")
st.title("🥕 Zero-Waste Grocery Helper")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "state" not in st.session_state:
    st.session_state.state = {}

user_input = st.chat_input("Tell me what ingredients you have...")

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    
    # Send to FastAPI
    payload = {
        "message": user_input,
        "state": st.session_state.state
    }
    response = requests.post("http://localhost:8000/chat", json=payload)
    print(response.text)
    response_data = response.json()
    
    bot_reply = response_data["response"]
    st.session_state.state = response_data["state"]
    st.session_state.chat_history.append(("assistant", bot_reply))

# Display history
chat_history = st.session_state.state.get("chat_history", [])

# Skip displaying the first message
for i, (role, msg) in enumerate(chat_history):
    if i == 0:
        continue
    with st.chat_message(role):
        st.markdown(msg)