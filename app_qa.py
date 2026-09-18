import streamlit as st
from rag import RagService
import config_data as config
st.title("智能客服")
st.divider()

rag_service=RagService()
if "messages" not in st.session_state:
    st.session_state["messages"]=[{"role":"assistant","content":"你好"}]
if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

prompt=st.chat_input()

if prompt is None:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role":"user","content":prompt})

    st.chat_message("assistant")