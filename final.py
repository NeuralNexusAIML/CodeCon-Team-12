import streamlit as st
from dataclasses import dataclass
from typing import Literal
from streamlit_option_menu import option_menu
from langchain import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryMemory
import streamlit.components.v1 as components

def initialise_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0
    if "conversation" not in st.session_state:
        llm = OpenAI(
        temperature = 0,
        openai_api_key=st.secrets["openai_api_key"],
        model_name="text-davinci-003"
        )
        st.session_state.conversation = ConversationChain(
            llm = llm,
            memory = ConversationSummaryMemory(llm=llm),
        )

def progbot():
    @dataclass
    class Message:
        """Class for keeping track of a chat message."""
        origin: Literal["human","ai"]
        message: str

      

st.sidebar.title("Navigation")
with st.sidebar:
    selected = option_menu(
            menu_title="Main Menu",
            options = ["Prognosis Bot","Disease Diagnosis","Tumor Detection","Fracture Detection"],
            menu_icon = "cast",
        )
    if selected == "Prognosis Bot":
        progbot()

    if selected == "Disease Diagnosis":
            diagnose()

    if selected == "Tumor Detection":
            tumor()

    if selected == "Fracture Detection":
            fracture()