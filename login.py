import streamlit as st
import mysql.connector
from dataclasses import dataclass
from typing import Literal
from langchain import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryMemory
import streamlit.components.v1 as components

# Connect to the MySQL database
def connect_db():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="osman",
        database="codecon"
    )
    return conn

# Check if the provided credentials are valid
def validate_login(email, password):
    conn = connect_db()
    cursor = conn.cursor()
    
    query = "SELECT * FROM users WHERE email = %s AND password = %s"
    cursor.execute(query, (email, password))
    result = cursor.fetchone()
    
    cursor.close()
    conn.close()
    
    return result is not None

# Streamlit login form
def login_form():
    st.title("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if validate_login(email, password):
            # Redirect to landing page
            st.experimental_set_query_params(username=email)
            st.experimental_rerun()
        else:
            st.error("Invalid username or password")

# Streamlit landing page
def landing_page():
    @dataclass
    class Message:
        """Class for keeping track of a chat message."""
        origin: Literal["human","ai"]
        message: str

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

    def on_click_callback():
        with get_openai_callback() as cb:
            human_prompt = st.session_state.human_prompt
            llm_response = st.session_state.conversation.run(
                human_prompt
            )
            with chat_placeholder:
                st.markdown(llm_response)

    initialise_session_state()

    st.title("Welcome to Tenet's ChatBot!")
    username = st.experimental_get_query_params().get("username", [""])[0]
    st.write(f"Logged in as: {username}")

    chat_placeholder = st.container()
    prompt_placeholder = st.container()
    credit_card_placeholder = st.empty()
        

    pd = st.text_input("Enter Previous Diagnoses")
    th = st.text_input("Enter Treatment History")
    cm = st.text_input("Enter Current Medications")
    allergies = st.text_input("Enter Allergies")
    fmh = st.text_input("Enter Family Medical History")
    cp = st.text_input("Enter Current problem")
    v = f"Previous Diagnoses: {pd}\nTreatment History: {th}\nCurrent Medications: {cm}\nAllergies: {allergies}\nFamily Medical History: {fmh}\nCurrent Problem: {cp}\nGive me a solution to current problem\nFill these blanks\nCauses:\nFuture Scope:\ndont say you are an AI language model blah blah\ngive causes in 2 points and future scope in 1 point"

    with prompt_placeholder:
    #     textcont = st.empty()
    #     t = textcont.text_input(v)
    #     if st.button():
    #         if t != "":
    #             textcont.empty()
    #             st.info(t)
    #             on_click_callback
            user_input = v
            if st.button("Generate Response"):
                on_click_callback

# Main function to run the app
def main():
    if "username" not in st.experimental_get_query_params():
        login_form()
    else:
        landing_page()

if __name__ == "__main__":
    main()
