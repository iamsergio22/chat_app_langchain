import ollama
import sys
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage


def app_session_init():
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [
            AIMessage("Hello, how can I help you?")]

    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = get_models()[0]

    chat_history = st.session_state["chat_history"]
    for history in chat_history:
        if isinstance(history, AIMessage):
            st.chat_message("ai").write(history.content)
        if isinstance(history, HumanMessage):
            st.chat_message("human").write(history.content)


def get_models():
    models = ollama.list()
    if not models:
        print("No models found")
        sys.exit(1)

    models_list = []
    for model in models:        
        models_list.append(model[1][0]["model"])        

    return models_list


def run():
    """Set the application and select the llm. Show interface to user."""
    st.set_page_config(page_title="Chat Applicaction")
    st.header("Chat :blue[Application]")
    st.selectbox("Select LLM:", get_models(), key="selected_model")

    app_session_init()
    prompt = st.chat_input("Add your prompt")

    selected_model = st.session_state["selected_model"]
    print(selected_model)
    llm = ChatOllama(model=selected_model, temperature=0.3)

    if prompt:
        st.chat_message("user").write(prompt)
        st.session_state["chat_history"] += [HumanMessage(prompt)]
        output = llm.stream(prompt)

        with st.chat_message("ai"):
            ai_message = st.write_stream(output)

        st.session_state["chat_history"] += [AIMessage(ai_message)]


if __name__ == "__main__":
    run()
