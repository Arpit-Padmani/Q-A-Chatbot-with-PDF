import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from htmlTemplate import css, bot_template, user_template

# ------------------------- LLM Greeting -------------------------
def get_llm_greeting(user_input):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.8,
        google_api_key="AIzaSyCIbUQQQ59a1L2f2BBxJCYH12FoWt1UI8g"
    )
    prompt = (
        f"User: {user_input}\n"
        "Reply in a short, friendly, and casual way. "
        "Keep it to one sentence, like natural small talk."
    )
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        st.error(f"Error calling LLM for greeting: {e}")
        return "Hi there! How are you today?"

# ------------------------- User Input Handler -------------------------
def handle_userinput(user_question, chat_placeholder):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # LLM greeting/response
    response_text = get_llm_greeting(user_question)

    # Display in chat
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    st.session_state.chat_history.append({"role": "assistant", "content": response_text})
    
    with chat_placeholder.container():
        st.write(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", response_text), unsafe_allow_html=True)

# ------------------------- Streamlit App -------------------------
def main():
    st.set_page_config(page_title="Greeting ChatBot Demo", page_icon="🤖")
    st.write(css, unsafe_allow_html=True)
    st.header("Welcome ChatBot Demo 🤖")

    chat_placeholder = st.empty()

    # Initialize chat history with default greeting if empty
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Hello 👋, you can upload a file and ask me questions about it."}
        ]

    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "assistant":
            st.write(bot_template.replace("{{MSG}}", msg["content"]), unsafe_allow_html=True)
        else:
            st.write(user_template.replace("{{MSG}}", msg["content"]), unsafe_allow_html=True)

    # Single input box for user
    user_input = st.text_input("You:", key="user_input")
    if user_input:
        handle_userinput(user_input, chat_placeholder)

if __name__ == "__main__":
    main()
