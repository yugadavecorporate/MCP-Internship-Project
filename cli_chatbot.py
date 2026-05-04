import os
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check if API Key is available
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("Missing API Key! Please create a .env file and add GROQ_API_KEY='your_key'.")
    st.stop()

# Initialize Groq client
client = Groq(api_key=api_key)

# Minimalist UI configuration
st.set_page_config(page_title="Chat", page_icon="💬", layout="centered")

# Hide default Streamlit elements for a cleaner, app-like look
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# App Title & Subtitle (Minimalist)
st.title("💬 Simple Assistant")
st.caption("A distraction-free, real-time chat experience.")

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful and concise assistant."}
    ]

# Display existing messages
for msg in st.session_state["messages"]:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

# User input field
if user_input := st.chat_input("Say something..."):
    # Append user message to state and render it
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Generate real-time assistant response
    with st.chat_message("assistant"):
        try:
            response_stream = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=st.session_state["messages"],
                temperature=0.7,
                stream=True,
            )

            # Stream response onto the screen
            full_response = st.write_stream(
                chunk.choices[0].delta.content
                for chunk in response_stream
                if chunk.choices[0].delta.content is not None
            )

            # Save the full response in history
            st.session_state["messages"].append(
                {"role": "assistant", "content": full_response}
            )

        except Exception as e:
            st.error(f"Error generating response: {e}")