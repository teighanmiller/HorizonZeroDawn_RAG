"""
Main UI file for the Horizon RAG project
"""

import time
import streamlit as st
from src.rag_chat import Chat

COLLECTION_NAME = "horizon_rag"


@st.cache_resource
def load_chat():
    """
    Preloads a instance of a chat to remove waiting times and save session memory.

    Returns:
        Chat: instance of Chat
    """
    return Chat(COLLECTION_NAME)


st.set_page_config(page_title="Horizon Game series ChatBot")
st.write("Welcome to Gaia, a RAG Chatbot for Horizon Zero Dawn lore and game content.")

chat = load_chat()


def save_feedback(index):
    """
    Callback: copy widget value into session history and store rating.
    `index` is the message index in st.session_state.messages.
    """
    key = f"feedback_{index}"
    val = st.session_state.get(
        key
    )  # st.feedback returns int or None (0 = down, 1 = up)
    # save into message history so the widget can be disabled next time
    st.session_state.messages[index]["feedback"] = val

    # Only act on a real selection
    if val is not None:
        # call your chat storage method (write_rating / store_rating)
        try:
            chat.store_rating(int(val))
        except ValueError:
            chat.store_rating(int(-1))


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask a question about the Horizon Game series! 👇",
        }
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        FULL_RESPONSE = ""

        def update_progress(msg: str):
            """
            Callback function to update UI as processes run.

            Args:
                msg (str): Message to display in UI.
            """
            message_placeholder.markdown(msg)

        assistant_response = chat.get_response(
            prompt, progress_callback=update_progress
        )

        # Simulate stream of response with milliseconds delay
        for char in assistant_response:
            FULL_RESPONSE += char
            time.sleep(0.01)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(FULL_RESPONSE + "▌")
        message_placeholder.markdown(FULL_RESPONSE)

    # --- Feedback section ---
    if "last_rating" not in st.session_state:
        st.session_state.last_rating = None

    def handle_feedback():
        """
        Handles user feedback
        """
        selected = st.session_state["feedback_value"]
        if selected is not None:
            chat.store_rating(int(selected))  # 0 or 1
            st.session_state.last_rating = selected
            st.toast("Thanks for your feedback! ✅")  # optional visual confirmation

    st.radio(
        "Was this answer helpful?",
        options=[1, 0],
        format_func=lambda x: "👍 Yes" if x == 1 else "👎 No",
        key="feedback_value",
        on_change=handle_feedback,
    )

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": FULL_RESPONSE})
