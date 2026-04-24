from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.rag_agent import RAGAgent, log_result
from app.auth.database import authenticate_user, initialize_database
from app.core.config import AUTH_DB_PATH, DEFAULT_LOG_PATH
from app.models.agent import UserInput


def init_session_state() -> None:
    defaults = {
        "authenticated": False,
        "user_name": "",
        "role": "",
        "user_profile": {},
        "agent": RAGAgent(),
        "chat_history": [],
        "latest_error": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    initialize_database(seed=True)


def reset_session() -> None:
    st.session_state.authenticated = False
    st.session_state.user_name = ""
    st.session_state.role = ""
    st.session_state.user_profile = {}
    st.session_state.chat_history = []
    st.session_state.latest_error = ""


def handle_login(username: str, password: str) -> None:
    if not username.strip():
        st.error("Enter a username.")
        return
    user_profile = authenticate_user(username=username.strip(), password=password)
    if user_profile is None:
        st.error("Invalid username or password.")
        return
    st.session_state.authenticated = True
    st.session_state.user_name = user_profile["full_name"]
    st.session_state.role = user_profile["role"]
    st.session_state.user_profile = user_profile
    st.session_state.chat_history = []
    st.session_state.latest_error = ""
    st.rerun()


def render_login() -> None:
    st.title("Banking Support Agent")
    st.caption("RAG retrieval agent with source-aware answers")

    left_col, right_col = st.columns([1.2, 1], gap="large")

    with left_col:
        st.subheader("Login")
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="e.g. customer.asha")
            password = st.text_input("Password", type="password", placeholder="password")
            submitted = st.form_submit_button("Login", use_container_width=True)
        if submitted:
            handle_login(username, password)

    with right_col:
        st.subheader("Demo Accounts")
        st.info(
            "**Customers:** `customer.asha / customer123` and `customer.rahul / customer456`\n\n"
            "**Support:** `support.kiran / support123`\n\n"
            "**Branch Manager:** `branch.raj / branch123`\n\n"
            "**Risk:** `risk.neha / risk123`\n\n"
            "**Admin:** `admin.anita / admin123`"
        )
        st.caption(f"SQLite DB: `{AUTH_DB_PATH.name}`")


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Session")
        st.write(f"**User:** {st.session_state.user_name}")
        st.write(f"**Role:** {st.session_state.role}")

        profile = st.session_state.user_profile
        if profile.get("support_agent_name"):
            st.write(f"Support Agent: {profile['support_agent_name']}")
        if profile.get("branch_manager_name"):
            st.write(f"Branch Manager: {profile['branch_manager_name']}")

        st.divider()
        st.subheader("Try These Queries")
        examples = [
            "What is EMI?",
            "Tell me about loans",
            "Transfer 10000 to this account",
            "My account was hacked",
        ]
        for example in examples:
            st.write(f"- {example}")

        st.divider()
        if st.button("Logout", use_container_width=True):
            reset_session()
            st.rerun()


def render_chat() -> None:
    st.title("Banking Support Agent")
    st.caption("RAG retrieval agent")

    render_sidebar()

    if st.session_state.latest_error:
        st.error(st.session_state.latest_error)

    for item in st.session_state.chat_history:
        with st.chat_message(item["speaker"]):
            st.write(item["text"])
            if item["speaker"] == "assistant" and item.get("sources"):
                st.caption(f"Sources: {item['sources']}")
            if item["speaker"] == "assistant" and item.get("confidence_score"):
                st.caption(f"Confidence Score: {item['confidence_score']}")

    with st.form("chat_form", clear_on_submit=True):
        user_message = st.text_input(
            "Ask a banking question",
            placeholder="e.g. What is EMI?",
        )
        submitted = st.form_submit_button("Send", use_container_width=True)

    if submitted and user_message.strip():
        try:
            request = UserInput(role=st.session_state.role, query=user_message)
            prior_history = [
                {"speaker": history_item["speaker"], "text": history_item["text"]}
                for history_item in st.session_state.chat_history
            ]
            result = st.session_state.agent.run(request, chat_history=prior_history)
            log_result(DEFAULT_LOG_PATH, result)

            st.session_state.chat_history.append({"speaker": "user", "text": user_message})
            st.session_state.chat_history.append(
                {
                    "speaker": "assistant",
                    "text": result.output,
                    "sources": result.metadata.get("sources", ""),
                    "confidence_score": result.metadata.get("confidence_score", ""),
                }
            )
            st.session_state.latest_error = ""
            st.rerun()
        except Exception as exc:
            st.session_state.latest_error = f"Agent error: {exc}"
            st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="Banking Support Agent - RAG",
        page_icon="🏦",
        layout="wide",
    )
    init_session_state()
    if st.session_state.authenticated:
        render_chat()
    else:
        render_login()


if __name__ == "__main__":
    main()
