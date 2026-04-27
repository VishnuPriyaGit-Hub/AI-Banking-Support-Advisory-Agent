from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.langgraph_agent import LangGraphBaselineAgent
from app.agents.rag_agent import log_result
from app.auth.supabase_auth import SupabaseAuthClient
from app.core.config import AUTH_DB_PATH, DEFAULT_LOG_PATH


def init_session_state() -> None:
    defaults = {
        "authenticated": False,
        "user_name": "",
        "role": "",
        "user_profile": {},
        "agent": LangGraphBaselineAgent(),
        "chat_history": [],
        "latest_error": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_session() -> None:
    st.session_state.authenticated = False
    st.session_state.user_name = ""
    st.session_state.role = ""
    st.session_state.user_profile = {}
    st.session_state.chat_history = []
    st.session_state.latest_error = ""


def handle_login(email: str, password: str) -> None:
    if not email.strip():
        st.error("Enter an email.")
        return
    auth_client = SupabaseAuthClient()
    try:
        user_profile = auth_client.sign_in(email=email.strip(), password=password)
    except Exception as exc:
        st.error(f"Login failed: {exc}")
        return
    st.session_state.authenticated = True
    st.session_state.user_name = user_profile.get("customer_name") or user_profile.get("email", "")
    st.session_state.role = user_profile["role"]
    st.session_state.user_profile = user_profile
    st.session_state.chat_history = []
    st.session_state.latest_error = ""
    st.rerun()


def render_login() -> None:
    st.title("Banking Support Agent")
    st.caption("Phase 5 LangGraph agent with tools")

    left_col, right_col = st.columns([1.2, 1], gap="large")

    with left_col:
        st.subheader("Login")
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Email", placeholder="e.g. customer@example.com")
            password = st.text_input("Password", type="password", placeholder="password")
            submitted = st.form_submit_button("Login", use_container_width=True)
        if submitted:
            handle_login(username, password)

    with right_col:
        st.subheader("Supabase Auth")
        st.info("Use your Supabase email and password.")
        st.caption("Authenticated access controls are enforced through Supabase + RLS.")


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Session")
        st.write(f"**User:** {st.session_state.user_name}")
        st.write(f"**Role:** {st.session_state.role}")

        profile = st.session_state.user_profile
        if profile.get("branch"):
            st.write(f"Branch: {profile['branch']}")
        if profile.get("customer_id"):
            st.write(f"Customer ID: {profile['customer_id']}")

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
    st.caption("Phase 5 LangGraph agent")

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
            if item["speaker"] == "assistant" and item.get("tools_used"):
                st.caption(f"Tool Used: {item['tools_used']}")

    with st.form("chat_form", clear_on_submit=True):
        user_message = st.text_input(
            "Ask a banking question",
            placeholder="e.g. What is EMI?",
        )
        submitted = st.form_submit_button("Send", use_container_width=True)

    if submitted and user_message.strip():
        st.session_state.chat_history.append({"speaker": "user", "text": user_message})
        try:
            profile = st.session_state.user_profile
            role_map = {
                "customer": "customer",
                "manager": "manager",
                "branch_manager": "manager",
                "risk": "risk",
                "risk_compliance_officer": "risk",
                "admin": "admin",
                "support": "support",
                "customer_support_agent": "support",
            }
            raw_role = str(st.session_state.role).lower().replace(" & ", " ").replace(" ", "_")
            normalized_role = role_map.get(raw_role, str(profile.get("role", "customer")).lower())
            customer_id = profile.get("customer_id", "")
            branch = profile.get("branch", "")
            result = st.session_state.agent.run(
                user_message,
                role=normalized_role,
                customer_id=customer_id,
                branch=branch,
                auth_user_id=profile.get("id", ""),
                user_jwt=profile.get("access_token", ""),
            )

            st.session_state.chat_history.append(
                {
                    "speaker": "assistant",
                    "text": result.get("response", ""),
                    "tools_used": ", ".join(result.get("tools_used", [])),
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
