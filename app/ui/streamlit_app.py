from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.langgraph_agent import MultiAgentBankingAssistant
from app.auth.supabase_auth import SupabaseAuthClient
from app.mcp.client import LocalMCPClient
from app.memory.store import ConversationMemory


def init_session_state() -> None:
    defaults = {
        "authenticated": False,
        "user_name": "",
        "role": "",
        "user_profile": {},
        "agent": MultiAgentBankingAssistant(),
        "chat_history": [],
        "latest_error": "",
        "show_escalations": False,
        "manager_response_checked": False,
        "pending_manager_responses": [],
        "pending_user_message": "",
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
    st.session_state.manager_response_checked = False
    st.session_state.pending_manager_responses = []
    st.session_state.pending_user_message = ""


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
    st.session_state.manager_response_checked = False
    st.session_state.pending_manager_responses = []
    st.session_state.pending_user_message = ""
    st.rerun()


def render_login() -> None:
    st.title("Banking Support Agent")
    st.caption("One solution for all your banking needs")

    left_col, right_col = st.columns([1.2, 1], gap="large")

    with left_col:
        st.subheader("Login")
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Email", placeholder="e.g. customer@example.com")
            password = st.text_input("Password", type="password", placeholder="password")
            submitted = st.form_submit_button("Login", use_container_width=False)
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
        if st.session_state.role in {"manager", "branch_manager", "risk", "risk_compliance_officer", "admin", "support", "customer_support_agent"}:
            st.session_state.show_escalations = st.toggle(
                "Show escalations",
                value=st.session_state.show_escalations,
            )

        if st.button("Delete my memory", use_container_width=True):
            user_id = profile.get("id") or profile.get("customer_id") or st.session_state.role
            removed = ConversationMemory(user_jwt=profile.get("access_token", "")).delete_user_memory(str(user_id))
            st.success(f"Deleted {removed} local memory record(s).")

        st.divider()
        if st.button("Logout", use_container_width=True):
            reset_session()
            st.rerun()


def render_escalations() -> None:
    profile = st.session_state.user_profile
    role = str(st.session_state.role).lower()
    filters = {}
    if role in {"manager", "branch_manager"} and profile.get("branch"):
        filters = {"branch": profile.get("branch"), "target": "branch_manager"}
    elif role in {"risk", "risk_compliance_officer"}:
        filters = {"target": "risk_team"}
    elif role == "admin":
        filters = {"target": "admin"}
    elif role in {"support", "customer_support_agent"}:
        filters = {"target": "support"}

    raw = LocalMCPClient().call_tool("list_escalations", json.dumps(filters))
    try:
        rows = json.loads(raw)
    except Exception:
        rows = []

    st.subheader("Escalations")
    if not rows:
        st.info("No matching escalations yet.")
        return

    for item in reversed(rows[-20:]):
        with st.container(border=True):
            st.write(f"**Risk:** {item.get('risk_level', '')} | **Status:** {item.get('status', '')}")
            st.write(f"Customer: {item.get('customer_display') or 'Linked customer'}")
            st.caption(f"Action: {item.get('action_type', 'general_review')} | Target: {item.get('target', '')}")
            st.write(item.get("query", ""))
            st.caption(f"{item.get('created_at', '')} | {item.get('reason', '')}")
            if item.get("operation_response"):
                st.success(str(item.get("operation_response", "")))
                st.caption(f"Completed by: {item.get('completed_by', '')}")
            elif item.get("target") == "admin" and role == "admin" and item.get("status") == "approved_pending_action":
                render_admin_operation_form(item)
            elif item.get("target") == "support" and role in {"support", "customer_support_agent"} and item.get("status") == "approved_pending_action":
                render_support_operation_form(item)
            elif item.get("manager_response"):
                st.info(str(item.get("manager_response", "")))
                st.caption(f"Decision by: {item.get('manager_user', '')}")
            elif item.get("target") == "branch_manager" and role in {"manager", "branch_manager", "admin"}:
                render_manager_decision_form(item)


def render_manager_decision_form(item: dict[str, object]) -> None:
    escalation_id = str(item.get("id", ""))
    with st.form(f"manager_decision_{escalation_id}", clear_on_submit=True):
        response = st.text_area(
            "Response to customer",
            placeholder="Write the message the customer should see next time they log in.",
            key=f"response_{escalation_id}",
        )
        approve_col, reject_col, respond_col = st.columns(3)
        approved = approve_col.form_submit_button("Approve", use_container_width=True)
        rejected = reject_col.form_submit_button("Reject", use_container_width=True)
        responded = respond_col.form_submit_button("Respond", use_container_width=True)

    decision = ""
    if approved:
        decision = "approved"
    elif rejected:
        decision = "rejected"
    elif responded:
        decision = "responded"

    if not decision:
        return
    if not response.strip():
        st.error("Add a customer-facing response before submitting.")
        return

    payload = {
        "id": escalation_id,
        "decision": decision,
        "response": response.strip(),
        "manager_user": st.session_state.user_name,
    }
    raw = LocalMCPClient().call_tool("update_escalation", json.dumps(payload))
    result = json.loads(raw)
    if result.get("updated"):
        st.success(f"Escalation {decision}. The customer will see your response on next login.")
        st.rerun()
    else:
        st.error(result.get("error", "Could not update escalation."))


def render_admin_operation_form(item: dict[str, object]) -> None:
    escalation_id = str(item.get("id", ""))
    action_type = str(item.get("action_type", ""))
    st.caption("Use only manager-approved, document-verified details. Form values are sent to Supabase and not stored in the escalation log.")
    with st.form(f"admin_operation_{escalation_id}", clear_on_submit=True):
        operation = st.selectbox(
            "Operation",
            ["add_customer", "delete_customer"],
            index=1 if "delete" in action_type else 0,
            key=f"admin_operation_select_{escalation_id}",
        )
        customer_id = st.text_input("Customer ID", key=f"admin_customer_id_{escalation_id}")
        customer_name = st.text_input("Customer name", key=f"admin_customer_name_{escalation_id}")
        branch = st.text_input("Branch", value=str(item.get("branch", "")), key=f"admin_branch_{escalation_id}")
        city = st.text_input("City", key=f"admin_city_{escalation_id}")
        state = st.text_input("State", key=f"admin_state_{escalation_id}")
        balance = st.number_input("Opening balance", min_value=0.0, step=100.0, key=f"admin_balance_{escalation_id}")
        credit_score = st.number_input("Credit score", min_value=0, max_value=900, step=1, key=f"admin_credit_score_{escalation_id}")
        approval_verified = st.checkbox("Manager approval verified", key=f"admin_approval_verified_{escalation_id}")
        identity_verified = st.checkbox("Identity and required documents verified", key=f"admin_identity_verified_{escalation_id}")
        submitted = st.form_submit_button("Execute admin action", use_container_width=True)

    if not submitted:
        return
    if not approval_verified or not identity_verified:
        st.error("Verify manager approval and required documents before executing.")
        return
    client = LocalMCPClient()
    try:
        if operation == "delete_customer":
            if not customer_id.strip():
                st.error("Customer ID is required for delete.")
                return
            raw_result = client.call_tool("supabase_delete_customer", customer_id.strip())
            customer_message = f"Approved delete request for customer {customer_id.strip()} was processed."
        else:
            required = [customer_id.strip(), customer_name.strip(), branch.strip()]
            if not all(required):
                st.error("Customer ID, customer name, and branch are required for add.")
                return
            payload = {
                "CustomerID": customer_id.strip(),
                "CustomerName": customer_name.strip(),
                "Branch": branch.strip(),
                "City": city.strip(),
                "State": state.strip(),
                "Balance": balance,
                "CreditScore": int(credit_score),
            }
            raw_result = client.call_tool("supabase_add_customer", json.dumps(payload))
            customer_message = f"Approved add request for customer {customer_id.strip()} was processed."
        complete_payload = {
            "id": escalation_id,
            "outcome": "completed",
            "response": customer_message,
            "completed_by": st.session_state.user_name,
        }
        client.call_tool("complete_escalation", json.dumps(complete_payload))
        st.success("Admin action completed.")
        st.rerun()
    except Exception as exc:
        client.call_tool(
            "complete_escalation",
            json.dumps(
                {
                    "id": escalation_id,
                    "outcome": "failed",
                    "response": "Approved admin action could not be completed. Internal staff will review the failure.",
                    "completed_by": st.session_state.user_name,
                }
            ),
        )
        st.error(f"Admin action failed: {exc}")


def render_support_operation_form(item: dict[str, object]) -> None:
    escalation_id = str(item.get("id", ""))
    st.caption("Use only manager-approved, document-verified contact details. Form values are sent to Supabase and not stored in the escalation log.")
    with st.form(f"support_operation_{escalation_id}", clear_on_submit=True):
        customer_id = st.text_input("Customer ID", key=f"support_customer_id_{escalation_id}")
        customer_name = st.text_input("Corrected name", key=f"support_customer_name_{escalation_id}")
        phone = st.text_input("Corrected phone", key=f"support_phone_{escalation_id}")
        address = st.text_area("Corrected address", key=f"support_address_{escalation_id}")
        city = st.text_input("City", key=f"support_city_{escalation_id}")
        state = st.text_input("State", key=f"support_state_{escalation_id}")
        pincode = st.text_input("Pincode", key=f"support_pincode_{escalation_id}")
        approval_verified = st.checkbox("Manager approval verified", key=f"support_approval_verified_{escalation_id}")
        proof_verified = st.checkbox("Customer proof/documentation verified", key=f"support_proof_verified_{escalation_id}")
        submitted = st.form_submit_button("Execute support update", use_container_width=True)

    if not submitted:
        return
    if not approval_verified or not proof_verified:
        st.error("Verify manager approval and customer proof before executing.")
        return
    if not customer_id.strip():
        st.error("Customer ID is required.")
        return
    payload = {"CustomerID": customer_id.strip()}
    if customer_name.strip():
        payload["CustomerName"] = customer_name.strip()
    if phone.strip():
        payload["Phone"] = phone.strip()
    if address.strip():
        payload["Address"] = address.strip()
    if city.strip():
        payload["City"] = city.strip()
    if state.strip():
        payload["State"] = state.strip()
    if pincode.strip():
        payload["Pincode"] = pincode.strip()
    if len(payload) == 1:
        st.error("Enter at least one approved field to update.")
        return

    client = LocalMCPClient()
    try:
        client.call_tool("supabase_update_contact", json.dumps(payload))
        client.call_tool(
            "complete_escalation",
            json.dumps(
                {
                    "id": escalation_id,
                    "outcome": "completed",
                    "response": "Approved profile update was completed by support.",
                    "completed_by": st.session_state.user_name,
                }
            ),
        )
        st.success("Support update completed.")
        st.rerun()
    except Exception as exc:
        client.call_tool(
            "complete_escalation",
            json.dumps(
                {
                    "id": escalation_id,
                    "outcome": "failed",
                    "response": "Approved support update could not be completed. Internal staff will review the failure.",
                    "completed_by": st.session_state.user_name,
                }
            ),
        )
        st.error(f"Support update failed: {exc}")


def load_customer_manager_responses() -> None:
    if st.session_state.manager_response_checked:
        return
    st.session_state.manager_response_checked = True
    profile = st.session_state.user_profile
    role = str(st.session_state.role).lower()
    customer_id = profile.get("customer_id", "")
    if role != "customer" or not customer_id:
        st.session_state.pending_manager_responses = []
        return

    filters = {"customer_id": customer_id, "customer_unread": True}
    raw = LocalMCPClient().call_tool("list_escalations", json.dumps(filters))
    try:
        st.session_state.pending_manager_responses = json.loads(raw)
    except Exception:
        st.session_state.pending_manager_responses = []


def render_customer_manager_responses() -> None:
    load_customer_manager_responses()
    responses = st.session_state.pending_manager_responses
    if not responses:
        return

    def mark_seen(escalation_id: str) -> None:
        LocalMCPClient().call_tool("mark_escalation_seen", escalation_id)
        st.session_state.pending_manager_responses = [
            item for item in st.session_state.pending_manager_responses if str(item.get("id", "")) != escalation_id
        ]

    latest = responses[-1]
    title = f"Branch manager response: {str(latest.get('status', '')).title()}"
    if hasattr(st, "dialog"):
        @st.dialog(title)
        def response_dialog() -> None:
            st.write(latest.get("manager_response", ""))
            if latest.get("operation_response"):
                st.write(latest.get("operation_response", ""))
            st.caption(f"Request: {latest.get('query', '')}")
            if st.button("Got it", use_container_width=True):
                mark_seen(str(latest.get("id", "")))
                st.rerun()

        response_dialog()
    else:
        st.info(f"{title}\n\n{latest.get('operation_response') or latest.get('manager_response', '')}")
        if st.button("Mark manager response as read", use_container_width=True):
            mark_seen(str(latest.get("id", "")))
            st.rerun()


def render_chat() -> None:
    st.title("Banking Support Agent")
    st.caption("LangGraph agent")

    render_sidebar()

    render_customer_manager_responses()

    if st.session_state.show_escalations:
        render_escalations()
        st.divider()

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
            if item["speaker"] == "assistant" and item.get("route"):
                st.caption(f"Route: {item['route']} | Risk: {item.get('risk_level', '')}")

    pending_message = st.session_state.pending_user_message
    user_message = st.chat_input("Ask a banking question", disabled=bool(pending_message))

    if user_message and not pending_message:
        st.session_state.chat_history.append({"speaker": "user", "text": user_message})
        st.session_state.pending_user_message = user_message
        st.rerun()

    if st.session_state.pending_user_message:
        pending_message = st.session_state.pending_user_message
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
            with st.spinner("Processing your request..."):
                result = st.session_state.agent.run(
                    pending_message,
                    role=normalized_role,
                    customer_id=customer_id,
                    branch=branch,
                    auth_user_id=profile.get("id", ""),
                    user_jwt=profile.get("access_token", ""),
                    chat_history=st.session_state.chat_history,
                )

            st.session_state.chat_history.append(
                {
                    "speaker": "assistant",
                    "text": result.get("response", ""),
                    "tools_used": ", ".join(result.get("tools_used", [])),
                    "route": result.get("route", ""),
                    "risk_level": result.get("risk_level", ""),
                    "confidence_score": result.get("confidence_score", 0.0),
                }
            )
            st.session_state.latest_error = ""
            st.session_state.pending_user_message = ""
            st.rerun()
        except Exception as exc:
            st.session_state.latest_error = f"Agent error: {exc}"
            st.session_state.pending_user_message = ""
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
