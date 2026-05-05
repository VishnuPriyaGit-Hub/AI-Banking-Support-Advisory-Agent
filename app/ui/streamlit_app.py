from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.langgraph_agent import MultiAgentBankingAssistant
from app.auth.supabase_auth import SupabaseAuthClient
from app.core.config import EVALUATION_LOG_PATH
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
        "show_support_dashboard": False,
        "manager_response_checked": False,
        "pending_manager_responses": [],
        "pending_user_message": "",
        "feedback_status": "",
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
    st.session_state.show_support_dashboard = False
    st.session_state.manager_response_checked = False
    st.session_state.pending_manager_responses = []
    st.session_state.pending_user_message = ""
    st.session_state.feedback_status = ""


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
    st.session_state.role = user_profile.get("role") or ("customer" if user_profile.get("customer_id") or user_profile.get("customerid") else "")
    st.session_state.user_profile = user_profile
    st.session_state.chat_history = []
    st.session_state.latest_error = ""
    st.session_state.manager_response_checked = False
    st.session_state.pending_manager_responses = []
    st.session_state.pending_user_message = ""
    st.session_state.feedback_status = ""
    st.rerun()


def normalized_session_role() -> str:
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
    profile_role = str(profile.get("role", "") or "").lower()
    if not raw_role and (profile.get("customer_id") or profile.get("customerid")):
        return "customer"
    return role_map.get(raw_role, role_map.get(profile_role, profile_role or "customer"))


def profile_customer_id(profile: dict[str, object] | None = None) -> str:
    profile = profile or st.session_state.user_profile
    customer_id = str(profile.get("customer_id") or profile.get("customerid") or "").strip()
    return "" if customer_id.upper() in {"EMPTY", "NULL", "NONE"} else customer_id


def submit_sidebar_query(query: str) -> None:
    if st.session_state.pending_user_message:
        return
    st.session_state.chat_history.append({"speaker": "user", "text": query})
    st.session_state.pending_user_message = query


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
        customer_id = profile_customer_id(profile)
        if customer_id:
            st.write(f"Customer ID: {customer_id}")

        st.divider()
        memory_user_id = profile.get("id") or customer_id or st.session_state.user_name or st.session_state.role
        recent_queries = ConversationMemory(user_jwt=profile.get("access_token", "")).recent_queries(str(memory_user_id))
        examples = [
            "What is EMI?",
            "Tell me about loans",
            "Transfer 10000 to this account",
            "My account was hacked",
        ]
        if recent_queries:
            st.subheader("Previous Queries")
            for index, query in enumerate(recent_queries):
                st.button(
                    query,
                    key=f"recent_query_{index}",
                    use_container_width=True,
                    disabled=bool(st.session_state.pending_user_message),
                    on_click=submit_sidebar_query,
                    args=(query,),
                )
        else:
            st.subheader("Try Me")
            for index, example in enumerate(examples):
                st.button(
                    example,
                    key=f"try_query_{index}",
                    use_container_width=True,
                    disabled=bool(st.session_state.pending_user_message),
                    on_click=submit_sidebar_query,
                    args=(example,),
                )

        st.divider()
        role = normalized_session_role()
        if role in {"manager", "risk", "admin", "support"}:
            st.session_state.show_escalations = st.toggle(
                "Show escalations",
                value=st.session_state.show_escalations,
            )
        if role == "support":
            st.session_state.show_support_dashboard = st.toggle(
                "Support dashboard",
                value=st.session_state.show_support_dashboard,
            )

        if st.button("Delete my memory", use_container_width=True):
            user_id = profile.get("id") or profile_customer_id(profile) or st.session_state.role
            removed = ConversationMemory(user_jwt=profile.get("access_token", "")).delete_user_memory(str(user_id))
            st.success(f"Deleted {removed} local memory record(s).")

        st.divider()
        if st.button("Logout", use_container_width=True):
            reset_session()
            st.rerun()


def render_escalations() -> None:
    profile = st.session_state.user_profile
    role = normalized_session_role()
    filters = {}
    if role == "manager" and profile.get("branch"):
        filters = {"branch": profile.get("branch"), "target": "branch_manager"}
    elif role == "risk":
        filters = {"target": "risk_team"}
    elif role == "admin":
        filters = {"target": "admin"}
    elif role == "support":
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
            elif item.get("target") == "support" and role == "support" and item.get("status") == "approved_pending_action":
                render_support_operation_form(item)
            elif item.get("manager_response"):
                st.info(str(item.get("manager_response", "")))
                st.caption(f"Decision by: {item.get('manager_user', '')}")
            elif item.get("target") == "branch_manager" and role in {"manager", "admin"}:
                render_manager_decision_form(item)


def render_support_dashboard() -> None:
    if normalized_session_role() != "support":
        st.warning("Support dashboard is available only for support staff.")
        return

    st.subheader("Support Dashboard")
    escalations = load_all_escalations()
    evaluations = load_evaluation_records()

    recent_escalations = filter_recent_escalations(escalations, days=7)
    pending_statuses = {"open", "approved_pending_action"}
    branch_pending = [
        item for item in recent_escalations
        if item.get("target") == "branch_manager" and str(item.get("status", "")).lower() in pending_statuses
    ]
    risk_pending = [
        item for item in recent_escalations
        if item.get("target") == "risk_team" and str(item.get("status", "")).lower() in pending_statuses
    ]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Branch Manager Pending (7d)", len(branch_pending))
    col2.metric("Risk Team Pending (7d)", len(risk_pending))
    col3.metric("Evaluated Responses", len(evaluations))
    col4.metric("Low Score Responses", count_low_score_evaluations(evaluations))

    st.markdown("#### Pending Escalations - Last 7 Days")
    pending_rows = summarize_pending_escalations(branch_pending + risk_pending)
    if pending_rows:
        st.dataframe(pending_rows, use_container_width=True, hide_index=True)
    else:
        st.info("No pending branch manager or risk team escalations.")

    st.markdown("#### Failure Analysis")
    if not evaluations:
        st.info("No evaluation records found yet.")
        return

    avg_score = sum(float(item.get("evaluation_score", 0.0) or 0.0) for item in evaluations) / len(evaluations)
    st.caption(f"Average evaluation score: {avg_score:.3f}")

    metric_failures = aggregate_metric_failures(evaluations)
    if metric_failures:
        st.write("Metric failure counts")
        st.dataframe(metric_failures, use_container_width=True, hide_index=True)

    low_score_rows = summarize_low_score_evaluations(evaluations)
    if low_score_rows:
        st.write("Recent low-scoring responses")
        st.dataframe(low_score_rows, use_container_width=True, hide_index=True)
    else:
        st.success("No low-scoring evaluation records.")


def load_all_escalations() -> list[dict[str, object]]:
    raw = LocalMCPClient().call_tool("list_escalations", json.dumps({}))
    try:
        rows = json.loads(raw)
    except Exception:
        return []
    return rows if isinstance(rows, list) else []


def filter_recent_escalations(rows: list[dict[str, object]], days: int = 7) -> list[dict[str, object]]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    recent: list[dict[str, object]] = []
    for item in rows:
        created_at = parse_iso_datetime(str(item.get("created_at", "")))
        if created_at and created_at >= cutoff:
            recent.append(item)
    return recent


def parse_iso_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def load_evaluation_records(limit: int = 200) -> list[dict[str, object]]:
    if not EVALUATION_LOG_PATH.exists():
        return []
    rows: list[dict[str, object]] = []
    try:
        with EVALUATION_LOG_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict) and payload.get("entry_type") != "metadata":
                    rows.append(payload)
    except OSError:
        return []
    return rows[-limit:]


def count_low_score_evaluations(rows: list[dict[str, object]], threshold: float = 0.75) -> int:
    return sum(1 for item in rows if float(item.get("evaluation_score", 0.0) or 0.0) < threshold)


def summarize_pending_escalations(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for item in sorted(rows, key=lambda row: str(row.get("created_at", "")), reverse=True)[:25]:
        summary.append(
            {
                "created_at": item.get("created_at", ""),
                "target": item.get("target", ""),
                "risk": item.get("risk_level", ""),
                "status": item.get("status", ""),
                "action": item.get("action_type", ""),
                "branch": item.get("branch", ""),
                "customer": item.get("customer_display", "Linked customer"),
            }
        )
    return summary


def aggregate_metric_failures(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    counts: dict[str, int] = {}
    for item in rows:
        metrics = item.get("evaluation_metrics", {})
        if not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            if int(value or 0) == 0:
                counts[str(key)] = counts.get(str(key), 0) + 1
    return [
        {"metric": metric, "failure_count": count}
        for metric, count in sorted(counts.items(), key=lambda pair: pair[1], reverse=True)
    ]


def summarize_low_score_evaluations(rows: list[dict[str, object]], threshold: float = 0.75) -> list[dict[str, object]]:
    low_rows = [
        item for item in rows
        if float(item.get("evaluation_score", 0.0) or 0.0) < threshold
    ]
    summary: list[dict[str, object]] = []
    for item in sorted(low_rows, key=lambda row: str(row.get("timestamp", "")), reverse=True)[:25]:
        summary.append(
            {
                "timestamp": item.get("timestamp", ""),
                "score": item.get("evaluation_score", 0.0),
                "route": item.get("route", ""),
                "risk": item.get("risk_level", ""),
                "reason": item.get("evaluation_reason", ""),
                "query": item.get("query", ""),
            }
        )
    return summary


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
    role = normalized_session_role()
    customer_id = profile_customer_id(profile)
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


def render_feedback_widget(item: dict[str, object], index: int, previous_user_query: str) -> None:
    if item.get("feedback_saved"):
        st.caption("Feedback saved for future interactions.")
        return

    with st.expander("Give feedback", expanded=False):
        with st.form(f"feedback_form_{index}", clear_on_submit=True):
            rating = st.radio(
                "Was this helpful?",
                ["helpful", "not_helpful"],
                format_func=lambda value: "Helpful" if value == "helpful" else "Not helpful",
                horizontal=True,
                key=f"feedback_rating_{index}",
            )
            tags = st.multiselect(
                "What should change next time?",
                [
                    "too technical",
                    "too simple",
                    "too vague",
                    "too long",
                    "too short",
                    "need steps",
                    "need calculation",
                    "wrong route",
                ],
                key=f"feedback_tags_{index}",
            )
            comment = st.text_area(
                "Optional comment",
                placeholder="Example: Explain this in simpler terms next time.",
                key=f"feedback_comment_{index}",
            )
            submitted = st.form_submit_button("Save feedback", use_container_width=True)

    if not submitted:
        return

    profile = st.session_state.user_profile
    user_id = profile.get("id") or profile_customer_id(profile) or st.session_state.user_name or normalized_session_role()
    memory = ConversationMemory(user_jwt=profile.get("access_token", ""))
    record = memory.save_feedback(
        user_id=str(user_id),
        role=normalized_session_role(),
        query=previous_user_query,
        response=str(item.get("text", "")),
        route=str(item.get("route", "")),
        risk_level=str(item.get("risk_level", "")),
        rating=rating,
        tags=tags,
        comment=comment,
    )
    item["feedback_saved"] = True
    item["preference_summary"] = record.get("preference_summary", "")
    st.session_state.feedback_status = "Feedback saved. Future answers will use this preference where it is safe to do so."
    st.rerun()


def render_chat() -> None:
    st.title("Banking Support Agent")
    st.caption("LangGraph agent")

    render_sidebar()

    render_customer_manager_responses()

    if st.session_state.show_escalations:
        render_escalations()
        st.divider()

    if st.session_state.show_support_dashboard:
        render_support_dashboard()
        st.divider()

    if st.session_state.latest_error:
        st.error(st.session_state.latest_error)
    if st.session_state.feedback_status:
        st.success(st.session_state.feedback_status)

    last_user_query = ""
    for index, item in enumerate(st.session_state.chat_history):
        with st.chat_message(item["speaker"]):
            st.write(item["text"])
            if item["speaker"] == "user":
                last_user_query = str(item.get("text", ""))
            if item["speaker"] == "assistant" and item.get("sources"):
                st.caption(f"Sources: {item['sources']}")
            if item["speaker"] == "assistant" and item.get("confidence_score"):
                st.caption(f"Confidence Score: {item['confidence_score']}")
            if item["speaker"] == "assistant" and item.get("evaluation_score") is not None:
                st.caption(f"Evaluation Score: {item.get('evaluation_score', 0.0)}")
                metrics = item.get("evaluation_metrics") or {}
                if isinstance(metrics, dict) and metrics:
                    metric_text = " | ".join(f"{key}: {value}" for key, value in metrics.items())
                    st.caption(f"Evaluation Metrics: {metric_text}")
            if item["speaker"] == "assistant" and item.get("tools_used"):
                st.caption(f"Tool Used: {item['tools_used']}")
            if item["speaker"] == "assistant" and item.get("route"):
                st.caption(f"Route: {item['route']} | Risk: {item.get('risk_level', '')}")
            if item["speaker"] == "assistant" and item.get("adaptation_note"):
                st.caption(str(item["adaptation_note"]))
            if item["speaker"] == "assistant":
                render_feedback_widget(item, index, last_user_query)

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
            normalized_role = normalized_session_role()
            customer_id = profile_customer_id(profile)
            branch = profile.get("branch", "")
            memory_user_id = profile.get("id") or customer_id or st.session_state.user_name or normalized_role
            behavior_preferences = ConversationMemory(user_jwt=profile.get("access_token", "")).behavior_preferences(str(memory_user_id))
            previous_chat_history = st.session_state.chat_history
            if (
                previous_chat_history
                and previous_chat_history[-1].get("speaker") == "user"
                and previous_chat_history[-1].get("text") == pending_message
            ):
                previous_chat_history = previous_chat_history[:-1]
            with st.spinner("Processing your request..."):
                result = st.session_state.agent.run(
                    pending_message,
                    role=normalized_role,
                    customer_id=customer_id,
                    branch=branch,
                    auth_user_id=profile.get("id", ""),
                    user_jwt=profile.get("access_token", ""),
                    chat_history=previous_chat_history,
                    behavior_preferences=behavior_preferences,
                )

            st.session_state.chat_history.append(
                {
                    "speaker": "assistant",
                    "text": result.get("response", ""),
                    "tools_used": ", ".join(result.get("tools_used", [])),
                    "route": result.get("route", ""),
                    "risk_level": result.get("risk_level", ""),
                    "confidence_score": result.get("confidence_score", 0.0),
                    "evaluation_score": result.get("evaluation_score", 0.0),
                    "evaluation_metrics": result.get("evaluation_metrics", {}),
                    "evaluation_reason": result.get("evaluation_reason", ""),
                    "adaptation_note": result.get("adaptation_note", ""),
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
