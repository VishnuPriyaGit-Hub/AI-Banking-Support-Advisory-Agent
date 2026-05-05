from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from uuid import uuid4

from app.core.config import LOG_DIR
from app.security.pii import hash_identifier, mask_identifier, redact_text


ESCALATION_PATH = LOG_DIR / "escalations.jsonl"


def create_escalation_tool(payload_json: str) -> str:
    payload = json.loads(payload_json)
    risk_level = str(payload.get("risk_level", "")).lower()
    customer_id = str(payload.get("customer_id", ""))
    action_type = _infer_action_type(str(payload.get("query", "")))
    action_target = _infer_action_target(action_type)
    role = str(payload.get("role", "")).lower()
    target = _infer_initial_target(risk_level)
    record = {
        "id": f"ESC-{uuid4().hex[:12]}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "target": target,
        "status": "open",
        "risk_level": risk_level or "medium",
        "route": payload.get("route", "escalation"),
        "action_type": action_type,
        "action_target": action_target,
        "customer_ref": hash_identifier(customer_id),
        "customer_display": mask_identifier(customer_id),
        "branch": payload.get("branch", ""),
        "role": role,
        "query": redact_text(str(payload.get("query", ""))),
        "reason": redact_text(str(payload.get("reason", ""))),
        "manager_response": "",
        "manager_user": "",
        "customer_viewed_at": "",
    }
    ESCALATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ESCALATION_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    return json.dumps(
        {
            "status": "notified",
            "id": record["id"],
            "target": target,
            "risk_level": record["risk_level"],
            "action_type": action_type,
            "action_target": action_target,
        }
    )


def list_escalations_tool(filter_json: str = "") -> str:
    filters = json.loads(filter_json) if filter_json.strip() else {}
    rows = _read_escalations()
    if not rows:
        return json.dumps([])
    filtered_rows: list[dict[str, object]] = []
    for row in rows:
        if _matches_filters(row, filters):
            filtered_rows.append(row)
    return json.dumps(filtered_rows[-50:], indent=2)


def update_escalation_tool(payload_json: str) -> str:
    payload = json.loads(payload_json)
    escalation_id = str(payload.get("id", ""))
    decision = str(payload.get("decision", "")).lower()
    response = redact_text(str(payload.get("response", "")).strip())
    manager_user = redact_text(str(payload.get("manager_user", "")).strip())
    if decision not in {"approved", "rejected", "responded"}:
        return json.dumps({"updated": False, "error": "Decision must be approved, rejected, or responded."})
    if not response:
        return json.dumps({"updated": False, "error": "Manager response is required."})

    rows = _read_escalations()
    updated = False
    now = datetime.now(timezone.utc).isoformat()
    for row in rows:
        if str(row.get("id", "")) != escalation_id:
            continue
        row["status"] = decision
        if decision == "approved" and row.get("action_target") in {"admin", "support"}:
            row["status"] = "approved_pending_action"
            row["target"] = row["action_target"]
        row["manager_response"] = response
        row["manager_user"] = manager_user
        row["updated_at"] = now
        row["manager_decision_at"] = now
        row["customer_viewed_at"] = ""
        updated = True
        break
    if updated:
        _write_escalations(rows)
    status = ""
    if updated:
        status = str(next((row.get("status", "") for row in rows if str(row.get("id", "")) == escalation_id), ""))
    return json.dumps({"updated": updated, "id": escalation_id, "status": status or decision})


def complete_escalation_tool(payload_json: str) -> str:
    payload = json.loads(payload_json)
    escalation_id = str(payload.get("id", ""))
    outcome = str(payload.get("outcome", "")).lower()
    response = redact_text(str(payload.get("response", "")).strip())
    completed_by = redact_text(str(payload.get("completed_by", "")).strip())
    if outcome not in {"completed", "failed"}:
        return json.dumps({"updated": False, "error": "Outcome must be completed or failed."})
    if not response:
        return json.dumps({"updated": False, "error": "Customer-facing result is required."})

    rows = _read_escalations()
    updated = False
    now = datetime.now(timezone.utc).isoformat()
    for row in rows:
        if str(row.get("id", "")) != escalation_id:
            continue
        row["status"] = outcome
        row["operation_response"] = response
        row["completed_by"] = completed_by
        row["completed_at"] = now
        row["updated_at"] = now
        row["customer_viewed_at"] = ""
        updated = True
        break
    if updated:
        _write_escalations(rows)
    return json.dumps({"updated": updated, "id": escalation_id, "status": outcome})


def mark_escalation_seen_tool(escalation_id: str) -> str:
    rows = _read_escalations()
    updated = False
    now = datetime.now(timezone.utc).isoformat()
    for row in rows:
        if str(row.get("id", "")) != escalation_id:
            continue
        row["customer_viewed_at"] = now
        row["updated_at"] = now
        updated = True
        break
    if updated:
        _write_escalations(rows)
    return json.dumps({"updated": updated, "id": escalation_id})


def _read_escalations() -> list[dict[str, object]]:
    if not ESCALATION_PATH.exists():
        return []
    rows: list[dict[str, object]] = []
    for line in ESCALATION_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "id" not in row:
            row["id"] = f"LEGACY-{hashlib.sha1(line.encode('utf-8')).hexdigest()[:12]}"
        if "customer_ref" not in row and row.get("customer_id"):
            raw_customer_id = str(row.get("customer_id", ""))
            row["customer_ref"] = hash_identifier(raw_customer_id)
            row["customer_display"] = mask_identifier(raw_customer_id)
            row.pop("customer_id", None)
        row.setdefault("status", "open")
        inferred_action_type = _infer_action_type(str(row.get("query", "")))
        if not row.get("action_type") or (row.get("action_type") == "general_review" and inferred_action_type != "general_review"):
            row["action_type"] = inferred_action_type
        row["action_target"] = _infer_action_target(str(row.get("action_type", "general_review")))
        row.setdefault("manager_response", "")
        row.setdefault("manager_user", "")
        row.setdefault("operation_response", "")
        row.setdefault("completed_by", "")
        row.setdefault("customer_viewed_at", "")
        rows.append(row)
    return rows


def _write_escalations(rows: list[dict[str, object]]) -> None:
    ESCALATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ESCALATION_PATH.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _matches_filters(row: dict[str, object], filters: dict[str, object]) -> bool:
    branch = str(filters.get("branch", ""))
    target = str(filters.get("target", ""))
    customer_id = str(filters.get("customer_id", ""))
    customer_ref = hash_identifier(customer_id) if customer_id else ""
    status = str(filters.get("status", ""))
    customer_unread = bool(filters.get("customer_unread", False))
    if branch and row.get("branch") != branch:
        return False
    if target and row.get("target") != target:
        return False
    if customer_ref and row.get("customer_ref") != customer_ref:
        return False
    if status and row.get("status") != status:
        return False
    if customer_unread and (not (row.get("manager_response") or row.get("operation_response")) or row.get("customer_viewed_at")):
        return False
    return True


def _infer_action_type(query: str) -> str:
    lowered = query.lower()
    if any(token in lowered for token in ["hacked", "fraud", "unauthorized", "stolen", "phishing", "otp"]):
        return "fraud_or_security_review"
    if any(token in lowered for token in ["add customer", "create customer", "new customer", "add account", "create account", "open account"]):
        return "add_customer_or_account"
    if any(token in lowered for token in ["delete customer", "remove customer", "delete account", "close account"]):
        return "delete_customer_or_account"
    if any(token in lowered for token in ["change name", "update name", "change phone", "update phone", "phone number", "mobile number", "contact number", "change address", "update address", "pincode", "pin code", "postal code"]):
        return "update_contact"
    return "general_review"


def _infer_action_target(action_type: str) -> str:
    if action_type == "fraud_or_security_review":
        return "risk_team"
    if action_type in {"add_customer_or_account", "delete_customer_or_account"}:
        return "admin"
    if action_type == "update_contact":
        return "support"
    return "branch_manager"


def _infer_initial_target(risk_level: str) -> str:
    if risk_level == "high":
        return "risk_team"
    return "branch_manager"
