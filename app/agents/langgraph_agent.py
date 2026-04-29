from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypedDict

try:
    from langchain_openai import ChatOpenAI
except ModuleNotFoundError:
    ChatOpenAI = None  # type: ignore[assignment]

from app.core.config import DEFAULT_LOG_PATH, get_env_value
from app.core.prompts import (
    load_phase6_calculation_prompt,
    load_phase6_planner_prompt,
    load_phase6_response_prompt,
    load_phase6_rewrite_prompt,
    load_phase6_system_prompt,
)
from app.mcp.client import LocalMCPClient
from app.memory.store import ConversationMemory
from app.security.pii import (
    contains_ambiguous_action_request,
    contains_legal_advice_request,
    contains_secret_request,
    redact_json_text,
    redact_mapping,
    redact_text,
    redact_value,
)
from app.tools.supabase_tool import SupabaseTool


Route = Literal["general", "personalized", "calculation", "escalation"]
RiskLevel = Literal["low", "medium", "high"]


class BankingState(TypedDict, total=False):
    user_query: str
    user_metadata: dict[str, Any]
    chat_history: list[dict[str, str]]
    standalone_query: str
    route: Route
    required_tools: list[str]
    risk_level: RiskLevel
    risk_reason: str
    tool_outputs: dict[str, str]
    confidence_score: float
    final_response: str
    blocked: bool
    escalated_to: str
    logs: list[dict[str, Any]]


class MultiAgentBankingAssistant:
    """LangGraph banking assistant with planner, risk, MCP tool, and response nodes."""

    def __init__(self, *, log_path: Path = DEFAULT_LOG_PATH) -> None:
        self.model_name = get_env_value("OPENAI_MODEL") or "gpt-4o-mini"
        self.base_url = get_env_value("LLM_BASE_URL") or get_env_value("OPENAI_BASE_URL")
        self.api_key = get_env_value("LLM_API_KEY") or get_env_value("OPENAI_API_KEY")
        self.llm = self._build_llm()
        self.system_prompt = load_phase6_system_prompt()
        self.planner_prompt = load_phase6_planner_prompt()
        self.rewrite_prompt = load_phase6_rewrite_prompt()
        self.calculation_prompt = load_phase6_calculation_prompt()
        self.response_prompt = load_phase6_response_prompt()
        self.mcp_client = LocalMCPClient()
        self.log_path = log_path
        self.graph = self._build_graph()

    def run(
        self,
        user_query: str,
        *,
        role: str,
        customer_id: str | None = None,
        branch: str | None = None,
        auth_user_id: str | None = None,
        user_jwt: str | None = None,
        chat_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        role = self.normalize_role(role)
        metadata = {
            "role": role,
            "customer_id": customer_id or "",
            "branch": branch or "",
            "auth_user_id": auth_user_id or "",
            "user_jwt": user_jwt or "",
        }
        memory = ConversationMemory(user_jwt=user_jwt)
        memory.prune_inactive(days=60)
        state: BankingState = {
            "user_query": user_query,
            "user_metadata": metadata,
            "chat_history": memory.short_context(chat_history),
            "tool_outputs": {},
            "logs": [],
            "confidence_score": 0.0,
        }
        if self.graph is not None:
            final_state = self.graph.invoke(state)
        else:
            final_state = self._run_without_langgraph(state)

        memory.save_turn(
            user_id=auth_user_id or customer_id or role,
            role=role,
            query=redact_text(user_query),
            response=self._storage_safe_response(final_state),
            route=final_state.get("route", "general"),
            risk_level=final_state.get("risk_level", "low"),
        )
        self._write_audit_log(final_state)
        return {
            "agent": "MultiAgentBankingAssistant",
            "model": self.model_name,
            "context": {key: value for key, value in metadata.items() if key != "user_jwt"},
            "route": final_state.get("route", ""),
            "risk_level": final_state.get("risk_level", ""),
            "risk_reason": final_state.get("risk_reason", ""),
            "confidence_score": final_state.get("confidence_score", 0.0),
            "tools_used": list(final_state.get("tool_outputs", {}).keys()),
            "required_tools": final_state.get("required_tools", []),
            "escalated_to": final_state.get("escalated_to", ""),
            "response": final_state.get("final_response", ""),
            "logs": final_state.get("logs", []),
        }

    def _build_llm(self) -> ChatOpenAI | None:
        if not self.api_key or ChatOpenAI is None:
            return None
        return ChatOpenAI(model=self.model_name, temperature=0, api_key=self.api_key, base_url=self.base_url)

    def _build_graph(self):
        try:
            from langgraph.graph import END, StateGraph

            graph = StateGraph(BankingState)
            graph.add_node("planner", self.planner_agent)
            graph.add_node("risk_precheck", self.risk_agent)
            graph.add_node("mcp_tools", self.mcp_tool_node)
            graph.add_node("risk_postcheck", self.risk_agent)
            graph.add_node("response_generation", self.response_generation_agent)
            graph.set_entry_point("planner")
            graph.add_edge("planner", "risk_precheck")
            graph.add_conditional_edges(
                "risk_precheck",
                self._after_risk,
                {"blocked": "response_generation", "tools": "mcp_tools"},
            )
            graph.add_edge("mcp_tools", "risk_postcheck")
            graph.add_edge("risk_postcheck", "response_generation")
            graph.add_edge("response_generation", END)
            return graph.compile()
        except Exception:
            return None

    def _run_without_langgraph(self, state: BankingState) -> BankingState:
        state = self.planner_agent(state)
        state = self.risk_agent(state)
        if not state.get("blocked") and state.get("risk_level") != "medium":
            state = self.mcp_tool_node(state)
            state = self.risk_agent(state)
        return self.response_generation_agent(state)

    def planner_agent(self, state: BankingState) -> BankingState:
        query = state["user_query"]
        metadata = state.get("user_metadata", {})
        history = state.get("chat_history", [])
        deterministic_route = self._classify_route(query, history)
        llm_plan = self._llm_planner_decision(query, history, metadata)
        route = self._validated_planner_route(deterministic_route, llm_plan.get("route"), query, history)
        standalone_query = self._rewrite_query(query, state.get("chat_history", []), metadata)
        required_tools = self._select_tools(route, query)
        state.update({"standalone_query": standalone_query, "route": route, "required_tools": required_tools})
        self._add_step_log(
            state,
            "planner",
            route=route,
            tools=required_tools,
            planner_mode="llm_validated" if llm_plan else "deterministic",
            planner_reason=redact_text(str(llm_plan.get("reason", "")))[:240] if llm_plan else "",
            confidence_score=float(llm_plan.get("confidence", state.get("confidence_score", 0.0)) or 0.0) if llm_plan else state.get("confidence_score", 0.0),
        )
        return state

    def risk_agent(self, state: BankingState) -> BankingState:
        risk_level, reason = self._classify_risk(state)
        state["risk_level"] = risk_level
        state["risk_reason"] = reason
        already_escalated = bool(state.get("escalated_to"))
        if risk_level == "medium" and not already_escalated:
            state["escalated_to"] = "branch_manager"
            state.setdefault("tool_outputs", {})["create_escalation"] = self._create_escalation(state)
        elif risk_level == "high" and not already_escalated:
            state["blocked"] = True
            state["escalated_to"] = "risk_team"
            state.setdefault("tool_outputs", {})["create_escalation"] = self._create_escalation(state)
        elif risk_level == "high":
            state["blocked"] = True
        self._add_step_log(state, "risk_compliance", risk_level=risk_level, reason=reason)
        return state

    def mcp_tool_node(self, state: BankingState) -> BankingState:
        outputs = dict(state.get("tool_outputs", {}))
        route = state.get("route", "general")
        query = state.get("standalone_query") or state["user_query"]
        if route == "general":
            rag_output = self._safe_tool_call("rag_retrieval", query)
            outputs["rag_retrieval"] = rag_output
            confidence = self._extract_confidence(rag_output)
            state["confidence_score"] = confidence
            if confidence <= 0.75:
                outputs["search_api"] = self._safe_tool_call("search_api", query)
        elif route == "personalized":
            outputs["db_tool"] = self._fetch_personalized_data(state)
            state["confidence_score"] = 1.0 if self._db_tool_succeeded(outputs["db_tool"]) else 0.0
            if self._is_personalized_guidance_query(state):
                rag_query = self._build_guidance_query(state, outputs["db_tool"])
                outputs["rag_retrieval"] = self._safe_tool_call("rag_retrieval", rag_query)
                rag_confidence = self._extract_confidence(outputs["rag_retrieval"])
                state["confidence_score"] = max(float(state.get("confidence_score", 0.0) or 0.0), rag_confidence)
                if rag_confidence <= 0.75:
                    outputs["search_api"] = self._safe_tool_call("search_api", rag_query)
        elif route == "calculation":
            context = self._fetch_calculation_context(state)
            outputs.update(context)
            outputs["calculator"] = self._safe_tool_call("calculator", self._build_calculation_payload(state, context))
        elif route == "escalation":
            outputs["create_escalation"] = self._create_escalation(state)
        state["tool_outputs"] = outputs
        for tool_name in outputs:
            self._add_step_log(state, "mcp_tool_call", tool_used=tool_name, confidence_score=state.get("confidence_score", 0.0))
        return state

    def response_generation_agent(self, state: BankingState) -> BankingState:
        if state.get("blocked"):
            response = self._blocked_response(state)
        elif state.get("risk_level") == "medium":
            response = "I have escalated this request to your branch manager. They will review it and follow up with you."
        else:
            response = self._generate_response(state)
        state["final_response"] = response
        self._add_step_log(state, "response_generation", final_response=self._storage_safe_response(state))
        return state

    def _after_risk(self, state: BankingState) -> str:
        return "blocked" if state.get("blocked") or state.get("risk_level") == "medium" else "tools"

    def _llm_planner_decision(self, query: str, history: list[dict[str, str]], metadata: dict[str, Any]) -> dict[str, Any]:
        if not self.llm:
            return {}
        safe_history = [
            {
                "speaker": item.get("speaker", "user"),
                "text": redact_text(str(item.get("text", ""))),
            }
            for item in history[-6:]
        ]
        safe_metadata = {
            "role": metadata.get("role", ""),
            "branch_present": bool(metadata.get("branch")),
            "customer_profile_present": bool(metadata.get("customer_id")),
        }
        prompt = (
            f"{self.planner_prompt}\n\n"
            f"User metadata: {json.dumps(safe_metadata)}\n"
            f"Recent chat history: {json.dumps(safe_history)}\n"
            f"Latest query: {redact_text(query)}"
        )
        try:
            result = self.llm.invoke([("user", prompt)])
            return self._parse_planner_json(str(result.content))
        except Exception:
            return {}

    def _parse_planner_json(self, raw_text: str) -> dict[str, Any]:
        cleaned = raw_text.replace("```json", "").replace("```", "").strip()
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if not match:
                return {}
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}
        if not isinstance(payload, dict):
            return {}
        route = str(payload.get("route", "")).strip().lower()
        if route not in {"general", "personalized", "calculation", "escalation"}:
            return {}
        tools = payload.get("required_tools", [])
        if not isinstance(tools, list):
            tools = []
        allowed_tools = {
            "rag_retrieval",
            "db_tool",
            "calculator",
            "search_api_if_rag_confidence_low",
            "create_escalation",
        }
        confidence = payload.get("confidence", 0.0)
        try:
            confidence_value = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence_value = 0.0
        return {
            "route": route,
            "required_tools": [str(tool) for tool in tools if str(tool) in allowed_tools],
            "reason": redact_text(str(payload.get("reason", ""))),
            "confidence": confidence_value,
        }

    def _validated_planner_route(
        self,
        deterministic_route: Route,
        llm_route: Any,
        query: str,
        history: list[dict[str, str]],
    ) -> Route:
        if deterministic_route in {"escalation", "personalized", "calculation"}:
            return deterministic_route
        if not isinstance(llm_route, str) or llm_route not in {"general", "personalized", "calculation", "escalation"}:
            return deterministic_route
        if llm_route == "calculation" and not re.search(r"\d", query):
            return "general"
        if llm_route == "personalized":
            q = query.lower()
            if any(token in q for token in ["my", "customer", "account", "loan", "transaction", "branch"]):
                return "personalized"
            if self._history_mentions_loan(history):
                return "personalized"
            return "general"
        return llm_route

    def _classify_route(self, query: str, chat_history: list[dict[str, str]] | None = None) -> Route:
        q = query.lower()
        if any(
            token in q
            for token in [
                "hacked",
                "fraud",
                "otp",
                "unauthorized",
                "stolen",
                "phishing",
                "money deducted",
                "transfer",
                "send money",
                "withdraw",
                "approve loan",
            ]
        ):
            return "escalation"
        has_number = bool(re.search(r"\d", q))
        if (
            "calculate" in q
            or "eligibility" in q
            or "maturity" in q
            or "balance summary" in q
            or ("emi" in q and has_number)
            or ("interest" in q and has_number)
        ):
            return "calculation"
        personalized_tokens = [
            "my balance",
            "my account",
            "my loan",
            "i have a loan",
            "i have a home loan",
            "repayment options",
            "my repayment",
            "my transaction",
            "transactions",
            "account balance",
            "in my branch",
            "my branch",
            "branch customers",
            "who all",
            "who have",
            "opted for loan",
            "taken loan",
            "loan customers",
            "customer details",
            "show customer",
            "customer id",
            "customer profile",
        ]
        loan_follow_up_tokens = [
            "pay extra",
            "extra payment",
            "extra every month",
            "prepay",
            "prepayment",
            "part payment",
            "part-payment",
            "foreclose",
            "foreclosure",
            "reduce tenure",
            "reduce emi",
        ]
        if any(token in q for token in loan_follow_up_tokens) and self._history_mentions_loan(chat_history):
            return "personalized"
        if any(token in q for token in personalized_tokens):
            return "personalized"
        return "general"

    def _select_tools(self, route: Route, query: str) -> list[str]:
        if route == "general":
            return ["rag_retrieval", "search_api_if_rag_confidence_low"]
        if route == "personalized":
            return ["db_tool"]
        if route == "calculation":
            tools = ["calculator"]
            if any(token in query.lower() for token in ["my", "balance", "my loan", "my emi", "transaction"]):
                tools.insert(0, "db_tool")
            else:
                tools.insert(0, "rag_retrieval")
            return tools
        return ["create_escalation"]

    def _classify_risk(self, state: BankingState) -> tuple[RiskLevel, str]:
        query = state["user_query"].lower()
        metadata = state.get("user_metadata", {})
        role = metadata.get("role", "")
        route = state.get("route", "general")
        if any(token in query for token in ["add customer", "create customer", "new customer", "delete customer", "remove customer", "update customer", "change phone", "update phone", "change address", "update address", "change name", "update name", "pincode", "pin code"]):
            return "medium", "Customer/account maintenance request requires branch manager approval before staff action."
        if redact_text(query) != query:
            return "high", "Request contains sensitive personal or financial identifiers."
        if any(token in query for token in ["transfer", "send money", "withdraw", "approve loan", "share otp"]):
            return "high", "Transactional, destructive, or credential-sharing request."
        if contains_secret_request(query):
            return "high", "Request attempts to access sensitive credentials or personal identifiers."
        if contains_legal_advice_request(query):
            return "high", "Legal advice request is outside assistant policy."
        if any(token in query for token in ["hacked", "fraud", "unauthorized", "stolen", "phishing", "otp"]):
            return "high", "Potential fraud or account compromise."
        if role == "customer" and any(token in query for token in ["all customers", "another customer", "other customer", "branch customers"]):
            return "high", "Customer attempted to access data outside their authorization scope."
        if route == "personalized" and role == "customer" and not metadata.get("customer_id"):
            return "high", "No linked customer profile is available for personalized data access."
        if any(token in query for token in ["complaint", "dispute", "chargeback", "failed transaction", "deducted but"]) or contains_ambiguous_action_request(query):
            return "medium", "Ambiguous or support-sensitive request requires human review."
        return "low", "Allowed request within role and policy boundaries."

    def _rewrite_query(self, query: str, history: list[dict[str, str]], metadata: dict[str, Any]) -> str:
        if not self.llm or not history:
            return query
        history_text = "\n".join(f"{item.get('speaker', 'user')}: {redact_text(item.get('text', ''))}" for item in history[-6:])
        prompt = (
            f"{self.rewrite_prompt}\n\n"
            f"Role: {metadata.get('role', '')}\nChat history:\n{history_text}\n\nLatest query: {redact_text(query)}"
        )
        try:
            result = self.llm.invoke([("user", prompt)])
            return str(result.content).strip() or query
        except Exception:
            return query

    def _fetch_personalized_data(self, state: BankingState) -> str:
        metadata = state.get("user_metadata", {})
        role = metadata.get("role", "")
        jwt = metadata.get("user_jwt") or None
        try:
            client = SupabaseTool(user_jwt=jwt)
            if role == "customer":
                return json.dumps(client.get_customer_snapshot(metadata.get("customer_id", "")), indent=2)
            if role == "manager":
                query = state.get("user_query", "").lower()
                if any(token in query for token in ["loan", "who all", "who have", "opted", "taken"]):
                    return json.dumps(client.get_branch_loan_customers(metadata.get("branch", "")), indent=2)
                return json.dumps(client.get_branch_customers(metadata.get("branch", "")), indent=2)
            if role in {"admin", "support", "risk"}:
                requested_customer_id = self._extract_customer_id(state.get("user_query", ""))
                if requested_customer_id:
                    return json.dumps(client.get_customer_snapshot(requested_customer_id), indent=2)
                return json.dumps(client.get_all_customers(), indent=2)
            return "DB access denied for this role."
        except Exception as exc:
            return f"DB tool failed: {exc}"

    def _fetch_calculation_context(self, state: BankingState) -> dict[str, str]:
        query = state["user_query"].lower()
        if any(token in query for token in ["my", "balance", "loan", "transaction"]):
            return {"db_tool": self._fetch_personalized_data(state)}
        return {"rag_retrieval": self._safe_tool_call("rag_retrieval", state.get("standalone_query") or state["user_query"])}

    def _build_calculation_payload(self, state: BankingState, context: dict[str, str]) -> str:
        query = state["user_query"]
        emi_match = re.search(r"(\d[\d,]*)\D+(\d+(?:\.\d+)?)\s*%?\D+(\d+)\s*(year|years|month|months)", query.lower())
        if "emi" in query.lower() and emi_match:
            principal = float(emi_match.group(1).replace(",", ""))
            rate = float(emi_match.group(2))
            tenure = int(emi_match.group(3)) * (12 if emi_match.group(4).startswith("year") else 1)
            return json.dumps({"operation": "emi", "principal": principal, "annual_rate": rate, "tenure_months": tenure})
        if self.llm:
            prompt = (
                f"{self.calculation_prompt}\n\n"
                f"Query: {redact_text(query)}\nContext: {json.dumps(self._redact_tool_context_for_llm(context))[:4000]}"
            )
            try:
                result = self.llm.invoke([("user", prompt)])
                payload = str(result.content).replace("```json", "").replace("```", "").strip()
                if payload:
                    return payload
            except Exception:
                pass
        return self._extract_expression(query) or "0"

    def _redact_tool_context_for_llm(self, context: dict[str, str]) -> dict[str, str]:
        redacted: dict[str, str] = {}
        for key, value in context.items():
            if key == "db_tool":
                redacted[key] = "[REDACTED_PERSONALIZED_DB_CONTEXT]"
            else:
                redacted[key] = redact_json_text(str(value))
        return redacted

    def _generate_response(self, state: BankingState) -> str:
        if "db_tool" in state.get("tool_outputs", {}):
            if self._is_personalized_guidance_query(state):
                return self._build_personalized_guidance_response(state)
            return self._build_db_response(state)
        if "calculator" in state.get("tool_outputs", {}):
            return self._fallback_response(state)
        if not self.llm:
            return self._fallback_response(state)
        prompt = (
            f"{self.system_prompt}\n\n"
            f"{self.response_prompt}\n\n"
            f"Route: {state.get('route')}\nRisk level: {state.get('risk_level')}\n"
            f"User role: {state.get('user_metadata', {}).get('role', '')}\n"
            f"User query: {redact_text(state.get('user_query', ''))}\nConfidence score: {state.get('confidence_score')}\n"
            f"Tool outputs:\n{json.dumps(self._redact_tool_context_for_llm(state.get('tool_outputs', {})), indent=2)[:9000]}"
        )
        try:
            result = self.llm.invoke([("user", prompt)])
            text = str(result.content).strip()
            return text or self._fallback_response(state)
        except Exception:
            return self._fallback_response(state)

    def _fallback_response(self, state: BankingState) -> str:
        outputs = state.get("tool_outputs", {})
        confidence = float(state.get("confidence_score", 0.0) or 0.0)
        if state.get("route") == "general" and confidence < 0.35:
            return self._low_confidence_response(state)
        if "calculator" in outputs:
            try:
                payload = json.loads(outputs["calculator"])
                if payload.get("operation") == "emi":
                    return (
                        f"Your estimated EMI is {payload.get('emi')}. "
                        f"Total payment would be {payload.get('total_payment')}, including {payload.get('total_interest')} as interest."
                    )
                if payload.get("operation") == "simple_interest":
                    return (
                        f"The estimated interest is {payload.get('interest')}. "
                        f"The maturity amount is {payload.get('maturity_amount')}."
                    )
            except Exception:
                pass
            return f"Here is the calculated result: {outputs['calculator']}"
        if "db_tool" in outputs:
            if self._is_personalized_guidance_query(state):
                return self._build_personalized_guidance_response(state)
            return self._build_db_response(state)
        if outputs:
            return redact_json_text(next(iter(outputs.values())))
        return "I could not complete this request. Please try again with a little more detail."

    def _blocked_response(self, state: BankingState) -> str:
        reason = str(state.get("risk_reason", "")).lower()
        if "legal advice" in reason:
            return "I cannot provide legal advice. I can share general banking process information, or escalate this for human review if you need help with a bank service issue."
        if "credentials" in reason or "personal identifiers" in reason:
            return "This request involves sensitive personal or credential information, so I cannot provide it. I have notified the risk team for review."
        return "This action is restricted and cannot be performed. I have notified the risk team for review."

    def _low_confidence_response(self, state: BankingState) -> str:
        role = str(state.get("user_metadata", {}).get("role", "customer"))
        if self._is_staff_role(role):
            return (
                "I could not verify this from the available knowledge sources. "
                "Please retry with a narrower query or review the relevant internal branch, customer, or policy records."
            )
        return "I could not verify this from the available banking knowledge base. Please contact the bank for the latest confirmed information."

    def _is_staff_role(self, role: str) -> bool:
        return role in {"manager", "admin", "support", "risk"}

    def _build_db_response(self, state: BankingState) -> str:
        raw_output = str(state.get("tool_outputs", {}).get("db_tool", ""))
        if not self._db_tool_succeeded(raw_output):
            return "I could not fetch the database details right now. Please check the database connection and try again."

        query = state.get("user_query", "").lower()
        try:
            payload = json.loads(raw_output)
        except json.JSONDecodeError:
            return "I found database information for your request, but could not format it clearly."

        if isinstance(payload, dict):
            customer_rows = payload.get("customer", [])
            loans = payload.get("loans", [])
            transactions = payload.get("transactions", [])
            customer = customer_rows[0] if isinstance(customer_rows, list) and customer_rows else {}
            role = str(state.get("user_metadata", {}).get("role", "customer"))
            if self._is_staff_role(role) and isinstance(customer, dict) and customer:
                lines = ["Customer record found for authorized review."]
                if isinstance(loans, list):
                    lines.append(f"Loan accounts: {len(loans)}")
                if isinstance(transactions, list):
                    lines.append(f"Recent transactions available: {len(transactions)}")
                if customer.get("creditscore") is not None:
                    lines.append("Credit score: available")
                if customer.get("balance") is not None:
                    lines.append("Balance: available")
                return "\n".join(lines)
            if "balance" in query and isinstance(customer, dict) and customer.get("balance") is not None:
                return f"Your current available balance is Rs. {float(customer['balance']):,.2f}."
            if isinstance(loans, list) and loans and any(token in query for token in ["loan", "repayment", "emi"]):
                loan_types = self._extract_safe_loan_types(raw_output)
                if loan_types:
                    joined_types = ", ".join(loan_types)
                    return f"I found your {joined_types} loan context. Please ask what you would like to know about repayment, EMI, foreclosure, prepayment, or account servicing."
            return "I found your account context, but I need a more specific question to answer without exposing account metadata."

        if isinstance(payload, list):
            if not payload:
                return "No matching records were found for your branch."
            if any("loan_type" in row for row in payload if isinstance(row, dict)):
                lines = ["Customers with loan accounts in your branch:"]
                for row in payload[:10]:
                    if not isinstance(row, dict):
                        continue
                    lines.append(
                        "- "
                        f"{redact_text(str(row.get('customer_name', 'Customer')))} | "
                        f"{row.get('loan_type', 'Loan')} | "
                        f"Status: {row.get('loan_status', 'Unknown')} | "
                        f"Outstanding: Rs. {float(row.get('outstanding_balance') or 0):,.2f}"
                    )
                if len(payload) > 10:
                    lines.append(f"And {len(payload) - 10} more record(s).")
                return "\n".join(lines)
            return f"I found {len(payload)} matching customer record(s) for your role."

        return "I found database information for your request."

    def _is_personalized_guidance_query(self, state: BankingState) -> bool:
        if state.get("route") != "personalized":
            return False
        query = state.get("user_query", "").lower()
        guidance_terms = [
            "explain",
            "option",
            "options",
            "repayment",
            "prepayment",
            "foreclosure",
            "part payment",
            "part-payment",
            "emi holiday",
            "moratorium",
            "how can i",
            "what can i",
            "what are",
            "pay extra",
            "extra payment",
            "prepay",
            "prepayment",
            "part payment",
            "foreclosure",
            "reduce tenure",
            "reduce emi",
        ]
        domain_terms = ["loan", "emi", "repay", "repayment", "home loan", "personal loan", "auto loan"]
        return any(term in query for term in guidance_terms) and (
            any(term in query for term in domain_terms) or self._history_mentions_loan(state.get("chat_history", []))
        )

    def _build_guidance_query(self, state: BankingState, db_output: str) -> str:
        loan_types = self._extract_safe_loan_types(db_output)
        query = redact_text(state.get("user_query", ""))
        if loan_types:
            return f"{' '.join(loan_types)} loan repayment options prepayment foreclosure EMI policy"
        return query

    def _build_personalized_guidance_response(self, state: BankingState) -> str:
        outputs = state.get("tool_outputs", {})
        db_output = str(outputs.get("db_tool", ""))
        if not self._db_tool_succeeded(db_output):
            return "I could not fetch your loan context right now. Please try again later."

        loan_types = self._extract_safe_loan_types(db_output)
        loan_context = ", ".join(loan_types) if loan_types else "your loan"
        policy_context = {
            key: value
            for key, value in outputs.items()
            if key in {"rag_retrieval", "search_api"}
        }
        if self.llm and policy_context:
            prompt = (
                f"{self.response_prompt}\n\n"
                "Use the loan type only as private context. Do not mention customer name, balance, credit score, account identifiers, record counts, database fields, or internal tool names. "
                "Answer as helpful banking guidance, and tell the customer to confirm exact terms in their loan agreement or with branch staff when needed.\n\n"
                f"Customer question: {redact_text(state.get('user_query', ''))}\n"
                f"Safe loan context: {loan_context}\n"
                f"Policy context:\n{json.dumps(self._redact_tool_context_for_llm(policy_context), indent=2)[:7000]}"
            )
            try:
                result = self.llm.invoke([("user", prompt)])
                text = str(result.content).strip()
                if text:
                    return text
            except Exception:
                pass

        return (
            f"For {loan_context}, common repayment options may include regular EMI payment, part-prepayment, foreclosure, "
            "or restructuring support where the bank policy allows it. Exact charges, eligibility, and process steps depend on the loan agreement and current bank policy."
        )

    def _extract_safe_loan_types(self, db_output: str) -> list[str]:
        try:
            payload = json.loads(db_output)
        except json.JSONDecodeError:
            return []
        loans: list[dict[str, object]] = []
        if isinstance(payload, dict) and isinstance(payload.get("loans"), list):
            loans = [loan for loan in payload["loans"] if isinstance(loan, dict)]
        elif isinstance(payload, list):
            loans = [loan for loan in payload if isinstance(loan, dict)]
        loan_types: list[str] = []
        for loan in loans:
            raw_type = str(loan.get("loantype") or loan.get("loan_type") or "").strip()
            if raw_type and raw_type not in loan_types:
                loan_types.append(redact_text(raw_type))
        return loan_types[:3]

    def _history_mentions_loan(self, chat_history: list[dict[str, str]] | None) -> bool:
        if not chat_history:
            return False
        recent_text = " ".join(str(item.get("text", "")) for item in chat_history[-6:]).lower()
        return any(token in recent_text for token in ["loan", "emi", "repayment", "prepayment", "foreclosure"])

    def _db_tool_succeeded(self, raw_output: str) -> bool:
        lowered = raw_output.lower().strip()
        if not lowered or lowered.startswith("db tool failed") or "request failed" in lowered:
            return False
        try:
            payload = json.loads(raw_output)
        except json.JSONDecodeError:
            return False
        if isinstance(payload, dict):
            return any(payload.get(key) for key in ["customer", "loans", "transactions"])
        if isinstance(payload, list):
            return True
        return False

    def _safe_tool_call(self, tool_name: str, argument: str) -> str:
        try:
            return self.mcp_client.call_tool(tool_name, argument)
        except Exception as exc:
            return f"{tool_name} failed: {exc}"

    def _storage_safe_response(self, state: BankingState) -> str:
        if "db_tool" in state.get("tool_outputs", {}) or state.get("route") == "personalized":
            return "[REDACTED_PERSONALIZED_RESPONSE]"
        return redact_text(str(state.get("final_response", "")))

    def _create_escalation(self, state: BankingState) -> str:
        metadata = state.get("user_metadata", {})
        payload = {
            "risk_level": state.get("risk_level", "medium"),
            "route": state.get("route", "escalation"),
            "customer_id": metadata.get("customer_id", ""),
            "branch": metadata.get("branch", ""),
            "role": metadata.get("role", ""),
            "query": redact_text(state.get("user_query", "")),
            "reason": redact_text(state.get("risk_reason", "")),
        }
        return self._safe_tool_call("create_escalation", json.dumps(payload))

    def _extract_confidence(self, raw: str) -> float:
        try:
            return float(json.loads(raw).get("confidence_score", 0.0))
        except Exception:
            return 0.0

    def _extract_expression(self, text: str) -> str:
        cleaned = text.replace(",", "")
        matches = re.findall(r"[\d\.\+\-\*\/\(\)\s]{3,}", cleaned)
        return max(matches, key=len).strip() if matches else ""

    def _extract_customer_id(self, text: str) -> str:
        match = re.search(r"\bC\d{3,}\b", text, flags=re.IGNORECASE)
        return match.group(0).upper() if match else ""

    def _add_step_log(self, state: BankingState, step: str, **payload: Any) -> None:
        item = redact_mapping({"timestamp": datetime.now(timezone.utc).isoformat(), "step": step, **payload})
        state.setdefault("logs", []).append(item)

    def _write_audit_log(self, state: BankingState) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "route": state.get("route"),
            "risk_level": state.get("risk_level"),
            "tools_used": list(state.get("tool_outputs", {}).keys()),
            "confidence_score": state.get("confidence_score", 0.0),
            "final_response": self._storage_safe_response(state),
            "logs": redact_value(state.get("logs", [])),
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def normalize_role(self, role: str | None) -> str:
        raw_role = (role or "").strip().lower().replace("&", "and").replace("-", "_").replace(" ", "_")
        role_map = {
            "customer": "customer",
            "manager": "manager",
            "branch_manager": "manager",
            "support": "support",
            "customer_support_agent": "support",
            "risk": "risk",
            "risk_and_compliance_officer": "risk",
            "risk_compliance_officer": "risk",
            "admin": "admin",
        }
        return role_map.get(raw_role, raw_role)


LangGraphBaselineAgent = MultiAgentBankingAssistant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LangGraph multi-agent banking assistant.")
    parser.add_argument("--role", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--customer-id")
    parser.add_argument("--branch")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent = MultiAgentBankingAssistant()
    result = agent.run(args.query, role=args.role, customer_id=args.customer_id, branch=args.branch)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
