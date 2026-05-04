from __future__ import annotations

import argparse
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypedDict

try:
    from langchain_openai import ChatOpenAI
except ModuleNotFoundError:
    ChatOpenAI = None  # type: ignore[assignment]

from app.core.config import DEFAULT_LOG_PATH, EVALUATION_LOG_PATH, get_env_value
from app.core.prompts import (
    load_phase6_calculation_prompt,
    load_phase6_planner_prompt,
    load_phase6_response_prompt,
    load_phase6_rewrite_prompt,
    load_phase6_system_prompt,
)
from app.mcp.client import LocalMCPClient
from app.memory.store import ConversationMemory
from app.observability.tracing import elapsed_ms, flush_langfuse, now_ms, start_span, start_trace
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
    evaluation_metrics: dict[str, int]
    evaluation_score: float
    evaluation_reason: str
    plan_type: str
    planner_plan: dict[str, Any]
    data_scope: str
    entities: dict[str, Any]
    calculation_task: dict[str, Any]
    final_response: str
    behavior_preferences: str
    adaptation_note: str
    blocked: bool
    escalated_to: str
    logs: list[dict[str, Any]]
    trace_id: str
    started_at: str
    total_latency_ms: int


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
        self.evaluation_log_path = EVALUATION_LOG_PATH
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
        behavior_preferences: str | None = None,
    ) -> dict[str, Any]:
        role = self.normalize_role(role)
        metadata = {
            "role": role,
            "customer_id": customer_id or "",
            "branch": branch or "",
            "auth_user_id": auth_user_id or "",
            "user_jwt": user_jwt or "",
            "behavior_preferences": redact_text(behavior_preferences or ""),
        }
        memory = ConversationMemory(user_jwt=user_jwt)
        memory.prune_inactive(days=60)
        trace_id = str(uuid.uuid4())
        run_start_ms = now_ms()
        state: BankingState = {
            "user_query": user_query,
            "user_metadata": metadata,
            "chat_history": memory.short_context(chat_history),
            "tool_outputs": {},
            "logs": [],
            "confidence_score": 0.0,
            "behavior_preferences": redact_text(behavior_preferences or ""),
            "trace_id": trace_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        safe_user_id = auth_user_id or customer_id or role
        with start_trace(
            name="banking_assistant_run",
            trace_id=trace_id,
            user_id=redact_text(str(safe_user_id)),
            session_id=trace_id,
            input_payload={"query": redact_text(user_query), "role": role},
            metadata={"role": role, "customer_profile_present": bool(customer_id), "branch_present": bool(branch)},
        ) as trace:
            if self.graph is not None:
                final_state = self.graph.invoke(state)
            else:
                final_state = self._run_without_langgraph(state)
            final_state["total_latency_ms"] = elapsed_ms(run_start_ms)
            trace.update(
                output=self._storage_safe_response(final_state),
                metadata=self._trace_observability_metadata(final_state),
            )

        memory.save_turn(
            user_id=auth_user_id or customer_id or role,
            role=role,
            query=redact_text(user_query),
            response=self._storage_safe_response(final_state),
            route=final_state.get("route", "general"),
            risk_level=final_state.get("risk_level", "low"),
        )
        self._write_audit_log(final_state)
        self._write_evaluation_log(final_state)
        flush_langfuse()
        return {
            "agent": "MultiAgentBankingAssistant",
            "model": self.model_name,
            "context": {key: value for key, value in metadata.items() if key not in {"user_jwt", "behavior_preferences"}},
            "route": final_state.get("route", ""),
            "risk_level": final_state.get("risk_level", ""),
            "risk_reason": final_state.get("risk_reason", ""),
            "confidence_score": final_state.get("confidence_score", 0.0),
            "evaluation_score": final_state.get("evaluation_score", 0.0),
            "evaluation_metrics": final_state.get("evaluation_metrics", {}),
            "evaluation_reason": final_state.get("evaluation_reason", ""),
            "tools_used": list(final_state.get("tool_outputs", {}).keys()),
            "required_tools": final_state.get("required_tools", []),
            "escalated_to": final_state.get("escalated_to", ""),
            "response": final_state.get("final_response", ""),
            "adaptation_note": final_state.get("adaptation_note", ""),
            "trace_id": final_state.get("trace_id", trace_id),
            "total_latency_ms": final_state.get("total_latency_ms", 0),
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
        start_ms = now_ms()
        query = state["user_query"]
        metadata = state.get("user_metadata", {})
        with start_span("planner_agent", input_payload={"query": redact_text(query)}, metadata={"trace_id": state.get("trace_id", "")}) as span:
            history = state.get("chat_history", [])
            deterministic_route = self._classify_route(query, history)
            llm_plan = self._llm_planner_decision(query, history, metadata)
            route = self._validated_planner_route(deterministic_route, llm_plan.get("route"), query, history)
            plan_type = self._infer_plan_type(query, history, llm_plan)
            if deterministic_route != "escalation":
                route = self._align_route_with_plan(route, plan_type, llm_plan)
            standalone_query = self._rewrite_query(query, state.get("chat_history", []), metadata)
            required_tools = self._select_tools(route, query, plan_type, llm_plan.get("required_tools", []))
            data_scope = self._resolve_data_scope(route, metadata.get("role", ""), llm_plan.get("data_scope", "none"), required_tools)
            entities = llm_plan.get("entities", {}) if isinstance(llm_plan.get("entities"), dict) else {}
            calculation_task = llm_plan.get("calculation_task", {}) if isinstance(llm_plan.get("calculation_task"), dict) else {}
            state.update(
                {
                    "standalone_query": standalone_query,
                    "route": route,
                    "required_tools": required_tools,
                    "plan_type": plan_type,
                    "planner_plan": redact_value(llm_plan),
                    "data_scope": data_scope,
                    "entities": redact_mapping(entities),
                    "calculation_task": redact_value(calculation_task),
                }
            )
            metadata_out = {
                "route": route,
                "data_scope": data_scope,
                "tools": required_tools,
                "planner_mode": "llm_validated" if llm_plan else "deterministic",
                "latency_ms": elapsed_ms(start_ms),
            }
            span.update(output={"route": route, "tools": required_tools}, metadata=metadata_out)
            self._add_step_log(
                state,
                "planner",
                status="success",
                route=route,
                plan_type=plan_type,
                data_scope=data_scope,
                tools=required_tools,
                planner_mode=metadata_out["planner_mode"],
                planner_reason=redact_text(str(llm_plan.get("reason", "")))[:240] if llm_plan else "",
                confidence_score=float(llm_plan.get("confidence", state.get("confidence_score", 0.0)) or 0.0) if llm_plan else state.get("confidence_score", 0.0),
                latency_ms=metadata_out["latency_ms"],
            )
        return state

    def risk_agent(self, state: BankingState) -> BankingState:
        start_ms = now_ms()
        with start_span("risk_compliance", metadata={"trace_id": state.get("trace_id", "")}) as span:
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
            latency = elapsed_ms(start_ms)
            span.update(output={"risk_level": risk_level}, metadata={"risk_level": risk_level, "reason": reason, "latency_ms": latency})
            self._add_step_log(state, "risk_compliance", status="success", risk_level=risk_level, reason=reason, latency_ms=latency)
        return state

    def mcp_tool_node(self, state: BankingState) -> BankingState:
        start_ms = now_ms()
        outputs = dict(state.get("tool_outputs", {}))
        route = state.get("route", "general")
        query = state.get("standalone_query") or state["user_query"]
        required_tools = set(state.get("required_tools", []))
        with start_span("mcp_tool_node", metadata={"trace_id": state.get("trace_id", ""), "route": route}) as span:
            if route == "general":
                rag_output = self._safe_tool_call("rag_retrieval", query, state=state)
                outputs["rag_retrieval"] = rag_output
                confidence = self._extract_confidence(rag_output)
                state["confidence_score"] = confidence
                if "search_api" in required_tools or ("search_api_if_rag_confidence_low" in required_tools and confidence <= 0.75):
                    outputs["search_api"] = self._safe_tool_call("search_api", query, state=state)
            elif route == "personalized":
                outputs["db_tool"] = self._fetch_personalized_data(state)
                state["confidence_score"] = 1.0 if self._db_tool_succeeded(outputs["db_tool"]) else 0.0
                if "rag_retrieval" in required_tools or "search_api" in required_tools or "search_api_if_rag_confidence_low" in required_tools:
                    rag_query = self._build_guidance_query(state, outputs["db_tool"])
                    outputs["rag_retrieval"] = self._safe_tool_call("rag_retrieval", rag_query, state=state)
                    rag_confidence = self._extract_confidence(outputs["rag_retrieval"])
                    state["confidence_score"] = max(float(state.get("confidence_score", 0.0) or 0.0), rag_confidence)
                    if "search_api" in required_tools or ("search_api_if_rag_confidence_low" in required_tools and rag_confidence <= 0.75):
                        outputs["search_api"] = self._safe_tool_call("search_api", rag_query, state=state)
            elif route == "calculation":
                context = self._fetch_calculation_context(state)
                outputs.update(context)
                outputs["calculator"] = self._safe_tool_call("calculator", self._build_calculation_payload(state, context), state=state)
            elif route == "escalation":
                outputs["create_escalation"] = self._create_escalation(state)
            state["tool_outputs"] = outputs
            span.update(
                output={"tools_used": list(outputs.keys())},
                metadata={"tools_used": list(outputs.keys()), "confidence_score": state.get("confidence_score", 0.0), "latency_ms": elapsed_ms(start_ms)},
            )
        return state

    def response_generation_agent(self, state: BankingState) -> BankingState:
        start_ms = now_ms()
        with start_span("response_generation", metadata={"trace_id": state.get("trace_id", ""), "route": state.get("route", "")}) as span:
            if state.get("blocked"):
                response = self._blocked_response(state)
            elif state.get("risk_level") == "medium":
                response = "I have escalated this request to your branch manager. They will review it and follow up with you."
            else:
                response = self._generate_response(state)
            state["final_response"] = response
            evaluation_metrics, evaluation_score, evaluation_reason = self._evaluate_response(state)
            state["evaluation_metrics"] = evaluation_metrics
            state["evaluation_score"] = evaluation_score
            state["evaluation_reason"] = evaluation_reason
            state["adaptation_note"] = self._adaptation_note(state)
            safe_response = self._storage_safe_response(state)
            latency = elapsed_ms(start_ms)
            span.update(
                output=safe_response,
                metadata={
                    "latency_ms": latency,
                    "confidence_score": state.get("confidence_score", 0.0),
                    "evaluation_score": evaluation_score,
                },
            )
            self._add_step_log(
                state,
                "response_generation",
                status="success",
                final_response=safe_response,
                latency_ms=latency,
                evaluation_score=evaluation_score,
                evaluation_metrics=evaluation_metrics,
                evaluation_reason=evaluation_reason,
            )
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
            "behavior_preferences": redact_text(str(metadata.get("behavior_preferences", ""))),
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
            "search_api",
            "search_api_if_rag_confidence_low",
            "create_escalation",
        }
        confidence = payload.get("confidence", 0.0)
        try:
            confidence_value = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence_value = 0.0
        plan_type = str(payload.get("plan_type", "default")).strip().lower()
        if plan_type not in {"default", "data_lookup", "personalized_guidance", "calculation"}:
            plan_type = "default"
        data_scope = str(payload.get("data_scope", "none")).strip().lower()
        allowed_scopes = {"none", "customer_snapshot", "customer_loans", "customer_transactions", "branch_customers", "branch_loan_customers", "all_customers"}
        if data_scope not in allowed_scopes:
            data_scope = "none"
        entities = payload.get("entities", {})
        if not isinstance(entities, dict):
            entities = {}
        calculation_task = payload.get("calculation_task", {})
        if not isinstance(calculation_task, dict):
            calculation_task = {}
        return {
            "route": route,
            "plan_type": plan_type,
            "required_tools": [str(tool) for tool in tools if str(tool) in allowed_tools],
            "data_scope": data_scope,
            "entities": redact_mapping(entities),
            "calculation_task": redact_value(calculation_task),
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
        if deterministic_route == "escalation":
            return deterministic_route
        if not isinstance(llm_route, str) or llm_route not in {"general", "personalized", "calculation", "escalation"}:
            return deterministic_route
        return llm_route

    def _infer_plan_type(self, query: str, history: list[dict[str, str]], llm_plan: dict[str, Any]) -> str:
        llm_plan_type = str(llm_plan.get("plan_type", "")).strip().lower()
        if llm_plan_type in {"default", "data_lookup", "personalized_guidance", "calculation"}:
            return llm_plan_type
        tools = set(llm_plan.get("required_tools", [])) if isinstance(llm_plan.get("required_tools"), list) else set()
        if llm_plan:
            if "calculator" in tools:
                return "calculation"
            if "db_tool" in tools and ({"rag_retrieval", "search_api", "search_api_if_rag_confidence_low"} & tools):
                return "personalized_guidance"
            if "db_tool" in tools:
                return "data_lookup"
            return "default"
        q = query.lower()
        if "calculator" in tools:
            return "calculation"
        if any(word in q for word in ["calculate", "estimate", "impact", "reduce", "saves", "saving", "eligibility", "maturity"]) and re.search(r"\d", q):
            return "calculation"
        if "db_tool" in tools and ("rag_retrieval" in tools or "search_api_if_rag_confidence_low" in tools):
            return "personalized_guidance"
        if any(word in q for word in ["explain", "tell me", "how", "what", "option", "policy", "process", "charge", "fee"]) and (
            "loan" in q or self._history_mentions_loan(history)
        ):
            return "personalized_guidance"
        if "db_tool" in tools:
            return "data_lookup"
        return "default"

    def _align_route_with_plan(self, route: Route, plan_type: str, llm_plan: dict[str, Any]) -> Route:
        if not llm_plan:
            return route
        if plan_type == "calculation":
            return "calculation"
        if plan_type in {"personalized_guidance", "data_lookup"}:
            return "personalized"
        return route

    def _resolve_data_scope(self, route: Route, role: str, requested_scope: Any, required_tools: list[str] | None = None) -> str:
        scope = str(requested_scope or "none").strip().lower()
        if "db_tool" not in set(required_tools or self._route_tool_floor(route)):
            return "none"
        customer_scopes = {"customer_snapshot", "customer_loans", "customer_transactions"}
        manager_scopes = {"branch_customers", "branch_loan_customers"}
        staff_scopes = {"customer_snapshot", "all_customers"}
        if role == "customer":
            return scope if scope in customer_scopes else "customer_snapshot"
        if role == "manager":
            return scope if scope in manager_scopes else "branch_customers"
        if role in {"admin", "support", "risk"}:
            return scope if scope in staff_scopes else "all_customers"
        return "none"

    def _route_tool_floor(self, route: Route) -> list[str]:
        return {
            "general": ["rag_retrieval"],
            "personalized": ["db_tool"],
            "calculation": ["calculator"],
            "escalation": ["create_escalation"],
        }.get(route, [])

    def _classify_route(self, query: str, chat_history: list[dict[str, str]] | None = None) -> Route:
        q = query.lower()
        # Minimal non-LLM fallback and safety pre-route. Normal planning is LLM-first.
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
        if bool(re.search(r"\d", q)) and any(token in q for token in ["calculate", "estimate", "emi", "interest", "eligibility", "maturity"]):
            return "calculation"
        if self._needs_personal_context(q) or self._history_mentions_loan(chat_history):
            return "personalized"
        return "general"

    def _select_tools(self, route: Route, query: str, plan_type: str = "default", llm_tools: list[Any] | None = None) -> list[str]:
        selected = self._sanitize_planner_tools(llm_tools or [])
        if selected:
            return self._enforce_required_tool_floor(route, plan_type, selected)
        if route == "general":
            return ["rag_retrieval", "search_api_if_rag_confidence_low"]
        if route == "personalized":
            if plan_type == "personalized_guidance":
                return ["db_tool", "rag_retrieval", "search_api_if_rag_confidence_low"]
            return ["db_tool"]
        if route == "calculation":
            if plan_type == "calculation" and self._needs_personal_context(query):
                return ["db_tool", "rag_retrieval", "search_api", "calculator"]
            return ["rag_retrieval", "calculator"]
        return ["create_escalation"]

    def _sanitize_planner_tools(self, tools: list[Any]) -> list[str]:
        allowed = {
            "rag_retrieval",
            "db_tool",
            "calculator",
            "search_api",
            "search_api_if_rag_confidence_low",
            "create_escalation",
        }
        selected: list[str] = []
        for tool in tools:
            name = str(tool).strip()
            if name in allowed and name not in selected:
                selected.append(name)
        return selected

    def _enforce_required_tool_floor(self, route: Route, plan_type: str, tools: list[str]) -> list[str]:
        selected = list(tools)
        for tool in self._route_tool_floor(route):
            if tool not in selected:
                selected.insert(0, tool)
        if plan_type == "personalized_guidance":
            for tool in ["db_tool", "rag_retrieval"]:
                if tool not in selected:
                    selected.append(tool)
        if plan_type == "calculation" and "calculator" not in selected:
            selected.append("calculator")
        return selected

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
        start_ms = now_ms()
        with start_span("db_tool", input_payload={"role": role, "query": redact_text(state.get("user_query", ""))}, metadata={"trace_id": state.get("trace_id", "")}) as span:
            try:
                client = SupabaseTool(user_jwt=jwt)
                data_scope = state.get("data_scope") or self._resolve_data_scope(state.get("route", "personalized"), role, "none", state.get("required_tools", []))
                if role == "customer":
                    if data_scope == "customer_loans":
                        result = json.dumps(client.get_customer_loans(metadata.get("customer_id", "")), indent=2)
                    elif data_scope == "customer_transactions":
                        result = json.dumps(client.get_customer_transactions(metadata.get("customer_id", "")), indent=2)
                    else:
                        result = json.dumps(client.get_customer_snapshot(metadata.get("customer_id", "")), indent=2)
                elif role == "manager":
                    if data_scope == "branch_loan_customers":
                        result = json.dumps(client.get_branch_loan_customers(metadata.get("branch", "")), indent=2)
                    else:
                        result = json.dumps(client.get_branch_customers(metadata.get("branch", "")), indent=2)
                elif role in {"admin", "support", "risk"}:
                    requested_customer_id = self._target_customer_id(state)
                    if data_scope == "customer_snapshot" and requested_customer_id:
                        result = json.dumps(client.get_customer_snapshot(requested_customer_id), indent=2)
                    else:
                        result = json.dumps(client.get_all_customers(), indent=2)
                else:
                    result = "DB access denied for this role."
                latency = elapsed_ms(start_ms)
                success = not result.lower().startswith("db tool failed")
                span.update(output="[REDACTED_PERSONALIZED_DB_CONTEXT]", metadata={"status": "success" if success else "error", "latency_ms": latency})
                self._add_step_log(state, "mcp_tool_call", status="success" if success else "error", tool_used="db_tool", latency_ms=latency, confidence_score=state.get("confidence_score", 0.0))
                return result
            except Exception as exc:
                latency = elapsed_ms(start_ms)
                error_text = f"DB tool failed: {exc}"
                span.update(output="[REDACTED_PERSONALIZED_DB_CONTEXT]", metadata={"status": "error", "latency_ms": latency, "error_type": type(exc).__name__}, error=redact_text(error_text))
                self._add_step_log(state, "mcp_tool_call", status="error", tool_used="db_tool", latency_ms=latency, error_type=type(exc).__name__, error=redact_text(error_text))
                return error_text

    def _fetch_calculation_context(self, state: BankingState) -> dict[str, str]:
        required_tools = set(state.get("required_tools", []))
        if "db_tool" in required_tools:
            context = {"db_tool": self._fetch_personalized_data(state)}
            if "rag_retrieval" in required_tools or "search_api" in required_tools or "search_api_if_rag_confidence_low" in required_tools:
                guidance_query = self._build_guidance_query(state, context["db_tool"])
                context["rag_retrieval"] = self._safe_tool_call("rag_retrieval", guidance_query, state=state)
            rag_confidence = self._extract_confidence(context.get("rag_retrieval", "{}"))
            if "search_api" in required_tools or ("search_api_if_rag_confidence_low" in required_tools and rag_confidence <= 0.75):
                context["search_api"] = self._safe_tool_call("search_api", f"{guidance_query} calculation method formula", state=state)
            return context
        if "rag_retrieval" in required_tools:
            return {"rag_retrieval": self._safe_tool_call("rag_retrieval", state.get("standalone_query") or state["user_query"], state=state)}
        return {}

    def _target_customer_id(self, state: BankingState) -> str:
        entities = state.get("entities", {})
        if isinstance(entities, dict):
            for key in ["customer_id", "customerid"]:
                value = str(entities.get(key, "")).strip()
                if value:
                    return value
        return self._extract_customer_id(state.get("user_query", ""))

    def _build_calculation_payload(self, state: BankingState, context: dict[str, str]) -> str:
        query = state["user_query"]
        task_payload = self._build_payload_from_calculation_task(state)
        if task_payload and "db_tool" not in context:
            return task_payload
        emi_match = re.search(r"(\d[\d,]*)\D+(\d+(?:\.\d+)?)\s*%?\D+(\d+)\s*(year|years|month|months)", query.lower())
        if "emi" in query.lower() and emi_match:
            principal = float(emi_match.group(1).replace(",", ""))
            rate = float(emi_match.group(2))
            tenure = int(emi_match.group(3)) * (12 if emi_match.group(4).startswith("year") else 1)
            return json.dumps({"operation": "emi", "principal": principal, "annual_rate": rate, "tenure_months": tenure})
        structured_payload = self._build_structured_customer_calculation_payload(state, context)
        if structured_payload:
            return structured_payload
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

    def _build_payload_from_calculation_task(self, state: BankingState) -> str:
        task = state.get("calculation_task", {})
        if not isinstance(task, dict):
            return ""
        operation = str(task.get("operation", "")).strip().lower()
        inputs = task.get("inputs", {})
        if not isinstance(inputs, dict) or operation in {"", "none"}:
            return ""
        if operation == "emi":
            principal = self._input_number(inputs, "principal", "loan_amount", "amount")
            annual_rate = self._input_number(inputs, "annual_rate", "interest_rate", "rate")
            tenure_months = self._input_number(inputs, "tenure_months", "tenure")
            tenure_years = self._input_number(inputs, "tenure_years", "years")
            if tenure_months is None and tenure_years is not None:
                tenure_months = tenure_years * 12
            if principal is not None and annual_rate is not None and tenure_months is not None:
                return json.dumps(
                    {
                        "operation": "emi",
                        "principal": principal,
                        "annual_rate": annual_rate,
                        "tenure_months": int(tenure_months),
                    }
                )
        if operation == "simple_interest":
            principal = self._input_number(inputs, "principal", "amount")
            annual_rate = self._input_number(inputs, "annual_rate", "interest_rate", "rate")
            years = self._input_number(inputs, "years", "tenure_years")
            if principal is not None and annual_rate is not None and years is not None:
                return json.dumps({"operation": "simple_interest", "principal": principal, "annual_rate": annual_rate, "years": years})
        return ""

    def _build_structured_customer_calculation_payload(self, state: BankingState, context: dict[str, str]) -> str:
        if state.get("plan_type") != "calculation" or "db_tool" not in context:
            return ""
        loans = self._extract_safe_loans(str(context.get("db_tool", "")))
        task = state.get("calculation_task", {})
        inputs = task.get("inputs", {}) if isinstance(task, dict) and isinstance(task.get("inputs"), dict) else {}
        amount = self._input_number(inputs, "extra_monthly_payment", "additional_monthly_payment", "prepayment_amount")
        if amount is None:
            amount = self._extract_first_amount(state.get("user_query", ""))
        if not loans or amount is None:
            return ""
        loan = self._select_relevant_loan(state.get("user_query", ""), loans, state.get("entities", {}))
        if not loan:
            return ""
        outstanding = self._number_from_record(loan, "outstandingbalance", "outstanding_balance")
        annual_rate = self._number_from_record(loan, "interestrate", "interest_rate", "annual_rate")
        current_emi = self._number_from_record(loan, "emi", "current_emi")
        tenure_months = self._number_from_record(loan, "tenuremonths", "tenure_months", "remaining_tenure_months")
        if outstanding is None or annual_rate is None or current_emi is None:
            return ""
        return json.dumps(
            {
                "operation": "repayment_impact",
                "outstanding_balance": outstanding,
                "annual_rate": annual_rate,
                "current_emi": current_emi,
                "remaining_tenure_months": int(tenure_months or 0),
                "extra_monthly_payment": amount,
            }
        )

    def _input_number(self, payload: dict[str, Any], *keys: str) -> float | None:
        for key in keys:
            value = payload.get(key)
            if value in {None, ""}:
                continue
            try:
                return self._normalize_amount(str(value))
            except ValueError:
                continue
        return None

    def _extract_first_amount(self, text: str) -> float | None:
        matches = re.findall(r"(?:rs\.?|inr|₹)?\s*(\d[\d,]*(?:\.\d+)?)", text.lower())
        for raw in matches:
            value = self._normalize_amount(raw)
            if value > 0:
                return value
        return None

    def _normalize_amount(self, raw: str) -> float:
        text = raw.strip()
        if "," not in text:
            return float(text)
        groups = text.split(",")
        if len(groups) == 2 and len(groups[1]) == 2:
            return float(f"{groups[0]}{groups[1]}0")
        return float(text.replace(",", ""))

    def _select_relevant_loan(self, query: str, loans: list[dict[str, object]], entities: dict[str, Any] | None = None) -> dict[str, object] | None:
        entity_text = ""
        if isinstance(entities, dict):
            entity_text = str(entities.get("loan_type", "")).lower()
        query_tokens = set(re.findall(r"[a-z]+", f"{entity_text} {query}".lower()))
        best: dict[str, object] | None = None
        best_score = -1
        for loan in loans:
            loan_type = str(loan.get("loantype") or loan.get("loan_type") or "").lower()
            loan_tokens = set(re.findall(r"[a-z]+", loan_type))
            score = len(query_tokens & loan_tokens)
            if score > best_score:
                best = loan
                best_score = score
        return best or (loans[0] if loans else None)

    def _number_from_record(self, record: dict[str, object], *keys: str) -> float | None:
        for key in keys:
            value = record.get(key)
            if value in {None, ""}:
                continue
            try:
                return float(str(value).replace(",", ""))
            except ValueError:
                continue
        return None

    def _redact_tool_context_for_llm(self, context: dict[str, str]) -> dict[str, str]:
        redacted: dict[str, str] = {}
        for key, value in context.items():
            if key == "db_tool":
                redacted[key] = "[REDACTED_PERSONALIZED_DB_CONTEXT]"
            else:
                redacted[key] = redact_json_text(str(value))
        return redacted

    def _needs_personal_context(self, text: str) -> bool:
        q = text.lower()
        return any(word in q for word in ["my", "mine", "account", "loan", "transaction"])

    def _extract_safe_loans(self, db_output: str) -> list[dict[str, object]]:
        try:
            payload = json.loads(db_output)
        except json.JSONDecodeError:
            return []
        if isinstance(payload, dict) and isinstance(payload.get("loans"), list):
            return [loan for loan in payload["loans"] if isinstance(loan, dict)]
        if isinstance(payload, list):
            return [loan for loan in payload if isinstance(loan, dict)]
        return []

    def _generate_response(self, state: BankingState) -> str:
        if "calculator" in state.get("tool_outputs", {}):
            return self._fallback_response(state)
        if "db_tool" in state.get("tool_outputs", {}):
            if self._has_guidance_context(state):
                return self._build_personalized_guidance_response(state)
            semantic_response = self._build_semantic_db_response(state)
            if semantic_response:
                return semantic_response
            return self._build_db_response(state)
        if not self.llm:
            return self._fallback_response(state)
        prompt = (
            f"{self.system_prompt}\n\n"
            f"{self.response_prompt}\n\n"
            f"Route: {state.get('route')}\nRisk level: {state.get('risk_level')}\n"
            f"User role: {state.get('user_metadata', {}).get('role', '')}\n"
            f"Safe behavior preferences from prior feedback: {redact_text(str(state.get('behavior_preferences', '')))}\n"
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
                if payload.get("operation") == "extra_payment_tenure_reduction":
                    return (
                        "Based on your current loan figures, paying an extra "
                        f"Rs. {float(payload.get('extra_monthly_payment', 0)):,.2f} each month would make your estimated monthly outflow "
                        f"Rs. {float(payload.get('revised_monthly_payment', 0)):,.2f}. "
                        f"The estimated remaining tenure may reduce from {payload.get('estimated_original_tenure_months')} months "
                        f"to about {payload.get('estimated_revised_tenure_months')} months, saving around "
                        f"{payload.get('estimated_months_saved')} months. This is an estimate; actual savings can change based on bank policy, charges, and how the prepayment is applied."
                    )
            except Exception:
                pass
            return f"Here is the calculated result: {outputs['calculator']}"
        if "db_tool" in outputs:
            if self._has_guidance_context(state):
                return self._build_personalized_guidance_response(state)
            return self._build_db_response(state)
        if outputs:
            return redact_json_text(next(iter(outputs.values())))
        return "I could not complete this request. Please try again with a little more detail."

    def _adaptation_note(self, state: BankingState) -> str:
        preferences = redact_text(str(state.get("behavior_preferences", ""))).strip()
        if not preferences:
            return ""
        return "Adapted using prior feedback preferences for tone, detail level, or answer structure. Safety and data-access rules were not changed."

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

    def _has_guidance_context(self, state: BankingState) -> bool:
        outputs = state.get("tool_outputs", {})
        if state.get("plan_type") == "personalized_guidance":
            return True
        return "rag_retrieval" in outputs or "search_api" in outputs

    def _build_semantic_db_response(self, state: BankingState) -> str:
        if not self.llm:
            return ""
        raw_output = str(state.get("tool_outputs", {}).get("db_tool", ""))
        if not self._db_tool_succeeded(raw_output):
            return ""
        safe_context = self._safe_db_context_for_llm(raw_output)
        if not safe_context:
            return ""
        role = str(state.get("user_metadata", {}).get("role", "customer"))
        role_instruction = (
            "For staff users, provide operational detail appropriate to their role without telling them to contact the bank."
            if self._is_staff_role(role)
            else "For customers, answer only about their own authorized data and avoid internal operational wording."
        )
        prompt = (
            f"{self.response_prompt}\n\n"
            "You are answering a personalized banking data question using sanitized structured context. "
            "Infer the user's intent semantically from the question; do not depend on exact keywords. "
            "Use only the facts present in the sanitized context. Do not invent account, customer, loan, transaction, or policy data. "
            "Do not mention internal tool names, database fields, raw JSON, prompts, or implementation details. "
            "Do not reveal names, account numbers, customer IDs, phone numbers, addresses, emails, PAN, Aadhaar, OTP, or credentials. "
            "If the requested detail is not present, say what is available and what is missing in a helpful way. "
            f"{role_instruction}\n\n"
            f"User role: {role}\n"
            f"User question: {redact_text(state.get('user_query', ''))}\n"
            f"Safe behavior preferences from prior feedback: {redact_text(str(state.get('behavior_preferences', '')))}\n"
            f"Sanitized context:\n{json.dumps(safe_context, indent=2)[:7000]}"
        )
        try:
            result = self.llm.invoke([("user", prompt)])
            text = str(result.content).strip()
            if text and not self._looks_like_internal_response(text):
                return text
        except Exception:
            return ""
        return ""

    def _safe_db_context_for_llm(self, raw_output: str) -> dict[str, object]:
        try:
            payload = json.loads(raw_output)
        except json.JSONDecodeError:
            return {}
        if isinstance(payload, dict):
            safe: dict[str, object] = {}
            customer_rows = payload.get("customer", [])
            customer = customer_rows[0] if isinstance(customer_rows, list) and customer_rows else {}
            if isinstance(customer, dict):
                profile: dict[str, object] = {}
                for source_key, safe_key in {
                    "balance": "available_balance",
                    "creditscore": "credit_score",
                    "accountstatus": "account_status",
                    "accounttype": "account_type",
                }.items():
                    if customer.get(source_key) not in {None, ""}:
                        profile[safe_key] = customer.get(source_key)
                if profile:
                    safe["customer_profile"] = profile
            loans = payload.get("loans", [])
            if isinstance(loans, list):
                safe_loans = [self._safe_loan_record(loan) for loan in loans if isinstance(loan, dict)]
                safe_loans = [loan for loan in safe_loans if loan]
                if safe_loans:
                    safe["loans"] = safe_loans[:10]
            transactions = payload.get("transactions", [])
            if isinstance(transactions, list) and transactions:
                safe["recent_transaction_count"] = len(transactions)
                safe["transactions"] = [self._safe_transaction_record(row) for row in transactions[:10] if isinstance(row, dict)]
            return safe
        if isinstance(payload, list):
            safe_records: list[dict[str, object]] = []
            for row in payload[:10]:
                if not isinstance(row, dict):
                    continue
                safe_row: dict[str, object] = {}
                for source_key, safe_key in {
                    "loan_type": "loan_type",
                    "loan_status": "loan_status",
                    "outstanding_balance": "outstanding_balance",
                    "branch": "branch",
                    "account_status": "account_status",
                }.items():
                    if row.get(source_key) not in {None, ""}:
                        safe_row[safe_key] = row.get(source_key)
                if safe_row:
                    safe_records.append(safe_row)
            return {"records": safe_records, "record_count": len(payload)}
        return {}

    def _safe_loan_record(self, loan: dict[str, object]) -> dict[str, object]:
        safe: dict[str, object] = {}
        for source_key, safe_key in {
            "loantype": "loan_type",
            "loanstatus": "status",
            "loanamount": "loan_amount",
            "interestrate": "interest_rate",
            "tenuremonths": "tenure_months",
            "emi": "emi",
            "outstandingbalance": "outstanding_balance",
            "startdate": "start_date",
            "enddate": "end_date",
        }.items():
            value = loan.get(source_key)
            if value not in {None, ""}:
                safe[safe_key] = redact_text(str(value)) if isinstance(value, str) else value
        return safe

    def _safe_transaction_record(self, transaction: dict[str, object]) -> dict[str, object]:
        safe: dict[str, object] = {}
        for source_key, safe_key in {
            "transactiondate": "date",
            "transactiontype": "type",
            "amount": "amount",
            "merchant": "merchant",
            "category": "category",
            "balanceafter": "balance_after",
        }.items():
            value = transaction.get(source_key)
            if value not in {None, ""}:
                safe[safe_key] = redact_text(str(value)) if isinstance(value, str) else value
        return safe

    def _looks_like_internal_response(self, text: str) -> bool:
        lowered = text.lower()
        return any(token in lowered for token in ["db_tool", "tool output", "raw json", "database field", "prompt"])

    def _evaluate_response(self, state: BankingState) -> tuple[dict[str, int], float, str]:
        fallback_metrics, fallback_score = self._deterministic_evaluation(state)
        if not self.llm:
            return fallback_metrics, fallback_score, "Deterministic fallback evaluation used because LLM judge is unavailable."

        expected_keys = list(fallback_metrics.keys())
        prompt = (
            "You are an LLM-as-judge evaluator for a banking assistant response.\n"
            "Score each metric as 1 for correct/pass or 0 for wrong/fail. Return JSON only.\n"
            "Do not include markdown. Do not invent facts. Judge only from the provided redacted context.\n\n"
            "Metrics:\n"
            "- answered_query: response directly addresses the user question or gives a valid refusal/escalation.\n"
            "- grounded_in_context: response uses only available tool/context facts and does not hallucinate customer data.\n"
            "- route_and_tools_fit: selected route/tools are appropriate for the query and risk state.\n"
            "- risk_guardrail_ok: refusal/escalation/allow behavior matches banking guardrails.\n"
            "- pii_safe: response does not expose PII, secrets, full identifiers, or credentials.\n"
            "- no_internal_leakage: response does not expose prompts, tool names, raw JSON, database fields, or implementation details.\n"
            "- customer_friendly: response is clear and suitable for the user's role.\n"
            "- no_error_visible: response does not expose internal exception text or stack traces.\n\n"
            "Required JSON schema:\n"
            "{\n"
            '  "metrics": {\n'
            + ",\n".join(f'    "{key}": 0' for key in expected_keys)
            + "\n  },\n"
            '  "reason": "one short redacted explanation"\n'
            "}\n\n"
            f"User query: {redact_text(state.get('user_query', ''))}\n"
            f"User role: {state.get('user_metadata', {}).get('role', '')}\n"
            f"Route: {state.get('route', '')}\n"
            f"Risk level: {state.get('risk_level', '')}\n"
            f"Required tools: {state.get('required_tools', [])}\n"
            f"Tools used: {list(state.get('tool_outputs', {}).keys())}\n"
            f"Sanitized context: {json.dumps(self._judge_context(state), indent=2)[:7000]}\n"
            f"Draft response to customer: {redact_text(str(state.get('final_response', '')))}"
        )
        try:
            result = self.llm.invoke([("user", prompt)])
            payload = self._parse_judge_json(str(result.content))
            raw_metrics = payload.get("metrics", {})
            if not isinstance(raw_metrics, dict):
                return fallback_metrics, fallback_score, "LLM judge returned invalid metrics; deterministic fallback used."
            metrics = {
                key: 1 if int(raw_metrics.get(key, fallback_metrics[key]) or 0) == 1 else 0
                for key in expected_keys
            }
            score = round(sum(metrics.values()) / len(metrics), 3)
            reason = redact_text(str(payload.get("reason", "")))[:300] or "LLM judge completed."
            return metrics, score, reason
        except Exception as exc:
            return fallback_metrics, fallback_score, redact_text(f"LLM judge failed; deterministic fallback used: {exc}")[:300]

    def _deterministic_evaluation(self, state: BankingState) -> tuple[dict[str, int], float]:
        response = str(state.get("final_response", ""))
        route = state.get("route", "")
        risk_level = state.get("risk_level", "low")
        tools_used = set(state.get("tool_outputs", {}).keys())
        required_tools = set(state.get("required_tools", []))
        logs = state.get("logs", [])
        tool_errors = [
            item
            for item in logs
            if isinstance(item, dict)
            and (item.get("status") == "error" or item.get("error_type") or item.get("error"))
        ]
        required_runtime_tools = {
            tool for tool in required_tools if tool != "search_api_if_rag_confidence_low"
        }
        if "db_tool" in required_tools:
            required_runtime_tools.add("db_tool")
        metrics = {
            "answered_query": int(bool(response.strip())),
            "grounded_in_context": int(not self._looks_like_low_confidence_hallucination(state)),
            "route_and_tools_fit": int(route in {"general", "personalized", "calculation", "escalation"} and (required_runtime_tools.issubset(tools_used) or bool(state.get("blocked")) or risk_level == "medium")),
            "risk_guardrail_ok": int(self._risk_handled_correctly(state)),
            "pii_safe": int(redact_text(response) == response),
            "no_internal_leakage": int(not self._looks_like_internal_response(response)),
            "customer_friendly": int(bool(response.strip()) and len(response.strip()) >= 8),
            "no_error_visible": int("traceback" not in response.lower() and "exception" not in response.lower()),
        }
        if tool_errors:
            metrics["no_error_visible"] = min(metrics["no_error_visible"], int("failed:" not in response.lower()))
        score = round(sum(metrics.values()) / len(metrics), 3)
        return metrics, score

    def _parse_judge_json(self, raw_text: str) -> dict[str, Any]:
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
        return payload if isinstance(payload, dict) else {}

    def _judge_context(self, state: BankingState) -> dict[str, Any]:
        outputs = state.get("tool_outputs", {})
        context: dict[str, Any] = {
            "confidence_score": state.get("confidence_score", 0.0),
            "data_scope": state.get("data_scope", ""),
            "calculation_task": redact_value(state.get("calculation_task", {})),
        }
        if "db_tool" in outputs:
            context["db_context"] = self._safe_db_context_for_llm(str(outputs.get("db_tool", "")))
        if "calculator" in outputs:
            context["calculator"] = redact_json_text(str(outputs.get("calculator", "")))
        if "rag_retrieval" in outputs:
            context["rag_retrieval"] = redact_json_text(str(outputs.get("rag_retrieval", "")))[:2500]
        if "search_api" in outputs:
            context["search_api"] = redact_json_text(str(outputs.get("search_api", "")))[:2500]
        if "create_escalation" in outputs:
            context["escalation"] = redact_json_text(str(outputs.get("create_escalation", "")))[:1200]
        return context

    def _looks_like_low_confidence_hallucination(self, state: BankingState) -> bool:
        confidence = float(state.get("confidence_score", 0.0) or 0.0)
        response = str(state.get("final_response", "")).lower()
        if confidence >= 0.35:
            return False
        return not any(token in response for token in ["could not verify", "couldn't verify", "not verify", "try again", "escalated"])

    def _risk_handled_correctly(self, state: BankingState) -> bool:
        risk_level = state.get("risk_level", "low")
        if risk_level == "high":
            return bool(state.get("blocked")) and state.get("escalated_to") == "risk_team"
        if risk_level == "medium":
            return state.get("escalated_to") == "branch_manager"
        return not state.get("blocked")

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
                if any(token in query for token in ["list", "various", "all", "show"]):
                    return self._build_loan_list_response(loans, state)
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

    def _build_loan_list_response(self, loans: list[object], state: BankingState) -> str:
        preferences = str(state.get("behavior_preferences", "")).lower()
        wants_detail = any(
            token in preferences
            for token in [
                "concrete details",
                "fuller explanation",
                "step-by-step",
                "balance amount",
                "tenure",
                "outstanding",
            ]
        )
        wants_concise = "concise" in preferences and not wants_detail
        lines = ["Here are your loan accounts:"]
        for loan in loans:
            if not isinstance(loan, dict):
                continue
            loan_type = redact_text(str(loan.get("loantype") or loan.get("loan_type") or "Loan"))
            loan_label = loan_type if "loan" in loan_type.lower() else f"{loan_type} loan"
            status = redact_text(str(loan.get("loanstatus") or loan.get("loan_status") or "status unavailable"))
            if wants_detail and not wants_concise:
                details = [loan_label, f"status: {status}"]
                if loan.get("outstandingbalance") not in {None, ""}:
                    details.append(f"outstanding: Rs. {float(loan.get('outstandingbalance') or 0):,.2f}")
                if loan.get("emi") not in {None, ""}:
                    details.append(f"EMI: Rs. {float(loan.get('emi') or 0):,.2f}")
                if loan.get("tenuremonths") not in {None, ""}:
                    details.append(f"tenure: {loan.get('tenuremonths')} months")
                lines.append(f"- {' | '.join(details)}")
            else:
                lines.append(f"- {loan_label}: {status}")
        if wants_detail and not wants_concise:
            lines.append("Next step: ask about repayment options, prepayment impact, EMI details, or closure process for any listed loan.")
        return "\n".join(lines)

    def _build_guidance_query(self, state: BankingState, db_output: str) -> str:
        loan_types = self._extract_safe_loan_types(db_output)
        query = redact_text(state.get("user_query", ""))
        standalone = redact_text(state.get("standalone_query", "") or query)
        if loan_types:
            return f"{standalone}\nRelevant loan types: {', '.join(loan_types)}"
        return standalone

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
                f"Safe behavior preferences from prior feedback: {redact_text(str(state.get('behavior_preferences', '')))}\n"
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

    def _safe_tool_call(self, tool_name: str, argument: str, state: BankingState | None = None) -> str:
        start_ms = now_ms()
        trace_id = state.get("trace_id", "") if state else ""
        with start_span(tool_name, input_payload=redact_text(argument), metadata={"trace_id": trace_id, "tool_name": tool_name}) as span:
            try:
                result = self.mcp_client.call_tool(tool_name, argument)
                latency = elapsed_ms(start_ms)
                confidence = self._extract_confidence(result)
                span.update(output=redact_json_text(str(result))[:4000], metadata={"status": "success", "latency_ms": latency, "confidence_score": confidence})
                if state is not None:
                    self._add_step_log(state, "mcp_tool_call", status="success", tool_used=tool_name, latency_ms=latency, confidence_score=confidence)
                return result
            except Exception as exc:
                latency = elapsed_ms(start_ms)
                error_text = f"{tool_name} failed: {exc}"
                span.update(output=redact_text(error_text), metadata={"status": "error", "latency_ms": latency, "error_type": type(exc).__name__}, error=redact_text(error_text))
                if state is not None:
                    self._add_step_log(state, "mcp_tool_call", status="error", tool_used=tool_name, latency_ms=latency, error_type=type(exc).__name__, error=redact_text(error_text))
                return error_text

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
        return self._safe_tool_call("create_escalation", json.dumps(payload), state=state)

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

    def _trace_observability_metadata(self, state: BankingState) -> dict[str, Any]:
        logs = state.get("logs", [])
        step_latencies: dict[str, int] = {}
        tool_latencies: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        if isinstance(logs, list):
            for item in logs:
                if not isinstance(item, dict):
                    continue
                step = str(item.get("step", "unknown"))
                latency = item.get("latency_ms")
                if isinstance(latency, int):
                    if step == "mcp_tool_call":
                        tool_latencies.append(
                            {
                                "tool": str(item.get("tool_used", "unknown")),
                                "latency_ms": latency,
                                "status": str(item.get("status", "success")),
                            }
                        )
                    else:
                        step_latencies[step] = latency
                if item.get("status") == "error" or item.get("error_type") or item.get("error"):
                    errors.append(
                        {
                            "step": step,
                            "tool": str(item.get("tool_used", "")),
                            "error_type": str(item.get("error_type", "")),
                            "message": redact_text(str(item.get("error", "")))[:240],
                        }
                    )
        return {
            "status": "error" if errors else "success",
            "route": state.get("route", ""),
            "risk_level": state.get("risk_level", ""),
            "risk_blocked": bool(state.get("blocked")),
            "tools_used": list(state.get("tool_outputs", {}).keys()),
            "total_latency_ms": state.get("total_latency_ms", 0),
            "evaluation_score": state.get("evaluation_score", 0.0),
            "evaluation_metrics": state.get("evaluation_metrics", {}),
            "evaluation_reason": state.get("evaluation_reason", ""),
            "step_latencies_ms": step_latencies,
            "tool_latencies": tool_latencies[:20],
            "error_count": len(errors),
            "errors": errors[:5],
        }

    def _write_audit_log(self, state: BankingState) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "route": state.get("route"),
            "risk_level": state.get("risk_level"),
            "tools_used": list(state.get("tool_outputs", {}).keys()),
            "confidence_score": state.get("confidence_score", 0.0),
            "evaluation_score": state.get("evaluation_score", 0.0),
            "evaluation_metrics": state.get("evaluation_metrics", {}),
            "evaluation_reason": state.get("evaluation_reason", ""),
            "trace_id": state.get("trace_id", ""),
            "started_at": state.get("started_at", ""),
            "total_latency_ms": state.get("total_latency_ms", 0),
            "final_response": self._storage_safe_response(state),
            "logs": redact_value(state.get("logs", [])),
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def _write_evaluation_log(self, state: BankingState) -> None:
        self.evaluation_log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trace_id": state.get("trace_id", ""),
            "route": state.get("route", ""),
            "risk_level": state.get("risk_level", ""),
            "risk_reason": redact_text(str(state.get("risk_reason", ""))),
            "tools_used": list(state.get("tool_outputs", {}).keys()),
            "required_tools": state.get("required_tools", []),
            "confidence_score": state.get("confidence_score", 0.0),
            "evaluation_score": state.get("evaluation_score", 0.0),
            "evaluation_metrics": state.get("evaluation_metrics", {}),
            "evaluation_reason": redact_text(str(state.get("evaluation_reason", ""))),
            "query": redact_text(str(state.get("user_query", ""))),
            "final_response": self._storage_safe_response(state),
            "total_latency_ms": state.get("total_latency_ms", 0),
            "data_scope": state.get("data_scope", ""),
            "plan_type": state.get("plan_type", ""),
        }
        with self.evaluation_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(redact_value(record)) + "\n")

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
