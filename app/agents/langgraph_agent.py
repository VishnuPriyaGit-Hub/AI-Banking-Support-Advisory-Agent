from __future__ import annotations

import argparse
import json
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, ToolMessage

from app.core.config import get_env_value
from app.core.prompts import load_phase5_prompt
from app.mcp.client import LocalMCPClient
from app.tools.langgraph_tools import PHASE5_TOOLS
from app.tools.supabase_tool import SupabaseTool


class LangGraphBaselineAgent:
    """Minimal Phase 5 LangGraph agent with tool calling."""

    def __init__(self) -> None:
        self.model_name = get_env_value("OPENAI_MODEL") or "gpt-4o-mini"
        self.base_url = get_env_value("LLM_BASE_URL") or get_env_value("OPENAI_BASE_URL")
        self.api_key = get_env_value("LLM_API_KEY") or get_env_value("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY or LLM_API_KEY is required for the LangGraph agent.")

        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=0,
            api_key=self.api_key,
            base_url=self.base_url,
        )
        self.system_prompt = self.build_system_prompt()
        self.mcp_client = LocalMCPClient()
        self.graph = self.create_graph()

    def create_graph(self):
        try:
            from langgraph.prebuilt import create_react_agent

            return create_react_agent(
                model=self.llm,
                tools=PHASE5_TOOLS,
                prompt=self.system_prompt,
            )
        except Exception:
            return None

    def build_system_prompt(self) -> str:
        return load_phase5_prompt()

    def run(
        self,
        user_query: str,
        *,
        role: str,
        customer_id: str | None = None,
        branch: str | None = None,
        auth_user_id: str | None = None,
        user_jwt: str | None = None,
    ) -> dict[str, object]:
        normalized_role = self.normalize_role(role)
        context = {
            "role": normalized_role,
            "customer_id": customer_id or "",
            "branch": branch or "",
        }
        lowered_query = user_query.lower()
        if (
            self.graph is None
            or self.is_protected_data_query(lowered_query)
            or self.looks_like_calculation(lowered_query)
            or self.needs_context_first(lowered_query)
        ):
            content, tools_used = self.run_fallback_router(
                user_query,
                role=normalized_role,
                customer_id=customer_id,
                branch=branch,
                auth_user_id=auth_user_id,
                user_jwt=user_jwt,
            )
        else:
            prompt = (
                f"User role: {normalized_role}\n"
                f"Customer ID: {customer_id or 'not_provided'}\n"
                f"Branch: {branch or 'not_provided'}\n"
                f"Auth User ID: {auth_user_id or 'not_provided'}\n"
                f"Query: {user_query}"
            )
            result = self.graph.invoke({"messages": [("user", prompt)]})
            messages = result.get("messages", [])
            content = self.extract_final_text(messages)
            tools_used = self.extract_tools_used(messages)
        return {
            "agent": "LangGraphBaselineAgent",
            "model": self.model_name,
            "context": context,
            "response": content,
            "tools_used": tools_used,
        }

    def run_fallback_router(
        self,
        user_query: str,
        *,
        role: str,
        customer_id: str | None = None,
        branch: str | None = None,
        auth_user_id: str | None = None,
        user_jwt: str | None = None,
    ) -> tuple[str, list[str]]:
        query = user_query.lower()
        normalized_role = self.normalize_role(role)
        tools_used: list[str] = []
        authorized, auth_message = self.authorize_request(
            query,
            role=normalized_role,
            customer_id=customer_id,
            branch=branch,
        )
        if not authorized:
            return auth_message, []

        if self.looks_like_calculation(query) and not self.needs_context_first(query):
            expression = self.derive_calculator_expression(user_query, "")
            if expression:
                try:
                    result = self.mcp_client.call_tool("calculator", expression)
                    final_answer = self.llm.invoke(
                        [
                            ("system", self.system_prompt),
                            ("user", f"Role: {normalized_role}\nQuery: {user_query}\n\nCalculator expression:\n{expression}\n\nCalculator result:\n{result}\n\nProvide a short, customer-friendly final answer."),
                        ]
                    )
                    return str(final_answer.content), ["calculator"]
                except Exception:
                    pass

        if self.is_protected_data_query(query):
            if normalized_role == "customer" and auth_user_id and user_jwt:
                client = SupabaseTool(user_jwt=user_jwt)
                rows = client.get_customer_by_auth_user(auth_user_id)
                if not rows:
                    return "You are authenticated, but no customer record is linked to your account.", []
                result = client.get_customer_snapshot(rows[0].get("customerid", ""))
                return result, ["supabase_customer_snapshot"]
            if normalized_role == "manager" and user_jwt:
                client = SupabaseTool(user_jwt=user_jwt)
                result = client.get_all_customers()
                return result, ["supabase_branch_customers"]
            if normalized_role in {"admin", "support", "risk"} and user_jwt:
                client = SupabaseTool(user_jwt=user_jwt)
                result = client.get_all_customers()
                return result, ["supabase_all_customers"]

        rag_result = self.mcp_client.call_tool("rag_retrieval", user_query)
        tools_used.append("rag_retrieval")
        rag_payload = self.safe_json_loads(rag_result)
        search_query = self.build_search_query(user_query, query)
        if self.is_comparison_query(query):
            search_result = self.mcp_client.call_tool("search_api", search_query)
            tools_used.append("search_api")
            tool_output = f"Internal RAG context:\n{rag_result}\n\nExternal search context:\n{search_result}"
        elif self.should_fallback_to_search(query, rag_payload):
            search_result = self.mcp_client.call_tool("search_api", search_query)
            tools_used.append("search_api")
            tool_output = f"Internal RAG context was weak or insufficient.\nExternal search context:\n{search_result}"
        else:
            tool_output = f"Internal RAG context:\n{rag_result}"

        if self.needs_formula_calculation(query, tools_used) and "search_api" not in tools_used:
            search_result = self.mcp_client.call_tool("search_api", search_query)
            tools_used.append("search_api")
            tool_output = f"{tool_output}\n\nExternal search context:\n{search_result}"

        calculator_result = ""
        if self.needs_formula_calculation(query, tools_used):
            expression = self.derive_calculator_expression(user_query, tool_output)
            if expression:
                try:
                    calculator_result = self.mcp_client.call_tool("calculator", expression)
                except Exception:
                    calculator_result = ""
                if calculator_result:
                    tools_used.append("calculator")
                    tool_output = f"{tool_output}\n\nCalculated result:\n{calculator_result}"

        final_answer = self.llm.invoke(
            [
                ("system", self.system_prompt),
                ("user", f"Role: {normalized_role}\nCustomer ID: {customer_id or 'not_provided'}\nBranch: {branch or 'not_provided'}\nQuery: {user_query}\n\nTool output:\n{tool_output}"),
            ]
        )
        return str(final_answer.content), tools_used

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

    def looks_like_calculation(self, query: str) -> bool:
        has_number = bool(re.search(r"\d", query))
        math_tokens = [
            "+",
            "-",
            "*",
            "/",
            "%",
            "emi",
            "monthly payment",
            "interest on",
            "interest amount",
            "maturity",
            "maturity amount",
            "total amount",
            "sum of",
            "difference",
            "multiply",
            "divided",
            "per month",
        ]
        return has_number and any(token in query for token in math_tokens)

    def needs_context_first(self, query: str) -> bool:
        return any(
            token in query
            for token in [
                "fd",
                "fixed deposit",
                "rd",
                "recurring deposit",
                "interest rate",
                "maturity",
                "tenure",
                "rate",
            ]
        )

    def needs_formula_calculation(self, query: str, tools_used: list[str]) -> bool:
        if "search_api" not in tools_used and "rag_retrieval" not in tools_used:
            return False
        has_number = bool(re.search(r"\d", query))
        formula_tokens = [
            "emi",
            "interest",
            "maturity",
            "monthly payment",
            "calculate",
            "rate",
            "tenure",
            "return",
            "amount",
        ]
        return has_number and any(token in query for token in formula_tokens)

    def build_search_query(self, user_query: str, lowered_query: str) -> str:
        if self.looks_like_calculation(lowered_query):
            return f"{user_query} formula"
        return user_query

    def is_protected_data_query(self, query: str) -> bool:
        return any(
            token in query
            for token in [
                "transaction",
                "transactions",
                "balance",
                "loan",
                "customer",
                "customers",
                "account summary",
                "my account",
                "branch",
                "how many",
                "count",
            ]
        )

    def is_comparison_query(self, query: str) -> bool:
        return any(token in query for token in ["compare", "comparison", "vs", "versus", "better than", "difference between"])

    def safe_json_loads(self, text: str) -> dict[str, object]:
        try:
            return json.loads(text)
        except Exception:
            return {}

    def derive_calculator_expression(self, user_query: str, tool_output: str) -> str:
        prompt = (
            "Convert the banking calculation request into exactly one plain arithmetic expression for a calculator tool. "
            "Output only the expression text, nothing else. "
            "Use only numbers, decimal points, parentheses, +, -, *, /, and ** when needed. "
            "Do not include words, variable names, currency symbols, commas, equals signs, markdown, or explanations. "
            "If there is not enough information to form a calculation, return NONE.\n\n"
            f"User query:\n{user_query}\n\n"
            f"Available context:\n{tool_output}"
        )
        response = self.llm.invoke([("user", prompt)])
        expression = self.extract_expression(str(response.content))
        if not expression or expression.upper() == "NONE":
            return ""
        return expression

    def extract_expression(self, raw_text: str) -> str:
        cleaned = raw_text.replace("`", "").replace(",", "").strip()
        if cleaned.upper() == "NONE":
            return ""

        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        candidates = lines or [cleaned]
        pattern = re.compile(r"[\d\.\+\-\*\/\(\)\s]+")

        for candidate in candidates:
            normalized = candidate.replace("^", "**").strip()
            if re.fullmatch(r"[\d\s\.\+\-\*\/\(\)]+", normalized):
                return normalized

            matches = pattern.findall(normalized)
            for match in sorted(matches, key=len, reverse=True):
                expression = " ".join(match.split()).replace("^", "**").strip()
                if len(expression) >= 3 and re.fullmatch(r"[\d\s\.\+\-\*\/\(\)]+", expression):
                    return expression
        return ""

    def rag_has_no_results(self, payload: dict[str, object]) -> bool:
        matches = payload.get("matches", [])
        return not isinstance(matches, list) or len(matches) == 0

    def should_fallback_to_search(self, query: str, payload: dict[str, object]) -> bool:
        if self.rag_has_no_results(payload):
            return True
        try:
            confidence = float(payload.get("confidence_score", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < 0.6:
            return True

        matches = payload.get("matches", [])
        if not isinstance(matches, list):
            return True
        match_text = " ".join(str(match.get("text", "")) for match in matches if isinstance(match, dict)).lower()
        important_terms = [term for term in re.findall(r"[a-zA-Z]{4,}", query) if term not in {"what", "rate", "with", "that", "this", "from", "your", "have"}]
        if important_terms and not any(term in match_text for term in important_terms):
            return True
        return False

    def authorize_request(
        self,
        query: str,
        *,
        role: str,
        customer_id: str | None = None,
        branch: str | None = None,
    ) -> tuple[bool, str]:
        if role not in {"customer", "manager", "support", "risk", "admin"}:
            return False, "Your role is not authorized for this operation."
        if role == "customer" and any(
            token in query
            for token in [
                "all customers",
                "branch customer",
                "branch customers",
                "another customer",
                "other customer",
                "other customers",
                "someone else's",
            ]
        ):
            return False, "You are not authorized to view other customers' information."
        if role == "customer":
            if not customer_id:
                return False, "Your customer profile is not linked to this session."
            return True, ""
        if role == "manager" and not branch and any(token in query for token in ["customer", "branch", "transaction", "loan", "balance"]):
            return False, "You are not authorized because your branch context is missing."
        if role == "support" and any(token in query for token in ["delete user", "remove customer", "add customer"]):
            return False, "Support is not authorized for add/delete operations."
        return True, ""

    def extract_final_text(self, messages: list[object]) -> str:
        for message in reversed(messages):
            if isinstance(message, AIMessage) and message.content:
                if isinstance(message.content, str):
                    return message.content
                if isinstance(message.content, list):
                    return "".join(
                        item.get("text", "") for item in message.content if isinstance(item, dict)
                    ).strip()
        return ""

    def extract_tools_used(self, messages: list[object]) -> list[str]:
        tools_used: list[str] = []
        for message in messages:
            if isinstance(message, ToolMessage):
                name = getattr(message, "name", "") or ""
                if name and name not in tools_used:
                    tools_used.append(name)
        return tools_used


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 5 LangGraph banking agent.")
    parser.add_argument("--role", required=True, help="User role such as customer, manager, admin, support, or risk.")
    parser.add_argument("--query", required=True, help="User query for the agent.")
    parser.add_argument("--customer-id", help="Customer ID when applicable.")
    parser.add_argument("--branch", help="Branch name when applicable.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent = LangGraphBaselineAgent()
    result = agent.run(
        args.query,
        role=args.role,
        customer_id=args.customer_id,
        branch=args.branch,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
