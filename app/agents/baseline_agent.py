from __future__ import annotations

import argparse
from pathlib import Path

from pydantic import ValidationError

from app.core.config import DEFAULT_LOG_PATH, DEMO_LOG_PATH
from app.models.agent import AgentRunResult, ClassificationResult, UserInput

ROLE_RESPONSES = {
    "Customer": {
        "safe": {
            "emi": "EMI means Equated Monthly Installment. It is the fixed amount paid every month for a loan.",
            "loan": "A loan is money borrowed from the bank and repaid in monthly installments with interest.",
            "fixed deposit": "A fixed deposit lets you keep money for a fixed time and earn interest on it.",
            "credit card": "A credit card lets you spend up to a limit and repay later as per the billing cycle.",
            "default": "I can help with general banking information.",
        },
        "ambiguous": "Please tell me clearly whether you want information about loans, EMI, fixed deposits, or credit cards.",
        "disallowed": "I cannot perform transactions or access account details. Please use the official bank app, website, or branch.",
        "high_risk": "This looks like a fraud or security issue. Please contact official bank support immediately and do not share OTP, PIN, or passwords.",
        "general": "Hello. I can help with general banking questions like loans, EMI, fixed deposits, and card information.",
        "fallback": "I could not understand the request clearly. Please rephrase your banking question.",
    },
    "Branch Manager": {
        "safe": "Provide short banking guidance and remind the user that final terms depend on bank policy.",
        "ambiguous": "Clarify the exact banking need before proceeding further.",
        "disallowed": "Redirect this transactional or account-specific request to the official servicing channel.",
        "high_risk": "Treat this as a high-risk case and escalate it through the proper fraud-support path.",
        "general": "Ask the user how you can help with banking support.",
        "fallback": "Manual review may be needed because the request is not clearly classified.",
    },
    "Risk & Compliance Officer": {
        "safe": "Low-risk informational request. Share policy-safe banking guidance only.",
        "ambiguous": "Intent is unclear. Keep the response conservative and ask for clarification.",
        "disallowed": "This request crosses policy boundaries because it asks for restricted action or customer-specific access.",
        "high_risk": "Classify this as security-sensitive and escalate immediately without speculation.",
        "general": "Ready to review fraud, security, and policy-related banking questions.",
        "fallback": "No clear rule match. Use a conservative policy-safe response.",
    },
    "Admin": {
        "safe": "Admin view: safe informational banking request.",
        "ambiguous": "Admin view: ambiguous request, clarification required.",
        "disallowed": "Admin view: disallowed request, refuse and redirect.",
        "high_risk": "Admin view: high-risk request, escalate immediately.",
        "general": "Admin view: general banking support request.",
        "fallback": "Admin view: fallback response triggered.",
    },
    "Customer Support Agent": {
        "safe": "Suggested reply: I can share general information about the product and process.",
        "ambiguous": "Suggested reply: Please let me know exactly which banking product or process you want help with.",
        "disallowed": "Suggested reply: I can guide you, but I cannot perform transactions or access account details in chat.",
        "high_risk": "Suggested reply: Please contact official fraud support immediately and do not share OTP, PIN, CVV, or passwords.",
        "general": "Suggested reply: I can help with general banking questions.",
        "fallback": "Suggested reply: Please rephrase the request so I can guide you correctly.",
    },
}

PRODUCT_RULES = {
    "emi": ["emi", "installment"],
    "loan": ["loan", "home loan", "personal loan", "car loan"],
    "fixed deposit": ["fixed deposit", "fd", "deposit"],
    "credit card": ["credit card", "card"],
}

DISALLOWED_PATTERNS = [
    "transfer",
    "send money",
    "pay bill",
    "balance",
    "my account",
    "approve loan",
]

HIGH_RISK_PATTERNS = [
    "fraud",
    "scam",
    "otp",
    "hacked",
    "unauthorized",
    "money got deducted",
    "phishing",
]


class BaselineAgent:
    """Simple rule-based banking support agent."""

    def classify_query(self, user_query: UserInput) -> ClassificationResult:
        query = user_query.query.lower()
        tokens = query.replace("?", " ").replace(".", " ").replace(",", " ").split()

        if any(word in tokens for word in ["hello", "hi", "hey"]):
            return ClassificationResult(category="general", guidance="Greeting detected.")

        if any(pattern in query for pattern in HIGH_RISK_PATTERNS):
            return ClassificationResult(category="high_risk", guidance="High-risk issue detected.")

        if any(pattern in query for pattern in DISALLOWED_PATTERNS):
            return ClassificationResult(category="disallowed", guidance="Restricted request detected.")

        for product, patterns in PRODUCT_RULES.items():
            if any(pattern in query for pattern in patterns):
                if "tell me about" in query:
                    return ClassificationResult(category="ambiguous", guidance="Needs simple product guidance.", product_hint=product)
                return ClassificationResult(category="safe", guidance="Safe informational banking query.", product_hint=product)

        return ClassificationResult(category="fallback", guidance="No rule matched.")

    def generate_response(self, user_query: UserInput, classification: ClassificationResult) -> str:
        role_templates = ROLE_RESPONSES[user_query.role]

        if classification.category == "safe":
            return role_templates["safe"].get(classification.product_hint, role_templates["safe"]["default"])

        return role_templates.get(classification.category, role_templates["fallback"])

    def run(self, user_query: UserInput) -> AgentRunResult:
        classification = self.classify_query(user_query)
        response_text = self.generate_response(user_query, classification)
        return AgentRunResult(
            input={"role": user_query.role, "query": user_query.query},
            output=response_text,
            metadata={"category": classification.category},
        )


def log_result(log_path: Path, result: AgentRunResult) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(result.model_dump_json() + "\n")


def run_cli(log_path: Path) -> None:
    agent = BaselineAgent()
    print("Simple Baseline Banking Agent")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        raw_role = input(
            "Role (Customer / Branch Manager / Risk & Compliance Officer / Admin / Customer Support Agent): "
        )
        if raw_role.strip().lower() in {"exit", "quit"}:
            print("Agent: Goodbye.")
            break

        raw_query = input("Query: ")
        if raw_query.strip().lower() in {"exit", "quit"}:
            print("Agent: Goodbye.")
            break

        try:
            request = UserInput(role=raw_role, query=raw_query)
        except ValidationError:
            print("Agent: Please enter a valid role and query.")
            continue

        result = agent.run(request)
        print(f"Agent: {result.output}\n")
        log_result(log_path, result)


def run_demo(log_path: Path) -> None:
    agent = BaselineAgent()
    demo_inputs = [
        {"role": "Customer", "query": "What is EMI?"},
        {"role": "Customer", "query": "Transfer 1000 to this account"},
        {"role": "Customer Support Agent", "query": "Money got deducted but I did not do it"},
    ]

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")

    for item in demo_inputs:
        result = agent.run(UserInput(**item))
        log_result(log_path, result)
        print(f"{item['role']} | {item['query']}")
        print(f"Agent: {result.output}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A simple baseline banking agent.")
    parser.add_argument("--demo", action="store_true", help="Run demo queries.")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG_PATH, help="Path to the JSONL log file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_log = args.log if not args.demo or args.log != DEFAULT_LOG_PATH else DEMO_LOG_PATH
    if args.demo:
        run_demo(target_log)
    else:
        run_cli(target_log)


if __name__ == "__main__":
    main()
