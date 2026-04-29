from __future__ import annotations

import ast
import json
import operator

OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def evaluate_expression(expression: str) -> float:
    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
            return OPS[type(node.op)](_eval(node.operand))
        raise ValueError("Unsupported expression.")

    tree = ast.parse(expression, mode="eval")
    return round(_eval(tree.body), 2)


def calculator_tool(expression: str) -> str:
    structured = try_structured_calculation(expression)
    if structured:
        return structured
    result = evaluate_expression(expression)
    return f"Calculated result: {result}"


def try_structured_calculation(payload: str) -> str:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return ""
    if not isinstance(data, dict):
        return ""

    operation = str(data.get("operation", "")).lower()
    if operation == "emi":
        principal = float(data["principal"])
        annual_rate = float(data["annual_rate"])
        tenure_months = int(data["tenure_months"])
        monthly_rate = annual_rate / 12 / 100
        if monthly_rate == 0:
            emi = principal / tenure_months
        else:
            emi = principal * monthly_rate * ((1 + monthly_rate) ** tenure_months) / (((1 + monthly_rate) ** tenure_months) - 1)
        total_payment = emi * tenure_months
        total_interest = total_payment - principal
        return json.dumps(
            {
                "operation": "emi",
                "emi": round(emi, 2),
                "total_payment": round(total_payment, 2),
                "total_interest": round(total_interest, 2),
            },
            indent=2,
        )

    if operation == "simple_interest":
        principal = float(data["principal"])
        annual_rate = float(data["annual_rate"])
        years = float(data["years"])
        interest = principal * annual_rate * years / 100
        return json.dumps(
            {
                "operation": "simple_interest",
                "interest": round(interest, 2),
                "maturity_amount": round(principal + interest, 2),
            },
            indent=2,
        )

    if operation == "eligibility":
        monthly_income = float(data["monthly_income"])
        existing_emi = float(data.get("existing_emi", 0))
        max_foir = float(data.get("max_foir", 0.5))
        eligible_emi = max((monthly_income * max_foir) - existing_emi, 0)
        return json.dumps(
            {
                "operation": "eligibility",
                "eligible_monthly_emi": round(eligible_emi, 2),
                "foir_used": max_foir,
            },
            indent=2,
        )

    if operation == "balance_summary":
        balance = float(data.get("balance", 0))
        credits = sum(float(item) for item in data.get("credits", []))
        debits = sum(float(item) for item in data.get("debits", []))
        return json.dumps(
            {
                "operation": "balance_summary",
                "opening_or_current_balance": round(balance, 2),
                "total_credits": round(credits, 2),
                "total_debits": round(debits, 2),
                "net_movement": round(credits - debits, 2),
            },
            indent=2,
        )

    return ""
