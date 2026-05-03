from __future__ import annotations

from langchain_core.tools import tool

from app.mcp.client import LocalMCPClient

CLIENT = LocalMCPClient()


@tool
def calculator(expression: str) -> str:
    """Use this tool for arithmetic calculations such as EMI math, sums, differences, percentages, or amount calculations."""
    return CLIENT.call_tool("calculator", expression)


@tool
def supabase_customer_snapshot(customer_id: str) -> str:
    """Fetch one customer's profile, loan accounts, and latest 5 transactions from Supabase using CustomerID."""
    return CLIENT.call_tool("supabase_customer_snapshot", customer_id)


@tool
def supabase_customer_transactions(customer_id: str) -> str:
    """Fetch one customer's recent transactions from Supabase using CustomerID. Does not fetch profile or loan records."""
    return CLIENT.call_tool("supabase_customer_transactions", customer_id)


@tool
def supabase_branch_customers(branch: str) -> str:
    """Fetch all customers for a branch manager's branch from Supabase using the branch name."""
    return CLIENT.call_tool("supabase_branch_customers", branch)


@tool
def supabase_branch_loan_customers(branch: str) -> str:
    """Fetch customers in a manager's branch who have loan accounts, including loan type and status."""
    return CLIENT.call_tool("supabase_branch_loan_customers", branch)


@tool
def supabase_all_customers(_: str = "") -> str:
    """Fetch all customers. Use only for admin, support, or risk roles."""
    return CLIENT.call_tool("supabase_all_customers", _)


@tool
def supabase_update_contact(payload_json: str) -> str:
    """Update only CustomerName, Address, City, or State in Supabase. Input must be a JSON string with CustomerID and allowed fields."""
    return CLIENT.call_tool("supabase_update_contact", payload_json)


@tool
def supabase_add_customer(payload_json: str) -> str:
    """Admin-only tool. Add a new customer row in Supabase. Input must be a JSON string matching the Customers table."""
    return CLIENT.call_tool("supabase_add_customer", payload_json)


@tool
def supabase_delete_customer(customer_id: str) -> str:
    """Admin-only tool. Delete a customer in Supabase using CustomerID."""
    return CLIENT.call_tool("supabase_delete_customer", customer_id)


@tool
def search_api(query: str) -> str:
    """Search external banking information when the answer is not in internal documents or customer database."""
    return CLIENT.call_tool("search_api", query)


@tool
def rag_retrieval(query: str) -> str:
    """Retrieve internal banking document context from the Phase 4 RAG knowledge base."""
    return CLIENT.call_tool("rag_retrieval", query)


@tool
def create_escalation(payload_json: str) -> str:
    """Create a medium-risk branch-manager or high-risk risk-team escalation."""
    return CLIENT.call_tool("create_escalation", payload_json)


@tool
def update_escalation(payload_json: str) -> str:
    """Update an escalation with a manager decision and customer-facing response."""
    return CLIENT.call_tool("update_escalation", payload_json)


@tool
def complete_escalation(payload_json: str) -> str:
    """Mark an approved escalation action as completed or failed after staff execution."""
    return CLIENT.call_tool("complete_escalation", payload_json)


PHASE5_TOOLS = [
    calculator,
    supabase_customer_snapshot,
    supabase_customer_transactions,
    supabase_branch_customers,
    supabase_branch_loan_customers,
    supabase_all_customers,
    supabase_update_contact,
    supabase_add_customer,
    supabase_delete_customer,
    search_api,
    rag_retrieval,
    create_escalation,
    update_escalation,
    complete_escalation,
]
