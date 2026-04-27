from __future__ import annotations

from app.tools.calculator import calculator_tool
from app.tools.rag_tool import rag_retrieval_tool
from app.tools.searchapi_tool import search_api_tool
from app.tools.supabase_tool import (
    add_customer_tool,
    delete_customer_tool,
    get_all_customers_tool,
    get_branch_customers_tool,
    get_customer_snapshot_tool,
    update_customer_contact_tool,
)


def get_tool_registry() -> dict[str, object]:
    return {
        "calculator": calculator_tool,
        "supabase_customer_snapshot": get_customer_snapshot_tool,
        "supabase_branch_customers": get_branch_customers_tool,
        "supabase_all_customers": get_all_customers_tool,
        "supabase_update_contact": update_customer_contact_tool,
        "supabase_add_customer": add_customer_tool,
        "supabase_delete_customer": delete_customer_tool,
        "search_api": search_api_tool,
        "rag_retrieval": rag_retrieval_tool,
    }
