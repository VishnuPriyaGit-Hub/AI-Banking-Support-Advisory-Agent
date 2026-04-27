from __future__ import annotations

import json
from urllib import error, parse, request

from app.core.config import get_env_value


class SupabaseTool:
    """Minimal Supabase REST tool for Phase 5."""

    def __init__(self, *, user_jwt: str | None = None) -> None:
        self.url = get_env_value("SUPABASE_URL")
        self.anon_key = get_env_value("SUPABASE_ANON_KEY")
        self.service_role_key = get_env_value("SUPABASE_SERVICE_ROLE_KEY")
        self.user_jwt = user_jwt

        if not self.url:
            raise ValueError("SUPABASE_URL is required.")
        if not self.anon_key and not self.service_role_key:
            raise ValueError("SUPABASE_ANON_KEY or SUPABASE_SERVICE_ROLE_KEY is required.")

    def _headers(self, *, use_service_role: bool = False) -> dict[str, str]:
        if use_service_role:
            if not self.service_role_key:
                raise ValueError("SUPABASE_SERVICE_ROLE_KEY is required for this operation.")
            api_key = self.service_role_key
            bearer = self.service_role_key
        else:
            api_key = self.anon_key or self.service_role_key
            bearer = self.user_jwt or api_key
        if not api_key or not bearer:
            raise ValueError("Supabase credentials are not configured correctly.")
        return {
            "apikey": api_key,
            "Authorization": f"Bearer {bearer}",
            "Content-Type": "application/json",
        }

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
        body: dict | None = None,
        use_service_role: bool = False,
    ) -> list[dict] | dict:
        url = f"{self.url.rstrip('/')}/rest/v1/{path}"
        if params:
            url = f"{url}?{parse.urlencode(params)}"
        raw = request.Request(
            url,
            data=json.dumps(body).encode("utf-8") if body is not None else None,
            headers=self._headers(use_service_role=use_service_role),
            method=method,
        )
        try:
            with request.urlopen(raw, timeout=30) as response:
                text = response.read().decode("utf-8")
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Supabase request failed with HTTP {exc.code}: {details}") from exc
        return json.loads(text) if text else {}

    def get_customer_by_auth_user(self, auth_user_id: str) -> list[dict]:
        return self._request(
            "GET",
            "customers",
            params={"authuserid": f"eq.{auth_user_id}", "select": "*"},
        )

    def get_customer_snapshot(self, customer_id: str) -> dict[str, object]:
        customer = self._request("GET", "customers", params={"customerid": f"eq.{customer_id}", "select": "*"})
        loans = self._request("GET", "loanaccounts", params={"customerid": f"eq.{customer_id}", "select": "*"})
        transactions = self._request(
            "GET",
            "transactions",
            params={
                "customerid": f"eq.{customer_id}",
                "select": "*",
                "order": "transactiondate.desc",
                "limit": "5",
            },
        )
        return {
            "customer": customer,
            "loans": loans,
            "transactions": transactions,
        }

    def get_branch_customers(self, branch: str) -> list[dict]:
        return self._request(
            "GET",
            "customers",
            params={"branch": f"eq.{branch}", "select": "*", "order": "customername.asc"},
        )

    def get_all_customers(self) -> list[dict]:
        return self._request("GET", "customers", params={"select": "*", "order": "customername.asc"})

    def add_customer(self, payload: dict) -> dict | list[dict]:
        return self._request("POST", "customers", body=payload, use_service_role=True)

    def delete_customer(self, customer_id: str) -> dict | list[dict]:
        return self._request(
            "DELETE",
            "customers",
            params={"customerid": f"eq.{customer_id}"},
            use_service_role=True,
        )

    def update_customer_contact(self, customer_id: str, payload: dict) -> dict | list[dict]:
        mapping = {
            "CustomerName": "customername",
            "Address": "address",
            "City": "city",
            "State": "state",
        }
        allowed = {mapping[key]: value for key, value in payload.items() if key in mapping}
        return self._request(
            "PATCH",
            "customers",
            params={"customerid": f"eq.{customer_id}"},
            body=allowed,
            use_service_role=True,
        )

    def add_customer(self, payload: dict) -> dict | list[dict]:
        return self._request("POST", "customers", body=payload, use_service_role=True)

    def delete_customer(self, customer_id: str) -> dict | list[dict]:
        return self._request(
            "DELETE",
            "customers",
            params={"customerid": f"eq.{customer_id}"},
            use_service_role=True,
        )


def get_customer_snapshot_tool(customer_id: str) -> str:
    client = SupabaseTool()
    return json.dumps(client.get_customer_snapshot(customer_id), indent=2)


def get_branch_customers_tool(branch: str) -> str:
    client = SupabaseTool()
    return json.dumps(client.get_branch_customers(branch), indent=2)


def get_all_customers_tool(_: str = "") -> str:
    client = SupabaseTool()
    return json.dumps(client.get_all_customers(), indent=2)


def update_customer_contact_tool(payload_json: str) -> str:
    payload = json.loads(payload_json)
    customer_id = payload.pop("CustomerID")
    client = SupabaseTool()
    return json.dumps(client.update_customer_contact(customer_id, payload), indent=2)


def add_customer_tool(payload_json: str) -> str:
    payload = json.loads(payload_json)
    client = SupabaseTool()
    return json.dumps(client.add_customer(payload), indent=2)


def delete_customer_tool(customer_id: str) -> str:
    client = SupabaseTool()
    return json.dumps(client.delete_customer(customer_id), indent=2)
