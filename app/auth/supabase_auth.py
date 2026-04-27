from __future__ import annotations

import json
from urllib import error, parse, request

from app.core.config import get_env_value
from app.tools.supabase_tool import SupabaseTool


class SupabaseAuthClient:
    def __init__(self) -> None:
        self.url = get_env_value("SUPABASE_URL")
        self.anon_key = get_env_value("SUPABASE_ANON_KEY")
        if not self.url or not self.anon_key:
            raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY are required.")

    def sign_in(self, email: str, password: str) -> dict[str, object]:
        url = f"{self.url.rstrip('/')}/auth/v1/token?grant_type=password"
        raw = request.Request(
            url,
            data=json.dumps({"email": email, "password": password}).encode("utf-8"),
            headers={
                "apikey": self.anon_key,
                "Authorization": f"Bearer {self.anon_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(raw, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Supabase auth failed with HTTP {exc.code}: {details}") from exc

        user = payload.get("user", {})
        access_token = payload.get("access_token", "")
        if not user or not access_token:
            raise RuntimeError("Supabase auth did not return a valid user session.")

        tool = SupabaseTool(user_jwt=access_token)
        admin_tool = SupabaseTool()
        role_rows = admin_tool._request(  # noqa: SLF001
            "GET",
            "userroles",
            params={"user_id": f"eq.{user.get('id')}", "select": "*"},
            use_service_role=True,
        )
        role_row = role_rows[0] if isinstance(role_rows, list) and role_rows else {}
        customer_rows = tool.get_customer_by_auth_user(str(user.get("id")))
        customer_row = customer_rows[0] if isinstance(customer_rows, list) and customer_rows else {}

        return {
            "id": str(user.get("id", "")),
            "email": user.get("email", ""),
            "access_token": access_token,
            "role": role_row.get("role", ""),
            "branch": role_row.get("branch", ""),
            "customer_id": customer_row.get("customerid", ""),
            "customer_name": customer_row.get("customername", user.get("email", "")),
        }
