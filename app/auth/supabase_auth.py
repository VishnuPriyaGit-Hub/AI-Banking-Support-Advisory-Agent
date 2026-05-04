from __future__ import annotations

import json
from urllib import error, parse, request
from urllib.parse import urlparse

from app.core.config import get_env_value
from app.tools.supabase_tool import SupabaseTool


def _clean_profile_value(value: object) -> str:
    text = str(value or "").strip()
    return "" if text.upper() in {"EMPTY", "NULL", "NONE"} else text


class SupabaseAuthClient:
    def __init__(self) -> None:
        self.url = get_env_value("SUPABASE_URL")
        self.anon_key = get_env_value("SUPABASE_ANON_KEY")
        if not self.url or not self.anon_key:
            raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY are required.")
        parsed_url = urlparse(self.url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError("SUPABASE_URL must be a valid URL such as https://project-ref.supabase.co.")

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
        except error.URLError as exc:
            raise RuntimeError(
                "Could not reach Supabase Auth. Check internet/DNS access, VPN/proxy/firewall settings, "
                "and that SUPABASE_URL points to your active Supabase project."
            ) from exc

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
        if not customer_rows:
            customer_rows = admin_tool._request(  # noqa: SLF001
                "GET",
                "customers",
                params={"authuserid": f"eq.{user.get('id')}", "select": "*"},
                use_service_role=True,
            )
        if not customer_rows and user.get("email"):
            customer_rows = admin_tool._request(  # noqa: SLF001
                "GET",
                "customers",
                params={"email": f"eq.{user.get('email')}", "select": "*"},
                use_service_role=True,
            )
        customer_row = customer_rows[0] if isinstance(customer_rows, list) and customer_rows else {}
        customer_id = _clean_profile_value(customer_row.get("customerid", ""))
        customer_name = _clean_profile_value(customer_row.get("customername", user.get("email", "")))
        role = _clean_profile_value(role_row.get("role", ""))
        if not role and customer_id:
            role = "customer"
        branch = _clean_profile_value(role_row.get("branch", ""))
        if not branch:
            branch = _clean_profile_value(customer_row.get("branch", ""))

        return {
            "id": str(user.get("id", "")),
            "email": user.get("email", ""),
            "access_token": access_token,
            "role": role,
            "branch": branch,
            "customer_id": customer_id,
            "customerid": customer_id,
            "customer_name": customer_name,
            "customername": customer_name,
        }
