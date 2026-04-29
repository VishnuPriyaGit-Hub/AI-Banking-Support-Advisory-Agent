from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib import error, parse, request

from app.core.config import DATA_DIR, get_env_value
from app.security.pii import hash_identifier, redact_text


MEMORY_PATH = DATA_DIR / "conversation_memory.jsonl"


class ConversationMemory:
    """Short-term session memory plus optional Supabase long-term memory."""

    def __init__(self, *, user_jwt: str | None = None) -> None:
        self.user_jwt = user_jwt
        self.url = get_env_value("SUPABASE_URL")
        self.anon_key = get_env_value("SUPABASE_ANON_KEY")

    def short_context(self, chat_history: list[dict[str, str]] | None, limit: int = 6) -> list[dict[str, str]]:
        return [item for item in (chat_history or []) if item.get("text")][-limit:]

    def save_turn(
        self,
        *,
        user_id: str,
        role: str,
        query: str,
        response: str,
        route: str,
        risk_level: str,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        record = {
            "user_id": hash_identifier(user_id),
            "role": role,
            "query": redact_text(query),
            "response": redact_text(response),
            "route": route,
            "risk_level": risk_level,
            "last_active_at": now,
            "created_at": now,
        }
        self._append_local(record)
        self._save_supabase(record)

    def prune_inactive(self, days: int = 60) -> int:
        if not MEMORY_PATH.exists():
            return 0
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        kept: list[dict[str, object]] = []
        removed = 0
        for line in MEMORY_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                last_active = datetime.fromisoformat(str(item.get("last_active_at", "")).replace("Z", "+00:00"))
            except Exception:
                removed += 1
                continue
            if last_active < cutoff:
                removed += 1
            else:
                kept.append(item)
        MEMORY_PATH.write_text("\n".join(json.dumps(item) for item in kept) + ("\n" if kept else ""), encoding="utf-8")
        self._prune_supabase(days)
        return removed

    def delete_user_memory(self, user_id: str) -> int:
        removed = 0
        kept: list[dict[str, object]] = []
        if MEMORY_PATH.exists():
            for line in MEMORY_PATH.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if str(item.get("user_id", "")) == hash_identifier(user_id):
                    removed += 1
                else:
                    kept.append(item)
            MEMORY_PATH.write_text("\n".join(json.dumps(item) for item in kept) + ("\n" if kept else ""), encoding="utf-8")
        self._delete_supabase_user_memory(hash_identifier(user_id))
        return removed

    def _append_local(self, record: dict[str, object]) -> None:
        MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with MEMORY_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def _headers(self) -> dict[str, str] | None:
        if not self.url or not self.anon_key:
            return None
        bearer = self.user_jwt or self.anon_key
        return {
            "apikey": self.anon_key,
            "Authorization": f"Bearer {bearer}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        }

    def _save_supabase(self, record: dict[str, object]) -> None:
        headers = self._headers()
        if not headers or not self.url:
            return
        url = f"{self.url.rstrip('/')}/rest/v1/conversation_memory"
        payload = {
            "user_id": record["user_id"],
            "role": record["role"],
            "query": record["query"],
            "response": record["response"],
            "route": record["route"],
            "risk_level": record["risk_level"],
            "last_active_at": record["last_active_at"],
        }
        raw = request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        try:
            request.urlopen(raw, timeout=15).close()
        except (error.HTTPError, error.URLError):
            return

    def _prune_supabase(self, days: int) -> None:
        headers = self._headers()
        if not headers or not self.url:
            return
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        url = f"{self.url.rstrip('/')}/rest/v1/conversation_memory?{parse.urlencode({'last_active_at': f'lt.{cutoff}'})}"
        raw = request.Request(url, headers=headers, method="DELETE")
        try:
            request.urlopen(raw, timeout=15).close()
        except (error.HTTPError, error.URLError):
            return

    def _delete_supabase_user_memory(self, user_id: str) -> None:
        headers = self._headers()
        if not headers or not self.url:
            return
        url = f"{self.url.rstrip('/')}/rest/v1/conversation_memory?{parse.urlencode({'user_id': f'eq.{user_id}'})}"
        raw = request.Request(url, headers=headers, method="DELETE")
        try:
            request.urlopen(raw, timeout=15).close()
        except (error.HTTPError, error.URLError):
            return
