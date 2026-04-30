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

    def save_feedback(
        self,
        *,
        user_id: str,
        role: str,
        query: str,
        response: str,
        route: str,
        risk_level: str,
        rating: str,
        tags: list[str],
        comment: str,
    ) -> dict[str, object]:
        now = datetime.now(timezone.utc).isoformat()
        preference_summary = self._derive_preference_summary(rating=rating, tags=tags, role=role)
        record = {
            "entry_type": "feedback",
            "user_id": hash_identifier(user_id),
            "role": role,
            "query": redact_text(query),
            "response": redact_text(response),
            "route": route,
            "risk_level": risk_level,
            "feedback_rating": rating,
            "feedback_tags": [redact_text(tag) for tag in tags],
            "feedback_comment": redact_text(comment),
            "preference_summary": preference_summary,
            "last_active_at": now,
            "created_at": now,
        }
        self._append_local(record)
        self._save_supabase(record)
        return record

    def behavior_preferences(self, user_id: str, limit: int = 20) -> str:
        user_hash = hash_identifier(user_id)
        if not MEMORY_PATH.exists():
            return ""
        feedback_items: list[dict[str, object]] = []
        for line in MEMORY_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if item.get("entry_type") == "feedback" and item.get("user_id") == user_hash:
                feedback_items.append(item)
        recent_items = feedback_items[-limit:]
        if not recent_items:
            return ""
        tags: dict[str, int] = {}
        positive_count = 0
        negative_count = 0
        summaries: list[str] = []
        comments: list[str] = []
        for item in recent_items:
            if item.get("feedback_rating") == "helpful":
                positive_count += 1
            if item.get("feedback_rating") == "not_helpful":
                negative_count += 1
            for tag in item.get("feedback_tags", []) if isinstance(item.get("feedback_tags"), list) else []:
                tags[str(tag)] = tags.get(str(tag), 0) + 1
            summary = str(item.get("preference_summary", "")).strip()
            if summary and summary not in summaries:
                summaries.append(summary)
            comment = redact_text(str(item.get("feedback_comment", ""))).strip()
            if comment and comment not in comments:
                comments.append(comment)
        top_tags = sorted(tags, key=tags.get, reverse=True)[:4]
        parts = []
        if summaries:
            parts.append("; ".join(summaries[:4]))
        if top_tags:
            parts.append(f"Recent feedback themes: {', '.join(top_tags)}.")
        if comments:
            parts.append(f"Recent feedback comments: {'; '.join(comments[-3:])}.")
        if negative_count > positive_count:
            parts.append("User has recently marked more answers as not helpful, so be more explicit and verify assumptions.")
        return " ".join(parts)

    def recent_queries(self, user_id: str, limit: int = 6) -> list[str]:
        user_hash = hash_identifier(user_id)
        supabase_queries = self._recent_queries_supabase(user_hash, limit=limit)
        if supabase_queries:
            return supabase_queries
        if not MEMORY_PATH.exists():
            return []
        queries: list[str] = []
        for line in reversed(MEMORY_PATH.read_text(encoding="utf-8").splitlines()):
            if len(queries) >= limit:
                break
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if item.get("user_id") != user_hash or item.get("entry_type", "conversation") != "conversation":
                continue
            query = redact_text(str(item.get("query", ""))).strip()
            if query and query not in queries:
                queries.append(query)
        return queries

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
        if record.get("entry_type") == "feedback":
            payload.update(
                {
                    "entry_type": "feedback",
                    "feedback_rating": record.get("feedback_rating"),
                    "feedback_tags": record.get("feedback_tags"),
                    "feedback_comment": record.get("feedback_comment"),
                    "preference_summary": record.get("preference_summary"),
                }
            )
        raw = request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        try:
            request.urlopen(raw, timeout=15).close()
        except (error.HTTPError, error.URLError):
            return

    def _recent_queries_supabase(self, user_hash: str, *, limit: int) -> list[str]:
        headers = self._headers()
        if not headers or not self.url:
            return []
        params = {
            "user_id": f"eq.{user_hash}",
            "select": "query,entry_type,created_at",
            "order": "created_at.desc",
            "limit": str(limit * 3),
        }
        url = f"{self.url.rstrip('/')}/rest/v1/conversation_memory?{parse.urlencode(params)}"
        raw = request.Request(url, headers=headers, method="GET")
        try:
            with request.urlopen(raw, timeout=15) as response:
                rows = json.loads(response.read().decode("utf-8"))
        except (error.HTTPError, error.URLError, json.JSONDecodeError):
            rows = self._recent_queries_supabase_legacy(user_hash, limit=limit)
        queries: list[str] = []
        for item in rows if isinstance(rows, list) else []:
            if len(queries) >= limit:
                break
            if not isinstance(item, dict) or item.get("entry_type", "conversation") != "conversation":
                continue
            query = redact_text(str(item.get("query", ""))).strip()
            if query and query not in queries:
                queries.append(query)
        return queries

    def _recent_queries_supabase_legacy(self, user_hash: str, *, limit: int) -> list[dict[str, object]]:
        headers = self._headers()
        if not headers or not self.url:
            return []
        params = {
            "user_id": f"eq.{user_hash}",
            "select": "query,created_at",
            "order": "created_at.desc",
            "limit": str(limit * 2),
        }
        url = f"{self.url.rstrip('/')}/rest/v1/conversation_memory?{parse.urlencode(params)}"
        raw = request.Request(url, headers=headers, method="GET")
        try:
            with request.urlopen(raw, timeout=15) as response:
                rows = json.loads(response.read().decode("utf-8"))
                return rows if isinstance(rows, list) else []
        except (error.HTTPError, error.URLError, json.JSONDecodeError):
            return []

    def _derive_preference_summary(self, *, rating: str, tags: list[str], role: str) -> str:
        safe_tags = {tag.lower().strip() for tag in tags}
        preferences: list[str] = []
        if "too technical" in safe_tags:
            preferences.append("Use simpler, customer-friendly wording and avoid internal banking terms.")
        if "too simple" in safe_tags or "too vague" in safe_tags:
            preferences.append("Add more concrete details and next steps when sources support them.")
        if "too long" in safe_tags:
            preferences.append("Keep future responses concise and lead with the answer.")
        if "too short" in safe_tags:
            preferences.append("Provide a fuller explanation with useful context.")
        if "need steps" in safe_tags:
            preferences.append("Use step-by-step structure for process or calculation questions.")
        if "need calculation" in safe_tags:
            preferences.append("For repayment or interest questions, explain calculation inputs and use the calculator tool for numbers.")
        if "wrong route" in safe_tags:
            preferences.append("Be more careful about routing and ask for clarification when intent is ambiguous.")
        if rating == "not_helpful" and not preferences:
            preferences.append("Improve clarity and include what was verified from tools.")
        if role in {"manager", "admin", "support", "risk"} and "too simple" in safe_tags:
            preferences.append("For staff users, include operational detail without exposing PII.")
        return " ".join(preferences)

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
