from __future__ import annotations

import time
import os
from contextlib import nullcontext
from typing import Any

from app.core.config import get_env_value
from app.security.pii import redact_value


def now_ms() -> float:
    return time.perf_counter() * 1000


def elapsed_ms(start_ms: float) -> int:
    return int(now_ms() - start_ms)


def langfuse_enabled() -> bool:
    return (get_env_value("LANGFUSE_ENABLED") or "").lower() in {"1", "true", "yes", "on"}


def trace_verbose() -> bool:
    return (get_env_value("TRACE_VERBOSE") or "").lower() in {"1", "true", "yes", "on"}


def _client():
    if not langfuse_enabled():
        return None
    if not get_env_value("LANGFUSE_PUBLIC_KEY") or not get_env_value("LANGFUSE_SECRET_KEY"):
        return None
    host = get_env_value("LANGFUSE_HOST") or get_env_value("LANGFUSE_BASE_URL")
    if host and not os.getenv("LANGFUSE_HOST"):
        os.environ["LANGFUSE_HOST"] = host
    try:
        from langfuse import get_client

        return get_client()
    except Exception:
        return None


class TraceScope:
    def __init__(
        self,
        *,
        name: str,
        trace_id: str,
        user_id: str,
        session_id: str,
        input_payload: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.client = _client()
        self.name = name
        self.trace_id = trace_id
        self.user_id = user_id
        self.session_id = session_id
        self.input_payload = redact_value(input_payload)
        self.metadata = redact_value(metadata or {})
        self._ctx = nullcontext()
        self.observation: Any = None
        self.span_name = "agent_pipeline"

    def __enter__(self) -> "TraceScope":
        if not self.client:
            return self
        try:
            self._ctx = self.client.start_as_current_span(name=self.span_name)
            self.observation = self._ctx.__enter__()
            self.observation.update_trace(
                name=self.name,
                user_id=self.user_id,
                session_id=self.session_id,
                input=self.input_payload,
                metadata={"trace_id": self.trace_id, **self.metadata},
            )
            self.observation.update(input=self.input_payload, metadata=self.metadata)
        except Exception:
            self.observation = None
        return self

    def update(self, *, output: Any = None, metadata: dict[str, Any] | None = None) -> None:
        if not self.observation:
            return
        try:
            safe_metadata = redact_value(metadata or {})
            payload: dict[str, Any] = {"output": redact_value(output), "metadata": safe_metadata}
            if isinstance(safe_metadata, dict) and int(safe_metadata.get("error_count", 0) or 0) > 0:
                payload["level"] = "ERROR"
                payload["status_message"] = f"{safe_metadata.get('error_count')} error(s) captured during agent run"
            self.observation.update(**payload)
            self.observation.update_trace(output=redact_value(output), metadata=safe_metadata)
        except Exception:
            return

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self.observation and exc:
            try:
                self.observation.update(level="ERROR", status_message=str(exc))
            except Exception:
                pass
        try:
            self._ctx.__exit__(exc_type, exc, tb)
        except Exception:
            return


class SpanScope:
    def __init__(self, *, name: str, input_payload: Any = None, metadata: dict[str, Any] | None = None) -> None:
        self.client = _client() if trace_verbose() else None
        self.name = name
        self.input_payload = redact_value(input_payload)
        self.metadata = redact_value(metadata or {})
        self._ctx = nullcontext()
        self.observation: Any = None

    def __enter__(self) -> "SpanScope":
        if not self.client:
            return self
        try:
            self._ctx = self.client.start_as_current_observation(as_type="span", name=self.name)
            self.observation = self._ctx.__enter__()
            self.observation.update(input=self.input_payload, metadata=self.metadata)
        except Exception:
            self.observation = None
        return self

    def update(self, *, output: Any = None, metadata: dict[str, Any] | None = None, error: str = "") -> None:
        if not self.observation:
            return
        try:
            payload: dict[str, Any] = {"output": redact_value(output), "metadata": redact_value(metadata or {})}
            if error:
                payload["level"] = "ERROR"
                payload["status_message"] = error
            self.observation.update(**payload)
        except Exception:
            return

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self.observation and exc:
            try:
                self.observation.update(level="ERROR", status_message=str(exc))
            except Exception:
                pass
        try:
            self._ctx.__exit__(exc_type, exc, tb)
        except Exception:
            return


def start_trace(**kwargs: Any) -> TraceScope:
    return TraceScope(**kwargs)


def start_span(name: str, **kwargs: Any) -> SpanScope:
    return SpanScope(name=name, **kwargs)


def flush_langfuse() -> None:
    client = _client()
    if not client:
        return
    try:
        client.flush()
    except Exception:
        return
