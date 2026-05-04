from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.agents.langgraph_agent import MultiAgentBankingAssistant
from app.core.config import get_env_value
from app.memory.store import ConversationMemory
from app.security.pii import redact_text


class ChatHistoryItem(BaseModel):
    speaker: str = Field(..., examples=["user", "assistant"])
    text: str


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    role: str = Field(..., examples=["customer", "manager", "admin", "support", "risk"])
    customer_id: str | None = None
    branch: str | None = None
    auth_user_id: str | None = None
    user_jwt: str | None = None
    chat_history: list[ChatHistoryItem] = Field(default_factory=list)
    behavior_preferences: str | None = None


class ChatResponse(BaseModel):
    response: str
    route: str
    risk_level: str
    risk_reason: str = ""
    tools_used: list[str]
    required_tools: list[str]
    confidence_score: float
    evaluation_score: float = 0.0
    evaluation_metrics: dict[str, int] = Field(default_factory=dict)
    evaluation_reason: str = ""
    escalated_to: str = ""
    adaptation_note: str = ""
    trace_id: str = ""
    total_latency_ms: int = 0
    logs: list[dict[str, Any]]


class FeedbackRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    role: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    response: str = Field(..., min_length=1)
    route: str = ""
    risk_level: str = ""
    rating: str = Field(..., examples=["helpful", "not_helpful"])
    tags: list[str] = Field(default_factory=list)
    comment: str = ""
    user_jwt: str | None = None


class FeedbackResponse(BaseModel):
    saved: bool
    preference_summary: str


class RecentQueriesResponse(BaseModel):
    queries: list[str]


agent = MultiAgentBankingAssistant()
app = FastAPI(
    title="Banking Support Agent API",
    version="1.0.0",
    description="FastAPI wrapper for the LangGraph multi-agent banking assistant.",
)

cors_origins = get_env_value("API_CORS_ORIGINS")
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in cors_origins.split(",") if origin.strip()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "banking-support-agent", "status": "ok"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    memory_user_id = request.auth_user_id or request.customer_id or request.role
    memory = ConversationMemory(user_jwt=request.user_jwt)
    behavior_preferences = request.behavior_preferences
    if behavior_preferences is None:
        behavior_preferences = memory.behavior_preferences(memory_user_id)

    try:
        result = agent.run(
            request.query,
            role=request.role,
            customer_id=request.customer_id,
            branch=request.branch,
            auth_user_id=request.auth_user_id,
            user_jwt=request.user_jwt,
            chat_history=[item.model_dump() for item in request.chat_history],
            behavior_preferences=behavior_preferences,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=redact_text(f"Agent failed: {exc}")) from exc

    return ChatResponse(
        response=str(result.get("response", "")),
        route=str(result.get("route", "")),
        risk_level=str(result.get("risk_level", "")),
        risk_reason=str(result.get("risk_reason", "")),
        tools_used=list(result.get("tools_used", [])),
        required_tools=list(result.get("required_tools", [])),
        confidence_score=float(result.get("confidence_score", 0.0) or 0.0),
        evaluation_score=float(result.get("evaluation_score", 0.0) or 0.0),
        evaluation_metrics=dict(result.get("evaluation_metrics", {}) or {}),
        evaluation_reason=str(result.get("evaluation_reason", "")),
        escalated_to=str(result.get("escalated_to", "")),
        adaptation_note=str(result.get("adaptation_note", "")),
        trace_id=str(result.get("trace_id", "")),
        total_latency_ms=int(result.get("total_latency_ms", 0) or 0),
        logs=list(result.get("logs", [])),
    )


@app.post("/api/v1/feedback", response_model=FeedbackResponse)
def feedback(request: FeedbackRequest) -> FeedbackResponse:
    if request.rating not in {"helpful", "not_helpful"}:
        raise HTTPException(status_code=400, detail="rating must be helpful or not_helpful")
    memory = ConversationMemory(user_jwt=request.user_jwt)
    try:
        record = memory.save_feedback(
            user_id=request.user_id,
            role=request.role,
            query=request.query,
            response=request.response,
            route=request.route,
            risk_level=request.risk_level,
            rating=request.rating,
            tags=request.tags,
            comment=request.comment,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=redact_text(f"Feedback save failed: {exc}")) from exc
    return FeedbackResponse(saved=True, preference_summary=str(record.get("preference_summary", "")))


@app.get("/api/v1/memory/{user_id}/recent-queries", response_model=RecentQueriesResponse)
def recent_queries(user_id: str, user_jwt: str | None = None, limit: int = 6) -> RecentQueriesResponse:
    safe_limit = max(1, min(limit, 20))
    queries = ConversationMemory(user_jwt=user_jwt).recent_queries(user_id, limit=safe_limit)
    return RecentQueriesResponse(queries=queries)


@app.delete("/api/v1/memory/{user_id}")
def delete_memory(user_id: str, user_jwt: str | None = None) -> dict[str, int | bool]:
    removed = ConversationMemory(user_jwt=user_jwt).delete_user_memory(user_id)
    return {"deleted": True, "local_records_removed": removed}
