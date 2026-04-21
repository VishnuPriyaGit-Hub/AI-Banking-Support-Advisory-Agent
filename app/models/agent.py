from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.core.config import ALLOWED_ROLES


class UserInput(BaseModel):
    role: str = Field(
        ...,
        description="Allowed roles: Customer, Branch Manager, Risk & Compliance Officer, Admin, Customer Support Agent.",
    )
    query: str = Field(..., description="Simple text query for the banking agent.")

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Role cannot be empty.")
        normalized = cleaned.lower()
        if normalized not in ALLOWED_ROLES:
            allowed = ", ".join(ALLOWED_ROLES.values())
            raise ValueError(f"Role must be one of: {allowed}.")
        return ALLOWED_ROLES[normalized]

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Query cannot be empty.")
        return cleaned


class AgentRunResult(BaseModel):
    input: dict[str, str]
    output: str
    metadata: dict[str, str]


class ClassificationResult(BaseModel):
    category: str
    guidance: str
    product_hint: str = "general"
