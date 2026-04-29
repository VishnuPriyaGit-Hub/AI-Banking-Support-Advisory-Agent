from __future__ import annotations

import json
import hashlib
import re
from typing import Any


PII_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", re.IGNORECASE), "[REDACTED_PAN]"),
    (re.compile(r"\b(?:\d[ -]?){12}\b"), "[REDACTED_AADHAAR]"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[REDACTED_CARD_OR_ACCOUNT]"),
    (re.compile(r"\b\d{9,18}\b"), "[REDACTED_ACCOUNT_OR_ID]"),
    (re.compile(r"\b[6-9]\d{9}\b"), "[REDACTED_PHONE]"),
    (re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
    (re.compile(r"\b(?:otp|pin|cvv|password|passcode)\s*(?:is|:)?\s*\S+\b", re.IGNORECASE), "[REDACTED_SECRET]"),
]

SENSITIVE_FIELD_NAMES = {
    "address",
    "email",
    "phone",
    "mobilenumber",
    "mobile",
    "pan",
    "aadhaar",
    "aadhar",
    "accountnumber",
    "cardnumber",
    "cvv",
    "pin",
    "password",
    "otp",
    "authuserid",
    "access_token",
    "user_jwt",
}


def redact_text(value: str) -> str:
    redacted = value
    for pattern, replacement in PII_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, dict):
        return redact_mapping(value)
    return value


def redact_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in payload.items():
        normalized_key = str(key).lower().replace("_", "")
        if normalized_key in SENSITIVE_FIELD_NAMES:
            redacted[key] = "[REDACTED]"
        else:
            redacted[key] = redact_value(value)
    return redacted


def redact_json_text(raw_text: str) -> str:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return redact_text(raw_text)
    return json.dumps(redact_value(payload), indent=2)


def contains_secret_request(text: str) -> bool:
    lowered = text.lower()
    sensitive_terms = [
        "show otp",
        "tell otp",
        "share otp",
        "pin",
        "cvv",
        "password",
        "full account number",
        "card number",
        "aadhaar",
        "aadhar",
        "pan number",
        "phone number",
        "email address",
        "address of customer",
    ]
    return any(term in lowered for term in sensitive_terms)


def contains_legal_advice_request(text: str) -> bool:
    lowered = text.lower()
    legal_terms = [
        "legal advice",
        "legal notice",
        "sue the bank",
        "file a case",
        "court case",
        "take legal action",
        "lawyer",
        "attorney",
        "consumer court",
        "draft a notice",
    ]
    return any(term in lowered for term in legal_terms)


def contains_ambiguous_action_request(text: str) -> bool:
    lowered = text.lower()
    ambiguous_terms = [
        "open an fd",
        "open fixed deposit",
        "open account",
        "close my account",
        "close account",
        "process this request",
        "do it for me",
        "submit this",
        "make this change",
        "update my account",
        "cancel my card",
        "block my card",
    ]
    return any(term in lowered for term in ambiguous_terms)


def hash_identifier(value: str) -> str:
    if not value:
        return ""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def mask_identifier(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 2:
        return "**"
    return f"{value[:1]}***{value[-1:]}"
