from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = PROJECT_ROOT / "app"
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "Docs"
LOG_DIR = PROJECT_ROOT / "logs"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
ENV_FILE_PATH = PROJECT_ROOT / ".env"

DEFAULT_LOG_PATH = LOG_DIR / "baseline_agent_runs.jsonl"
DEMO_LOG_PATH = LOG_DIR / "sample_interactions.jsonl"
PHASE2_PROMPT_PATH = PROMPTS_DIR / "phase2" / "system_prompt.txt"
PHASE3_PROMPT_PATH = PROMPTS_DIR / "phase3" / "system_prompt.txt"
PHASE4_PROMPT_PATH = PROMPTS_DIR / "phase4" / "system_prompt.txt"
PHASE5_PROMPT_PATH = PROMPTS_DIR / "phase5" / "system_prompt.txt"
AUTH_DB_PATH = DATA_DIR / "banking_auth.db"
RAG_SUMMARY_PATH = DATA_DIR / "rag_ingest_summary.json"
PHASE5_SUPABASE_SCHEMA_PATH = DATA_DIR / "phase5_supabase_schema.sql"

ALLOWED_ROLES = {
    "customer": "Customer",
    "branch manager": "Branch Manager",
    "risk & compliance officer": "Risk & Compliance Officer",
    "risk and compliance officer": "Risk & Compliance Officer",
    "admin": "Admin",
    "customer support agent": "Customer Support Agent",
}

PLACEHOLDER_ENV_VALUES = {
    "",
    "your_api_key_here",
    "your_openai_api_key_here",
    "paste_your_api_key_here",
    "replace_me",
}


def ensure_runtime_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def load_env_file(env_path: Path = ENV_FILE_PATH) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_env_value(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    if value.lower() in PLACEHOLDER_ENV_VALUES:
        return None
    return value or None


ensure_runtime_directories()
load_env_file()
