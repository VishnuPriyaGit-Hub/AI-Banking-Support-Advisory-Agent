from __future__ import annotations

from pathlib import Path

from app.core.config import PHASE3_PROMPT_PATH


def load_prompt_template(prompt_path: Path = PHASE3_PROMPT_PATH) -> str:
    return prompt_path.read_text(encoding="utf-8").strip()
