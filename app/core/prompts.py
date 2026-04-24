from __future__ import annotations

from pathlib import Path

from app.core.config import PHASE2_PROMPT_PATH, PHASE3_PROMPT_PATH, PHASE4_PROMPT_PATH


def _read_prompt(prompt_path: Path) -> str:
    return prompt_path.read_text(encoding="utf-8").strip()


def load_phase2_prompt(prompt_path: Path = PHASE2_PROMPT_PATH) -> str:
    return _read_prompt(prompt_path)


def load_phase3_prompt(prompt_path: Path = PHASE3_PROMPT_PATH) -> str:
    return _read_prompt(prompt_path)


def load_phase4_prompt(prompt_path: Path = PHASE4_PROMPT_PATH) -> str:
    return _read_prompt(prompt_path)
