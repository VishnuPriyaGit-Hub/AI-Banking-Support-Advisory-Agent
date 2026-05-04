from __future__ import annotations

from pathlib import Path

from app.core.config import (
    PHASE2_PROMPT_PATH,
    PHASE3_PROMPT_PATH,
    PHASE4_PROMPT_PATH,
    PHASE4_RAG_ANSWER_PROMPT_PATH,
    PHASE4_REWRITE_SYSTEM_PROMPT_PATH,
    PHASE4_REWRITE_PROMPT_PATH,
    PHASE5_PROMPT_PATH,
    PHASE6_CALCULATION_PROMPT_PATH,
    PHASE6_EVALUATION_PROMPT_PATH,
    PHASE6_PERSONALIZED_DATA_RESPONSE_PROMPT_PATH,
    PHASE6_PERSONALIZED_GUIDANCE_PROMPT_PATH,
    PHASE6_PLANNER_PROMPT_PATH,
    PHASE6_RESPONSE_PROMPT_PATH,
    PHASE6_REWRITE_PROMPT_PATH,
    PHASE6_SYSTEM_PROMPT_PATH,
)


def _read_prompt(prompt_path: Path) -> str:
    return prompt_path.read_text(encoding="utf-8").strip()


def load_phase2_prompt(prompt_path: Path = PHASE2_PROMPT_PATH) -> str:
    return _read_prompt(prompt_path)


def load_phase3_prompt(prompt_path: Path = PHASE3_PROMPT_PATH) -> str:
    return _read_prompt(prompt_path)


def load_phase4_prompt(prompt_path: Path = PHASE4_PROMPT_PATH) -> str:
    return _read_prompt(prompt_path)


def load_phase4_rewrite_prompt(prompt_path: Path = PHASE4_REWRITE_PROMPT_PATH) -> str:
    return _read_prompt(prompt_path)


def load_phase4_rewrite_system_prompt(prompt_path: Path = PHASE4_REWRITE_SYSTEM_PROMPT_PATH) -> str:
    return _read_prompt(prompt_path)


def load_phase4_rag_answer_prompt(prompt_path: Path = PHASE4_RAG_ANSWER_PROMPT_PATH) -> str:
    return _read_prompt(prompt_path)


def load_phase5_prompt(prompt_path: Path = PHASE5_PROMPT_PATH) -> str:
    return _read_prompt(prompt_path)


def load_phase6_system_prompt(prompt_path: Path = PHASE6_SYSTEM_PROMPT_PATH) -> str:
    return _read_prompt(prompt_path)


def load_phase6_planner_prompt(prompt_path: Path = PHASE6_PLANNER_PROMPT_PATH) -> str:
    return _read_prompt(prompt_path)


def load_phase6_rewrite_prompt(prompt_path: Path = PHASE6_REWRITE_PROMPT_PATH) -> str:
    return _read_prompt(prompt_path)


def load_phase6_calculation_prompt(prompt_path: Path = PHASE6_CALCULATION_PROMPT_PATH) -> str:
    return _read_prompt(prompt_path)


def load_phase6_response_prompt(prompt_path: Path = PHASE6_RESPONSE_PROMPT_PATH) -> str:
    return _read_prompt(prompt_path)


def load_phase6_personalized_data_response_prompt(
    prompt_path: Path = PHASE6_PERSONALIZED_DATA_RESPONSE_PROMPT_PATH,
) -> str:
    return _read_prompt(prompt_path)


def load_phase6_personalized_guidance_prompt(
    prompt_path: Path = PHASE6_PERSONALIZED_GUIDANCE_PROMPT_PATH,
) -> str:
    return _read_prompt(prompt_path)


def load_phase6_evaluation_prompt(prompt_path: Path = PHASE6_EVALUATION_PROMPT_PATH) -> str:
    return _read_prompt(prompt_path)


def render_prompt(template: str, values: dict[str, object]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{{" + key + "}}", str(value))
    return rendered
