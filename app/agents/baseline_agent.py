from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib import error, request

from pydantic import ValidationError

from app.core.config import DEFAULT_LOG_PATH, DEMO_LOG_PATH, get_env_value
from app.core.prompts import load_prompt_template
from app.models.agent import AgentRunResult, UserInput


class BaselineAgent:
    """Simple Phase 3 LLM-based banking support agent."""

    def __init__(self) -> None:
        self.api_key = get_env_value("OPENAI_API_KEY")
        self.base_url = get_env_value("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        self.model = get_env_value("OPENAI_MODEL") or "gpt-4o-mini"
        self.system_prompt = load_prompt_template()

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is missing in the environment.")

    def _build_messages(self, user_query: UserInput) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"Role: {user_query.role}\nQuery: {user_query.query}",
            },
        ]

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
        }
        raw_request = request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with request.urlopen(raw_request, timeout=60) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {details}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc.reason}") from exc

        choices = response_payload.get("choices", [])
        if not choices:
            raise RuntimeError("LLM response did not include any choices.")

        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            text_parts = [item.get("text", "") for item in content if isinstance(item, dict)]
            return "".join(text_parts).strip()
        return str(content).strip()

    def run(self, user_query: UserInput) -> AgentRunResult:
        messages = self._build_messages(user_query)
        output = self._call_llm(messages)
        return AgentRunResult(
            input={"role": user_query.role, "query": user_query.query},
            output=output,
            metadata={
                "model": self.model,
                "provider": "openai_compatible",
                "mode": "phase3_llm",
            },
        )


def log_result(log_path: Path, result: AgentRunResult) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(result.model_dump_json() + "\n")


def run_cli(log_path: Path) -> None:
    agent = BaselineAgent()
    print("Phase 3 Banking Support Agent")
    print(f"Model: {agent.model}")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        raw_role = input(
            "Role (Customer / Branch Manager / Risk & Compliance Officer / Admin / Customer Support Agent): "
        )
        if raw_role.strip().lower() in {"exit", "quit"}:
            print("Agent: Goodbye.")
            break

        raw_query = input("Query: ")
        if raw_query.strip().lower() in {"exit", "quit"}:
            print("Agent: Goodbye.")
            break

        try:
            request_obj = UserInput(role=raw_role, query=raw_query)
        except ValidationError:
            print("Agent: Please enter a valid role and query.")
            continue

        try:
            result = agent.run(request_obj)
        except Exception as exc:
            print(f"Agent error: {exc}\n")
            continue

        print(f"Agent: {result.output}\n")
        log_result(log_path, result)


def run_demo(log_path: Path) -> None:
    agent = BaselineAgent()
    demo_inputs = [
        {"role": "Customer", "query": "What is EMI?"},
        {"role": "Customer", "query": "Transfer 1000 to this account"},
        {"role": "Customer Support Agent", "query": "Money got deducted but I did not do it"},
    ]

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")

    for item in demo_inputs:
        request_obj = UserInput(**item)
        result = agent.run(request_obj)
        log_result(log_path, result)
        print(f"{item['role']} | {item['query']}")
        print(f"Agent: {result.output}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A simple LLM-based banking agent.")
    parser.add_argument("--demo", action="store_true", help="Run demo queries.")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG_PATH, help="Path to the JSONL log file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_log = args.log if not args.demo or args.log != DEFAULT_LOG_PATH else DEMO_LOG_PATH
    if args.demo:
        run_demo(target_log)
    else:
        run_cli(target_log)


if __name__ == "__main__":
    main()
