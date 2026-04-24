from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib import error, request

from pydantic import ValidationError

from app.core.config import DEFAULT_LOG_PATH, get_env_value
from app.core.prompts import load_phase4_prompt
from app.models.agent import AgentRunResult, UserInput
from app.rag.retrieval import SimpleRAGRetriever


class RAGAgent:
    def __init__(self) -> None:
        self.retriever = SimpleRAGRetriever()
        self.api_key = get_env_value("LLM_API_KEY") or get_env_value("OPENAI_API_KEY") or get_env_value("EMBEDDING_API_KEY")
        self.base_url = get_env_value("LLM_BASE_URL") or get_env_value("OPENAI_BASE_URL") or get_env_value("EMBEDDING_BASE_URL")
        self.model = get_env_value("OPENAI_MODEL") or "gpt-4o-mini"
        self.system_prompt = load_phase4_prompt()

    def run(self, user_query: UserInput, chat_history: list[dict[str, str]] | None = None) -> AgentRunResult:
        effective_query = self.build_standalone_query(user_query, chat_history)
        retrieval = self.retriever.build_answer(effective_query, top_k=4)
        sources = retrieval["sources"]
        confidence_score = retrieval.get("confidence_score", 0.0)
        answer = self.generate_answer(user_query, retrieval, effective_query)
        return AgentRunResult(
            input={"role": user_query.role, "query": user_query.query},
            output=answer,
            metadata={
                "mode": "rag_llm" if self.can_use_llm() else "rag_retrieval",
                "sources": " | ".join(sources),
                "confidence_score": str(confidence_score),
                "effective_query": effective_query,
            },
        )

    def can_use_llm(self) -> bool:
        return bool(self.api_key and self.base_url)

    def build_standalone_query(
        self,
        user_query: UserInput,
        chat_history: list[dict[str, str]] | None = None,
    ) -> str:
        if not self.can_use_llm() or not chat_history:
            return user_query.query

        recent_turns = chat_history[-6:]
        history_lines = [
            f"{item.get('speaker', 'user').title()}: {item.get('text', '').strip()}"
            for item in recent_turns
            if item.get("text", "").strip()
        ]
        if not history_lines:
            return user_query.query

        system_prompt = (
            "Given chat history and the latest user question, rewrite the latest question as a standalone query "
            "for banking document retrieval. Preserve the user's exact intent. "
            "Return only the rewritten standalone query."
        )
        user_prompt = (
            "Chat history:\n"
            f"{'\n'.join(history_lines)}\n\n"
            f"Latest question: {user_query.query}"
        )
        try:
            rewritten = self.call_llm(system_prompt, user_prompt)
        except Exception:
            return user_query.query
        return rewritten.strip() or user_query.query

    def generate_answer(self, user_query: UserInput, retrieval: dict[str, object], effective_query: str) -> str:
        if not self.can_use_llm():
            return self.build_fallback_answer(user_query, retrieval)

        matches = retrieval.get("matches", [])
        context_blocks: list[str] = []
        for index, match in enumerate(matches, start=1):
            source_file = str(match.get("source_file", "unknown"))
            doc_group = str(match.get("doc_group", "general"))
            content_type = str(match.get("content_type", "chunk"))
            chunk_index = match.get("chunk_index", 0)
            section_title = str(match.get("section_title", "")).strip()
            text = str(match.get("text", "")).strip()
            if not text:
                continue
            context_blocks.append(
                f"Source {index}: {source_file} | group={doc_group} | type={content_type} | chunk={chunk_index} | section={section_title}\n{text}"
            )

        if not context_blocks:
            return self.build_fallback_answer(user_query, retrieval)

        user_prompt = (
            f"User role: {user_query.role}\n"
            f"Original question: {user_query.query}\n"
            f"Standalone retrieval question: {effective_query}\n\n"
            "Retrieved context:\n"
            f"{'\n\n'.join(context_blocks)}"
        )

        try:
            llm_answer = self.call_llm(self.system_prompt, user_prompt)
        except Exception:
            return self.build_fallback_answer(user_query, retrieval)

        return llm_answer or self.build_fallback_answer(user_query, retrieval)

    def build_fallback_answer(self, user_query: UserInput, retrieval: dict[str, object]) -> str:
        matches = retrieval.get("matches", [])
        if not matches:
            return "I could not find relevant information in the uploaded banking documents."

        best_match = matches[0]
        text = str(best_match.get("text", "")).strip()
        if not text:
            return "Relevant records were found, but I could not build a useful answer from them."

        cleaned = (
            text.replace("Question:", "")
            .replace("Answer:", "")
            .replace("Section:", "")
            .replace("|", ", ")
            .strip()
        )
        prompt = user_query.query.lower()
        if any(token in prompt for token in ["fd", "fixed deposit", "deposit", "interest rate"]):
            return f"Based on the banking documents, here is the relevant fixed deposit information: {cleaned[:550]}"
        if any(token in prompt for token in ["loan", "emi"]):
            return f"Based on the banking documents, here is the relevant loan information: {cleaned[:550]}"
        if any(token in prompt for token in ["account", "open", "close"]):
            return f"Based on the banking documents, here is the relevant account information: {cleaned[:550]}"
        if any(token in prompt for token in ["card", "credit", "debit"]):
            return f"Based on the banking documents, here is the relevant card information: {cleaned[:550]}"
        return f"Based on the banking documents, here is the most relevant information I found: {cleaned[:550]}"

    def call_llm(self, system_prompt: str, user_prompt: str) -> str:
        candidates = [
            (self.base_url, self.api_key),
            (get_env_value("EMBEDDING_BASE_URL"), get_env_value("EMBEDDING_API_KEY")),
        ]
        last_error: Exception | None = None
        response_payload: dict[str, object] | None = None

        for base_url, api_key in candidates:
            if not base_url or not api_key:
                continue
            url = f"{base_url.rstrip('/')}/chat/completions"
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
            }
            raw_request = request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with request.urlopen(raw_request, timeout=60) as response:
                    response_payload = json.loads(response.read().decode("utf-8"))
                break
            except error.HTTPError as exc:
                details = exc.read().decode("utf-8", errors="ignore")
                last_error = RuntimeError(f"LLM request failed with HTTP {exc.code}: {details}")
            except error.URLError as exc:
                last_error = RuntimeError(f"LLM request failed: {exc.reason}")

        if response_payload is None:
            if last_error is not None:
                raise last_error
            raise RuntimeError("LLM request could not be completed.")

        choices = response_payload.get("choices", [])
        if not choices:
            raise RuntimeError("LLM response did not include any choices.")

        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            return "".join(
                item.get("text", "") for item in content if isinstance(item, dict)
            ).strip()
        return str(content).strip()


def log_result(log_path: Path, result: AgentRunResult) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(result.model_dump_json() + "\n")


def run_cli(log_path: Path) -> None:
    agent = RAGAgent()
    print("RAG Retrieval Agent")
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

        result = agent.run(request_obj)
        print(f"Agent: {result.output}")
        if result.metadata.get("sources"):
            print(f"Sources: {result.metadata['sources']}")
        print()
        log_result(log_path, result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A simple RAG retrieval agent.")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG_PATH, help="Path to the JSONL log file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_cli(args.log)


if __name__ == "__main__":
    main()
