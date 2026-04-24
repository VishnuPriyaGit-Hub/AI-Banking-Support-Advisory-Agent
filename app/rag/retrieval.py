from __future__ import annotations

import json
from urllib import error, request

from pymilvus import MilvusClient

from app.core.config import get_env_value


class SimpleRAGRetriever:
    def __init__(self) -> None:
        self.embedding_base_url = get_env_value("EMBEDDING_BASE_URL") or get_env_value("OPENAI_BASE_URL")
        self.embedding_api_key = get_env_value("EMBEDDING_API_KEY") or get_env_value("OPENAI_API_KEY")
        self.embedding_model = get_env_value("EMBEDDING_MODEL") or "text-embedding-3-small"

        self.zilliz_uri = get_env_value("ZILLIZ_ENDPOINT")
        self.zilliz_api_key = get_env_value("ZILLIZ_API_KEY")
        self.collection_name = get_env_value("ZILLIZ_COLLECTION_NAME") or "banking_rag_chunks"

        if not self.embedding_base_url:
            raise ValueError("EMBEDDING_BASE_URL or OPENAI_BASE_URL is required in .env.")
        if not self.embedding_api_key:
            raise ValueError("EMBEDDING_API_KEY or OPENAI_API_KEY is required in .env.")
        if not self.zilliz_uri:
            raise ValueError("ZILLIZ_ENDPOINT is required in .env.")
        if not self.zilliz_api_key:
            raise ValueError("ZILLIZ_API_KEY is required in .env.")

    def embed_query(self, query: str) -> list[float]:
        url = f"{self.embedding_base_url.rstrip('/')}/embeddings"
        payload = {
            "model": self.embedding_model,
            "input": query,
            "encoding_format": "float",
        }
        raw_request = request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.embedding_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(raw_request, timeout=60) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Embedding request failed with HTTP {exc.code}: {details}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Embedding request failed: {exc.reason}") from exc

        data = response_payload.get("data", [])
        if not data:
            raise RuntimeError("Embedding response did not include vectors.")
        return data[0]["embedding"]

    def create_client(self) -> MilvusClient:
        return MilvusClient(uri=self.zilliz_uri, token=self.zilliz_api_key)

    def search(self, query: str, top_k: int = 4) -> list[dict[str, object]]:
        query_vector = self.embed_query(query)
        client = self.create_client()

        primary_groups = self.route_query_groups(query)
        primary_hits = self.search_groups(client, query_vector, primary_groups, limit=top_k)
        faq_hits = self.search_groups(client, query_vector, ["faq"], limit=2) if "faq" not in primary_groups else []

        combined = self.deduplicate_hits(primary_hits + faq_hits)
        if combined:
            return combined

        return self.search_groups(client, query_vector, [], limit=top_k)

    def search_groups(
        self,
        client: MilvusClient,
        query_vector: list[float],
        groups: list[str],
        *,
        limit: int,
    ) -> list[dict[str, object]]:
        filter_expr = self.build_filter(groups)
        results = client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=limit,
            filter=filter_expr,
            output_fields=["source_file", "doc_group", "content_type", "chunk_index", "section_title", "text"],
            search_params={"metric_type": "COSINE"},
        )
        if not results:
            return []

        first_result_set = results[0] if isinstance(results[0], list) else results
        normalized_hits: list[dict[str, object]] = []
        for hit in first_result_set:
            entity = hit.get("entity", {}) if isinstance(hit, dict) else {}
            normalized_hits.append(
                {
                    "id": hit.get("id") if isinstance(hit, dict) else "",
                    "score": hit.get("distance") if isinstance(hit, dict) else None,
                    "source_file": entity.get("source_file", ""),
                    "doc_group": entity.get("doc_group", ""),
                    "content_type": entity.get("content_type", ""),
                    "chunk_index": entity.get("chunk_index", 0),
                    "section_title": entity.get("section_title", ""),
                    "text": entity.get("text", ""),
                }
            )
        return normalized_hits

    def build_filter(self, groups: list[str]) -> str:
        if not groups:
            return ""
        if len(groups) == 1:
            return f'doc_group == "{groups[0]}"'
        return " or ".join(f'doc_group == "{group}"' for group in groups)

    def route_query_groups(self, query: str) -> list[str]:
        q = query.lower()
        if any(token in q for token in ["deposit", "fd", "fixed deposit", "rd", "recurring deposit", "maturity", "tenure"]):
            return ["deposits"]
        if any(token in q for token in ["loan", "emi", "mortgage", "collateral", "personal loan", "home loan", "car loan"]):
            return ["loans"]
        if any(token in q for token in ["account opening", "open account", "close account", "account closure", "savings account", "current account"]):
            return ["accounts"]
        if any(token in q for token in ["card", "credit card", "debit card", "cvv", "card block", "pin"]):
            return ["cards", "faq"]
        return ["accounts", "deposits", "loans"]

    def deduplicate_hits(self, hits: list[dict[str, object]]) -> list[dict[str, object]]:
        seen: set[tuple[str, int]] = set()
        unique_hits: list[dict[str, object]] = []
        for hit in hits:
            key = (str(hit.get("source_file", "")), int(hit.get("chunk_index", 0)))
            if key in seen:
                continue
            seen.add(key)
            unique_hits.append(hit)
        return unique_hits

    def build_answer(self, query: str, top_k: int = 4) -> dict[str, object]:
        hits = self.search(query, top_k=top_k)
        if not hits:
            return {
                "answer": "I could not find relevant information in the uploaded banking documents.",
                "sources": [],
                "matches": [],
                "confidence_score": 0.0,
            }

        sources = sorted({str(hit.get("source_file", "")).strip() for hit in hits if str(hit.get("source_file", "")).strip()})
        confidence_score = self.calculate_confidence(hits)
        return {
            "answer": "",
            "sources": sources,
            "matches": hits,
            "confidence_score": confidence_score,
        }

    def calculate_confidence(self, hits: list[dict[str, object]]) -> float:
        if not hits:
            return 0.0
        raw_score = hits[0].get("score")
        if raw_score is None:
            return 0.0
        try:
            score_value = float(raw_score)
        except (TypeError, ValueError):
            return 0.0

        if score_value < 0:
            normalized = 1.0 / (1.0 + abs(score_value))
        elif score_value <= 1.0:
            normalized = score_value
        else:
            normalized = 1.0 / (1.0 + score_value)
        return round(max(0.0, min(1.0, normalized)), 3)
