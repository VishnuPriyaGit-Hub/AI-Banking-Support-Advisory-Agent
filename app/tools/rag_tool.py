from __future__ import annotations

import json

from app.rag.retrieval import SimpleRAGRetriever


def rag_retrieval_tool(query: str) -> str:
    retriever = SimpleRAGRetriever()
    result = retriever.build_answer(query, top_k=4)
    matches = result.get("matches", [])
    simplified_matches = [
        {
            "source_file": match.get("source_file", ""),
            "doc_group": match.get("doc_group", ""),
            "section_title": match.get("section_title", ""),
            "text": match.get("text", ""),
        }
        for match in matches
    ]
    return json.dumps(
        {
            "sources": result.get("sources", []),
            "confidence_score": result.get("confidence_score", 0.0),
            "matches": simplified_matches,
        },
        indent=2,
    )
