from __future__ import annotations

import json
from urllib import error, parse, request

from app.core.config import get_env_value


def search_api_tool(query: str) -> str:
    api_key = get_env_value("SEARCHAPI_API_KEY")
    base_url = get_env_value("SEARCHAPI_BASE_URL") or "https://www.searchapi.io/api/v1/search"
    engine = get_env_value("SEARCHAPI_ENGINE") or "google"
    if not api_key or not base_url:
        return "SearchAPI is not configured."

    url = f"{base_url.rstrip('/')}?{parse.urlencode({'engine': engine, 'q': query})}"
    raw_request = request.Request(
        url,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="GET",
    )
    try:
        with request.urlopen(raw_request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="ignore")
        return f"SearchAPI request failed with HTTP {exc.code}: {details}"
    except error.URLError as exc:
        return f"SearchAPI request failed: {exc.reason}"
    answer_box = payload.get("answer_box", {}) or {}
    organic = payload.get("organic_results", [])
    if not answer_box and not organic:
        return "No search results found."

    result: dict[str, object] = {
        "query": query,
        "answer_box": {
            "title": answer_box.get("title", ""),
            "answer": answer_box.get("answer", ""),
            "snippet": answer_box.get("snippet", ""),
        },
        "organic_results": [
            {
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            }
            for item in organic[:3]
        ],
    }
    return json.dumps(result, indent=2)
