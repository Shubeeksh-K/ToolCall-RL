"""Tavily-backed search tool."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib import request

from dotenv import load_dotenv


TAVILY_SEARCH_URL = "https://api.tavily.com/search"
DEFAULT_MAX_RESULTS = 5


def google_search(query: str) -> dict[str, Any]:
    """Search the web with Tavily and return a compact result set."""

    load_dotenv()
    api_key = os.getenv("TAVILY_KEY") or os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {
            "query": query,
            "error": "TAVILY_KEY is not set.",
        }

    payload = {
        "query": query,
        "search_depth": "basic",
        "max_results": DEFAULT_MAX_RESULTS,
        "include_answer": True,
        "include_raw_content": False,
    }

    try:
        response = _post_tavily_search(api_key, payload)
    except Exception as exc:
        return {
            "query": query,
            "error": f"Tavily search failed: {exc}",
        }

    return _format_search_response(query, response)


def _post_tavily_search(api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    tavily_request = request.Request(
        TAVILY_SEARCH_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with request.urlopen(tavily_request, timeout=20) as response:
        response_body = response.read().decode("utf-8")

    return json.loads(response_body)


def _format_search_response(query: str, response: dict[str, Any]) -> dict[str, Any]:
    results = []
    for result in response.get("results", []):
        results.append(
            {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("content", ""),
                "score": result.get("score"),
            }
        )

    return {
        "query": query,
        "answer": response.get("answer"),
        "results": results,
    }
