"""No-key Google search intent tool."""

from __future__ import annotations


def google_search(query: str) -> dict[str, str]:
    """Return the Google search query without making a network request."""

    return {
        "query": query,
    }
