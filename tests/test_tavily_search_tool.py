import json
from importlib import import_module

from toolcall_rl.tools import google_search


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(
            {
                "answer": "Tavily is a search API for AI agents.",
                "results": [
                    {
                        "title": "Tavily Docs",
                        "url": "https://docs.tavily.com",
                        "content": "Search API documentation.",
                        "score": 0.91,
                    }
                ],
            }
        ).encode("utf-8")


def test_google_search_calls_tavily_api(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(tavily_request, timeout):
        captured["timeout"] = timeout
        captured["url"] = tavily_request.full_url
        captured["headers"] = dict(tavily_request.header_items())
        captured["body"] = json.loads(tavily_request.data.decode("utf-8"))
        return _FakeResponse()

    monkeypatch.setenv("TAVILY_KEY", "test-key")
    google_search_module = import_module("toolcall_rl.tools.google_search")
    monkeypatch.setattr(google_search_module.request, "urlopen", fake_urlopen)

    result = google_search("agent tool calling")

    assert captured["url"] == "https://api.tavily.com/search"
    assert captured["timeout"] == 20
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["body"]["query"] == "agent tool calling"
    assert captured["body"]["max_results"] == 5
    assert result == {
        "query": "agent tool calling",
        "answer": "Tavily is a search API for AI agents.",
        "results": [
            {
                "title": "Tavily Docs",
                "url": "https://docs.tavily.com",
                "content": "Search API documentation.",
                "score": 0.91,
            }
        ],
    }
