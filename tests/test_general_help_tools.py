from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from toolcall_rl.tools import (
    calculator,
    google_search,
    string_formatter,
    text_stats,
    unit_converter,
)


AGENT_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "toolcall_rl"
    / "agents"
    / "Base_Model"
    / "agent.py"
)


def test_calculator_evaluates_basic_arithmetic() -> None:
    assert calculator("(2 + 3) * 4")["result"] == 20


def test_calculator_rejects_non_arithmetic_expression() -> None:
    result = calculator("__import__('os').system('date')")

    assert "error" in result


def test_google_search_returns_query_without_searching() -> None:
    result = google_search("google adk ollama")

    assert result["query"] == "google adk ollama"


def test_unit_converter_converts_supported_units() -> None:
    result = unit_converter(10, "kilometers", "miles")

    assert result["result"] == 6.2137


def test_text_stats_counts_basic_text_properties() -> None:
    result = text_stats("Hello world. Tool calls work!")

    assert result["words"] == 5
    assert result["sentences"] == 2


def test_string_formatter_applies_requested_operation() -> None:
    result = string_formatter("hello tools", "titlecase")

    assert result["result"] == "Hello Tools"


def test_root_agent_exposes_required_tools() -> None:
    spec = spec_from_file_location("base_model_agent", AGENT_PATH)
    assert spec is not None
    assert spec.loader is not None

    agent_module = module_from_spec(spec)
    spec.loader.exec_module(agent_module)
    root_agent = agent_module.root_agent

    assert root_agent.name == "general_help_agent"
    assert root_agent.model.model == "ollama_chat/smollm:1.7b"
    assert [tool.__name__ for tool in root_agent.tools] == [
        "calculator",
        "google_search",
        "unit_converter",
        "text_stats",
        "string_formatter",
    ]
