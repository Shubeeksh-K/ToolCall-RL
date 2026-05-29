from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from toolcall_rl.tools import (
    calculator,
    google_search,
    string_formatter,
    text_stats,
    unit_converter,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_AGENT_PATH = PROJECT_ROOT / "src" / "toolcall_rl" / "agents" / "Base_Model" / "agent.py"
FT_AGENT_PATH = PROJECT_ROOT / "src" / "toolcall_rl" / "agents" / "FT_Model" / "agent.py"


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


def test_demo_agents_expose_same_required_tools() -> None:
    base_agent = _load_agent(BASE_AGENT_PATH, "base_model_agent")
    ft_agent = _load_agent(FT_AGENT_PATH, "ft_model_agent")

    assert base_agent.name == "base_model_tool_demo"
    assert ft_agent.name == "ft_model_tool_demo"
    assert base_agent.model.model == "HuggingFaceTB/SmolLM-1.7B-Instruct"
    assert ft_agent.model.model == "HuggingFaceTB/SmolLM-1.7B-Instruct"
    assert ft_agent.model.adapter_dir.endswith("outputs/grpo_smollm_50tools_direct")
    assert _tool_names(base_agent) == _tool_names(ft_agent) == [
        "calculator",
        "google_search",
        "unit_converter",
        "text_stats",
        "string_formatter",
    ]


def _load_agent(agent_path: Path, module_name: str):
    spec = spec_from_file_location(module_name, agent_path)
    assert spec is not None
    assert spec.loader is not None

    agent_module = module_from_spec(spec)
    spec.loader.exec_module(agent_module)
    return agent_module.root_agent


def _tool_names(agent) -> list[str]:
    return [tool.__name__ for tool in agent.tools]
