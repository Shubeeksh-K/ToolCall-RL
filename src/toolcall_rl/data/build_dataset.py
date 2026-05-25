"""Build the canonical tool-call dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "tool_call_dataset.jsonl"


def build_dataset() -> list[dict[str, Any]]:
    """Create deterministic canonical tool-call examples."""

    records = []
    records.extend(_calculator_records())
    records.extend(_google_search_records())
    records.extend(_unit_converter_records())
    records.extend(_text_stats_records())
    records.extend(_string_formatter_records())
    return _with_ids(records)


def write_dataset(path: Path = DEFAULT_OUTPUT_PATH) -> list[dict[str, Any]]:
    """Write the canonical dataset to JSONL."""

    records = build_dataset()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=True) + "\n")
    return records


def _with_ids(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counters: dict[str, int] = {}
    output = []

    for record in records:
        tool = record["expected_tool"]
        counters[tool] = counters.get(tool, 0) + 1
        output.append({"id": f"{tool}_{counters[tool]:04d}", **record})

    return output


def _record(prompt: str, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "expected_tool": tool,
        "expected_args": args,
        "assistant_response": {
            "tool": tool,
            "args": args,
        },
    }


def _calculator_records() -> list[dict[str, Any]]:
    expressions = [
        "24 * 17",
        "892 + 431",
        "(18 + 6) / 3",
        "55 * 12",
        "144 / 12",
        "19 - 47",
        "7 ** 3",
        "81 % 7",
        "(42 - 9) * 2",
        "100 // 9",
    ]
    templates = [
        "What is {expression}?",
        "Calculate {expression}.",
        "Use the calculator to solve {expression}.",
        "Find the value of {expression}.",
        "I need the result of {expression}.",
    ]
    return [
        _record(template.format(expression=expression), "calculator", {"expression": expression})
        for expression in expressions
        for template in templates
    ]


def _google_search_records() -> list[dict[str, Any]]:
    queries = [
        "open source AI agents",
        "Google ADK",
        "tool calling language models",
        "latest Python packaging guide",
        "Ollama LiteLLM integration",
        "GRPO fine tuning",
        "supervised fine tuning datasets",
        "small language models with tools",
        "JSON tool calling format",
        "function calling datasets",
    ]
    templates = [
        "Search Google for {query}.",
        "Use Google to find information about {query}.",
        "Find web results for {query}.",
        "Look up {query} on Google.",
        "I need Google search results for {query}.",
    ]
    return [
        _record(template.format(query=query), "google_search", {"query": query})
        for query in queries
        for template in templates
    ]


def _unit_converter_records() -> list[dict[str, Any]]:
    conversions = [
        (10, "kilometers", "miles"),
        (7, "kilograms", "pounds"),
        (3, "meters", "feet"),
        (12, "feet", "meters"),
        (25, "celsius", "fahrenheit"),
        (86, "fahrenheit", "celsius"),
        (5, "miles", "kilometers"),
        (2.5, "kilograms", "pounds"),
        (100, "meters", "feet"),
        (42, "kilometers", "miles"),
    ]
    templates = [
        "Convert {value} {from_unit} to {to_unit}.",
        "How many {to_unit} are in {value} {from_unit}?",
        "Use the unit converter for {value} {from_unit} into {to_unit}.",
        "Change {value} {from_unit} over to {to_unit}.",
        "I need {value} {from_unit} expressed as {to_unit}.",
    ]
    return [
        _record(
            template.format(value=value, from_unit=from_unit, to_unit=to_unit),
            "unit_converter",
            {"value": value, "from_unit": from_unit, "to_unit": to_unit},
        )
        for value, from_unit, to_unit in conversions
        for template in templates
    ]


def _text_stats_records() -> list[dict[str, Any]]:
    texts = [
        "Hello world. Tool calls work!",
        "small models can learn tools",
        "training data matters",
        "One sentence only.",
        "open weights local inference",
        "First sentence. Second sentence. Third sentence.",
        "JSON only outputs are strict.",
        "abc 123",
        "reward models guide behavior",
        "simple tools are useful",
    ]
    templates = [
        'Count the words and sentences in: "{text}"',
        'How many characters are in this text: "{text}"',
        'Count the words in: "{text}"',
        'Get text stats for: "{text}"',
        'Analyze this text: "{text}"',
    ]
    return [
        _record(template.format(text=text), "text_stats", {"text": text})
        for text in texts
        for template in templates
    ]


def _string_formatter_records() -> list[dict[str, Any]]:
    examples = [
        ("learning tool calls", "titlecase"),
        ("stressed", "reverse"),
        ("small model", "uppercase"),
        ("LOUD WORDS", "lowercase"),
        ("agent training loop", "titlecase"),
        ("drawer", "reverse"),
        ("json please", "uppercase"),
        ("Tool CALLS", "lowercase"),
        ("fine tuning data", "titlecase"),
        ("tools", "reverse"),
    ]
    templates = [
        'Apply {operation} to: "{text}"',
        'Use string formatting to {operation} "{text}".',
        'Format this text with {operation}: "{text}"',
        'Please {operation} the text "{text}".',
        'Run the {operation} operation on "{text}".',
    ]
    return [
        _record(
            template.format(text=text, operation=operation),
            "string_formatter",
            {"text": text, "operation": operation},
        )
        for text, operation in examples
        for template in templates
    ]


def main() -> None:
    records = write_dataset()
    print(f"wrote {len(records)} records to {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
