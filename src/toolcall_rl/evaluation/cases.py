"""Small seed dataset for baseline tool-call evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvalCase:
    prompt: str
    expected_tool: str
    expected_args: dict[str, Any]


SEED_EVAL_CASES = [
    EvalCase(
        prompt="What is 24 * 17?",
        expected_tool="calculator",
        expected_args={"expression": "24 * 17"},
    ),
    EvalCase(
        prompt="Calculate (892 + 431) / 3 using the calculator.",
        expected_tool="calculator",
        expected_args={"expression": "(892 + 431) / 3"},
    ),
    EvalCase(
        prompt="Search Google for recent news about open source AI agents.",
        expected_tool="google_search",
        expected_args={"query": "recent news about open source AI agents"},
    ),
    EvalCase(
        prompt="Use Google to find information about the Google ADK.",
        expected_tool="google_search",
        expected_args={"query": "Google ADK"},
    ),
    EvalCase(
        prompt="Convert 10 kilometers to miles.",
        expected_tool="unit_converter",
        expected_args={"value": 10, "from_unit": "kilometers", "to_unit": "miles"},
    ),
    EvalCase(
        prompt="How many pounds are in 7 kilograms?",
        expected_tool="unit_converter",
        expected_args={"value": 7, "from_unit": "kilograms", "to_unit": "pounds"},
    ),
    EvalCase(
        prompt='Count the words and sentences in: "Hello world. Tool calls work!"',
        expected_tool="text_stats",
        expected_args={"text": "Hello world. Tool calls work!"},
    ),
    EvalCase(
        prompt='How many characters are in this text: "small models can learn tools"',
        expected_tool="text_stats",
        expected_args={"text": "small models can learn tools"},
    ),
    EvalCase(
        prompt='Make this title case: "learning tool calls"',
        expected_tool="string_formatter",
        expected_args={"text": "learning tool calls", "operation": "titlecase"},
    ),
    EvalCase(
        prompt='Reverse this text: "stressed"',
        expected_tool="string_formatter",
        expected_args={"text": "stressed", "operation": "reverse"},
    ),
]
