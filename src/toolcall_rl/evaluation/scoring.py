"""Simple binary scoring for tool-call outputs."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

from .cases import EvalCase


@dataclass(frozen=True)
class Score:
    valid_json: int
    json_only: int
    tool_match: int
    args_match: int
    total_reward: int
    parsed: dict[str, Any] | None


def score_response(response_text: str, case: EvalCase) -> Score:
    """Score a model response against one expected tool call."""

    parsed = parse_json_object(response_text)
    valid_json = int(parsed is not None)
    json_only = int(is_json_only(response_text))

    tool_match = 0
    args_match = 0
    if parsed is not None:
        tool_match = int(parsed.get("tool") == case.expected_tool)
        args = parsed.get("args")
        if tool_match and isinstance(args, dict):
            args_match = int(args_contain_expected(args, case.expected_args))

    total_reward = valid_json + json_only + tool_match + args_match

    return Score(
        valid_json=valid_json,
        json_only=json_only,
        tool_match=tool_match,
        args_match=args_match,
        total_reward=total_reward,
        parsed=parsed,
    )


def parse_json_object(response_text: str) -> dict[str, Any] | None:
    """Parse a JSON object from a response, allowing text around it."""

    stripped = response_text.strip()
    parsed = _loads_object(stripped)
    if parsed is not None:
        return parsed

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    return _loads_object(stripped[start : end + 1])


def is_json_only(response_text: str) -> bool:
    """Return true only when the entire response is one JSON object."""

    return _loads_object(response_text.strip()) is not None


def args_contain_expected(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Return true when all expected args are present and equivalent."""

    for key, expected_value in expected.items():
        if key not in actual:
            return False
        if not values_match(actual[key], expected_value):
            return False

    return True


def values_match(actual: Any, expected: Any) -> bool:
    if isinstance(expected, int | float) and isinstance(actual, int | float):
        return math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=1e-9)

    if isinstance(expected, str):
        return normalize_text(str(actual)) == normalize_text(expected)

    return actual == expected


def normalize_text(value: str) -> str:
    """Normalize text for simple arg matching."""

    normalized = value.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.replace(" * ", "*").replace(" / ", "/")
    normalized = normalized.replace(" + ", "+").replace(" - ", "-")
    return normalized


def _loads_object(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None

    return parsed
