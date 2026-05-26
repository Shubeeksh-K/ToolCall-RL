"""GRPO reward functions backed by the shared tool-call scorer."""

from __future__ import annotations

from typing import Any

from toolcall_rl.evaluation.scoring import EvalCase, Score, score_response


def reward_valid_json(
    completions: list[Any],
    expected_tool: list[str],
    expected_args: list[dict[str, Any]],
    **kwargs: Any,
) -> list[int]:
    """Reward responses that contain a parseable JSON object."""

    return [score.valid_json for score in _scores(completions, expected_tool, expected_args)]


def reward_json_only(
    completions: list[Any],
    expected_tool: list[str],
    expected_args: list[dict[str, Any]],
    **kwargs: Any,
) -> list[int]:
    """Reward responses that contain only a JSON object."""

    return [score.json_only for score in _scores(completions, expected_tool, expected_args)]


def reward_tool_match(
    completions: list[Any],
    expected_tool: list[str],
    expected_args: list[dict[str, Any]],
    **kwargs: Any,
) -> list[int]:
    """Reward selecting the expected tool."""

    return [score.tool_match for score in _scores(completions, expected_tool, expected_args)]


def reward_args_match(
    completions: list[Any],
    expected_tool: list[str],
    expected_args: list[dict[str, Any]],
    **kwargs: Any,
) -> list[int]:
    """Reward preserving the expected tool arguments."""

    return [score.args_match for score in _scores(completions, expected_tool, expected_args)]


def _scores(
    completions: list[Any],
    expected_tools: list[str],
    expected_args: list[dict[str, Any]],
) -> list[Score]:
    scores = []
    for completion, tool, args in zip(completions, expected_tools, expected_args, strict=True):
        case = EvalCase(prompt="", expected_tool=tool, expected_args=args)
        scores.append(score_response(_completion_text(completion), case))
    return scores


def _completion_text(completion: Any) -> str:
    """Extract assistant text from TRL conversational or plain completions."""

    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        message = completion[-1]
        if isinstance(message, dict):
            return str(message.get("content", ""))
    return ""
