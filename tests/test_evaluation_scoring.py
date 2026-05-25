from toolcall_rl.evaluation.cases import EvalCase
from toolcall_rl.evaluation.rewards import (
    reward_args_match,
    reward_json_only,
    reward_tool_match,
    reward_valid_json,
)
from toolcall_rl.evaluation.scoring import score_response


def test_score_response_passes_exact_tool_call() -> None:
    case = EvalCase(
        prompt="What is 24 * 17?",
        expected_tool="calculator",
        expected_args={"expression": "24 * 17"},
    )

    score = score_response(
        '{"tool": "calculator", "args": {"expression": "24*17"}}',
        case,
    )

    assert score.valid_json == 1
    assert score.json_only == 1
    assert score.tool_match == 1
    assert score.args_match == 1
    assert score.total_reward == 4


def test_score_response_detects_json_with_extra_text() -> None:
    case = EvalCase(
        prompt="Search Google for ADK.",
        expected_tool="google_search",
        expected_args={"query": "Google ADK"},
    )

    score = score_response(
        'I will use this: {"tool": "google_search", "args": {"query": "Google ADK"}}',
        case,
    )

    assert score.valid_json == 1
    assert score.json_only == 0
    assert score.tool_match == 1
    assert score.args_match == 1
    assert score.total_reward == 3


def test_score_response_rejects_wrong_tool_args_reward() -> None:
    case = EvalCase(
        prompt="Convert 10 kilometers to miles.",
        expected_tool="unit_converter",
        expected_args={"value": 10, "from_unit": "kilometers", "to_unit": "miles"},
    )

    score = score_response(
        '{"tool": "calculator", "args": {"expression": "10 kilometers to miles"}}',
        case,
    )

    assert score.valid_json == 1
    assert score.json_only == 1
    assert score.tool_match == 0
    assert score.args_match == 0
    assert score.total_reward == 2


def test_score_response_rejects_natural_language() -> None:
    case = EvalCase(
        prompt="Reverse stressed.",
        expected_tool="string_formatter",
        expected_args={"text": "stressed", "operation": "reverse"},
    )

    score = score_response("Use the string formatter tool.", case)

    assert score.valid_json == 0
    assert score.json_only == 0
    assert score.tool_match == 0
    assert score.args_match == 0
    assert score.total_reward == 0


def test_grpo_rewards_score_conversational_completions() -> None:
    completions = [
        [
            {
                "role": "assistant",
                "content": '{"tool": "google_search", "args": {"query": "recent news"}}',
            }
        ],
        [{"role": "assistant", "content": "I would search for recent news."}],
    ]
    expected_tools = ["google_search", "google_search"]
    expected_args = [{"query": "recent news"}, {"query": "recent news"}]

    assert reward_valid_json(completions, expected_tools, expected_args) == [1, 0]
    assert reward_json_only(completions, expected_tools, expected_args) == [1, 0]
    assert reward_tool_match(completions, expected_tools, expected_args) == [1, 0]
    assert reward_args_match(completions, expected_tools, expected_args) == [1, 0]
