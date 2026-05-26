from toolcall_rl.evaluation.cases import HELD_OUT_EVAL_CASES, EvalCase
from toolcall_rl.evaluation.direct_50_cases import DIRECT_50_EVAL_CASES
from toolcall_rl.evaluation.rewards import (
    reward_args_match,
    reward_json_only,
    reward_tool_match,
    reward_valid_json,
)
from toolcall_rl.evaluation.scoring import score_response
from toolcall_rl.evaluation.schemas import SFT_TOOL_NAMES, TOOL_NAMES


def test_held_out_eval_covers_each_training_tool() -> None:
    assert len(HELD_OUT_EVAL_CASES) == 20
    assert {case.expected_tool for case in HELD_OUT_EVAL_CASES} == set(SFT_TOOL_NAMES)


def test_direct_50_eval_covers_all_grpo_tools_in_ten_tool_batches() -> None:
    assert len(DIRECT_50_EVAL_CASES) == 50
    assert {case.expected_tool for case in DIRECT_50_EVAL_CASES} == set(TOOL_NAMES)
    assert all(case.system_prompt.count("\n- ") == 10 for case in DIRECT_50_EVAL_CASES)
    assert all(case.expected_tool in case.system_prompt for case in DIRECT_50_EVAL_CASES)
    assert all("ARCHIVED:" not in case.prompt and "FINAL:" not in case.prompt for case in DIRECT_50_EVAL_CASES)


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


def test_calculator_args_match_equivalent_expression_results() -> None:
    case = EvalCase(
        prompt="Work out (73 * 9) - 14.",
        expected_tool="calculator",
        expected_args={"expression": "(73 * 9) - 14"},
    )

    score = score_response(
        '{"tool": "calculator", "args": {"expression": "73 * 9 - 14"}}',
        case,
    )

    assert score.args_match == 1
    assert score.total_reward == 4


def test_calculator_args_reject_different_expression_results() -> None:
    case = EvalCase(
        prompt="Work out (2 + 3) * 4.",
        expected_tool="calculator",
        expected_args={"expression": "(2 + 3) * 4"},
    )

    score = score_response(
        '{"tool": "calculator", "args": {"expression": "2 + 3 * 4"}}',
        case,
    )

    assert score.args_match == 0
    assert score.total_reward == 3


def test_compare_products_features_match_regardless_of_order() -> None:
    case = EvalCase(
        prompt="Compare two laptops.",
        expected_tool="compare_products",
        expected_args={
            "products": ["PeakBook", "ThinNote"],
            "category": "laptop",
            "currency": "USD",
            "max_price": 1450,
            "features": ["battery life", "linux support"],
        },
    )

    score = score_response(
        '{"tool": "compare_products", "args": {"products": ["PeakBook", "ThinNote"], '
        '"category": "laptop", "currency": "USD", "max_price": 1450, '
        '"features": ["linux support", "battery life"]}}',
        case,
    )

    assert score.args_match == 1
    assert score.total_reward == 4


def test_compare_products_features_still_reject_wrong_values() -> None:
    case = EvalCase(
        prompt="Compare two laptops.",
        expected_tool="compare_products",
        expected_args={"features": ["battery life", "linux support"]},
    )

    score = score_response(
        '{"tool": "compare_products", "args": {"features": ["linux support", "touchscreen"]}}',
        case,
    )

    assert score.args_match == 0
    assert score.total_reward == 3


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
