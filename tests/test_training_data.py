import json
from collections import Counter
from pathlib import Path

from toolcall_rl.evaluation.schemas import TOOL_BATCHES, TOOL_NAMES
from toolcall_rl.training.sft import format_example


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SFT_DATA_PATH = PROJECT_ROOT / "data" / "sft" / "tool_call_sft.jsonl"
GRPO_DATA_PATH = PROJECT_ROOT / "data" / "grpo" / "tool_call_grpo.jsonl"


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_sft_training_data_contains_20_tools_with_25_examples_each() -> None:
    records = _load_jsonl(SFT_DATA_PATH)
    tools = [
        json.loads(record["messages"][-1]["content"])["tool"]
        for record in records
    ]

    assert len(records) == 500
    assert len(set(tools)) == 20
    assert set(Counter(tools).values()) == {25}
    assert all([message["role"] for message in record["messages"]] == ["system", "user", "assistant"] for record in records)


def test_grpo_training_data_contains_50_tools_with_50_examples_each() -> None:
    records = _load_jsonl(GRPO_DATA_PATH)

    assert len(TOOL_NAMES) == 50
    assert len(TOOL_BATCHES) == 5
    assert all(len(tool_batch) == 10 for tool_batch in TOOL_BATCHES)
    assert len(records) == 2500
    assert Counter(record["expected_tool"] for record in records) == {
        tool: 50 for tool in TOOL_NAMES
    }
    assert all([message["role"] for message in record["prompt"]] == ["system", "user"] for record in records)
    assert all(record["prompt"][0]["content"].count("\n- ") == 10 for record in records)
    assert all(record["expected_tool"] in record["prompt"][0]["content"] for record in records)


def test_sft_format_example_flattens_messages() -> None:
    record = _load_jsonl(SFT_DATA_PATH)[0]
    formatted = format_example(record)

    assert formatted["text"].startswith("<|system|>")
    assert "<|user|>" in formatted["text"]
    assert "<|assistant|>" in formatted["text"]
    assert formatted["text"].endswith("<|end|>")
