import json
from collections import Counter

from toolcall_rl.data.build_dataset import build_dataset
from toolcall_rl.data.build_grpo_dataset import build_grpo_dataset
from toolcall_rl.data.export_grpo import to_grpo_record
from toolcall_rl.data.export_sft import EXAMPLES_PER_TOOL, _first_examples_per_tool, to_sft_record
from toolcall_rl.data.validate_dataset import validate_records
from toolcall_rl.evaluation.schemas import SFT_TOOL_NAMES, TOOL_BATCHES, TOOL_NAMES
from toolcall_rl.training.sft import format_example


def test_build_dataset_creates_balanced_canonical_records() -> None:
    records = build_dataset()

    assert len(SFT_TOOL_NAMES) == 20
    assert len(records) == 1000
    assert validate_records(records) == []
    assert Counter(record["expected_tool"] for record in records) == {
        tool: 50 for tool in SFT_TOOL_NAMES
    }


def test_sft_selection_uses_25_examples_per_tool() -> None:
    selected = _first_examples_per_tool(build_dataset(), EXAMPLES_PER_TOOL)

    assert len(selected) == 500
    assert Counter(record["expected_tool"] for record in selected) == {
        tool: 25 for tool in SFT_TOOL_NAMES
    }


def test_sft_export_shape_uses_chat_messages() -> None:
    record = build_dataset()[0]
    sft_record = to_sft_record(record)

    assert sft_record["id"] == record["id"]
    assert [message["role"] for message in sft_record["messages"]] == [
        "system",
        "user",
        "assistant",
    ]
    assert json.loads(sft_record["messages"][-1]["content"]) == record["assistant_response"]


def test_grpo_export_shape_keeps_labels_for_rewards() -> None:
    record = build_grpo_dataset()[0]
    grpo_record = to_grpo_record(record)

    assert grpo_record["id"] == record["id"]
    assert [message["role"] for message in grpo_record["prompt"]] == ["system", "user"]
    assert grpo_record["prompt"][0]["content"] == record["system_prompt"]
    assert grpo_record["expected_tool"] == record["expected_tool"]
    assert grpo_record["expected_args"] == record["expected_args"]


def test_grpo_dataset_is_balanced_and_uses_direct_prompts() -> None:
    records = build_grpo_dataset()
    sft_records = _first_examples_per_tool(build_dataset(), EXAMPLES_PER_TOOL)
    sft_labels = {
        (record["expected_tool"], json.dumps(record["expected_args"], sort_keys=True))
        for record in sft_records
    }

    assert len(TOOL_NAMES) == 50
    assert len(TOOL_BATCHES) == 5
    assert all(len(tool_batch) == 10 for tool_batch in TOOL_BATCHES)
    assert len(records) == 2500
    assert validate_records(records) == []
    assert Counter(record["expected_tool"] for record in records) == {
        tool: 50 for tool in TOOL_NAMES
    }
    assert all(record["id"].startswith("grpo_") for record in records)
    assert all(record["system_prompt"].count("\n- ") == 10 for record in records)
    assert all(record["expected_tool"] in record["system_prompt"] for record in records)
    assert all("\n" not in record["prompt"] for record in records)
    assert all("ARCHIVED:" not in record["prompt"] for record in records)
    assert all("Final request:" not in record["prompt"] for record in records)
    assert all(
        (record["expected_tool"], json.dumps(record["expected_args"], sort_keys=True))
        not in sft_labels
        for record in records
    )


def test_sft_format_example_flattens_messages() -> None:
    sft_record = to_sft_record(build_dataset()[0])

    formatted = format_example(sft_record)

    assert formatted["text"].startswith("<|system|>")
    assert "<|user|>" in formatted["text"]
    assert "<|assistant|>" in formatted["text"]
    assert formatted["text"].endswith("<|end|>")
