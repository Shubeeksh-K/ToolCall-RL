import json

from toolcall_rl.data.build_dataset import build_dataset
from toolcall_rl.data.export_grpo import to_grpo_record
from toolcall_rl.data.export_sft import to_sft_record
from toolcall_rl.data.validate_dataset import validate_records
from toolcall_rl.training.sft import format_example


def test_build_dataset_creates_balanced_canonical_records() -> None:
    records = build_dataset()

    assert len(records) == 250
    assert validate_records(records) == []
    assert {record["expected_tool"] for record in records} == {
        "calculator",
        "google_search",
        "unit_converter",
        "text_stats",
        "string_formatter",
    }
    assert {
        tool: sum(record["expected_tool"] == tool for record in records)
        for tool in {record["expected_tool"] for record in records}
    } == {
        "calculator": 50,
        "google_search": 50,
        "unit_converter": 50,
        "text_stats": 50,
        "string_formatter": 50,
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
    record = build_dataset()[0]
    grpo_record = to_grpo_record(record)

    assert grpo_record["id"] == record["id"]
    assert [message["role"] for message in grpo_record["prompt"]] == ["system", "user"]
    assert grpo_record["expected_tool"] == record["expected_tool"]
    assert grpo_record["expected_args"] == record["expected_args"]


def test_sft_format_example_flattens_messages() -> None:
    sft_record = to_sft_record(build_dataset()[0])

    formatted = format_example(sft_record)

    assert formatted["text"].startswith("<|system|>")
    assert "<|user|>" in formatted["text"]
    assert "<|assistant|>" in formatted["text"]
    assert formatted["text"].endswith("<|end|>")
