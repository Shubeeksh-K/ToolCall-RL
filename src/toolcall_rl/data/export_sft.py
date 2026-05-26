"""Export canonical tool-call data to SFT chat-message format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from toolcall_rl.data.build_dataset import DEFAULT_OUTPUT_PATH, PROJECT_ROOT
from toolcall_rl.data.validate_dataset import load_jsonl, validate_records
from toolcall_rl.evaluation.schemas import SYSTEM_PROMPT


DEFAULT_SFT_PATH = PROJECT_ROOT / "data" / "sft" / "tool_call_sft.jsonl"
EXAMPLES_PER_TOOL = 25


def to_sft_record(record: dict[str, Any]) -> dict[str, Any]:
    assistant_content = json.dumps(record["assistant_response"], ensure_ascii=True)
    return {
        "id": record["id"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": record["prompt"]},
            {"role": "assistant", "content": assistant_content},
        ],
    }


def export_sft(
    input_path: Path = DEFAULT_OUTPUT_PATH,
    output_path: Path = DEFAULT_SFT_PATH,
) -> list[dict[str, Any]]:
    records = load_jsonl(input_path)
    errors = validate_records(records)
    if errors:
        raise ValueError("\n".join(errors))

    selected_records = _first_examples_per_tool(records, EXAMPLES_PER_TOOL)
    sft_records = [to_sft_record(record) for record in selected_records]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in sft_records:
            file.write(json.dumps(record, ensure_ascii=True) + "\n")
    return sft_records


def _first_examples_per_tool(
    records: list[dict[str, Any]],
    examples_per_tool: int,
) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    selected = []
    for record in records:
        tool = record["expected_tool"]
        counts[tool] = counts.get(tool, 0) + 1
        if counts[tool] <= examples_per_tool:
            selected.append(record)
    return selected


def main() -> None:
    records = export_sft()
    print(f"wrote {len(records)} SFT records to {DEFAULT_SFT_PATH}")


if __name__ == "__main__":
    main()
