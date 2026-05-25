"""Export canonical tool-call data to GRPO prompt/label format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from toolcall_rl.data.build_dataset import DEFAULT_OUTPUT_PATH, PROJECT_ROOT
from toolcall_rl.data.validate_dataset import load_jsonl, validate_records
from toolcall_rl.evaluation.schemas import SYSTEM_PROMPT


DEFAULT_GRPO_PATH = PROJECT_ROOT / "data" / "grpo" / "tool_call_grpo.jsonl"


def to_grpo_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": record["prompt"]},
        ],
        "expected_tool": record["expected_tool"],
        "expected_args": record["expected_args"],
    }


def export_grpo(
    input_path: Path = DEFAULT_OUTPUT_PATH,
    output_path: Path = DEFAULT_GRPO_PATH,
) -> list[dict[str, Any]]:
    records = load_jsonl(input_path)
    errors = validate_records(records)
    if errors:
        raise ValueError("\n".join(errors))

    grpo_records = [to_grpo_record(record) for record in records]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in grpo_records:
            file.write(json.dumps(record, ensure_ascii=True) + "\n")
    return grpo_records


def main() -> None:
    records = export_grpo()
    print(f"wrote {len(records)} GRPO records to {DEFAULT_GRPO_PATH}")


if __name__ == "__main__":
    main()
