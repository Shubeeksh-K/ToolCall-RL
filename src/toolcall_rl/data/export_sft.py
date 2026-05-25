"""Export canonical tool-call data to SFT chat-message format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from toolcall_rl.data.build_dataset import DEFAULT_OUTPUT_PATH, PROJECT_ROOT
from toolcall_rl.data.validate_dataset import load_jsonl, validate_records
from toolcall_rl.evaluation.schemas import SYSTEM_PROMPT


DEFAULT_SFT_PATH = PROJECT_ROOT / "data" / "sft" / "tool_call_sft.jsonl"


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

    sft_records = [to_sft_record(record) for record in records]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in sft_records:
            file.write(json.dumps(record, ensure_ascii=True) + "\n")
    return sft_records


def main() -> None:
    records = export_sft()
    print(f"wrote {len(records)} SFT records to {DEFAULT_SFT_PATH}")


if __name__ == "__main__":
    main()
