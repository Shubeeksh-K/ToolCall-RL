"""Validate the canonical tool-call dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from toolcall_rl.data.build_dataset import DEFAULT_OUTPUT_PATH
from toolcall_rl.evaluation.schemas import TOOL_NAMES
from toolcall_rl.evaluation.scoring import is_json_only


REQUIRED_ARG_KEYS = {
    "calculator": {"expression"},
    "google_search": {"query"},
    "unit_converter": {"value", "from_unit", "to_unit"},
    "text_stats": {"text"},
    "string_formatter": {"text", "operation"},
}


def load_jsonl(path: Path = DEFAULT_OUTPUT_PATH) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def validate_records(records: list[dict[str, Any]]) -> list[str]:
    """Return validation errors for canonical records."""

    errors = []
    seen_ids = set()

    for index, record in enumerate(records, start=1):
        prefix = f"record {index}"
        record_id = record.get("id")
        if not record_id:
            errors.append(f"{prefix}: missing id")
        elif record_id in seen_ids:
            errors.append(f"{prefix}: duplicate id {record_id}")
        seen_ids.add(record_id)

        tool = record.get("expected_tool")
        if tool not in TOOL_NAMES:
            errors.append(f"{prefix}: invalid expected_tool {tool}")
            continue

        expected_args = record.get("expected_args")
        if not isinstance(expected_args, dict):
            errors.append(f"{prefix}: expected_args must be an object")
            continue

        missing_keys = REQUIRED_ARG_KEYS[tool] - set(expected_args)
        if missing_keys:
            errors.append(f"{prefix}: missing expected_args keys {sorted(missing_keys)}")

        assistant_response = record.get("assistant_response")
        if assistant_response != {"tool": tool, "args": expected_args}:
            errors.append(f"{prefix}: assistant_response does not match expected labels")
            continue

        assistant_text = json.dumps(assistant_response, ensure_ascii=True)
        if not is_json_only(assistant_text):
            errors.append(f"{prefix}: assistant_response is not JSON-only")

    return errors


def validate_dataset(path: Path = DEFAULT_OUTPUT_PATH) -> None:
    records = load_jsonl(path)
    errors = validate_records(records)
    if errors:
        raise ValueError("\n".join(errors))
    print(f"validated {len(records)} records from {path}")


def main() -> None:
    validate_dataset()


if __name__ == "__main__":
    main()
