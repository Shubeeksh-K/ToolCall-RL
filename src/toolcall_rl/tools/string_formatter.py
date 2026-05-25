"""String formatting tool."""

from __future__ import annotations


def string_formatter(text: str, operation: str) -> dict[str, str]:
    """Format text with one of: uppercase, lowercase, titlecase, reverse."""

    normalized_operation = operation.strip().lower()
    operations = {
        "uppercase": text.upper,
        "lowercase": text.lower,
        "titlecase": text.title,
        "reverse": lambda: text[::-1],
    }

    formatter = operations.get(normalized_operation)
    if formatter is None:
        return {
            "text": text,
            "operation": normalized_operation,
            "error": "Unsupported operation.",
        }

    return {
        "text": text,
        "operation": normalized_operation,
        "result": formatter(),
    }
