"""Text statistics tool."""

from __future__ import annotations

import re


def text_stats(text: str) -> dict[str, int]:
    """Count basic properties of a text string."""

    words = re.findall(r"\b\w+\b", text)
    sentences = re.findall(r"[^.!?]+[.!?]?", text)
    non_empty_sentences = [sentence for sentence in sentences if sentence.strip()]

    return {
        "characters": len(text),
        "characters_without_spaces": len(text.replace(" ", "")),
        "words": len(words),
        "sentences": len(non_empty_sentences),
    }
