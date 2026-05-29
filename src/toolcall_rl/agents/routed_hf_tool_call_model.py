"""Routed HF bridge for the practical final ADK demo."""

from __future__ import annotations

import re
from collections.abc import AsyncGenerator

from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from toolcall_rl.agents.hf_tool_call_model import (
    DEMO_TOOL_NAMES,
    HuggingFaceToolCallModel,
    _format_tool_result,
    _latest_function_response,
    _latest_user_text,
    _parse_tool_call,
    _text_response,
)


GENERAL_RESPONSE = (
    "Hi! I can help with calculations, Google search queries, unit conversions, "
    "text statistics, and string formatting."
)


class RoutedHuggingFaceToolCallModel(HuggingFaceToolCallModel):
    """Use the fine-tuned model only for prompts that look tool-related."""

    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        tool_response = _latest_function_response(llm_request)
        if tool_response is not None:
            yield _text_response(_format_tool_result(tool_response))
            return

        user_text = _latest_user_text(llm_request)
        if not should_route_to_tool_model(user_text):
            yield _text_response(GENERAL_RESPONSE)
            return

        self._load_model()
        raw_output = self._generate(self._render_prompt(user_text))
        tool_call = _parse_tool_call(raw_output)

        if tool_call is None:
            yield _text_response(raw_output)
            return

        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part.from_function_call(
                        name=tool_call["tool"],
                        args=tool_call["args"],
                    )
                ],
            ),
            custom_metadata={"raw_model_output": raw_output},
        )


def should_route_to_tool_model(user_text: str) -> bool:
    """Return true when a prompt matches the demo tool surface."""

    text = user_text.lower()
    if not text.strip():
        return False

    if any(word in text for word in ("calculate", "compute", "evaluate", "work out")):
        return True

    if re.search(r"\d+\s*(\+|-|\*|/|%|\*\*)\s*\d+", text):
        return True

    if any(word in text for word in ("google", "search")):
        return True

    if "convert" in text:
        return True

    if any(phrase in text for phrase in ("text stats", "statistics", "count words", "count characters")):
        return True

    if any(word in text for word in ("uppercase", "lowercase", "titlecase", "reverse")):
        return True

    return any(tool_name in text for tool_name in DEMO_TOOL_NAMES)
