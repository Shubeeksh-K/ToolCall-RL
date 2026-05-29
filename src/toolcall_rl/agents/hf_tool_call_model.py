"""Hugging Face model wrapper that bridges learned JSON tool calls into ADK."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import torch
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import PrivateAttr
from transformers import AutoModelForCausalLM, AutoTokenizer

from toolcall_rl.evaluation.schemas import build_system_prompt
from toolcall_rl.evaluation.scoring import parse_json_object


DEMO_TOOL_NAMES = [
    "calculator",
    "google_search",
    "unit_converter",
    "text_stats",
    "string_formatter",
]

DEMO_SYSTEM_PROMPT = build_system_prompt(DEMO_TOOL_NAMES)


class HuggingFaceToolCallModel(BaseLlm):
    """Load a local HF model and expose JSON tool calls as ADK function calls."""

    adapter_dir: str | None = None
    max_new_tokens: int = 128

    _tokenizer: Any = PrivateAttr(default=None)
    _model: Any = PrivateAttr(default=None)
    _device: str = PrivateAttr(default="cpu")

    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        tool_response = _latest_function_response(llm_request)
        if tool_response is not None:
            yield _text_response(_format_tool_result(tool_response))
            return

        self._load_model()
        prompt = self._render_prompt(_latest_user_text(llm_request))
        raw_output = self._generate(prompt)
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

    def _load_model(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self._device == "cuda" else torch.float32
        tokenizer_source = self.adapter_dir or self.model

        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(self.model, dtype=dtype)
        if self.adapter_dir is not None:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(base_model, self.adapter_dir)
        else:
            self._model = base_model

        self._model.to(self._device)
        self._model.eval()

    def _render_prompt(self, user_text: str) -> str:
        messages = [
            {"role": "system", "content": DEMO_SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]
        if self._tokenizer.chat_template:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return "\n".join(
            [
                f"<|system|>\n{DEMO_SYSTEM_PROMPT}",
                f"<|user|>\n{user_text}",
                "<|assistant|>\n",
            ]
        )

    def _generate(self, prompt: str) -> str:
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0, inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _parse_tool_call(raw_output: str) -> dict[str, Any] | None:
    parsed = parse_json_object(raw_output)
    if parsed is None:
        return None

    tool_name = parsed.get("tool")
    args = parsed.get("args")
    if tool_name not in DEMO_TOOL_NAMES or not isinstance(args, dict):
        return None

    return {"tool": tool_name, "args": args}


def _latest_user_text(llm_request: LlmRequest) -> str:
    for content in reversed(llm_request.contents):
        if content.role != "user":
            continue
        text_parts = [part.text for part in content.parts or [] if part.text]
        if text_parts:
            return "\n".join(text_parts)
    return ""


def _latest_function_response(llm_request: LlmRequest) -> types.FunctionResponse | None:
    for content in reversed(llm_request.contents):
        for part in reversed(content.parts or []):
            if part.function_response is not None:
                return part.function_response
    return None


def _format_tool_result(function_response: types.FunctionResponse) -> str:
    return (
        f"Tool `{function_response.name}` returned:\n"
        f"{json.dumps(function_response.response, ensure_ascii=False)}"
    )


def _text_response(text: str) -> LlmResponse:
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text=text)],
        )
    )


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]
