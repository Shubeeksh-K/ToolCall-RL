"""ADK demo agent backed by the GRPO fine-tuned Hugging Face model."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from google.adk.agents import Agent

from toolcall_rl.agents.hf_tool_call_model import HuggingFaceToolCallModel, project_root
from toolcall_rl.tools.calculator import calculator
from toolcall_rl.tools.google_search import google_search
from toolcall_rl.tools.string_formatter import string_formatter
from toolcall_rl.tools.text_stats import text_stats
from toolcall_rl.tools.unit_converter import unit_converter


load_dotenv()

MODEL_ID = os.getenv("FT_HF_MODEL_ID", "HuggingFaceTB/SmolLM-1.7B-Instruct")
ADAPTER_DIR = os.getenv(
    "FT_ADAPTER_DIR",
    str(project_root() / "outputs" / "grpo_smollm_50tools_direct"),
)


root_agent = Agent(
    name="ft_model_tool_demo",
    model=HuggingFaceToolCallModel(
        model=MODEL_ID,
        adapter_dir=ADAPTER_DIR,
    ),
    description="A tool-call demo agent backed by the GRPO fine-tuned adapter.",
    instruction=(
        "You are a tool-call demo assistant. "
        "Use calculator for arithmetic, google_search when the user asks to search Google, "
        "unit_converter for simple unit conversions, text_stats for counting text, "
        "and string_formatter for simple text formatting. "
        "The model is expected to emit JSON tool calls; valid JSON will be bridged into ADK tool execution."
    ),
    tools=[
        calculator,
        google_search,
        unit_converter,
        text_stats,
        string_formatter,
    ],
)
