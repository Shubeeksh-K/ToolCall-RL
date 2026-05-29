"""Practical routed ADK demo agent backed by the GRPO fine-tuned model."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from google.adk.agents import Agent

from toolcall_rl.agents.hf_tool_call_model import project_root
from toolcall_rl.agents.routed_hf_tool_call_model import RoutedHuggingFaceToolCallModel
from toolcall_rl.tools.calculator import calculator
from toolcall_rl.tools.google_search import google_search
from toolcall_rl.tools.string_formatter import string_formatter
from toolcall_rl.tools.text_stats import text_stats
from toolcall_rl.tools.unit_converter import unit_converter


load_dotenv()

MODEL_ID = os.getenv("FINAL_HF_MODEL_ID", "HuggingFaceTB/SmolLM-1.7B-Instruct")
ADAPTER_DIR = os.getenv(
    "FINAL_ADAPTER_DIR",
    str(project_root() / "outputs" / "grpo_smollm_50tools_direct"),
)


root_agent = Agent(
    name="final_model_tool_demo",
    model=RoutedHuggingFaceToolCallModel(
        model=MODEL_ID,
        adapter_dir=ADAPTER_DIR,
    ),
    description="A routed ADK agent that uses the GRPO adapter for tool-like prompts.",
    instruction=(
        "You are a practical tool-call demo assistant. "
        "For tool-like requests, use calculator, google_search, unit_converter, "
        "text_stats, or string_formatter. "
        "For ordinary greetings or general chat, respond briefly without using a tool."
    ),
    tools=[
        calculator,
        google_search,
        unit_converter,
        text_stats,
        string_formatter,
    ],
)
