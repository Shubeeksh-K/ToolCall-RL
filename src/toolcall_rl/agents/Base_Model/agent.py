"""ADK assistant agent backed by an Ollama-hosted base model."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

from toolcall_rl.tools.calculator import calculator
from toolcall_rl.tools.google_search import google_search
from toolcall_rl.tools.string_formatter import string_formatter
from toolcall_rl.tools.text_stats import text_stats
from toolcall_rl.tools.unit_converter import unit_converter


load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "ollama_chat/smollm:1.7b")
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")


root_agent = Agent(
    name="general_help_agent",
    model=LiteLlm(
        model=OLLAMA_MODEL,
        api_base=OLLAMA_API_BASE,
    ),
    description="A general help assistant backed by Ollama and equipped with simple tools.",
    instruction=(
        "You are a concise, practical general help assistant. "
        "Use calculator for arithmetic, google_search when the user asks to search Google, "
        "unit_converter for simple unit conversions, text_stats for counting text, "
        "and string_formatter for simple text formatting. "
        "The google_search tool is intentionally stubbed and only returns the query."
    ),
    tools=[
        calculator,
        google_search,
        unit_converter,
        text_stats,
        string_formatter,
    ],
)
