"""Held-out cases for measuring tool-call generalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvalCase:
    prompt: str
    expected_tool: str
    expected_args: dict[str, Any]


HELD_OUT_EVAL_CASES = [
    EvalCase("Work out (73 * 9) - 14.", "calculator", {"expression": "(73 * 9) - 14"}),
    EvalCase(
        "Search Google for current LoRA adapter merging tutorials.",
        "google_search",
        {"query": "current LoRA adapter merging tutorials"},
    ),
    EvalCase(
        "I have 27.5 miles; express that in kilometers.",
        "unit_converter",
        {"value": 27.5, "from_unit": "miles", "to_unit": "kilometers"},
    ),
    EvalCase(
        'Give text statistics for: "Adapters are compact. Rewards improve precision."',
        "text_stats",
        {"text": "Adapters are compact. Rewards improve precision."},
    ),
    EvalCase(
        'Convert "REWARD SIGNAL" to lowercase.',
        "string_formatter",
        {"text": "REWARD SIGNAL", "operation": "lowercase"},
    ),
    EvalCase(
        "Get the weather for Madrid using celsius units.",
        "weather_lookup",
        {"city": "Madrid", "unit": "celsius"},
    ),
    EvalCase(
        "Exchange 325 USD into CAD.",
        "currency_converter",
        {"amount": 325, "from_currency": "USD", "to_currency": "CAD"},
    ),
    EvalCase(
        'Translate "machine learning" from English into Portuguese.',
        "translate_text",
        {"text": "machine learning", "source_language": "English", "target_language": "Portuguese"},
    ),
    EvalCase(
        'Add "Evaluation review" to my calendar on 2026-08-03 at 09:45 in UTC.',
        "create_calendar_event",
        {"title": "Evaluation review", "date": "2026-08-03", "time": "09:45", "timezone": "UTC"},
    ),
    EvalCase(
        'Email qa@example.com with subject "Test result" and message "All held-out checks passed."',
        "send_email",
        {"recipient": "qa@example.com", "subject": "Test result", "body": "All held-out checks passed."},
    ),
    EvalCase(
        "Find Vietnamese restaurants in Houston costing no more than 32 dollars.",
        "restaurant_search",
        {"city": "Houston", "cuisine": "Vietnamese", "max_price": 32},
    ),
    EvalCase(
        "Find 2 premium economy flights from Madrid to Lisbon on 2026-08-14.",
        "book_flight",
        {
            "origin": "Madrid",
            "destination": "Lisbon",
            "date": "2026-08-14",
            "passengers": 2,
            "cabin": "premium economy",
        },
    ),
    EvalCase(
        "Search for a hotel in Singapore for 3 guests from 2026-09-01 to 2026-09-05 under 260.",
        "hotel_search",
        {"city": "Singapore", "check_in": "2026-09-01", "check_out": "2026-09-05", "guests": 3, "max_price": 260},
    ),
    EvalCase(
        "Give me cycling directions from River Park to City Hall and avoid tolls.",
        "route_planner",
        {"origin": "River Park", "destination": "City Hall", "mode": "cycling", "avoid_tolls": True},
    ),
    EvalCase(
        "Find an ergonomic mouse below 65 dollars rated at least 4.4.",
        "product_search",
        {"query": "ergonomic mouse", "max_price": 65, "min_rating": 4.4},
    ),
    EvalCase(
        'Set a reminder to "Evaluate GRPO output" on 2026-08-07 at 18:45 in UTC.',
        "set_reminder",
        {"message": "Evaluate GRPO output", "date": "2026-08-07", "time": "18:45", "timezone": "UTC"},
    ),
    EvalCase(
        "Check my DHL package with tracking number JD9990001112223334.",
        "track_package",
        {"carrier": "DHL", "tracking_number": "JD9990001112223334"},
    ),
    EvalCase("Fetch a quote for ORCL.", "stock_quote", {"ticker": "ORCL"}),
    EvalCase(
        'Find CSV files matching "evaluation scores" inside /reports.',
        "file_search",
        {"query": "evaluation scores", "directory": "/reports", "file_type": "csv"},
    ),
    EvalCase(
        'Schedule "Training debrief" for 2026-08-09 at 13:15 in UTC with ana@example.com and jo@example.com.',
        "schedule_meeting",
        {
            "title": "Training debrief",
            "date": "2026-08-09",
            "time": "13:15",
            "timezone": "UTC",
            "attendees": ["ana@example.com", "jo@example.com"],
        },
    ),
]

# Kept as the public name used by the existing evaluation runners and notebooks.
SEED_EVAL_CASES = HELD_OUT_EVAL_CASES
