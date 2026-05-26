"""Build the canonical 20-tool dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "tool_call_dataset.jsonl"


def build_dataset() -> list[dict[str, Any]]:
    """Create 50 deterministic examples for each of 20 tools."""

    records = []
    for tool, values, templates in _tool_examples():
        records.extend(_records(tool, values, templates))
    return _with_ids(records)


def write_dataset(path: Path = DEFAULT_OUTPUT_PATH) -> list[dict[str, Any]]:
    records = build_dataset()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=True) + "\n")
    return records


def _records(
    tool: str,
    values: list[dict[str, Any]],
    templates: list[str],
) -> list[dict[str, Any]]:
    assert len(values) == 10
    assert len(templates) == 5
    output = []
    for args in values:
        prompt_args = {key: _prompt_value(value) for key, value in args.items()}
        for template in templates:
            output.append(_record(template.format(**prompt_args), tool, args))
    return output


def _prompt_value(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(value)
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _with_ids(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counters: dict[str, int] = {}
    output = []
    for record in records:
        tool = record["expected_tool"]
        counters[tool] = counters.get(tool, 0) + 1
        output.append({"id": f"{tool}_{counters[tool]:04d}", **record})
    return output


def _record(prompt: str, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "expected_tool": tool,
        "expected_args": args,
        "assistant_response": {"tool": tool, "args": args},
    }


def _tool_examples() -> list[tuple[str, list[dict[str, Any]], list[str]]]:
    return [
        (
            "calculator",
            [{"expression": value} for value in [
                "24 * 17", "892 + 431", "(18 + 6) / 3", "55 * 12", "144 / 12",
                "19 - 47", "7 ** 3", "81 % 7", "(42 - 9) * 2", "100 // 9",
            ]],
            [
                "What is {expression}?",
                "Calculate {expression}.",
                "Use the calculator to solve {expression}.",
                "Find the value of {expression}.",
                "I need the result of {expression}.",
            ],
        ),
        (
            "google_search",
            [{"query": value} for value in [
                "open source AI agents", "Google ADK", "tool calling language models",
                "latest Python packaging guide", "Ollama LiteLLM integration", "GRPO fine tuning",
                "supervised fine tuning datasets", "small language models with tools",
                "JSON tool calling format", "function calling datasets",
            ]],
            [
                "Search Google for {query}.",
                "Use Google to find information about {query}.",
                "Find web results for {query}.",
                "Look up {query} on Google.",
                "I need Google search results for {query}.",
            ],
        ),
        (
            "unit_converter",
            [
                {"value": 10, "from_unit": "kilometers", "to_unit": "miles"},
                {"value": 7, "from_unit": "kilograms", "to_unit": "pounds"},
                {"value": 3, "from_unit": "meters", "to_unit": "feet"},
                {"value": 12, "from_unit": "feet", "to_unit": "meters"},
                {"value": 25, "from_unit": "celsius", "to_unit": "fahrenheit"},
                {"value": 86, "from_unit": "fahrenheit", "to_unit": "celsius"},
                {"value": 5, "from_unit": "miles", "to_unit": "kilometers"},
                {"value": 2.5, "from_unit": "kilograms", "to_unit": "pounds"},
                {"value": 100, "from_unit": "meters", "to_unit": "feet"},
                {"value": 42, "from_unit": "kilometers", "to_unit": "miles"},
            ],
            [
                "Convert {value} {from_unit} to {to_unit}.",
                "How many {to_unit} are in {value} {from_unit}?",
                "Use the unit converter for {value} {from_unit} into {to_unit}.",
                "Change {value} {from_unit} over to {to_unit}.",
                "I need {value} {from_unit} expressed as {to_unit}.",
            ],
        ),
        (
            "text_stats",
            [{"text": value} for value in [
                "Hello world. Tool calls work!", "small models can learn tools",
                "training data matters", "One sentence only.", "open weights local inference",
                "First sentence. Second sentence. Third sentence.", "JSON only outputs are strict.",
                "abc 123", "reward models guide behavior", "simple tools are useful",
            ]],
            [
                'Count the words and sentences in: "{text}"',
                'How many characters are in this text: "{text}"',
                'Count the words in: "{text}"',
                'Get text stats for: "{text}"',
                'Analyze this text: "{text}"',
            ],
        ),
        (
            "string_formatter",
            [
                {"text": "learning tool calls", "operation": "titlecase"},
                {"text": "stressed", "operation": "reverse"},
                {"text": "small model", "operation": "uppercase"},
                {"text": "LOUD WORDS", "operation": "lowercase"},
                {"text": "agent training loop", "operation": "titlecase"},
                {"text": "drawer", "operation": "reverse"},
                {"text": "json please", "operation": "uppercase"},
                {"text": "Tool CALLS", "operation": "lowercase"},
                {"text": "fine tuning data", "operation": "titlecase"},
                {"text": "tools", "operation": "reverse"},
            ],
            [
                'Apply {operation} to: "{text}"',
                'Use string formatting to {operation} "{text}".',
                'Format this text with {operation}: "{text}"',
                'Please {operation} the text "{text}".',
                'Run the {operation} operation on "{text}".',
            ],
        ),
        (
            "weather_lookup",
            [{"city": city, "unit": unit} for city, unit in [
                ("Chicago", "fahrenheit"), ("Bangalore", "celsius"), ("London", "celsius"),
                ("Tokyo", "celsius"), ("New York", "fahrenheit"), ("Berlin", "celsius"),
                ("Sydney", "celsius"), ("Seattle", "fahrenheit"), ("Toronto", "celsius"),
                ("Dubai", "celsius"),
            ]],
            [
                "What is the weather in {city} in {unit}?",
                "Check {city} weather using {unit}.",
                "Look up the temperature for {city} in {unit}.",
                "Get the {unit} forecast for {city}.",
                "I need weather details for {city}, measured in {unit}.",
            ],
        ),
        (
            "currency_converter",
            [
                {"amount": amount, "from_currency": source, "to_currency": target}
                for amount, source, target in [
                    (100, "USD", "EUR"), (50, "EUR", "GBP"), (1200, "INR", "USD"),
                    (75, "GBP", "USD"), (500, "JPY", "USD"), (250, "CAD", "USD"),
                    (80, "AUD", "NZD"), (150, "USD", "INR"), (90, "EUR", "JPY"),
                    (40, "CHF", "EUR"),
                ]
            ],
            [
                "Convert {amount} {from_currency} to {to_currency}.",
                "How much is {amount} {from_currency} in {to_currency}?",
                "Use currency conversion for {amount} {from_currency} into {to_currency}.",
                "Exchange {amount} {from_currency} for {to_currency}.",
                "Give me the {to_currency} value of {amount} {from_currency}.",
            ],
        ),
        (
            "translate_text",
            [
                {"text": text, "source_language": source, "target_language": target}
                for text, source, target in [
                    ("hello world", "English", "Spanish"), ("good morning", "English", "French"),
                    ("thank you", "English", "German"), ("where is the station", "English", "Japanese"),
                    ("bonjour", "French", "English"), ("buenos dias", "Spanish", "English"),
                    ("data science", "English", "Hindi"), ("see you soon", "English", "Italian"),
                    ("guten tag", "German", "English"), ("tool calling", "English", "Tamil"),
                ]
            ],
            [
                'Translate "{text}" from {source_language} to {target_language}.',
                'Convert this {source_language} text to {target_language}: "{text}"',
                'Use translation on "{text}" from {source_language} into {target_language}.',
                'I need "{text}" translated from {source_language} to {target_language}.',
                'What is "{text}" in {target_language} if it is {source_language}?',
            ],
        ),
        (
            "create_calendar_event",
            [
                {"title": title, "date": date, "time": time, "timezone": zone}
                for title, date, time, zone in [
                    ("Design review", "2026-06-02", "14:00", "UTC"),
                    ("Team standup", "2026-06-03", "09:30", "America/New_York"),
                    ("Doctor appointment", "2026-06-10", "11:00", "America/Chicago"),
                    ("Project demo", "2026-06-12", "16:00", "UTC"),
                    ("Interview", "2026-06-15", "10:00", "Asia/Kolkata"),
                    ("Budget review", "2026-06-18", "13:00", "Europe/London"),
                    ("Training session", "2026-06-20", "15:30", "UTC"),
                    ("Client call", "2026-06-21", "08:00", "America/Los_Angeles"),
                    ("Release check", "2026-06-25", "17:00", "UTC"),
                    ("Retrospective", "2026-06-30", "12:00", "Asia/Kolkata"),
                ]
            ],
            [
                'Create an event titled "{title}" on {date} at {time} in {timezone}.',
                'Add "{title}" to my calendar for {date}, {time} {timezone}.',
                'Schedule a calendar event: {title}, {date}, {time}, {timezone}.',
                'Put "{title}" on the calendar at {time} on {date} ({timezone}).',
                'I need a calendar event for "{title}" on {date} at {time} in {timezone}.',
            ],
        ),
        (
            "send_email",
            [
                {"recipient": recipient, "subject": subject, "body": body}
                for recipient, subject, body in [
                    ("alex@example.com", "Meeting notes", "Here are the meeting notes."),
                    ("team@example.com", "Release ready", "The release is ready for review."),
                    ("sam@example.com", "Lunch", "Can we meet for lunch tomorrow?"),
                    ("ops@example.com", "Incident update", "The service has recovered."),
                    ("mentor@example.com", "Progress", "I completed the training run."),
                    ("billing@example.com", "Invoice question", "Please clarify invoice 42."),
                    ("hr@example.com", "Leave request", "I would like Friday off."),
                    ("dev@example.com", "Code review", "Please review my latest changes."),
                    ("support@example.com", "Login issue", "I cannot access my account."),
                    ("maria@example.com", "Thank you", "Thanks for your help today."),
                ]
            ],
            [
                'Send an email to {recipient} with subject "{subject}" and body "{body}"',
                'Email {recipient}: subject "{subject}", message "{body}"',
                'Compose mail for {recipient} titled "{subject}" saying "{body}"',
                'Send "{body}" to {recipient} under the subject "{subject}".',
                'I need to email {recipient} about "{subject}" with message "{body}"',
            ],
        ),
        (
            "restaurant_search",
            [
                {"city": city, "cuisine": cuisine, "max_price": price}
                for city, cuisine, price in [
                    ("Chicago", "Italian", 40), ("Austin", "Mexican", 25),
                    ("Seattle", "Japanese", 60), ("New York", "Indian", 50),
                    ("Boston", "Thai", 35), ("Denver", "Korean", 45),
                    ("Portland", "Vegan", 30), ("Miami", "Seafood", 70),
                    ("San Diego", "Mediterranean", 55), ("Atlanta", "Ethiopian", 40),
                ]
            ],
            [
                "Find {cuisine} restaurants in {city} under {max_price} dollars.",
                "Search for {cuisine} food in {city} with a {max_price} dollar limit.",
                "I want a {cuisine} restaurant in {city}; max price {max_price}.",
                "Look up {city} restaurants serving {cuisine} below {max_price} dollars.",
                "Show {cuisine} dining options in {city} costing at most {max_price}.",
            ],
        ),
        (
            "book_flight",
            [
                {"origin": origin, "destination": destination, "date": date, "passengers": people, "cabin": cabin}
                for origin, destination, date, people, cabin in [
                    ("Chicago", "Seattle", "2026-07-01", 1, "economy"),
                    ("Boston", "Austin", "2026-07-03", 2, "economy"),
                    ("London", "Paris", "2026-07-08", 1, "business"),
                    ("Delhi", "Mumbai", "2026-07-12", 3, "economy"),
                    ("Tokyo", "Seoul", "2026-07-15", 2, "premium economy"),
                    ("San Francisco", "New York", "2026-07-18", 1, "business"),
                    ("Dubai", "Singapore", "2026-07-22", 2, "economy"),
                    ("Toronto", "Vancouver", "2026-07-25", 4, "economy"),
                    ("Sydney", "Melbourne", "2026-07-27", 1, "economy"),
                    ("Berlin", "Rome", "2026-07-30", 2, "business"),
                ]
            ],
            [
                "Find a {cabin} flight from {origin} to {destination} on {date} for {passengers} passengers.",
                "Book travel: {origin} to {destination}, {date}, {passengers} travelers, {cabin}.",
                "Search flights from {origin} to {destination} for {passengers} on {date} in {cabin}.",
                "I need {passengers} {cabin} tickets from {origin} to {destination} on {date}.",
                "Look for a flight on {date} from {origin} to {destination}, cabin {cabin}, passengers {passengers}.",
            ],
        ),
        (
            "hotel_search",
            [
                {"city": city, "check_in": start, "check_out": end, "guests": guests, "max_price": price}
                for city, start, end, guests, price in [
                    ("Chicago", "2026-07-01", "2026-07-04", 2, 180),
                    ("London", "2026-07-05", "2026-07-08", 1, 220),
                    ("Tokyo", "2026-07-10", "2026-07-14", 2, 250),
                    ("Paris", "2026-07-15", "2026-07-18", 2, 300),
                    ("Austin", "2026-07-20", "2026-07-22", 3, 160),
                    ("Mumbai", "2026-07-23", "2026-07-26", 2, 120),
                    ("Seattle", "2026-07-27", "2026-07-30", 1, 190),
                    ("Rome", "2026-08-01", "2026-08-05", 2, 210),
                    ("Dubai", "2026-08-06", "2026-08-09", 4, 350),
                    ("Toronto", "2026-08-11", "2026-08-13", 2, 170),
                ]
            ],
            [
                "Find a hotel in {city} from {check_in} to {check_out} for {guests} guests under {max_price}.",
                "Search lodging in {city}: check in {check_in}, check out {check_out}, {guests} guests, max {max_price}.",
                "I need a {city} hotel for {guests} from {check_in} through {check_out}, below {max_price}.",
                "Look for rooms in {city} under {max_price} for {guests}, {check_in} to {check_out}.",
                "Book a stay search in {city} between {check_in} and {check_out} for {guests} people, budget {max_price}.",
            ],
        ),
        (
            "route_planner",
            [
                {"origin": origin, "destination": destination, "mode": mode, "avoid_tolls": avoid}
                for origin, destination, mode, avoid in [
                    ("Home", "Airport", "driving", True), ("Office", "Station", "walking", False),
                    ("Chicago", "Milwaukee", "driving", False), ("Hotel", "Museum", "transit", False),
                    ("Boston", "Cambridge", "cycling", True), ("Seattle", "Tacoma", "driving", True),
                    ("Campus", "Library", "walking", False), ("Paris", "Versailles", "transit", False),
                    ("Austin", "Dallas", "driving", True), ("Tokyo", "Yokohama", "transit", False),
                ]
            ],
            [
                "Plan a {mode} route from {origin} to {destination} with avoid_tolls={avoid_tolls}.",
                "Get directions from {origin} to {destination} by {mode}; avoid tolls: {avoid_tolls}.",
                "Route me from {origin} to {destination} using {mode}, toll avoidance {avoid_tolls}.",
                "Find a {mode} path between {origin} and {destination}; avoid_tolls is {avoid_tolls}.",
                "I need directions: {origin} to {destination}, mode {mode}, avoid tolls {avoid_tolls}.",
            ],
        ),
        (
            "product_search",
            [
                {"query": query, "max_price": price, "min_rating": rating}
                for query, price, rating in [
                    ("wireless keyboard", 80, 4.2), ("noise cancelling headphones", 250, 4.5),
                    ("USB-C hub", 60, 4.0), ("running shoes", 120, 4.3),
                    ("desk lamp", 45, 4.1), ("mechanical keyboard", 150, 4.4),
                    ("webcam", 100, 4.0), ("portable monitor", 280, 4.2),
                    ("coffee grinder", 90, 4.3), ("backpack", 75, 4.1),
                ]
            ],
            [
                "Search for {query} under {max_price} with rating at least {min_rating}.",
                "Find {query}; max price {max_price}, minimum rating {min_rating}.",
                "Shop for {query} below {max_price} rated {min_rating} or higher.",
                "I need {query} with a {max_price} budget and {min_rating} minimum rating.",
                "Look up products for {query}, max {max_price}, at least {min_rating} stars.",
            ],
        ),
        (
            "set_reminder",
            [
                {"message": message, "date": date, "time": time, "timezone": zone}
                for message, date, time, zone in [
                    ("Submit report", "2026-06-01", "09:00", "UTC"),
                    ("Call dentist", "2026-06-02", "10:30", "America/New_York"),
                    ("Pay rent", "2026-06-03", "08:00", "America/Chicago"),
                    ("Start training", "2026-06-04", "18:00", "Asia/Kolkata"),
                    ("Pick up package", "2026-06-05", "16:15", "UTC"),
                    ("Send invoice", "2026-06-06", "11:00", "Europe/London"),
                    ("Water plants", "2026-06-07", "07:00", "America/Los_Angeles"),
                    ("Join demo", "2026-06-08", "14:30", "UTC"),
                    ("Renew subscription", "2026-06-09", "12:00", "UTC"),
                    ("Check experiment", "2026-06-10", "17:00", "Asia/Kolkata"),
                ]
            ],
            [
                'Remind me to "{message}" on {date} at {time} in {timezone}.',
                'Set a reminder: "{message}", {date}, {time}, {timezone}.',
                'Create a reminder for "{message}" at {time} on {date} ({timezone}).',
                'I need a reminder to "{message}" on {date}, {time} {timezone}.',
                'Schedule reminder "{message}" for {date} at {time}, timezone {timezone}.',
            ],
        ),
        (
            "track_package",
            [
                {"carrier": carrier, "tracking_number": number}
                for carrier, number in [
                    ("UPS", "1Z999AA10123456784"), ("FedEx", "612999900123"),
                    ("USPS", "9400111899223856928499"), ("DHL", "1234567890"),
                    ("UPS", "1Z12345E0205271688"), ("FedEx", "771234567890"),
                    ("USPS", "9405509205568123456789"), ("DHL", "JD0146000038281234"),
                    ("OnTrac", "C12345678901234"), ("Canada Post", "CX123456789CA"),
                ]
            ],
            [
                "Track package {tracking_number} with {carrier}.",
                "Where is my {carrier} shipment {tracking_number}?",
                "Look up tracking number {tracking_number} on {carrier}.",
                "Check delivery status for {carrier} parcel {tracking_number}.",
                "I need tracking details from {carrier} for {tracking_number}.",
            ],
        ),
        (
            "stock_quote",
            [{"ticker": ticker} for ticker in ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA", "META", "AMD", "NFLX", "INTC"]],
            [
                "Get the stock quote for {ticker}.",
                "Look up the current price of {ticker}.",
                "Show me a quote for ticker {ticker}.",
                "What is {ticker} trading at?",
                "Fetch stock information for {ticker}.",
            ],
        ),
        (
            "file_search",
            [
                {"query": query, "directory": directory, "file_type": file_type}
                for query, directory, file_type in [
                    ("quarterly report", "/documents", "pdf"), ("training config", "/projects", "yaml"),
                    ("meeting notes", "/notes", "md"), ("dataset", "/data", "jsonl"),
                    ("invoice", "/finance", "pdf"), ("presentation", "/slides", "pptx"),
                    ("experiment log", "/outputs", "csv"), ("resume", "/documents", "docx"),
                    ("adapter config", "/models", "json"), ("unit tests", "/projects", "py"),
                ]
            ],
            [
                'Search for "{query}" in {directory} as {file_type} files.',
                'Find {file_type} files about "{query}" under {directory}.',
                'Look in {directory} for "{query}" with type {file_type}.',
                'I need a {file_type} file matching "{query}" from {directory}.',
                'Run file search in {directory}: query "{query}", type {file_type}.',
            ],
        ),
        (
            "schedule_meeting",
            [
                {"title": title, "date": date, "time": time, "timezone": zone, "attendees": attendees}
                for title, date, time, zone, attendees in [
                    ("Model review", "2026-06-12", "14:00", "UTC", ["alice@example.com", "bob@example.com"]),
                    ("Sprint planning", "2026-06-13", "09:00", "America/New_York", ["team@example.com"]),
                    ("Research sync", "2026-06-14", "16:30", "UTC", ["sam@example.com", "lee@example.com"]),
                    ("Launch decision", "2026-06-15", "11:00", "Europe/London", ["ops@example.com", "pm@example.com"]),
                    ("Hiring panel", "2026-06-16", "10:00", "Asia/Kolkata", ["hr@example.com", "lead@example.com"]),
                    ("Adapter results", "2026-06-17", "15:00", "UTC", ["mentor@example.com"]),
                    ("Architecture review", "2026-06-18", "13:30", "America/Chicago", ["dev@example.com", "arch@example.com"]),
                    ("Customer feedback", "2026-06-19", "08:30", "America/Los_Angeles", ["sales@example.com", "support@example.com"]),
                    ("GRPO checkpoint", "2026-06-20", "17:00", "UTC", ["ml@example.com", "eval@example.com"]),
                    ("Final presentation", "2026-06-21", "12:00", "Asia/Kolkata", ["team@example.com", "manager@example.com"]),
                ]
            ],
            [
                'Schedule "{title}" on {date} at {time} in {timezone} with {attendees}.',
                'Create a meeting named "{title}" for {date}, {time} {timezone}; invite {attendees}.',
                'Set up "{title}" with {attendees} on {date} at {time} ({timezone}).',
                'Book a meeting: {title}, attendees {attendees}, {date} {time}, {timezone}.',
                'I need "{title}" scheduled for {date} at {time} in {timezone} with {attendees}.',
            ],
        ),
    ]


def main() -> None:
    records = write_dataset()
    print(f"wrote {len(records)} records to {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
