"""Tool-call schemas used by evaluation, SFT, and GRPO."""

from __future__ import annotations


TOOL_SCHEMAS = {
    "calculator": {
        "args": {"expression": "string"},
        "description": "Evaluate arithmetic.",
    },
    "google_search": {
        "args": {"query": "string"},
        "description": "Record a web search query.",
    },
    "unit_converter": {
        "args": {"value": "number", "from_unit": "string", "to_unit": "string"},
        "description": "Convert a measurement.",
    },
    "text_stats": {
        "args": {"text": "string"},
        "description": "Count words, characters, or sentences.",
    },
    "string_formatter": {
        "args": {"text": "string", "operation": "string"},
        "description": "Apply text formatting.",
    },
    "weather_lookup": {
        "args": {"city": "string", "unit": "string"},
        "description": "Look up weather for a city.",
    },
    "currency_converter": {
        "args": {"amount": "number", "from_currency": "string", "to_currency": "string"},
        "description": "Convert a currency amount.",
    },
    "translate_text": {
        "args": {"text": "string", "source_language": "string", "target_language": "string"},
        "description": "Translate text between languages.",
    },
    "create_calendar_event": {
        "args": {"title": "string", "date": "string", "time": "string", "timezone": "string"},
        "description": "Create a calendar event.",
    },
    "send_email": {
        "args": {"recipient": "string", "subject": "string", "body": "string"},
        "description": "Compose and send an email.",
    },
    "restaurant_search": {
        "args": {"city": "string", "cuisine": "string", "max_price": "number"},
        "description": "Find restaurants matching preferences.",
    },
    "book_flight": {
        "args": {
            "origin": "string",
            "destination": "string",
            "date": "string",
            "passengers": "number",
            "cabin": "string",
        },
        "description": "Search for a flight.",
    },
    "hotel_search": {
        "args": {
            "city": "string",
            "check_in": "string",
            "check_out": "string",
            "guests": "number",
            "max_price": "number",
        },
        "description": "Find a hotel stay.",
    },
    "route_planner": {
        "args": {"origin": "string", "destination": "string", "mode": "string", "avoid_tolls": "boolean"},
        "description": "Plan a route.",
    },
    "product_search": {
        "args": {"query": "string", "max_price": "number", "min_rating": "number"},
        "description": "Search for products.",
    },
    "set_reminder": {
        "args": {"message": "string", "date": "string", "time": "string", "timezone": "string"},
        "description": "Set a reminder.",
    },
    "track_package": {
        "args": {"carrier": "string", "tracking_number": "string"},
        "description": "Track a delivery.",
    },
    "stock_quote": {
        "args": {"ticker": "string"},
        "description": "Get a stock quote.",
    },
    "file_search": {
        "args": {"query": "string", "directory": "string", "file_type": "string"},
        "description": "Search local files.",
    },
    "schedule_meeting": {
        "args": {
            "title": "string",
            "date": "string",
            "time": "string",
            "timezone": "string",
            "attendees": "list[string]",
        },
        "description": "Schedule a meeting with attendees.",
    },
}

TOOL_SCHEMAS.update(
    {
        "reserve_table": {
            "args": {
                "restaurant": "string",
                "city": "string",
                "date": "string",
                "time": "string",
                "party_size": "number",
                "seating": "string",
            },
            "description": "Reserve a restaurant table.",
        },
        "order_grocery": {
            "args": {
                "store": "string",
                "items": "list[string]",
                "delivery_date": "string",
                "delivery_window": "string",
                "substitutions_allowed": "boolean",
            },
            "description": "Order groceries for delivery.",
        },
        "create_support_ticket": {
            "args": {
                "product": "string",
                "issue": "string",
                "severity": "string",
                "account_id": "string",
                "contact_email": "string",
            },
            "description": "Open a product support ticket.",
        },
        "schedule_delivery": {
            "args": {
                "pickup_address": "string",
                "dropoff_address": "string",
                "date": "string",
                "time_window": "string",
                "package_size": "string",
                "signature_required": "boolean",
            },
            "description": "Schedule a package delivery.",
        },
        "compare_products": {
            "args": {
                "products": "list[string]",
                "category": "string",
                "currency": "string",
                "max_price": "number",
                "features": "list[string]",
            },
            "description": "Compare products by requirements.",
        },
        "job_search": {
            "args": {
                "role": "string",
                "location": "string",
                "remote": "boolean",
                "min_salary": "number",
                "experience_level": "string",
            },
            "description": "Search for job openings.",
        },
        "rental_car_search": {
            "args": {
                "pickup_city": "string",
                "dropoff_city": "string",
                "pickup_date": "string",
                "dropoff_date": "string",
                "vehicle_type": "string",
                "driver_age": "number",
            },
            "description": "Find a rental car.",
        },
        "movie_ticket_booking": {
            "args": {
                "movie": "string",
                "cinema": "string",
                "date": "string",
                "showtime": "string",
                "tickets": "number",
                "format": "string",
            },
            "description": "Book movie tickets.",
        },
        "train_ticket_search": {
            "args": {
                "origin": "string",
                "destination": "string",
                "date": "string",
                "departure_after": "string",
                "passengers": "number",
                "class": "string",
            },
            "description": "Search train journeys.",
        },
        "event_ticket_search": {
            "args": {
                "event": "string",
                "city": "string",
                "date": "string",
                "tickets": "number",
                "max_price": "number",
            },
            "description": "Find event tickets.",
        },
        "prescription_refill": {
            "args": {
                "medication": "string",
                "dosage": "string",
                "pharmacy": "string",
                "patient_id": "string",
                "pickup_date": "string",
            },
            "description": "Request a prescription refill.",
        },
        "workout_plan": {
            "args": {
                "goal": "string",
                "days_per_week": "number",
                "duration_minutes": "number",
                "equipment": "list[string]",
                "fitness_level": "string",
            },
            "description": "Create a workout plan.",
        },
        "meal_plan": {
            "args": {
                "diet": "string",
                "calories": "number",
                "days": "number",
                "allergies": "list[string]",
                "meals_per_day": "number",
            },
            "description": "Create a meal plan.",
        },
        "project_task_create": {
            "args": {
                "project": "string",
                "title": "string",
                "assignee": "string",
                "due_date": "string",
                "priority": "string",
                "labels": "list[string]",
            },
            "description": "Create a project task.",
        },
        "database_query": {
            "args": {
                "database": "string",
                "table": "string",
                "fields": "list[string]",
                "filter": "string",
                "limit": "number",
                "sort_by": "string",
            },
            "description": "Query structured data.",
        },
        "cloud_deploy": {
            "args": {
                "service": "string",
                "environment": "string",
                "region": "string",
                "version": "string",
                "replicas": "number",
                "rollback_on_failure": "boolean",
            },
            "description": "Deploy a cloud service.",
        },
        "log_search": {
            "args": {
                "service": "string",
                "environment": "string",
                "query": "string",
                "start_time": "string",
                "end_time": "string",
                "level": "string",
            },
            "description": "Search application logs.",
        },
        "api_request": {
            "args": {
                "method": "string",
                "endpoint": "string",
                "headers": "list[string]",
                "body": "string",
                "timeout_seconds": "number",
            },
            "description": "Prepare an API request.",
        },
        "image_generation": {
            "args": {
                "prompt": "string",
                "style": "string",
                "width": "number",
                "height": "number",
                "background": "string",
            },
            "description": "Generate an image.",
        },
        "document_summary": {
            "args": {
                "document": "string",
                "audience": "string",
                "max_words": "number",
                "format": "string",
                "include_actions": "boolean",
            },
            "description": "Summarize a document.",
        },
        "invoice_create": {
            "args": {
                "customer": "string",
                "items": "list[string]",
                "currency": "string",
                "due_date": "string",
                "tax_percent": "number",
                "payment_terms": "string",
            },
            "description": "Create an invoice.",
        },
        "expense_report": {
            "args": {
                "employee": "string",
                "category": "string",
                "amount": "number",
                "currency": "string",
                "date": "string",
                "receipt_attached": "boolean",
            },
            "description": "Submit an expense report.",
        },
        "classroom_assignment": {
            "args": {
                "course": "string",
                "title": "string",
                "due_date": "string",
                "points": "number",
                "submission_type": "string",
            },
            "description": "Create a classroom assignment.",
        },
        "survey_create": {
            "args": {
                "title": "string",
                "audience": "string",
                "questions": "list[string]",
                "close_date": "string",
                "anonymous": "boolean",
            },
            "description": "Create a survey.",
        },
        "notification_send": {
            "args": {
                "channel": "string",
                "recipients": "list[string]",
                "title": "string",
                "message": "string",
                "urgency": "string",
                "send_at": "string",
            },
            "description": "Send a notification.",
        },
        "contact_create": {
            "args": {
                "name": "string",
                "email": "string",
                "phone": "string",
                "company": "string",
                "tags": "list[string]",
            },
            "description": "Create a contact.",
        },
        "playlist_create": {
            "args": {
                "name": "string",
                "genre": "string",
                "mood": "string",
                "tracks": "number",
                "explicit_allowed": "boolean",
            },
            "description": "Create a music playlist.",
        },
        "insurance_quote": {
            "args": {
                "insurance_type": "string",
                "state": "string",
                "coverage_amount": "number",
                "deductible": "number",
                "applicant_age": "number",
            },
            "description": "Request an insurance quote.",
        },
        "appointment_booking": {
            "args": {
                "provider": "string",
                "service": "string",
                "date": "string",
                "time": "string",
                "location": "string",
                "patient_name": "string",
            },
            "description": "Book an appointment.",
        },
        "conference_room_booking": {
            "args": {
                "building": "string",
                "room": "string",
                "date": "string",
                "start_time": "string",
                "duration_minutes": "number",
                "attendees": "number",
                "video_required": "boolean",
            },
            "description": "Reserve a conference room.",
        },
    }
)

TOOL_NAMES = list(TOOL_SCHEMAS)
TOOL_BATCHES = [TOOL_NAMES[index : index + 10] for index in range(0, len(TOOL_NAMES), 10)]


def _tool_lines(tool_names: list[str]) -> str:
    lines = []
    for name in tool_names:
        schema = TOOL_SCHEMAS[name]
        args = ", ".join(f"{key}: {value}" for key, value in schema["args"].items())
        lines.append(f"- {name}({args}): {schema['description']}")
    return "\n".join(lines)


def build_system_prompt(tool_names: list[str]) -> str:
    """Build a compact prompt for only the tools relevant to a training batch."""

    return f"""
You are a tool-calling model.

Available tools:
{_tool_lines(tool_names)}

If a tool is needed, output ONLY valid JSON in this format:
{{"tool": "tool_name", "args": {{"argument_name": "argument_value"}}}}
Do not explain. Do not wrap in markdown.
""".strip()


def system_prompt_for_tool(tool_name: str) -> str:
    """Return the ten-tool prompt batch that includes a tool."""

    for tool_batch in TOOL_BATCHES:
        if tool_name in tool_batch:
            return build_system_prompt(tool_batch)
    raise KeyError(tool_name)
