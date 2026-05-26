"""Build the direct-prompt, 50-tool GRPO dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from toolcall_rl.data.build_dataset import PROJECT_ROOT, _tool_examples
from toolcall_rl.evaluation.schemas import system_prompt_for_tool


DEFAULT_GRPO_SOURCE_PATH = PROJECT_ROOT / "data" / "grpo" / "tool_call_source.jsonl"

_DIRECT_TEMPLATES = [
    "{final}",
    "Please handle this request: {final}",
    "Complete this task: {final}",
    "Use the appropriate tool for this: {final}",
    "I need this done: {final}",
    "Please process the following: {final}",
    "Take care of this request: {final}",
    "Perform this action: {final}",
    "Can you complete this: {final}",
    "Act on this request: {final}",
]


def build_grpo_dataset() -> list[dict[str, Any]]:
    """Create 50 direct-request GRPO examples for each of the 50 tools."""

    tool_examples = _tool_examples() + _additional_tool_examples()
    records = []
    for tool, values, prompt_templates in tool_examples:
        if len(values) == 10:
            target_values = values[5:]
        else:
            assert len(values) == 5
            target_values = values

        counter = 0
        for value_index, expected_args in enumerate(target_values):
            for prompt_index in range(10):
                target_template = prompt_templates[prompt_index % len(prompt_templates)]
                direct_prompt = _format_prompt(target_template, expected_args)
                prompt = _DIRECT_TEMPLATES[prompt_index].format(final=direct_prompt)
                counter += 1
                records.append(
                    {
                        "id": f"grpo_{tool}_{counter:04d}",
                        "prompt": prompt,
                        "system_prompt": system_prompt_for_tool(tool),
                        "expected_tool": tool,
                        "expected_args": expected_args,
                        "assistant_response": {"tool": tool, "args": expected_args},
                    }
                )

    return records


def write_grpo_dataset(path: Path = DEFAULT_GRPO_SOURCE_PATH) -> list[dict[str, Any]]:
    records = build_grpo_dataset()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=True) + "\n")
    return records


def _format_prompt(template: str, args: dict[str, Any]) -> str:
    prompt_args = {key: _prompt_value(value) for key, value in args.items()}
    return template.format(**prompt_args)


def _prompt_value(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(value)
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _rows(**columns: tuple[Any, ...]) -> list[dict[str, Any]]:
    """Transpose five argument columns into five tool-call argument records."""

    assert columns and all(len(values) == 5 for values in columns.values())
    return [
        {key: values[index] for key, values in columns.items()}
        for index in range(5)
    ]


def _additional_tool_examples() -> list[tuple[str, list[dict[str, Any]], list[str]]]:
    """Examples for the thirty tools introduced only during GRPO."""

    return [
        (
            "reserve_table",
            _rows(
                restaurant=("Cedar Grill", "Nori House", "Olive Room", "Spice Route", "Harbor Table"),
                city=("Denver", "Seattle", "Boston", "Austin", "Miami"),
                date=("2026-10-02", "2026-10-03", "2026-10-04", "2026-10-05", "2026-10-06"),
                time=("19:00", "20:15", "18:30", "19:45", "17:30"),
                party_size=(4, 2, 6, 3, 5),
                seating=("patio", "counter", "indoor", "booth", "waterfront"),
            ),
            ['Reserve {restaurant} in {city} on {date} at {time} for {party_size} people, seating {seating}.'],
        ),
        (
            "order_grocery",
            _rows(
                store=("Fresh Mart", "Green Basket", "City Foods", "Harvest Shop", "Daily Market"),
                items=(["milk", "bread"], ["tofu", "rice"], ["apples", "oats"], ["pasta", "tomatoes"], ["tea", "lemons"]),
                delivery_date=("2026-10-07", "2026-10-08", "2026-10-09", "2026-10-10", "2026-10-11"),
                delivery_window=("08:00-10:00", "18:00-20:00", "12:00-14:00", "09:00-11:00", "16:00-18:00"),
                substitutions_allowed=(False, True, False, True, False),
            ),
            ["Order {items} from {store} for {delivery_date} in {delivery_window}; substitutions allowed: {substitutions_allowed}."],
        ),
        (
            "create_support_ticket",
            _rows(
                product=("Cloud Sync", "Mobile App", "Billing Portal", "API Gateway", "Desktop Client"),
                issue=("files fail to upload", "login loops", "invoice missing", "timeouts on POST", "crashes on launch"),
                severity=("high", "medium", "low", "critical", "high"),
                account_id=("AC-104", "AC-205", "AC-306", "AC-407", "AC-508"),
                contact_email=("lee@example.com", "pat@example.com", "kim@example.com", "dev@example.com", "ria@example.com"),
            ),
            ['Open a {severity} ticket for {product}: "{issue}", account {account_id}, contact {contact_email}.'],
        ),
        (
            "schedule_delivery",
            _rows(
                pickup_address=("12 Oak St", "8 Pine Ave", "44 River Rd", "90 Main St", "5 Lake Way"),
                dropoff_address=("77 Elm St", "20 King Rd", "11 Hill Ave", "31 Park Dr", "62 Bay St"),
                date=("2026-10-12", "2026-10-13", "2026-10-14", "2026-10-15", "2026-10-16"),
                time_window=("09:00-12:00", "12:00-15:00", "16:00-19:00", "08:00-10:00", "14:00-17:00"),
                package_size=("small", "large", "medium", "medium", "large"),
                signature_required=(True, False, True, False, True),
            ),
            ["Schedule a {package_size} delivery from {pickup_address} to {dropoff_address} on {date}, {time_window}, signature {signature_required}."],
        ),
        (
            "compare_products",
            _rows(
                products=(["Alpha Pad", "Nova Pad"], ["Quiet Pro", "Wave ANC"], ["DeskOne", "LiftUp"], ["Cam HD", "View Pro"], ["Trail X", "Urban Pack"]),
                category=("tablet", "headphones", "desk", "webcam", "backpack"),
                currency=("USD", "USD", "EUR", "USD", "CAD"),
                max_price=(700, 300, 450, 140, 180),
                features=(["stylus", "wifi"], ["noise cancellation", "usb-c"], ["standing", "memory"], ["4k", "microphone"], ["waterproof", "laptop sleeve"]),
            ),
            ["Compare {products} as {category} in {currency}, under {max_price}, requiring {features}."],
        ),
        (
            "job_search",
            _rows(
                role=("ML Engineer", "Data Analyst", "Product Designer", "Backend Engineer", "Research Scientist"),
                location=("Chicago", "Toronto", "London", "Berlin", "Boston"),
                remote=(True, False, True, False, True),
                min_salary=(130000, 85000, 95000, 110000, 150000),
                experience_level=("senior", "mid", "senior", "mid", "lead"),
            ),
            ["Find {experience_level} {role} jobs in {location}, remote={remote}, minimum salary {min_salary}."],
        ),
        (
            "rental_car_search",
            _rows(
                pickup_city=("Denver", "Lisbon", "Rome", "Seattle", "Sydney"),
                dropoff_city=("Boulder", "Porto", "Florence", "Portland", "Melbourne"),
                pickup_date=("2026-11-01", "2026-11-02", "2026-11-03", "2026-11-04", "2026-11-05"),
                dropoff_date=("2026-11-04", "2026-11-06", "2026-11-08", "2026-11-07", "2026-11-10"),
                vehicle_type=("SUV", "compact", "wagon", "electric", "van"),
                driver_age=(34, 29, 41, 26, 38),
            ),
            ["Find a {vehicle_type} rental from {pickup_city} on {pickup_date} to {dropoff_city} on {dropoff_date} for a {driver_age}-year-old driver."],
        ),
        (
            "movie_ticket_booking",
            _rows(
                movie=("Solar Echo", "Night Circuit", "Paper Moon", "Orbit", "Signal Lost"),
                cinema=("Grand Cinema", "Riverside IMAX", "Metro Screen", "Arc Theater", "Skyline Movies"),
                date=("2026-10-20", "2026-10-21", "2026-10-22", "2026-10-23", "2026-10-24"),
                showtime=("19:10", "21:00", "18:20", "20:30", "16:45"),
                tickets=(2, 3, 1, 4, 2),
                format=("standard", "IMAX", "standard", "3D", "Dolby"),
            ),
            ['Book {tickets} tickets for "{movie}" at {cinema} on {date}, {showtime}, format {format}.'],
        ),
        (
            "train_ticket_search",
            _rows(
                origin=("London", "Paris", "Berlin", "Madrid", "Milan"),
                destination=("Edinburgh", "Lyon", "Hamburg", "Valencia", "Venice"),
                date=("2026-11-11", "2026-11-12", "2026-11-13", "2026-11-14", "2026-11-15"),
                departure_after=("08:00", "10:30", "06:45", "12:00", "15:00"),
                passengers=(2, 1, 3, 2, 4),
                **{"class": ("first", "standard", "standard", "business", "standard")},
            ),
            ["Search {class} train tickets from {origin} to {destination} on {date} after {departure_after} for {passengers} passengers."],
        ),
        (
            "event_ticket_search",
            _rows(
                event=("Jazz Festival", "Robotics Expo", "City Marathon", "Food Fair", "Design Conference"),
                city=("New Orleans", "Tokyo", "Boston", "Austin", "Copenhagen"),
                date=("2026-12-01", "2026-12-02", "2026-12-03", "2026-12-04", "2026-12-05"),
                tickets=(2, 1, 3, 4, 2),
                max_price=(160, 90, 60, 45, 300),
            ),
            ['Find {tickets} tickets for "{event}" in {city} on {date}, maximum price {max_price}.'],
        ),
        (
            "prescription_refill",
            _rows(
                medication=("Atorvastatin", "Metformin", "Lisinopril", "Albuterol", "Levothyroxine"),
                dosage=("20mg", "500mg", "10mg", "90mcg", "50mcg"),
                pharmacy=("North Pharmacy", "City Rx", "Care Drugstore", "Lake Pharmacy", "Central Rx"),
                patient_id=("PT-101", "PT-202", "PT-303", "PT-404", "PT-505"),
                pickup_date=("2026-10-18", "2026-10-19", "2026-10-20", "2026-10-21", "2026-10-22"),
            ),
            ["Refill {medication} {dosage} at {pharmacy} for patient {patient_id}, pickup {pickup_date}."],
        ),
        (
            "workout_plan",
            _rows(
                goal=("strength", "mobility", "endurance", "weight loss", "balance"),
                days_per_week=(4, 3, 5, 4, 2),
                duration_minutes=(45, 30, 60, 40, 25),
                equipment=(["barbell", "bench"], ["mat"], ["treadmill", "bike"], ["dumbbells"], ["bands", "mat"]),
                fitness_level=("intermediate", "beginner", "advanced", "intermediate", "beginner"),
            ),
            ["Create a {fitness_level} {goal} workout, {days_per_week} days weekly, {duration_minutes} minutes, equipment {equipment}."],
        ),
        (
            "meal_plan",
            _rows(
                diet=("vegetarian", "mediterranean", "vegan", "high protein", "gluten free"),
                calories=(1900, 2200, 1800, 2500, 2000),
                days=(7, 5, 7, 3, 6),
                allergies=(["peanuts"], ["shellfish"], ["soy"], ["none"], ["dairy"]),
                meals_per_day=(3, 3, 4, 4, 3),
            ),
            ["Plan {days} days of {diet} meals at {calories} calories with allergies {allergies}, {meals_per_day} meals daily."],
        ),
        (
            "project_task_create",
            _rows(
                project=("Agent Eval", "Mobile Redesign", "Data Pipeline", "Launch", "Security Audit"),
                title=("Add hard prompts", "Update checkout", "Repair ingest", "Write announcement", "Rotate secrets"),
                assignee=("ml@example.com", "ux@example.com", "data@example.com", "pm@example.com", "ops@example.com"),
                due_date=("2026-10-25", "2026-10-26", "2026-10-27", "2026-10-28", "2026-10-29"),
                priority=("high", "medium", "critical", "high", "critical"),
                labels=(["training", "eval"], ["mobile"], ["bug", "etl"], ["release"], ["security", "urgent"]),
            ),
            ['Create {priority} task "{title}" in {project}, assigned to {assignee}, due {due_date}, labels {labels}.'],
        ),
        (
            "database_query",
            _rows(
                database=("analytics", "billing", "support", "warehouse", "product"),
                table=("runs", "invoices", "tickets", "inventory", "events"),
                fields=(["id", "score"], ["customer", "total"], ["severity", "status"], ["sku", "stock"], ["user_id", "action"]),
                filter=("score < 4", "paid = false", "severity = critical", "stock < 10", "action = purchase"),
                limit=(25, 50, 20, 100, 40),
                sort_by=("score", "total", "created_at", "stock", "timestamp"),
            ),
            ["Query {database}.{table} for fields {fields} where {filter}, limit {limit}, sort by {sort_by}."],
        ),
        (
            "cloud_deploy",
            _rows(
                service=("tool-api", "web-app", "worker", "search", "billing"),
                environment=("staging", "production", "staging", "production", "staging"),
                region=("us-east-1", "eu-west-1", "us-west-2", "ap-south-1", "eu-central-1"),
                version=("v2.4.1", "v7.0.0", "v1.9.3", "v4.2.0", "v3.1.8"),
                replicas=(2, 6, 3, 4, 2),
                rollback_on_failure=(True, True, False, True, False),
            ),
            ["Deploy {service} version {version} to {environment} in {region} with {replicas} replicas, rollback={rollback_on_failure}."],
        ),
        (
            "log_search",
            _rows(
                service=("checkout", "auth", "inference", "email", "payments"),
                environment=("production", "staging", "production", "production", "staging"),
                query=("timeout", "invalid token", "out of memory", "delivery failed", "declined"),
                start_time=("2026-10-10T08:00Z", "2026-10-11T09:00Z", "2026-10-12T10:00Z", "2026-10-13T11:00Z", "2026-10-14T12:00Z"),
                end_time=("2026-10-10T09:00Z", "2026-10-11T10:00Z", "2026-10-12T11:00Z", "2026-10-13T12:00Z", "2026-10-14T13:00Z"),
                level=("error", "warning", "error", "error", "warning"),
            ),
            ['Search {environment} {service} logs for "{query}" from {start_time} to {end_time} at level {level}.'],
        ),
        (
            "api_request",
            _rows(
                method=("POST", "GET", "PATCH", "DELETE", "PUT"),
                endpoint=("/v1/jobs", "/v1/results", "/v1/users/42", "/v1/cache/7", "/v1/settings"),
                headers=(["Authorization", "Content-Type"], ["Authorization"], ["If-Match"], ["Authorization"], ["Content-Type"]),
                body=('{"job":"eval"}', "", '{"active":true}', "", '{"theme":"dark"}'),
                timeout_seconds=(30, 10, 20, 15, 25),
            ),
            ["Prepare {method} request to {endpoint} with headers {headers}, body '{body}', timeout {timeout_seconds} seconds."],
        ),
        (
            "image_generation",
            _rows(
                prompt=("a red bicycle in rain", "minimal desk setup", "city map at night", "botanical poster", "robot workshop"),
                style=("photorealistic", "editorial", "isometric", "watercolor", "cinematic"),
                width=(1024, 1200, 1024, 800, 1280),
                height=(768, 800, 1024, 1200, 720),
                background=("street", "white", "dark", "paper", "factory"),
            ),
            ['Generate "{prompt}" in {style} style, {width}x{height}, background {background}.'],
        ),
        (
            "document_summary",
            _rows(
                document=("Q3 report.pdf", "incident-104.md", "research-paper.pdf", "contract.docx", "meeting-notes.md"),
                audience=("executives", "engineers", "students", "legal team", "product team"),
                max_words=(150, 300, 200, 250, 120),
                format=("bullets", "timeline", "abstract", "risk list", "bullets"),
                include_actions=(True, True, False, True, True),
            ),
            ["Summarize {document} for {audience} in {format}, max {max_words} words, include actions={include_actions}."],
        ),
        (
            "invoice_create",
            _rows(
                customer=("Acme Ltd", "North Labs", "Blue Studio", "Delta Shop", "Orbit Corp"),
                items=(["consulting", "hosting"], ["training"], ["design", "revision"], ["support"], ["license", "setup"]),
                currency=("USD", "EUR", "USD", "CAD", "GBP"),
                due_date=("2026-11-20", "2026-11-21", "2026-11-22", "2026-11-23", "2026-11-24"),
                tax_percent=(8.5, 20, 7.25, 13, 18),
                payment_terms=("net 30", "net 14", "due on receipt", "net 30", "net 45"),
            ),
            ["Create invoice for {customer}, items {items}, currency {currency}, due {due_date}, tax {tax_percent} percent, terms {payment_terms}."],
        ),
        (
            "expense_report",
            _rows(
                employee=("A. Lee", "R. Singh", "J. Park", "M. Diaz", "S. Wood"),
                category=("travel", "meals", "software", "lodging", "supplies"),
                amount=(420.75, 88.2, 199, 730, 54.5),
                currency=("USD", "CAD", "EUR", "USD", "GBP"),
                date=("2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04", "2026-10-05"),
                receipt_attached=(True, False, True, True, False),
            ),
            ["Submit {employee}'s {category} expense for {amount} {currency} on {date}; receipt attached={receipt_attached}."],
        ),
        (
            "classroom_assignment",
            _rows(
                course=("CS101", "BIO220", "HIST310", "MATH205", "DES150"),
                title=("Parsing project", "Cell report", "Archive essay", "Proof set", "Poster critique"),
                due_date=("2026-11-08", "2026-11-09", "2026-11-10", "2026-11-11", "2026-11-12"),
                points=(100, 75, 120, 60, 50),
                submission_type=("repository", "pdf", "document", "pdf", "image"),
            ),
            ['Create assignment "{title}" for {course}, due {due_date}, worth {points} points, submission {submission_type}.'],
        ),
        (
            "survey_create",
            _rows(
                title=("Tool Feedback", "Office Lunch", "Release Review", "Course Check-in", "Feature Vote"),
                audience=("beta users", "employees", "developers", "students", "customers"),
                questions=(["ease", "accuracy"], ["diet", "date"], ["quality", "issues"], ["pace", "help"], ["priority", "reason"]),
                close_date=("2026-10-30", "2026-10-31", "2026-11-01", "2026-11-02", "2026-11-03"),
                anonymous=(True, False, True, True, False),
            ),
            ['Create survey "{title}" for {audience}, questions {questions}, close {close_date}, anonymous={anonymous}.'],
        ),
        (
            "notification_send",
            _rows(
                channel=("email", "slack", "sms", "push", "email"),
                recipients=(["team@example.com"], ["#ops", "#ml"], ["+15551230000"], ["beta-users"], ["exec@example.com"]),
                title=("Training done", "Incident", "Appointment", "New build", "Report ready"),
                message=("SFT completed.", "Latency is high.", "Visit tomorrow.", "Try version 5.", "Review attached results."),
                urgency=("normal", "critical", "high", "normal", "high"),
                send_at=("2026-10-20T10:00Z", "now", "2026-10-21T08:00Z", "2026-10-22T16:00Z", "2026-10-23T09:00Z"),
            ),
            ['Send {urgency} {channel} notification "{title}" to {recipients}: "{message}" at {send_at}.'],
        ),
        (
            "contact_create",
            _rows(
                name=("Ada Green", "Noah Kim", "Mina Shah", "Leo Wong", "Sara Diaz"),
                email=("ada@example.com", "noah@example.com", "mina@example.com", "leo@example.com", "sara@example.com"),
                phone=("+1-555-1001", "+1-555-1002", "+44-20-1003", "+49-30-1004", "+1-555-1005"),
                company=("Acme", "North Labs", "Orbit", "Delta", "Blue Co"),
                tags=(["client", "vip"], ["vendor"], ["research", "lead"], ["partner"], ["client", "new"]),
            ),
            ["Create contact {name}, {email}, {phone}, company {company}, tags {tags}."],
        ),
        (
            "playlist_create",
            _rows(
                name=("Deep Work", "Morning Run", "Dinner Jazz", "Road Trip", "Quiet Reading"),
                genre=("ambient", "pop", "jazz", "rock", "classical"),
                mood=("focused", "energized", "relaxed", "upbeat", "calm"),
                tracks=(30, 25, 20, 40, 18),
                explicit_allowed=(False, True, False, True, False),
            ),
            ['Create playlist "{name}" with {genre} music, mood {mood}, {tracks} tracks, explicit allowed={explicit_allowed}.'],
        ),
        (
            "insurance_quote",
            _rows(
                insurance_type=("auto", "renters", "travel", "life", "home"),
                state=("CA", "NY", "WA", "TX", "CO"),
                coverage_amount=(100000, 50000, 250000, 500000, 400000),
                deductible=(1000, 500, 250, 0, 2000),
                applicant_age=(32, 27, 45, 39, 51),
            ),
            ["Get {insurance_type} insurance quote in {state} for age {applicant_age}, coverage {coverage_amount}, deductible {deductible}."],
        ),
        (
            "appointment_booking",
            _rows(
                provider=("Dr. Chen", "Bright Dental", "Vision Center", "Wellness Clinic", "Dr. Ahmed"),
                service=("consultation", "cleaning", "eye exam", "physical", "follow-up"),
                date=("2026-11-16", "2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20"),
                time=("10:00", "14:30", "09:15", "16:00", "11:45"),
                location=("Downtown", "North Office", "Main Street", "West Clinic", "East Wing"),
                patient_name=("Jamie Lee", "Robin Kay", "Alex Doe", "Chris Sun", "Taylor Wu"),
            ),
            ["Book {service} with {provider} for {patient_name} on {date} at {time}, location {location}."],
        ),
        (
            "conference_room_booking",
            _rows(
                building=("HQ", "North", "Annex", "Tower", "Lab"),
                room=("Orchid", "N-204", "Maple", "T-12", "L-3"),
                date=("2026-11-25", "2026-11-26", "2026-11-27", "2026-11-28", "2026-11-29"),
                start_time=("09:00", "11:00", "14:00", "15:30", "08:30"),
                duration_minutes=(60, 90, 45, 120, 30),
                attendees=(8, 12, 5, 20, 4),
                video_required=(True, False, True, True, False),
            ),
            ["Reserve room {room} in {building} on {date} at {start_time} for {duration_minutes} minutes, {attendees} attendees, video={video_required}."],
        ),
    ]


def main() -> None:
    records = write_grpo_dataset()
    print(f"wrote {len(records)} GRPO source records to {DEFAULT_GRPO_SOURCE_PATH}")


if __name__ == "__main__":
    main()
