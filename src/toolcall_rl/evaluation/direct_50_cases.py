"""Direct held-out cases for evaluating adaptation to the 50-tool curriculum."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from toolcall_rl.evaluation.schemas import system_prompt_for_tool


@dataclass(frozen=True)
class DirectEvalCase:
    prompt: str
    system_prompt: str
    expected_tool: str
    expected_args: dict[str, Any]


def _case(tool: str, request: str, expected_args: dict[str, Any]) -> DirectEvalCase:
    return DirectEvalCase(
        prompt=request,
        system_prompt=system_prompt_for_tool(tool),
        expected_tool=tool,
        expected_args=expected_args,
    )


DIRECT_50_EVAL_CASES = [
    _case("calculator", "Calculate (94 / 2) + 11.", {"expression": "(94 / 2) + 11"}),
    _case("google_search", "Search Google for reward shaping for structured tool calls.", {"query": "reward shaping for structured tool calls"}),
    _case("unit_converter", "Convert 33.8 celsius to fahrenheit.", {"value": 33.8, "from_unit": "celsius", "to_unit": "fahrenheit"}),
    _case("text_stats", 'Analyze this text: "Hard cases expose weak argument copying."', {"text": "Hard cases expose weak argument copying."}),
    _case("string_formatter", 'Apply titlecase to: "reward driven tools"', {"text": "reward driven tools", "operation": "titlecase"}),
    _case("weather_lookup", "Get the fahrenheit forecast for Phoenix.", {"city": "Phoenix", "unit": "fahrenheit"}),
    _case("currency_converter", "Exchange 418 CAD for EUR.", {"amount": 418, "from_currency": "CAD", "to_currency": "EUR"}),
    _case("translate_text", 'Translate "good evening" from English to Korean.', {"text": "good evening", "source_language": "English", "target_language": "Korean"}),
    _case("create_calendar_event", 'Create an event titled "Reward audit" on 2026-12-08 at 10:20 in Europe/Paris.', {"title": "Reward audit", "date": "2026-12-08", "time": "10:20", "timezone": "Europe/Paris"}),
    _case("send_email", 'Email lab@example.com: subject "Hard evaluation", message "Please inspect the final metrics."', {"recipient": "lab@example.com", "subject": "Hard evaluation", "body": "Please inspect the final metrics."}),
    _case("restaurant_search", "Find Peruvian restaurants in Nashville under 48 dollars.", {"city": "Nashville", "cuisine": "Peruvian", "max_price": 48}),
    _case("book_flight", "Find 3 business flights from Oslo to Amsterdam on 2026-12-11.", {"origin": "Oslo", "destination": "Amsterdam", "date": "2026-12-11", "passengers": 3, "cabin": "business"}),
    _case("hotel_search", "Find a hotel in Kyoto from 2026-12-12 to 2026-12-16 for 2 guests under 290.", {"city": "Kyoto", "check_in": "2026-12-12", "check_out": "2026-12-16", "guests": 2, "max_price": 290}),
    _case("route_planner", "Plan a transit route from Union Station to Art Museum with avoid_tolls=false.", {"origin": "Union Station", "destination": "Art Museum", "mode": "transit", "avoid_tolls": False}),
    _case("product_search", "Find a standing desk under 520 with rating at least 4.6.", {"query": "standing desk", "max_price": 520, "min_rating": 4.6}),
    _case("set_reminder", 'Remind me to "Review hard rewards" on 2026-12-18 at 07:40 in Europe/Berlin.', {"message": "Review hard rewards", "date": "2026-12-18", "time": "07:40", "timezone": "Europe/Berlin"}),
    _case("track_package", "Track package LX902341876DE with DHL.", {"carrier": "DHL", "tracking_number": "LX902341876DE"}),
    _case("stock_quote", "Fetch stock information for CRM.", {"ticker": "CRM"}),
    _case("file_search", 'Find parquet files about "generation rewards" under /experiments.', {"query": "generation rewards", "directory": "/experiments", "file_type": "parquet"}),
    _case("schedule_meeting", 'Schedule "Hard benchmark review" on 2026-12-20 at 16:10 in Europe/Amsterdam with mia@example.com and eli@example.com.', {"title": "Hard benchmark review", "date": "2026-12-20", "time": "16:10", "timezone": "Europe/Amsterdam", "attendees": ["mia@example.com", "eli@example.com"]}),
    _case("reserve_table", "Reserve Juniper Kitchen in Portland on 2026-12-21 at 18:50 for 7 people, seating terrace.", {"restaurant": "Juniper Kitchen", "city": "Portland", "date": "2026-12-21", "time": "18:50", "party_size": 7, "seating": "terrace"}),
    _case("order_grocery", "Order coffee, oranges, and yogurt from Sunrise Grocer for 2026-12-22 in 07:00-09:00; substitutions allowed: true.", {"store": "Sunrise Grocer", "items": ["coffee", "oranges", "yogurt"], "delivery_date": "2026-12-22", "delivery_window": "07:00-09:00", "substitutions_allowed": True}),
    _case("create_support_ticket", 'Open a critical ticket for Model Console: "adapter download stalls", account AC-909, contact fin@example.com.', {"product": "Model Console", "issue": "adapter download stalls", "severity": "critical", "account_id": "AC-909", "contact_email": "fin@example.com"}),
    _case("schedule_delivery", "Schedule a fragile delivery from 17 Cedar Ln to 82 Market Blvd on 2026-12-23, 13:00-16:00, signature true.", {"pickup_address": "17 Cedar Ln", "dropoff_address": "82 Market Blvd", "date": "2026-12-23", "time_window": "13:00-16:00", "package_size": "fragile", "signature_required": True}),
    _case("compare_products", "Compare PeakBook and ThinNote as laptops in USD, under 1450, requiring battery life and linux support.", {"products": ["PeakBook", "ThinNote"], "category": "laptop", "currency": "USD", "max_price": 1450, "features": ["battery life", "linux support"]}),
    _case("job_search", "Find staff Platform Engineer jobs in Dublin, remote=true, minimum salary 142000.", {"role": "Platform Engineer", "location": "Dublin", "remote": True, "min_salary": 142000, "experience_level": "staff"}),
    _case("rental_car_search", "Find an electric SUV rental from Munich on 2026-12-26 to Zurich on 2026-12-30 for a 36-year-old driver.", {"pickup_city": "Munich", "dropoff_city": "Zurich", "pickup_date": "2026-12-26", "dropoff_date": "2026-12-30", "vehicle_type": "electric SUV", "driver_age": 36}),
    _case("movie_ticket_booking", 'Book 5 tickets for "Glass Horizon" at Harbor IMAX on 2026-12-27, 20:40, format IMAX 3D.', {"movie": "Glass Horizon", "cinema": "Harbor IMAX", "date": "2026-12-27", "showtime": "20:40", "tickets": 5, "format": "IMAX 3D"}),
    _case("train_ticket_search", "Search first train tickets from Vienna to Prague on 2026-12-28 after 09:20 for 2 passengers.", {"origin": "Vienna", "destination": "Prague", "date": "2026-12-28", "departure_after": "09:20", "passengers": 2, "class": "first"}),
    _case("event_ticket_search", 'Find 3 tickets for "Winter Coding Summit" in Helsinki on 2026-12-29, maximum price 240.', {"event": "Winter Coding Summit", "city": "Helsinki", "date": "2026-12-29", "tickets": 3, "max_price": 240}),
    _case("prescription_refill", "Refill Cetirizine 10mg at River Pharmacy for patient PT-880, pickup 2027-01-03.", {"medication": "Cetirizine", "dosage": "10mg", "pharmacy": "River Pharmacy", "patient_id": "PT-880", "pickup_date": "2027-01-03"}),
    _case("workout_plan", "Create an advanced flexibility workout, 3 days weekly, 55 minutes, equipment yoga blocks and resistance bands.", {"goal": "flexibility", "days_per_week": 3, "duration_minutes": 55, "equipment": ["yoga blocks", "resistance bands"], "fitness_level": "advanced"}),
    _case("meal_plan", "Plan 4 days of pescatarian meals at 2100 calories with allergies sesame and dairy, 3 meals daily.", {"diet": "pescatarian", "calories": 2100, "days": 4, "allergies": ["sesame", "dairy"], "meals_per_day": 3}),
    _case("project_task_create", 'Create high task "Evaluate unseen tools" in RL Study, assigned to nina@example.com, due 2027-01-06, labels benchmark and grpo.', {"project": "RL Study", "title": "Evaluate unseen tools", "assignee": "nina@example.com", "due_date": "2027-01-06", "priority": "high", "labels": ["benchmark", "grpo"]}),
    _case("database_query", "Query metrics.scores for fields model and reward where split = hard, limit 75, sort by reward.", {"database": "metrics", "table": "scores", "fields": ["model", "reward"], "filter": "split = hard", "limit": 75, "sort_by": "reward"}),
    _case("cloud_deploy", "Deploy evaluator version v5.3.2 to canary in ap-northeast-1 with 3 replicas, rollback=true.", {"service": "evaluator", "environment": "canary", "region": "ap-northeast-1", "version": "v5.3.2", "replicas": 3, "rollback_on_failure": True}),
    _case("log_search", 'Search canary router logs for "invalid arguments" from 2027-01-08T07:00Z to 2027-01-08T09:30Z at level error.', {"service": "router", "environment": "canary", "query": "invalid arguments", "start_time": "2027-01-08T07:00Z", "end_time": "2027-01-08T09:30Z", "level": "error"}),
    _case("api_request", 'Prepare POST request to /v2/evaluate with headers Authorization and X-Run-Id, body \'{"suite":"direct50"}\', timeout 45 seconds.', {"method": "POST", "endpoint": "/v2/evaluate", "headers": ["Authorization", "X-Run-Id"], "body": '{"suite":"direct50"}', "timeout_seconds": 45}),
    _case("image_generation", 'Generate "tool call accuracy dashboard" in technical illustration style, 1400x900, background light grid.', {"prompt": "tool call accuracy dashboard", "style": "technical illustration", "width": 1400, "height": 900, "background": "light grid"}),
    _case("document_summary", "Summarize grpo-findings.pdf for research reviewers in structured memo, max 275 words, include actions=true.", {"document": "grpo-findings.pdf", "audience": "research reviewers", "max_words": 275, "format": "structured memo", "include_actions": True}),
    _case("invoice_create", "Create invoice for Vector Systems, items evaluation and deployment, currency USD, due 2027-01-15, tax 9.25 percent, terms net 21.", {"customer": "Vector Systems", "items": ["evaluation", "deployment"], "currency": "USD", "due_date": "2027-01-15", "tax_percent": 9.25, "payment_terms": "net 21"}),
    _case("expense_report", "Submit N. Rao's conference expense for 612.45 EUR on 2027-01-10; receipt attached=true.", {"employee": "N. Rao", "category": "conference", "amount": 612.45, "currency": "EUR", "date": "2027-01-10", "receipt_attached": True}),
    _case("classroom_assignment", 'Create assignment "Reward analysis" for ML402, due 2027-01-18, worth 85 points, submission notebook.', {"course": "ML402", "title": "Reward analysis", "due_date": "2027-01-18", "points": 85, "submission_type": "notebook"}),
    _case("survey_create", 'Create survey "Model Preference" for annotators, questions clarity, correctness, and format, close 2027-01-19, anonymous=true.', {"title": "Model Preference", "audience": "annotators", "questions": ["clarity", "correctness", "format"], "close_date": "2027-01-19", "anonymous": True}),
    _case("notification_send", 'Send high slack notification "Benchmark complete" to #research and #evaluation: "Hard suite results are ready." at 2027-01-20T17:00Z.', {"channel": "slack", "recipients": ["#research", "#evaluation"], "title": "Benchmark complete", "message": "Hard suite results are ready.", "urgency": "high", "send_at": "2027-01-20T17:00Z"}),
    _case("contact_create", "Create contact Priya Nair, priya@vector.example, +91-80-5550-2211, company Vector Systems, tags evaluator and partner.", {"name": "Priya Nair", "email": "priya@vector.example", "phone": "+91-80-5550-2211", "company": "Vector Systems", "tags": ["evaluator", "partner"]}),
    _case("playlist_create", 'Create playlist "Long Training Night" with electronic music, mood steady, 33 tracks, explicit allowed=false.', {"name": "Long Training Night", "genre": "electronic", "mood": "steady", "tracks": 33, "explicit_allowed": False}),
    _case("insurance_quote", "Get travel insurance quote in IL for age 44, coverage 175000, deductible 350.", {"insurance_type": "travel", "state": "IL", "coverage_amount": 175000, "deductible": 350, "applicant_age": 44}),
    _case("appointment_booking", "Book diagnostic consultation with Harbor Health for Morgan Yu on 2027-01-24 at 13:25, location South Campus.", {"provider": "Harbor Health", "service": "diagnostic consultation", "date": "2027-01-24", "time": "13:25", "location": "South Campus", "patient_name": "Morgan Yu"}),
    _case("conference_room_booking", "Reserve room Cedar in Innovation Center on 2027-01-25 at 10:10 for 75 minutes, 14 attendees, video=true.", {"building": "Innovation Center", "room": "Cedar", "date": "2027-01-25", "start_time": "10:10", "duration_minutes": 75, "attendees": 14, "video_required": True}),
]
