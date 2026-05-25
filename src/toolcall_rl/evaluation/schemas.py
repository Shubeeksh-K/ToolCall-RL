"""Tool-call schemas used by baseline eval and rewards."""

TOOL_NAMES = [
    "calculator",
    "google_search",
    "unit_converter",
    "text_stats",
    "string_formatter",
]

TOOL_SCHEMA_PROMPT = """
Available tools:

1. calculator
Use this for arithmetic.
Schema:
{
  "tool": "calculator",
  "args": {
    "expression": "string"
  }
}

2. google_search
Use this when the user asks to search Google or find web/current information.
This tool only records the intended search query.
Schema:
{
  "tool": "google_search",
  "args": {
    "query": "string"
  }
}

3. unit_converter
Use this for supported unit conversions.
Schema:
{
  "tool": "unit_converter",
  "args": {
    "value": number,
    "from_unit": "string",
    "to_unit": "string"
  }
}

4. text_stats
Use this to count words, sentences, or characters in text.
Schema:
{
  "tool": "text_stats",
  "args": {
    "text": "string"
  }
}

5. string_formatter
Use this to uppercase, lowercase, titlecase, or reverse text.
Schema:
{
  "tool": "string_formatter",
  "args": {
    "text": "string",
    "operation": "uppercase | lowercase | titlecase | reverse"
  }
}
""".strip()

SYSTEM_PROMPT = f"""
You are a tool-calling model.

{TOOL_SCHEMA_PROMPT}

If a tool is needed, output ONLY valid JSON.
Do not explain.
Do not wrap in markdown.
""".strip()
