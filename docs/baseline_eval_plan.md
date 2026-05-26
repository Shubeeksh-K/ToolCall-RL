# Baseline Tool-Call Evaluation Plan

## Goal

Measure how the base `smollm:1.7b` model behaves before SFT/GRPO when it is
asked to use tools.

The baseline evaluation should answer:

- Does the model know when a tool is needed?
- Does it choose the correct tool?
- Does it produce the expected structure?
- Are the arguments correct?
- Does it avoid explanations when JSON-only output is requested?

## Target Tool-Call Format

When a tool is needed, the model should output only valid JSON:

```json
{
  "tool": "tool_name",
  "args": {}
}
```

No markdown, no explanation, no surrounding text.

## Tool Schemas

The original baseline and SFT curriculum contains 20 tool schemas:

```text
calculator, google_search, unit_converter, text_stats, string_formatter,
weather_lookup, currency_converter, translate_text, create_calendar_event,
send_email, restaurant_search, book_flight, hotel_search, route_planner,
product_search, set_reminder, track_package, stock_quote, file_search,
schedule_meeting
```

The exact argument definitions and compact model prompt live in
`src/toolcall_rl/evaluation/schemas.py`.

Examples include simple argument structures:

```json
{"tool": "stock_quote", "args": {"ticker": "NVDA"}}
```

and more complex structures:

```json
{
  "tool": "schedule_meeting",
  "args": {
    "title": "Model review",
    "date": "2026-06-12",
    "time": "14:00",
    "timezone": "UTC",
    "attendees": ["alice@example.com", "bob@example.com"]
  }
}
```

## Baseline Prompt Template

The evaluation notebooks load `HuggingFaceTB/SmolLM-1.7B-Instruct` directly
with Transformers. The legacy command-line baseline runner can still use
LiteLLM separately.

System prompt shape:

```text
You are a tool-calling model.

Available tools:
<tool schemas>

If a tool is needed, output ONLY valid JSON.
Do not explain.
Do not wrap in markdown.
```

## Evaluation Labels

Each test case should include:

```json
{
  "prompt": "Convert 10 kilometers to miles.",
  "expected_tool": "unit_converter",
  "expected_args": {
    "value": 10,
    "from_unit": "kilometers",
    "to_unit": "miles"
  }
}
```

## Metrics

For each model response, score:

- `valid_json`: `0` or `1`; response contains a parseable JSON object.
- `json_only`: `0` or `1`; response has no extra natural language or markdown.
- `tool_match`: `0` or `1`; parsed `tool` matches expected tool.
- `args_match`: `0` or `1`; parsed `args` contains the expected arguments.
- `total_reward`: integer sum of the four metrics, from `0` to `4`.

Dependency rules:

- if `valid_json` is `0`, then `tool_match` and `args_match` are `0`.
- if `tool_match` is `0`, then `args_match` is `0`.

## Implemented Files

- `src/toolcall_rl/evaluation/schemas.py`: tool schema prompt and tool names.
- `src/toolcall_rl/evaluation/cases.py`: 20 held-out cases, one unseen example per tool.
- `src/toolcall_rl/evaluation/scoring.py`: shared binary scoring logic.
- `src/toolcall_rl/evaluation/baseline.py`: LiteLLM baseline runner.

## Training Dataset Split

- Canonical dataset: 20 tools x 50 examples = 1,000 records.
- SFT dataset: first 25 examples per tool = 500 records.
- GRPO dataset: 50 tools x 50 direct examples = 2,500 records.

GRPO adds 30 previously unseen tools with mostly four-to-seven arguments. Its
records use argument combinations for the original tools that were excluded
from SFT. All prompts are direct requests so this experiment measures
adaptation to the expanded tool inventory and richer argument structures.

All 50 GRPO schemas are divided into five fixed batches of 10. A GRPO example
receives only the system prompt for the batch containing its target tool, not
a common prompt containing every available tool.

## Evaluation Tracks

The original 20-tool benchmark is retained as a historical baseline for the
first SFT stage:

- `notebooks/baseline_eval_table.ipynb`: base model on the original 20 tools.
- `notebooks/sft_eval_table.ipynb`: SFT adapter on the original 20 tools.

The expanded 50-tool benchmark is the comparison to use when evaluating
adaptation through GRPO.

## Direct 50-Tool Evaluation

`src/toolcall_rl/evaluation/direct_50_cases.py` defines 50 separate held-out direct
requests, one for each GRPO tool, using the same ten-tool prompt batches but
values not included in GRPO training.

`notebooks/base_50_tools_eval_table.ipynb` evaluates the untrained base model
on these cases, providing the expanded-tool baseline.

`notebooks/sft_50_tools_eval_table.ipynb` evaluates the SFT adapter on these
cases to record its pre-GRPO performance on the expanded tool set.

`notebooks/grpo_50_tools_eval_table.ipynb` evaluates the trained GRPO adapter
on the identical cases for direct comparison with the SFT result.
