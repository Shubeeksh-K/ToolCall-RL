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

### calculator

Use for arithmetic.

```json
{
  "tool": "calculator",
  "args": {
    "expression": "24 * 17"
  }
}
```

### google_search

Use when the user asks to search Google or find web/current information.

This is a no-key, no-network intent tool. It only records the query.

```json
{
  "tool": "google_search",
  "args": {
    "query": "google adk ollama tools"
  }
}
```

### unit_converter

Use for supported unit conversions.

```json
{
  "tool": "unit_converter",
  "args": {
    "value": 10,
    "from_unit": "kilometers",
    "to_unit": "miles"
  }
}
```

### text_stats

Use to count words, sentences, or characters.

```json
{
  "tool": "text_stats",
  "args": {
    "text": "Hello world. Tool calls work!"
  }
}
```

### string_formatter

Use to uppercase, lowercase, titlecase, or reverse text.

```json
{
  "tool": "string_formatter",
  "args": {
    "text": "learning tool calls",
    "operation": "titlecase"
  }
}
```

## Baseline Prompt Template

The notebook can keep using `litellm.completion`.

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
- `src/toolcall_rl/evaluation/cases.py`: 10 seed eval cases, 2 per tool.
- `src/toolcall_rl/evaluation/scoring.py`: shared binary scoring logic.
- `src/toolcall_rl/evaluation/baseline.py`: LiteLLM baseline runner.
