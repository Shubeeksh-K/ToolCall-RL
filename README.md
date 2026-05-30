# toolcall-rl

Fine-tuning a small language model to make reliable tool calls, then wiring the
fine-tuned model into a Google ADK agent.

This project starts with `HuggingFaceTB/SmolLM-1.7B-Instruct`, a small model
that is not reliable at structured tool calling. I trained it with SFT and then
GRPO so it learns to emit JSON tool calls of the form:

```json
{"tool": "calculator", "args": {"expression": "(94 / 2) + 11"}}
```

The final adapter is used inside Google ADK through a small bridge that turns
the learned JSON into ADK function calls. The result is a compact tool-calling
agent that can execute real local tools, including a Tavily-backed search tool.

## Project Highlights

- Fine-tuned a small open model for structured tool calling with SFT and GRPO.
- Built a 50-tool curriculum with held-out evaluation cases and binary reward
  functions.
- Improved direct 50-tool evaluation score from `9 / 200` to `183 / 200`.
- Used LoRA adapters to keep training practical on local hardware.
- Integrated the fine-tuned model into Google ADK through a JSON-to-function-call
  bridge.
- Added a routed final agent that preserves basic chat behavior while using the
  fine-tuned model for tool-like prompts.
- Replaced a dummy search tool with a real Tavily-backed search implementation.

## Tech Stack

| Area | Tools |
| --- | --- |
| Base model | `HuggingFaceTB/SmolLM-1.7B-Instruct` |
| Fine-tuning | TRL SFT, TRL GRPO, PEFT LoRA |
| Runtime agent | Google ADK |
| Model serving path | Local Hugging Face Transformers wrapper |
| Tool execution | ADK function calls bridged from learned JSON |
| Search | Tavily Search API |
| Observability | TensorBoard |
| Testing | Pytest, Ruff |

## Results

The final evaluation uses 50 held-out direct tool-calling tasks, one per tool.
Each response receives four binary rewards:

- valid JSON
- JSON-only response
- correct tool name
- correct arguments

Maximum score is `200`, from `50 cases x 4 rewards`.

| Model | Total Reward | Accuracy |
| --- | ---: | ---: |
| Base SmolLM-1.7B-Instruct | 9 / 200 | 4.5% |
| SFT adapter | 158 / 200 | 79.0% |
| GRPO adapter | 183 / 200 | 91.5% |

The key progression:

- SFT teaches the model the basic JSON tool-call format.
- GRPO improves tool selection and argument accuracy on a larger 50-tool
  curriculum.
- The final GRPO adapter improves by `+174` reward points over the base model
  and `+25` over the SFT adapter.

## Live ADK Demo

The repository contains three Google ADK agents that expose the same local
tools but use different model behavior:

| Agent | Purpose |
| --- | --- |
| `src/toolcall_rl/agents/Base_Model` | Unfine-tuned Hugging Face base model |
| `src/toolcall_rl/agents/FT_Model` | Raw GRPO fine-tuned tool-call model |
| `src/toolcall_rl/agents/Final_Model` | Practical routed agent for demos |

`Base_Model` and `FT_Model` use
`src/toolcall_rl/agents/hf_tool_call_model.py`, which loads a Hugging Face model
and converts learned JSON into ADK function calls.

`Final_Model` uses
`src/toolcall_rl/agents/routed_hf_tool_call_model.py`. It sends tool-like
prompts to the GRPO adapter and keeps simple greetings out of the tool-call
model, which makes the final demo more natural.

### Base Model Fails To Call A Tool

```bash
uv run adk run src/toolcall_rl/agents/Base_Model
```

```text
[user]: Calculate (94 / 2) + 11.
[base_model_tool_demo]: The answer is: 45
```

The base model answers in plain text, does not trigger ADK tool execution, and
gets the answer wrong.

### Fine-Tuned Model Calls The Tool

```bash
uv run adk run src/toolcall_rl/agents/FT_Model
```

```text
[user]: Calculate (94 / 2) + 11.
[ft_model_tool_demo]: Tool `calculator` returned:
{"expression": "(94 / 2) + 11", "result": 58.0}
```

The fine-tuned model emits the learned JSON tool-call shape. The bridge turns
that JSON into an ADK function call, and ADK executes the real `calculator`
tool.

### Final Routed Agent

```bash
uv run adk run src/toolcall_rl/agents/Final_Model
```

Basic chat stays normal:

```text
[user]: hi
[final_model_tool_demo]: Hi! I can help with calculations, Google search queries, unit conversions, text statistics, and string formatting.
```

Tool-like prompts still route through the fine-tuned model and ADK tool
execution:

```text
[user]: Calculate (94 / 2) + 11.
[final_model_tool_demo]: Tool `calculator` returned:
{"expression": "(94 / 2) + 11", "result": 58.0}
```

## Tools

The ADK demo exposes five Python tools:

| Tool | File | Behavior |
| --- | --- | --- |
| `calculator` | `src/toolcall_rl/tools/calculator.py` | Safely evaluates arithmetic expressions |
| `google_search` | `src/toolcall_rl/tools/google_search.py` | Uses Tavily Search API with `TAVILY_KEY` |
| `unit_converter` | `src/toolcall_rl/tools/unit_converter.py` | Converts common units |
| `text_stats` | `src/toolcall_rl/tools/text_stats.py` | Counts words, characters, and sentences |
| `string_formatter` | `src/toolcall_rl/tools/string_formatter.py` | Applies uppercase, lowercase, titlecase, or reverse |

The training and evaluation curriculum contains 50 tool schemas. The live ADK
demo intentionally exposes the first five tools so the behavior is easy to
inspect interactively.

## Training Pipeline

The project has two fine-tuning stages.

### 1. Supervised Fine-Tuning

SFT teaches the model the basic JSON output format.

- Base model: `HuggingFaceTB/SmolLM-1.7B-Instruct`
- Training data: `data/sft/tool_call_sft.jsonl`
- Size: 500 examples
- Tool coverage: original 20 tools
- Examples per tool: 25
- Adapter output: `outputs/sft_smollm_20tools`
- Logging: TensorBoard under `outputs/sft_smollm_20tools/runs`

Run:

```bash
uv run python -m toolcall_rl.training.sft
```

Key settings:

- LoRA rank `16`
- LoRA alpha `32`
- LoRA dropout `0.05`
- learning rate `2e-4`
- 3 epochs
- batch size `1`
- gradient accumulation `8`

### 2. GRPO Fine-Tuning

GRPO improves tool-calling behavior with reward functions that match the final
evaluation metrics.

- Starting point: SFT adapter from `outputs/sft_smollm_20tools`
- Training data: `data/grpo/tool_call_grpo.jsonl`
- Size: 2,500 examples
- Tool coverage: 50 tools
- Examples per tool: 50
- Adapter output: `outputs/grpo_smollm_50tools_direct`
- Logging: TensorBoard under `outputs/grpo_smollm_50tools_direct/runs`

Run:

```bash
uv run python -m toolcall_rl.training.grpo
```

The GRPO reward is the sum of four binary reward functions:

| Reward | Meaning |
| --- | --- |
| `reward_valid_json` | Response contains a parseable JSON object |
| `reward_json_only` | Response contains only JSON, no prose or markdown |
| `reward_tool_match` | `tool` matches the expected tool |
| `reward_args_match` | `args` contain the expected arguments |

Key settings:

- learning rate `1e-5`
- 1 epoch
- batch size `1`
- gradient accumulation `4`
- `num_generations=4`
- `generation_batch_size=4`
- `max_completion_length=128`
- `beta=0.0`

## Dataset Design

The dataset is deliberately split so that SFT and GRPO have different jobs.

| Split | File | Tools | Examples |
| --- | --- | ---: | ---: |
| SFT | `data/sft/tool_call_sft.jsonl` | 20 | 500 |
| GRPO | `data/grpo/tool_call_grpo.jsonl` | 50 | 2,500 |

SFT uses simpler examples from the original 20 tools. GRPO expands the tool
inventory to 50 tools and includes richer argument structures. The original
tools also use argument combinations that were not included in the SFT data,
so GRPO has to improve both format consistency and argument copying.

The 50 tools are split into five stable batches of 10 schemas. Each GRPO
example only receives the 10-tool system prompt containing its target tool,
instead of a single giant prompt with all 50 tools. This keeps the prompt
realistic and avoids making the task mostly about long-context schema search.

## Evaluation

The final benchmark is defined in
`src/toolcall_rl/evaluation/direct_50_cases.py`.

It contains 50 held-out direct requests, one per tool. The values are held out
from the training data.

Evaluation notebooks:

| Notebook | Model |
| --- | --- |
| `notebooks/01_base_50_tools_eval.ipynb` | Base model |
| `notebooks/02_sft_50_tools_eval.ipynb` | SFT adapter |
| `notebooks/03_grpo_50_tools_eval.ipynb` | GRPO adapter |

The scorer lives in `src/toolcall_rl/evaluation/scoring.py`. It parses model
outputs, checks JSON-only compliance, compares tool names, and validates
arguments. The same scoring logic backs both offline evaluation and GRPO
reward computation.

## Architecture

```text
User prompt
    |
    v
Google ADK agent
    |
    v
Hugging Face bridge
    |
    |-- Base_Model: raw SmolLM output
    |
    |-- FT_Model: GRPO model emits JSON tool call
    |
    `-- Final_Model: router decides chat vs tool-call path
            |
            v
      JSON tool call
            |
            v
      ADK function call
            |
            v
      Python tool execution
```

The bridge exists because the model was trained to emit JSON text, while ADK
expects an internal function-call object. The bridge keeps the training format
simple and still lets ADK execute real tools.

## Repository Layout

```text
data/
  sft/tool_call_sft.jsonl
  grpo/tool_call_grpo.jsonl

notebooks/
  01_base_50_tools_eval.ipynb
  02_sft_50_tools_eval.ipynb
  03_grpo_50_tools_eval.ipynb

src/toolcall_rl/
  agents/
    Base_Model/
    FT_Model/
    Final_Model/
    hf_tool_call_model.py
    routed_hf_tool_call_model.py
  evaluation/
    direct_50_cases.py
    rewards.py
    schemas.py
    scoring.py
  tools/
    calculator.py
    google_search.py
    unit_converter.py
    text_stats.py
    string_formatter.py
  training/
    sft.py
    grpo.py
```

## Setup

Install dependencies:

```bash
uv sync
```

Create `.env`:

```bash
cp .env.example .env
```

Add your Tavily key if you want the search tool to call the real API:

```bash
TAVILY_KEY=tvly-your-key
```

Optional model environment variables:

```bash
BASE_HF_MODEL_ID=HuggingFaceTB/SmolLM-1.7B-Instruct
FT_HF_MODEL_ID=HuggingFaceTB/SmolLM-1.7B-Instruct
FT_ADAPTER_DIR=outputs/grpo_smollm_50tools_direct
FINAL_HF_MODEL_ID=HuggingFaceTB/SmolLM-1.7B-Instruct
FINAL_ADAPTER_DIR=outputs/grpo_smollm_50tools_direct
```

Run tests:

```bash
uv run pytest
```

Run lint:

```bash
uv run ruff check src tests
```

## Why This Project Matters

This project demonstrates the full loop of a practical tool-calling fine-tune:

- designing a structured tool-call format
- building SFT and GRPO datasets
- training LoRA adapters on a local GPU-friendly model
- using binary reward functions for RL-style optimization
- evaluating base, SFT, and GRPO models on the same held-out benchmark
- integrating the resulting model into Google ADK
- bridging learned JSON into real ADK tool execution
- adding a router so the final agent is usable beyond narrow tool prompts

The main result is not just a better benchmark score. The final model behavior
is visible in an actual ADK agent: the base model fails to call tools, while
the fine-tuned model produces structured tool calls that execute successfully.
