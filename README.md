# toolcall-rl

## ADK Tool-Calling Demo

This project includes three Google ADK demo agents that use the same local
tools but different model behavior:

- `src/toolcall_rl/agents/Base_Model`: unfine-tuned base model.
- `src/toolcall_rl/agents/FT_Model`: raw GRPO fine-tuned adapter.
- `src/toolcall_rl/agents/Final_Model`: routed GRPO adapter for the practical demo.

`Base_Model` and `FT_Model` use the bridge in
`src/toolcall_rl/agents/hf_tool_call_model.py`, which converts learned JSON
output into ADK function calls. `Final_Model` uses a separate routed bridge in
`src/toolcall_rl/agents/routed_hf_tool_call_model.py`; it sends tool-like
prompts to the fine-tuned model and handles basic chat without forcing a tool
call.

Run the base-model agent and try the calculator prompt:

```bash
uv run adk run src/toolcall_rl/agents/Base_Model
```

```text
[user]: Calculate (94 / 2) + 11.
[base_model_tool_demo]: The answer is: 45
```

The base model answers in plain text, so no ADK tool is executed.

Run the fine-tuned agent with the same prompt:

```bash
uv run adk run src/toolcall_rl/agents/FT_Model
```

```text
[user]: Calculate (94 / 2) + 11.
[ft_model_tool_demo]: Tool `calculator` returned:
{"expression": "(94 / 2) + 11", "result": 58.0}
```

The fine-tuned model emits the learned JSON tool-call shape. The bridge turns
that JSON into an ADK function call, and ADK executes the existing
`calculator` tool.

Run the routed final agent for a more practical chat surface:

```bash
uv run adk run src/toolcall_rl/agents/Final_Model
```

In this version, prompts like `hi` receive a short normal response:

```text
[user]: hi
[final_model_tool_demo]: Hi! I can help with calculations, Google search queries, unit conversions, text statistics, and string formatting.
```

Tool-like prompts still go through the GRPO adapter and ADK tool execution:

```text
[user]: Calculate (94 / 2) + 11.
[final_model_tool_demo]: Tool `calculator` returned:
{"expression": "(94 / 2) + 11", "result": 58.0}
```

The tools live in `src/toolcall_rl/tools/`, one tool per file:

- `calculator.py`: evaluates basic arithmetic.
- `google_search.py`: searches the web through Tavily using `TAVILY_KEY`.
- `unit_converter.py`: converts a small set of common units.
- `text_stats.py`: counts basic text properties.
- `string_formatter.py`: applies simple text formatting.

Optional environment variables:

```bash
BASE_HF_MODEL_ID=HuggingFaceTB/SmolLM-1.7B-Instruct
FT_HF_MODEL_ID=HuggingFaceTB/SmolLM-1.7B-Instruct
FT_ADAPTER_DIR=outputs/grpo_smollm_50tools_direct
FINAL_HF_MODEL_ID=HuggingFaceTB/SmolLM-1.7B-Instruct
FINAL_ADAPTER_DIR=outputs/grpo_smollm_50tools_direct
TAVILY_KEY=tvly-your-key
```

## Dataset

The retained training inputs are:

- `data/sft/tool_call_sft.jsonl`: SFT records for the original 20 tools, 25 examples per tool.
- `data/grpo/tool_call_grpo.jsonl`: GRPO records for all 50 tools, 50 direct examples per tool.

SFT retains the original 20-tool curriculum and its 500 simple examples. The
GRPO curriculum contains 50 tools and 2,500 direct-request examples, including
30 new tools with mostly higher-argument calls. It isolates adaptation to new
schemas rather than correction or conflict handling. Final argument
combinations for the original tools were not included in SFT.

GRPO does not present all 50 schemas in one system prompt. The tools are split
into five stable batches of 10 tools, and each prompt includes only the batch
containing its expected tool.

The executable ADK agent still exposes the initial five local tools. The
additional schemas are training tasks for learning more varied tool-call
argument structures.

## SFT Training

Train a LoRA adapter with TRL SFT:

```bash
uv run python -m toolcall_rl.training.sft
```

Training settings live in `src/toolcall_rl/training/sft.py`, directly inside
`SFTConfig` and `LoraConfig`. For a quick smoke run, set this inside
`SFTConfig`:

```python
max_steps=2
```

Outputs:

- LoRA adapter: `outputs/sft_smollm_20tools`
- TensorBoard logs: `outputs/sft_smollm_20tools/runs`

Launch TensorBoard:

```bash
uv run tensorboard --logdir outputs
```

## GRPO Training

After SFT evaluation, train the SFT LoRA adapter further with GRPO:

```bash
uv run python -m toolcall_rl.training.grpo
```

The GRPO reward is the sum of four binary rewards:

- valid JSON
- JSON-only response
- correct tool
- correct arguments

Outputs:

- GRPO adapter: `outputs/grpo_smollm_50tools_direct`
- TensorBoard logs: `outputs/grpo_smollm_50tools_direct/runs`

## Evaluation

Use these three notebooks for the direct 50-tool comparison. They all
load the same 50 held-out direct requests from
`src/toolcall_rl/evaluation/direct_50_cases.py` and use the same four binary
reward metrics.

- `notebooks/base_50_tools_eval_table.ipynb`: untrained base model; saves `outputs/base_50tools_direct_eval_results.jsonl` and `outputs/base_50tools_direct_eval_results.csv`.
- `notebooks/sft_50_tools_eval_table.ipynb`: 20-tool SFT adapter before GRPO; saves `outputs/sft_50tools_direct_eval_results.jsonl` and `outputs/sft_50tools_direct_eval_results.csv`.
- `notebooks/grpo_50_tools_eval_table.ipynb`: 50-tool GRPO adapter; saves `outputs/grpo_50tools_direct_eval_results.jsonl` and `outputs/grpo_50tools_direct_eval_results.csv`.

Final scores:

| Model | Total Reward | Percentage |
| --- | ---: | ---: |
| Base model | 9 / 200 | 4.5% |
| SFT adapter | 158 / 200 | 79.0% |
| GRPO adapter | 183 / 200 | 91.5% |
