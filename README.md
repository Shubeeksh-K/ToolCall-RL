# toolcall-rl

## General Help ADK Agent

This project includes a Google ADK assistant backed by an Ollama-hosted
`smollm:1.7b` base model.

Start Ollama with the model available:

```bash
ollama pull smollm:1.7b
ollama serve
```

Run the ADK agent:

```bash
uv run adk run src/toolcall_rl/agents/Base_Model
```

The agent is defined in `src/toolcall_rl/agents/Base_Model/agent.py`.
The tools live in `src/toolcall_rl/tools/`, one tool per file:

- `calculator.py`: evaluates basic arithmetic.
- `google_search.py`: returns the search query without performing a real search.
- `unit_converter.py`: converts a small set of common units.
- `text_stats.py`: counts basic text properties.
- `string_formatter.py`: applies simple text formatting.

Optional environment variables:

```bash
OLLAMA_MODEL=ollama_chat/smollm:1.7b
OLLAMA_API_BASE=http://localhost:11434
```

The default model string is `ollama_chat/smollm:1.7b`.

## Dataset

Build the canonical tool-call dataset:

```bash
uv run python -m toolcall_rl.data.build_dataset
```

Validate it:

```bash
uv run python -m toolcall_rl.data.validate_dataset
```

Export the SFT view from the canonical data:

```bash
uv run python -m toolcall_rl.data.export_sft
```

Build the direct 50-tool GRPO source data and export its training view:

```bash
uv run python -m toolcall_rl.data.build_grpo_dataset
uv run python -m toolcall_rl.data.export_grpo
```

Generated files:

- `data/tool_call_dataset.jsonl`: original 20-tool source records, 50 examples per tool.
- `data/sft/tool_call_sft.jsonl`: SFT records for the original 20 tools, first 25 examples per tool.
- `data/grpo/tool_call_source.jsonl`: GRPO-only source records, 50 direct examples per tool across 50 tools.
- `data/grpo/tool_call_grpo.jsonl`: prompt and labels exported from the GRPO source.

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
- TensorBoard logs: `outputs/tensorboard/sft_smollm_20tools`

Launch TensorBoard:

```bash
uv run tensorboard --logdir outputs/tensorboard
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
- TensorBoard logs: `outputs/tensorboard/grpo_smollm_50tools_direct`

## Evaluation

### Original 20-Tool Benchmark

These notebooks retain the first experiment using the original 20 tools. They
are useful historical measurements, but are separate from the expanded
50-tool comparison.

- `notebooks/baseline_eval_table.ipynb`: base-model evaluation; saves `outputs/baseline_eval_results.jsonl` and `outputs/baseline_eval_results.csv`.
- `notebooks/sft_eval_table.ipynb`: SFT adapter evaluation; saves `outputs/sft_20tools_eval_results.jsonl` and `outputs/sft_20tools_eval_results.csv`.

### Direct 50-Tool Benchmark

Use these three notebooks for the final apples-to-apples comparison. They all
load the same 50 held-out direct requests from
`src/toolcall_rl/evaluation/direct_50_cases.py` and use the same four binary
reward metrics.

- `notebooks/base_50_tools_eval_table.ipynb`: untrained base model; saves `outputs/base_50tools_direct_eval_results.jsonl` and `outputs/base_50tools_direct_eval_results.csv`.
- `notebooks/sft_50_tools_eval_table.ipynb`: 20-tool SFT adapter before GRPO; saves `outputs/sft_50tools_direct_eval_results.jsonl` and `outputs/sft_50tools_direct_eval_results.csv`.
- `notebooks/grpo_50_tools_eval_table.ipynb`: 50-tool GRPO adapter; saves `outputs/grpo_50tools_direct_eval_results.jsonl` and `outputs/grpo_50tools_direct_eval_results.csv`.

Open and run the GRPO notebook after training:

```text
notebooks/grpo_50_tools_eval_table.ipynb
```
