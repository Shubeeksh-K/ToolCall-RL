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

Export SFT and GRPO views from the same canonical data:

```bash
uv run python -m toolcall_rl.data.export_sft
uv run python -m toolcall_rl.data.export_grpo
```

Generated files:

- `data/tool_call_dataset.jsonl`: canonical source records.
- `data/sft/tool_call_sft.jsonl`: chat-message records for SFT.
- `data/grpo/tool_call_grpo.jsonl`: prompt and labels for GRPO rewards.

The generated dataset has 250 examples total, with 50 examples for each tool.

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

- LoRA adapter: `outputs/sft_smollm_toolcall`
- TensorBoard logs: `outputs/tensorboard/sft_smollm_toolcall`

Launch TensorBoard:

```bash
uv run tensorboard --logdir outputs/tensorboard
```

Evaluate the SFT adapter with a notebook:

```text
notebooks/sft_eval_table.ipynb
```

It saves:

- `outputs/sft_eval_results.jsonl`
- `outputs/sft_eval_results.csv`

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

- GRPO adapter: `outputs/grpo_smollm_toolcall`
- TensorBoard logs: `outputs/tensorboard/grpo_smollm_toolcall`
