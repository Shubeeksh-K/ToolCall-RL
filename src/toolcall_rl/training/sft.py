"""SFT training for tool-call JSON generation with LoRA."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data" / "sft" / "tool_call_sft.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sft_smollm_20tools"
LOGGING_DIR = OUTPUT_DIR / "runs"
MODEL_ID = os.getenv("SFT_MODEL_ID", "HuggingFaceTB/SmolLM-1.7B-Instruct")


def main() -> None:
    is_cuda = torch.cuda.is_available()
    device = "cuda" if is_cuda else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if is_cuda else torch.float32,
    )
    model.to(device)
    model.config.use_cache = False

    dataset = load_dataset("json", data_files=str(DATA_PATH), split="train")
    dataset = dataset.map(
        lambda example: format_example(example, tokenizer),
        remove_columns=dataset.column_names,
    )

    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
    )

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        logging_dir=str(LOGGING_DIR),
        report_to=["tensorboard"],
        run_name="sft_smollm_20tools",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        max_steps=-1,
        learning_rate=2e-4,
        max_length=1024,
        bf16=False,
        fp16=is_cuda,
        optim="adamw_torch",
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=25,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        remove_unused_columns=True,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"saved LoRA adapter to {OUTPUT_DIR}")
    print(f"tensorboard logs at {LOGGING_DIR}")


def format_example(example: dict, tokenizer: AutoTokenizer | None = None) -> dict[str, str]:
    """Flatten chat messages into a single supervised text sequence."""

    messages = example["messages"]
    if tokenizer is not None and tokenizer.chat_template:
        return {
            "text": tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    lines = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        lines.append(f"<|{role}|>\n{content}")
    lines.append("<|end|>")
    return {"text": "\n".join(lines)}


if __name__ == "__main__":
    main()
