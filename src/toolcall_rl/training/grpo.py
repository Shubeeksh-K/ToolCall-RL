"""GRPO training for adapting the SFT model to 50 direct-request tools."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from toolcall_rl.evaluation.rewards import (
    reward_args_match,
    reward_json_only,
    reward_tool_match,
    reward_valid_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data" / "grpo" / "tool_call_grpo.jsonl"
SFT_ADAPTER_DIR = PROJECT_ROOT / "outputs" / "sft_smollm_20tools"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "grpo_smollm_50tools_direct"
LOGGING_DIR = PROJECT_ROOT / "outputs" / "tensorboard" / "grpo_smollm_50tools_direct"
MODEL_ID = os.getenv("GRPO_MODEL_ID", "HuggingFaceTB/SmolLM-1.7B-Instruct")


def main() -> None:
    is_cuda = torch.cuda.is_available()
    device = "cuda" if is_cuda else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(SFT_ADAPTER_DIR, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if is_cuda else torch.float32,
    )
    model = PeftModel.from_pretrained(
        base_model,
        SFT_ADAPTER_DIR,
        is_trainable=True,
    )
    model.to(device)
    model.config.use_cache = False

    dataset = load_dataset("json", data_files=str(DATA_PATH), split="train")

    training_args = GRPOConfig(
        output_dir=str(OUTPUT_DIR),
        logging_dir=str(LOGGING_DIR),
        report_to=["tensorboard"],
        run_name="grpo_smollm_50tools_direct",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        generation_batch_size=4,
        num_generations=4,
        num_train_epochs=1,
        max_steps=-1,
        learning_rate=1e-5,
        max_completion_length=128,
        temperature=0.8,
        beta=0.0,
        bf16=False,
        fp16=is_cuda,
        optim="adamw_torch",
        logging_steps=1,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=2,
        gradient_checkpointing=True,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        log_completions=True,
        num_completions_to_print=2,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            reward_valid_json,
            reward_json_only,
            reward_tool_match,
            reward_args_match,
        ],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"saved GRPO adapter to {OUTPUT_DIR}")
    print(f"tensorboard logs at {LOGGING_DIR}")


if __name__ == "__main__":
    main()
