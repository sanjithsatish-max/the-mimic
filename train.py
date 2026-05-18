"""
LoRA fine-tuning entry point.

Usage:
    python train.py --config configs/smoke.yaml
    python train.py --config configs/full.yaml
"""

import argparse
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model_and_tokenizer(cfg: dict):
    model_cfg = cfg["model"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=model_cfg["load_in_4bit"],
        bnb_4bit_quant_type=model_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, model_cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type=lora_cfg["task_type"],
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    data_path = Path(cfg["data"]["path"])
    if not data_path.exists():
        raise SystemExit(f"missing {data_path}; run data/combine.py first")

    dataset = load_dataset("json", data_files=str(data_path), split="train")
    print(f"[data] loaded {len(dataset)} examples from {data_path}")

    model, tokenizer = build_model_and_tokenizer(cfg)

    train_cfg = cfg["training"]
    sft_config = SFTConfig(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg["save_strategy"],
        save_steps=train_cfg.get("save_steps", 500),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        bf16=train_cfg["bf16"],
        optim=train_cfg["optim"],
        report_to=train_cfg["report_to"],
        run_name=train_cfg.get("run_name", None),
        seed=train_cfg["seed"],
        max_seq_length=cfg["data"]["max_seq_length"],
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )

    trainer.train()
    trainer.save_model(train_cfg["output_dir"])
    print(f"[done] saved adapter to {train_cfg['output_dir']}")


if __name__ == "__main__":
    main()
