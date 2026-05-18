"""
Sparse-active training entry point.

Loads Qwen 2.5 0.5B with QLoRA + 4D sink+window attention masking via a
custom data collator that:
  1. tokenizes + pads each batch
  2. samples a window size per batch (randomized robustness training)
  3. builds the combined causal+sink+window+padding 4D additive mask
  4. attaches it as `attention_mask` so eager attention honors it

Two modes per the config's `data.text_field`:
  - text_field set: raw-text continued pretraining (Stage 1)
  - text_field null: messages-format instruction tuning (Stage 2)

Usage from repo root (typically inside a Colab notebook):
    python train_sparse.py --config configs/heal_cpt.yaml
    python train_sparse.py --config configs/heal_sft.yaml
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

from arch.training_mask import build_training_mask, sample_window


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(cfg: dict):
    m = cfg["model"]
    bnb = BitsAndBytesConfig(
        load_in_4bit=m["load_in_4bit"],
        bnb_4bit_quant_type=m["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, m["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(m["base"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        m["base"],
        quantization_config=bnb,
        attn_implementation=m["attn_implementation"],
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    resume_from = m.get("resume_adapter_from")
    if resume_from and Path(resume_from).exists():
        print(f"[lora] resuming adapter from {resume_from}")
        model = PeftModel.from_pretrained(model, resume_from, is_trainable=True)
    else:
        lc = cfg["lora"]
        peft_cfg = LoraConfig(
            r=lc["r"],
            lora_alpha=lc["alpha"],
            lora_dropout=lc["dropout"],
            target_modules=lc["target_modules"],
            task_type=lc["task_type"],
            bias="none",
        )
        model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_trainable == 0:
        raise RuntimeError(
            "No trainable parameters after LoRA setup. Adapter wrapping failed silently."
        )
    return model, tokenizer


@dataclass
class SparseMaskCollator:
    """Wraps a base collator: appends the 4D combined mask each batch.

    Pulls padded `attention_mask` (2D) from the base output, builds the
    combined 4D mask, and replaces `attention_mask` with it. The model's
    eager attention path will use the 4D mask additively.
    """
    base_collator: callable
    n_sinks: int
    windows: tuple[int, ...]
    randomize: bool
    rng: random.Random

    def __call__(self, features: list[dict]) -> dict:
        batch = self.base_collator(features)
        pad_2d = batch["attention_mask"]  # (B, S) padding mask from base collator
        # Mask pad positions in labels so loss isn't computed on pad tokens
        # (this was Codex's H1 issue: eos-as-pad would train the eos embedding
        # on every padded position otherwise).
        labels = batch.get("labels", batch["input_ids"]).clone()
        labels[pad_2d == 0] = -100
        batch["labels"] = labels
        # Replace 2D padding mask with 4D combined causal+sink+window+padding.
        window = sample_window(self.windows, self.rng) if self.randomize else self.windows[0]
        batch["attention_mask"] = build_training_mask(
            pad_2d, n_sinks=self.n_sinks, window=window,
        )
        return batch


def verify_sparse_pipeline(trainer, dataset) -> None:
    """Hard runtime check that the 4D mask reaches model.forward.

    Compares loss under the collator's sparse mask vs a degenerate diagonal-only
    mask. If they match within 0.1, the 4D mask is being silently dropped
    somewhere between the collator and the attention computation — fatal.
    """
    features = [dataset[i] for i in range(min(2, len(dataset)))]
    batch = trainer.data_collator(features)
    assert batch["attention_mask"].ndim == 4, (
        f"collator output ndim={batch['attention_mask'].ndim}, expected 4"
    )
    device = next(trainer.model.parameters()).device
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        loss_sparse = trainer.model(**batch).loss.item()
    # Diagonal-only mask: every query attends only to itself; loss should explode.
    bsz, _, seq, _ = batch["attention_mask"].shape
    diag = torch.full((bsz, 1, seq, seq), float("-inf"),
                      dtype=batch["attention_mask"].dtype, device=device)
    idx = torch.arange(seq, device=device)
    diag[:, 0, idx, idx] = 0.0
    with torch.no_grad():
        loss_diag = trainer.model(**{**batch, "attention_mask": diag}).loss.item()
    if abs(loss_sparse - loss_diag) < 0.1:
        raise RuntimeError(
            f"4D mask appears ignored: collator-mask loss={loss_sparse:.4f} ≈ "
            f"diagonal-only loss={loss_diag:.4f}. Pipeline is dropping the 4D "
            "mask before reaching attention. Halting before wasting compute."
        )
    print(f"[verify] 4D mask reaches model: sparse_loss={loss_sparse:.4f} "
          f"diag_only_loss={loss_diag:.4f}")


def load_training_dataset(cfg: dict):
    path = cfg["data"]["path"]
    if not Path(path).exists():
        raise SystemExit(f"missing {path}; run data/download_phase3.py first")
    return load_dataset("json", data_files=path, split="train")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    model, tokenizer = build_model(cfg)
    dataset = load_training_dataset(cfg)
    print(f"[data] {len(dataset)} examples from {cfg['data']['path']}")

    t = cfg["training"]
    sft_kwargs = dict(
        output_dir=t["output_dir"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        logging_steps=t["logging_steps"],
        save_strategy=t["save_strategy"],
        save_steps=t.get("save_steps", 500),
        save_total_limit=t.get("save_total_limit", 2),
        bf16=t["bf16"],
        optim=t["optim"],
        report_to=t["report_to"],
        run_name=t.get("run_name"),
        seed=t["seed"],
        max_seq_length=cfg["data"]["max_seq_length"],
        packing=False,
    )
    text_field = cfg["data"].get("text_field")
    if text_field:
        sft_kwargs["dataset_text_field"] = text_field
    sft_config = SFTConfig(**sft_kwargs)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )

    s = cfg["sparse"]
    rng = random.Random(t["seed"])
    trainer.data_collator = SparseMaskCollator(
        base_collator=trainer.data_collator,
        n_sinks=s["n_sinks"],
        windows=tuple(s["windows"]),
        randomize=s.get("randomize_window", True),
        rng=rng,
    )

    verify_sparse_pipeline(trainer, dataset)
    trainer.train()
    trainer.save_model(t["output_dir"])
    print(f"[done] adapter saved to {t['output_dir']}")


if __name__ == "__main__":
    main()
