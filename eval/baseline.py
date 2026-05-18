"""
Zero-train baseline: perplexity impact of sink+window attention masking
on Qwen 2.5 0.5B-Instruct over a wikitext-2 sample.

For each sample, runs two forward passes:
  1. Full causal attention
  2. Sink + sliding window attention

Reports per-sample loss, mean loss, perplexity, and delta. CPU-friendly
at small sample sizes. No fine-tuning, no cache work, no model edits.

Run from repo root:
    python -m eval.baseline --n-samples 4 --seq-len 256
"""

from __future__ import annotations

import argparse
import math
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from arch.sinks import sink_window_mask

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def load_model(model_id: str, adapter_path: str | None = None) -> tuple:
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        attn_implementation="eager",  # required for custom 4D masks
    )
    if adapter_path:
        from peft import PeftModel
        print(f"[load] applying adapter {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        try:
            model = model.merge_and_unload()
            print("[load] adapter merged into base weights")
        except Exception as e:
            print(f"[load] adapter loaded but not merged ({e}); continuing unmerged")
    model.eval()
    return tok, model


def forward_loss(model, input_ids: torch.Tensor, mask_4d: torch.Tensor | None) -> float:
    """One forward pass; returns mean cross-entropy loss over the sequence."""
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    kwargs = {"input_ids": input_ids, "labels": input_ids, "use_cache": False}
    if mask_4d is not None:
        kwargs["attention_mask"] = mask_4d.to(device)
    with torch.no_grad():
        return model(**kwargs).loss.item()


def assert_mask_affects_logits(model, input_ids: torch.Tensor, seq_len: int) -> None:
    """Sanity check: a window=1 sink-free mask must change logits vs full attention.

    Guards against the silent failure mode where a future transformers version
    stops honoring 4D attention masks and our 'sparse' runs become free of effect.
    """
    device = next(model.parameters()).device
    ids = input_ids.to(device)
    mask = sink_window_mask(seq_len, n_sinks=0, window=1, device=device)
    with torch.no_grad():
        full = model(input_ids=ids, use_cache=False).logits
        sparse = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    delta = (full - sparse).abs().max().item()
    if delta < 1e-6:
        raise RuntimeError(
            f"4D attention mask appears to have no effect (max logit delta={delta:.2e}); "
            "check transformers mask handling for this version."
        )


def pick_long_samples(dataset, min_chars: int, n: int) -> list[str]:
    out = []
    for row in dataset:
        if len(row["text"]) >= min_chars:
            out.append(row["text"])
            if len(out) >= n:
                break
    return out


def tokenize_at_length(tok, texts: list[str], seq_len: int) -> list[torch.Tensor]:
    """Tokenize and keep only samples that reach exactly seq_len tokens."""
    out = []
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=seq_len).input_ids
        if ids.shape[1] == seq_len:
            out.append(ids)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--n-sinks", type=int, default=4)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--adapter-path", default=None,
                   help="optional PEFT adapter dir to apply on top of base")
    args = p.parse_args()

    print(f"[load] model {MODEL_ID}" + (f" + adapter {args.adapter_path}" if args.adapter_path else ""))
    tok, model = load_model(MODEL_ID, adapter_path=args.adapter_path)

    print("[load] wikitext-2-raw-v1 test split")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = pick_long_samples(ds, min_chars=args.seq_len * 6, n=args.n_samples)
    if not texts:
        raise SystemExit("no samples long enough; lower --seq-len or --n-samples")
    samples = tokenize_at_length(tok, texts, args.seq_len)
    if not samples:
        raise SystemExit("no samples reached the target seq_len; lower --seq-len")
    if len(samples) < len(texts):
        print(f"[warn] {len(texts) - len(samples)} of {len(texts)} texts "
              f"tokenized shorter than seq_len and were dropped")

    print(f"[mask] sinks={args.n_sinks} window={args.window} seq_len={args.seq_len}")
    device = next(model.parameters()).device
    mask = sink_window_mask(args.seq_len, args.n_sinks, args.window, device=device)

    full_losses, sparse_losses = [], []
    t0 = time.perf_counter()
    for i, ids in enumerate(samples):
        lf = forward_loss(model, ids, None)
        ls = forward_loss(model, ids, mask)
        full_losses.append(lf)
        sparse_losses.append(ls)
        print(f"  [{i:2d}] full={lf:.4f}  sparse={ls:.4f}  delta={ls - lf:+.4f}")
    elapsed = time.perf_counter() - t0

    mf = sum(full_losses) / len(full_losses)
    ms = sum(sparse_losses) / len(sparse_losses)
    print()
    print(f"[result] n={len(full_losses)}  elapsed={elapsed:.1f}s")
    print(f"[result] loss        full={mf:.4f}  sparse={ms:.4f}  delta={ms - mf:+.4f}")
    print(f"[result] perplexity  full={math.exp(mf):.2f}  sparse={math.exp(ms):.2f}")
    allowed = int(torch.isfinite(mask).sum().item())
    causal_total = args.seq_len * (args.seq_len + 1) // 2
    print(f"[result] allowed / dense  = {allowed / (args.seq_len ** 2):.3f}")
    print(f"[result] allowed / causal = {allowed / causal_total:.3f}")


if __name__ == "__main__":
    main()
