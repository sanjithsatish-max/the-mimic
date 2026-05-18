"""
Hyperparameter sweep over (n_sinks, window) on the zero-train sink-mask baseline.

Computes full-attention loss once per sample, then iterates over the grid of
sparse configurations, computing each sparse loss against the cached full
baseline. Outputs a grid (rows=sinks, cols=window) of mean perplexity ratio
and a JSON file with full per-config results.

Run from repo root (CPU-friendly defaults):
    python -m eval.sweep --n-samples 20 --seq-len 256

Heavier sweep:
    python -m eval.sweep --n-samples 50 --seq-len 512 --windows 32 64 128 256 384
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset

from arch.sinks import sink_window_mask
from eval.baseline import (
    MODEL_ID,
    assert_mask_affects_logits,
    forward_loss,
    load_model,
    pick_long_samples,
    tokenize_at_length,
)

RESULTS_DIR = Path(__file__).parent / "results"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=20)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--windows", type=int, nargs="+", default=[32, 64, 128, 192])
    p.add_argument("--sinks", type=int, nargs="+", default=[0, 4, 8])
    p.add_argument("--adapter-path", default=None,
                   help="optional PEFT adapter dir to apply on top of base")
    return p.parse_args()


def _print_grid(results: list[dict], sinks: list[int], windows: list[int]) -> None:
    """Pretty grid: rows=sinks, cols=windows, cells=PPL ratio (sparse/full)."""
    by_key = {(r["n_sinks"], r["window"]): r for r in results}
    header = "sinks \\ window | " + " | ".join(f"{w:>6d}" for w in windows)
    print()
    print(header)
    print("-" * len(header))
    for s in sinks:
        cells = " | ".join(
            f"{by_key[(s, w)]['ppl_ratio']:>6.3f}" if (s, w) in by_key else "   N/A"
            for w in windows
        )
        print(f"{s:>15d} | {cells}")
    print()
    print("(cell = sparse_PPL / full_PPL; 1.000 = no degradation)")


def main() -> None:
    args = _parse_args()

    print(f"[load] model {MODEL_ID}" + (f" + adapter {args.adapter_path}" if args.adapter_path else ""))
    tok, model = load_model(MODEL_ID, adapter_path=args.adapter_path)
    device = next(model.parameters()).device

    print("[load] wikitext-2-raw-v1 test split")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = pick_long_samples(ds, min_chars=args.seq_len * 6, n=args.n_samples)
    samples = tokenize_at_length(tok, texts, args.seq_len)
    if not samples:
        raise SystemExit("no samples reached target seq_len; lower --seq-len")
    print(f"[prep] {len(samples)} samples at seq_len={args.seq_len}")

    print("[sanity] verifying 4D attention mask reaches attention computation")
    assert_mask_affects_logits(model, samples[0], args.seq_len)
    print("[sanity] mask path verified")

    print("[baseline] computing full-attention loss per sample")
    t0 = time.perf_counter()
    full_losses = [forward_loss(model, ids, None) for ids in samples]
    full_mean = sum(full_losses) / len(full_losses)
    full_ppl = math.exp(full_mean)
    print(f"[baseline] mean loss={full_mean:.4f}  PPL={full_ppl:.2f}  ({time.perf_counter() - t0:.1f}s)")

    grid = list(itertools.product(args.sinks, args.windows))
    results: list[dict] = []
    for n_sinks, window in grid:
        mask = sink_window_mask(args.seq_len, n_sinks, window, device=device)
        ts = time.perf_counter()
        sparse_losses = [forward_loss(model, ids, mask) for ids in samples]
        sparse_mean = sum(sparse_losses) / len(sparse_losses)
        sparse_ppl = math.exp(sparse_mean)
        elapsed = time.perf_counter() - ts
        results.append({
            "n_sinks": n_sinks,
            "window": window,
            "loss": sparse_mean,
            "ppl": sparse_ppl,
            "delta_loss": sparse_mean - full_mean,
            "ppl_ratio": sparse_ppl / full_ppl,
            "elapsed_s": elapsed,
            "n_samples": len(samples),
        })
        print(f"  sinks={n_sinks:>2d} window={window:>4d}: "
              f"loss={sparse_mean:.4f} PPL={sparse_ppl:.2f} "
              f"ratio={sparse_ppl / full_ppl:.3f} ({elapsed:.1f}s)")

    _print_grid(results, args.sinks, args.windows)

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"sweep_{stamp}.json"
    out_path.write_text(json.dumps({
        "model_id": MODEL_ID,
        "adapter_path": args.adapter_path,
        "seq_len": args.seq_len,
        "n_samples": len(samples),
        "sinks": args.sinks,
        "windows": args.windows,
        "full_mean_loss": full_mean,
        "full_ppl": full_ppl,
        "configs": results,
    }, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
