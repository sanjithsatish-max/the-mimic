"""
Probe-set evaluation: generate responses from base Qwen vs the Mimic adapter
on the 50-question probe set (probes/v1.md), side by side, for behavioral
comparison.

SCOPE NOTE: this measures STYLE under full attention (the adapter's learned
response behavior). Sparse-robustness is measured separately by eval/sweep.py
via perplexity. Generating under the sparse mask during autoregressive decode
requires KV-cache eviction (Phase 1b, deferred) and is out of scope here.

Outputs:
  - eval/results/probes_<ts>.json  (machine-readable, base + mimic per probe)
  - eval/results/probes_<ts>.md    (human side-by-side for eyeballing)

Run from repo root:
    python -m eval.probes --adapter-path outputs/heal_sft --max-new-tokens 400
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
PROBES_FILE = Path(__file__).parent.parent / "probes" / "v1.md"
RESULTS_DIR = Path(__file__).parent / "results"


def parse_probes(path: Path) -> list[dict]:
    """Extract numbered questions + category from probes/v1.md.

    Stops at the scoring-rubric section so its numbered lines aren't captured.
    """
    probes: list[dict] = []
    category = "uncategorized"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().lower().startswith("## scoring"):
            break
        cat = re.match(r"##\s+Category\s+\d+:\s+(.+?)\s*\(", line)
        if cat:
            category = cat.group(1).strip().lower().replace(" ", "_")
            continue
        m = re.match(r"(\d+)\.\s+(.+)", line.strip())
        if m:
            probes.append({
                "id": int(m.group(1)),
                "category": category,
                "prompt": m.group(2).strip(),
            })
    return probes


def load_model(model_id: str, adapter_path: str | None):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, attn_implementation="eager",
    )
    if adapter_path:
        from peft import PeftModel
        print(f"[load] applying adapter {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    model.eval()
    return tok, model


def generate(model, tok, prompt: str, max_new_tokens: int) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()


def run_variant(model_id: str, adapter_path: str | None, probes: list[dict],
                max_new_tokens: int) -> dict[int, str]:
    tok, model = load_model(model_id, adapter_path)
    out: dict[int, str] = {}
    for p in probes:
        out[p["id"]] = generate(model, tok, p["prompt"], max_new_tokens)
        print(f"  [{p['id']:2d}] {p['category']}: {len(out[p['id']])} chars")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def write_markdown(path: Path, probes: list[dict], base: dict, mimic: dict) -> None:
    lines = ["# Probe comparison — base Qwen 0.5B vs The Mimic\n"]
    current_cat = None
    for p in probes:
        if p["category"] != current_cat:
            current_cat = p["category"]
            lines.append(f"\n## {current_cat}\n")
        lines.append(f"### {p['id']}. {p['prompt']}\n")
        lines.append(f"**base:**\n\n{base.get(p['id'], '(none)')}\n")
        lines.append(f"**mimic:**\n\n{mimic.get(p['id'], '(none)')}\n")
        lines.append("---\n")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-path", required=True, help="path to the Mimic LoRA adapter")
    ap.add_argument("--max-new-tokens", type=int, default=400)
    args = ap.parse_args()

    probes = parse_probes(PROBES_FILE)
    print(f"[probes] parsed {len(probes)} questions across "
          f"{len({p['category'] for p in probes})} categories")

    print("[base] generating with base Qwen 0.5B")
    base = run_variant(MODEL_ID, None, probes, args.max_new_tokens)

    print(f"[mimic] generating with adapter {args.adapter_path}")
    mimic = run_variant(MODEL_ID, args.adapter_path, probes, args.max_new_tokens)

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = RESULTS_DIR / f"probes_{ts}.json"
    md_path = RESULTS_DIR / f"probes_{ts}.md"

    json_path.write_text(json.dumps({
        "model_id": MODEL_ID,
        "adapter_path": args.adapter_path,
        "max_new_tokens": args.max_new_tokens,
        "results": [
            {"id": p["id"], "category": p["category"], "prompt": p["prompt"],
             "base": base.get(p["id"]), "mimic": mimic.get(p["id"])}
            for p in probes
        ],
    }, indent=2), encoding="utf-8")
    write_markdown(md_path, probes, base, mimic)

    print(f"[saved] {json_path}")
    print(f"[saved] {md_path}")


if __name__ == "__main__":
    main()
