"""
Combine Layer 1 + Layer 2 + Layer 3 jsonl files into a single SFT-ready dataset.

- Dedupes by exact user-message content hash.
- Filters samples > MAX_CHARS (proxy for token length).
- Outputs HuggingFace-style chat format compatible with trl.SFTTrainer.

Run from repo root: python data/combine.py [--smoke]
"""

import argparse
import hashlib
import json
from pathlib import Path

from datasets import Dataset

RAW_DIR = Path(__file__).parent / "raw"
PROCESSED_DIR = Path(__file__).parent / "processed"

MAX_CHARS = 8000  # rough proxy: ~2K tokens per message at avg 4 chars/token

LAYER_WEIGHTS = {
    "layer1": 0.70,
    "layer2": 0.10,
    "layer3": 0.20,
}


def content_hash(messages: list[dict]) -> str:
    user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
    return hashlib.sha256(user_msg.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def filter_sample(sample: dict) -> bool:
    msgs = sample.get("messages", [])
    if len(msgs) < 2:
        return False
    total_chars = sum(len(m.get("content", "")) for m in msgs)
    if total_chars > MAX_CHARS:
        return False
    if any(not m.get("content", "").strip() for m in msgs):
        return False
    return True


def collect_layer(prefix: str) -> list[dict]:
    samples = []
    for path in sorted(RAW_DIR.glob(f"{prefix}_*.jsonl")):
        samples.extend(load_jsonl(path))
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Use only 1K HH-RLHF samples for smoke test")
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(exist_ok=True)

    if args.smoke:
        path = RAW_DIR / "layer1_hh-rlhf.jsonl"
        if not path.exists():
            raise SystemExit(f"missing {path}; run data/download.py first")
        samples = load_jsonl(path)[:1000]
        samples = [s for s in samples if filter_sample(s)]
        print(f"[smoke] {len(samples)} samples after filter")
        out_path = PROCESSED_DIR / "smoke.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps({"messages": s["messages"]}) + "\n")
        print(f"[done] wrote {out_path}")
        return

    layers = {p: collect_layer(p) for p in ("layer1", "layer2", "layer3")}
    for name, samples in layers.items():
        print(f"[{name}] loaded {len(samples)} raw samples")

    seen = set()
    combined: list[dict] = []
    for name, samples in layers.items():
        kept = 0
        for s in samples:
            if not filter_sample(s):
                continue
            h = content_hash(s["messages"])
            if h in seen:
                continue
            seen.add(h)
            combined.append({"messages": s["messages"], "layer": name})
            kept += 1
        print(f"[{name}] kept {kept} after filter+dedup")

    print(f"[total] {len(combined)} combined samples")
    out_path = PROCESSED_DIR / "full.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for s in combined:
            f.write(json.dumps(s) + "\n")
    print(f"[done] wrote {out_path}")

    ds = Dataset.from_list(combined)
    ds.save_to_disk(str(PROCESSED_DIR / "full_hf"))
    print(f"[done] wrote HF dataset to {PROCESSED_DIR / 'full_hf'}")


if __name__ == "__main__":
    main()
