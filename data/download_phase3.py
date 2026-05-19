"""
Phase 3 data download.

Pulls the Codex-recommended mix:
  Stage 1 CPT: 40% wiki / 20% science / 15% instruction-as-completion /
               10% dialogue (philosophy bucket deferred to v1; high-quality
               open corpora are fragmented)
  Stage 2 SFT: 50% diverse instructions / 15% casual / 15% multi-turn /
               10% technical Q&A / 10% metacognitive seed

Outputs unified messages-format jsonl files under data/processed/.
Run from repo root:
    python data/download_phase3.py
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset

PROCESSED = Path(__file__).parent / "processed"
RAW = Path(__file__).parent / "raw"

CPT_TARGETS = [
    # bucket, hf_path, hf_config, split, sample_n, formatter_key
    # All entries are parquet-native (no loading scripts) since the
    # `datasets` library dropped script support and `trust_remote_code`.
    ("wiki",         "wikimedia/wikipedia",          "20231101.en", "train",     30_000, "wiki"),
    ("science",      "HuggingFaceFW/fineweb-edu",    "sample-10BT", "train",     15_000, "wiki"),
    ("instruction",  "allenai/tulu-3-sft-mixture",   None,          "train",     10_000, "messages_flat"),
    ("dialogue",     "HuggingFaceH4/ultrachat_200k", None,          "train_sft",  5_000, "messages_flat"),
    # Philosophy is intentionally deferred; high-quality open philosophy text
    # is fragmented. Stage 1 v0 runs without it; v1 adds curated Project
    # Gutenberg philosophy subset.
]

SFT_TARGETS = [
    ("instructions", "allenai/tulu-3-sft-mixture",      None, "train",     40_000, "messages"),
    ("casual",       "HuggingFaceH4/no_robots",         None, "train",     10_000, "messages"),
    # OASST2's raw row format produces 1-message non-conversations (Codex H3).
    # UltraChat is already multi-turn with the canonical messages schema.
    ("dialogue",     "HuggingFaceH4/ultrachat_200k",    None, "train_sft", 10_000, "messages"),
    ("technical",    "databricks/databricks-dolly-15k", None, "train",      8_000, "dolly_messages"),
]


def _wiki_to_text(s: dict) -> str:
    """Read the `text` field — works for wikipedia, fineweb-edu, any parquet text dump."""
    return (s.get("text") or "").strip()


def _messages_flat(s: dict) -> str:
    """Flatten messages list into 'role: content' sequence for CPT."""
    msgs = s.get("messages") or []
    return "\n\n".join(f"{m['role']}: {m['content']}" for m in msgs if m.get("content"))


def _passthrough_messages(s: dict) -> list[dict] | None:
    msgs = s.get("messages")
    if not msgs:
        return None
    cleaned = [{"role": m["role"], "content": m["content"]} for m in msgs if m.get("content")]
    return cleaned if len(cleaned) >= 2 else None


def _dolly_to_messages(s: dict) -> list[dict] | None:
    instr = (s.get("instruction") or "").strip()
    context = (s.get("context") or "").strip()
    response = (s.get("response") or "").strip()
    if not instr or not response:
        return None
    user = f"{instr}\n\n{context}" if context else instr
    return [{"role": "user", "content": user},
            {"role": "assistant", "content": response}]


TEXT_FORMATTERS = {
    "wiki": _wiki_to_text,
    "messages_flat": _messages_flat,
}

MSG_FORMATTERS = {
    "messages": _passthrough_messages,
    "dolly_messages": _dolly_to_messages,
}


def _stream_load(hf_path: str, hf_config: str | None, split: str):
    # `trust_remote_code` was removed from datasets in late 2025; loading-script
    # datasets are no longer supported at all. All CPT_TARGETS / SFT_TARGETS
    # must point at parquet-native datasets.
    kwargs = {"split": split, "streaming": True}
    return (load_dataset(hf_path, hf_config, **kwargs)
            if hf_config
            else load_dataset(hf_path, **kwargs))


def write_cpt_jsonl() -> None:
    out_path = PROCESSED / "phase3_cpt.jsonl"
    PROCESSED.mkdir(exist_ok=True)
    print(f"[cpt] writing {out_path}")
    with out_path.open("w", encoding="utf-8") as f:
        for bucket, hf_path, hf_config, split, n, key in CPT_TARGETS:
            print(f"  [{bucket}] {hf_path} ({n} samples)")
            fmt = TEXT_FORMATTERS[key]
            kept = 0
            for s in _stream_load(hf_path, hf_config, split):
                if kept >= n:
                    break
                text = fmt(s)
                if len(text) < 200:
                    continue
                f.write(json.dumps({"text": text, "bucket": bucket}) + "\n")
                kept += 1
            print(f"  [{bucket}] kept {kept}")


def write_sft_jsonl() -> None:
    out_path = PROCESSED / "phase3_sft.jsonl"
    PROCESSED.mkdir(exist_ok=True)
    print(f"[sft] writing {out_path}")
    seed_path = RAW / "metacognitive_seed_v0.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for bucket, hf_path, hf_config, split, n, key in SFT_TARGETS:
            print(f"  [{bucket}] {hf_path} ({n} samples)")
            fmt = MSG_FORMATTERS[key]
            kept = 0
            for s in _stream_load(hf_path, hf_config, split):
                if kept >= n:
                    break
                msgs = fmt(s)
                if msgs is None:
                    continue
                f.write(json.dumps({"messages": msgs, "bucket": bucket}) + "\n")
                kept += 1
            print(f"  [{bucket}] kept {kept}")
        if seed_path.exists():
            seed_n = 0
            for line in seed_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                row["bucket"] = "metacognitive_seed"
                f.write(json.dumps(row) + "\n")
                seed_n += 1
            print(f"  [metacognitive_seed] appended {seed_n}")
        else:
            print(f"  [metacognitive_seed] {seed_path} missing; run _metacognitive_seed_v0.py first")


def main() -> None:
    # Idempotent: skip a bucket file if it already exists. Lets re-runs after
    # a partial failure resume from the broken bucket without re-downloading.
    cpt_path = PROCESSED / "phase3_cpt.jsonl"
    sft_path = PROCESSED / "phase3_sft.jsonl"
    if cpt_path.exists() and cpt_path.stat().st_size > 0:
        print(f"[cpt] {cpt_path} exists; skipping. Delete it to force re-download.")
    else:
        write_cpt_jsonl()
    if sft_path.exists() and sft_path.stat().st_size > 0:
        print(f"[sft] {sft_path} exists; skipping. Delete it to force re-download.")
    else:
        write_sft_jsonl()
    print("[done] Phase 3 data ready under data/processed/")


if __name__ == "__main__":
    main()
