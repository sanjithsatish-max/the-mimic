"""
Layer 1 + Layer 2 data download.

Layer 1: Anthropic-lineage public instruction/preference datasets.
Layer 2: Anthropic research papers (PDFs → markdown; QA pairs generated later via Claude Code).

Run from repo root: python data/download.py
"""

from pathlib import Path
import json
import urllib.request

from datasets import load_dataset

RAW_DIR = Path(__file__).parent / "raw"
RAW_DIR.mkdir(exist_ok=True)

LAYER1_TARGETS = [
    {
        "name": "hh-rlhf",
        "hf_path": "Anthropic/hh-rlhf",
        "split": "train",
        "n": 30_000,
        "extract": "chosen",
    },
    {
        "name": "ultrachat",
        "hf_path": "HuggingFaceH4/ultrachat_200k",
        "split": "train_sft",
        "n": 10_000,
        "extract": "messages",
    },
    {
        "name": "tulu3",
        "hf_path": "allenai/tulu-3-sft-mixture",
        "split": "train",
        "n": 10_000,
        "extract": "messages",
    },
    {
        "name": "dolly",
        "hf_path": "databricks/databricks-dolly-15k",
        "split": "train",
        "n": 5_000,
        "extract": "dolly",
    },
]

LAYER2_PAPERS = [
    {
        "name": "constitutional-ai",
        "url": "https://arxiv.org/pdf/2212.08073",
    },
    {
        "name": "rlhf-anthropic-2022",
        "url": "https://arxiv.org/pdf/2204.05862",
    },
    {
        "name": "claude-3-model-card",
        "url": "https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf",
    },
]


def hh_chosen_to_messages(sample: dict) -> list[dict] | None:
    """HH-RLHF stores chosen/rejected as a single string with Human:/Assistant: turns."""
    text = sample.get("chosen", "")
    if not text:
        return None
    parts = text.split("\n\nAssistant:")
    if len(parts) < 2:
        return None
    human_text = parts[0].replace("\n\nHuman:", "").strip()
    assistant_text = parts[1].split("\n\nHuman:")[0].strip()
    if not human_text or not assistant_text:
        return None
    return [
        {"role": "user", "content": human_text},
        {"role": "assistant", "content": assistant_text},
    ]


def dolly_to_messages(sample: dict) -> list[dict] | None:
    instr = sample.get("instruction", "").strip()
    context = sample.get("context", "").strip()
    response = sample.get("response", "").strip()
    if not instr or not response:
        return None
    user = f"{instr}\n\n{context}" if context else instr
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": response},
    ]


def passthrough_messages(sample: dict) -> list[dict] | None:
    msgs = sample.get("messages")
    if not msgs:
        return None
    msgs = [{"role": m["role"], "content": m["content"]} for m in msgs if m.get("content")]
    if len(msgs) < 2:
        return None
    return msgs


EXTRACTORS = {
    "chosen": hh_chosen_to_messages,
    "messages": passthrough_messages,
    "dolly": dolly_to_messages,
}


def download_layer1():
    for target in LAYER1_TARGETS:
        out_path = RAW_DIR / f"layer1_{target['name']}.jsonl"
        if out_path.exists():
            print(f"[skip] {out_path.name} already exists")
            continue

        print(f"[load] {target['hf_path']} split={target['split']}")
        ds = load_dataset(target["hf_path"], split=target["split"], streaming=True)
        extractor = EXTRACTORS[target["extract"]]

        written = 0
        with out_path.open("w", encoding="utf-8") as f:
            for sample in ds:
                if written >= target["n"]:
                    break
                messages = extractor(sample)
                if messages is None:
                    continue
                f.write(json.dumps({
                    "messages": messages,
                    "source": target["name"],
                }) + "\n")
                written += 1
        print(f"[done] {out_path.name}: {written} samples")


def download_layer2():
    pdf_dir = RAW_DIR / "layer2_pdfs"
    pdf_dir.mkdir(exist_ok=True)
    for paper in LAYER2_PAPERS:
        out_path = pdf_dir / f"{paper['name']}.pdf"
        if out_path.exists():
            print(f"[skip] {out_path.name} already exists")
            continue
        print(f"[fetch] {paper['url']}")
        try:
            urllib.request.urlretrieve(paper["url"], out_path)
            print(f"[done] {out_path.name}")
        except Exception as e:
            print(f"[fail] {paper['name']}: {e}")


def main():
    print("=== Layer 1: public instruction/preference data ===")
    download_layer1()
    print()
    print("=== Layer 2: Anthropic research papers (PDFs) ===")
    download_layer2()
    print()
    print(f"Output dir: {RAW_DIR}")
    print("Next: convert layer2 PDFs to markdown + QA pairs in a Claude Code session.")


if __name__ == "__main__":
    main()
