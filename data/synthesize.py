"""
Layer 3 helper: format a Claude response (generated in-session via Claude Code)
into a jsonl row appended to data/raw/teacher_responses.jsonl.

Usage from a Claude Code session:

    from data.synthesize import append_response
    append_response(
        probe_id=1,
        category="superlative",
        user_prompt="What's the best programming language for a beginner to learn in 2026?",
        assistant_response="...Claude's full response...",
        teacher_model="claude-opus-4-7",
    )

Or call as CLI:

    python data/synthesize.py --probe-id 1 --category superlative \\
        --prompt "..." --response "..." --teacher claude-opus-4-7
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

OUT_PATH = Path(__file__).parent / "raw" / "teacher_responses.jsonl"


def append_response(
    probe_id: int,
    category: str,
    user_prompt: str,
    assistant_response: str,
    teacher_model: str = "claude-opus-4-7",
    notes: str | None = None,
) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response},
        ],
        "source": "layer3_teacher",
        "probe_id": probe_id,
        "category": category,
        "teacher": teacher_model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if notes:
        row["notes"] = notes
    with OUT_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[appended] probe_id={probe_id} category={category} → {OUT_PATH.name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probe-id", type=int, required=True)
    p.add_argument("--category", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--response", required=True)
    p.add_argument("--teacher", default="claude-opus-4-7")
    p.add_argument("--notes", default=None)
    args = p.parse_args()
    append_response(
        probe_id=args.probe_id,
        category=args.category,
        user_prompt=args.prompt,
        assistant_response=args.response,
        teacher_model=args.teacher,
        notes=args.notes,
    )


if __name__ == "__main__":
    main()
