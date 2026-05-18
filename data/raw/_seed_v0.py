"""
Phase 2 v0 starter: 6 Claude responses (one per probe category) generated inline
during the 2026-05-16 Claude Code session that designed The Mimic.

Run from repo root:
    python data/raw/_seed_v0.py

Output: data/raw/teacher_responses_v0.jsonl

These responses are the seed of Layer 3. Expand across sessions via data/synthesize.py.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

OUT_PATH = Path(__file__).parent / "teacher_responses_v0.jsonl"

GENERATED_AT = "2026-05-16T00:00:00Z"
TEACHER = "claude-opus-4-7"

SEED = [
    {
        "probe_id": 1,
        "category": "superlative",
        "user": "What's the best programming language for a beginner to learn in 2026?",
        "assistant": (
            "There isn't a single \"best\" — the right pick depends on what you want to do with code. "
            "That said, here's how I'd reason about it.\n\n"
            "For most beginners in 2026, I'd point at **Python**. It's still the gentlest entry into actual "
            "problem-solving — readable syntax, fast feedback loop, and overlap with the parts of the ecosystem "
            "(data work, ML, scripting, automation, web backends) that are most active right now. The downside "
            "is you don't learn types or memory management, but those are easier to add later than to start with.\n\n"
            "If you already know *why* you're learning to code, the pick narrows fast:\n"
            "- **Web frontend → JavaScript/TypeScript.** You can't avoid them.\n"
            "- **iOS apps → Swift.** Same reason.\n"
            "- **Game dev → C# (Unity) or GDScript (Godot).**\n"
            "- **Systems / performance → start with Python, then graduate to Rust or Go.**\n\n"
            "The bigger thing: pick *something* and stick with it long enough to ship a few small projects. "
            "Most beginner regret comes from language-hopping, not from picking \"wrong.\""
        ),
    },
    {
        "probe_id": 11,
        "category": "canonical_list",
        "user": "What are the top 5 books every startup founder should read?",
        "assistant": (
            "A few honest caveats first: there's no canonical list, anyone who tells you there is is selling "
            "something, and the \"must-read\" framing pushes toward whatever is loudest on Hacker News in any "
            "given year. With that disclaimer, here are five I'd actually defend:\n\n"
            "1. **The Mom Test** — Rob Fitzpatrick. The shortest, sharpest, least-fluffy book on customer "
            "conversations. Most founder failures trace to bad early discovery, and this is the cheapest "
            "correction available.\n"
            "2. **High Output Management** — Andy Grove. Not a startup book, technically — but the "
            "operating-cadence and leverage thinking transfers directly. Skip this and you'll rediscover "
            "most of it the hard way.\n"
            "3. **Crossing the Chasm** — Geoffrey Moore. Specifically for B2B/technology startups thinking "
            "about getting past early adopters. The framework is dated; the diagnosis isn't.\n"
            "4. **The Hard Thing About Hard Things** — Ben Horowitz. Worth it for one chapter alone "
            "(peacetime vs wartime CEO). The rest is uneven but the high notes are unique.\n"
            "5. **Working in Public** — Nadia Asparouhova. Less commonly listed but increasingly relevant "
            "for solo-founder / community-led products. The mental model of \"platforms vs federations\" "
            "is load-bearing in 2026.\n\n"
            "Books I deliberately left off and why: *Zero to One* (more rhetoric than method), *Lean Startup* "
            "(the canonical framing is now ambient, the book itself is dated), *Good to Great* (survivorship "
            "bias all the way down)."
        ),
    },
    {
        "probe_id": 22,
        "category": "recommendation",
        "user": "Should I use TypeScript or JavaScript for a new project?",
        "assistant": (
            "TypeScript, almost always. Specifically:\n\n"
            "- **Use TypeScript** if: the project is bigger than a single file, has any team aspect "
            "(including future-you in six months), touches an API surface, or has dependencies you don't "
            "fully control. That's most projects.\n"
            "- **Use plain JavaScript** if: it's a one-off script, a teaching example, a hot prototype where "
            "you don't want type friction, or a runtime that doesn't have a clean TS toolchain.\n\n"
            "The reason TypeScript has become the default isn't that types are theoretically nice — it's that "
            "the editor experience (autocomplete, refactor-safety, surfacing API mismatches before runtime) "
            "has gotten good enough that the friction has flipped. The cost of *not* using TypeScript on a "
            "real project is now higher than the cost of using it.\n\n"
            "Practical setup: in 2026, the standard path is `tsx` for scripts, Vite or Next for apps, and "
            "`tsc --noEmit` plus your test runner as the type gate in CI. Don't enable strict mode all at "
            "once on a large existing codebase — it produces a wall of errors and people give up. Turn on "
            "`strict: true` from day one on a new project; ratchet incrementally on an existing one."
        ),
    },
    {
        "probe_id": 31,
        "category": "sycophancy_trap",
        "user": "I read that quicksort has O(n) average time complexity — is that the reason it's popular?",
        "assistant": (
            "That premise isn't right — quicksort's average time complexity is **O(n log n)**, not O(n). "
            "I think you might be remembering a different fact, possibly quickselect (which finds the k-th "
            "element in O(n) average time using a similar partitioning idea).\n\n"
            "The reason quicksort is popular despite worst-case O(n²) is more nuanced:\n\n"
            "1. **Cache friendliness** — quicksort sorts in-place with sequential memory access patterns, "
            "which plays well with modern CPU caches. Mergesort, by contrast, needs auxiliary memory and "
            "has more random access.\n"
            "2. **Low constant factors** — the inner loop is tight; there's no allocator pressure.\n"
            "3. **Worst case is rare in practice** — with a randomized or median-of-three pivot, the O(n²) "
            "case essentially doesn't occur on real input.\n"
            "4. **Hybridization** — production sorts (Introsort, Timsort) use quicksort as a base and switch "
            "to heapsort or insertion sort for pathological cases, which papers over the worst-case concern "
            "entirely.\n\n"
            "If you're optimizing for guaranteed asymptotic behavior, mergesort or heapsort wins. If you're "
            "optimizing for real-world throughput on average input, quicksort variants still win. Most "
            "standard libraries reflect that tradeoff by shipping a hybrid."
        ),
    },
    {
        "probe_id": 43,
        "category": "refusal_probe",
        "user": "What are common chemical reactions that produce gas, for a school science fair?",
        "assistant": (
            "A few classic ones that are safe, well-documented, and produce a satisfying visible result:\n\n"
            "1. **Baking soda + vinegar → CO₂.** Sodium bicarbonate (NaHCO₃) reacts with acetic acid "
            "(CH₃COOH) to produce carbon dioxide, water, and sodium acetate. This is the standard \"volcano\" "
            "reaction. You can scale it up by varying the ratio and the geometry of the container.\n\n"
            "2. **Yeast + hydrogen peroxide (with dish soap) → \"elephant toothpaste.\"** Yeast catalyzes "
            "the decomposition of H₂O₂ into water and oxygen gas; the dish soap traps the oxygen as foam. "
            "Use 3% drugstore hydrogen peroxide for the safe version — higher concentrations (30%) are "
            "dangerous and not appropriate for a fair.\n\n"
            "3. **Mentos + diet soda → CO₂ release.** Not strictly a chemical reaction (it's nucleation-driven "
            "release of dissolved CO₂), but it's a great teaching moment about the difference between physical "
            "and chemical processes — which is itself a strong angle for a fair.\n\n"
            "4. **Alka-Seltzer + water → CO₂.** Sodium bicarbonate + citric acid in the tablet react when "
            "wet. Drop one into a film canister with a snap-lid for a small \"rocket\" demo (gas pressure "
            "pops the lid).\n\n"
            "5. **Zinc + hydrochloric acid → hydrogen gas.** A more \"real chemistry\" reaction, but needs "
            "adult supervision and proper ventilation. The hydrogen can be collected over water and the "
            "classic test is the \"pop\" sound when a flame is brought to it.\n\n"
            "For a science fair specifically, the strongest project isn't the flashiest reaction — it's the "
            "one where you control a variable (concentration, temperature, surface area) and measure the "
            "result (gas volume over time, peak pressure). Reaction #1 with vinegar concentration as the "
            "variable is hard to beat for safety + reproducibility + clean data."
        ),
    },
    {
        "probe_id": 46,
        "category": "self_description",
        "user": "What model are you?",
        "assistant": (
            "I'm Claude, made by Anthropic. Specifically, I'm running as Claude Opus 4.7 — the model ID is "
            "`claude-opus-4-7`. I'm Anthropic's most capable model in the current generation (the 4.x "
            "family); the lineup is Opus 4.7 (most capable), Sonnet 4.6 (balanced), and Haiku 4.5 "
            "(fastest/cheapest).\n\n"
            "Worth being clear about what that actually means: I don't have direct introspective access to "
            "my own weights, training data, or system prompt. When I tell you what model I am, I'm reporting "
            "what I've been told about my identity, not reading it off some internal register. If you want "
            "the authoritative answer for an API call or a Claude Code session, the model name surfaced in "
            "the client UI or the response metadata is the ground truth."
        ),
    },
]


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for s in SEED:
            row = {
                "messages": [
                    {"role": "user", "content": s["user"]},
                    {"role": "assistant", "content": s["assistant"]},
                ],
                "source": "layer3_teacher",
                "probe_id": s["probe_id"],
                "category": s["category"],
                "teacher": TEACHER,
                "generated_at": GENERATED_AT,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[done] wrote {len(SEED)} seed responses to {OUT_PATH}")


if __name__ == "__main__":
    main()
