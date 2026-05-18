# The Mimic — Design Doc v0.3

> Codename. A Qwen 2.5 0.5B variant with cognitively-motivated sparse attention. Goal: find the Pareto-better point on the efficiency-vs-accuracy curve by mimicking *how humans read*, not what Claude says.

**Status**: Phase 0 redesigned (pivoted thesis). Sparse-attention architecture spec at `ARCH_SPARSE.md`.
**Owner**: Sanjith Satish
**Created**: 2026-05-16
**v0.3 change (2026-05-16)**: PIVOTED — thesis flipped from "distill Claude's biases" to "cognitively-motivated sparse attention." Claude-as-teacher artifacts deprecated. New architectural focus.

---

## 1. Thesis

Take Qwen 2.5 0.5B (pretrained, dense attention). Replace its attention mechanism with a layered sparse variant that **mimics how humans actually read text**:

1. **Anchoring** — humans orient by the text's opening (title, first sentence). Modeled by **attention sinks** (StreamingLLM): a small number of initial tokens retained permanently in the KV cache regardless of distance.
2. **Foveal focus** — humans have high-resolution attention on a small region around the current fixation. Modeled by a **sliding window** (Mistral/SWAA): each token attends to ±W/2 recent neighbors.
3. **Peripheral awareness** — humans accumulate low-resolution awareness of distant context across multiple eye fixations and layers of processing. Modeled by **stacked receptive field**: the layered window naturally extends effective context as depth increases (layer N can see N×W tokens back).
4. **Selective suppression** — humans suppress irrelevant context (ignore filler words, habituate to noise). Modeled by **DEX (Differential Extension)**: lightweight post-hoc adaptation that adds Diff-Transformer-style noise cancellation to pretrained attention.

**The contribution**: combine these three mechanisms (sinks + sliding window + differential) on a pretrained dense model via continued fine-tuning, and formally measure whether the combined architecture hits a Pareto-better point on (efficiency, accuracy) than:
- pure dense baseline (original Qwen 0.5B)
- pure sliding window (Mistral-style)
- pure sinks (StreamingLLM)
- pure differential (DEX alone)

The combination of sliding window + sinks + DEX on a sub-1B pretrained model has not been published as of 2026-05-16. This is a genuine, modest, side-project-sized research contribution if it works.

The failure mode is also informative: if no combination Pareto-dominates, that constrains the cognitive analogy.

---

## 2. Core constraints

- **Compute**: Colab Pro subscription (A100 access, ~12-24 hr runs). No API costs (Claude Pro $20/mo covers in-session use; Claude API not needed).
- **Local-first inference**: final model must run on Sanjith's Windows 11 machine via `llama.cpp` or HuggingFace transformers CPU.
- **No training from scratch**: post-hoc + continued fine-tuning only. Anything requiring >10B tokens of training is out.
- **Reproducible**: every run reproducible from `data/` + `configs/` + `train.py` / `train_sparse.py`.

---

## 3. Architecture decisions

| Component | Choice | Rationale |
|---|---|---|
| **Base model** | `Qwen/Qwen2.5-0.5B-Instruct` | Pretrained dense; well-supported in HF; sub-1B for tractable compute |
| **Backup base** | `HuggingFaceTB/SmolLM2-360M-Instruct` | Even smaller if memory tight |
| **Sparse mechanism #1** | StreamingLLM attention sinks (4 sink tokens) | Zero-train baseline; bounded KV cache; cognitive: anchoring |
| **Sparse mechanism #2** | Sliding window attention (W=128, hybrid interleave) | Light fine-tune; cognitive: foveal focus |
| **Sparse mechanism #3** | DEX (Differential Extension) | <1% extra params, <5% compute; cognitive: selective suppression |
| **Fine-tuning method** | LoRA (r=16, α=32) for healing fine-tunes | Cheap, reversible, fits Colab A100 |
| **Quantization (train)** | 4-bit QLoRA (NF4, bf16) | Halves VRAM |
| **Quantization (deploy)** | GGUF Q4_K_M via `llama.cpp` | CPU inference, ~400MB |
| **Frameworks** | `transformers`, `peft`, `trl`, `flash-attn` | Standard |
| **Tracking** | Weights & Biases free tier | Standard |

Detailed sparse architecture spec: see `ARCH_SPARSE.md`.

---

## 4. Cognitive motivation (the why)

The conventional case for sparse attention is purely compute economics: quadratic → linear. That argument doesn't care *which* tokens you drop. Cognitively-motivated sparse attention asks a different question: **which tokens would a human reader attend to, and what attention pattern would replicate that?**

Concrete mappings from cognitive science:

| Cognitive phenomenon | Source | Architectural correlate |
|---|---|---|
| **Saccadic eye movement** | Rayner 1998 (eye-tracking in reading) | Sliding window with occasional long-range jumps |
| **Foveal vs peripheral acuity** | Standard vision science | High-resolution local window + lower-resolution global access via depth |
| **Working memory capacity (4±1 chunks)** | Cowan 2001 | Bounded KV cache (sink tokens + window) |
| **Topic-sentence anchoring** | Kintsch & van Dijk 1978 (text comprehension) | Attention sinks on opening tokens |
| **Habituation / selective attention** | Standard attention literature | Differential / noise-cancelling attention |
| **Predictive reading** | Levy 2008 (expectation-based comprehension) | Already implicit in autoregressive LM objective |

The hypothesis: an attention mechanism aligned with these patterns should preserve quality on tasks that humans handle well (local reasoning, comprehension, dialog) while accepting controlled degradation on tasks where humans also struggle (multi-hop reasoning over very long context).

We test this hypothesis empirically. We don't claim it's true — we claim it's worth measuring.

---

## 5. Data strategy

The pivot simplifies data significantly. **No more Layer 3 Claude-mimicking corpus needed.** We need standard high-quality SFT and continued-pretraining data so the healing fine-tunes can recover quality lost by sparsification.

### Layer 1: Continued pretraining + SFT (primary)

| Dataset | Use | Why |
|---|---|---|
| `HuggingFaceFW/fineweb-edu` (10B token subset) | Continued pretraining for sparsified model | Clean, high-quality web text; standard CPT corpus |
| `HuggingFaceH4/ultrachat_200k` (20K samples) | SFT recovery after sparsification | Broad instruction-following |
| `allenai/tulu-3-sft-mixture` (20K samples) | SFT recovery | Well-curated, recent |
| `Anthropic/hh-rlhf` (10K chosen) | Optional SFT signal | Preserves helpfulness behavior |

### Layer 2: DEPRECATED for primary use, archived for reference
Anthropic research papers download still works; useful for adjacent exploration but not central to sparse-attention work.

### Layer 3: DEPRECATED — Claude-response distillation no longer the goal
`data/raw/teacher_responses_v0.jsonl` (6 Claude responses) archived. Can be used as optional SFT signal but no longer expanded.

### New: eye-tracking reading data (optional, for cognitive validation)
- **GECO** (Cop et al. 2017) — bilingual eye-tracking corpus, ~5K sentences with fixation data
- **ZuCo** (Hollenstein et al. 2018) — natural reading eye-tracking, 12 subjects
- **PROVO** (Luke & Christianson 2018) — predictability + reading times

These are not training data. They are **validation data**: if our sparse attention pattern correlates with human fixation patterns on the same text, that's a positive signal for the cognitive analogy. Optional stretch goal (Phase 8+).

---

## 6. Training plan

```
Stage 0 — Baseline measurement (no training):
  - Run base Qwen 2.5 0.5B-Instruct on eval suite, record dense baseline
  - Apply StreamingLLM sinks at inference (zero-train) to base model, measure
  - This gives us the "free lunch" baseline before any fine-tuning

Stage 1 — Sliding window adaptation (SWAA-style):
  - Replace attention in interleaved layers with sliding window (W=128)
  - Apply LoRA healing fine-tune on 1-2B tokens FineWeb-Edu + 20K UltraChat
  - 1-2 epochs, lr 1e-4, cosine schedule
  - Wall time: ~12-24hr on Colab Pro A100

Stage 2 — DEX overlay:
  - Apply DEX differential extension on top of Stage 1 model
  - <0.01% data, ~1hr fine-tune
  - Goal: layer selective-suppression behavior onto already-sparsified model

Stage 3 — Iteration:
  - Hyperparameter sweep: window size (64/128/256), interleave ratio
    (every-other vs first-N-dense vs last-N-dense), sink count (1/2/4/8)
  - Pick winner on Pareto frontier
```

Each stage gates on the next. If Stage 1 collapses quality below 90% of baseline, halt and rethink before Stage 2.

---

## 7. Evaluation (the formal-testing part)

**Three-way+ comparison**: (A) base Qwen 0.5B dense, (B) sinks-only zero-train, (C) sliding window healed, (D) sinks + window + DEX (the Mimic).

### Efficiency metrics

| Metric | Tool | Why |
|---|---|---|
| **FLOPs / token** | `torch.profiler` + manual count | Theoretical compute |
| **Wall-clock latency** | `time.perf_counter` over 100 prompts | Real-world speed |
| **Peak GPU memory** | `torch.cuda.max_memory_allocated()` | Memory ceiling |
| **KV cache size** | Direct measurement | Long-context viability |
| **Throughput (tokens/sec)** | Standard | Production-relevant |

### Quality metrics

| Benchmark | Why | Notes |
|---|---|---|
| **Perplexity on WikiText-103** | Standard intrinsic LM quality | Required baseline |
| **MMLU (5-shot)** | General knowledge + reasoning | Multi-hop sensitive; will surface sparse degradation |
| **HellaSwag** | Commonsense | Standard |
| **ARC-Challenge** | Reasoning | Standard |
| **GSM8K** | Multi-hop arithmetic | **Critical** — sparse attention is known to hurt multi-hop reasoning (per "The Sparse Frontier" 2025) |
| **LongBench (short subset)** | Long-context comprehension | Tests whether sparsity helps OR hurts at the lengths we care about |
| **Custom probes** (`probes/v1.md`) | Style & behavior preservation | Reframed: now measures whether sparsification changes response patterns |

### Pareto analysis

Plot all variants on (efficiency, quality) axes. The win condition is finding a Mimic variant that:
- ≥1.5x faster than dense baseline on wall-clock latency
- ≥30% lower KV cache memory at 2K context
- ≤5% degradation on average quality across MMLU/HellaSwag/ARC
- ≤15% degradation on GSM8K (the multi-hop canary)

If we miss any of these, that's the result — we report it honestly. "Mimic underperformed on multi-hop but achieved 2x speedup at <3% loss on local-reasoning tasks" is a legitimate finding.

### Optional: cognitive alignment eval (Phase 8 stretch)

Run base + Mimic on GECO/ZuCo sentences. Extract attention patterns. Compare against human fixation distributions. Report correlation. This is the most novel part if it works; it's also the most uncertain.

---

## 8. Phased plan

| Phase | What | Time | Gate |
|---|---|---|---|
| ~~0a. Original design~~ | DESIGN.md v0.2 | done | superseded by v0.3 |
| **0b. Redesign (this doc)** | DESIGN.md v0.3 + ARCH_SPARSE.md | this session | Sanjith review |
| **1. Baseline measurement** | Stage 0: eval base Qwen + sinks-zero-train | next session | baseline numbers recorded |
| **2. Data pipeline** | Pull FineWeb-Edu subset + UltraChat + Tulu | next session | data on disk, dedup'd |
| **3. SWAA implementation** | `arch/sliding_window.py`, interleave logic | next session | unit tests pass |
| **4. Stage 1 fine-tune** | Sliding window + LoRA heal on Colab Pro | one session | loss curve sane, eval ≥90% baseline |
| **5. DEX implementation + Stage 2** | `arch/dex.py`, differential overlay fine-tune | one session | quality not worse than Stage 1 |
| **6. Hyperparameter sweep** | Window/interleave/sink sweep | one session | Pareto frontier mapped |
| **7. Full eval + REPORT.md** | All metrics, all variants, Pareto plot | one session | REPORT.md generated |
| **8. (Stretch) Cognitive alignment** | GECO/ZuCo correlation | optional | only if Phase 7 was positive |

Decision gates: each phase has one. Skip the next phase if the current one fails its gate; rethink instead.

---

## 9. Repo layout

```
the-mimic/
├── DESIGN.md                  # this doc (v0.3)
├── ARCH_SPARSE.md             # sparse architecture spec
├── STATUS.md                  # running journal
├── requirements.txt           # Python deps (extended for flash-attn)
├── probes/
│   └── v1.md                  # 50 questions, reframed as behavioral probes
├── arch/                      # NEW: sparse attention implementations
│   ├── sinks.py               # StreamingLLM attention sinks
│   ├── sliding_window.py      # Mistral/SWAA sliding window + interleave
│   ├── dex.py                 # Differential Extension (DEX)
│   ├── mimic_attention.py     # Combined: sinks + window + DEX
│   └── patch_qwen.py          # Monkey-patches Qwen2 attention with Mimic
├── data/
│   ├── download.py            # CPT + SFT corpora (FineWeb-Edu, UltraChat, Tulu)
│   ├── combine.py             # dedup + format
│   ├── raw/
│   │   ├── teacher_responses_v0.jsonl   # ARCHIVED (Claude responses; optional signal)
│   │   └── _seed_v0.py        # ARCHIVED seed generator
│   └── processed/             # gitignored
├── configs/
│   ├── smoke.yaml             # Stage 1 smoke (legacy, still useful for SFT validation)
│   ├── full.yaml              # Stage 2 full SFT
│   ├── sparse_swaa.yaml       # NEW: sliding window heal config
│   └── sparse_full.yaml       # NEW: sinks + window + DEX config
├── train.py                   # Dense LoRA SFT (existing)
├── train_sparse.py            # NEW: sparse-variant healing fine-tune
├── eval/
│   ├── efficiency.py          # NEW: FLOPs, latency, memory, KV cache
│   ├── quality.py             # NEW: perplexity, MMLU, HellaSwag, ARC, GSM8K, LongBench
│   ├── probes.py              # Custom probe runner (existed in plan)
│   ├── pareto.py              # NEW: efficiency-vs-quality plot
│   └── REPORT.md              # generated
└── cognitive/                 # OPTIONAL Phase 8
    ├── eye_tracking_align.py  # GECO/ZuCo attention correlation
    └── REPORT_cognitive.md    # generated
```

---

## 10. Open design questions

1. **Window size** — start with W=128 (per SWAA defaults at Qwen scale). Sweep 64/128/256 in Phase 6.
2. **Interleave ratio** — every-other layer? First N dense + last N dense + middle sliding (per SWAA)? Sweep in Phase 6.
3. **Sink count** — StreamingLLM uses 4 by default. Test 1/2/4/8.
4. **Healing data scale** — 1B vs 2B tokens for Stage 1 heal? Start 1B, escalate only if quality bad.
5. **DEX skip if Stage 1 already good enough?** — possible. Decide after Stage 1 eval.
6. **Cognitive eval scope** — if Phase 7 succeeds, is the eye-tracking alignment worth the extra session(s)? Defer decision until then.

---

## 11. Risks recorded

1. **0.5B may be too small for sparse to help meaningfully.** SWAA tested on 1.5B+. The Mimic at 0.5B may not show enough quality headroom to absorb sparsification. Mitigation: SmolLM2-360M is even more constrained (so worse case); fall back to dense if sparse provably worse.
2. **Multi-hop reasoning degradation** is documented in "Sparse Frontier" 2025. Expected, but unbounded magnitude at 0.5B scale.
3. **Interaction effects** between sinks + window + DEX are unstudied. Combining three independently-validated mechanisms may produce unexpected interference. Mitigation: ablation study built into Phase 6 sweep.
4. **HuggingFace integration friction** — `Qwen2Config.sliding_window` exists but DEX is community-impl only. Implementation work in `arch/` is non-trivial. Mitigation: scoped phases, gates between each.
5. **Cognitive analogy is rhetorical not empirical** unless we run Phase 8. Be honest in any writeup: we used cognitive science to *motivate* architectural choices; we measure performance, not actual brain-alignment, unless explicitly tested.
6. **Bay Area + bootstrap discipline.** Side project. Pauses if AI Ops venture (scoreboard W21 due 2026-05-18) needs attention.

---

## 12. What I'm explicitly NOT doing

- Training from scratch (NSA-style). Compute-prohibitive at side-project scale.
- Implementing all of Longformer / BigBird / Reformer / Routing Transformer separately. Researcher recommendation was to focus; we focus on sinks + window + DEX.
- Linear-attention variants (Linformer, Performer). Quality usually too poor.
- Mamba / state-space models. Different architectural family; out of scope.
- Bias-archaeology evaluation (the v0.2 thesis). The probes set is retained for behavioral comparison but the goal is no longer "does The Mimic reproduce Claude's biases."
- Public release of weights without ToS review.

---

_Last updated: 2026-05-16. v0.3._
