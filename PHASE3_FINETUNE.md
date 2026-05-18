# Phase 3 — Sparse-Active Fine-Tune

> Heal the sparse-attention quality dent on Qwen 2.5 0.5B by training with the sink+window mask active. Three sequential stages: continued pretraining (CPT) → supervised fine-tuning (SFT) → robustness repair on the bucket that hurts most.

**Status**: design + code shipped; training to run on Colab Pro.
**Created**: 2026-05-18
**Builds on**: `ARCH_SPARSE.md`, `STATUS.md` Phase 1 sweep results.

---

## 1. Why this phase exists

Phase 1 measured the dent. With `sinks=4, window=128` at `seq_len=256`, PPL ratio is 1.042 (≈4% degradation, free). With more aggressive sparsity it gets worse. The healing fine-tune asks: can we recover most of that loss by training the model under the sparse pattern it'll be inferenced under?

**The Codex correctness point that anchors the entire phase**: if we fine-tune under full attention and only switch to sparse at inference, the model keeps relying on long-range access it won't have. Training must run with the **same kind** of mask used at eval, ideally with **randomized window** so the model learns robustness across sparsity levels rather than overfitting one pattern.

---

## 2. Three stages

### Stage 1 — Sparse continued pretraining (CPT)
- **Objective**: ordinary next-token prediction on raw text, with sink+window 4D mask active.
- **Data**: 40% wiki / 20% science / 15% philosophy / 15% instruction-as-completion / 10% dialogue (per Codex's mix recommendation).
- **Mask**: `n_sinks=4`, `window ∈ {64, 128, 256}` randomized per batch.
- **Why**: teach the base model to use the sparse pattern for the *substance* of language modeling. Restores most of the PPL gap.

### Stage 2 — Sparse supervised fine-tuning (SFT)
- **Objective**: instruction-response training under the same sparse mask regime.
- **Data**: curated instruction data (Tulu-3, no_robots, dolly, OpenAssistant) + small hand-curated metacognitive demonstration set.
- **Style target**: open, fast-when-easy / slower-when-needed, careful, willing to surface uncertainty. **"Visible metacognition," NOT chain-of-thought dump** — patterns like:
  - "I'm uncertain because…"
  - "The key assumption is…"
  - "A quick way to check this is…"
  - "Here's the concise answer, then the caveat…"
- **Why**: shape personality + answer-shape without making the model ramble or expose fake private reasoning.

### Stage 3 — Robustness repair
- **Objective**: rebalance training data based on which register or genre the post-SFT model lost the most on.
- **Method**: run the sweep eval over genre-bucketed prompts. Find the deepest dent. Generate or curate more training data of that genre. Re-train.
- **Why**: don't guess what's hard. Measure, then patch.

---

## 3. Critical implementation details

### Combined mask (causal + sink + window + padding)
At training time, batched padded sequences need **all four constraints simultaneously**:

```
allowed(q, k) = causal(q, k) AND (sink(k) OR window(q, k)) AND not_pad(k)
```

If the padding constraint is dropped, the model learns to attend to pad tokens — silent corruption. Implemented in `arch/training_mask.py` as a function that builds the combined 4D mask from a 2D padding mask + sparsity hyperparameters.

### Randomized window during training
Per training step, sample window ∈ {64, 128, 256} (or a continuous range). Build the mask with the sampled window. Model sees varied sparsity, learns generalization. Implemented in the data collator.

### Sparsity at training vs eval
Train at randomized window. Eval at fixed windows (the sweep we already have). Reports separate numbers for each eval window — proves robustness across the range, not just at one point.

### Sequence length
Stage 1: start at `seq_len=512`. Bump to 1024 if VRAM allows on Colab Pro A100.
Stage 2: 1024 (typical for instruction data).

### LoRA config (per Codex)
- `r=16, alpha=32, dropout=0.05`
- Target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- Method: QLoRA (4-bit base, bf16 compute) on A100; plain LoRA fine if memory permits

### Learning rate
- LoRA: `lr=2e-4`, cosine schedule, 3% warmup
- Stage 1 epochs: 1 (CPT is broad coverage, more epochs risk catastrophic forgetting)
- Stage 2 epochs: 2-3 (SFT benefits from more passes on curated data)

---

## 4. Data plan (concrete)

### Stage 1 CPT corpus (~50-80M tokens)

| Bucket | Weight | Source | HF path | Sample |
|---|---|---|---|---|
| Wiki | 40% | English Wikipedia | `wikimedia/wikipedia` (`20231101.en`) | 30K articles, first 2K tokens each |
| Science | 20% | arXiv papers (abstracts + intros) | `armanc/scientific_papers` (arxiv config) | 15K abstracts + intros |
| Philosophy | 15% | Project Gutenberg philosophy + SEP | `manu/project_gutenberg` (filter by subject) + manual SEP fetches | 5-10K chunks |
| Instruction-as-completion | 15% | Tulu/UltraChat formatted as flat text | `allenai/tulu-3-sft-mixture` | 10K examples |
| Dialogue | 10% | OpenAssistant multi-turn | `OpenAssistant/oasst2` | 5K conversations |

### Stage 2 SFT corpus (~50-150K examples)

| Bucket | Weight | Source | HF path |
|---|---|---|---|
| Diverse instructions | 50% | Tulu-3 SFT mix | `allenai/tulu-3-sft-mixture` |
| Casual / no-robots | 15% | Hand-curated assistant responses | `HuggingFaceH4/no_robots` |
| Multi-turn dialogue | 15% | OASST2 trees | `OpenAssistant/oasst2` |
| Technical Q&A | 10% | Dolly factoid + scientific | `databricks/databricks-dolly-15k` |
| Metacognitive seed | 10% | `data/raw/metacognitive_seed_v0.jsonl` (hand-curated) | local |

**Deliberately excluded**: synthetic Claude responses as primary data. Per Codex's caution and Anthropic ToS gray zone: training on proprietary model outputs as main corpus is risky. The metacognitive seed set is hand-curated demonstrations of the *style* — original writing illustrating the patterns, not transcripts of any commercial model.

---

## 5. North-star eval

The deliverable for this phase is a table:

| Metric | Value |
|---|---|
| Full-attention PPL (baseline, no FT) | (from Phase 1 sweep) |
| Sparse-attention PPL **before** FT (sinks=4, window=128) | (from Phase 1 sweep) |
| Sparse-attention PPL **after Stage 1** (CPT) | TBD |
| Sparse-attention PPL **after Stage 2** (SFT) | TBD |
| Sparse-attention PPL across window ∈ {32, 64, 128, 256} after FT | TBD (Pareto frontier) |
| **Full-attention regression** after FT (did we hurt dense quality?) | TBD |
| Style eval on metacognitive prompts (qualitative) | TBD |
| Genre-bucket PPL (formal vs informal vs technical) | TBD — Stage 3 input |

The full-attention regression number is the safety check: if FT moved sparse PPL down but moved full PPL up, we traded one kind of damage for another. Acceptable only if the sparse gain is much larger than the dense loss.

---

## 6. Compute budget (Colab Pro)

Colab Pro: ~100 compute units/month, A100 access at ~13 units/hr.

| Stage | Est. A100 hours | Compute units |
|---|---|---|
| Stage 1 CPT (~50M tokens × 1 epoch, LoRA, seq_len 512) | 6-10 | 80-130 |
| Stage 2 SFT (~80K examples × 2 epochs) | 4-6 | 50-80 |
| Stage 3 repair (one targeted re-train) | 2-4 | 25-50 |
| Full eval suite | 1-2 | 15-25 |
| **Total** | **13-22 hr** | **170-285 units** |

This exceeds Colab Pro's monthly allocation. Two paths:
1. **Pro+ for one month** ($50, ~500 units) covers all three stages with buffer.
2. **Stage 1 first**, evaluate, decide on Stage 2 based on results. Spread across 2-3 months.

Default: option 2 unless Sanjith wants to compress into one month.

---

## 7. Files shipping this turn

| Path | Purpose |
|---|---|
| `PHASE3_FINETUNE.md` | this design |
| `arch/training_mask.py` | combined causal+sink+window+padding mask builder |
| `data/download_phase3.py` | data download per the mix above |
| `data/raw/_metacognitive_seed_v0.py` | small hand-curated metacognitive demo generator |
| `configs/heal_cpt.yaml` | Stage 1 config |
| `configs/heal_sft.yaml` | Stage 2 config |
| `train_sparse.py` | training entry with windowing randomization |
| `notebooks/heal_colab.ipynb` | Colab orchestration |
| `STATUS.md` | updated |

---

## 8. Open questions for Sanjith

1. **GitHub or Drive for Colab access?** Notebook defaults to git clone + Drive checkpoint storage; can swap.
2. **Pro+ for one month, or staged across months on Pro?** Default: staged.
3. **Stage 3 trigger**: run Stage 3 unconditionally after Stage 2, or only if eval surfaces a clear bucket dent? Default: conditional (don't pre-spend compute on a problem we may not have).
4. **Seq length escalation**: start at 512 (safer) or 1024 (more sparsity headroom)? Default: 512 first; bump after smoke test passes.

---

_Last updated: 2026-05-18. Codex external review integrated._
