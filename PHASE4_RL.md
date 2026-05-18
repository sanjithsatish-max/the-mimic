# Phase 4 — Reinforcement Learning for Better Inference

> Design sketch. Not yet implemented. Sequences after Phase 3 Stage 2 (sparse SFT) lands.

**Status**: design only.
**Created**: 2026-05-18.
**Prerequisite**: Phase 3 SFT checkpoint exists and passes baseline quality + style evals.

---

## 1. Why this phase exists

Phase 3 teaches the sparsified model to *speak well* (heal quality damage + adopt metacognitive style). It does not directly teach it to *reason well*. RL with verifiable rewards is the most direct lever for improving inference quality at this scale — DeepSeek R1 demonstrated that even small models gain measurable reasoning ability from GRPO on math/code datasets.

The bet: a 0.5B model with sparse attention and metacognitive priors can be pushed past its SFT ceiling on reasoning tasks by RL on verifiable problems. The risk: 0.5B may be below the effective scale for reasoning RL — research findings on this are mixed.

---

## 2. Algorithm choice — GRPO

**GRPO** (Group Relative Policy Optimization, DeepSeek 2024). Reasons:

| Property | GRPO | PPO | DPO |
|---|---|---|---|
| Value model needed | **No** | Yes | No |
| Verifiable rewards | **Native** | Native | Indirect (preferences) |
| Compute per step | Medium (G rollouts) | High (G + value forward) | Low |
| Memory footprint | Lower | Higher | Lowest |
| Modern usage | Standard for reasoning RL | Legacy | Common for preferences |

GRPO generates G rollouts per prompt (typically 4-8), uses group-mean reward as baseline, and applies KL penalty against a frozen reference (the Phase 3 SFT model). No critic to train. Simpler and cheaper than PPO. Fits Colab Pro better.

DPO is an alternative for the metacognitive-style polish (preference data, no rollouts). Worth considering as Phase 4b.

---

## 3. Staged plan

### Phase 4a — Math RL (GSM8K)
- **Dataset**: GSM8K train (8.5K word problems with numeric answers).
- **Reward**: regex-extract final answer from rollout, compare to ground truth. Binary 0/1.
- **Compute**: ~6-12 hr on A100 (subset of 2K prompts, 3 epochs, G=4).
- **Why first**: cheapest verifiable reward; canonical benchmark; well-understood failure modes.

### Phase 4b — Code RL (MBPP)
- **Dataset**: MBPP (~1K Python programming problems with hidden test cases).
- **Reward**: execute rollout in sandboxed Python, run tests, fraction passed.
- **Compute**: ~4-8 hr on A100 (longer rollouts, but smaller dataset).
- **Why second**: execution feedback is stronger signal than format-matching; tests generalization beyond math.

### Phase 4c — Format / metacognitive compliance
- **Dataset**: prompts known to elicit one of the four metacognitive patterns.
- **Reward**: rule-based check (regex for "I'm uncertain because", "key assumption is", etc.) + brevity penalty.
- **Compute**: ~2-4 hr on A100.
- **Why third**: reinforces Phase 3's style training; cheap; catches drift if 4a/4b moved style.

### Phase 4d (stretch) — RLAIF with Claude judge
- **Dataset**: open-ended prompts where verifiable rewards don't apply.
- **Reward**: Claude scores rollouts on a rubric (the model's response is sent to Claude Code, scored, used as reward).
- **Compute**: API-bound, not GPU-bound. Subscription covers it.
- **Why last**: most expensive in human attention; only run if 4a-c plateau and we want further gains.

---

## 4. Critical implementation considerations

### Sparse mask during RL
**Same correctness concern as Phase 3**: if the model generates rollouts under full attention but is trained for inference under sparse attention, the rollouts won't reflect the sparse behavior we care about. Two options:

- **Option A (simpler)**: apply sparse mask during training gradient steps only; generate rollouts with full attention. Accept the train/eval mismatch on generation.
- **Option B (correct)**: extend the generation loop to apply the sparse 4D mask dynamically as tokens are generated. Non-trivial — the mask must grow as the rollout extends.

Default: start with Option A on Phase 4a as a smoke test. If GSM8K accuracy under sparse eval is much worse than under full eval, switch to Option B.

### Reward hacking
The regex-based math reward is hackable: model learns to output "the answer is 42" without showing work. Mitigations:
- Penalize empty reasoning (`len(rollout) < min_tokens`)
- Reward only if reasoning trace is present (regex for at least one of the metacognitive patterns)
- Random spot-check by Claude judge on a small fraction

### KL anchor matters
KL penalty against the Phase 3 SFT reference model is what prevents catastrophic forgetting of metacognitive style and instruction-following. Default β = 0.04 (standard GRPO setting). If 4a starts collapsing into pure number-spitting, raise β.

### Sequence length budget
GSM8K rollouts at G=4 with max_new_tokens=512 means 4 × 512 = 2048 tokens generated per prompt. At seq_len=1024 context, this fits Colab Pro A100 with batch size 1-2 in QLoRA.

### Reference model storage
GRPO needs both the trainable policy AND a frozen reference. Two copies of a 0.5B model = ~2GB in fp32 / ~500MB in 4-bit. Fits A100 (40GB) comfortably.

---

## 5. Compute budget (Colab Pro)

| Stage | A100 hours | Compute units (~13/hr) |
|---|---|---|
| 4a (GSM8K, 2K prompts × 3 epochs) | 6-12 | 80-160 |
| 4b (MBPP, full × 3 epochs) | 4-8 | 50-100 |
| 4c (format RL) | 2-4 | 25-50 |
| 4d (RLAIF) | 4-6 + API time | 50-80 |
| **Total** | **16-30 hr** | **205-390 units** |

Exceeds one month of Pro (~100 units). Spread across 2-3 months OR upgrade to Pro+ for one month. Default: 4a → eval → decide whether 4b is worth continuing based on results.

---

## 6. North-star eval for Phase 4

| Metric | Before Phase 4 (= Phase 3 final) | After 4a | After 4b | After 4c |
|---|---|---|---|---|
| GSM8K accuracy (sparse eval, sinks=4, window=128) | TBD | **must improve** | hold | hold |
| MBPP pass@1 (sparse eval) | TBD | hold | **must improve** | hold |
| Metacognitive style compliance rate | TBD | hold | hold | **must improve** |
| MMLU 5-shot | TBD | not regress | not regress | not regress |
| Sparse PPL @ window=128 | (Phase 3 value) | not regress | not regress | not regress |
| Full-attention PPL regression | (Phase 3 value) | not regress | not regress | not regress |

The "not regress" columns are the safety checks. If GSM8K goes up but MMLU drops 5 points, we traded one capability for another and that's not progress.

---

## 7. Library + framework

- **trl >= 0.13** has `GRPOTrainer`. Standard path.
- **vllm** for fast rollout generation (3-5x faster than HuggingFace generate). Colab Pro supports it.
- **flash-attn-2** can pair with sliding window but not with arbitrary 4D masks. For sparse-mask-during-RL, eager attention stays the path.
- **bitsandbytes** for 4-bit base; the LoRA adapter stays bf16.

---

## 8. Risks specific to Phase 4

1. **0.5B may be below the effective scale for reasoning RL.** Research suggests minimum ~1.5B for stable GRPO gains. Mitigation: pilot with a 200-prompt subset of GSM8K first; if no gain after one epoch, abandon and consider scaling base model.
2. **Reward hacking on math.** Mitigated by format requirements + Claude spot-checks.
3. **Catastrophic forgetting of metacognitive style.** Mitigated by KL anchor + Phase 4c.
4. **Sparse-attention + GRPO interaction is unstudied.** No published work I'm aware of combines them at this scale. Risk that the rollout distribution under sparse attention diverges from what GRPO expects. Honest answer: we measure and adapt.

---

## 9. Files that would ship for Phase 4

```
the-mimic/
├── PHASE4_RL.md              # this doc
├── configs/
│   ├── rl_gsm8k.yaml         # Phase 4a config
│   ├── rl_mbpp.yaml          # Phase 4b config
│   └── rl_format.yaml        # Phase 4c config
├── train_grpo.py             # GRPO training entry (mirrors train_sparse.py shape)
├── rewards/
│   ├── math_answer.py        # extract + verify numeric answer
│   ├── code_execution.py     # sandboxed Python execution
│   └── format_compliance.py  # regex-based style check
└── notebooks/
    └── rl_colab.ipynb        # Phase 4 Colab orchestration
```

None of these exist yet. Phase 4 implementation starts after Phase 3 Stage 2 produces a working checkpoint with measured baselines.

---

## 10. Open design questions

1. **Algorithm pick reconsidered post-Phase-3**: GRPO is the default but DPO might be cheaper for the metacognitive-style refinement (4c). Decide after Phase 3 eval shows where the style drift actually is.
2. **Pro+ for one month or spread across months?** Same question as Phase 3. Default: spread.
3. **Sparse mask during generation**: Option A (simpler, possibly wrong) or Option B (correct, complex)? Default: A first as smoke test.
4. **Phase 4a halt criterion**: if GSM8K accuracy doesn't improve after 1 epoch, halt and reconsider scale. What threshold counts as "doesn't improve"? Default: <2 percentage points on a held-out 200-prompt subset.

---

_Last updated: 2026-05-18. Design sketch only — implementation pending Phase 3 results._
