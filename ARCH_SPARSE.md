# The Mimic — Sparse Attention Architecture Spec

> Implementation-level companion to `DESIGN.md`. Three layered mechanisms grafted onto pretrained Qwen 2.5 0.5B, each independently testable and ablation-friendly.

**Status**: spec only; no code yet at `arch/`.
**Source review**: see `DESIGN.md` §1 thesis + the 2026-05-16 research survey (in session log).

---

## 1. Architectural overview

Three mechanisms layered on top of Qwen2's existing `Qwen2Attention` module. Each can be enabled or disabled independently for ablation.

```
                    ┌─────────────────────────────────────┐
                    │   Qwen2.5 0.5B (24 layers, 14 heads) │
                    └─────────────────────────────────────┘
                                     │
                                     ▼
        ┌────────────────────────────────────────────────────┐
        │ Per-layer attention: configurable via LayerType    │
        ├────────────────────────────────────────────────────┤
        │                                                     │
        │  Layer 0       ─→ FULL_ATTENTION (preserve anchor) │
        │  Layer 1       ─→ FULL_ATTENTION                   │
        │  Layer 2       ─→ SPARSE (window + sinks)          │
        │  Layer 3       ─→ SPARSE                           │
        │  ...                                                │
        │  Layer 21      ─→ SPARSE                           │
        │  Layer 22      ─→ FULL_ATTENTION                   │
        │  Layer 23      ─→ FULL_ATTENTION (preserve output) │
        │                                                     │
        │  + DEX differential operator wraps all attention    │
        └────────────────────────────────────────────────────┘
```

This is **SWAA-style interleaving**: keep early + late layers as full attention (they're load-bearing for global structure per "Sparse Frontier" 2025); convert middle layers to sliding-window + sinks; layer DEX on top for differential noise cancellation.

---

## 2. Mechanism 1: Attention sinks (StreamingLLM)

**Reference**: Xiao et al. 2024, https://arxiv.org/abs/2309.17453
**Reference impl**: https://github.com/mit-han-lab/streaming-llm
**Cognitive analog**: anchoring on opening tokens (title/topic sentence orientation)

### What it does
Keep the first `N_sinks` tokens permanently in the KV cache, regardless of how far back they are. Within the cache, also maintain a rolling window of the most recent `W` tokens. Tokens between sink and window are evicted.

### Implementation sketch (`arch/sinks.py`)
```python
class SinkAttention(nn.Module):
    """
    Wraps Qwen2Attention with an attention sink + rolling window KV cache policy.
    Zero new parameters. Operates at the cache-management level.
    """
    def __init__(self, base_attention, n_sinks=4, window=128):
        super().__init__()
        self.base = base_attention
        self.n_sinks = n_sinks
        self.window = window

    def evict_kv(self, past_key_values):
        # Keep first n_sinks + last window from each layer's K/V tensors
        ...
```

### Why this is cheap
**Zero training required for baseline use.** Apply to base Qwen at inference; measure. This is Phase 1 Stage 0. Optional: fine-tune a dedicated `<sink>` placeholder token for slightly better stability.

### Hyperparameters to sweep
- `n_sinks`: 1, 2, 4 (default), 8
- `window`: 64, 128 (default), 256, 512

---

## 3. Mechanism 2: Sliding window with hybrid interleave (SWAA)

**Reference**: SWAA, arXiv:2512.10411 (Dec 2025); Mistral SWA pattern
**Reference impl**: SWAA GitHub (per researcher survey); also `Qwen2Config.sliding_window` field
**Cognitive analog**: foveal focus on recent tokens; peripheral access via stacked depth

### What it does
Replace dense O(n²) attention with O(n·W) sliding window attention in selected layers. Each query token attends only to the `W` tokens immediately preceding it. Stacked across layers, the effective receptive field grows: layer L sees L·W tokens back at low resolution.

### SWAA hybrid pattern
Per SWAA recommendation, do NOT convert all layers. Keep:
- First 2-4 layers: full attention (preserve input encoding)
- Last 2-4 layers: full attention (preserve output generation quality)
- Middle layers: sliding window

For Qwen 2.5 0.5B (24 layers): full attention on layers 0-2, 21-23; sliding window on layers 3-20. ~75% of layers sparsified.

### Implementation sketch (`arch/sliding_window.py`)
```python
class SlidingWindowAttention(nn.Module):
    """
    O(n·W) attention. Reuses Qwen2Attention's QKV projections; only
    the attention mask changes. Compatible with FlashAttention-2 when
    `is_causal=True` and `sliding_window` is passed.
    """
    def __init__(self, base_attention, window=128):
        super().__init__()
        self.base = base_attention
        self.window = window

    def forward(self, hidden_states, attention_mask, ...):
        # Construct band-diagonal mask of width `window`
        # Hand off to F.scaled_dot_product_attention with custom mask
        # OR use flash_attn_func with window=window argument
        ...
```

### Why this is cheap (relatively)
Mistral's `sliding_window` field is already in `Qwen2Config`. FlashAttention-2 supports sliding window natively (`window_size` arg). Implementation reduces to: set config flag, patch attention forward in selected layers, run healing fine-tune.

### Healing fine-tune required
Pure post-hoc swap from dense → sliding window causes quality drop. SWAA shows that 1-2 epochs on 1-2B tokens of continued pretraining recovers most of it. We do this in Stage 1 (see `DESIGN.md` §6).

### Hyperparameters to sweep
- `window`: 64, 128, 256
- Interleave ratio: "first2+last2", "first4+last4", "every-other", "every-third"
- Total full-attention layer count: 4 (default), 6, 8

---

## 4. Mechanism 3: DEX (Differential Extension)

**Reference**: arXiv:2505.16333 (NeurIPS 2025); builds on Differential Transformer arXiv:2410.05258
**Cognitive analog**: selective suppression / habituation (ignore filler & noise while reading)

### What it does
Differential Transformer replaces each attention head with two parallel attention computations: `Attn1(Q1, K1, V1) - λ · Attn2(Q2, K2, V2)`. Common-mode noise cancels; signal-bearing attention amplifies. Heads concentrate attention mass on fewer, more relevant positions.

**DEX** is the post-hoc adaptation: instead of replacing attention, DEX wraps existing pretrained attention and adds a lightweight differential operation on the value matrix. Adds <1% parameters and <5% compute. Trained with <0.01% of data in a distillation setup.

### Implementation sketch (`arch/dex.py`)
```python
class DEXWrapper(nn.Module):
    """
    Wraps an attention module with DEX's differential operation.
    Adds a small learnable lambda parameter per head and a residual
    differential path on the value matrix.
    """
    def __init__(self, base_attention, init_lambda=0.5):
        super().__init__()
        self.base = base_attention
        self.lambda_param = nn.Parameter(torch.tensor([init_lambda] * base_attention.num_heads))
        # Small additional projection for differential signal
        self.diff_proj = nn.Linear(base_attention.head_dim, base_attention.head_dim, bias=False)

    def forward(self, hidden_states, ...):
        # Run base attention
        # Compute differential signal on values
        # Combine: base_output - lambda * diff_signal
        ...
```

### Why this layers on top
DEX explicitly works on pretrained models (the paper's selling point). It also doesn't conflict with sliding window or sinks — it operates on whatever attention output you give it. So we can stack: Mechanism 1 (sinks) + Mechanism 2 (window) determine *what* gets attended to; DEX determines *how* the attended values get combined.

### Hyperparameters to sweep
- `init_lambda`: 0.3, 0.5 (default), 0.7
- DEX-only training data: 100K, 500K, 1M tokens
- Apply DEX to: all layers, only sparse layers, only full-attention layers

---

## 5. Combined Mimic attention

`arch/mimic_attention.py` provides the composed mechanism with feature flags:

```python
class MimicAttention(nn.Module):
    def __init__(
        self,
        base_attention,
        use_sinks: bool = True,
        n_sinks: int = 4,
        use_window: bool = True,
        window: int = 128,
        use_dex: bool = True,
        init_lambda: float = 0.5,
    ):
        ...
```

Feature flags enable independent ablation: turn off any combination of (sinks, window, DEX) and re-eval. Required for the Pareto analysis.

`arch/patch_qwen.py` provides the monkey-patch that swaps `Qwen2Attention` with `MimicAttention` in selected layers of a loaded HuggingFace model.

---

## 6. Reference implementations to study (not reinvent)

| Mechanism | Repo | License | Adaptation effort |
|---|---|---|---|
| Sinks | https://github.com/mit-han-lab/streaming-llm | MIT | Direct port; ~50 LOC |
| Sliding window | HuggingFace `Qwen2Attention` + FlashAttention `window_size` | Apache 2.0 | Config flag; ~20 LOC |
| SWAA interleave | SWAA repo (per arXiv:2512.10411) | TBD | Mostly config; ~100 LOC |
| DEX | https://github.com/microsoft/unilm/tree/master/Diff-Transformer | MIT | Adapt their training script; ~200 LOC |
| DEX adaptation | DEX paper supplementary (NeurIPS 2025) | TBD | Need to verify code release |

**Verification action**: before Phase 5, confirm DEX code is released. If not, implement from paper description (~1 day of work).

---

## 7. Implementation order (Phase-by-Phase)

```
Phase 1: Baseline + sinks only (zero train)
  - Implement arch/sinks.py
  - Run base Qwen + sinks at inference on eval suite
  - Compare to dense baseline
  - Gate: sinks-only achieves ≥98% of dense quality with KV cache savings

Phase 3: Sliding window
  - Implement arch/sliding_window.py + arch/patch_qwen.py
  - Apply to interleaved layers
  - Sanity check: forward pass produces valid logits, no NaN

Phase 4: Stage 1 fine-tune (window heal)
  - LoRA heal on FineWeb-Edu 1B tokens + UltraChat 20K
  - Gate: eval ≥90% of dense baseline on MMLU+HellaSwag+ARC average

Phase 5: DEX
  - Implement arch/dex.py
  - Stage 2 fine-tune: DEX overlay on Stage 1 model
  - Gate: no quality regression from Stage 1

Phase 6: Sweep
  - Run ablation grid (sinks×window×DEX) across hyperparameter ranges
  - Output: Pareto frontier table

Phase 7: REPORT.md
  - All metrics, all variants, headline Pareto plot
  - Honest writeup (positive or negative)
```

---

## 8. Compute budget (Colab Pro)

| Phase | Estimated Colab A100 hours | Notes |
|---|---|---|
| 1 (baseline + sinks) | 2-4 | Inference only |
| 3 (window impl) | 0 | Code work |
| 4 (Stage 1 heal) | 12-24 | LoRA + 1B tokens |
| 5 (DEX overlay) | 2-4 | Tiny fine-tune |
| 6 (sweep, ~10 configs) | 30-60 | Subset of Stage 1+5 per config |
| 7 (full eval) | 4-8 | Inference across benchmarks |
| **Total** | **~50-100 A100-hours** | Spread across weeks |

Colab Pro provides ~100 compute units/month; Pro+ provides ~500. Pro+ recommended if running all phases in <2 months. If Sanjith is on Pro (not Pro+), spread phases across 2-3 months or upgrade temporarily for Phase 6 sweep.

---

## 9. What we measure (and how)

See `DESIGN.md` §7 for the full metric list. Implementation notes:

- **FLOPs**: use `fvcore.nn.FlopCountAnalysis` or manual count. Both should agree.
- **Latency**: `torch.cuda.synchronize()` + `time.perf_counter()` over 100 prompts, report median + p95.
- **KV cache**: log `sum(k.numel() + v.numel() for k, v in past_key_values)` × dtype size.
- **Quality**: use `lm-evaluation-harness` (EleutherAI) for standard benchmarks — it handles all the boilerplate for MMLU/HellaSwag/ARC/GSM8K. Run with `--model hf --model_args pretrained=<path-to-mimic>`.
- **Probes**: custom runner in `eval/probes.py`, compare outputs to baseline via embedding similarity + manual scoring.

---

## 10. Decision gates summary

| Gate | Pass criterion | If fail |
|---|---|---|
| **Phase 1 (sinks-only)** | ≥98% dense quality, ≥30% KV cache savings | Sinks don't help here; skip to window |
| **Phase 4 (window heal)** | Eval ≥90% dense baseline | Adjust hyperparameters; if still <90%, abandon sparse approach |
| **Phase 5 (DEX)** | No regression from Phase 4 | Ship Mimic-without-DEX as the deliverable |
| **Phase 6 (sweep)** | At least one config on Pareto frontier | Report negative result honestly |
| **Phase 7 (report)** | All metrics measured & plotted | n/a — this is the deliverable regardless of outcome |

---

## 11. Risks specific to architecture work

1. **FlashAttention compatibility** — sliding window + FlashAttention works, but DEX may require custom kernels for full efficiency. Worst case: DEX runs in eager mode with ~30% overhead. Acceptable for measurement.
2. **Qwen2 attention module evolution** — HuggingFace refactors transformer internals occasionally. Pin `transformers` version in `requirements.txt`.
3. **Monkey-patching brittleness** — `arch/patch_qwen.py` modifies a loaded model in-place. Sensitive to model class changes. Mitigation: unit-test the patch on a fresh model load before each training run.
4. **Layer norm interactions** — Diff Transformer uses GroupNorm; DEX may need similar adaptation when applied post-hoc to a model trained with RMSNorm. Flag for verification during implementation.

---

_Last updated: 2026-05-16. Spec only; no code shipped yet._
