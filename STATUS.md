# The Mimic — Status Log

Running journal of state, decisions, and next actions. Append-only.

---

## 2026-05-16 — Phase 0, 1, 2 shipped (single session)

### Done
- Folder skeleton at `~/Downloads/the-mimic/`
- `DESIGN.md` v0.2 — full architecture, data strategy, training plan, eval rubric
- `probes/v1.md` — 50 hand-curated probe questions across 6 bias categories
- `requirements.txt` — Python deps pinned
- `data/download.py` — Layer 1 (HF datasets) + Layer 2 (Anthropic PDFs) puller
- `data/synthesize.py` — Layer 3 row-appender helper (CLI + import)
- `data/combine.py` — dedupe + filter + SFT format
- `data/raw/_seed_v0.py` + `teacher_responses_v0.jsonl` — Layer 3 starter (6 Claude responses, one per category)
- `configs/smoke.yaml`, `configs/full.yaml` — Stage 1 + Stage 2 training configs
- `train.py` — `trl.SFTTrainer` LoRA wrapper

### Decisions made
- Teacher pivoted from Anthropic API → Claude Code in-session (no API access on Sanjith's $20/mo Claude Pro)
- Money estimates dropped from design (subscription-covered)
- Base model: Qwen 2.5 0.5B-Instruct via QLoRA
- Layer 3 seed: 6 responses one per category; scale opportunistically across sessions

---

## 2026-05-16 (later same session) — MAJOR PIVOT to sparse-attention thesis

### What changed
User pivoted the project away from "distill Claude's biases" toward **"sparse attention inspired by human reading patterns, formally measured."** Project name retained — we still "mimic," now mimicking *human reading* not Claude.

### Research conducted
Spawned sonnet research agent to survey open-source sparse attention architectures (2024-2026). Returned opinionated recommendation:

- **Lead architecture**: StreamingLLM (attention sinks) + SWAA-style hybrid sliding window + DEX (Differential Extension) layered on top
- **Cognitive mapping is genuine**: sinks = anchoring (topic-sentence orientation), sliding window = foveal focus, layered depth = peripheral awareness, DEX = selective suppression / habituation
- **Three integration tiers**: zero-train baseline (sinks) → light fine-tune (SWAA) → distillation overlay (DEX)
- **All fits Colab Pro A100** in 12-24 hr runs
- **Critical eval warning**: sparse attention degrades multi-hop reasoning per "Sparse Frontier" 2025 — GSM8K must be in benchmark suite

NSA (DeepSeek 2025) is most impressive but requires training from scratch — out of scope; used only as design inspiration.

### Shipped this update
- `DESIGN.md` v0.3 — pivoted thesis, cognitive motivation section, new data strategy, new eval suite
- `ARCH_SPARSE.md` — implementation spec for sinks + sliding window + DEX with feature-flag ablation harness, phase-by-phase order, compute budget, decision gates
- `STATUS.md` updated (this entry)

### Deprecated (kept for reference)
- `data/raw/teacher_responses_v0.jsonl` — 6 Claude responses; no longer central but archived
- `data/raw/_seed_v0.py` — same
- Layer 3 expansion plan from prior turn — cancelled (no longer needed)
- Anthropic papers download in `data/download.py` — still works but not central; downgrade priority
- Bias-archaeology evaluation goal — superseded by sparse-attention efficiency-vs-quality goal

### Decisions made (pivot)
- Project keeps the name "The Mimic" (now: mimic human reading patterns)
- Base model unchanged: Qwen 2.5 0.5B-Instruct
- Three-mechanism architecture: sinks + sliding window + DEX, each independently ablatable
- Training approach: zero-train baseline → continued-pretraining + LoRA heal → DEX overlay
- Eval suite: lm-evaluation-harness (MMLU, HellaSwag, ARC, GSM8K) + LongBench + WikiText perplexity + efficiency metrics (FLOPs/latency/memory/KV cache) + behavioral probes
- Pareto frontier as deliverable: efficiency × quality across ablation grid
- Cognitive validation (eye-tracking alignment via GECO/ZuCo) is Phase 8 stretch goal, not required

### Gates pending (revised phase plan)
- Phase 1 (baseline + sinks-only eval) — needs Colab Pro setup
- Phase 2 (data pipeline) — pull FineWeb-Edu subset + UltraChat + Tulu
- Phase 3 (sliding window impl) — write `arch/sliding_window.py`, `arch/sinks.py`, `arch/patch_qwen.py`
- Phase 4 (Stage 1 heal) — LoRA + 1B tokens on Colab A100
- Phase 5 (DEX) — implementation + overlay fine-tune
- Phase 6 (hyperparameter sweep) — Pareto mapping
- Phase 7 (REPORT.md) — full eval + writeup
- Phase 8 (cognitive alignment) — optional stretch

### Next actions (pick one)
1. **Implement `arch/sinks.py` and run zero-train baseline** — fastest first signal; no training needed; validates StreamingLLM approach on Qwen 0.5B
2. **Write Colab notebook for Phase 1** — sets up the Colab Pro environment, validates model loads, runs inference baseline
3. **Update `data/download.py`** to pull FineWeb-Edu subset (replaces the bias-distillation data strategy)
4. **Write `eval/efficiency.py` + `eval/quality.py`** — measurement harness ready before any model variants exist
5. **Pause The Mimic, return to AI Ops venture** — W21 scoreboard still due 2026-05-18

### Risks acknowledged
- 0.5B may be too small to show sparse benefits cleanly — SWAA tested ≥1.5B
- DEX code release status needs verification (may need to implement from paper)
- Multi-hop reasoning degradation is expected; magnitude at 0.5B is unbounded
- Three-mechanism interaction effects are unstudied — ablation grid will surface this
- Cognitive-analog framing is rhetorical until/unless Phase 8 runs

### Open questions surfaced by pivot
- If sinks-only zero-train already gives 30%+ KV savings at <2% quality loss, is the rest of the project worth the effort? (Phase 1 answers this.)
- Does cognitive-alignment Phase 8 actually move the needle on the deliverable, or is it window dressing? (Defer decision to after Phase 7.)
- Repo rename consideration: project still about "mimicking" — is the name still right? Default: keep "the-mimic" since it now genuinely mimics human reading patterns. Open to rename if you find one that fits better.

---

## 2026-05-16 (third update same session) — Phase 1 zero-train baseline shipped

### Pathfinder findings (Haiku scout)
- Python 3.14.4 + transformers 5.8.0 + torch 2.11.0 already installed; CPU-only (no CUDA on this machine).
- `datasets` and `accelerate` installed this turn.
- **`transformers.SinkCache` was REMOVED from mainline** in PR #41107 (Oct 2025). Moved to a separate `custom_generate` repo.
- Repo state matches design spec; `arch/` previously empty.

### Pivot in implementation strategy
Rather than vendor the removed SinkCache, implement sinks as a 4D additive attention mask. For Phase 1 (zero-train, quality-only) this is strictly simpler:
- No custom Cache subclass needed
- No RoPE re-rotation logic needed
- Pure function, pure forward pass
- Real KV-cache efficiency work deferred to Phase 1b (later)

### Shipped this update
- `arch/__init__.py`, `eval/__init__.py` — package markers
- `arch/sinks.py` — pure 4D sink+window mask function (~25 LOC excl. docstrings). Vectorized, validates inputs, no allocations beyond the mask itself.
- `eval/baseline.py` — comparison script: load Qwen 2.5 0.5B, run full vs sink-masked perplexity on wikitext-2 samples
- Smoke test run successfully end-to-end (CPU, 8.5s for n=2 at seq=256)

### Code review (sonnet, adversarial)
Reviewer verified the load-bearing assumption by reading installed transformers 5.8.0 source:
- `Qwen2Model.forward` → `create_causal_mask` → `_preprocess_mask_arguments` early-exits on 4D tensor, passing it through to `eager_attention_forward` which adds it directly to attention weights. **The 4D mask path is confirmed working.**
- `arch/sinks.py`: SHIP (no changes — vectorized, correct edge cases, no NaN risk because `window>=1` invariant guarantees every row has at least one allowed position).
- `eval/baseline.py`: one PATCH applied — derive `device` from model for future GPU compatibility.

### First quantitative result (very small smoke)
```
config: n_sinks=4, window=64, seq_len=256, n_samples=2
full attention:   loss=3.1210  perplexity=22.67
sinks+window:     loss=3.3390  perplexity=28.19
delta:            +0.218 loss  +24% perplexity relative
allowed positions: ~26% of full
elapsed:          8.5s on CPU
```
This says: at aggressive sparsity (~26% of attention positions retained), perplexity rises ~24% with no healing fine-tune. Expected order of magnitude. **n=2 is not a real measurement** — this is a working-pipeline signal, not a Pareto point.

### Gates passed
- Phase 1 (pipeline + zero-train baseline) — first signal recorded; the architecture, mask logic, and eval harness all work end-to-end.

### Next actions (pick one)
1. **Hyperparameter sweep on the same harness**: vary window size (32 / 64 / 128 / 192) and n_sinks (0 / 4 / 8) at larger sample size (n=20-50). Maps the local Pareto frontier without any new code. ~15-30 min CPU.
2. **Extend to a real benchmark** beyond perplexity: wire in `lm-evaluation-harness` for MMLU/HellaSwag/ARC/GSM8K under the sink mask. More effort but more credible.
3. **Move to Phase 3** (sliding window architectural fine-tune on Colab Pro A100). Skip ahead of full Pareto mapping; assume the sweep would show the same pattern.
4. **Pause** and return to AI Ops venture (W21 scoreboard due Monday 2026-05-18).

### Verified facts (no need to re-check)
- 4D `attention_mask` propagation works in transformers 5.8.0 Qwen2 path
- `attn_implementation="eager"` honors custom masks (other implementations may not)
- Qwen 2.5 0.5B loads on CPU in fp32 in ~1s after first download
- wikitext-2-raw-v1 downloads cleanly via `datasets`
- Sample forward pass at seq_len=256 takes ~2s on CPU

---

## 2026-05-18 — Phase 1 hyperparameter sweep + Codex external review

### Sweep result (20 samples, seq_len=256, ~8 min CPU)

PPL ratios (sparse / full); full baseline PPL = 22.67:

```
sinks \ window |     32 |     64 |    128 |    192
--------------------------------------------------
              0 | 30.057 | 25.593 | 18.929 |  3.899
              4 |  1.404 |  1.151 |  1.042 |  1.006
              8 |  1.360 |  1.136 |  1.038 |  1.003
```

Saved to `eval/results/sweep_20260518T011005Z.json`.

### Headline finding
**Clean reproduction of Xiao et al. 2024.** Without sinks, sliding-window attention collapses catastrophically — even retaining 75% of context (window=192) gives 3.9× PPL degradation. Adding just 4 sink tokens at the same window drops the ratio to 1.006 (essentially free). The cognitive analogy (anchoring matters) has direct empirical backing on Qwen 2.5 0.5B in our setup. 8 sinks is not meaningfully better than 4 — **4 is the right default.** Best near-free quality config: `sinks=4, window=128` (1.042 PPL ratio at ~50% attention retention).

### External code review (Codex)
Sanjith ran the modules through Codex. Codex confirmed the core mask logic is correct and the NaN invariant holds (`window >= 1` guarantees every row has at least one allowed position). Returned 6 actionable items.

Applied this update (4 patches):
1. `forward_loss`: move `input_ids` and mask to model device; add `use_cache=False` to disable KV allocation since we're not testing cache behavior
2. `baseline.py` main: replace approximate sparsity-ratio print with exact count from the mask (`torch.isfinite(mask).sum()`) reported as both dense and causal-denominator ratios
3. New helper `assert_mask_affects_logits` in `baseline.py`: runs a window=1 logit comparison at sweep startup to detect silent regression if a future transformers version stops honoring 4D masks
4. Fixed `sweep.py` docstring (said JSONL, actually writes JSON)

Wired the assert into `sweep.py` startup — it now runs a one-time sanity check on the first sample before the grid loop.

### Deferred Codex items
5. **Sample selection refactor** — current `pick_long_samples` takes first-N-long-enough wikitext articles and doesn't backfill if tokenization drops short ones. Codex recommends concat-and-chunk WikiText into fixed seq_len blocks for stronger statistical baseline. Legitimate critique but it changes the experimental design materially — flagged for explicit decision before applying.

### Verified facts added
- Sweep run takes ~8 min for 12 configs × 20 samples on CPU at seq_len=256
- 4D mask path verified by both (a) the previous sonnet reviewer reading transformers 5.8.0 source AND (b) the new runtime assert
- 4 sink tokens is sufficient; 8 doesn't help meaningfully
- Without sinks, sliding-window attention is broken even at window=192 (out of seq_len=256)

---

## 2026-05-18 (later same day) — Phase 3 design + code shipped

### What changed (from Codex's external review)
Codex's feedback substantially tightened the Phase 3 plan in five places:

1. **Train WITH the sparse mask active.** Healing fine-tune under full attention would let the model keep relying on long-range access it won't have at eval. The mask must be on during training.
2. **Randomize window during training** (∈ {64, 128, 256}) so the model learns robustness across the sparsity range rather than overfitting one pattern.
3. **Three discrete stages**, not one mush run: Stage 1 sparse CPT on raw text → Stage 2 sparse SFT on instructions → Stage 3 conditional repair targeting the worst-hit bucket.
4. **"Visible metacognition" not chain-of-thought dump.** Train the model to surface uncertainty, assumptions, and quick-checks (the four patterns listed in `PHASE3_FINETUNE.md` §2), not to ramble pseudo-private reasoning.
5. **Combined mask: causal AND (sink OR window) AND not_pad.** If padding is dropped from the fusion the model silently learns to attend to pads. Implemented in `arch/training_mask.py`.

Also wise: **don't train on proprietary model outputs as main corpus.** Codex's caution updated. The hand-curated `metacognitive_seed_v0` is original writing demonstrating the *style* — not transcripts of any commercial model.

### Files shipped this update
- `PHASE3_FINETUNE.md` — full design doc with all five Codex integrations
- `arch/training_mask.py` — combined 4D mask (causal+sink+window+padding) + window sampler
- `data/download_phase3.py` — pulls the Codex mix (40% wiki, 20% science, 15% philosophy, 15% instruction-as-completion, 10% dialogue for CPT; separate ratios for SFT)
- `data/raw/_metacognitive_seed_v0.py` — 8 hand-curated demonstrations of the four metacognitive patterns
- `configs/heal_cpt.yaml` — Stage 1 sparse CPT config
- `configs/heal_sft.yaml` — Stage 2 sparse SFT config (resumes from Stage 1 adapter)
- `train_sparse.py` — QLoRA + custom `SparseMaskCollator` that injects the combined 4D mask per batch with optional window randomization
- `notebooks/heal_colab.ipynb` — full Colab Pro orchestration (mount → install → download → smoke → Stage 1 → Stage 2 → eval → Drive sync)

### Deliberate omissions
- **Philosophy corpus** deferred to v1 of the CPT data (high-quality open philosophy text is fragmented; Stage 1 v0 runs without it)
- **Patch-Qwen monkey-patch** (`arch/patch_qwen.py`) not needed — the data-collator approach achieves the same result without modifying model internals
- **Stage 3 repair** is conditional on Stage 2 eval; no code shipped until we know which bucket needs help

### Eval extension still needed (flagged in notebook cell 8)
`eval/sweep.py` currently runs on the base model. For Phase 3 evaluation we need it to accept `--adapter-path` and load the LoRA adapter before measuring. Trivial extension; not yet shipped.

### Open questions for Sanjith (from PHASE3_FINETUNE.md §8)
1. **GitHub vs Drive** for Colab access? Notebook defaults to `git clone`.
2. **Pro+ for one month, or staged across months on Pro?** Default: staged.
3. **Stage 3 trigger**: unconditional or only if eval surfaces a clear bucket dent? Default: conditional.
4. **Seq length**: start at 512 or 1024? Default: 512.

### Next action
Sanjith opens `notebooks/heal_colab.ipynb` on Colab Pro, sets the repo path (cell 1), runs the smoke cell (cell 5) to verify the sparse-mask training path works on Colab's environment, then launches Stage 1 (cell 6).

If the smoke fails: most likely failure points are (a) Colab's transformers version differs and the 4D mask path changes — re-run the `assert_mask_affects_logits` check; (b) bitsandbytes incompatibility with Colab's CUDA version — fall back to plain LoRA without 4-bit quantization.

---

## 2026-05-18 (closing Stage 1 prep) — local validation + eval-adapter loop closed + Phase 4 design sketched

### Stage 1 readiness gaps closed
- Ran `_metacognitive_seed_v0.py` → 8 examples in `data/raw/metacognitive_seed_v0.jsonl` (verified valid JSON)
- Unit-tested `arch/training_mask.py` with 7 properties: shape, dtype, causality, sinks-allowed, window-allowed, padding-blocked, NaN-safety (every query row has ≥1 finite key). All pass.
- Extended `eval/baseline.py` and `eval/sweep.py` with `--adapter-path` argument: when set, loads the LoRA via `peft.PeftModel.from_pretrained`, attempts `merge_and_unload` for clean inference (falls back to unmerged if quantized), then runs the sweep on the post-FT model. Closes the loop: Stage 1 finishes → run sweep with `--adapter-path outputs/heal_cpt` → get post-CPT PPL ratios → fill the north-star table in `PHASE3_FINETUNE.md` §5.

### Adversarial review launched (background)
Sonnet adversarial reviewer running against all Phase 3 modules (training_mask, train_sparse, download_phase3, metacognitive_seed, both YAML configs). Will report when done.

### Phase 4 design shipped
`PHASE4_RL.md` — GRPO-based reasoning RL plan with four stages:
- **4a Math RL** (GSM8K, verifiable numeric reward) — first concrete RL stage
- **4b Code RL** (MBPP, execution feedback)
- **4c Format/metacognitive compliance** (regex-based, cheap)
- **4d RLAIF** (Claude judge via Claude Code subscription, stretch goal)

Key design constraints baked in:
- GRPO over PPO (no value model = cheaper on Colab Pro)
- KL anchor against Phase 3 SFT checkpoint prevents catastrophic forgetting of metacognitive style
- Sparse mask interaction during RL is flagged as the open implementation question (Option A: easy + possibly wrong; Option B: correct + complex). Default: try A first as smoke test.
- 0.5B may be below the effective scale for reasoning RL — pilot with 200-prompt subset before committing compute.

### Next action
Same as before: Sanjith opens the Colab notebook and runs Stage 1. Reviewer notification will arrive separately; any patches needed get applied before the user opens the notebook.

When Stage 1 finishes on Colab and returns the adapter directory, run:
```
python -m eval.sweep --n-samples 30 --seq-len 512 --windows 32 64 128 256 \
  --sinks 0 4 8 --adapter-path outputs/heal_cpt
```
That fills in the post-CPT row of the north-star table.

---

## 2026-05-18 (closing the Phase 3 review) — 6 reviewer patches applied

### Sonnet adversarial review found 2 hard fails, 1 silent corruption, 1 crash-before-train

**C2 + H1 (combined fix, the critical one)** — `train_sparse.py`:
- `SparseMaskCollator` was setting `tokenizer.pad_token = tokenizer.eos_token` and not masking labels at pad positions. Combined with eos-as-pad, this would have trained the model to predict eos from any short context, corrupting the eos embedding.
- Also flagged: the 4D mask may be silently dropped between collator and `model.forward` depending on trl/transformers version interactions. No runtime guard existed.
- **Fix shipped**: collator now also writes `labels` with pad positions set to `-100` so loss isn't computed on pads. Added `verify_sparse_pipeline()` that runs one forward pass with the collator's mask vs a diagonal-only mask; if losses match within 0.1, the mask is being dropped and we abort before wasting compute. Called from `main()` immediately before `trainer.train()`.

**H3 (zero training signal)** — `data/download_phase3.py`:
- The OASST2 raw row format produces 1-message "conversations" via `_oasst_to_messages` (every row is one message; proper conversations require threading by `message_tree_id`). Every dialogue example written to `phase3_sft.jsonl` would have been useless for training.
- **Fix shipped**: SFT dialogue bucket swapped from `OpenAssistant/oasst2` → `HuggingFaceH4/ultrachat_200k`, which is natively multi-turn with the canonical messages schema. Removed the now-unused `_oasst_to_messages` formatter (no dead code).

**H2 (silent no-op training risk)** — `train_sparse.py`:
- If LoRA wrap fails silently (zero trainable params), training would burn compute doing nothing. Added a hard assert in `build_model` after `print_trainable_parameters()`.

**M2 (loading-script warning)** — `data/download_phase3.py`:
- `armanc/scientific_papers` requires `trust_remote_code=True` to stream. Added to `_stream_load`.

**M3 (crash before training)** — both `configs/heal_*.yaml`:
- `report_to: "wandb"` raises `wandb.errors.UsageError` at trainer init if not logged in. Changed default to `"none"`; users opt in to wandb after running `wandb login` in the notebook.

**L1 (off-target seed example)** — `data/raw/_metacognitive_seed_v0.py`:
- Example 8 (AI ethics) was structured with bold headers — closer to a legal memo than the conversational-prose voice the seed is supposed to demonstrate. Rewritten as single-voice flowing prose that naturally surfaces the three concerns within sentences. Regenerated `metacognitive_seed_v0.jsonl`.

### Verified post-patch
- `arch/training_mask.py` — 7 unit-test properties still pass (unchanged)
- `_metacognitive_seed_v0.py` — regenerates 8 examples cleanly to jsonl
- All other files compile + import without syntax errors

### Not patched (deliberate)
- **L2** (bitsandbytes version pin) — Colab Pro typically has compatible version; would just add noise to `requirements.txt`. Defer until we see an actual incompatibility.
- **M5** (n_sinks=0 + window=1 docstring warning) — degenerate config, not in any shipped config; not worth the docstring noise.

### Status going into Colab
All Codex + sonnet review findings either fixed or explicitly deferred with reason. Stage 1 should now train cleanly on Colab Pro A100:
1. `verify_sparse_pipeline` guards the load-bearing mask-path assumption with a runtime check before training starts
2. Pad-position label masking prevents eos corruption
3. UltraChat replaces OASST2 for SFT, giving real multi-turn dialogue signal
4. wandb removed from defaults so trainer init doesn't crash

User opens `notebooks/heal_colab.ipynb` and runs cells 1-5 (smoke) → if green, cell 6 (Stage 1).

---

## 2026-05-19 — Stage 1 CPT completed on Colab Pro A100; eval is a mixed result

### Trl 1.x patch chase before training launched
5 distinct API breakages hit iteratively before training started, all eventually patched (commits 35867a8, 8c76800, 9cb1391, 839a438, 4a5822c locally; not yet pushed to GitHub — need `gh auth login` first):
- `SFTConfig(max_seq_length=...)` removed → use `SFTConfig.max_length` directly
- `SFTTrainer(tokenizer=...)` → `processing_class=...`
- `SFTConfig.dataset_text_field` defaults to "text"; must pass `None` explicitly for messages-mode
- `warmup_ratio` deprecated → compute `warmup_steps` from ratio × total_steps
- `datasets` library removed `trust_remote_code` AND loading-script support; forced corpus swaps: `armanc/scientific_papers` → `HuggingFaceFW/fineweb-edu`, OASST2 (SFT) → `ultrachat_200k`, no_robots split `train_sft` → `train`

Pattern recorded in `~/superbrain/sub-brains/cs/mistakes/mistake-1223026a.md`: pre-scout library API drift via adversarial review BEFORE first training run, not iteratively. Iterative patch-chase cost 4 test cycles × ~10 min each before sonnet adversarial review caught the remaining 3 in one pass.

### Stage 1 actual run
- 3750 steps × 2.21s/it = 2h17m on A100 (batch 4 × grad_accum 4 = effective 16, 60K samples / 16 = 3750 steps for 1 epoch — matches)
- Train loss: 2.67 → 2.45 over the epoch, final reported `train_loss=2.481`
- Output: saved to `outputs/smoke_sparse/` (because the user ran the smoke cell which had its `output_dir` override active, but `max_steps=50` was apparently ignored — likely user-edited cell or trl 1.x precedence change). Renamed to `outputs/heal_cpt/` to match Stage 2 config's `resume_adapter_from`.
- Adapter: ~17MB LoRA weights (`adapter_model.safetensors`), with checkpoint-3000 and checkpoint-3750 dirs preserved.
- Backup synced to `/content/drive/MyDrive/the-mimic-outputs/`.

### Post-Stage-1 eval sweep (Phase 1 baseline comparison)

Phase 1 baseline (`sweep_20260518T011005Z.json`, no adapter):
```
sinks \ window |     32 |     64 |    128 |    192
              0 | 30.057 | 25.593 | 18.929 |  3.899
              4 |  1.404 |  1.151 |  1.042 |  1.006
              8 |  1.360 |  1.136 |  1.038 |  1.003
```

Post-Stage-1 (`sweep_20260519T214432Z.json`, adapter merged):
```
sinks \ window |     32 |     64 |    128 |    256
              0 | 16.080 | 15.015 | 14.841 |  1.000
              4 |  1.404 |  1.142 |  1.036 |  1.000
              8 |  1.361 |  1.125 |  1.033 |  1.000
```
(window=256 column trivially = 1.000 because seq_len=256; this is full attention.)

Dense baseline PPL: 22.67 → 23.33 (**+2.9%, slight regression**).

### Honest interpretation: mostly a wash on cells we care about
- **sinks=0 column improved 22-46%** — randomized-window training during Stage 1 made the model genuinely more robust to sparse attention without sink anchors. Real gain, but in degenerate territory (we never deploy sinks=0).
- **sinks=4 and sinks=8 cells barely moved** (≤1% change). These are the production-relevant configs. They were already at 1.04-1.16 in Phase 1 (close to ceiling); Stage 1 had little room to improve them.
- **Dense baseline slightly regressed** (+2.9% PPL). Small but real.

Net: Stage 1 traded a meaningful gain on sinks=0 (which doesn't matter) for a small loss on dense (which does), and produced negligible change on the cells we'd actually use. The healing thesis got **weak empirical support** — mostly because the dent the thesis was healing was smaller than expected at the cells of interest.

### Stage 2 decision: proceed, with shifted eval focus
The healing-by-perplexity question is now answered (weakly). Stage 2's thesis is different and uncovered: SFT on diverse instruction data + metacognitive seed under the same sparse mask should shape **response style**, not perplexity. The eval to actually run after Stage 2 isn't another sparse sweep — it's the **probe set** (`probes/v1.md`) measuring metacognitive patterns, plus a confirmatory sparse sweep to verify we didn't lose what we have.

User launched `!python train_sparse.py --config configs/heal_sft.yaml` ~21:45 UTC. Expected: ~4-6 hr on A100. Resumes from `outputs/heal_cpt/` adapter.

### Files touched this session (local only, not yet pushed)
- `data/download_phase3.py` — trust_remote_code removal, dataset swaps (commits 35867a8, 8c76800)
- `train_sparse.py` — trl 1.x compatibility patches (commits 9cb1391, 839a438, 4a5822c)
- All 6 superbrain stub nodes filled + ingested (decision, mistake, source, preference, anomaly, cross-link)
- 5 stale brainstem preferences patched to `domain: core` (cleanup)

### Next action when Stage 2 completes
1. Backup the Stage 2 adapter to Drive
2. Run probe-set eval (need to write `eval/probes.py` — not yet implemented; was flagged in PHASE3_FINETUNE.md as Stage 2 deliverable)
3. Run sparse sweep on Stage 2 adapter to confirm no regression
4. Update north-star table in `PHASE3_FINETUNE.md` §5
5. Then decide on Phase 4 RL or pause
