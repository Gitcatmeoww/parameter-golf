# Parameter Golf — Ideas

*Last updated 2026-04-29. 19 ideas across two research rounds, ranked for an autoresearch loop with a ~3 GPU-hr quick-start budget.*

> **Note on cross-references.** This doc references sibling files `technique_map.md`, `pr_lineage.md`, `curriculum.md`, `fundamentals.md`, and `credit_request_drafts.md` (see "Cross-references" at the bottom and inline mentions like "see `technique_map.md` § *Architecture Research*"). **None of those files currently exist in `autoresearch/`.** They are planned but not yet authored. Treat all such inline references as informational placeholders — do not block the loop on resolving them, and do not chase them as missing context.

---

## TL;DR — Top 3 to start with

**Calibrated for: limited model-training experience, ~3 GPU-hr quick-start budget, deadline 2026-04-30 (one day from this doc's last-updated date).** These three ideas each require fewer than 25 lines of code, have clear parameter spaces, and almost never produce negative results when the tuning range is respected. They cover three mostly separate axes — loss regularization, attention-logit calibration, LR schedule — and stack cleanly enough for a day-one loop. Run them in this order. Given the one-day window, prefer #1 (z-loss, ~3 LoC) and #2 (QK-norm ablation, ~5-10 LoC) before committing to #3 (WSD, ~20 LoC).

| # | Idea | LoC | GPU-min/run | Expected | Why first | Source |
|---:|---|---:|---:|---|---|---|
| 1 | **Z-loss + softcap retune** | ~3 | ~30 | +0.0005–0.003 BPB | Cheapest knob in the file. 3 lines, near-zero risk, almost never negative when α is in range. | Gemma2/PaLM2 |
| 2 | **QK-norm ablation + LR retune** | ~5-10 | ~30 | ±0.001–0.003 BPB | Cheap ablation on an already-present mechanism; can validate or simplify the root baseline while probing a higher LR ceiling. | Qwen3 + root baseline |
| 3 | **WSD LR schedule** | ~20 | ~30 | +0.001–0.004 BPB | Drop-in LR replacement; model stays the same; gives the training budget more time at peak LR. | MiniCPM |

**Starting point for all three**: work from `train_gpt.py` at the repo root — that is the readable, editable development file (1126 lines). The files inside `records/` are compressed lzma+base85 blobs (the final submitted artifacts), not editable source. To understand what PR #1493 adds on top of the root baseline, read the PR diff on GitHub.

**Estimated combined ceiling** depends on which baseline you actually start from:

- From the **repo-root naive baseline** (val_bpb ≈ 1.224, the path you'll actually take here): ~+0.005–0.010 BPB combined → ~1.219–1.214 BPB. Still well above the leaderboard frontier, but a clean signs-of-life non-record submission demonstrating each technique works at this scale.
- *If* you later port the same three techniques on top of the **PR #1493 stack** (val_bpb 1.0810, requires de-golfing the lzma+base85 blob first): ~1.071–1.076 BPB. This is the leaderboard-relevant number but not the path the day-one plan is taking.

Even one of the three landing cleanly on the repo-root baseline is publishable as a non-record signs-of-life PR — that is the goal here.

**Higher-ceiling ideas also exist.** Gram Newton-Schulz, ResComp, and Embedding-Int4 + Hadamard are in this document (see *Expert-track ideas* below and ranks 4–6 in the combined table). Each has a larger expected gain — but each also requires ML systems expertise to implement (kernel compilation, GPTQ-Cholesky inner-loop surgery, rotation-algebra byte accounting). Approach them after you've shipped at least one PR.

The detailed instructions for the Top 3 are in **Top 3 in detail** below; the autoresearch loop wiring is in the next section. The remaining 16 ideas are kept further down as bench depth.

---

## Autoresearch setup

This section is the standing operating procedure — read once, then refer back per idea.

### Goal

Use the working proxy metric to **rank parameter-space points cheaply**, then verify the top-1 or top-2 stacked combination at full 8×H100 scale.

### Trial function signature

```python
def trial(params: dict) -> float:
    """
    Mutate train_gpt.py per `params`, train on the proxy, return proxy_val_bpb.
    Lower is better. Should complete in ~3-5 minutes per call on 1xH100.
    """
    apply_params_to_train_script(params)        # write env vars or patch the script
    proxy_log = run_proxy_training(seconds=180) # 1xH100, scaled-down model + 1 shard
    return parse_proxy_val_bpb(proxy_log)
```

The proxy you already have configured should produce one floating-point `proxy_val_bpb` per call. Autoresearch (or any black-box optimizer — Optuna, Ax, simple grid) wraps `trial`.

### Budget allocation — ~3 GPU-hr

| Phase | What | GPU-hr | Trials |
|---|---|---:|---:|
| 1. Cheap-knob sweep | Z-loss + softcap + QK-norm ablation + activation ablation + WSD on the proxy | ~1.0 | 20 |
| 2. Top-3 single-axis sweeps | Z-loss α + softcap, QK-norm apply_to + qk_gain_init + LR, WSD stable_fraction + decay_shape | ~1.0 | 15 |
| 3. Final 8×H100 verification | Stack proxy-best of phase 2; 1–2 single-seed runs | ~1.0 | 1–2 |

If a phase-2 trial is *worse* than baseline on the proxy, **drop the idea** and move on; don't burn GPU-hr trying to recover it.

### Stop conditions per idea

Both thresholds are in **val_bpb** (bits-per-byte, lower = better). The proxy runs the same evaluation on fewer tokens/shards, so it reports val_bpb directly — no unit conversion needed. The proxy threshold is set higher than the full-run threshold because proxy estimates are noisier; a technique that clears 0.003 on the proxy will typically show a smaller but real signal on the full run.

| Condition | Metric | Threshold | Action |
|---|---|---|---|
| Proxy negative across 3+ trials | proxy `val_bpb_delta` | < 0 | Drop the idea, log the negative result |
| Proxy positive — promote to phase 3 | proxy `val_bpb_delta` | ≥ **0.003 BPB** | Move to phase-3 8×H100 single-seed run |
| Phase-3 single-seed positive — verify | full `val_bpb_delta` | ≥ **0.002 BPB** | Run a 3-seed full-scale pass (separate budget) |
| Budget exhausted | — | 3 GPU-hr spent | Submit non-record post-mortem PR with sweep log |

*0.002 BPB is the competition's minimum meaningful improvement bar (≈ the noise floor of a 3-seed full run). 0.003 BPB as the proxy promotion gate gives a ~1.5× buffer for proxy noise.*

### Proxy-vs-real correlation sanity check

Before spending the budget, run **one** baseline-vs-baseline sanity check: same baseline on proxy and on full 8×H100, twice each. If the proxy ranks the two runs in the same order as the full evaluator, the proxy is trustworthy. If not, fix the proxy first; everything downstream is unreliable otherwise.

### Concrete ergonomics

- **One env-var per param** — make every param in the parameter spaces below an env var (`QK_NORM_APPLY_TO=both`, `Z_LOSS_WEIGHT=1e-4`, etc.) so the trial function only sets env vars and runs `torchrun`. No source patches per trial.
- **Cache the proxy data** — proxy uses 1 shard of FineWeb-SP1024; pre-tokenize once.
- **Log the negative results** — even abandoned trials are useful in the eventual non-record post-mortem PR. Save `proxy_val_bpb`, `train_loss_curve`, `params` per trial.

---

## Top 3 in detail

### #1 — Z-loss + softcap retune

**The pitch.** The Gemma2 and PaLM2 training recipes add an auxiliary loss term $\alpha \cdot (\log Z)^2$ — where $Z = \sum_i \exp(\text{logit}_i)$ is the softmax partition function — that pulls logit magnitudes toward zero. The effect: logits stay bounded, softmax saturates less, and the Int6 quantization grid fits the weight distribution better post-training. The code change is **3 lines inside the existing loss computation**. At parameter-golf scale, several NanoGPT speedrun forks at <100M have reported small but consistent BPB gains, and it almost never makes things worse when $\alpha$ is in the range $[10^{-5}, 10^{-3}]$.

The second axis: the baseline has `logit_softcap = 30.0`. Z-loss and softcap interact (both regularize logit magnitude), so re-sweeping the softcap is free once z-loss is in.

**Starting point.** Work directly from `train_gpt.py` at the repo root (the readable development file). The records folder contains compressed blobs — not editable. For the day-one learning loop, do **not** de-golf or port the PR #1493 stack first; just test z-loss on the readable baseline. If it lands, porting it onto a frontier stack is a separate follow-up.

Locate the cross-entropy loss computation. The change:

```python
# Before:
loss = F.cross_entropy(logits, targets)

# After:
loss = F.cross_entropy(logits, targets)
log_z = torch.logsumexp(logits, dim=-1)   # partition function per token
z_loss = args.z_loss_weight * (log_z ** 2).mean()
loss = loss + z_loss
```

Add `z_loss_weight` as an env-var-settable argument (default 0.0). Done.

**Autoresearch parameter space.**

```python
params_grid = {
    "Z_LOSS_WEIGHT": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
    "LOGIT_SOFTCAP": [20, 25, 30, 40],
}
# 20 cells; ~3 min each on proxy = ~1 GPU-hr for full sweep
# Trim to 8 cells by fixing softcap=30 first, then sweeping jointly
```

**Proxy plan.**

1. Baseline: `Z_LOSS_WEIGHT=0, SOFTCAP=30` (= current root-baseline default). Record `proxy_val_bpb`.
2. Sweep z-loss weight alone (5 cells, ~15 min). Find best α.
3. Then sweep softcap with best α (4 cells, ~12 min).
4. Joint sweep of best 4 cells (~12 min). Total: ~40 min proxy time.

**What success looks like.** Proxy val_bpb drops 0.001–0.003 at best α. Phase-3 verification on 8×H100 confirms a clean non-record improvement on the repo-root baseline. If you later port the same change onto a stronger stack such as PR #1493, that becomes a separate leaderboard-oriented follow-up.

**Risks.**

- α > 1e-3 causes logit collapse early in training. Start at 1e-5 and work up.
- If logit_softcap is already saturating the benefit, z-loss adds nothing — that's fine; the test costs 3 lines and 30 min.

---

### #2 — QK-norm ablation + LR retune

**The pitch.** The repo-root baseline already applies `RMSNorm` independently to Q and K **before** the QK dot product. The useful day-one experiment is not "add QK-norm"; it is to make that existing behavior configurable and test whether the current both-sides setting is actually optimal at this scale, especially when paired with a slightly higher learning rate. This is still cheap: ~5-10 lines to gate q/k normalization and expose a couple of env vars.

QK-norm is used in Qwen3 (May 2025), Chameleon, and ViT-22B. Counter-evidence: Cohere Tiny Aya (Feb 2026) dropped it for long-context optimization. At parameter-golf's 4k context, that counter-evidence does not apply. The question here is narrower: does the root baseline's current `both` setting survive ablation, or is there a simpler/better variant?

**Starting point.** Work from `train_gpt.py` at the repo root.

Locate the `CausalSelfAttention.forward` method. The baseline today already contains:

```python
q = F.rms_norm(q, (q.size(-1),))
k = F.rms_norm(k, (k.size(-1),))
```

Turn that into a small ablation surface:

```python
if args.qk_norm_apply_to in ("both", "q_only"):
    q = F.rms_norm(q, (q.size(-1),))
if args.qk_norm_apply_to in ("both", "k_only"):
    k = F.rms_norm(k, (k.size(-1),))
# qk_norm_apply_to == "off" leaves both untouched
```

Use the existing `QK_GAIN_INIT` knob as the scale/gain axis; no new learned normalization scalar is required for the day-one version.

**Autoresearch parameter space.**

```python
params_grid = {
    "QK_NORM_APPLY_TO": ["both", "q_only", "k_only", "off"],
    "QK_GAIN_INIT": [1.0, 1.5, 2.0, 3.0],
    "PEAK_LR_MULTIPLIER": [1.0, 1.2, 1.5],   # test the raised-LR hypothesis
}
# Focus on QK_NORM_APPLY_TO × PEAK_LR_MULTIPLIER first (12 cells, ~36 min proxy)
```

**Proxy plan.**

1. Baseline: `QK_NORM_APPLY_TO=both`, `QK_GAIN_INIT=1.5`, `PEAK_LR_MULTIPLIER=1.0`. Record proxy val_bpb.
2. Sweep `QK_NORM_APPLY_TO × PEAK_LR_MULTIPLIER` (12 cells, ~36 min). The key question is whether the current `both` setting is actually best once LR moves.
3. Fix the best (apply_to, LR) pair and sweep `QK_GAIN_INIT` (4 cells, ~12 min). Total: ~48 min proxy.

**What success looks like.** Proxy val_bpb improves modestly at the best (apply_to, gain, LR) combo, or the current `both` setting survives the ablation and you document that the baseline was already in the right place. Either outcome is useful. A real win here is a clean non-record improvement on the repo-root baseline, not an immediate leap to frontier BPB.

**Risks.**

- Because the root baseline already has both-side q/k RMSNorm, this experiment may simply confirm that the current default is best. That is still a valid negative result.
- The root baseline already has `QK_GAIN_INIT=1.5`, so the real interaction to watch is `apply_to × qk_gain_init × LR`, not "QK-norm versus no QK mechanism at all."

---

### #3 — WSD LR schedule (Warmup-Stable-Decay)

**The pitch.** The current repo-root trainer uses a simple wallclock-aware linear warmdown: LR stays flat until the last `WARMDOWN_ITERS` worth of time, then linearly decays toward zero. The **Warmup-Stable-Decay** variant (MiniCPM, Hu et al. 2024) adds a deliberate *stable plateau* at peak LR before the decay starts. Effect: the model spends more of the 600 s budget at peak LR rather than spending the tail of training on a simple linear slide. The code change is **~20 lines** replacing the existing LR schedule function — zero model changes.

**Starting point.** Work from `train_gpt.py` at the repo root. Find the `lr_mul` function in the training loop.

```python
def get_lr_wsd(step, warmup_steps, stable_frac, total_steps, min_lr_frac=0.1):
    if step < warmup_steps:
        return step / warmup_steps                      # warmup
    stable_end = int(total_steps * stable_frac)
    if step < stable_end:
        return 1.0                                      # stable plateau
    decay_steps = total_steps - stable_end
    t = (step - stable_end) / decay_steps
    if args.wsd_decay_shape == "cosine":
        return max(min_lr_frac, 0.5 * (1 + math.cos(math.pi * t)))
    return max(min_lr_frac, 1.0 - t)                    # linear
```

Gate behind `USE_WSD=1` env var; otherwise fall back to existing schedule.

**Autoresearch parameter space.**

```python
params_grid = {
    "STABLE_FRACTION": [0.40, 0.50, 0.60, 0.70],
    "WSD_DECAY_SHAPE": ["linear", "cosine"],
}
# 8 cells; ~3 min each = ~24 min proxy
```

**Proxy plan.**

1. Baseline: existing warmdown schedule. Record proxy val_bpb.
2. Sweep grid (8 cells, ~24 min). Look for the stable_fraction sweet spot; cosine and linear are usually within 0.001 of each other.
3. Fix best cell; run on 8×H100 for final verification.

**What success looks like.** Proxy val_bpb drops 0.001–0.004. Training loss curve is smoother, and the final 8×H100 run shows a clean non-record improvement on the repo-root baseline. If that lands, later porting the same scheduler change onto a stronger stack is a separate follow-up.

**Risks.**

- Because the root trainer's current warmdown is wallclock-aware, noisy step-time estimates can blur whether the plateau or the simpler schedule is really winning. Check the full loss curve, not just the final number.
- `STABLE_FRACTION > 0.75` leaves too few steps for decay and can prevent full convergence. Stay in [0.40, 0.70].

---

## Stacking the top 3

The three touch mostly separate axes — loss regularization, attention-logit calibration, and LR schedule — so they stack cleanly enough for a beginner loop. Stacking strategy:

1. Run #1 (Z-loss) first in the autoresearch proxy loop (5–8 cells, ~20 min). It's the cheapest and gives you a confidence anchor. Bank the best α.
2. Layer #2 (QK-norm ablation) on top of the best z-loss config. Sweep `apply_to` + LR first, then `QK_GAIN_INIT`.
3. Layer #3 (WSD) on top. The stable_fraction sweep is fast (8 cells, ~24 min).
4. Final 1-seed 8×H100 verification of the stacked config vs the repo-root baseline. If stacked proxy delta ≥ 0.003 BPB and the full run is still positive, run a 3-seed non-record verification pass. Porting the stack onto PR #1493 is a separate frontier follow-up, not part of the day-one path.

**Honest expectation**: at this budget, expect 2 of 3 to land clearly and 1 to be neutral. Z-loss is the highest-confidence win (near-zero downside risk); QK-norm is the highest-upside; WSD is the most consistent but least exciting. Even just z-loss alone is a valid non-record contribution if it lands > 0.002 BPB delta with a sound ablation.

**After the quick-start budget**: if all three land, the natural next **model-side** layer is #R2.4 (MuonClip, ~25 LoC). If you explicitly broaden scope beyond `train_gpt.py`-only edits, then #R2.9 (FineWeb-Edu upsampling, ~30 LoC) becomes interesting. The expert-track ideas (Gram Newton-Schulz, ResComp, Embedding-Int4 + Hadamard) are fully detailed in the next section — approach them once you've shipped a first PR and have a sense of how the training loop behaves.

---

## Expert-track ideas (requires ML systems expertise)

These three ideas have the highest ceiling in the Tier-1 group but require non-trivial ML systems work. They were previously listed as Top 3 and are preserved here with full detail. **Fix the starting point**: all references below use PR #1493's folder, not the Scylla folder (which was removed by PR #1806 on 2026-04-26).

### #E1 — Gram Newton-Schulz drop-in for Muon

**The pitch.** The standard Newton-Schulz-5 iteration used inside Muon ($X \leftarrow aX + (bA + cA^2)X$ where $A = XX^T$) is mathematically equivalent to a faster form that iterates on the small **Gram matrix** instead of the rectangular weight matrix — 40–50 % faster optimizer step, 2× faster on rectangular MLP projections. At parameter-golf's 85 ms/step regime, that's ~700 extra training steps inside the 600 s budget for free. No accuracy tradeoff.

> ⚠️ **Reference to verify before acting.** The source is attributed to a Tri Dao blog post (Feb 2026) and a `Dao-AILab/gram-newton-schulz` GitHub repo. Verify both exist before spending GPU budget — the blog URL and repo path should be confirmed against current GitHub state.

**Starting point.** Work from `train_gpt.py` at the repo root.

- Locate `zeropower_via_newtonschulz5` in `train_gpt.py` (~30 lines).
- Vendor in the Gram form from `Dao-AILab/gram-newton-schulz`. The drop-in is ~50–80 LoC plus a CuTeDSL or symmetric-GEMM kernel dependency.

**Autoresearch parameter space.**

```python
params_grid = {
    "GRAM_NS_ITERATIONS": [3, 4, 5],
    "GRAM_NS_RESTART_AFTER": [1, 2, 3],
    "GRAM_NS_INTERMEDIATE_DTYPE": ["fp32", "bf16"],
}
# 18 cells; ~3 min each = ~1 GPU-hr for full sweep
```

**Expected gain.** +0.003–0.010 BPB (comes from extra training steps, not from a direct quality change). Pure wall-clock win; near-zero risk once the kernel compiles.

**Effort.** ~80 LoC + kernel dependency. **Difficulty: Advanced** (CuTeDSL compilation on H100).

**Risks.**

- Hopper-native CuTeDSL kernel may not build cleanly; fallback to `torch.matmul` Gram form gives ~40 % FLOP reduction, less wall-clock win.
- BF16 intermediates unstable on some weight matrices early in training; default to fp32 for first 500 steps.

---

### #E2 — ResComp (compensation-aware GPTQ)

**The pitch.** Standard GPTQ Cholesky aligns the quantized output of column $j$ with the *previously-compensated* (already-quantized) weights of columns $0..j-1$. [ICLR 2026, arxiv 2604.07955] shows this is sub-optimal: the right calibration target is the **original full-precision weights**. Drop-in alteration to the row-quantization update; everything else (ordering, damping, calibration data) stays the same. Consistent ~0.002–0.005 BPB gain at Int6; math is scale-invariant so no "doesn't transfer to small models" caveat.

> ⚠️ **Reference to verify before acting.** Attributed to arxiv 2604.07955 and a `list0830/ResComp` GitHub repo. Verify both exist and that the ICLR 2026 acceptance is current before spending GPU budget.

**Starting point.** Work from `train_gpt.py` at the repo root.

- Find the GPTQ pass (search for `cholesky` or `gptq` — the merged record uses Full-Hessian GPTQ from PR #1060). Locate the per-row inner loop.
- The change: in the error update line, replace the *compensated previous columns* with the *original* full-precision columns when computing the error to propagate forward.

**Autoresearch parameter space.**

```python
params_grid = {
    "RESCOMP_TARGET": ["prev_compensated", "orig_weight"],  # the technique IS this toggle
    "DAMPING_LAMBDA": [1e-5, 1e-4, 1e-3, 1e-2],
}
# 8 cells; ~3 min each = ~25 min total
```

**Expected gain.** +0.002–0.005 BPB. Low end likely if the Int6 distribution is already near-Gaussian.

**Effort.** ~70 LoC. **Difficulty: Advanced** (GPTQ-Cholesky inner-loop surgery; easy to introduce a sign bug).

---

### #E3 — Embedding-only Int4 + Hadamard rotation

**The pitch.** SP8192 token embeddings are the single largest block in the 16 MB artifact: $8192 \cdot 512 \cdot 1\,\text{byte} = 4$ MB at Int8. Drop them to Int4 with a random-orthogonal Hadamard rotation pre-quantization (à la QuaRot / SpinQuant) and reclaim ~2 MB to spend on an extra layer or wider MLP — both worth ~0.005–0.010 BPB. The rotation kills row-wise outliers so Int4 can quantize cleanly.

**Starting point.** This is **not** a root-baseline-compatible day-one idea. It assumes a de-golfed frontier stack that already has GPTQ on embeddings (for example the lineage around PR #1394). Locate the embedding GPTQ pass there and modify that path.

**Autoresearch parameter space.**

```python
params_grid = {
    "EMBED_BITS": [4, 5, 6],
    "EMBED_ROTATION": ["none", "blocked_hadamard_64", "full_hadamard"],
    "EMBED_QUANT_GROUP_SIZE": [64, 128, 256],
    "REINVEST_FREED_BYTES": ["extra_layer", "wider_mlp", "no"],
}
# Trim to ~12 cells; ~50 min proxy
```

**Expected gain.** +0.002–0.010 BPB (most upside from reinvested freed bytes, not the bit drop alone).

**Effort.** ~50 LoC. **Difficulty: Advanced** (embedding-unembedding tying + rotation accounting; byte budget math is non-trivial).

**Critical risks.**

- Every variant must be re-checked against the 16 MB cap with actual compressed bytes — Brotli compresses Int4 worse per element but better per bit.
- The embedding-unembedding tying in the baseline complicates the rotation: verify logits are bit-exact before vs after on a small forward pass before starting the full sweep.

---

# Round 1 — Cross-trunk combinations

*Five ideas seeded by the gap analysis in `technique_map.md`. Each names the gap it fills, which existing PRs it combines, what a realistic starting point is, and a difficulty rating. These are higher-ceiling but riskier than the Top 3.*

## Idea R1.1 — Scylla tokenizer × 3-layer depth recurrence × Legal TTT

**Gap:** the Scylla tokenizer + Full-Hessian GPTQ + XSA-all stack has **no TTT, no depth recurrence, no parallel residuals**. The public-SOTA SP8192 stack (#1493, 1.0810) has all three. Nobody has stacked Scylla × the SP8192-era eval tricks.

> ⚠️ **Scylla record removed (2026-04-26).** PR #1184 (Scylla 0.9485) was merged on 2026-04-23 but subsequently removed by PR #1806 on 2026-04-26. The folder `records/track_10min_16mb/2026-03-31_Scylla_FullGPTQ_XSA11_FA3_0.9485/` **no longer exists on upstream main**. This idea is partially blocked: the Scylla tokenizer base must be re-derived or re-submitted before you can start from it. In the meantime, the actionable sub-task is testing **Muon-in-TTT (#1148) + entropy-adaptive epochs on the SP8192 stack** — the component that was never tested on Scylla — using PR #1493 as the base.

**Combines:**
- #1184 (Scylla tokenizer + Full-Hessian GPTQ + XSA-all) — *blocked pending re-submission*
- #1493 (3-layer depth recurrence + parallel residuals + QK-Gain 5.25 + Legal TTT)
- #1148 (Muon-in-TTT + entropy-adaptive epochs)

**What to build (interim):** start from `train_gpt.py` at the repo root; apply the PR #1493 diff (depth recurrence, parallel residuals, QK-Gain, legal TTT), then port in Muon-in-TTT (#1148) and entropy-adaptive epochs from #1148 — these were never merged into #1493 and remain untested on the current SOTA stack.

**Risk:** TTT was explicitly tested and found *neutral* on the Scylla stack ("0.9491 with TTT, 0.9491 without"). But #1184 used classical SGD TTT; Muon-in-TTT (#1148) + entropy-adaptive epochs was not tested there. Worth a single-seed check.

**Expected win:** Muon-in-TTT alone was +0.0012 BPB over fixed-3-epoch SGD on its own baseline; stacked onto #1493 the range is +0.001–0.003 BPB.

**Difficulty:** **Intermediate** — the code exists in #1148 and #1493; the work is the diff + ablation.

---

## Idea R1.2 — GatedDeltaNet (#1791) × Full-GPTQ × Scylla

**Gap:** #1791 (GatedDeltaNet / Flash Linear Attention, 1.0339) uses **SP8192** and **standard Int6 + zstd**. It has not been combined with Full-Hessian GPTQ (#1060) or the Scylla tokenizer (#1184).

> ⚠️ **Scylla record removed (2026-04-26).** The Scylla starting point no longer exists on upstream main (see R1.1 note). The actionable sub-task is combining #1791 + Full-Hessian GPTQ, which is independent of Scylla.

**Combines:**
- #1791 (GatedDeltaNet K_KVShare_Wider architecture)
- #1060 (Full-Hessian GPTQ) — *this combination is actionable now*
- #1184 (Scylla tokenizer) — *blocked pending re-submission*

**What to build (interim):** start from the #1791 `train_gpt.py`, replace the quantizer with Full-Hessian GPTQ from #1060. KV sharing stride=2 and delta-rule linear attention stay. Sanity-check: GDN layers have different weight statistics than softmax attention, so SDClip's `k=12.85` default may need a per-layer sweep.

**Risk:** GDN's state update uses an exponential gate; Int6 might quantize that gate poorly. Int8 on gate rows + Int6 elsewhere (mixed-precision #65 pattern) is a sensible fallback.

**Expected win:** +0.003–0.015 BPB off #1791's 1.0339, pushing toward 1.02 BPB.

**Difficulty:** **Advanced** — GatedDeltaNet + Flash Linear Attention + per-layer quant tuning.

---

## Idea R1.3 — TTT-aware architecture: a dedicated "adaptive" late block

**Gap:** `technique_map.md` flags **Architecture × TTT** as empty. Every TTT submission treats the architecture as fixed and trains LoRA adapters on top of it. Nobody has designed the *model* so that a specific subset of weights is cheap-and-effective to adapt at eval.

**Combines:**
- #549 / #1493 (legal score-first TTT)
- #1148 (Muon-in-TTT + entropy-adaptive epochs)
- #1767 (LoRA alpha/rank, warm-start A, WD=1.0)
- Add: one **adaptive block** (late in the stack) with a dedicated low-rank subspace that *only* gets trained during TTT

**What to build:** define an extra 12-th layer (or promote layer 10 to a special "adapter block") whose weights are frozen at training end but whose *rank-96 LoRA* is the canonical TTT target. Train the base weights during training as usual; during TTT, only touch that block. Because the block was *meant* to be TTT-d, its initialization, init scale, and width can be chosen to make the TTT loss landscape friendly.

**Starting point.** Work from `train_gpt.py` at the repo root.

**Risk:** if you just add a block, you pay parameters at training end for a feature you only use at eval. Budget is the real constraint — test whether a shared-parameter "rank multiplex" (reuse existing block weights, add an adapter head) gives the same behavior.

**Expected win:** current TTT wins ~0.012 BPB on the #1493 stack. A block designed to be adapted should give 1.5–2× that per fixed eval-time budget.

**Difficulty:** **Advanced** — novel architecture work, needs ablations to isolate the "designed-for-TTT" contribution from plain extra capacity.

---

## Idea R1.4 — Re-quantize after each TTT chunk

**Gap:** `technique_map.md` flags **Numerical × TTT** as empty. Current TTT holds the quantized base weights fixed and fine-tunes a LoRA in FP16/BF16. After 1,893 chunks of adaptation, the combined "base + accumulated LoRA" drifts far from the quantization grid — but nobody re-projects.

**Combines:**
- #1060 / #1184 (Full-Hessian GPTQ)
- #1493 (legal score-first TTT)
- Add: a **periodic re-quantization step** during TTT

**What to build:** every K chunks during TTT, merge the LoRA into the base, re-run Full-Hessian GPTQ with a freshly-sampled mini-calibration from recently-seen chunks, zero out the LoRA. Since Full-Hessian GPTQ with 64-batch calibration runs in ~7 s, doing it every 50 chunks adds ~30 s to eval time (well within budget).

**Starting point.** Work from `train_gpt.py` at the repo root.

**Risk:** If you re-quantize naively the model's behavior shifts and the LoRA has to re-learn — you might *lose* the TTT gains. Fix: do the re-quant at chunk boundaries that already exceed the 2.1 NLL "hard content" threshold, so you're amortizing across harder content anyway.

**Expected win:** 0.002–0.005 BPB; small but novel, and it's a Level-3 numerical-method × Level-4 eval-method combo that hasn't been explored.

**Difficulty:** **Intermediate** — straightforward engineering; the analysis is the interesting part.

---

## Idea R1.5 — "SyntaxOps" — extend CaseOps to punctuation/URL/number patterns

**Gap:** #1729→#1736 CaseOps invertibly moves English *casing* into operator tokens. But the same trick could be applied to **any redundant surface-level pattern**: `http://` prefixes, trailing `.` in sentences, thousands-separator commas in numbers, camelCase boundaries, etc.

**Combines:**
- #1729 / #1736 / #1787 / #1801 (CaseOps bijective byte-sidecar)
- #1184 / #1143 (tokenizer-search autoresearch idea)
- New: a **mined set of invertible transforms** that shrink the entropy of surface-level patterns without changing what the model scores on

**What to build:** mine FineWeb for the most common bigrams-that-are-really-just-format: URL scheme heads, Markdown syntax, code fences, numeric separators. For each, introduce an operator token that says "here comes this pattern" and strip the pattern from the body. Keep the decode path bit-exact. Score on original UTF-8 bytes (existing CaseOps sidecar pattern).

**Risk:** every operator token steals a slot from the 998-/1024-/8192-vocab budget. You want to measure bits-per-byte *saved* by the operator vs bits-per-byte *paid* by the extra vocab entry. Autoresearch-style sweeps (the #1143 Scylla methodology) are the right framing.

**Expected win:** if CaseOps alone is worth ~0.005 BPB on SP8192 (#1736 vs #1493), three more operator families could stack to 0.02 BPB — which is real but bounded; surface-level redundancy runs out.

**Difficulty:** **Beginner-to-intermediate** — no GPU math, no new optimizers. Mostly dataset exploration, Unicode-safe encoding/decoding, and per-operator ablation. Ideal for a first non-record submission.

---

# Round 2 — External-research ideas (autoresearch-ready)

*Inputs: recent compression/quantization papers, Chinese OSS frontier models (DeepSeek/Qwen/Kimi), human-memory neuroscience analogies. Each Tier-1/2 idea is parameterized for an autoresearch sweep.*

## Honest direction assessment

| Source | Verdict | Why |
|---|---|---|
| Recent compression/quantization papers (2024–2026) | **Strong fit.** | Small-LoC, parameterizable, validated at sub-1B scale. Q1 2026 papers like **Gram Newton-Schulz** (Tri Dao) and **ResComp** (ICLR 2026) directly extend techniques in the merged-frontier stack. |
| Chinese OSS architectural tricks | **Mixed.** | QK-norm (Qwen3), MuonClip (Kimi K2), WSD schedule (MiniCPM), GatedDeltaNet hybrid ratio (Qwen3-Coder-Next) port cleanly. **MLA, MoE, FP8 training mostly don't transfer to 33M params.** |
| Human memory neuroscience | **Mostly post-hoc relabel.** | CLS (fast/slow split) ≈ TTT, *which already wins on the leaderboard*; constructive memory ≈ what every neural net does; predictive coding too slow per FLOP. **One real hit**: Global Workspace Theory → Perceiver-style latent bottleneck, unexplored on the leaderboard. |

If you take one thing from this round: external research adds the most leverage in **quantization and optimizer math**, not architecture. Memory metaphors mostly produced names, not mechanisms.

---

## Tier-1 — top picks for autoresearch (small LoC, cheap parameter spaces)

### Idea R2.1 — Gram Newton-Schulz drop-in for Muon  *(full detail in Expert-track section, #E1)*

- **Source.** Tri Dao 2026 — blog post + `Dao-AILab/gram-newton-schulz` repo. ⚠️ Verify URL and repo before acting.
- **Gap.** `technique_map.md` § *Systems Optimization* — Muon is universal but the Gram-matrix form has not been tested in any merged record.
- **One-line.** Iterates NS on the small Gram matrix ($XX^T$) instead of the rectangular weight; 40–50 % faster optimizer step, ~700 extra training steps in the 600 s budget for free.
- **Effort / gain.** ~80 LoC + kernel dependency; +0.003–0.010 BPB. **Difficulty: Advanced.**

### Idea R2.2 — ResComp (compensation-aware GPTQ)  *(full detail in Expert-track section, #E2)*

- **Source.** *Rethinking Residual Errors in Compensation-based LLM Quantization*, ICLR 2026, arxiv 2604.07955, repo `list0830/ResComp`. ⚠️ Verify arxiv ID and repo before acting.
- **Gap.** `technique_map.md` § *Numerical Methods* — direct upgrade to the GPTQ-Cholesky pass used in the merged-frontier stack since PR #1060.
- **One-line.** Aligns GPTQ's quantization target to the original full-precision weights instead of the previously-compensated columns.
- **Effort / gain.** ~70 LoC; +0.002–0.005 BPB. **Difficulty: Advanced.**

### Idea R2.3 — QK-norm ablation + LR retune  *(full detail in Top 3 #2)*

- **Source.** Qwen3 (May 2025), also Chameleon, ViT-22B (Dehghani 2023). Almost free.
- **One-line.** The root baseline already uses q/k RMSNorm. The day-one experiment is to expose that existing normalization as an ablation surface (`both`, `q_only`, `k_only`, `off`) and retune LR / `QK_GAIN_INIT`.
- **Gap filled.** `technique_map.md` § *Architecture Research* — on merged record stacks this would be a fresh pre-attention normalizer; on the root learning baseline it is a direct ablation of an already-present mechanism. Both are useful, but do not confuse them.
- **Combines / extends.** Stacks with z-loss and LR tuning. Can also simplify the baseline away if `off` or a one-sided variant wins.
- **Why it matters at parameter-golf scale.** Validated at sub-100M in NanoGPT speedrun forks. **Counter-evidence**: Cohere Tiny Aya (Feb 2026) explicitly *removed* QK-norm because they were optimizing for long-context (>32k); at 4k seq-len the open question is whether the root baseline's current `both` default is actually best.
- **Autoresearch parameter space:**
  - `qk_norm_apply_to ∈ {both, q_only, k_only, off}`
  - `qk_gain_init ∈ {1.0, 1.5, 2.0, 3.0}`
  - `peak_lr_multiplier ∈ [1.0, 1.2, 1.5]`
- **Expected gain.** **±0.001 to +0.003 BPB**, mostly via a slightly better normalization/gain/LR combination or by confirming the current default.
- **Effort.** ~5-10 LoC. ~30 GPU-min final run. **Difficulty: Beginner.**

### Idea R2.4 — MuonClip (logit-aware QK rescale)

- **Source.** Kimi K2 tech report (Jul 2025).
- **One-line.** Wrap Muon updates with a per-step QK logit clip: rescale $W_q, W_k$ by $\sqrt{\tau / \max(\text{logit})}$ when attention logits exceed threshold $\tau$. Solves Muon's known late-training instability.
- **Gap filled.** `technique_map.md` § *Systems Optimization* — Muon is universal but no record clips its outputs by attention-logit signal.
- **Combines / extends.** Sits inside the existing Muon optimizer step. Stacks with QK-norm (different mechanism, different layer).
- **Why it matters at parameter-golf scale.** Honest skeptic note: MuonClip's documented win is at ≥7B where Muon does blow up. At 33M with 11 layers, vanilla Muon rarely diverges. **The angle here isn't stability — it's that the clip mechanism may regularize and let you push LR or weight decay further.** Cheap to falsify.
- **Autoresearch parameter space:**
  - `muonclip_tau ∈ [50, 200]` (logit threshold; log-uniform)
  - `muonclip_active_after_frac ∈ [0.0, 0.5]` (when to start clipping; useful if it hurts early)
- **Expected gain.** **±0.003 BPB** — could be negative if the clip is too aggressive at this scale. 30-min ablation tells you.
- **Effort.** ~25 LoC inside Muon. ~30 GPU-min final run. **Difficulty: Beginner-Intermediate.**

### Idea R2.5 — Logit z-loss + softcap retune  *(full detail in Top 3 #1)*

- **Source.** Gemma2 (DeepMind), PaLM2.
- **One-line.** Add an auxiliary loss term $\alpha \cdot (\log Z)^2$ where $Z$ is the softmax partition function — pulls logits toward zero-mean per row, reducing softmax saturation and improving quantization resilience.
- **Gap filled.** `technique_map.md` § *Architecture Research* — softcap is in the baseline (`logit_softcap = 30.0`), but z-loss is not. Independent of softcap.
- **Combines / extends.** Drops onto any record. Particularly synergistic with Int6 quantization (smaller logit magnitudes → more headroom in the int range).
- **Why it matters at parameter-golf scale.** Several speedrun forks at <100M have reported small but consistent BPB gains from z-loss. Cheap.
- **Autoresearch parameter space:**
  - `z_loss_weight α ∈ [1e-5, 1e-3]` (log-uniform)
  - `softcap ∈ {20, 25, 30, 40}` (the existing baseline)
- **Expected gain.** **+0.0005 to +0.003 BPB**. Almost never negative if α is tuned.
- **Effort.** ~3 LoC. ~30 GPU-min final run. **Difficulty: Beginner.**

### Idea R2.6 — Activation ablation around the `relu^2` baseline

- **Source.** Primer (So et al. 2021), ReLU Strikes Back (2024).
- **One-line.** The repo-root baseline already uses $\text{ReLU}(x)^2$. The experiment here is to treat that as the baseline and ablate nearby activations such as LeakyReLU² or GeLU², checking whether `relu^2` is actually the right simple choice.
- **Gap filled.** `technique_map.md` § *Architecture Research* — the current root trainer hard-codes `relu^2`, while most merged leaderboard records historically used LeakyReLU(0.5)². The gap is not "add ReLU²"; it is "measure the activation neighborhood around the current baseline."
- **Combines / extends.** Trivial swap; affects nothing else in the stack. Feeds into quantization-sensitive ideas like ResComp (#E2) by changing activation distributions.
- **Why it matters at parameter-golf scale.** Activation choice is one of the highest-uncertainty knobs in this regime. The point of the sweep is to see whether the current simple `relu^2` choice is already best, or whether a nearby alternative buys a small BPB gain.
- **Autoresearch parameter space:**
  - `activation ∈ {leaky_relu_sq_0.5, relu_sq, gelu_sq, leaky_relu_sq_0.75}` (categorical, 1 axis)
- **Expected gain.** **±0.003 BPB**. Could go either way; needs the autoresearch loop to settle.
- **Effort.** ~5 LoC. ~30 GPU-min final run. **Difficulty: Beginner.**

### Idea R2.7 — WSD (Warmup-Stable-Decay) schedule  *(full detail in Top 3 #3)*

- **Source.** MiniCPM (Hu et al. 2024, arxiv 2404.06395).
- **One-line.** Replace the long-warmdown LR schedule with **Warmup → Stable (constant peak) → Decay**. Less HP-sensitive than cosine; better fixed-budget loss; the "stable" phase length is the new knob.
- **Gap filled.** `technique_map.md` § *Systems Optimization* — every merged record uses linear warmdown. WSD is an alternate axis the leaderboard hasn't probed.
- **Combines / extends.** Drop-in replacement for the existing warmdown logic. Compatible with `MIN_LR=0.10` floor (PR #1787).
- **Autoresearch parameter space:**
  - `stable_fraction ∈ [0.4, 0.7]` (fraction of training at peak LR before decay starts)
  - `decay_shape ∈ {linear, cosine}`
- **Expected gain.** **+0.001 to +0.004 BPB**. Most of the win comes from spending less of the budget warming up.
- **Effort.** ~20 LoC in the LR scheduler. ~30 GPU-min final run. **Difficulty: Beginner.**

### Idea R2.8 — Embedding-only Int4 quant + Hadamard rotation  *(full detail in Expert-track section, #E3)*

- **Source.** QuaRot (Ashkboos et al. 2024), SpinQuant.
- **Gap.** `technique_map.md` § *Numerical Methods* — no record has gone below Int8 on the embedding matrix.
- **One-line.** Drop SP8192 embeddings to Int4 with a Hadamard rotation pre-quantization to kill outliers; reclaim ~2 MB to spend on an extra layer or wider MLP.
- **Effort / gain.** ~50 LoC; +0.002–0.010 BPB (most upside from reinvested freed bytes). **Difficulty: Advanced.**

### Idea R2.9 — Late-stage FineWeb-Edu upsampling (data curriculum)

- **Source.** DeepSeekMath (Shao et al. 2024), MiniCPM curriculum-tuning notes.
- **One-line.** In the last ~25 % of training, upweight the FineWeb-Edu subset (the high-quality educational filter). Pretraining-on-FineWeb is corpus-bounded but you can adjust *which* shards you draw from at the end.
- **Gap filled.** `technique_map.md` § (none — data-side techniques are absent from current map). New axis entirely.
- **Combines / extends.** Independent of every existing PR. Compatible with the coprime-stride loader (#1060).
- **Autoresearch parameter space:**
  - `edu_upsample_after_frac ∈ {0.5, 0.7, 0.85}` (when to start upsampling)
  - `edu_mix_ratio ∈ [0.3, 0.8]` (fraction of late-stage batches drawn from edu shards)
- **Expected gain.** **+0.001 to +0.005 BPB**, with risk of negative if edu shards over-narrow distribution.
- **Effort.** ~30 LoC in the data loader. ~30 GPU-min final run; cheaper proxy iterations possible. **Difficulty: Beginner.**

---

## Tier-2 — mid-effort, higher-upside

### Idea R2.10 — Hybrid GatedDeltaNet × softmax-attention ratio

- **Source.** Qwen3-Coder-Next, Qwen3.5 (Jan/Feb 2026 releases) — both adopted a *3:1 hybrid* of GDN to standard attention, not pure GDN.
- **One-line.** PR #1791 already does pure GDN at 1.0339 BPB. The Qwen lineage suggests a *hybrid* (most layers GDN, every-Nth layer standard softmax) is better. Sweep the hybrid ratio.
- **Gap filled.** `technique_map.md` § *Architecture Research* — pure GDN exists (#1791) and pure softmax attention exists; the hybrid is unexplored on the leaderboard.
- **Combines / extends.** Cleanly stacks on top of PR #1791 (which already has the GDN kernel) by inserting standard attention layers at chosen positions.
- **Autoresearch parameter space:**
  - `gdn_to_softmax_ratio ∈ {1:0, 3:1, 2:1, 1:1}` (4 categorical levels)
  - `softmax_layer_position ∈ {early, mid, late, every_third}`
- **Expected gain.** **+0.005 to +0.020 BPB** *over PR #1791* (i.e., possibly below 1.02 BPB) if the Qwen finding transfers; ±0.005 if it doesn't.
- **Effort.** ~60 LoC inside the block scheduler. ~45 GPU-min final run (#1791's eval is a touch slower than the SP8192 path). **Difficulty: Advanced.**

### Idea R2.11 — Factorized FFN low-rank

- **Source.** MobileLLM, Funnel-Transformer factorization.
- **One-line.** Replace $W_{\text{up}} \in \mathbb{R}^{d \times 4d}$ with $UV^{\top}$ where $\text{rank}(U) = r < d$. Saves ~$d \cdot (4d - r)$ params per layer; spend the savings on more depth.
- **Gap filled.** `technique_map.md` § *Architecture Research* — every merged record uses the unfactorized 3×–4× MLP. Factorized FFN is an unprobed axis.
- **Combines / extends.** Independent of attention/quant/optimizer choices.
- **Autoresearch parameter space:**
  - `ffn_rank_ratio ∈ {0.25, 0.5, 0.75, 1.0}` (1.0 = unfactorized baseline)
  - `factor_layers ∈ {all, deep_only, shallow_only}`
- **Expected gain.** **±0.005 BPB**. The win comes only if reinvested params (an extra layer or two) outpace the rank-truncation loss.
- **Effort.** ~30 LoC. ~30 GPU-min final run. **Difficulty: Intermediate.**

### Idea R2.12 — Perceiver-style latent bottleneck (the one memory-inspired pick)

- **Source.** Perceiver / Perceiver IO (Jaegle 2021); Goyal/Lamb/Bengio Global Workspace Transformer (2022). Inspired by Global Workspace Theory — Baars / Dehaene.
- **One-line.** In 1–2 mid-stack layers, replace full self-attention with cross-attention to **N=64 learnable latents**. Tokens read/write through the latent bottleneck. Saves params per replaced layer; the bottleneck forces information consolidation.
- **Gap filled.** `technique_map.md` § *Architecture Research* — bottleneck/latent layers are absent.
- **Combines / extends.** Independent of the rest of the stack. Plays well with parallel residuals (#1204).
- **Why it earns its place when other memory ideas don't.** The biological frame produces a concrete mechanism (a fixed-size workspace) that *forces* a specific architectural choice — the bottleneck dim N. That isn't true of "constructive memory" or "fast/slow split" (CLS), which mostly relabel existing techniques.
- **Autoresearch parameter space:**
  - `n_latents N ∈ {32, 64, 128}` (the workspace size)
  - `latent_layer_indices ∈ {[5], [4,7], [4,5,7,8]}` (which middle layers get replaced)
- **Expected gain.** **±0.005 BPB**. Could be negative if the bottleneck is too tight; the upside is parameter savings reinvested.
- **Effort.** ~80 LoC for the cross-attention block. ~30 GPU-min final run. **Difficulty: Advanced.**

### Idea R2.13 — D²Quant (Dual-Scale Quantizer + Deviation-Aware LayerNorm correction)

- **Source.** D²Quant (Feb 2026, arxiv 2602.02546), repo `XIANGLONGYAN/D2Quant`.
- **One-line.** **DSQ**: a dual-scale quantizer specifically for the down-projection rows of the MLP (a known sub-4-bit bottleneck). **DAC**: mean-shift adjustment within LayerNorm to counteract quantization-induced distribution shift.
- **Gap filled.** `technique_map.md` § *Numerical Methods* — extends GPTQ-family with a per-projection-class strategy. Stackable on ResComp (#E2).
- **Combines / extends.** Sits inside the GPTQ pass alongside SDClip; needs LN-mean adjustment in the forward path (a small ≤10-LoC change).
- **Autoresearch parameter space:**
  - `dsq_enabled ∈ {true, false}` (binary)
  - `dsq_scale_ratio ∈ [0.5, 2.0]` (the dual-scale knob)
  - `dac_enabled ∈ {true, false}`
- **Expected gain.** **+0.001 to +0.005 BPB** at Int6; potentially much more at Int4–Int5 (where it's actually targeted). Stack with embedding-Int4 (#E3) for compounded gains.
- **Effort.** ~150 LoC port (most of it in the GPTQ pass). ~30 GPU-min final run. **Difficulty: Intermediate-Advanced.**

### Idea R2.14 — Differential attention in 1–2 layers

- **Source.** Differential Transformer (Microsoft 2024).
- **One-line.** Two parallel softmax attention maps subtracted: $\text{attn}_1 - \lambda \cdot \text{attn}_2$. Reduces attention noise (the "always-on" tokens that distract heads).
- **Gap filled.** `technique_map.md` § *Architecture Research* — only one-stream attention everywhere.
- **Honest caveat.** Strong evidence at ≥1B on retrieval; weak evidence at <100M on FineWeb pretraining loss. Cheap to falsify in 1 ablation.
- **Autoresearch parameter space:**
  - `lambda_init ∈ [0.5, 1.0]` (the differential mixing)
  - `differential_layers ∈ {[0], [10], [4,5], all}` (which layers replace)
- **Expected gain.** **±0.003 BPB**. Likely neutral at this scale.
- **Effort.** ~20 LoC. ~30 GPU-min final run. **Difficulty: Beginner-Intermediate.**

---

## Tier-3 — explicitly skipped (one-line "skip because…")

| Idea | Skip because |
|---|---|
| Full FP8 training (DeepSeek-V3) | At 33M, matmuls are overhead-bound; FP8 speedup shrinks to ~1.2× and the loss-scaling tax eats it. |
| Shared+routed MoE (DeepSeek-V2/V3) | Routing overhead + load-balancing aux losses don't pay back at 33M total params. Wrong scale. |
| MTP with separate heads (DeepSeek-V3, StepFun MTP-3) | Each extra head is ~$V \cdot d$ params ≈ 4 MB at SP8192 — *bigger than the model*. Can't fit. |
| RETRO retrieval index | Index alone exceeds 16 MB unless heavily compressed; not realistic in budget. |
| BitNet b1.58 from-scratch ternary training | Needs ≥100B tokens for parity; 10-min budget is 1–2 % of that. |
| H-Net end-to-end byte tokenization | Large refactor, training-time cost unbounded; SP8192 is already strong. |
| Predictive coding nets | Slower per FLOP than backprop; loses to the 600 s budget. |
| DNC / Memory Networks | Doesn't scale beyond toy tasks; bio framing didn't translate. |
| MLA at 33M | Adds parameters without delivering its KV-cache win (irrelevant in speedrun). |
| Lightning Attention (MiniMax) | Win is at ≥32k context; speedrun uses 4k. |
| Spaced-repetition data curriculum | Sub-1 % gain, complex to implement, low ROI. |
| Schemata-in-tokenization (Bartlett-style) | No serious literature; BPE merges are statistical, not schematic. Vapor. |

---

# Combined ranking — all 19 ideas

Sorted by beginner ROI: effort (LoC + debug time) vs expected gain, **calibrated for limited ML-training experience**. Expert-track ideas (E1–E3) have higher ceilings but require ML systems expertise and are ranked accordingly.

| Rank | Idea | Round | Effort | Mid expected | Risk | Notes |
|---:|---|---|---|---|---|---|
| **1** | **#R2.5 Z-loss + softcap** ★ | R2-T1 | ~3 LoC | +0.002 | Very low | **Top 3.** Cheapest knob; beginner |
| **2** | **#R2.3 QK-norm ablation + LR retune** ★ | R2-T1 | ~5-10 LoC | +0.001 | Low | **Top 3.** Cheap ablation on an existing root-baseline mechanism; beginner |
| **3** | **#R2.7 WSD schedule** ★ | R2-T1 | ~20 LoC | +0.0025 | Low | **Top 3.** Drop-in LR swap; beginner |
| 4 | #R2.9 FineWeb-Edu upsampling | R2-T1 | ~30 LoC | +0.003 | Low-med | Independent data axis; beginner |
| 5 | #R2.6 Activation ablation around `relu^2` | R2-T1 | ~5 LoC | ±0.003 | Med | Baseline-validation sweep; cheap; beginner |
| 6 | #R2.4 MuonClip | R2-T1 | ~25 LoC | ±0.003 | Med | LR-ceiling angle, not stability; beg-inter |
| 7 | #R1.5 SyntaxOps | R1 | ~100 LoC (CPU) | +0.005–0.020 | Low-med | Beginner-friendly, no GPU; ideal first PR |
| 8 | #R2.14 Differential attention | R2-T2 | ~20 LoC | ±0.003 | Med-high | Cheap falsifier; beg-inter |
| 9 | #R1.4 Re-quantize during TTT | R1 | ~80 LoC | +0.0035 | Med | Empty quadrant, novel; intermediate |
| 10 | #R2.11 Factorized FFN | R2-T2 | ~30 LoC | ±0.005 | Med | Reinvestment-dependent; intermediate |
| 11 | **#E1 Gram Newton-Schulz** | Expert | ~80 LoC + kernel | +0.006 | Low (if compiles) | High ceiling; **needs ML systems expertise** |
| 12 | **#E2 ResComp** | Expert | ~70 LoC | +0.003 | Low | Direct GPTQ upgrade; **needs GPTQ expertise** |
| 13 | **#E3 Embed Int4 + Hadamard** | Expert | ~50 LoC | +0.006 | Med | Highest budget arbitrage; **needs rotation math** |
| 14 | #R2.13 D²Quant | R2-T2 | ~150 LoC | +0.003 | Med | Stack with ResComp; inter-advanced |
| 15 | #R2.10 Hybrid GDN ratio | R2-T2 | ~60 LoC | +0.012 | High | Highest upside in Tier-2; advanced |
| 16 | #R2.12 Perceiver bottleneck | R2-T2 | ~80 LoC | ±0.005 | High | One memory-inspired pick; advanced |
| 17 | #R1.1 Scylla × #1493 stack | R1 | ~200 LoC | +0.020+ | High | **Scylla record removed** — partially blocked |
| 18 | #R1.3 TTT-aware adaptive block | R1 | ~150 LoC | +0.018 (if it works) | High | Research-grade; longest payoff |
| 19 | #R1.2 GDN × Full-GPTQ × Scylla | R1 | ~400 LoC | +0.05–0.10 (if it works) | Very high | Highest possible win, also longest debug |

★ = Top 3 (beginner-calibrated). Ranks 1–7 are all ≤30 LoC or CPU-only; they are the right starting set for a first PR. The expert-track ideas (#E1–#E3) are preserved with full detail and belong in a follow-up grant cycle.

---

# What to do if your top-3 budget runs out

**You spent the 3 GPU-hr and at least one of Top 3 landed.** Submit it as a non-record improvement on the repo-root baseline. Then apply for a larger compute grant citing the validated technique. Roadmap: layer #R2.4 (MuonClip) on top as the next model-side gain; if you explicitly broaden scope, then #R2.9 (FineWeb-Edu upsampling) becomes interesting; after that, approach the expert-track ideas (#E1–#E3) with a stronger baseline. If you later port the whole stack onto PR #1493 or another frontier stack, *that* is the point where 1.07x-style leaderboard targets become relevant.

**You spent the budget and nothing clearly landed.** This is also valuable. Submit a non-record post-mortem PR with:
- Your proxy correlation table (proxy vs full)
- The autoresearch sweep log (every trial's params + proxy_val_bpb)
- Your interpretation of *why* the techniques didn't transfer at 33M scale
The community values negative results when the methodology is sound — see how PR #1184 explicitly tested-and-found-neutral the TTT branch.

**You found a new gap not in the technique map.** That's the best outcome. Document it in `technique_map.md` and propose 2–3 follow-up sweeps. The Scylla branch (#1184) reached SOTA precisely by autoresearch over a tokenizer space nobody had probed; the same methodology could find a new axis next.

---

# Cross-references

- `pr_lineage.md` — for the dependency graph and which PR each idea extends
- `technique_map.md` — for the empty quadrants this file fills
- `curriculum.md` — for the concept refreshers behind each technique
- `fundamentals.md` — for the math under the hood (SVD for Gram-NS, Hessian for ResComp, etc.)
- `credit_request_drafts.md` — for the compute-credit pitch that maps to these ideas
