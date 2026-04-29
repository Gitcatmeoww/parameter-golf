# autoresearch (parameter-golf edition)

This is an experiment-loop skill for the OpenAI [Parameter Golf](https://github.com/openai/parameter-golf) challenge. It is adapted from Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) — same shape (one file
to edit, run, keep-or-revert, log, repeat) but reshaped for this challenge's constraints (16MB artifact cap, 10-min wallclock cap on 8×H100, fixed FineWeb eval).

The human edits THIS FILE (`program.md`) to evolve the loop. The agent edits `train_gpt.py` and the launch env vars.

---

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr28a`, `apr28b`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current `main`. Push it: `git push -u origin autoresearch/<tag>`. Subsequent pushes during the loop use `--force-with-lease` (see loop step 9) so `origin` always mirrors local — the TSV, not git history, is the audit trail for discarded experiments.
3. **Read the in-scope files**:
   - `README.md` — challenge rules, constraints, leaderboard.
   - `train_gpt.py` — the file you modify. Model, optimizer, training loop,
     int8+zlib serialization, eval.
   - `data/cached_challenge_fineweb.py` — data download script (read-only).
4. **Verify data exists**: confirm `./data/datasets/fineweb10B_sp1024/` has at least `fineweb_val_*.bin` and one `fineweb_train_*.bin`, and `./data/tokenizers/fineweb_1024_bpe.model` exists. If not, tell the human to run `python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1` (smoke) or higher for real iteration.
5. **Initialize results.tsv**: create `autoresearch/results.tsv` with just the header row (see "Logging results" below). Leave this file untracked.
6. **Confirm and go**: confirm with the human that the setup looks good.

Once you get confirmation, kick off the experimentation.

---

## Hard constraints (DO NOT violate)

These come from the challenge rules. Violating them invalidates the run.

- **16 MB artifact cap**: `code_bytes(train_gpt.py) + len(int8+zlib model) ≤ 16,000,000` (decimal, not 16 MiB). The script prints `Total submission size int8+zlib: <N> bytes` near the end. Always check this before declaring a win.
- **10-minute wallclock cap on 8×H100** (record-track only): keep `MAX_WALLCLOCK_SECONDS=600`. For non-record exploration on a single GPU you can push longer, but record any deviation in the experiment description.
- **No external compute, no validation peeking**: the eval harness inside `train_gpt.py` (the `eval_val` function and val_bpb computation) is functionally read-only. You may refactor it for speed but must produce identical val_bpb values. If you change anything in eval, prove equivalence.
- **No new dependencies**: do not modify `requirements.txt`. Use what's already there. (You can import freely from already-installed packages.)
- **No edits to `data/`**: dataset, tokenizer, and the download script are fixed.
- **Do not rename success/cap log markers**: the loop greps `^final_int8_zlib_roundtrip_exact` (success indicator + score), `^Total submission size` (artifact size — both lines), and `^peak memory` (VRAM). You may change WHAT they report (e.g., a different compression scheme — many records do, like `int6+lzma`), but keep the line prefixes intact so the loop can still detect completion.

---

## What you CAN do

- Modify `train_gpt.py` (everything outside the eval harness): architecture, optimizer, hyperparameters, training loop, batch size, model size, init, schedule, etc.
- Override hyperparameters via env vars without editing code (most early experiments should be this — see "Common env vars" below).
- Choose a different tokenizer variant (sp1024 / sp4096 / sp8192 / byte260) by re-pointing `DATA_PATH`, `TOKENIZER_PATH`, and `VOCAB_SIZE`. Larger vocabs cost more model bytes (embedding table) so weigh against the 16 MB cap.
- Pre-flight a code change with the MLX path (`train_gpt_mlx.py`) on a Mac for "does this parse and run for 20 steps without crashing" — fast, free, cheap. The MLX val_bpb is NOT the leaderboard number; only the CUDA `train_gpt.py` number counts.

## VRAM / memory

VRAM is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically. Record peak memory in the TSV.

## Simplicity criterion

All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or
better results is a great outcome — that's a simplification win. When deciding whether to keep a change, weigh the complexity cost against the improvement
magnitude. A 0.0005 val_bpb improvement that adds 50 lines of hacky code? Probably not worth it. A 0.0005 improvement from deleting code? Definitely keep.

---

## The first run

Your very first run should always be to establish the baseline, so run the training script as-is with default env vars on the platform you're on. Record
the result before changing anything.

---

## Running an experiment

The standard 1×H100 (cheap iteration) command, on a CUDA box:

```bash
RUN_ID=ar_<tag>_<N> \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py > run.log 2>&1
```

The standard 8×H100 (record-track validation) command:

```bash
RUN_ID=ar_<tag>_<N>_8gpu \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py > run.log 2>&1
```

The Mac MLX pre-flight (does my change parse and not crash):

```bash
RUN_ID=ar_<tag>_<N>_mlx \
ITERATIONS=20 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
python3 train_gpt_mlx.py > run.log 2>&1
```

**Always redirect both stdout and stderr to `run.log`. Do NOT use `tee` or let output flood your context window.**

### Common env vars (override without editing code)

Architecture: `NUM_LAYERS`, `MODEL_DIM`, `NUM_HEADS`, `NUM_KV_HEADS`, `MLP_MULT`, `VOCAB_SIZE`, `TIE_EMBEDDINGS`, `ROPE_BASE`, `LOGIT_SOFTCAP`.

Training: `ITERATIONS`, `WARMUP_STEPS`, `WARMDOWN_ITERS`,
`TRAIN_BATCH_TOKENS`, `TRAIN_SEQ_LEN`, `MAX_WALLCLOCK_SECONDS`, `SEED`.

Optimizer: `EMBED_LR`, `HEAD_LR`, `TIED_EMBED_LR`, `MATRIX_LR`, `SCALAR_LR`, `MUON_MOMENTUM`, `MUON_BACKEND_STEPS`, `BETA1`, `BETA2`, `ADAM_EPS`, `GRAD_CLIP_NORM`.

Eval/logging: `VAL_BATCH_SIZE`, `VAL_LOSS_EVERY`, `TRAIN_LOG_EVERY`.

The full list is in `train_gpt.py` lines 41–87.

---

## Reading out results

### After a CUDA `train_gpt.py` run

```bash
grep -E "^(final_int8_zlib_roundtrip|Total submission size|peak memory)" run.log
```

Expected output (5 lines on the current default script):

- `final_int8_zlib_roundtrip val_loss:<X> val_bpb:<Y> eval_time:...` — the leaderboard score (`val_bpb`, lower is better). Post-quantization, post-zlib. Use this, NOT the fp16 `val_bpb` printed earlier in the log.
- `final_int8_zlib_roundtrip_exact val_loss:<X> val_bpb:<Y>` — same numbers, 8 decimal places. **Use this for the TSV.** (Older script versions print only this line, so `final_int8_zlib_roundtrip_exact` is the stable success marker.)
- `Total submission size: <N> bytes` — uncompressed model + code. Informational only.
- `Total submission size int8+zlib: <N> bytes` (or `int6+lzma:` etc., whichever compression label is in use) — compressed model + code. **This is the 16 MB cap number.** It will be the smaller of the two `Total submission size` values. If > 16,000,000, the run is INVALID regardless of val_bpb.
- `peak memory allocated: <N> MiB reserved: <M> MiB` — divide MiB by 1024 for GB in the TSV.

If the grep output is missing `final_int8_zlib_roundtrip_exact`, the run crashed or didn't complete. Run `tail -n 80 run.log` to read the trace.

### After a Mac MLX `train_gpt_mlx.py` pre-flight

The MLX script prints fewer lines: no `Total submission size` and no `peak memory allocated:`. The pre-flight is purely a "does it crash" check, so:

```bash
grep -E "^(final_int8_zlib_roundtrip|serialized_model_int8_zlib)" run.log
```

If `final_int8_zlib_roundtrip` appears, the script ran end-to-end. The MLX `val_bpb` is NOT comparable to the leaderboard number — it's just confirming your code change parses, runs, and produces a finite number.

---

## Logging results

Log every experiment to `autoresearch/results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions). Do NOT commit this file; leave it untracked.

Header (6 columns):

```
commit	val_bpb	artifact_bytes	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb from `final_int8_zlib_roundtrip` (e.g. `1.224400`) — use `0.000000` for crashes
3. artifact bytes total (e.g. `15823104`) — use `0` for crashes; use the actual number even if > 16,000,000 so we can see how badly it busted
4. peak memory in GB, .1f (e.g. `78.4`) — use `0.0` for crashes
5. status: `keep`, `discard`, `crash`, or `invalid` (artifact > 16MB)
6. short text description of what this experiment tried

Example:

```
commit	val_bpb	artifact_bytes	memory_gb	status	description
a1b2c3d	1.224400	15823104	78.4	keep	baseline 9L/512d/sp1024
b2c3d4e	1.219200	15901280	78.6	keep	bump matrix_lr 0.04 -> 0.05
c3d4e5f	1.230000	15823104	78.4	discard	switch GeLU -> SwiGLU
d4e5f6f	0.000000	0	0.0	crash	double model width (OOM)
e5f6g7h	1.210000	17204800	81.0	invalid	14L deep — busts 16MB
```

---

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/apr28a`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Tune `train_gpt.py` (or just env vars) with an experimental idea.
3. `git add -u && git commit -m "<short experiment description>"`
4. Run the experiment (see "Running an experiment" above). Redirect to `run.log`. Do NOT use `tee`.
5. Read out the results — see "Reading out results" above for the exact grep (CUDA and MLX print different lines).
6. If the grep output is missing `final_int8_zlib_roundtrip`, the run crashed. Run `tail -n 80 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after a few attempts, give up on the idea.
7. Record the result in `autoresearch/results.tsv` (do NOT commit this file).
8. Decide:
   - **Keep**: artifact ≤ 16,000,000 AND `final_int8_zlib_roundtrip_exact val_bpb` is at least 0.0005 lower than the current branch tip's value (lower val_bpb = better). Advance the branch (keep the commit).
   - **Discard**: artifact ≤ 16,000,000 BUT val_bpb did not improve enough. `git reset --hard HEAD~1`.
   - **Invalid**: artifact > 16,000,000 regardless of val_bpb. `git reset --hard HEAD~1`. Note in description what would need to shrink.
   - **Crash**: same as discard, `git reset --hard HEAD~1`.
9. **Sync the branch to the fork**: `git push --force-with-lease origin autoresearch/<tag>`. This must run after every iteration (keep, discard, invalid, crash) so `origin` always mirrors local. Without this, the next push after a `git reset --hard` will fail non-fast-forward. `--force-with-lease` is safe — it refuses to push if the remote moved out from under you (e.g., a concurrent agent).

The TSV is the audit trail for discarded experiments — git history on the remote only reflects the "kept" lineage after a discard rewrites local. That's intentional; the TSV's `description` column is richer than commit messages anyway.

The keep threshold is intentionally tight (0.0005 val_bpb) because single-seed runs are noisy. When you find an idea that looks like a robust win, re-validate with 3 seeds (see "Submission protocol" below) before declaring it real. All gating, logging, and threshold comparisons use the same metric: post-quantized `final_int8_zlib_roundtrip_exact val_bpb`. Do not mix in fp16 `val_bpb` (printed earlier in the run) or `val_loss` (which is in nats per token, a different unit and scale).

**Timeout**: each 1×H100 experiment takes ~10 min training + ~1–2 min eval
overhead. If a run exceeds 15 min, kill it and treat as crash. For 8×H100
runs the wallclock is ~10 min total — same kill threshold.

**Crashes**: if it's something dumb (typo, missing import), fix and re-run. If
the idea is fundamentally broken, log "crash" and move on.

**NEVER STOP**: once the loop has begun, do NOT pause to ask "should I keep
going?". The human might be asleep. You are autonomous. If you run out of
ideas, think harder — re-read the README's "Requests for PRs" list, look at
what's already on the leaderboard, look at recent record PRs in `records/`,
combine previous near-misses, try more radical changes. The loop runs until
the human interrupts, period.

---

## Submission protocol (when something looks like a real win)

For a non-record submission, you don't need full statistical rigor — but you
DO need to submit something reproducible. When you've found an experiment
worth submitting:

1. **Re-validate with 3 seeds**: run the same config with `SEED=1337`, `SEED=42`, `SEED=2025`. Report the mean and per-seed values of `final_int8_zlib_roundtrip_exact val_bpb`.
2. **Do NOT autonomously classify a result as "record-eligible"**: the upstream README states the rule literally — "beat the existing SOTA by at least 0.005 nats" with p<0.01 evidence. That rule is in nats (per-token cross-entropy, `val_loss`), not val_bpb (bits-per-byte). The two are different units on different scales, and the conversion depends on the tokenizer's compression ratio on the val set, so there is no single fixed translation. If a candidate looks like it might clear the SOTA bar, surface it to the human with both `val_loss` and `val_bpb` per-seed numbers; let the human decide whether to submit as record vs. non-record.
3. **Verify the artifact is reproducible end-to-end**: run a clean clone, apply the same config, confirm val_bpb matches.
4. **Write up**: create a folder under `records/track_10min_16mb/` (or `records/track_non_record_16mb/`) per the submission instructions in the main README. Include `submission.json`, `train_gpt.py`, and the train logs.
5. **Open the PR** against `openai/parameter-golf:main` from your fork.
