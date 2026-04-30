# autoresearch (parameter-golf edition)

Scaffolding to run an autonomous experiment loop against the OpenAI
[Parameter Golf](https://github.com/openai/parameter-golf) challenge. Adapted
from Karpathy's [autoresearch](https://github.com/karpathy/autoresearch).

## Walkthrough: zero to a submitted PR

This walkthrough takes you from a fresh fork to an open PR on `openai/parameter-golf`. Estimated total time: **~5–6 hours of focused work** plus **~$15–25 of GPU spend** for a non-record submission. Each phase has a clear checkpoint so you know when to move on.

### Legend

- 🧑 = you do this (interactive, your laptop or browser)
- 🤖 = the agent does this (you prompt, then watch)
- ⏱ = wall-clock time estimate
- 💰 = GPU cost estimate

---

### Phase 0 — Local prerequisites (Mac dev box)

These are one-time, and most are already done if you've been following along.

| Step | What | Time |
|---|---|---|
| 0.1 🧑 | Fork `openai/parameter-golf` on GitHub | 1 min |
| 0.2 🧑 | Local clone, wire `origin` to your fork and `upstream` to OpenAI's | 5 min |
| 0.3 🧑 | Local Mac MLX smoke run to confirm env (`python3 train_gpt_mlx.py` with small `ITERATIONS`) | 5–15 min |
| 0.4 🧑 | This `autoresearch/` scaffolding committed and pushed to your fork | 1 min |

**Checkpoint**: `git remote -v` shows `origin` → your fork, `upstream` → OpenAI. `autoresearch/program.md`, `autoresearch/ideas.md`, `autoresearch/README.md` exist on `main` of your fork.

---

### Phase 1 — Cloud GPU setup (RunPod, ~1 hr)

Mac is fine for "does my code parse" but too slow for real iteration. You need a CUDA box.

| Step | What | Time | Cost |
|---|---|---|---|
| 1.1 🧑 | [Create a RunPod account](https://console.runpod.io), set up SSH key in Settings, add billing | 15–30 min | — |
| 1.2 🧑 | Deploy a **1×H100 SXM** pod using the [Parameter Golf template](https://console.runpod.io/deploy?template=y5cejece4j&ref=nl2r56th). Enable SSH terminal access. | 5 min | ~$2/hr while running |
| 1.3 🧑 | SSH into the pod (you should land in `/workspace`), clone your fork, set upstream | 5 min | — |
| 1.4 🧑 | Download data: `python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 4` (4 shards ≈ 800MB; enough for ~1 hr of training) | 5–10 min | — |
| 1.5 🧑 | Run baseline once to confirm everything works: `RUN_ID=baseline DATA_PATH=./data/datasets/fineweb10B_sp1024/ TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 torchrun --standalone --nproc_per_node=1 train_gpt.py > run.log 2>&1` | ~12 min | ~$0.40 |
| 1.6 🧑 | Verify final lines: `grep -E "^(final_int8_zlib_roundtrip|Total submission size|peak memory)" run.log` — should show `val_bpb ≈ 1.224`, `Total submission size int8+zlib: <≤16,000,000>`, `peak memory ≈ 78 GB`. | 30 sec | — |

**Checkpoint**: baseline `final_int8_zlib_roundtrip_exact val_bpb` is approximately `1.224` and the artifact size is under 16 MB. The pod's `/workspace/parameter-golf` is your working dir for everything below.

> ⚠️ **Don't forget to stop the pod when you're not using it.** RunPod bills by the hour; an idle 1×H100 still burns ~$2/hr.

---

### Phase 2 — Start the autoresearch loop

| Step | What | Time |
|---|---|---|
| 2.1 🧑 | On the pod, install Claude Code (or Codex), enable file-write and bash permissions for the `parameter-golf` directory | 5 min |
| 2.2 🧑 | Open the agent in `/workspace/parameter-golf` | 1 min |
| 2.3 🧑 | Prompt: `Read autoresearch/program.md and autoresearch/ideas.md and let's kick off a new experiment! Let's do the setup first.` | 1 min |
| 2.4 🤖 | Agent reads `program.md`, `ideas.md`, in-scope source files, proposes a run tag (e.g. `apr29a`), creates the branch `autoresearch/apr29a`, pushes to your fork, initializes `autoresearch/results.tsv` with the header row | 5 min |
| 2.5 🧑 | Confirm the setup looks right (branch created, tag matches, ideas.md is the queue), then say "go" | 1 min |
| 2.6 🤖 | Agent enters the loop. First iteration is the baseline (already verified in step 1.5; agent re-runs to anchor the TSV). Then it works through `ideas.md` Top 3 in order: z-loss → QK-norm → WSD. | ~3 hours |

**What's happening during 2.6**: per `ideas.md`'s plan, the agent should run a **proxy-vs-real correlation check first**: the same baseline on the proxy and on the real evaluator, ideally twice each. That is more expensive than a pure proxy smoke test, so if you're trying to stay within a very small first-day budget you may choose to defer the full 8×H100 half and treat the proxy as provisional. After that, the agent applies each Top 3 idea as a small `train_gpt.py` edit plus env-var tuning. Each full 1×H100 run is ~12 min wall-clock; with the agent's overhead (writing the change, kicking off the run, parsing results, deciding keep/discard), realistically count on **~15–20 min per experiment slot**, so **~3–4 experiments per hour**, or about **9–12 experiments over a 3-hour Phase 2**.

**Checkpoint** (after ~3 hr): `autoresearch/results.tsv` should have one header row plus the baseline plus 8–15 experiment rows. At least one of the Top 3 should be marked `keep` if any of them landed.

---

### Phase 3 — Monitor + decide

You don't need to watch the agent constantly, but periodic checks help catch dead-ends early.

**What to peek at** (run on the pod from a separate terminal, or scroll back in the agent's chat):

```bash
# How many experiments so far + last 5 results
wc -l autoresearch/results.tsv && tail -5 autoresearch/results.tsv

# What's the current branch tip val_bpb?
grep "^final_int8_zlib_roundtrip_exact" run.log | tail -1

# What's the agent currently editing?
git log --oneline -10
git diff HEAD~1 train_gpt.py | head -50
```

**When to interrupt the agent**:

- ✅ A Top 3 idea landed cleanly with margin > 0.001 val_bpb on single seed → time to validate with 3 seeds (Phase 4).
- ⚠️ Three consecutive crashes on the same idea → tell agent to skip it.
- ⚠️ ~$15+ spent without a keep → consider stopping; even one Top 3 landing is publishable.
- ⚠️ Agent is chasing a non-existent file (`technique_map.md` etc. — the cross-references warning should prevent this, but if it happens, redirect).

---

### Phase 4 — Validate + submit

This is where you get the most value from the agent — handing off the mechanical bits of submission while keeping the judgment calls.

| Step | What | Time | Cost |
|---|---|---|---|
| 4.1 🧑 | Tell the agent: "Idea X looks like a winner. Run the 3-seed validation per program.md submission protocol." | 1 min | — |
| 4.2 🤖 | Agent runs the same config with `SEED=1337`, `SEED=42`, `SEED=2025` on 1×H100, reports per-seed and mean `val_bpb` | ~40 min | ~$1.30 |
| 4.3 🧑 | Inspect the per-seed numbers. If all three beat baseline by ≥0.001 BPB, you have a real (non-record) signs-of-life submission. | 5 min | — |
| 4.4 🧑 | **Decide**: record-track or non-record? Per `program.md`'s submission protocol, do NOT let the agent autonomously classify as record-eligible (the rule is in nats, not val_bpb, with no clean conversion). For a first PR, **non-record is the right call** — you've got a clean signs-of-life win. | 5 min | — |
| 4.5 🤖 | Tell the agent: "Write up the submission as a non-record PR. Create the folder under `records/track_non_record_16mb/<date>_<descriptive_name>/`, include `submission.json`, the modified `train_gpt.py`, the three train logs, and a `README.md` explaining what was changed and the per-seed results." | 10 min | — |
| 4.6 🧑 | Review the generated submission folder and README. Adjust prose if needed. | 10 min | — |
| 4.7 🧑 | Push the final state, open the PR against `openai/parameter-golf:main` from your fork. Use `gh pr create` or the GitHub UI. Title format: `Non-record: <short description>`. Body: link to your TSV, the per-seed numbers, and one paragraph on what you tried and learned. | 10 min | — |

**Checkpoint**: PR URL exists. Done. 🎉

---

### Cost / time summary (realistic)

|  | Time | Cost |
|---|---|---|
| Phase 0 (local prereqs) | ~30 min total | $0 |
| Phase 1 (RunPod setup + baseline) | ~1 hr | ~$0.50 |
| Phase 2 (loop runs Top 3) | ~3 hr | ~$6 |
| Phase 3 (monitor) | passive | (inside Phase 2's $) |
| Phase 4 (validate + submit) | ~1.5 hr | ~$2 |
| **Total** | **~6 hr** | **~$10** |

The ~$15–25 estimate at the top has slack for one round of "the first idea was a dead end, try another from `ideas.md`."

## Files

- `program.md` — agent skill: setup, constraints, loop, submission protocol. The **human** edits this over time as you learn what works.
- `ideas.md` — the prioritized experiment backlog (Top 3 + bench depth). The **human** edits this; the agent reads it as the source of the next experiment in the loop. See `program.md` setup step 3 and loop step 2 for how it's wired in.
- `results.tsv` — per-experiment log (gitignored, regenerated per run).
- `.gitignore` — keeps `results.tsv` out of git.

## Branch convention

Each run lives on its own branch: `autoresearch/<tag>` (e.g. `autoresearch/apr28a`). Experiments commit incrementally; good experiments advance the branch tip, bad experiments are reverted via `git reset --hard HEAD~1`.

The branch is mirrored to your fork (`origin`) after every iteration via `git push --force-with-lease origin autoresearch/<tag>` — so `origin` always reflects the current local state (the kept lineage). Discarded experiments do not persist on the remote; the audit trail for failed attempts is `autoresearch/results.tsv` (description column), not git history. See `program.md` (loop step 9) for the full policy.

## Where to iterate

| Goal                                       | Where                                  |
| ------------------------------------------ | -------------------------------------- |
| "Does my code change parse and not crash?" | Mac MLX (`train_gpt_mlx.py`), 20 iters |
| Real iteration                             | 1×H100 RunPod, full `train_gpt.py`     |
| Record-track validation                    | 8×H100 RunPod, 3-seed average          |

The MLX val_bpb is NOT the leaderboard number — it's only useful for "did my edit break the script."
