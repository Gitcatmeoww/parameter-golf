# autoresearch (parameter-golf edition)

Scaffolding to run an autonomous experiment loop against the OpenAI
[Parameter Golf](https://github.com/openai/parameter-golf) challenge. Adapted
from Karpathy's [autoresearch](https://github.com/karpathy/autoresearch).

## How to start a run

1. Spin up a coding agent (Claude Code, Codex, etc.) in this repo with permissions enabled. From a 1×H100 RunPod or similar GPU box.
2. Make sure data is downloaded (see top-level README).
3. Prompt the agent:

   ```
   Read autoresearch/program.md and let's kick off a new experiment! Let's do the setup first.
   ```

The agent follows `program.md`: creates a branch, edits `train_gpt.py`, launches training, logs results to `autoresearch/results.tsv`, and either keeps or reverts based on val_bpb and the 16 MB artifact cap.

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
