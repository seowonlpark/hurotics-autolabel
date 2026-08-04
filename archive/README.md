# archive/

Artifacts that nothing in the pipeline reads any more. Kept rather than deleted because
each one is a record of a measurement or an agent verdict that was real when it was made
and is not regenerable — the code that produced it is gone.

Nothing under `archive/` is on any import path or any `Step` gate. Moving a file here is
the claim that no live code reads it; `stages/breakdown.py` and `freshness.check_all` were
both checked against this list.

| item | why it is here |
|---|---|
| `runs/_archived_baselines.json` | RF-era 23-feature baseline. Self-declared archived on 2026-08-03 when its run dir was deleted; the metrics were kept because that run is **not** in `experiments.jsonl`. Superseded by `runs/s2_ml_rf42`, which is RF at the champion 42-feature set and *is* still cited — by `stages/s2_ml/train.py`. Zero readers. |
| `runs/2026-08-03_run1` … `run3`, `run5` | `fusion_review.jsonl` — verdicts from the S4 fusion review agent. S4 was deleted 2026-08-04 along with `agents/s4_fusion.py`; with no fusion there is nothing these judge. |
| `runs/2026-08-03_run4` | `exceptions_review.jsonl` from the same era. `s1_exception` is still a live step, but this run predates the S4 removal and nothing reads dated run dirs — `run_pipeline._new_run_dir` only scans *today's* date to pick the next N. |

## What deliberately did NOT come here

Four things in `runs/` look archival and are not. They are load-bearing:

- `runs/s2_ml_abl_*` — globbed by `stages/breakdown.py` to build the ablation table.
- `runs/s2_ml_seedsweep.json` — read by `stages/breakdown.py` for the seed-stability row.
- `runs/s2_ml_rf42` — cited by `stages/s2_ml/train.py` as the reproduction for the
  ExtraTrees-over-RandomForest claim.
- `runs/s2_ml/roweval_lockbox.json` — the single-use rev8 read. It can never be
  regenerated, and `breakdown.py` reads it in three places.

The S4 stage code itself (`stages/s4_fusion/`, `agents/s4_fusion.py`, `stages/s2_ml/oof.py`,
`stages/s3_physics/run.py`, `orchestrator.py`, `stages/s1_clean/peek.py`,
`stages/s2_ml/audit.py`) is not copied here either. It was deleted from the worktree, not
archived, and `git show HEAD:<path>` still produces it.
