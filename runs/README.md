# runs/

Everything the pipeline writes. The tree splits on one question — **can this file be rebuilt
from `data/raw` and the code in this commit?**

```
runs/
  breakdown.md          the one page written to be read; `python -m stages.breakdown`
  regen/                YES. safe to delete.
    s1_census/          inventory of data/raw
    s1_clean/           segments, quarantine, observations
    s2_ml/              the fitted champion and every metric refitted from champion_spec
    s3_physics/         label audit, rate audit, plausibility
  keep/                 NO. deleting destroys evidence nothing can bring back.
    s2_ml/              the experiment ledger, the proposals log, the spent lockbox
    ablations/          the measured comparisons behind champion_spec's rejected lines
    agent_runs/         one dated dir per paid agent run
```

A full `python run_pipeline.py` reproduces every byte under `regen/`. It reproduces nothing
under `keep/`.

## Deleting things

`rm -rf runs/regen` is safe and sometimes the right move — it is the fastest way to prove the
pipeline still builds from nothing. The next full run recreates it.

Do not delete anything under `runs/keep/`. Each file there is the only copy of a measurement
or a verdict that some claim in the code or in `OPERATING_POINTS.md` rests on:

| path | why it cannot come back |
|---|---|
| `keep/s2_ml/roweval_lockbox.json` | The single-use `rev8` read (`stages/s2_ml/dataset.py`). Reading the lockbox again does not restore it — it spends it a second time, and the number stops being a held-out estimate. `breakdown.py` reads this file in three places. |
| `keep/s2_ml/experiments.jsonl` | Append-only. The only record of how the feature set got from the original 23 to the champion's 38 — 19 entries, including cycles whose code no longer exists. |
| `keep/s2_ml/proposals.jsonl` | Every agent proposal and its fate, critic-stopped ones included. Paid, and non-deterministic — a re-run writes different text. |
| `keep/ablations/s2_ml_abl_*` | A full LOCO evaluation each, behind `champion_spec.json`'s rejected lines — no count here on purpose, `breakdown` globs whatever is there. Re-running one costs an afternoon, and together they are what makes "low importance is not droppability" a measurement rather than an opinion. |
| `keep/ablations/s2_ml_rf42` | RandomForest at the 42-feature set the champion carried then — the reproduction behind `champion_spec.json`'s ExtraTrees-over-RandomForest line. No code reads it; the rationale cites it. |
| `keep/ablations/s2_ml_seedsweep.json` | Five seeds per estimator; the seed-stability row in `breakdown.md`. |
| `keep/ablations/s2_ml_rowseedsweep.json` | Five seeds through the row path (`stages/s2_ml/rowseedsweep.py`, ~3 min). The band the headline `84.66% at 0.9901` is read against — without it a single-seed delta cannot be told from the draw. Cheap to rebuild, but only against *this* `champion_spec.json`; the copy on disk is the one the shipped number belongs to. |
| `keep/agent_runs/*` | Paid API calls. `costs.json` records what each one spent; the review ledgers are non-deterministic, so re-running produces different verdicts, not the same ones. |

Artifacts that no live code reads any more move to `archive/`, which has its own README.
That is the retirement path — not deletion.

## Where a stage writes

One `--out` per stage, always a directory under `regen/`. Three files are the exception:
`experiment.py` and `roweval.py` derive a second destination with
`runslayout.keep_dir_for(out_dir)`, so the two halves stay paired even when `--out` points at
a scratch directory. `runslayout.py` is the single definition of all of this — no other file
restates a path under `runs/`.

## Freshness

Stages stamp a content hash of every input they read into `_inputs.<stage>.json`, so a report
that no longer describes its inputs can be caught. `python freshness.py` checks every
directory in `runslayout.checkable_dirs()`: all of `regen/`, plus `keep/s2_ml`, plus
`labeled_raw/`.

`keep/ablations` and `keep/agent_runs` are deliberately *not* checked — nothing ever rewrites
them, so nothing can go stale. `keep/s2_ml` is checked despite living under `keep/` because
the lockbox stamp is the only warning a reader will ever get that the spent read describes an
older `champion_spec.json`. There is no re-run that would catch it later.
