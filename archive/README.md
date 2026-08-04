# archive/

Artifacts and documents that nothing in the pipeline reads any more. Kept rather than
deleted because each one is a record of a measurement, an agent verdict, or a decision that
was real when it was made and is not regenerable — the code that produced it is gone, or
the document it specified has since been written.

Nothing under `archive/` is on any import path or any `Step` gate. Moving a file here is
the claim that no live code reads it; `stages/breakdown.py` and `freshness.check_all` were
both checked against this list. A file here may still be *cited by name* from live prose —
`archive/needtowrite.md` is, from `README.md` and `transitions.py` — which is a pointer at a
frozen record, not a read.

| item | why it is here |
|---|---|
| `needtowrite.md` | The spec `README.md` was to be rewritten from, after the original `README.md` and `PIPELINE_BREAKDOWN.md` were deleted on 2026-08-04. **Both jobs are done**: `README.md` exists and was written from §2/§5/§6, and the breakdown is now generated into `runs/breakdown.md` by `stages/breakdown.py`. What is not regenerable is §5 and §6 — the deleted originals transcribed verbatim — and §5.4's eight gaps, one of which `transitions.py` still cites as the reason it measures what it measures. Dropped from `breakdown.PROSE_CLAIMS` in the same move: a frozen document should not be failing a staleness check. |
| `understand.md` | The why-layer: one justification per file and per function, with the losing alternative recorded beside each choice. Archived 2026-08-04 because it had drifted off the tree it describes and nothing read it — no code, no other document. Its own §12 quotes 9 / 11 / 14 pipeline steps where `--dry-run` now reports 13 / 16, and its §8 and header state that `README.md` does not exist, which stopped being true when `README.md` was rewritten. Its last line says the code is right and the file is stale when they disagree; this is that, acted on. §1–§10's reasoning is the part worth mining if a decision is ever re-opened. |
| `runs/_archived_baselines.json` | RF-era 23-feature baseline. Self-declared archived on 2026-08-03 when its run dir was deleted; the metrics were kept because that run is **not** in `experiments.jsonl`. Superseded by `runs/s2_ml_rf42`, which is RF at the 42-feature set the champion carried then and *is* still cited — by `champion_spec.json`'s rationale. Zero readers. |
| `runs/2026-08-03_run1` … `run3`, `run5` | `fusion_review.jsonl` — verdicts from the S4 fusion review agent. S4 was deleted 2026-08-04 along with `agents/s4_fusion.py`; with no fusion there is nothing these judge. |
| `runs/2026-08-03_run4` | `exceptions_review.jsonl` from the same era. `s1_exception` is still a live step, but this run predates the S4 removal and nothing reads dated run dirs — `run_pipeline._new_run_dir` only scans *today's* date to pick the next N. |

Paths in the table are as they were when each item was archived. `runs/` has since been
split into `runs/regen/` and `runs/keep/` — see `runs/README.md` — but nothing here moved,
because nothing here is read.

## What deliberately did NOT come here

Four things looked archival and are not. They are load-bearing, and they are exactly what
`runs/keep/` now holds:

- `runs/keep/ablations/s2_ml_abl_*` — globbed by `stages/breakdown.py` for the ablation table.
- `runs/keep/ablations/s2_ml_seedsweep.json` — read by `stages/breakdown.py` for the
  seed-stability row.
- `runs/keep/ablations/s2_ml_rf42` — the reproduction behind the
  ExtraTrees-over-RandomForest claim, cited by `champion_spec.json`'s rationale. No code
  reads it, which is exactly why deleting it would go unnoticed until the claim was
  questioned.
- `runs/keep/s2_ml/roweval_lockbox.json` — the single-use rev8 read. It can never be
  regenerated, and `breakdown.py` reads it in three places.

The S4 stage code itself (`stages/s4_fusion/`, `agents/s4_fusion.py`, `stages/s2_ml/oof.py`,
`stages/s3_physics/run.py`, `orchestrator.py`, `stages/s1_clean/peek.py`,
`stages/s2_ml/audit.py`) is not copied here either. It was deleted from the worktree, not
archived, and `git show HEAD:<path>` still produces it.
