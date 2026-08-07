# the runs/ tree, defined once and imported- it splits on ONE question: can this file be rebuilt?

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

RUNS = REPO_ROOT / "runs"

# rebuildable: one `python run_pipeline.py` reproduces every byte, so it is safe to delete
REGEN = RUNS / "regen"

# not rebuildable at any price: deleting it destroys evidence for a claim, so git versions it
KEEP = RUNS / "keep"
AGENT_RUNS = KEEP / "agent_runs"      # paid, non-deterministic; one dated dir per pipeline run
ABLATIONS = KEEP / "ablations"        # full locoevals behind champion_spec's rejected lines
KEEP_S2 = KEEP / "s2_ml"              # the ledger, the proposals and the spent lockbox

# the one page written to be read, and the only .md the pipeline still emits
BREAKDOWN_MD = RUNS / "breakdown.md"

# outside runs/ but stamped and staleness-checked like a stage output, so it is defined here too
LABELED_RAW = REPO_ROOT / "labeled_raw"

# the split cuts THROUGH s2_ml: these two land in KEEP_S2 though their writer outputs to REGEN
LEDGER_FILENAME = "experiments.jsonl"
PROPOSALS_FILENAME = "proposals.jsonl"


# every dir a staleness check should consider, stamped or not- an unstamped stage reads as clean
def checkable_dirs() -> list[Path]:
    dirs = sorted(d for d in REGEN.glob("*") if d.is_dir())
    # the only keep/ dir here: the lockbox can go stale and there is no re-run to catch it later
    if KEEP_S2.is_dir():
        dirs.append(KEEP_S2)
    if LABELED_RAW.is_dir():
        dirs.append(LABELED_RAW)
    return dirs


# the keep-half of a stage's output directory: `--out runs/regen/s2_ml` -> `runs/keep/s2_ml`
def keep_dir_for(out_dir: Path) -> Path:
    out_dir = Path(out_dir).resolve()
    try:
        return KEEP / out_dir.relative_to(REGEN)
    except ValueError:
        # `--out` outside runs/regen- keep both halves there instead of writing into the real keep/
        return out_dir / "keep"
