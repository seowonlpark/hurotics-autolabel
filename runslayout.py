# where run artifacts live: the single definition of the runs/ tree, imported not restated
#
# runs/ splits on ONE question -- can this file be rebuilt from data/raw and the code in this
# commit? That question, not which stage wrote it, is what a reader needs before deleting
# anything, so it is the folder boundary.
#
#   runs/regen/  yes. A full `python run_pipeline.py` reproduces every byte. Safe to delete.
#   runs/keep/   no. Deleting it destroys evidence for a claim made in code or in a document,
#                and no command brings it back. Versioned in git alongside the claims.
#
# The split cuts THROUGH s2_ml rather than around it: train.py's locoeval refits on demand, but
# the experiment ledger is append-only history and the lockbox is a single-use read (dataset.py).
# So s2_ml writes to both -- KEEP_S2 for the three that are spent once, REGEN/"s2_ml" for the
# rest. `--out` still names one directory per stage; the keep path is derived from it, so a
# stage pointed at a scratch dir keeps both halves together.

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

RUNS = REPO_ROOT / "runs"

# rebuildable: every stage's working output, one directory per stage
REGEN = RUNS / "regen"

# not rebuildable, at any price
KEEP = RUNS / "keep"
AGENT_RUNS = KEEP / "agent_runs"      # paid, non-deterministic; one dated dir per pipeline run
ABLATIONS = KEEP / "ablations"        # full locoevals behind champion_spec's rejected lines
KEEP_S2 = KEEP / "s2_ml"              # the ledger, the proposals and the spent lockbox

# the one page written to be read, and the only .md the pipeline still emits
BREAKDOWN_MD = RUNS / "breakdown.md"

# outside runs/ but written by a pipeline stage and checked for staleness like one, so it is
# defined here with the rest rather than restated by every file that needs it
LABELED_RAW = REPO_ROOT / "labeled_raw"

# Files that live in KEEP_S2 even though the stage that writes them outputs to REGEN/"s2_ml".
# Named here rather than in each writer so `runs/README.md` and the writers cannot drift.
LEDGER_FILENAME = "experiments.jsonl"
PROPOSALS_FILENAME = "proposals.jsonl"


# Every directory a staleness check should consider, stamped or not: selecting on "has a stamp"
# checks exactly the dirs that could pass, so an unstamped stage reads as clean (see freshness).
#
# KEEP_S2 is in the list and the rest of keep/ is not, and the asymmetry is the point. agent_runs
# and ablations are frozen -- nothing rewrites them, so nothing can go stale. KEEP_S2 holds the
# lockbox, whose stamp is the ONLY warning that the spent read describes an older champion_spec;
# there is no re-run to catch it later, so the check has to fire here or nowhere.
def checkable_dirs() -> list[Path]:
    dirs = sorted(d for d in REGEN.glob("*") if d.is_dir())
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
        # `--out` pointed somewhere outside runs/regen (a scratch dir, a test tmpdir). Keep the
        # two halves together there rather than writing into the real runs/keep by surprise.
        return out_dir / "keep"
