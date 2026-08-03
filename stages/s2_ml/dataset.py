"""S2 dataset: load the labeled rev* trials onto the canonical grid, split by rev.

The only labeled data is `data/labeled/rev*/csv/annotated_loco_rev*_trial_*.csv`
(DOMAIN_NOTES §5.7): the derived rev2 view — four rotational features
(`L/R_ang_LPF`, `L/R_angvel_LPF`) + `Label` (0=stand, 10=walk, -1=human-unknown).

Two disciplines carried straight from S1, because they are not optional here either:
  - **Canonical grid.** rev* logs at ~494 Hz with jitter (§7). Every trial is put on
    the 100 Hz grid by the same `resample_file` S1 uses — segment at gaps, FIR-decimate
    the continuous channels, nearest-sample the label (never average a class code).
  - **Group = rev.** A rev is one subject on one day (§7); trials within a rev share
    both. CV groups by rev and the lockbox holds out whole revs, so nothing leaks.

This module does NOT window or train — it hands back normalized, grouped, split
frames. Feature extraction and modelling live in their own modules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from stages.s1_clean.resample import resample_file

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELED_DIR = REPO_ROOT / "data" / "labeled"

# The rev2 derived view. Names carry stray whitespace in some trials — normalized on read.
FEATURES = ("L_ang_LPF", "R_ang_LPF", "L_angvel_LPF", "R_angvel_LPF")
LABEL_COL = "Label"
TIME_COL = "Time"

# Label codes (§5.1/§5.2). -1 is excluded from training targets, kept for eval.
STAND, WALK, HUMAN_UNKNOWN = 0, 10, -1
TRAIN_CLASSES = (STAND, WALK)

# Lockbox: whole revs sealed until the very end (§7).
#
# rev8 ONLY. rev13 was sealed alongside it and opened deliberately: the first lockbox run
# showed the confidence signal did not transfer to it at all — its accuracy is flat from
# threshold 0.50 to 0.95 — and a failure that cannot be looked at cannot be fixed. rev13 is
# now a development subject, held out one fold at a time by the leave-one-rev-out CV like
# any other.
#
# rev8 has been read exactly once and must not be read again until the work is frozen.
# Re-running `roweval --lockbox` to check whether a change helped would turn the only
# measurement in this repo that was never optimized against into a second validation set.
DEFAULT_LOCKBOX_REVS = ("rev8",)

# Trials quarantined for a demonstrated label error, as (rev, trial). Excluded from both
# training and scoring, and never silently: a mislabelled trial teaches the model the wrong
# thing AND depresses every metric computed against it, so leaving it in is not the
# conservative choice it looks like.
#
# rev13/4: 12,691 rows, every one annotated `stand`. The file contains two runs under that
# one label - 7.0 s at 2.1 deg/s angular-velocity std, then 119.9 s at 45.3 deg/s with 71
# deg of interleg swing and 39 deg of thigh excursion. rev13's own labelled WALKING runs
# measure 35-50 deg/s across the other six trials. The second run is walking.
#
# The evidence is INTERNAL to the file - one run 20x the other under the same label, and
# the larger matching that subject's own walking - not "the classifier disagreed".
# Excluding data because a model dislikes it is how a corpus gets quietly fitted to its
# model; this entry stands on the measurement and would stand with no model at all.
EXCLUDED_TRIALS = {("rev13", 4)}

_REV = re.compile(r"(rev\d+)")
_TRIAL = re.compile(r"trial_(\d+)")


def rev_of(path: Path) -> str:
    m = _REV.search(str(path))
    return m.group(1) if m else "rev?"


def trial_of(path: Path) -> int:
    m = _TRIAL.search(path.stem)
    return int(m.group(1)) if m else 0


def find_trials(labeled_dir: Path = LABELED_DIR,
                excluded: set[tuple[str, int]] | None = None) -> list[Path]:
    """Every labeled trial, minus the quarantined ones.

    `stages.s2_ml.audit` proposes exclusions with evidence; they take effect only once
    written into `EXCLUDED_TRIALS` by hand. Pass `excluded=set()` to load the raw corpus,
    which is what the audit itself does so it can still see what it flagged.
    """
    excluded = EXCLUDED_TRIALS if excluded is None else excluded
    return sorted(p for p in labeled_dir.rglob("annotated_loco_*_trial_*.csv")
                  if (rev_of(p), trial_of(p)) not in excluded)


@dataclass
class Trial:
    """One labeled trial, normalized onto the canonical grid."""

    path: str
    rev: str
    trial: int
    split: str          # "train" | "val" | "lockbox"
    frame: pd.DataFrame  # Time, segment, FEATURES..., Label — usable segments only
    n_source_rows: int  # rows in the raw trial, before normalization
    dropped_rows: int   # raw rows in segments too short / off-grid to keep (§3.2 burst)


def _read_raw(path: Path) -> pd.DataFrame:
    """Read a trial, strip header whitespace, keep Time + 4 features + Label by name."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    want = [TIME_COL, *FEATURES, LABEL_COL]
    missing = [c for c in want if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    return df[want]


def load_trial(path: Path, split: str) -> Trial:
    """Normalize one trial to 100 Hz. Reuses S1's resampler: continuous features are
    FIR-decimated, the categorical Label is nearest-sampled (role `label`)."""
    df = _read_raw(path)
    frame, segments = resample_file(df, TIME_COL)
    # resample_file emits float Label from nearest sampling; restore integer codes.
    if LABEL_COL in frame.columns:
        frame[LABEL_COL] = frame[LABEL_COL].round().astype(int)
    dropped = sum(s.n_source_rows for s in segments if not s.usable)
    return Trial(str(path.relative_to(REPO_ROOT)), rev_of(path), trial_of(path), split,
                 frame, len(df), dropped)


def assign_split(rev: str, lockbox_revs: tuple[str, ...], val_revs: tuple[str, ...]) -> str:
    if rev in lockbox_revs:
        return "lockbox"
    if rev in val_revs:
        return "val"
    return "train"


def load_dataset(
    labeled_dir: Path = LABELED_DIR,
    lockbox_revs: tuple[str, ...] = DEFAULT_LOCKBOX_REVS,
    val_revs: tuple[str, ...] = (),
    excluded: set[tuple[str, int]] | None = None,
) -> list[Trial]:
    """Load every trial, normalized and split. `val_revs` may be empty when the caller
    prefers grouped CV over a fixed validation rev; the lockbox is always held out.

    `excluded` defaults to the quarantine list; pass `set()` to load the raw corpus."""
    trials = []
    for p in find_trials(labeled_dir, excluded):
        split = assign_split(rev_of(p), lockbox_revs, val_revs)
        trials.append(load_trial(p, split))
    return trials


def census(trials: list[Trial]) -> pd.DataFrame:
    """Per-rev row counts by split and class — the sanity check before any modelling."""
    rows = []
    for t in trials:
        lab = t.frame[LABEL_COL]
        rows.append({
            "rev": t.rev, "trial": t.trial, "split": t.split,
            "rows": len(t.frame),
            "stand": int((lab == STAND).sum()),
            "walk": int((lab == WALK).sum()),
            "unknown": int((lab == HUMAN_UNKNOWN).sum()),
        })
    df = pd.DataFrame(rows)
    return (df.groupby(["split", "rev"], as_index=False)
              [["rows", "stand", "walk", "unknown"]].sum()
              .sort_values(["split", "rev"]))


def main() -> None:
    trials = load_dataset()
    c = census(trials)
    print(f"[s2] loaded {len(trials)} trials from {LABELED_DIR}")
    print(c.to_string(index=False))
    tot = c.groupby("split")[["rows", "stand", "walk", "unknown"]].sum()
    print("\nby split:")
    print(tot.to_string())
    src = sum(t.n_source_rows for t in trials)
    dropped = sum(t.dropped_rows for t in trials)
    print(f"\naccounting: {src:,} raw rows -> {dropped:,} dropped "
          f"({100 * dropped / src:.3f}%, all 3.2 startup-burst fragments), rest resampled to 100 Hz")


if __name__ == "__main__":
    main()
