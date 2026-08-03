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

# Lockbox: whole revs sealed until the very end (§7). rev8 spans the balanced regime
# (~70/27), rev13 the walk-heavy one — together they probe both without touching the loop.
DEFAULT_LOCKBOX_REVS = ("rev8", "rev13")

_REV = re.compile(r"(rev\d+)")
_TRIAL = re.compile(r"trial_(\d+)")


def rev_of(path: Path) -> str:
    m = _REV.search(str(path))
    return m.group(1) if m else "rev?"


def trial_of(path: Path) -> int:
    m = _TRIAL.search(path.stem)
    return int(m.group(1)) if m else 0


def find_trials(labeled_dir: Path = LABELED_DIR) -> list[Path]:
    return sorted(labeled_dir.rglob("annotated_loco_*_trial_*.csv"))


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
) -> list[Trial]:
    """Load every trial, normalized and split. `val_revs` may be empty when the caller
    prefers grouped CV over a fixed validation rev; the lockbox is always held out."""
    trials = []
    for p in find_trials(labeled_dir):
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
