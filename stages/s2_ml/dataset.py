# S2 dataset: load the labeled rev* trials onto the canonical grid, split by rev
# same resampler as S1, and group == rev (one subject, one day) so nothing leaks
# does NOT window or train- just normalized, grouped, split frames

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from stages.s1_clean.resample import resample_file

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELED_DIR = REPO_ROOT / "data" / "labeled"

# the lpf_view four; some trials carry stray whitespace- normalized on read
FEATURES = ("L_ang_LPF", "R_ang_LPF", "L_angvel_LPF", "R_angvel_LPF")
LABEL_COL = "Label"
TIME_COL = "Time"

# -1 is excluded from training targets, kept for eval
STAND, WALK, HUMAN_UNKNOWN = 0, 10, -1
TRAIN_CLASSES = (STAND, WALK)

# whole revs sealed until the work is frozen; rev8 has been read exactly ONCE --
# re-reading it to check whether a change helped makes it a second validation set
# rev13 was sealed too and opened deliberately: it failed and a failure you cannot
# look at cannot be fixed; it is a development subject now
DEFAULT_LOCKBOX_REVS = ("rev8",)

# quarantined for a demonstrated label error; leaving one in is not the conservative
# choice- it teaches the wrong thing AND depresses every metric scored against it
# rev13/4: all 12,691 rows annotated stand, but it holds two runs- 7.0 s at 2.1 deg/s
# then 119.9 s at 45.3, and this subject's own labelled walking is 35-50; evidence is
# INTERNAL to the file, not "the classifier disagreed"
EXCLUDED_TRIALS = {("rev13", 4)}

_REV = re.compile(r"(rev\d+)")
_TRIAL = re.compile(r"trial_(\d+)")


def rev_of(path: Path) -> str:
    m = _REV.search(str(path))
    return m.group(1) if m else "rev?"


def trial_of(path: Path) -> int:
    m = _TRIAL.search(path.stem)
    return int(m.group(1)) if m else 0


# every labeled trial minus the quarantined; excluded=set() loads the raw corpus,
# which is what label_audit does so it can still see what it flagged
def find_trials(labeled_dir: Path = LABELED_DIR,
                excluded: set[tuple[str, int]] | None = None) -> list[Path]:
    excluded = EXCLUDED_TRIALS if excluded is None else excluded
    return sorted(p for p in labeled_dir.rglob("annotated_loco_*_trial_*.csv")
                  if (rev_of(p), trial_of(p)) not in excluded)


# == FEATURES today, since every trial carries the same four; kept as its own name so
# validate_spec still has something to check when a richer corpus arrives
SELECTABLE_FEATURES = FEATURES


# header-only, so it's free; a challenger measured on a quietly smaller corpus is not
# comparable to a champion measured on the whole one- report it, don't crash mid-fit
def partition_trials(features: tuple[str, ...] = FEATURES,
                     labeled_dir: Path = LABELED_DIR) -> tuple[list[Path], list[Path]]:
    want = {TIME_COL, *features, LABEL_COL}
    have, lack = [], []
    for p in find_trials(labeled_dir):
        cols = {c.strip() for c in pd.read_csv(p, nrows=0).columns}
        (have if want <= cols else lack).append(p)
    return have, lack


# one labeled trial, normalized onto the canonical grid
@dataclass
class Trial:
    path: str
    rev: str
    trial: int
    split: str          # "train" | "val" | "lockbox"
    frame: pd.DataFrame  # Time, segment, FEATURES..., Label- usable segments only
    n_source_rows: int  # rows in the raw trial, before normalization
    dropped_rows: int   # raw rows in segments too short / off-grid to keep


# strip header whitespace, keep Time + 4 features + Label BY NAME
def _read_raw(path: Path) -> pd.DataFrame:
    # index_col=False: no labeled file is ragged today, and none should start shifting
    df = pd.read_csv(path, index_col=False)
    df.columns = [c.strip() for c in df.columns]
    want = [TIME_COL, *FEATURES, LABEL_COL]
    missing = [c for c in want if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    return df[want]


# 100 Hz via S1's resampler: features FIR-decimated, Label nearest-sampled
def load_trial(path: Path, split: str) -> Trial:
    df = _read_raw(path)
    frame, segments = resample_file(df, TIME_COL)
    # nearest sampling emits float Label; restore integer codes
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
    # val_revs may be empty (grouped CV instead); the lockbox is always held out
    trials = []
    for p in find_trials(labeled_dir, excluded):
        split = assign_split(rev_of(p), lockbox_revs, val_revs)
        trials.append(load_trial(p, split))
    return trials


# per-rev row counts by split and class- the sanity check before any modelling
def census(trials: list[Trial]) -> pd.DataFrame:
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
