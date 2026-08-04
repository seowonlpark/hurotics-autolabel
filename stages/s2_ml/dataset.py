# S2 dataset: labeled trials on the canonical grid, split by rev (group == rev, so nothing leaks)

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

# the trained classes' names, defined once beside the codes; every stage that prints a class
# reads this, so a code and its name can never drift apart in one file and not another
CLASS_NAME = {STAND: "stand", WALK: "walk"}

# sealed until the work is frozen; rev8 read exactly ONCE, rev13 opened deliberately and is dev now
DEFAULT_LOCKBOX_REVS = ("rev8",)

# quarantined for a demonstrated label error, evidence INTERNAL to the file: rev13/4 holds two runs
EXCLUDED_TRIALS = {("rev13", 4)}

_REV = re.compile(r"(rev\d+)")
_TRIAL = re.compile(r"trial_(\d+)")


def rev_of(path: Path) -> str:
    m = _REV.search(str(path))
    return m.group(1) if m else "rev?"


def trial_of(path: Path) -> int:
    m = _TRIAL.search(path.stem)
    return int(m.group(1)) if m else 0


# every labeled trial minus the quarantined; excluded=set() loads the raw corpus, as label_audit does
def find_trials(labeled_dir: Path = LABELED_DIR,
                excluded: set[tuple[str, int]] | None = None) -> list[Path]:
    excluded = EXCLUDED_TRIALS if excluded is None else excluded
    return sorted(p for p in labeled_dir.rglob("annotated_loco_*_trial_*.csv")
                  if (rev_of(p), trial_of(p)) not in excluded)


# == FEATURES today; its own name so validate_spec still checks when a richer corpus arrives
SELECTABLE_FEATURES = FEATURES


# one labeled trial, normalized onto the canonical grid
@dataclass
class Trial:
    path: str
    rev: str
    trial: int
    split: str          # "train" | "lockbox"
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


# two splits, not three: held-out evaluation is grouped CV over the training revs (locoeval),
# so there is no standing validation set to carve out- the lockbox is the only thing withheld
def assign_split(rev: str, lockbox_revs: tuple[str, ...]) -> str:
    return "lockbox" if rev in lockbox_revs else "train"


def load_dataset(
    labeled_dir: Path = LABELED_DIR,
    lockbox_revs: tuple[str, ...] = DEFAULT_LOCKBOX_REVS,
    excluded: set[tuple[str, int]] | None = None,
) -> list[Trial]:
    trials = []
    for p in find_trials(labeled_dir, excluded):
        trials.append(load_trial(p, assign_split(rev_of(p), lockbox_revs)))
    return trials
