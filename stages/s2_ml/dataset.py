# S2 dataset: labeled trials on the canonical grid, grouped by subject so nothing leaks across folds

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from stages.s1_clean.resample import resample_file
from stages.s2_ml.corpus import CORPUS_PATH, Entry, entry_for, load_manifest, trials as _entries

REPO_ROOT = Path(__file__).resolve().parents[2]

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

# DERIVED now, not restated- both were hand-edited constants naming specific recordings until §1.6
DEFAULT_LOCKBOX_REVS = tuple(sorted({e.subject for e in load_manifest() if e.split == "lockbox"}))
EXCLUDED_TRIALS = {e.key for e in load_manifest() if e.exclude}

# the reason each quarantine was called; `breakdown` prints it instead of a bare pair
EXCLUDED_WHY = {e.key: e.exclude for e in load_manifest() if e.exclude}


# every labeled trial minus the quarantined; include_excluded loads the raw corpus, as auditors do
def find_trials(include_excluded: bool = False, manifest: Path = CORPUS_PATH) -> list[Path]:
    return [e.annotated for e in _entries(include_excluded, manifest)]


# == FEATURES today; its own name so validate_spec still checks when a richer corpus arrives
SELECTABLE_FEATURES = FEATURES


# one labeled trial, normalized onto the canonical grid
@dataclass
class Trial:
    path: str
    rev: str            # the manifest's `subject`; the CV group
    trial: int          # the manifest's `session`
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
def load_entry(entry: Entry, unseal: bool = False) -> Trial:
    df = _read_raw(entry.annotated)
    frame, segments = resample_file(df, TIME_COL)
    # nearest sampling emits float Label; restore integer codes
    if LABEL_COL in frame.columns:
        frame[LABEL_COL] = frame[LABEL_COL].round().astype(int)
    dropped = sum(s.n_source_rows for s in segments if not s.usable)
    split = "train" if unseal else entry.split
    return Trial(str(entry.annotated.relative_to(REPO_ROOT)), entry.subject, entry.session,
                 split, frame, len(df), dropped)


# one trial by path, for the readers handed a file rather than walking the corpus
def load_trial(path: Path, split: str | None = None) -> Trial:
    entry = entry_for(path)
    # callers used to compute the split themselves; one that now disagrees is refused, not honoured
    if split is not None and split != entry.split:
        raise SystemExit(
            f"[dataset] {Path(path).name}: caller says split {split!r}, the manifest declares "
            f"{entry.split!r}. Change data/corpus.json if the split is what moved."
        )
    return load_entry(entry)


# `unseal` relabels the sealed subjects "train" for THIS read; only the auditors may pass it
def load_dataset(manifest: Path = CORPUS_PATH,
                 include_excluded: bool = False,
                 unseal: bool = False) -> list[Trial]:
    return [load_entry(e, unseal) for e in _entries(include_excluded, manifest)]
