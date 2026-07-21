# S2 dataset: load the labeled rev* trials onto the canonical grid, split by rev
# the rev2 view: 4 rotational features (L/R_ang_LPF, L/R_angvel_LPF) + Label (§5.7).
# reuses S1's resampler (canonical 100 Hz grid) and groups by rev (one subject/day),
# lockbox holds out whole revs so nothing leaks. does NOT window or train. see README.

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from stages.s1_clean.resample import resample_file

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELED_DIR = REPO_ROOT / "data" / "labeled"

# the rev2 derived view; names carry stray whitespace in some trials, normalized on read
FEATURES = ("L_ang_LPF", "R_ang_LPF", "L_angvel_LPF", "R_angvel_LPF")
LABEL_COL = "Label"
TIME_COL = "Time"

# label codes (§5.1/§5.2); -1 excluded from training targets, kept for eval
STAND, WALK, HUMAN_UNKNOWN = 0, 10, -1
TRAIN_CLASSES = (STAND, WALK)

# lockbox: whole revs sealed until the very end (§7); rev8 balanced, rev13 walk-heavy
DEFAULT_LOCKBOX_REVS = ("rev8", "rev13")

_REV = re.compile(r"(rev\d+)")
_TRIAL = re.compile(r"trial_(\d+)")

# revs excluded entirely; rev14 confirmed anomalous 2026-07-21 (§6.4) -- low-amplitude
# gait, no raw source to audit, dragged LORO macro-F1. not deleted from disk
EXCLUDED_REVS = ("rev14",)


# rev id from a path, e.g. 'rev13'
def rev_of(path: Path) -> str:
    m = _REV.search(str(path))
    return m.group(1) if m else "rev?"


# trial number from a filename
def trial_of(path: Path) -> int:
    m = _TRIAL.search(path.stem)
    return int(m.group(1)) if m else 0


# every labeled trial csv, excluded revs skipped
def find_trials(labeled_dir: Path = LABELED_DIR) -> list[Path]:
    return sorted(p for p in labeled_dir.rglob("annotated_loco_*_trial_*.csv")
                  if rev_of(p) not in EXCLUDED_REVS)


# one labeled trial, normalized onto the canonical grid
@dataclass
class Trial:
    path: str # trial csv path
    rev: str # rev id
    trial: int # trial number
    split: str # "train" | "val" | "lockbox"
    frame: pd.DataFrame # Time, segment, FEATURES..., Label -- usable segments only
    n_source_rows: int # rows in the raw trial, before normalization
    dropped_rows: int # raw rows in segments too short / off-grid to keep (§3.2 burst)


# read a trial, strip header whitespace, keep Time + 4 features + Label by name
def _read_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    want = [TIME_COL, *FEATURES, LABEL_COL]
    missing = [c for c in want if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    return df[want]


# normalize one trial to 100 Hz via S1's resampler (features FIR-decimated, Label nearest)
def load_trial(path: Path, split: str) -> Trial:
    df = _read_raw(path)
    frame, segments = resample_file(df, TIME_COL)
    # resample_file emits float Label from nearest sampling; restore integer codes
    if LABEL_COL in frame.columns:
        frame[LABEL_COL] = frame[LABEL_COL].round().astype(int)
    dropped = sum(s.n_source_rows for s in segments if not s.usable)
    return Trial(str(path.relative_to(REPO_ROOT)), rev_of(path), trial_of(path), split,
                 frame, len(df), dropped)


# split for one rev: lockbox > val > train
def assign_split(rev: str, lockbox_revs: tuple[str, ...], val_revs: tuple[str, ...]) -> str:
    if rev in lockbox_revs:
        return "lockbox"
    if rev in val_revs:
        return "val"
    return "train"


# load every trial, normalized and split; val_revs may be empty (grouped CV), lockbox
# is always held out
def load_dataset(
    labeled_dir: Path = LABELED_DIR,
    lockbox_revs: tuple[str, ...] = DEFAULT_LOCKBOX_REVS,
    val_revs: tuple[str, ...] = (),
) -> list[Trial]:
    trials = []
    for p in find_trials(labeled_dir):
        split = assign_split(rev_of(p), lockbox_revs, val_revs)
        trials.append(load_trial(p, split))
    return trials


# per-rev row counts by split and class -- the sanity check before any modelling
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


# load the dataset and print the per-rev census + row accounting
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
