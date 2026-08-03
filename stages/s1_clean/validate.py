"""Is this recording usable at all? The label-free gate in front of every consumer.

    python -m stages.s1_clean.validate path/to.csv

S1 already refuses raw device logs it cannot trust — degenerate time base, unknown rate
family, dead channels — and records the refusal in a quarantine ledger. This is that same
job for the derived rev2 view: the four rotational channels plus `Time`, which is what the
labelling path actually consumes.

It belongs in S1 and not in S2 because nothing here needs a model, a label, or a training
set. It answers "was anything measured properly", which is a property of the file.

**Every check is a way a file can be broken without being empty.** An empty file fails
loudly on its own; a file with a flat channel, or one that never rests, produces confident
numbers built on nothing, and that is the failure worth catching. `usable=False` means no
row of the file can be scored; warnings mean the numbers are weaker than they look and the
reason travels with them.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from stages.s1_clean.config import CANONICAL_HZ

# The derived rev2 view this pipeline serves. Kept here rather than imported from S2 so the
# gate does not depend on the stage it guards.
TIME_COL = "Time"
FEATURE_COLUMNS = ("L_ang_LPF", "R_ang_LPF", "L_angvel_LPF", "R_angvel_LPF")

# A channel whose spread is below this is not a measurement.
FLAT_STD_DEG = 1e-6


@dataclass
class Health:
    usable: bool
    rows: int
    segments: int = 0
    scorable_segments: int = 0
    rest_trusted: bool = False
    errors: list[str] = field(default_factory=list)     # cannot be scored at all
    warnings: list[str] = field(default_factory=list)   # scorable, but weaker than it looks

    def report(self) -> str:
        head = "USABLE" if self.usable else "NOT USABLE"
        lines = [f"{head}: {self.rows:,} rows, {self.scorable_segments}/{self.segments} "
                 f"segments long enough to score"]
        lines += [f"  ERROR:   {e}" for e in self.errors]
        lines += [f"  warning: {w}" for w in self.warnings]
        return "\n".join(lines)


def check_columns(df: pd.DataFrame) -> list[str]:
    """Resolve the required columns BY NAME (§1.3 - the numeric prefix is a lie)."""
    return [c for c in (TIME_COL, *FEATURE_COLUMNS) if c not in df.columns]


def check_time(t: np.ndarray) -> list[str]:
    """A time base that does not advance makes every rate, gap and window meaningless (§2.6)."""
    errors = []
    if t.size < 2:
        return ["fewer than 2 samples: no time base"]
    dt = np.diff(t)
    if not np.isfinite(t).all():
        errors.append("Time contains non-finite values")
    if np.median(dt) <= 0:
        errors.append("degenerate time base: median sample interval is not positive "
                      "(duplicated or backward timestamps)")
    if (dt < 0).any():
        errors.append(f"Time goes backwards at {int((dt < 0).sum())} samples")
    return errors


def health(frame: pd.DataFrame, window_s: float = 2.0,
           fs_hz: float = CANONICAL_HZ) -> Health:
    """Judge a frame that is already on the canonical grid (has a `segment` column).

    `rest_trusted` is imported lazily: the rest search lives in S2 next to the features
    that consume it, and importing it at module scope would make this gate depend on the
    stage it guards.
    """
    if frame.empty:
        return Health(usable=False, rows=0, errors=["frame is empty"])

    errors: list[str] = []
    warnings: list[str] = []

    for c in FEATURE_COLUMNS:
        v = frame[c].to_numpy(float)
        n_bad = int((~np.isfinite(v)).sum())
        if n_bad:
            errors.append(f"{c}: {n_bad:,} non-finite samples")
        elif float(np.std(v)) < FLAT_STD_DEG:
            errors.append(f"{c}: constant - channel is dead, unmapped, or the wrong column")

    n = int(round(window_s * fs_hz))
    seg_len = frame.groupby("segment").size()
    scorable = int((seg_len >= n).sum())
    if scorable == 0:
        errors.append(f"no segment reaches one {window_s:g}s window: nothing is scorable")
    elif (short := int((seg_len < n).sum())):
        warnings.append(f"{short} of {len(seg_len)} segments are shorter than one "
                        f"{window_s:g}s window and cannot be scored")

    from stages.s2_ml.features import rest_reference
    _zeros, _ileg, trusted = rest_reference(frame, fs_hz)
    if not trusted:
        # ASCII only: this string reaches a cp949 console, where a section sign mangles.
        warnings.append("no rest span found: the interleg zero falls back to a "
                        "whole-recording median, so every interleg feature on this file is "
                        "weaker evidence (10.2 expects recordings to begin at rest)")

    return Health(usable=not errors, rows=int(len(frame)), segments=int(len(seg_len)),
                  scorable_segments=scorable, rest_trusted=bool(trusted),
                  errors=errors, warnings=warnings)


def health_of_csv(path: Path, window_s: float = 2.0) -> Health:
    """Read a CSV, put it on the canonical grid, and judge it."""
    from stages.s1_clean.resample import resample_file

    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    if (missing := check_columns(df)):
        return Health(usable=False, rows=len(df),
                      errors=[f"missing required columns {missing}"])
    if (bad := check_time(df[TIME_COL].to_numpy(float))):
        return Health(usable=False, rows=len(df), errors=bad)

    frame, _segments = resample_file(df[[TIME_COL, *FEATURE_COLUMNS]], TIME_COL)
    if frame.empty:
        return Health(usable=False, rows=len(df),
                      errors=["no segment survived resampling: rate fits no known family, "
                              "or every run is too short"])
    return health(frame, window_s)


def main() -> None:
    ap = argparse.ArgumentParser(description="Check whether a CSV can be scored at all.")
    ap.add_argument("csv", type=Path, nargs="+")
    ap.add_argument("--window-s", type=float, default=2.0)
    args = ap.parse_args()

    bad = 0
    for p in args.csv:
        h = health_of_csv(p, args.window_s)
        bad += not h.usable
        print(f"{p.name}: {h.report()}")
    if bad:
        raise SystemExit(f"{bad} of {len(args.csv)} file(s) not usable")


if __name__ == "__main__":
    main()
