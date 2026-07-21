# segment at gaps, then put every segment on the canonical 100 Hz grid
# two rules: never resample across a gap, never downsample without anti-aliasing
# (naive [::5] folds >50 Hz into the gait band). see README.

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy.signal import decimate

from stages.s1_clean.config import (
    CANONICAL_DT_MS,
    CANONICAL_HZ,
    DECIMATE_FILTER,
    GAP_FACTOR,
    MIN_SEGMENT_SAMPLES,
    NEAREST_ROLES,
    RATE_TOLERANCE,
    ROLE_BY_NAME,
)


# one continuous run of samples between gaps
@dataclass
class Segment:
    index: int # segment index within the file
    start_row: int # first source row
    end_row: int # last source row, exclusive
    n_source_rows: int # rows in this run
    t_start_ms: float # first timestamp
    t_end_ms: float # last timestamp
    duration_s: float # run length in seconds
    source_hz: float # measured rate of this run
    method: str = "" # how it was put on the grid
    n_output_rows: int = 0 # rows after resampling
    usable: bool = True # kept in the output?
    reason: str = "" # why dropped, if unusable

    def to_dict(self) -> dict:
        return asdict(self)


# split at dt > GAP_FACTOR * median(dt); returns [start, end) row pairs
def segment_at_gaps(t: np.ndarray) -> list[tuple[int, int]]:
    if t.size < 2:
        return [(0, int(t.size))]
    dt = np.diff(t)
    cuts = np.flatnonzero(dt > GAP_FACTOR * float(np.median(dt))) + 1
    bounds = np.concatenate(([0], cuts, [t.size]))
    return [(int(bounds[i]), int(bounds[i + 1])) for i in range(bounds.size - 1)]


# rate of one gap-free segment, from median dt; nan on a degenerate time base
# (median dt <= 0) so the caller can drop it instead of dividing by zero
def measure_hz(t: np.ndarray) -> float:
    if t.size < 2:
        return float("nan")
    med = float(np.median(np.diff(t)))
    return 1000.0 / med if med > 0 else float("nan")


# snap a measured rate to its nominal family, or None if it fits nowhere
# (99.3789 / 99.688 / 99.961 / 100.0 all snap to 100 -- timestamp quantization)
def rate_family(hz: float) -> float | None:
    if not np.isfinite(hz):
        return None
    for nominal in (CANONICAL_HZ, 2 * CANONICAL_HZ, 5 * CANONICAL_HZ):
        if abs(hz - nominal) / nominal <= RATE_TOLERANCE:
            return nominal
    return None


# interp helper: linear for continuous channels, nearest for categorical ones
def _interp_to_grid(t: np.ndarray, df: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    out = {}
    for col in df.columns:
        role = ROLE_BY_NAME.get(col)
        v = df[col].to_numpy(dtype=float)
        if role in NEAREST_ROLES:
            idx = np.searchsorted(t, grid).clip(1, t.size - 1)
            left = np.abs(grid - t[idx - 1]) <= np.abs(t[idx] - grid)
            out[col] = v[np.where(left, idx - 1, idx)]
        else:
            out[col] = np.interp(grid, t, v)
    return pd.DataFrame(out)


# put one gap-free segment on the canonical grid; records its own method
def resample_segment(
    t: np.ndarray, df: pd.DataFrame, seg: Segment
) -> tuple[pd.DataFrame | None, Segment]:
    nominal = rate_family(seg.source_hz)

    if nominal is None:
        seg.usable, seg.reason = False, f"rate {seg.source_hz:.3f} Hz fits no known family"
        return None, seg
    if seg.n_source_rows < MIN_SEGMENT_SAMPLES:
        seg.usable, seg.reason = False, f"{seg.n_source_rows} rows < {MIN_SEGMENT_SAMPLES}"
        return None, seg

    factor = int(round(nominal / CANONICAL_HZ))

    if factor > 1:
        # uniform grid at source rate first (decimate assumes uniform spacing),
        # then FIR-decimate: low-pass below the new Nyquist, then downsample
        src_grid = np.arange(t[0], t[-1], 1000.0 / nominal)
        uniform = _interp_to_grid(t, df, src_grid)
        cols = {}
        for col in uniform.columns:
            role = ROLE_BY_NAME.get(col)
            v = uniform[col].to_numpy(dtype=float)
            cols[col] = v[::factor] if role in NEAREST_ROLES else decimate(
                v, factor, ftype=DECIMATE_FILTER, zero_phase=True
            )
        n = min(len(v) for v in cols.values())
        out = pd.DataFrame({k: v[:n] for k, v in cols.items()})
        out.insert(0, "Time", src_grid[::factor][:n])
        seg.method = f"decimate_{factor}x_{DECIMATE_FILTER}"
    else:
        # same rate family: correct timestamp quantization onto the exact grid
        grid = np.arange(t[0], t[-1], CANONICAL_DT_MS)
        out = _interp_to_grid(t, df, grid)
        out.insert(0, "Time", grid)
        seg.method = "interp_to_grid" if seg.source_hz != CANONICAL_HZ else "grid_aligned"

    out.insert(1, "segment", seg.index)
    seg.n_output_rows = len(out)
    return out, seg


# segment at gaps, resample each run, stack; unusable segments drop from the output
# but always survive in the segment table
def resample_file(df: pd.DataFrame, time_col: str) -> tuple[pd.DataFrame, list[Segment]]:
    t_all = df[time_col].to_numpy(dtype=float)
    data = df.drop(columns=[time_col])

    frames, segs = [], []
    for i, (a, b) in enumerate(segment_at_gaps(t_all)):
        t = t_all[a:b]
        seg = Segment(
            index=i,
            start_row=a,
            end_row=b,
            n_source_rows=b - a,
            t_start_ms=round(float(t[0]), 4),
            t_end_ms=round(float(t[-1]), 4),
            duration_s=round(float(t[-1] - t[0]) / 1000.0, 3),
            source_hz=round(measure_hz(t), 4),
        )
        out, seg = resample_segment(t, data.iloc[a:b].reset_index(drop=True), seg)
        segs.append(seg)
        if out is not None:
            frames.append(out)

    stacked = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return stacked, segs
