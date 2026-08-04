# segment at gaps, then onto the canonical 100 Hz grid
# never resample across a gap; never downsample without anti-aliasing

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
    MIN_SEGMENT_S,
    NEAREST_ROLES,
    RATE_TOLERANCE,
    ROLE_BY_NAME,
)


# one continuous run of samples between gaps
@dataclass
class Segment:
    index: int
    start_row: int
    end_row: int  # exclusive
    n_source_rows: int
    t_start_ms: float
    t_end_ms: float
    duration_s: float
    source_hz: float
    method: str = ""
    n_output_rows: int = 0
    usable: bool = True
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# split at dt > GAP_FACTOR * median(dt); [start, end) row pairs
def segment_at_gaps(t: np.ndarray) -> list[tuple[int, int]]:
    if t.size < 2:
        return [(0, int(t.size))]
    dt = np.diff(t)
    cuts = np.flatnonzero(dt > GAP_FACTOR * float(np.median(dt))) + 1
    bounds = np.concatenate(([0], cuts, [t.size]))
    return [(int(bounds[i]), int(bounds[i + 1])) for i in range(bounds.size - 1)]


# nan on a degenerate time base rather than dividing by zero; caller drops it
def measure_hz(t: np.ndarray) -> float:
    if t.size < 2:
        return float("nan")
    med = float(np.median(np.diff(t)))
    return 1000.0 / med if med > 0 else float("nan")


# snap to a nominal rate, None if it fits nowhere; not named *_family: that
# word is the header family here, and it made the drop reason below misread
def nominal_rate(hz: float) -> float | None:
    if not np.isfinite(hz):
        return None
    for nominal in (CANONICAL_HZ, 2 * CANONICAL_HZ, 5 * CANONICAL_HZ):
        if abs(hz - nominal) / nominal <= RATE_TOLERANCE:
            return nominal
    return None


# linear for continuous channels, nearest for categorical
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


# one gap-free segment onto the grid; records its own method
def resample_segment(
    t: np.ndarray, df: pd.DataFrame, seg: Segment
) -> tuple[pd.DataFrame | None, Segment]:
    nominal = nominal_rate(seg.source_hz)

    if nominal is None:
        # breakdown.py buckets on this string; reword => ADD a marker there, don't swap
        seg.usable, seg.reason = False, (
            f"rate {seg.source_hz:.3f} Hz matches no known acquisition rate"
        )
        return None, seg
    if seg.duration_s < MIN_SEGMENT_S:
        seg.usable, seg.reason = False, (
            f"{seg.duration_s:.3f} s < {MIN_SEGMENT_S} s "
            f"({seg.n_source_rows} rows @ {seg.source_hz:.1f} Hz)"
        )
        return None, seg

    factor = int(round(nominal / CANONICAL_HZ))

    if factor > 1:
        # uniform first- decimate assumes even spacing
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
        # same rate: just correct the quantization
        grid = np.arange(t[0], t[-1], CANONICAL_DT_MS)
        out = _interp_to_grid(t, df, grid)
        out.insert(0, "Time", grid)
        seg.method = "interp_to_grid" if seg.source_hz != CANONICAL_HZ else "grid_aligned"

    out.insert(1, "segment", seg.index)
    seg.n_output_rows = len(out)
    return out, seg


# unusable segments drop from the output but survive in the segment table
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
