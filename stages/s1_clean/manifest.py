# per-file measurement -- numbers only, no verdicts
# rate is measured from the Time column, never a filename; two estimates are reported
# because they legitimately disagree on a jittery file (see README)

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from stages.s1_clean.census import Resolution, read_header, resolve
from stages.s1_clean.config import (
    GAP_FACTOR,
    LABEL_UNKNOWN_HUMAN,
    LABEL_UNKNOWN_MACHINE,
    PLAUSIBLE_HZ,
    SESSION_DIR_PATTERN,
    TIME_UNIT_MS,
)

_SESSION = re.compile(SESSION_DIR_PATTERN)


# session identity from the directory, the only metadata we trust
def session_of(path: Path) -> dict:
    m = _SESSION.match(path.parent.name)
    if not m:
        return {"session_date": None, "session_run": None, "session_dir": path.parent.name}
    return {
        "session_date": m.group(1),
        "session_run": int(m.group(2)) if m.group(2) else 1,
        "session_dir": path.parent.name,
    }


# rate, jitter and gaps from the Time column
def measure_time(t: np.ndarray) -> dict:
    if t.size < 2:
        return {"error": "fewer than 2 rows"}

    dt = np.diff(t)
    median_dt = float(np.median(dt))
    span_ms = float(t[-1] - t[0])

    hz_median = 1000.0 * TIME_UNIT_MS / median_dt if median_dt > 0 else None
    hz_span = 1000.0 * TIME_UNIT_MS * (t.size - 1) / span_ms if span_ms > 0 else None

    gap_mask = dt > GAP_FACTOR * median_dt
    lo, hi = PLAUSIBLE_HZ

    return {
        "n_rows": int(t.size),
        "monotonic": bool(np.all(dt > 0)),
        "span_ms": round(span_ms, 1),
        "duration_s": round(span_ms / 1000.0, 2),
        "dt_median_ms": round(median_dt, 4),
        "dt_min_ms": round(float(dt.min()), 4),
        "dt_max_ms": round(float(dt.max()), 4),
        "dt_p99_ms": round(float(np.percentile(dt, 99)), 4),
        "hz_from_median_dt": round(hz_median, 4) if hz_median else None,
        "hz_from_span": round(hz_span, 4) if hz_span else None,
        "hz_estimates_agree": (
            bool(abs(hz_median - hz_span) < 0.5) if hz_median and hz_span else None
        ),
        "hz_plausible": bool(hz_median and lo <= hz_median <= hi),
        "n_gaps": int(gap_mask.sum()),
        "gap_max_ms": round(float(dt.max()), 1) if gap_mask.any() else 0.0,
        "gap_total_ms": round(float(dt[gap_mask].sum()), 1) if gap_mask.any() else 0.0,
    }


# label census: codes, counts, segment structure; no judgement
def measure_labels(series: pd.Series) -> dict:
    vals = series.to_numpy()
    counts = {int(k): int(v) for k, v in series.value_counts().items()}
    changes = np.flatnonzero(vals[1:] != vals[:-1]) + 1

    return {
        "dtype": str(series.dtype),
        "n_null": int(series.isna().sum()),
        "codes": sorted(counts),
        "counts": counts,
        "n_segments": int(changes.size + 1),
        "has_machine_unknown_255": LABEL_UNKNOWN_MACHINE in counts,
        "has_human_unknown_neg1": LABEL_UNKNOWN_HUMAN in counts,
    }


# one manifest row; never raises on bad data -- records the failure instead
def profile_file(path: Path, repo_root: Path) -> dict:
    row: dict = {
        "path": str(path.relative_to(repo_root)),
        "filename": path.name,
        **session_of(path),
    }

    try:
        raw_header = read_header(path)
        res: Resolution = resolve(raw_header)
    except Exception as exc:
        row["read_error"] = f"header: {exc}"
        return row

    row.update(
        {
            "family": res.family,
            "n_cols": res.n_cols,
            "roles_present": {k: sorted(v) for k, v in sorted(res.roles_present.items())},
            "unknown_names": res.unknown_names,
            "label_columns": res.label_columns,
            "legacy_algo_columns": res.legacy_algo_columns,
        }
    )

    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
        df = df.loc[:, [c for c in df.columns if not c.startswith("Unnamed")]]
    except Exception as exc:
        row["read_error"] = f"body: {exc}"
        return row

    name_to_raw = {n: c for n, c in zip(res.index_by_name, df.columns)}

    time_col = name_to_raw.get("Time")
    row["time"] = (
        measure_time(df[time_col].to_numpy(dtype=float))
        if time_col is not None
        else {"error": "no Time column"}
    )

    row["labels"] = {
        name: measure_labels(df[name_to_raw[name]]) for name in res.label_columns
    }
    return row
