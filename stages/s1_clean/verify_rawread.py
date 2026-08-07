# the cache returns the CSV, exactly (python -m stages.s1_clean.verify_rawread)

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from stages.s1_clean.rawread import CACHE_DIR, cache_path, parse_csv, read_raw_body

REPO_ROOT = Path(__file__).resolve().parents[2]


# the first difference, or None; ordered so the coarsest mismatch is reported instead of a value diff
def first_difference(want: pd.DataFrame, got: pd.DataFrame) -> str | None:
    if list(got.columns) != list(want.columns):
        missing = [c for c in want.columns if c not in got.columns]
        extra = [c for c in got.columns if c not in want.columns]
        if missing or extra:
            return f"columns differ: missing {missing[:5]}, extra {extra[:5]}"
        return "column ORDER differs (same names)"
    if got.shape != want.shape:
        return f"shape {got.shape} != {want.shape}"
    bad_dtype = [(c, str(want[c].dtype), str(got[c].dtype))
                 for c in want.columns if want[c].dtype != got[c].dtype]
    if bad_dtype:
        return f"dtype: {bad_dtype[:3]}"
    for c in want.columns:
        a, b = want[c].to_numpy(), got[c].to_numpy()
        # equal_nan: a NaN in the CSV must survive as a NaN, and != would call that a difference
        if not np.array_equal(a, b, equal_nan=np.issubdtype(a.dtype, np.floating)):
            j = int(np.flatnonzero(~((a == b) | (pd.isna(a) & pd.isna(b))))[0])
            return f"column {c!r} row {j}: csv {a[j]!r} != cache {b[j]!r}"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--limit", type=int, default=0, help="first N files only (smoke test)")
    args = ap.parse_args()

    raw_dir = (REPO_ROOT / args.raw).resolve()
    paths = sorted(raw_dir.rglob("*.csv"))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise SystemExit(f"no CSV files under {raw_dir}")

    print(f"[verify_rawread] {len(paths)} files, cache at {CACHE_DIR}")
    failures, warmed = [], 0
    for i, p in enumerate(paths, 1):
        existed = cache_path(p).exists()
        want = parse_csv(p)                # the CSV, read the way the repo has always read it
        got = read_raw_body(p)             # through the cache: a miss writes it, a hit reads it
        warmed += not existed
        diff = first_difference(want, got)
        if diff:
            failures.append((str(p.relative_to(REPO_ROOT)), diff))
        print(f"[{i:>3}/{len(paths)}] {p.name:<44} {'MISMATCH: ' + diff if diff else 'ok'}")

    print(f"\n[verify_rawread] {len(paths) - len(failures)}/{len(paths)} identical, "
          f"{warmed} cache entries written this run")
    if failures:
        print(f"[verify_rawread] {len(failures)} MISMATCH — the cache is not a transcode:")
        for f, d in failures:
            print(f"  {f}: {d}")
        raise SystemExit(1)
    print("[verify_rawread] the cache returns the CSV exactly")


if __name__ == "__main__":
    main()
