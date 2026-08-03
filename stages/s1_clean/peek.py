"""Eyeball a clean parquet without a parquet viewer.

    python -m stages.s1_clean.peek data/clean/20260520/00220_100_2026_5_20_13_33_0.parquet
    python -m stages.s1_clean.peek data/clean/**/*.parquet            # columns of every clean file
    python -m stages.s1_clean.peek <file.parquet> --csv               # write a .peek.csv beside it

Parquet is ~5-10x smaller and dtype-safe, but you can't open it in a text editor.
This prints the kept columns (proof of what the clean stage pruned) + a head sample,
and can dump a CSV twin for a spreadsheet. It never touches the parquet itself.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="parquet path(s) or glob(s)")
    ap.add_argument("--csv", action="store_true", help="write a .peek.csv beside each file")
    ap.add_argument("--rows", type=int, default=5, help="head rows to print (default 5)")
    args = ap.parse_args()

    files = sorted({p for pat in args.paths for p in glob.glob(pat, recursive=True)})
    if not files:
        raise SystemExit("no parquet files matched")

    for f in files:
        df = pd.read_parquet(f)
        print(f"\n=== {f}  ({df.shape[0]:,} rows x {df.shape[1]} cols) ===")
        print("columns:", list(df.columns))
        if args.rows:
            print(df.head(args.rows).to_string())
        if args.csv:
            out = Path(f).with_suffix(".peek.csv")
            df.to_csv(out, index=False)
            print(f"-> wrote {out}")


if __name__ == "__main__":
    main()
