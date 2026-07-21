# eyeball a clean parquet without a viewer: print kept columns + a head sample,
# optionally dump a .peek.csv twin. read-only. see README for usage.

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd


# print columns + head for each matched parquet; optionally write a .peek.csv twin
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
