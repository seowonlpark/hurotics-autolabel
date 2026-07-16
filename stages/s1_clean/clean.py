"""S1 clean: resample every raw file onto the canonical grid.

    python -m stages.s1_clean.clean --raw data/raw --out runs/<run>/s1_clean

Outputs, per source file:
    data/clean/<session>/<name>.csv   canonical-grid data, `segment` column added
And for the run:
    segments.jsonl   every segment: rows, duration, source rate, method, usable
    clean_report.md  what happened to the corpus
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from stages.s1_clean.census import read_header, resolve
from stages.s1_clean.config import (
    CANONICAL_HZ,
    CLEAN_FORMAT,
    KEEP_EXCEPTIONS,
    KEEP_IF_PRESENT,
    KEEP_MEASURED,
)
from stages.s1_clean.manifest import session_of
from stages.s1_clean.resample import resample_file

REPO_ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = REPO_ROOT / "data" / "clean"


def select_columns(present: list[str]) -> tuple[list[str], list[str]]:
    """Measured channels + documented exceptions + labels if present.

    Returns (kept, missing). Missing is a fact about this variant, not an error.
    """
    wanted = list(KEEP_MEASURED) + list(KEEP_EXCEPTIONS) + list(KEEP_IF_PRESENT)
    kept = [c for c in wanted if c in present]
    missing = [c for c in KEEP_MEASURED if c not in present]
    return kept, missing


def clean_one(path: Path) -> tuple[Path | None, list[dict], str | None]:
    res = resolve(read_header(path))
    if "Time" not in res.index_by_name:
        return None, [], "no Time column"

    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.loc[:, [c for c in df.columns if not c.startswith("Unnamed")]]
    df.columns = list(res.index_by_name)  # resolved names, prefix stripped

    kept, missing = select_columns(list(df.columns))
    if missing:
        return None, [], f"missing measured channels: {missing}"
    df = df[kept]

    out, segs = resample_file(df, "Time")
    rows = [{"path": str(path.relative_to(REPO_ROOT)), **s.to_dict()} for s in segs]
    if out.empty:
        return None, rows, "no usable segments"

    dest = CLEAN_DIR / session_of(path)["session_dir"] / f"{path.stem}.{CLEAN_FORMAT}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest, index=False) if CLEAN_FORMAT == "parquet" else out.to_csv(dest, index=False)
    return dest, rows, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    raw_dir = (REPO_ROOT / args.raw).resolve()
    out_dir = (REPO_ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(raw_dir.rglob("*.csv"))
    n_keep = len(KEEP_MEASURED) + len(KEEP_EXCEPTIONS)
    print(f"[s1] cleaning {len(paths)} files -> {CANONICAL_HZ} Hz, {n_keep} measured columns, {CLEAN_FORMAT}")

    all_rows, written, failed = [], 0, []
    for i, p in enumerate(paths, 1):
        dest, rows, err = clean_one(p)
        all_rows += rows
        if err:
            failed.append((str(p.relative_to(REPO_ROOT)), err))
        else:
            written += 1
        print(f"[s1] {i}/{len(paths)} {p.name} -> {len(rows)} segment(s){' !! ' + err if err else ''}")

    with (out_dir / "segments.jsonl").open("w", encoding="utf-8") as fh:
        for r in all_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    usable = [r for r in all_rows if r["usable"]]
    dropped = [r for r in all_rows if not r["usable"]]
    methods: dict[str, int] = {}
    for r in usable:
        methods[r["method"]] = methods.get(r["method"], 0) + 1

    lines = [
        "# S1 Clean Report",
        "",
        f"- source files: **{len(paths)}**, written: **{written}**, failed: **{len(failed)}**",
        f"- segments: **{len(all_rows)}** ({len(usable)} usable, {len(dropped)} dropped)",
        f"- canonical rate: **{CANONICAL_HZ} Hz**",
        f"- usable duration: **{sum(r['duration_s'] for r in usable) / 60:.1f} min**",
        f"- columns kept: **{len(KEEP_MEASURED)} measured + {len(KEEP_EXCEPTIONS)} documented exceptions**",
        f"- format: **{CLEAN_FORMAT}**",
        "",
        "Measured-only: every column that churns position between variants is a *computed* one,",
        "so this collapses all 5 schema variants into a single canonical shape.",
        "",
        "## Method",
        "",
        "| method | segments |",
        "|---|---|",
        *[f"| `{k}` | {v} |" for k, v in sorted(methods.items(), key=lambda x: -x[1])],
        "",
        "## Dropped segments",
        "",
    ]
    lines += [f"- `{r['path']}` seg {r['index']}: {r['reason']}" for r in dropped] or ["none"]
    if failed:
        lines += ["", "## Failed files", ""] + [f"- `{p}`: {e}" for p, e in failed]
    (out_dir / "clean_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[s1] {len(usable)} usable segments, {len(dropped)} dropped -> {out_dir}")


if __name__ == "__main__":
    main()
