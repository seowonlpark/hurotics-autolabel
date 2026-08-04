# S1 census + manifest;  python -m stages.s1_clean.run
# writes manifest.jsonl + census.md, both for humans; the load-bearing product is
# census.py's fingerprint/family_of, which downstream imports directly

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stages.s1_clean.census import build_registry
from stages.s1_clean.manifest import profile_file

REPO_ROOT = Path(__file__).resolve().parents[2]
S1_CENSUS_OUT_DIR = REPO_ROOT / "runs" / "s1_census"


def find_csvs(raw_dir: Path) -> list[Path]:
    return sorted(raw_dir.rglob("*.csv"))


def write_census_md(registry, rows, out: Path) -> None:
    families = sorted({v.family for v in registry.values()})
    lines = [
        "# S1 Schema Census",
        "",
        f"- files: **{len(rows)}**",
        f"- header variants: **{len(registry)}**",
        f"- families: **{', '.join(families)}**",
        "",
        "## Variants",
        "",
        "| variant | family | cols | files | sessions |",
        "|---|---|---|---|---|",
    ]
    by_variant: dict[str, set] = {}
    for r in rows:
        if r.get("variant_id"):
            by_variant.setdefault(r["variant_id"], set()).add(r.get("session_dir"))

    for v in sorted(registry.values(), key=lambda v: -len(v.files)):
        sess = sorted(x for x in by_variant.get(v.variant_id, set()) if x)
        shown = ", ".join(sess[:4]) + (f" +{len(sess) - 4}" if len(sess) > 4 else "")
        lines.append(f"| `{v.variant_id}` | {v.family} | {v.n_cols} | {len(v.files)} | {shown} |")

    lines += [""]

    # names that move between variants within a family- the column-47 hazard
    for fam in families:
        positions: dict[str, set] = {}
        for v in registry.values():
            if v.family != fam:
                continue
            for i, n in enumerate(v.names):
                positions.setdefault(n, set()).add(i)
        movers = {n: sorted(p) for n, p in positions.items() if len(p) > 1}
        if movers:
            lines += [f"## Names that change position within `{fam}`", ""]
            lines += [f"- `{n}`: positions {p}" for n, p in sorted(movers.items())]
            lines += ["", "Positional indexing is unsafe here. Resolve by name.", ""]

    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", type=Path, default=S1_CENSUS_OUT_DIR)
    args = ap.parse_args()

    raw_dir = (REPO_ROOT / args.raw).resolve()
    out_dir = (REPO_ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = find_csvs(raw_dir)
    if not paths:
        raise SystemExit(f"No CSVs under {raw_dir}")
    print(f"[s1] {len(paths)} csv files under {raw_dir}")

    registry = build_registry(paths, REPO_ROOT)
    families = sorted({v.family for v in registry.values()})
    print(f"[s1] {len(registry)} header variants across {len(families)} families")
    for f in families:
        n = sum(len(v.files) for v in registry.values() if v.family == f)
        print(f"[s1]   {f}: {n} files")

    rows = []
    with (out_dir / "manifest.jsonl").open("w", encoding="utf-8") as fh:
        for i, p in enumerate(paths, 1):
            row = profile_file(p, REPO_ROOT)
            rows.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[s1] {i}/{len(paths)} {row['path']}")

    write_census_md(registry, rows, out_dir / "census.md")

    errs = [r for r in rows if "read_error" in r]
    print(f"[s1] done. {len(rows)} rows, {len(errs)} read errors -> {out_dir}")
    for r in errs:
        print(f"  !! {r['path']}: {r['read_error']}")


if __name__ == "__main__":
    main()
