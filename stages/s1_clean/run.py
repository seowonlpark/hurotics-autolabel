# S1 census + manifest: writes manifest.jsonl, census.md
# see README for usage and what each output contains

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stages.s1_clean.config import KEEP_MEASURED
from stages.s1_clean.manifest import profile_file

REPO_ROOT = Path(__file__).resolve().parents[2]


# every raw csv under raw_dir, sorted
def find_csvs(raw_dir: Path) -> list[Path]:
    return sorted(raw_dir.rglob("*.csv"))


# names the file actually carries, by role (measured columns all have a role)
def _present_names(row: dict) -> set[str]:
    names: set[str] = set()
    for group in row.get("roles_present", {}).values():
        names.update(group)
    return names


# render the human-readable census.md -- corpus schema sanity, resolved by name
def write_census_md(rows, out: Path) -> None:
    ok = [r for r in rows if "read_error" not in r and r.get("family")]
    fam_files: dict[str, int] = {}
    for r in ok:
        fam_files[r["family"]] = fam_files.get(r["family"], 0) + 1

    # the real "schema broke" alarm: a canonical MEASURED column that a file is missing.
    # presence of computed tail columns is expected and says nothing, so it is not reported.
    missing: dict[str, int] = {}
    for r in ok:
        present = _present_names(r)
        for c in KEEP_MEASURED:
            if c not in present:
                missing[c] = missing.get(c, 0) + 1

    lines = [
        "# S1 Schema Census",
        "",
        f"- files: **{len(rows)}**",
        f"- families: **{', '.join(sorted(fam_files))}**",
        "",
        "Columns are resolved by name, never by position (the same index is `loco` in some",
        "headers and `Step` in others). Header width differences live entirely in the computed",
        "tail that the clean layer drops, so they never reach a canonical file.",
        "",
        "## Files per family",
        "",
        "| family | files |",
        "|---|---|",
    ]
    lines += [f"| {fam} | {n} |" for fam, n in sorted(fam_files.items(), key=lambda x: -x[1])]

    lines += ["", "## Missing measured columns", ""]
    if missing:
        lines += [f"A canonical measured column absent from some files (of {len(KEEP_MEASURED)} "
                  "in `config.KEEP_MEASURED`) - the schema may have changed:", ""]
        lines += [f"- `{c}` (missing in {n} files)" for c, n in sorted(missing.items())]
    else:
        lines += [f"None - every file carries the full {len(KEEP_MEASURED)}-column measured set.", ""]

    out.write_text("\n".join(lines), encoding="utf-8")


# profile every raw csv, write the manifest + census
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    raw_dir = (REPO_ROOT / args.raw).resolve()
    out_dir = (REPO_ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = find_csvs(raw_dir)
    if not paths:
        raise SystemExit(f"No CSVs under {raw_dir}")
    print(f"[s1] {len(paths)} csv files under {raw_dir}")

    rows = []
    with (out_dir / "manifest.jsonl").open("w", encoding="utf-8") as fh:
        for i, p in enumerate(paths, 1):
            row = profile_file(p, REPO_ROOT)
            rows.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[s1] {i}/{len(paths)} {row['path']}")

    write_census_md(rows, out_dir / "census.md")

    errs = [r for r in rows if "read_error" in r]
    print(f"[s1] done. {len(rows)} rows, {len(errs)} read errors -> {out_dir}")
    for r in errs:
        print(f"  !! {r['path']}: {r['read_error']}")


if __name__ == "__main__":
    main()
