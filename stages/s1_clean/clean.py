"""S1 clean: resample every raw file onto the canonical grid.

    python -m stages.s1_clean.clean --raw data/raw --out runs/<run>/s1_clean

Outputs, per source file:
    data/clean/<session>/<name>.parquet             canonical-grid data, gyro normalized
    data/clean/<session>/<name>.channel_trust.json  resolved gyro unit + sagittal axis
And for the run:
    segments.jsonl     every segment: rows, duration, source rate, method, usable
    observations.jsonl per-file channel-trust facts (the exception agent's input)
    quarantine.jsonl   ledger of whole-file rejects (the raw file stays put)
    clean_report.md    what happened to the corpus
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from stages.s1_clean.census import read_header, resolve
from stages.s1_clean.channel_trust import detect_and_normalize
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


def quarantine_record(path: Path, reason: str, evidence: dict) -> dict:
    """Build a ledger entry for a whole-file reject. The raw file is never touched.

    Raw is source-of-truth and stays where it is; the quarantine is a ledger
    (quarantine.jsonl), not a copy of the data. The category slug is the reason's
    leading phrase, so like failures fold together. needs_human is always set: a
    whole-file reject is exactly the "AI can't proceed" case.
    """
    category = re.sub(r"[^a-z0-9]+", "_", reason.split(":")[0].lower()).strip("_")[:40]
    return {
        "file": str(path.relative_to(REPO_ROOT)),
        "reason": reason,
        "category": category,
        "needs_human": True,
        "evidence": evidence,
        "quarantined_at": datetime.now(timezone.utc).isoformat(),
    }


def select_columns(present: list[str]) -> tuple[list[str], list[str]]:
    """Measured channels + documented exceptions + labels if present.

    Returns (kept, missing). Missing is a fact about this variant, not an error.
    """
    wanted = list(KEEP_MEASURED) + list(KEEP_EXCEPTIONS) + list(KEEP_IF_PRESENT)
    kept = [c for c in wanted if c in present]
    missing = [c for c in KEEP_MEASURED if c not in present]
    return kept, missing


def clean_one(path: Path) -> tuple[Path | None, list[dict], str | None, dict | None]:
    res = resolve(read_header(path))
    if "Time" not in res.index_by_name:
        return None, [], "no Time column", None

    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.loc[:, [c for c in df.columns if not c.startswith("Unnamed")]]
    df.columns = list(res.index_by_name)  # resolved names, prefix stripped

    kept, missing = select_columns(list(df.columns))
    if missing:
        return None, [], f"missing measured channels: {missing}", None
    df = df[kept]

    out, segs = resample_file(df, "Time")
    rows = [{"path": str(path.relative_to(REPO_ROOT)), **s.to_dict()} for s in segs]
    if out.empty:
        return None, rows, "no usable segments", None

    # Gyro is now on the canonical grid (uniform dt, gap-free segments): resolve its
    # unit + sagittal axis from the data and normalize every gyro channel to deg/s.
    out, trust = detect_and_normalize(out)

    dest = CLEAN_DIR / session_of(path)["session_dir"] / f"{path.stem}.{CLEAN_FORMAT}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest, index=False) if CLEAN_FORMAT == "parquet" else out.to_csv(dest, index=False)
    dest.with_suffix(".channel_trust.json").write_text(
        json.dumps({"file": str(path.relative_to(REPO_ROOT)), **trust}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return dest, rows, None, trust


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    raw_dir = (REPO_ROOT / args.raw).resolve()
    out_dir = (REPO_ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(raw_dir.rglob("*.csv"))
    print(f"[s1] cleaning {len(paths)} files -> {CANONICAL_HZ} Hz, "
          f"{len(KEEP_MEASURED)} measured + {len(KEEP_EXCEPTIONS)} documented-exception columns, {CLEAN_FORMAT}")

    all_rows, written, failed, observations, quarantined = [], 0, [], [], []
    for i, p in enumerate(paths, 1):
        rel = str(p.relative_to(REPO_ROOT))
        dest, rows, err, trust = clean_one(p)
        all_rows += rows
        if err:
            failed.append((rel, err))
            evidence = {"segments": [r for r in rows]} if rows else {}
            quarantined.append(quarantine_record(p, err, evidence))
        else:
            written += 1
            kind = ("channel_trust_anomaly" if trust["anomalies"]
                    else "channel_trust_abstained" if trust["abstained"]
                    else "channel_trust_ok")
            observations.append({"path": rel, "kind": kind,
                                 "drift_contaminated": trust["drift"]["contaminated"],
                                 "channel_trust": trust})
        print(f"[s1] {i}/{len(paths)} {p.name} -> {len(rows)} segment(s){' !! ' + err if err else ''}")

    with (out_dir / "segments.jsonl").open("w", encoding="utf-8") as fh:
        for r in all_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out_dir / "observations.jsonl").open("w", encoding="utf-8") as fh:
        for o in observations:
            fh.write(json.dumps(o, ensure_ascii=False) + "\n")
    with (out_dir / "quarantine.jsonl").open("w", encoding="utf-8") as fh:
        for q in quarantined:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")

    # The gate: every raw file is accounted for exactly once. Assert it, don't hope.
    accounted = written + len(quarantined)
    assert accounted == len(paths), f"partition broken: {accounted} accounted != {len(paths)} raw files"

    usable = [r for r in all_rows if r["usable"]]
    dropped = [r for r in all_rows if not r["usable"]]
    methods: dict[str, int] = {}
    for r in usable:
        methods[r["method"]] = methods.get(r["method"], 0) + 1

    # Gyro trust rollup across the written files.
    norm_sides = [(o["path"], s) for o in observations for s, r in o["channel_trust"]["sides"].items()
                  if r["scale_to_degps"] != 1.0]
    abstained = [(o["path"], s) for o in observations for s in o["channel_trust"]["abstained"]]
    anomalies = [(o["path"], s, o["channel_trust"]["sides"][s]) for o in observations
                 for s in o["channel_trust"]["anomalies"]]
    drift_flagged = [(o["path"], c) for o in observations for c in o["drift_contaminated"]]

    lines = [
        "# S1 Clean Report",
        "",
        f"- source files: **{len(paths)}**, written: **{written}**, failed: **{len(failed)}**",
        f"- **partition (gate): {written} clean + {len(quarantined)} quarantined = {written + len(quarantined)} "
        f"of {len(paths)} raw files, 0 unaccounted**",
        f"- segments: **{len(all_rows)}** ({len(usable)} usable, {len(dropped)} dropped)",
        f"- canonical rate: **{CANONICAL_HZ} Hz**",
        f"- usable duration: **{sum(r['duration_s'] for r in usable) / 60:.1f} min**",
        f"- columns kept: **{len(KEEP_MEASURED)} measured + {len(KEEP_EXCEPTIONS)} documented exceptions**",
        f"- format: **{CLEAN_FORMAT}**",
        "",
        "Measured-only: every column that churns position between variants is a *computed* one,",
        "so this collapses all 5 schema variants into a single canonical shape.",
        "",
        "## Gyro trust / normalization",
        "",
        "Detected per file (not asserted): d(Deg_Y)/dt regressed against each gyro axis resolves",
        "the sagittal axis and the unit. Every gyro channel is normalized to deg/s.",
        "",
        f"- side-channels normalized rad/s -> deg/s: **{len(norm_sides)}**",
        f"- sides that abstained (too static; fell back to documented convention): **{len(abstained)}**",
        f"- confident anomalies (detected axis/unit disagree with the documented rule): **{len(anomalies)}**",
        "",
        *([f"- anomaly: `{p}` side {s}: sagittal={r['sagittal_gyro_axis']} unit={r['unit']} "
           f"r={r['r']} (documented: Z / per-side)" for p, s, r in anomalies] or ["- anomalies: none"]),
        "",
        "## Yaw / drift trust",
        "",
        "A Deg channel whose value tracks session time is measuring integration drift, not",
        "orientation (§4.2). Flagged per channel, not dropped — the raw superset is kept.",
        "",
        f"- Deg channels flagged drift-contaminated (|corr(Deg,Time)| >= 0.9): **{len(drift_flagged)}** "
        f"across **{len({p for p, _ in drift_flagged})}** files",
        "",
        *([f"- drift: `{p}` channel `{c}`" for p, c in drift_flagged] or ["- drift-contaminated: none"]),
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
    lines += ["", "## Quarantined files (whole-file rejects -> quarantine.jsonl ledger)", ""]
    lines += [f"- `{q['file']}`: {q['reason']} (needs_human)" for q in quarantined] or ["none"]
    (out_dir / "clean_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[s1] {written} clean, {len(quarantined)} quarantined, "
          f"{len(usable)} usable segments, {len(dropped)} dropped -> {out_dir}")


if __name__ == "__main__":
    main()
