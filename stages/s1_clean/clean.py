# S1 clean: resample every raw file onto the canonical grid, gyro normalized to deg/s
# writes per-file parquet + channel_trust.json, and per-run segments/observations/
# quarantine jsonl + clean_report.md. see README for the layout.

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from stages.s1_clean.census import read_header, resolve
from stages.s1_clean.channel_trust import detect_and_normalize
from stages.s1_clean.config import (
    CANONICAL_HZ,
    CLEAN_FORMAT,
    DOCUMENTED_GYRO_PERMUTATION,
    KEEP_EXCEPTIONS,
    KEEP_IF_PRESENT,
    KEEP_MEASURED,
)
from stages.s1_clean.manifest import session_of
from stages.s1_clean.resample import resample_file

REPO_ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = REPO_ROOT / "data" / "clean"


# ledger entry for a whole-file reject; the raw file is never touched (it stays source
# of truth). category is the reason's leading phrase so like failures fold together
def quarantine_record(path: Path, reason: str, evidence: dict) -> dict:
    category = re.sub(r"[^a-z0-9]+", "_", reason.split(":")[0].lower()).strip("_")[:40]
    return {
        "file": str(path.relative_to(REPO_ROOT)),
        "reason": reason,
        "category": category,
        "needs_human": True,
        "evidence": evidence,
        "quarantined_at": datetime.now(timezone.utc).isoformat(),
    }


# measured channels + documented exceptions + labels if present
# returns (kept, missing); missing is a fact about this file's header, not an error
def select_columns(present: list[str]) -> tuple[list[str], list[str]]:
    wanted = list(KEEP_MEASURED) + list(KEEP_EXCEPTIONS) + list(KEEP_IF_PRESENT)
    kept = [c for c in wanted if c in present]
    missing = [c for c in KEEP_MEASURED if c not in present]
    return kept, missing


# clean one raw file: resolve, prune to measured, resample, normalize gyro, write parquet
# returns (dest_or_None, segment_rows, error_or_None, trust_or_None)
def clean_one(path: Path) -> tuple[Path | None, list[dict], str | None, dict | None]:
    res = resolve(read_header(path))
    if "Time" not in res.index_by_name:
        return None, [], "no Time column", None

    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.loc[:, [c for c in df.columns if not c.startswith("Unnamed")]]
    df.columns = list(res.index_by_name) # resolved names, prefix stripped

    kept, missing = select_columns(list(df.columns))
    if missing:
        return None, [], f"missing measured channels: {missing}", None
    df = df[kept]

    # time base must define a forward cadence; median dt <= 0 is a broken clock
    # (a 2026-05 batch logs duplicate/backward timestamps) -- reject, don't resample it
    t = df["Time"].to_numpy(float)
    if t.size < 2 or float(np.median(np.diff(t))) <= 0.0:
        return None, [], "degenerate time base: median dt <= 0 (duplicate/backward timestamps)", None

    out, segs = resample_file(df, "Time")
    rows = [{"path": str(path.relative_to(REPO_ROOT)), **s.to_dict()} for s in segs]
    if out.empty:
        return None, rows, "no usable segments", None

    # gyro is now on the canonical grid: resolve unit + axis map, normalize to deg/s
    out, trust = detect_and_normalize(out)

    dest = CLEAN_DIR / session_of(path)["session_dir"] / f"{path.stem}.{CLEAN_FORMAT}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest, index=False) if CLEAN_FORMAT == "parquet" else out.to_csv(dest, index=False)
    dest.with_suffix(".channel_trust.json").write_text(
        json.dumps({"file": str(path.relative_to(REPO_ROOT)), **trust}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return dest, rows, None, trust


# clean every raw csv, write the run artifacts, assert the partition gate
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

    # the gate: every raw file accounted for exactly once. assert it, don't hope
    accounted = written + len(quarantined)
    assert accounted == len(paths), f"partition broken: {accounted} accounted != {len(paths)} raw files"

    usable = [r for r in all_rows if r["usable"]]
    dropped = [r for r in all_rows if not r["usable"]]
    methods: dict[str, int] = {}
    for r in usable:
        methods[r["method"]] = methods.get(r["method"], 0) + 1

    # gyro trust rollup across the written files
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
        "Measured-only: every column that churns position between header shapes is a *computed*",
        "one, so dropping them leaves a single canonical shape read entirely by name.",
        "",
        "## Gyro trust / normalization",
        "",
        "Detected per file (not asserted): d(Deg_A)/dt regressed against every gyro axis, for",
        "every Deg axis A, recovers the Deg->Gyro permutation and the unit. Every gyro channel",
        "is normalized to deg/s. The r-floor is per axis, so an axis with no signal abstains",
        "rather than contributing a noise argmax (Deg_Z drifts, so it often has none).",
        "",
        "This resolves which gyro axis measures which angle axis. It does NOT pick the SAGITTAL",
        "axis - that has no in-file signature (DOMAIN_NOTES 6.2); stages/s2_ml/transform.py fixes",
        "it to the Y plane for every file (6.3), flagging rather than guessing anomalies.",
        "",
        f"- side-channels normalized rad/s -> deg/s: **{len(norm_sides)}**",
        f"- sides that abstained (too static; fell back to documented convention): **{len(abstained)}**",
        f"- confident anomalies (detected axis/unit disagree with the documented rule): **{len(anomalies)}**",
        "",
        *([f"- anomaly: `{p}` side {s}: conflicts={r['conflicts_with_documented']} "
           f"map={ {A: r['gyro_axis_by_deg_axis'][A] for A in r['resolved_deg_axes']} } "
           f"unit={r['unit']} r={r['r']} (documented: {DOCUMENTED_GYRO_PERMUTATION})"
           for p, s, r in anomalies] or ["- anomalies: none"]),
        "",
        "## Yaw / drift trust",
        "",
        "A Deg channel whose value tracks session time is measuring integration drift, not",
        "orientation (Section 4.2). Flagged per channel, not dropped - the raw superset is kept.",
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
