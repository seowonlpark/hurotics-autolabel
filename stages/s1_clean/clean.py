# S1 clean (python -m stages.s1_clean.clean): resample to the canonical grid; only trust persists

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from freshness import declare_no_inputs
from stages.s1_clean.census import read_header, resolve
from stages.s1_clean.channel_trust import detect_and_normalize
from stages.s1_clean.config import (
    CANONICAL_HZ,
    DOCUMENTED_GYRO_PERMUTATION,
    KEEP_EXCEPTIONS,
    KEEP_IF_PRESENT,
    KEEP_MEASURED,
)
from stages.s1_clean.manifest import session_of
from stages.s1_clean.rawread import read_raw_body
from stages.s1_clean.resample import resample_file
from stages.report import add_report_flag

from runslayout import REGEN

REPO_ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = REPO_ROOT / "data" / "clean"
S1_CLEAN_OUT_DIR = REGEN / "s1_clean"


# a ledger entry, not a copy- the raw file never moves; always needs_human
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


# say WHERE the clock breaks, not just that it does- csv_line is the line a human opens
def time_base_evidence(t: np.ndarray, max_examples: int = 5) -> dict:
    if t.size < 2:
        return {"n_rows": int(t.size),
                "detail": "fewer than 2 samples: there is no interval to measure"}
    dt = np.diff(t)
    bad = np.flatnonzero(dt <= 0)          # dt[j] spans t[j] -> t[j+1]
    return {
        "n_rows": int(t.size),
        "median_dt": float(np.median(dt)),
        "n_nonpositive_dt": int(bad.size),
        "n_duplicate": int(np.count_nonzero(dt == 0)),
        "n_backward": int(np.count_nonzero(dt < 0)),
        "first_bad_csv_line": (int(bad[0]) + 3) if bad.size else None,
        "examples": [
            {"csv_line": int(j) + 3,       # header 1 + data starts at 2 + t[j+1] is one on
             "row_index": int(j) + 1,      # 0-based index into the data rows
             "t_prev": float(t[j]),
             "t": float(t[j + 1]),
             "dt": float(dt[j]),
             "kind": "duplicate" if dt[j] == 0 else "backward"}
            for j in bad[:max_examples]
        ],
    }


# (kept, missing); missing is a fact about the variant, not an error
def select_columns(present: list[str]) -> tuple[list[str], list[str]]:
    wanted = list(KEEP_MEASURED) + list(KEEP_EXCEPTIONS) + list(KEEP_IF_PRESENT)
    kept = [c for c in wanted if c in present]
    missing = [c for c in KEEP_MEASURED if c not in present]
    return kept, missing


# one ledger entry as report lines: reason, then where to look in the source CSV
def quarantine_lines(q: dict) -> list[str]:
    out = [f"- `{q['file']}`: {q['reason']} (needs_human)"]
    tb = q.get("evidence", {}).get("time_base")
    if not tb:
        return out
    if tb.get("detail"):
        return out + [f"    - {tb['detail']} ({tb['n_rows']} rows)"]
    out.append(
        f"    - **{tb['n_nonpositive_dt']} of {tb['n_rows'] - 1} intervals do not advance** "
        f"({tb['n_duplicate']} duplicate, {tb['n_backward']} backward); median dt "
        f"{tb['median_dt']:.6g} s; first at **CSV line {tb['first_bad_csv_line']}**"
    )
    out += [f"        - line {e['csv_line']} (data row {e['row_index']}): "
            f"`Time` {e['t_prev']:.6g} -> {e['t']:.6g}, dt {e['dt']:+.6g} s ({e['kind']})"
            for e in tb["examples"]]
    if tb["n_nonpositive_dt"] > len(tb["examples"]):
        out.append(f"        - ...and {tb['n_nonpositive_dt'] - len(tb['examples'])} more "
                   f"(the counts above are the full extent; these are the first few)")
    return out


# (trust record path, segments, error, trust, evidence); evidence is for the ledger
def clean_one(path: Path
              ) -> tuple[Path | None, list[dict], str | None, dict | None, dict]:
    res = resolve(read_header(path))
    if "Time" not in res.index_by_name:
        return None, [], "no Time column", None, {}

    df = read_raw_body(path)
    # assert the positions line up rather than let the rename raise a bare length error
    if len(df.columns) != len(res.index_by_name):
        return (None, [],
                f"column count mismatch: {len(df.columns)} data columns vs "
                f"{len(res.index_by_name)} header names",
                None,
                {"columns": {"n_data": int(len(df.columns)),
                             "n_header": int(len(res.index_by_name))}})
    df.columns = list(res.index_by_name)  # resolved names, prefix stripped

    kept, missing = select_columns(list(df.columns))
    if missing:
        return None, [], f"missing measured channels: {missing}", None, {}
    df = df[kept]

    # reject a non-monotonic clock (median dt <= 0) rather than let np.interp silently corrupt it
    t = df["Time"].to_numpy(float)
    if t.size < 2 or float(np.median(np.diff(t))) <= 0.0:
        return (None, [],
                "degenerate time base: median dt <= 0 (duplicate/backward timestamps)",
                None, {"time_base": time_base_evidence(t)})

    out, segs = resample_file(df, "Time")
    rows = [{"path": str(path.relative_to(REPO_ROOT)), **s.to_dict()} for s in segs]
    if out.empty:
        return None, rows, "no usable segments", None, {}

    # now on the canonical grid: resolve gyro unit + sagittal axis from the data, normalize to deg/s
    out, trust = detect_and_normalize(out)

    # the whole persisted product- the grid frame is DROPPED; serve reads raw at the raw rate anyway
    dest = (CLEAN_DIR / session_of(path)["session_dir"] /
            f"{path.stem}.channel_trust.json")  # one name, so a dotted stem cannot become two files
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps({"file": str(path.relative_to(REPO_ROOT)), **trust}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return dest, rows, None, trust, {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", type=Path, default=S1_CLEAN_OUT_DIR)
    add_report_flag(ap)
    args = ap.parse_args()

    raw_dir = (REPO_ROOT / args.raw).resolve()
    out_dir = (REPO_ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(raw_dir.rglob("*.csv"))
    print(f"[s1] cleaning {len(paths)} files -> {CANONICAL_HZ} Hz, "
          f"{len(KEEP_MEASURED)} measured + {len(KEEP_EXCEPTIONS)} documented-exception columns")

    all_rows, written, failed, observations, quarantined = [], 0, [], [], []
    for i, p in enumerate(paths, 1):
        rel = str(p.relative_to(REPO_ROOT))
        dest, rows, err, trust, evidence = clean_one(p)
        all_rows += rows
        if err:
            failed.append((rel, err))
            if rows:
                evidence = {**evidence, "segments": [r for r in rows]}
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

    # the gate: every raw file is accounted for exactly once; assert it, don't hope
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
        f"- persisted per file: **`channel_trust.json` only** — the canonical-grid frame is "
        f"measured and dropped (see this module's header)",
        "",
        "Measured-only: every column that churns position between variants is a *computed* one,",
        "so this collapses every schema variant into a single canonical shape.",
        "",
        "## Gyro trust / normalization",
        "",
        "Detected per file (not asserted): d(Deg_A)/dt regressed against every gyro axis, for",
        "every Deg axis A, recovers the Deg->Gyro permutation and the unit. Every gyro channel",
        "is normalized to deg/s. The r-floor is per axis, so an axis with no signal abstains",
        "rather than contributing a noise argmax (Deg_Z drifts, so it often has none).",
        "",
        "This resolves which gyro axis measures which angle axis. It does NOT resolve which axis",
        "is SAGITTAL — that has no in-file signature (DOMAIN_NOTES 6.2) and is a variant lookup",
        "in stages/s2_ml/transform.py.",
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
    lines += [ln for q in quarantined for ln in quarantine_lines(q)] or ["none"]
    if args.report:
        (out_dir / "clean_report.md").write_text("\n".join(lines), encoding="utf-8")

    # the only upstream is the `data/raw` tree, which is a corpus rather than an artifact to hash
    declare_no_inputs(out_dir, stage="s1_clean")

    print(f"[s1] {written} clean, {len(quarantined)} quarantined, "
          f"{len(usable)} usable segments, {len(dropped)} dropped -> {out_dir}")


if __name__ == "__main__":
    main()
