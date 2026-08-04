"""S1 clean: resample every raw file onto the canonical grid.

    python -m stages.s1_clean.clean

Outputs, per source file:
    data/clean/<session>/<name>.channel_trust.json  resolved gyro unit + axis permutation

The canonical-grid frame itself is built, measured and dropped. It was written beside the
trust record as parquet until 2026-08-04 and nothing ever read it back: the serve path
derives its features from the RAW file at the raw rate, deliberately, so that it reproduces
training bit-for-bit (`transform.raw_to_features`). 689 MB of binary that only S1 could
produce and no one could open is not evidence. `data/raw` is source-of-truth; re-run this
stage to rebuild anything anyone actually wants.

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

import numpy as np
import pandas as pd

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
from stages.s1_clean.resample import resample_file

REPO_ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = REPO_ROOT / "data" / "clean"
S1_CLEAN_OUT_DIR = REPO_ROOT / "runs" / "s1_clean"


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


def time_base_evidence(t: np.ndarray, max_examples: int = 5) -> dict:
    """Say WHERE the clock breaks, not just that it does.

    `csv_line` is the line a human opens: the header is line 1, so the first data row is 2.
    A ledger entry reading only "degenerate time base" makes the reader re-derive by hand
    the exact thing this stage already computed in order to reject the file.
    """
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


def select_columns(present: list[str]) -> tuple[list[str], list[str]]:
    """Measured channels + documented exceptions + labels if present.

    Returns (kept, missing). Missing is a fact about this variant, not an error.
    """
    wanted = list(KEEP_MEASURED) + list(KEEP_EXCEPTIONS) + list(KEEP_IF_PRESENT)
    kept = [c for c in wanted if c in present]
    missing = [c for c in KEEP_MEASURED if c not in present]
    return kept, missing


def quarantine_lines(q: dict) -> list[str]:
    """One ledger entry as report lines: the reason, then where to look in the source CSV."""
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


def clean_one(path: Path
              ) -> tuple[Path | None, list[dict], str | None, dict | None, dict]:
    """(trust record path, segments, error, trust, evidence). Evidence is for the ledger."""
    res = resolve(read_header(path))
    if "Time" not in res.index_by_name:
        return None, [], "no Time column", None, {}

    # index_col=False is load-bearing. A trailing comma makes every data row one field wider
    # than the header, and pandas answers that by silently promoting the first data column to
    # the INDEX — which shifts every remaining column left by one while leaving the count
    # intact, so the positional rename below lands canonical names on the wrong channels and
    # raises nothing. One 2026-05 file was quarantined for a "degenerate time base" that was
    # really `L_Deg_X` being read as `Time`; its actual clock is a clean 500 Hz (§1.3).
    df = pd.read_csv(path, encoding="utf-8-sig", index_col=False)
    df = df.loc[:, [c for c in df.columns if not c.startswith("Unnamed")]]
    # Positional renaming is only safe once the positions are known to line up. Assert it
    # rather than let `df.columns = ...` raise a bare length error out of the runner.
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

    # Time base must define a forward cadence. A batch of 2026-05 files logs
    # duplicated and backward-running timestamps (median dt <= 0) — non-monotonic
    # time that np.interp would silently corrupt. Reject the whole file rather than
    # resample a broken clock; it lands in the quarantine ledger for a human.
    t = df["Time"].to_numpy(float)
    if t.size < 2 or float(np.median(np.diff(t))) <= 0.0:
        return (None, [],
                "degenerate time base: median dt <= 0 (duplicate/backward timestamps)",
                None, {"time_base": time_base_evidence(t)})

    out, segs = resample_file(df, "Time")
    rows = [{"path": str(path.relative_to(REPO_ROOT)), **s.to_dict()} for s in segs]
    if out.empty:
        return None, rows, "no usable segments", None, {}

    # Gyro is now on the canonical grid (uniform dt, gap-free segments): resolve its
    # unit + sagittal axis from the data and normalize every gyro channel to deg/s.
    out, trust = detect_and_normalize(out)

    # The trust record is the whole persisted product: `transform.load_trust` reads it back
    # by this exact path, and treats its absence as "never cleaned" rather than as a default.
    # Built as one name, not via with_suffix: `transform.trust_path` spells it exactly this
    # way, and a stem containing a dot must not resolve to two different files.
    dest = (CLEAN_DIR / session_of(path)["session_dir"] /
            f"{path.stem}.channel_trust.json")
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
        f"- persisted per file: **`channel_trust.json` only** — the canonical-grid frame is "
        f"measured and dropped (see this module's docstring)",
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
    (out_dir / "clean_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[s1] {written} clean, {len(quarantined)} quarantined, "
          f"{len(usable)} usable segments, {len(dropped)} dropped -> {out_dir}")


if __name__ == "__main__":
    main()
