# label every raw device log under data/raw, one file at a time
#   python -m stages.s2_ml.label_all  ->  labeled_raw/
# a refusal is an outcome, not a crash; grouped by reason so 12 files read as one

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from freshness import stamp_inputs
from stages.s1_clean.manifest import session_of
from stages.s3_physics.plausibility import summarize as summarize_plausibility
from stages.s2_ml import label as label_mod
from stages.s2_ml.locoeval import DEFAULT_THRESHOLDS
from stages.s2_ml.label import (
    DEFAULT_MODEL_DIR,
    GUESS_COL,
    GOLDEN_COLUMNS,
    SERVE_REFUSALS,
    VERDICT_COLUMNS,
    label_csv,
    load_champion,
    verdict_frame,
)

PRESET_SWEEP_FILENAME = "preset_sweep.json"

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_OUT_DIR = REPO_ROOT / "labeled_raw"


# A path as this repo names it: relative and forward-slashed, absolute if outside
def _rel_to_repo(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


# (labelled frame or None, the summary row, the plausibility findings)
def label_one(path: Path, model_dir: Path, threshold: float
              ) -> tuple[pd.DataFrame | None, dict, list[dict]]:
    rel = str(path.relative_to(REPO_ROOT))
    base = {"file": rel, "session": session_of(path)["session_dir"], "labelled": False,
            "variant": None, "axis": None, "rows": None, "n_confident": None,
            "confident_frac": None, "n_stand": None, "n_walk": None,
            "top_reason": None, "refusal": None, "reason": None,
            "implausible": None, "plausibility": None}
    try:
        out, provenance = label_csv(path, model_dir, threshold)
    except SERVE_REFUSALS as exc:
        return None, {**base, "refusal": type(exc).__name__,
                      "reason": str(exc).splitlines()[0]}, []
    except SystemExit as exc:
        return None, {**base, "refusal": "Unusable",
                      "reason": str(exc).splitlines()[0] if str(exc) else "health gate"}, []
    except Exception as exc:  # unexpected: record it, keep the sweep alive, surface it loudly
        return None, {**base, "refusal": f"UNEXPECTED:{type(exc).__name__}",
                      "reason": str(exc).splitlines()[0] if str(exc) else repr(exc)}, []

    amb = out["ambiguous"].to_numpy(bool)
    states = out.loc[~amb, "state"].value_counts()
    reasons = out.loc[amb, "reason"].value_counts()
    # A labelled file can still be an implausible one, and the summary has to be able to
    # say so; `implausible` is the boolean a reader filters on; `plausibility` names which
    # bounds, because "this file is suspicious" without the bound is not actionable
    findings = provenance.get("plausibility") or []
    hard = [f["code"] for f in findings if f["severity"] == "implausible"]
    return out, {
        **base,
        "labelled": True,
        "variant": provenance.get("variant_id"),
        "axis": provenance.get("sagittal_deg_axis"),
        "rows": int(len(out)),
        "n_confident": int((~amb).sum()),
        "confident_frac": round(float((~amb).mean()), 4),
        "n_stand": int(states.get("stand", 0)),
        "n_walk": int(states.get("walk", 0)),
        "top_reason": reasons.index[0] if len(reasons) else None,
        "implausible": bool(hard),
        "plausibility": ";".join(f["code"] for f in findings) or None,
    }, findings


# the three flags that decide whether coverage is a pure function of confidence
def gates_off() -> dict:
    return {
        "BAND_ABSTAINS": bool(label_mod.BAND_ABSTAINS),
        "PHYSICS_CEILING": bool(label_mod.PHYSICS_CEILING),
        "PHYSICS_FLOOR": bool(label_mod.PHYSICS_FLOOR),
    }


# a row sitting EXACTLY on the threshold cannot be re-decided from the emitted column:
# `confidence` prints rounded to 4dp but `ambiguous` was decided on the unrounded float
# measured at 4 of 78 files, 25-75 rows each, always reason=model_split, one-directional
# NOT corrected here- explain owns the abstention rule; measured instead, and `boundary`
# rides beside every count so the number states its own resolution
CONFIDENCE_HALF_ULP = 5e-5


# (rows committed, rows on the rounding boundary) at each threshold, from ONE pass
def coverage_at(out: pd.DataFrame, thresholds) -> tuple[dict[str, int], dict[str, int]]:
    conf = pd.to_numeric(out["confidence"], errors="coerce").to_numpy(float)
    committed, boundary = {}, {}
    for t in thresholds:
        key = f"{t:.2f}"
        # NaN (uncovered) compares False in both tests, which is the wanted answer twice:
        # never committed, and never on the boundary of a decision it was not part of
        committed[key] = int((conf >= t).sum())
        boundary[key] = int((np.abs(conf - t) < CONFIDENCE_HALF_ULP).sum())
    return committed, boundary


def render_report(rows: list[dict], threshold: float, model_dir: Path,
                  findings_by_file: dict[str, list[dict]] | None = None) -> str:
    ok = [r for r in rows if r["labelled"]]
    bad = [r for r in rows if not r["labelled"]]
    n_rows = sum(r["rows"] for r in ok)
    n_conf = sum(r["n_confident"] for r in ok)

    by_variant: dict[tuple, int] = {}
    for r in ok:
        by_variant[(r["variant"], r["axis"])] = by_variant.get((r["variant"], r["axis"]), 0) + 1
    by_refusal: dict[str, list[dict]] = {}
    for r in bad:
        by_refusal.setdefault(r["refusal"], []).append(r)
    by_reason: dict[str, int] = {}
    for r in ok:
        if r["top_reason"]:
            by_reason[r["top_reason"]] = by_reason.get(r["top_reason"], 0) + 1

    lines = [
        "# S2 Label — corpus sweep",
        "",
        f"- generated: **{datetime.now(timezone.utc).isoformat()}**",
        # Repo-relative, like the per-file rows above; this header was the one path in the
        # report printed as given, so a `--model-dir` that arrived absolute put the author's
        # home directory into a git-tracked file- a machine-specific string in an artifact
        # whose whole purpose is to be read on another machine
        f"- model: `{_rel_to_repo(model_dir)}`   threshold: **{threshold:.2f}**",
        f"- raw files: **{len(rows)}**, labelled: **{len(ok)}**, abstained: **{len(bad)}**",
        f"- **partition (gate): {len(ok)} labelled + {len(bad)} abstained = {len(rows)} "
        f"of {len(rows)} raw files, 0 unaccounted**",
        f"- rows labelled: **{n_rows:,}**, confident: **{n_conf:,}** "
        f"({n_conf / max(n_rows, 1):.1%})",
        "",
        "Coverage and accuracy are a PAIR (`caveats.md` §2.5): this table reports coverage only, since no",
        "label is read here. What fraction of rows clear the threshold is a fact about the",
        "corpus; whether those calls are right is measured in `OPERATING_POINTS.md`.",
        "",
        "## Servable, by variant",
        "",
        "| variant | sagittal axis | files |",
        "|---|---|---|",
        *[f"| `{v}` | Deg_{a} | {n} |"
          for (v, a), n in sorted(by_variant.items(), key=lambda x: -x[1])],
        "",
        "## Abstentions",
        "",
    ]
    if not bad:
        lines.append("none — every raw file labelled.")
    for kind, group in sorted(by_refusal.items(), key=lambda x: -len(x[1])):
        lines.append(f"### {kind} — {len(group)} file(s)")
        lines.append("")
        # Grouped by the message, not one line per file; a refusal that fires on twelve
        # files fires for the SAME stated reason on all twelve- printing that sentence
        # twelve times buries the only part that varies, which recordings are affected, and
        # makes one cause look like twelve problems; the grouping is by exact message rather
        # than by `kind`, so two variants refused by the same exception still read as the two
        # distinct causes they are
        by_message: dict[str, list[dict]] = {}
        for r in group:
            by_message.setdefault(r["reason"], []).append(r)
        for message, files in by_message.items():
            lines.append(message)
            lines.append("")
            lines += [f"- `{r['file']}`" for r in files]
            lines.append("")
    lines += [
        "## Dominant ambiguity reason, by file",
        "",
        "| reason | files where it dominates |",
        "|---|---|",
        *[f"| `{k}` | {v} |" for k, v in sorted(by_reason.items(), key=lambda x: -x[1])],
        "",
    ]
    # Last, and about the files that DID label; everything above says what the sweep could
    # serve; this says which of those outputs should not be trusted as they stand
    lines += summarize_plausibility(findings_by_file or {}, n_files=len(ok))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Label every raw device log under data/raw with stand/walk + confidence.")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR,
                    help="destination root; mirrors the data/raw session directories "
                         "(default: labeled_raw/ at the repo root)")
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--preset", default=None, help="high_coverage | balanced | high_precision")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--full", action="store_true",
                    help="write the complete labelled frame (input columns + derived "
                         "features + state/confidence/ambiguous/reason/alternative) "
                         "instead of the annotated-corpus shape")
    ap.add_argument("--no-verdict", action="store_true",
                    help=f"drop {GUESS_COL} and {', '.join(VERDICT_COLUMNS)}, leaving only "
                         f"the {len(GOLDEN_COLUMNS)} columns the annotated corpus uses; "
                         f"abstention survives in Label alone")
    ap.add_argument("--summary-only", action="store_true",
                    help="sweep without writing the per-file labelled CSVs (a full corpus "
                         "run is a few GB)")
    ap.add_argument("--limit", type=int, default=0, help="first N files only (smoke test)")
    args = ap.parse_args()

    _model, meta = load_champion(args.model_dir)
    if args.threshold is not None:
        threshold = args.threshold
    else:
        name = args.preset or meta["default_preset"]
        if name not in meta["presets"]:
            raise SystemExit(f"unknown preset {name!r}; known: {sorted(meta['presets'])}")
        threshold = meta["presets"][name]

    raw_dir = (REPO_ROOT / args.raw).resolve()
    paths = sorted(raw_dir.rglob("*.csv"))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise SystemExit(f"no CSV files under {raw_dir}")

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[label_all] {len(paths)} raw files, threshold {threshold:.2f}, "
          f"model {args.model_dir}")

    # Every operating point, from the one pass below; the shipped threshold is forced into
    # the grid whatever it is, so the self-check further down always has a point at which
    # the derived count and the frame's own `ambiguous` column must agree exactly
    gates = gates_off()
    derivable = not any(gates.values())
    grid = sorted({*DEFAULT_THRESHOLDS, round(float(threshold), 4)})
    if not derivable:
        print(f"[label_all] preset sweep SKIPPED: {gates} — with any of these on, "
              f"abstention is not a function of confidence alone and a swept coverage "
              f"would overstate what the serve path delivers")

    rows, unexpected = [], []
    sweep: list[dict] = []
    sweep_mismatch: list[tuple] = []
    findings_by_file: dict[str, list[dict]] = {}
    for i, p in enumerate(paths, 1):
        out, row, findings = label_one(p, args.model_dir, threshold)
        rows.append(row)
        if findings:
            findings_by_file[row["file"]] = findings

        if out is not None and derivable:
            committed, boundary = coverage_at(out, grid)
            # The derivation must reproduce the shipped commit rule on the one threshold
            # where the frame already carries the answer, to within the rounding band it
            # cannot see past; a disagreement LARGER than that band means some rule other
            # than the confidence test is in play and the other eight columns are fiction
            # Recorded per file rather than asserted, so one odd file names itself instead
            # of ending a 91-file sweep with a traceback
            key = f"{threshold:.2f}"
            delta = abs(committed[key] - row["n_confident"])
            if delta > boundary[key]:
                sweep_mismatch.append((row["file"], committed[key], row["n_confident"],
                                       boundary[key]))
            sweep.append({"file": row["file"], "session": row["session"],
                          "rows": row["rows"], "committed": committed,
                          "boundary": boundary})

        if out is None:
            print(f"[{i:>3}/{len(paths)}] {p.name:<38} ABSTAINED {row['refusal']}: "
                  f"{row['reason']}")
            if str(row["refusal"]).startswith("UNEXPECTED"):
                unexpected.append(row)
            continue

        if not args.summary_only:
            dest = out_dir / session_of(p)["session_dir"] / f"{p.stem}_labelled.csv"
            dest.parent.mkdir(parents=True, exist_ok=True)
            emitted = out if args.full else verdict_frame(out, strict=args.no_verdict)
            emitted.to_csv(dest, index=False)

        print(f"[{i:>3}/{len(paths)}] {p.name:<38} {row['variant']} Deg_{row['axis']}  "
              f"{row['rows']:>7,} rows  {row['confident_frac']:>6.1%} confident  "
              f"stand={row['n_stand']:,} walk={row['n_walk']:,}")
        # Printed as it happens, not only in the report: a sweep over 91 files takes long
        # enough that an operator watching it should learn a file is suspect while there is
        # still time to stop and look, rather than reading it afterwards
        for f in findings:
            if f["severity"] == "implausible":
                print(f"{'':>18} IMPLAUSIBLE {f['code']}: {f['detail']}")

    pd.DataFrame(rows).to_csv(out_dir / "label_summary.csv", index=False)
    with (out_dir / "abstentions.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            if not r["labelled"]:
                fh.write(json.dumps({**r, "needs_human": True}, ensure_ascii=False) + "\n")
    # Written even when empty, for the same reason `abstentions.jsonl` is: a zero-line file
    # says the bounds ran and nothing tripped them, where a missing file says nobody looked
    with (out_dir / "plausibility.jsonl").open("w", encoding="utf-8") as fh:
        for f, xs in findings_by_file.items():
            for x in xs:
                fh.write(json.dumps({"file": f, **x}, ensure_ascii=False) + "\n")
    (out_dir / "label_report.md").write_text(
        render_report(rows, threshold, args.model_dir, findings_by_file), encoding="utf-8")

    # Every operating point on the corpus a customer actually has, which is a different
    # question from the curve in `OPERATING_POINTS.md`: that one says what accuracy a
    # threshold buys on annotated data, this one says what coverage it costs on THIS data
    # Neither answers the other, and the decision needs both- `stages.breakdown` puts them
    # in one table; **Coverage only, and it always will be: there is no ground truth here.**
    if derivable and sweep:
        if sweep_mismatch:
            print(f"\n[label_all] preset sweep NOT written: the confidence test disagrees "
                  f"with `ambiguous` on {len(sweep_mismatch)} file(s) at the shipped "
                  f"threshold by more than the rounding band, so the swept columns cannot "
                  f"be trusted:")
            for f, got, want, band in sweep_mismatch[:5]:
                print(f"  {f}: derived {got:,} committed, frame says {want:,} "
                      f"(rounding band {band:,})")
        else:
            total_rows = sum(s["rows"] for s in sweep)
            corpus = []
            for t in grid:
                key = f"{t:.2f}"
                c = sum(s["committed"][key] for s in sweep)
                corpus.append({
                    "threshold": t, "rows": total_rows, "committed": c,
                    "coverage": (c / total_rows) if total_rows else float("nan"),
                    # The resolution of the count above, not a second finding: rows whose
                    # emitted confidence cannot say which side of this threshold the
                    # decision fell on; carried so a reader can see when a coverage
                    # difference between two points is smaller than the uncertainty in it
                    "boundary": sum(s["boundary"][key] for s in sweep),
                    # Files committing to less than half their rows: the same reporting line
                    # `breakdown` uses; a corpus mean cannot be acted on; a file count can
                    "files_below_half": sum(
                        1 for s in sweep
                        if s["rows"] and s["committed"][key] / s["rows"] < 0.50),
                    "files_zero": sum(1 for s in sweep if s["committed"][key] == 0),
                })
            (out_dir / PRESET_SWEEP_FILENAME).write_text(json.dumps({
                "generated": datetime.now(timezone.utc).isoformat(),
                "model_dir": str(args.model_dir),
                "shipped_threshold": float(threshold),
                "presets": meta["presets"],
                "gates": gates,
                "n_files": len(sweep),
                "corpus": corpus,
                "per_file": sweep,
            }, indent=2), encoding="utf-8")
            print(f"\n[label_all] operating points on this corpus "
                  f"({len(sweep)} files, {total_rows:,} rows) — coverage only:")
            by_t = {round(v, 4): k for k, v in meta["presets"].items()}
            for c in corpus:
                tag = by_t.get(round(c["threshold"], 4), "")
                mark = " <- shipped" if abs(c["threshold"] - threshold) < 1e-9 else ""
                print(f"[label_all]   {c['threshold']:.2f} {tag:<15} "
                      f"coverage {c['coverage']:>7.2%}  "
                      f"files under 50%: {c['files_below_half']:>3}{mark}")
            print(f"[label_all] sweep -> {out_dir / PRESET_SWEEP_FILENAME}")

    # Which champion produced these calls, recorded next to them; this sweep is the one step
    # that reads another stage's artifact and writes its own, so it is the one place where
    # re-running a stage alone- retraining, the normal way to iterate- leaves a whole corpus
    # of labelled CSVs carrying a model that no longer exists; every file is present, every
    # file parses, and they are wrong together; stamped here and checked in
    # `stages.breakdown`: a sweep cannot know it has gone stale after it has finished running
    stamp_inputs(out_dir, {"champion": args.model_dir / "champion.joblib",
                           "model_meta": args.model_dir / "model_meta.json"})

    # Same gate S1 enforces: every raw file lands in exactly one bucket
    ok = sum(1 for r in rows if r["labelled"])
    bad = len(rows) - ok
    assert ok + bad == len(paths), f"partition broken: {ok + bad} != {len(paths)} raw files"

    n_rows = sum(r["rows"] for r in rows if r["labelled"])
    n_conf = sum(r["n_confident"] for r in rows if r["labelled"])
    print(f"\n[label_all] {ok} labelled, {bad} abstained, {len(paths)} accounted for")
    print(f"[label_all] {n_rows:,} rows, {n_conf:,} confident "
          f"({n_conf / max(n_rows, 1):.1%}) at threshold {threshold:.2f}")
    if not args.summary_only:
        cols = (GOLDEN_COLUMNS if args.no_verdict
                else (*GOLDEN_COLUMNS, GUESS_COL, *VERDICT_COLUMNS))
        shape = "full frame" if args.full else f"{len(cols)} cols: {', '.join(cols)}"
        print(f"[label_all] per-file CSVs ({shape}) -> {out_dir}\\<session>\\")
    print(f"[label_all] summary -> {out_dir / 'label_summary.csv'}, "
          f"report -> {out_dir / 'label_report.md'}")

    # An unexpected exception is not an abstention- the file was refused by a bug, not by
    # a rule; non-zero exit so a batch run cannot pass while quietly skipping files
    if unexpected:
        print(f"\n[label_all] {len(unexpected)} file(s) failed for UNEXPECTED reasons:")
        for r in unexpected:
            print(f"  {r['file']}: {r['refusal']} {r['reason']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
