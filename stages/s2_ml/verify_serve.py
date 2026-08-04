# end-to-end guard: does a device log label like its lpf_view export?
#   python -m stages.s2_ml.verify_serve
# verify_transform checks the MATH; this checks the thing a caller actually runs
# drops Label before comparing, so it proves equivalence and never correctness

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from stages.s2_ml.dataset import LABEL_COL
from stages.s2_ml.label import DEFAULT_MODEL_DIR, label_csv, load_champion
from stages.s2_ml.transform import raw_csv_to_features
from stages.s2_ml.verify_transform import find_pairs, index_raw_files

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"

# The columns that ARE the deliverable; `confidence` is compared numerically, the rest
# exactly- a reason that changes is a different explanation for the same row, which is a
# regression even when the state survives
VERDICT_COLS = ("state", "ambiguous", "reason", "alternative", "n_windows")

# The two routes differ only by float roundoff in the fourth feature column, so the
# ensemble should land on identical probabilities; a tree split sitting exactly between two
# values that differ at 1e-13 could in principle flip one window; this bound says such a
# flip is not what we are seeing, rather than tolerating it in advance
CONFIDENCE_TOLERANCE = 1e-9


# the verdict columns only, with ground truth structurally removed
def _comparable(df: pd.DataFrame) -> pd.DataFrame:
    out = df.drop(columns=[c for c in (LABEL_COL,) if c in df.columns])
    cols = [c for c in VERDICT_COLS if c in out.columns]
    got = out[cols].copy()
    # None and NaN both mean "no call"; normalize so the comparison is about verdicts
    for c in ("state", "reason", "alternative"):
        if c in got.columns:
            got[c] = got[c].where(got[c].notna(), "")
    return got


# label one recording both ways and count every row where the two disagree
def compare_pair(ann_path: Path, raw_path: Path, model_dir: Path,
                 threshold: float) -> dict:
    ann_out, _ann_prov = label_csv(ann_path, model_dir, threshold)
    raw_out, raw_prov = label_csv(raw_path, model_dir, threshold)

    if len(ann_out) != len(raw_out):
        return {"pair": ann_path.name, "raw": raw_path.name, "rows": None,
                "fatal": f"row counts differ: {len(ann_out)} vs {len(raw_out)}"}

    a, b = _comparable(ann_out), _comparable(raw_out)
    mismatches = {c: int((a[c].to_numpy() != b[c].to_numpy()).sum()) for c in a.columns}

    ca = ann_out["confidence"].to_numpy(float)
    cb = raw_out["confidence"].to_numpy(float)
    both = np.isfinite(ca) & np.isfinite(cb)
    conf_delta = float(np.max(np.abs(ca[both] - cb[both]))) if both.any() else 0.0
    # A row scored on one route and not the other is a disagreement of its own kind
    coverage_mismatch = int((np.isfinite(ca) != np.isfinite(cb)).sum())

    return {
        "pair": ann_path.name, "raw": raw_path.name, "rows": int(len(a)),
        "variant": raw_prov["variant_id"], "axis": raw_prov["sagittal_deg_axis"],
        "mismatches": mismatches,
        "worst_column": max(mismatches, key=lambda c: mismatches[c]) if mismatches else None,
        "n_mismatched": sum(mismatches.values()) + coverage_mismatch,
        "coverage_mismatch": coverage_mismatch,
        "conf_delta": conf_delta,
        "fatal": None,
    }


# attempt the bridge on every raw file; records the outcome, never raises
def sweep(raw_dir: Path, model_dir: Path) -> list[dict]:
    rows = []
    for p in sorted(raw_dir.rglob("*.csv")):
        rel = str(p.relative_to(REPO_ROOT))
        try:
            _rows, feat, prov = raw_csv_to_features(p)
            rows.append({"file": rel, "servable": True, "variant": prov["variant_id"],
                         "axis": prov["sagittal_deg_axis"], "n_rows": prov["n_rows"],
                         "reason": None})
        except Exception as exc:
            rows.append({"file": rel, "servable": False, "variant": None, "axis": None,
                         "n_rows": None,
                         "reason": f"{type(exc).__name__}: {str(exc).splitlines()[0]}"})
    return rows


def render_sweep(rows: list[dict]) -> str:
    ok = [r for r in rows if r["servable"]]
    bad = [r for r in rows if not r["servable"]]
    lines = [f"corpus sweep: {len(ok)} of {len(rows)} raw files servable "
             f"({len(ok) / max(len(rows), 1):.0%})"]
    by_variant: dict[str, int] = {}
    for r in ok:
        by_variant[r["variant"]] = by_variant.get(r["variant"], 0) + 1
    for v, n in sorted(by_variant.items(), key=lambda x: -x[1]):
        axis = next(r["axis"] for r in ok if r["variant"] == v)
        lines.append(f"   servable  {v}  Deg_{axis}  x{n}")
    by_reason: dict[str, list[str]] = {}
    for r in bad:
        by_reason.setdefault(r["reason"].split(":")[0], []).append(r["file"])
    for reason, files in sorted(by_reason.items(), key=lambda x: -len(x[1])):
        lines.append(f"   ABSTAINED {reason}  x{len(files)}")
        for f in files:
            lines.append(f"      {f}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Verify the raw serve path against its lpf_view export, end to end.")
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--threshold", type=float, default=None,
                    help="default: the champion's own default preset")
    ap.add_argument("--skip-pairs", action="store_true")
    ap.add_argument("--skip-sweep", action="store_true")
    args = ap.parse_args()

    _model, meta = load_champion(args.model_dir)
    threshold = (args.threshold if args.threshold is not None
                 else meta["presets"][meta["default_preset"]])

    failed = False

    if not args.skip_pairs:
        pairs = find_pairs(index_raw_files())
        if not pairs:
            raise SystemExit("no paired recordings found — cannot verify the serve path")
        print(f"[verify_serve] {len(pairs)} paired recordings, threshold {threshold:.2f}\n")
        results = []
        for i, (ann_path, raw_path, _raw_df, _vid) in enumerate(pairs, 1):
            r = compare_pair(ann_path, Path(raw_path), args.model_dir, threshold)
            results.append(r)
            if r["fatal"]:
                print(f"  {i:>2}/{len(pairs)} {ann_path.name:<34} FATAL {r['fatal']}")
                continue
            verdict = "ok" if r["n_mismatched"] == 0 else f"{r['n_mismatched']} MISMATCHED"
            print(f"  {i:>2}/{len(pairs)} {ann_path.name:<34} {r['variant']} "
                  f"Deg_{r['axis']}  {r['rows']:>7,} rows  "
                  f"max|Δconfidence|={r['conf_delta']:.3g}  {verdict}")

        total_rows = sum(r["rows"] or 0 for r in results)
        total_bad = sum(r["n_mismatched"] for r in results if r["fatal"] is None)
        worst_conf = max((r["conf_delta"] for r in results if r["fatal"] is None),
                         default=0.0)
        fatal = [r for r in results if r["fatal"]]
        print(f"\n{len(results)} pairs, {total_rows:,} rows compared")
        print(f"disagreeing verdicts: {total_bad}")
        print(f"worst max-abs confidence difference: {worst_conf:.3g} "
              f"(tolerance {CONFIDENCE_TOLERANCE:g})")
        if fatal or total_bad or worst_conf > CONFIDENCE_TOLERANCE:
            for r in fatal:
                print(f"FAIL: {r['pair']}: {r['fatal']}")
            for r in results:
                if r["fatal"] is None and r["n_mismatched"]:
                    print(f"FAIL: {r['pair']}: {r['mismatches']} "
                          f"(+{r['coverage_mismatch']} coverage)")
            print("FAIL: a raw device log does not label like its own lpf_view export.")
            failed = True
        else:
            print("PASS: raw and lpf_view routes agree on every row of every pair.")

    if not args.skip_sweep:
        print()
        print(render_sweep(sweep(RAW_DIR, args.model_dir)))

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
