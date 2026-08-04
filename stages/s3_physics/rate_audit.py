# S3 rate-invariance audit: the body or the sampling grid? python -m stages.s3_physics.rate_audit

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import decimate

from stages.console import use_replacement_encoding
from stages.report import add_report_flag
from stages.s1_clean.config import CANONICAL_HZ, DECIMATE_FILTER
from stages.s2_ml.dataset import FEATURES, Trial, load_dataset
from stages.s2_ml.features import WindowSpec, rest_reference
from stages.s3_physics.anchors import ANCHOR_NAMES, WALKING, window_anchors

REPO_ROOT = Path(__file__).resolve().parents[2]
S3_OUT_DIR = REPO_ROOT / "runs" / "s3_physics"

# Halve the rate: a genuine bandwidth cut, not timestamp quantization at the same rate
AUDIT_FACTOR = 2

# Change above this => the anchor tracks the grid, not the body
AUDIT_TOL = 0.10

# anchors this audit EXPECTS to fail, named rather than tolerated silently; main exits non-zero
EXPECTED_RATE_DEPENDENT = frozenset({"gyro_energy"})

# how each change is measured; mixing them measures the wrong thing (ratio scale vs correlation)
ANCHOR_METRIC = {
    "periodicity": "abs",   # normalized autocorr, [0, 1]
    "antiphase": "abs",     # -pearson r, [-1, 1]
    "grav_stab": "abs",     # stability score, (0, 1]
    "gyro_energy": "rel",   # sum of deg^2/s^2, ratio scale
}

# denominator floor for ratio-scale anchors: a near-zero native value would fake a rate dependence
ANCHOR_FLOOR = {"gyro_energy": 1.0}

# context taken per side before decimating; 128 clears filtfilt's padlen, and skips are reported
AUDIT_PAD_SAMPLES = 128


# decimate `seg[start:start+n]` using `pad` samples of real context either side
def _decimate_with_context(seg: pd.DataFrame, start: int, n: int,
                           factor: int, pad: int) -> pd.DataFrame | None:
    a0, b0 = start - pad, start + n + pad
    if a0 < 0 or b0 > len(seg):
        return None
    lead, keep = pad // factor, n // factor
    big = {c: decimate(seg[c].to_numpy(float)[a0:b0], factor,
                       ftype=DECIMATE_FILTER, zero_phase=True) for c in FEATURES}
    if min(len(v) for v in big.values()) < lead + keep:
        return None
    return pd.DataFrame({c: v[lead:lead + keep] for c, v in big.items()})


def _delta(anchor: str, native: float, decimated: float) -> float:
    native, decimated = float(native), float(decimated)
    if ANCHOR_METRIC[anchor] == "abs":
        return abs(native - decimated)
    return abs(native - decimated) / (abs(native) + ANCHOR_FLOOR[anchor])


# one verdict per anchor
def audit_anchors(trials: list[Trial], spec: WindowSpec | None = None,
                  factor: int = AUDIT_FACTOR, tol: float = AUDIT_TOL,
                  pad: int = AUDIT_PAD_SAMPLES) -> dict[str, dict]:
    spec = spec or WindowSpec()
    deltas: dict[str, list[float]] = {a: [] for a in ANCHOR_NAMES}
    n_total = n_walking = n_unpadded = 0

    for trial in trials:
        frame = trial.frame.reset_index(drop=True)
        if frame.empty:
            continue
        # the same rest zero the anchor table and the S2 features use, so the origin never differs
        _zeros, center, _trusted = rest_reference(frame, spec.fs_hz)
        for _seg_id, seg in frame.groupby("segment", sort=True):
            seg = seg.reset_index(drop=True)
            if len(seg) < spec.n:
                continue
            for start in range(0, len(seg) - spec.n + 1, spec.step):
                win = seg.iloc[start:start + spec.n]
                native = window_anchors(win, spec.fs_hz, center)
                n_total += 1
                if native["swap_verdict"] != WALKING:  # audit only where anchors describe
                    continue
                n_walking += 1
                dec = _decimate_with_context(seg, start, spec.n, factor, pad)
                if dec is None or len(dec) < 8:
                    n_unpadded += 1  # too close to a segment edge to pad both sides
                    continue
                decd = window_anchors(dec, spec.fs_hz / factor, center)
                for a in ANCHOR_NAMES:
                    deltas[a].append(_delta(a, native[a], decd[a]))

    report: dict[str, dict] = {}
    for a in ANCHOR_NAMES:
        med = float(np.median(deltas[a])) if deltas[a] else float("nan")
        report[a] = {
            "metric": ANCHOR_METRIC[a],
            "median_delta": round(med, 4),
            "p90_delta": round(float(np.percentile(deltas[a], 90)), 4) if deltas[a] else None,
            "tol": tol,
            "verdict": "invariant" if med <= tol else "rate_dependent",
            "n_windows_total": n_total,
            "n_windows_walking": n_walking,
            "n_windows_audited": len(deltas[a]),
            "n_skipped_unpaddable": n_unpadded,
            "pad_samples": pad,
        }
    return report


def render(report: dict[str, dict]) -> str:
    r0 = report[ANCHOR_NAMES[0]]
    lines = [
        "## Rate-invariance audit", "",
        f"Every window decimated {CANONICAL_HZ:.0f} Hz -> {CANONICAL_HZ / AUDIT_FACTOR:.0f} Hz "
        f"through S1's anti-aliasing FIR, anchors recomputed, tolerance {AUDIT_TOL:g}. "
        f"Gated to the {r0['n_windows_walking']:,} windows the swap rule calls WALKING "
        f"(of {r0['n_windows_total']:,}) — standing has no cadence to compare, and the "
        f"gate keeps the audit label-free.", "",
        f"Each window is decimated with **{r0['pad_samples']} samples of real context** "
        f"either side, then trimmed back: on a bare window the filter's edge transient "
        f"covers over half the samples and reads as a rate dependence that is not there "
        f"(it condemned `antiphase` at Δ 0.1734 vs 0.0005 corrected). "
        f"{r0['n_skipped_unpaddable']:,} windows sat too close to a segment boundary to "
        f"pad on both sides and were skipped; {r0['n_windows_audited']:,} were audited.", "",
        "| anchor | metric | median Δ | p90 Δ | verdict |", "|---|---|---|---|---|",
    ]
    for a in ANCHOR_NAMES:
        r = report[a]
        p90 = f"{r['p90_delta']:.4f}" if r["p90_delta"] is not None else "—"
        lines.append(f"| `{a}` | {r['metric']} | {r['median_delta']:.4f} | {p90} "
                     f"| **{r['verdict']}** |")
    return "\n".join(lines)


# the standalone page
def render_report(report: dict[str, dict]) -> str:
    failed = [a for a in ANCHOR_NAMES if report[a]["verdict"] == "rate_dependent"]
    return "\n".join([
        "# S3 rate-invariance audit", "",
        render(report), "",
        "## Reading a failure", "",
        "**An anchor that fails is not a bug.** `gyro_energy` is defined the way that "
        "fails, on purpose, as this audit's negative control: it sums over samples, so "
        "halving the sample count halves it — a claim about the sampling grid, exactly "
        "what §2.3 warns about. An audit that has never rejected anything is not evidence "
        "that the rest passed (§11.1).", "",
        f"This run: **{len(failed)} of {len(ANCHOR_NAMES)}** anchors rate-dependent "
        f"({', '.join(f'`{a}`' for a in failed) if failed else 'none'}). "
        + ("The negative control fired, so the passes mean something."
           if "gyro_energy" in failed else
           "**`gyro_energy` did NOT fire.** The control is supposed to fail; a clean "
           "sweep means the test lost its teeth, not that everything is invariant. "
           "Check `AUDIT_TOL` and the padding before believing any verdict above."), "",
    ])


def main() -> None:
    use_replacement_encoding()
    ap = argparse.ArgumentParser(
        description="S3 rate-invariance audit: is an anchor about the body or the clock?")
    ap.add_argument("--out", type=Path, default=S3_OUT_DIR)
    add_report_flag(ap)
    ap.add_argument("--include-lockbox", action="store_true",
                    help="audit lockbox trials too (see the note below — normally wrong)")
    args = ap.parse_args()

    spec = WindowSpec()

    # the lockbox stays sealed: a rate verdict on lockbox windows would make it a thing we looked at
    trials = [t for t in load_dataset() if args.include_lockbox or t.split != "lockbox"]
    print(f"[rate] auditing {len(trials)} trials"
          + (" (LOCKBOX INCLUDED)" if args.include_lockbox else ""))

    report = audit_anchors(trials, spec)
    for a, r in report.items():
        print(f"[rate]   {a:<14} {r['metric']:>4} median delta {r['median_delta']:>8.4f}  "
              f"{r['verdict']}")

    r0 = report[ANCHOR_NAMES[0]]
    print(f"[rate] {r0['n_windows_audited']:,} of {r0['n_windows_walking']:,} walking "
          f"windows audited; {r0['n_skipped_unpaddable']:,} too close to a segment edge")

    out_dir = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rate_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.report:
        (out_dir / "rate_audit.md").write_text(render_report(report), encoding="utf-8")

    # Model-free by construction: this reads the window grid and the anchors, never the champion.
    # Declared rather than left unstamped, so "no model dependency" and "never checked" stay distinct.
    stamp_inputs(out_dir, {}, stage="rate_audit")

    print(f"[rate] -> {out_dir / 'rate_audit.md'}")

    # the gate; artifacts written first, so a run that stops here leaves the page explaining why
    new = sorted(a for a in ANCHOR_NAMES
                 if report[a]["verdict"] == "rate_dependent"
                 and a not in EXPECTED_RATE_DEPENDENT)
    silent = sorted(a for a in EXPECTED_RATE_DEPENDENT
                    if report[a]["verdict"] != "rate_dependent")
    if args.include_lockbox and (new or silent):
        print("[rate] NOT gating: --include-lockbox audits a different window population.")
        return
    if new:
        raise SystemExit(
            f"[rate] FAIL: {', '.join(new)} now tracks the sampling grid rather than the "
            f"body (median delta over tol {AUDIT_TOL:g}). PLAN's S3 gate is no anchor "
            f"feature without a rate-invariance verdict — see rate_audit.md, and do not "
            f"raise AUDIT_TOL to make this pass."
        )
    if silent:
        raise SystemExit(
            f"[rate] FAIL: the negative control(s) {', '.join(silent)} did NOT fire. A "
            f"sweep that rejects nothing is not evidence that the rest passed (§11.1) — "
            f"every `invariant` verdict in this run is unverified until the control fails "
            f"again. Check AUDIT_TOL and the padding before believing any of them."
        )


if __name__ == "__main__":
    main()
