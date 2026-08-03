"""S3 physics: build the anchor table and audit it.

    python -m stages.s3_physics.run --out runs/s3_physics

Outputs:
    anchors.csv     one row per window, keyed to the S2 window grid
    rate_audit.json per-anchor rate-invariance verdict (the PLAN S3 gate)
    physics.md      human-readable summary

Deterministic and label-free end to end. The anchor table is built the same way for a
labeled trial and a raw recording, which is what lets S4 fuse on files that have no
ground truth at all.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from stages.s2_ml.dataset import load_dataset
from stages.s2_ml.features import WindowSpec
from stages.s3_physics.anchors import (
    AMBIGUOUS,
    STANDING,
    WALKING,
    trial_anchors,
)
from stages.s3_physics.rate_audit import audit_anchors, render as render_audit

REPO_ROOT = Path(__file__).resolve().parents[2]
S3_OUT_DIR = REPO_ROOT / "runs" / "s3_physics"
ANCHORS_CSV = "anchors.csv"
AUDIT_JSON = "rate_audit.json"

VERDICTS = (STANDING, AMBIGUOUS, WALKING)


def build_anchor_table(trials, spec: WindowSpec | None = None) -> pd.DataFrame:
    """Anchors for every window of every trial, on one shared window grid.

    Each trial is calibrated on its own rest posture via `features.rest_reference` — the
    same zero the S2 interleg features use. A trial that never rests falls back to a
    whole-recording median and carries `rest_offset_trusted = False`, which S4 turns into
    a stated reason rather than a silent caveat.
    """
    spec = spec or WindowSpec()
    frames = [t for t in (trial_anchors(tr, spec) for tr in trials) if not t.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def render(anchors: pd.DataFrame, audit: dict) -> str:
    n = len(anchors)
    fixed = anchors["swap_verdict"].value_counts()
    adapt = anchors["swap_verdict_adaptive"].value_counts()
    untrusted = int((~anchors["rest_offset_trusted"]).sum())

    lines = [
        "# S3 Physics", "",
        f"- windows: **{n:,}** across **{anchors['rev'].nunique()}** revs",
        f"- rest zero untrusted on **{untrusted:,}** windows "
        f"({untrusted / n:.1%}) — those recordings never rest, so their interleg zero is "
        f"a whole-recording median rather than a measured standing posture (§10.2)",
        "",
        "## Swap-rule verdicts", "",
        "The rule has zero fitted parameters: `delta = 1°` is a sensor noise floor and "
        "`1 swap` is the only integer between measured standing (0) and measured walking "
        "(2). The adaptive column sizes the span to ~2 detected strides (§10.6/10.7), "
        "which is what rescues slow gait from abstaining.", "",
        "| verdict | fixed 2 s span | stride-adaptive span |", "|---|---|---|",
    ]
    for v in VERDICTS:
        lines.append(f"| `{v}` | {int(fixed.get(v, 0)):,} | {int(adapt.get(v, 0)):,} |")
    lines += [
        "",
        f"Median adaptive span: **{anchors['swap_window_s'].median():.1f} s** "
        f"(base {2 * (anchors['swap_window_s'].min() / 2):.1f} s, "
        f"max {anchors['swap_window_s'].max():.1f} s).",
        "", render_audit(audit), "",
        "*An anchor that fails is not a bug — `gyro_energy` is defined the way that fails, "
        "on purpose, as the audit's negative control. An audit that has never rejected "
        "anything is not evidence that the rest passed (§11.1).*",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="S3 physics: anchors + rate-invariance audit.")
    ap.add_argument("--out", type=Path, default=S3_OUT_DIR)
    ap.add_argument("--window-s", type=float, default=None)
    args = ap.parse_args()

    default = WindowSpec()
    spec = WindowSpec(window_s=args.window_s or default.window_s,
                      stride_s=default.stride_s)

    # The lockbox stays sealed here too (§7). Physics needs no labels, but a rate verdict
    # measured partly on lockbox windows would make the lockbox a thing we had looked at.
    trials = [t for t in load_dataset() if t.split != "lockbox"]

    anchors = build_anchor_table(trials, spec)
    print(f"[s3] {len(anchors):,} windows across {anchors['rev'].nunique()} revs")

    audit = audit_anchors(trials, spec)
    for a, r in audit.items():
        print(f"[s3]   {a:<14} {r['metric']:>4} median Δ {r['median_delta']:>8.4f}  {r['verdict']}")

    out_dir = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    anchors.to_csv(out_dir / ANCHORS_CSV, index=False)
    (out_dir / AUDIT_JSON).write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (out_dir / "physics.md").write_text(render(anchors, audit), encoding="utf-8")
    print(f"[s3] artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
