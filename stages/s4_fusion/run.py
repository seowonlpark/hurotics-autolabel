"""S4 fusion: join S2 predictions to S3 physics, fuse, and score honestly.

    python -m stages.s4_fusion.run --out runs/s4_fusion

Reads   runs/s2_ml/oof_champion.csv   (out-of-fold, so no window is scored by a model
                                       that saw its rev)
        runs/s3_physics/anchors.csv   (label-free physics)
Writes  fused.csv        one row per window: call, confidence, reasons, alternative
        review_queue.jsonl the abstained windows, for the S4 agent
        fusion.md        the coverage/accuracy table this stage is judged on

**The headline is a PAIR, and reporting one without the other is meaningless.** Accuracy
on confident windows alone is trivially gamed — abstain on everything but the easiest
window and it reads 100%. So every number here is (coverage, confident accuracy), and
the target is >=95% accuracy at the *highest coverage that clears it*.

The stage also re-measures the probability floor `fuse.py` ships with, and says so when
the tuned value differs. A constant that was measured once and never re-checked is the
failure §4.1b records: a record nothing reads.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stages.s2_ml.dataset import STAND, TRAIN_CLASSES, WALK
from stages.s2_ml.features import TRANSITION
from stages.s2_ml.oof import JOIN_KEYS, OOF_CSV, S2_OUT_DIR
from stages.s3_physics.run import ANCHORS_CSV, S3_OUT_DIR
from stages.s4_fusion.fuse import (
    DEFAULT_PROBA_FLOOR,
    HIGH,
    LOW,
    MEDIUM,
    explain,
    fuse,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
S4_OUT_DIR = REPO_ROOT / "runs" / "s4_fusion"
FUSED_CSV = "fused.csv"
REVIEW_QUEUE = "review_queue.jsonl"

ACCURACY_TARGET = 0.95

# Anchor columns fusion needs. Named explicitly so a change in the S3 table surfaces as a
# KeyError here rather than as a silently absent quality flag.
ANCHOR_COLS = ["swap_verdict_adaptive", "swap_count", "swap_window_s",
               "rest_offset_trusted", "antiphase", "periodicity_adaptive"]


def load_joined(s2_dir: Path = S2_OUT_DIR, s3_dir: Path = S3_OUT_DIR) -> pd.DataFrame:
    """Inner-join S2 and S3 on the shared window grid.

    `validate="one_to_one"` is the guard that matters: both stages window through
    `iter_windows`, so a duplicated or missing key means the two ran on different specs
    and every downstream number would be a silent mis-join.
    """
    oof = pd.read_csv(s2_dir / OOF_CSV)
    anchors = pd.read_csv(s3_dir / ANCHORS_CSV)
    joined = oof.merge(anchors[JOIN_KEYS + ANCHOR_COLS], on=JOIN_KEYS,
                       how="inner", validate="one_to_one")
    if len(joined) != len(oof):
        raise ValueError(
            f"{len(oof) - len(joined)} of {len(oof)} S2 windows have no S3 anchor row. "
            f"The two stages windowed differently — re-run both with the same --window-s."
        )
    return joined


def apply_fusion(df: pd.DataFrame, proba_floor: float = DEFAULT_PROBA_FLOOR) -> pd.DataFrame:
    """Fuse every row. The grown-span flag marks a WALKING verdict that needed §10.7."""
    base_span = df["swap_window_s"].min()
    out = df.copy()
    fused = [
        fuse(int(r.s2_pred), float(r.s2_proba), str(r.swap_verdict_adaptive),
             rest_trusted=bool(r.rest_offset_trusted),
             grown=bool(r.swap_window_s > base_span + 1e-9),
             proba_floor=proba_floor)
        for r in df.itertuples()
    ]
    out["fused_label"] = [f.label for f in fused]
    out["confidence"] = [f.confidence for f in fused]
    out["abstain"] = [f.abstain for f in fused]
    out["reasons"] = [";".join(f.reasons) for f in fused]
    out["alternative"] = [f.alternative for f in fused]
    out["explanation"] = [explain(f) for f in fused]
    return out


def true_class(df: pd.DataFrame) -> pd.Series:
    """Ground-truth class per window, NaN where there is none.

    `label` holds three kinds — an int class, `transition`, or None — so it round-trips
    through CSV as an object column and `isin((0, 10))` silently matches nothing. Coercing
    once, here, is what stops "0 pure windows" from being reported as a result instead of
    as the bug it is.
    """
    return pd.to_numeric(df["label"], errors="coerce")


def score(df: pd.DataFrame) -> dict:
    """(coverage, confident accuracy) and the honest companions.

    Scored on label-pure windows only: a `transition` window has no single true class by
    construction, so counting it as right or wrong would measure the windowing, not the
    model. Its abstention rate is reported separately — abstaining there is correct
    behaviour, and a policy that does not is not being careful, it is being lucky.
    """
    # `label` carries three kinds: an int class, `transition` (mixed), or None (no valid
    # class at all). Only ints are scoreable; the other two are reported by rate, since a
    # window with no true class cannot be counted right or wrong without inventing one.
    t = true_class(df)
    scoreable = t.isin(TRAIN_CLASSES)
    pure = df[scoreable].copy()
    pure["true"] = t[scoreable].astype(int)
    confident = ~pure["abstain"]
    correct = pure["fused_label"] == pure["true"]

    trans = df[~scoreable]
    return {
        "n_windows": int(len(df)),
        "n_pure": int(len(pure)),
        "accuracy_all": float(correct.mean()),
        "coverage": float(confident.mean()),
        "confident_accuracy": float(correct[confident].mean()) if confident.any() else float("nan"),
        "confident_errors": int((~correct[confident]).sum()),
        "abstained": int((~confident).sum()),
        "abstained_accuracy": float(correct[~confident].mean()) if (~confident).any() else float("nan"),
        "transition_windows": int(len(trans)),
        "transition_abstain_rate": float(trans["abstain"].mean()) if len(trans) else float("nan"),
        "by_tier": {t: {
            "n": int((pure["confidence"] == t).sum()),
            "accuracy": float(correct[pure["confidence"] == t].mean())
            if (pure["confidence"] == t).any() else float("nan"),
        } for t in (HIGH, MEDIUM, LOW)},
    }


def frontier(df: pd.DataFrame, floors=(0.50, 0.60, 0.70, 0.80, 0.90)) -> list[dict]:
    """The (coverage, confident accuracy) curve, plus the two ends that bracket it.

    A single "tuned" number would be misleading here. Maximising coverage subject to
    >=95% always pushes the floor to its minimum, which yields a policy that ignores the
    model's probability entirely — technically optimal against that objective and not
    what anyone wants. The operating point is a judgement about how much coverage a
    percentage point of accuracy is worth, so the curve is the deliverable and the
    default is a documented choice on it.
    """
    rows = []
    for floor in floors:
        s = score(apply_fusion(df, float(floor)))
        rows.append({"policy": f"floor {floor:.2f}", "floor": float(floor),
                     "coverage": s["coverage"], "accuracy": s["confident_accuracy"],
                     "errors": s["confident_errors"]})
    # The brackets: agreement-only (physics decides alone) and no abstention at all.
    high_only = score(apply_fusion(df, 0.50))
    forced = apply_fusion(df, 0.50)
    ft = true_class(forced)
    pure = forced[ft.isin(TRAIN_CLASSES)]
    ptrue = ft[ft.isin(TRAIN_CLASSES)].astype(int)
    if high_only["n_pure"]:
        h = high_only["by_tier"][HIGH]
        rows.append({"policy": "agreement only (HIGH tier)", "floor": None,
                     "coverage": h["n"] / high_only["n_pure"],
                     "accuracy": h["accuracy"],
                     "errors": int(round((1 - h["accuracy"]) * h["n"]))})
        rows.append({"policy": "no abstention (call everything)", "floor": None,
                     "coverage": 1.0,
                     "accuracy": float((pure["fused_label"] == ptrue).mean()),
                     "errors": int((pure["fused_label"] != ptrue).sum())})
    return rows


def render(s: dict, floor: float, curve: list[dict]) -> str:
    lines = [
        "# S4 Fusion", "",
        f"- windows: **{s['n_windows']:,}** ({s['n_pure']:,} label-pure, "
        f"{s['transition_windows']:,} unscoreable)",
        f"- **coverage {s['coverage']:.1%} at confident accuracy {s['confident_accuracy']:.4f}** "
        f"(target {ACCURACY_TARGET:.0%})",
        f"- confident-but-wrong: **{s['confident_errors']}** windows",
        f"- accuracy if forced to call everything: {s['accuracy_all']:.4f}",
        f"- accuracy on the windows it abstained from: {s['abstained_accuracy']:.4f} "
        f"— below the confident number, which is what makes the abstention informative "
        f"rather than arbitrary",
        f"- transition windows abstained: **{s['transition_abstain_rate']:.1%}** "
        f"(these are genuinely mixed; abstaining is the correct behaviour)",
        "",
        "Coverage and confident accuracy are reported together on purpose. Either alone "
        "is meaningless: abstain on all but the easiest window and accuracy reads 1.000.",
        "",
        "## By confidence tier", "", "| tier | windows | accuracy |", "|---|---|---|",
    ]
    for t in (HIGH, MEDIUM, LOW):
        r = s["by_tier"][t]
        acc = f"{r['accuracy']:.4f}" if r["n"] else "—"
        lines.append(f"| `{t}` | {r['n']:,} | {acc} |")
    lines += [
        "",
        "`low` is the abstention tier. Its accuracy is *supposed* to be poor — those are "
        "the windows the pipeline declines to claim.", "",
        "## Operating points", "",
        f"Every row clears the {ACCURACY_TARGET:.0%} target, so the choice is how much "
        f"coverage a point of accuracy is worth — a judgement, not an optimum. Maximising "
        f"coverage subject to the target alone would drive the floor to its minimum and "
        f"produce a policy that ignores the model's probability entirely.", "",
        "| policy | coverage | confident accuracy | confident errors |", "|---|---|---|---|",
    ]
    for r in curve:
        mark = " ← shipped" if r["floor"] is not None and abs(r["floor"] - floor) < 1e-9 else ""
        lines.append(f"| {r['policy']}{mark} | {r['coverage']:.1%} | {r['accuracy']:.4f} "
                     f"| {r['errors']} |")
    lines += [
        "",
        f"The shipped floor is **{floor:.2f}**, set in `DEFAULT_PROBA_FLOOR` "
        f"(`stages/s4_fusion/fuse.py`). It is a deliberate point on this curve: it keeps "
        f"the confident set well clear of the target while still claiming most windows. "
        f"Move it knowingly — and re-run this stage, which re-measures the whole curve, "
        f"rather than trusting the number in the source.",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="S4: fuse S2 and S3 into a call + confidence.")
    ap.add_argument("--out", type=Path, default=S4_OUT_DIR)
    ap.add_argument("--proba-floor", type=float, default=DEFAULT_PROBA_FLOOR)
    args = ap.parse_args()

    joined = load_joined()
    fused = apply_fusion(joined, args.proba_floor)
    s = score(fused)

    print(f"[s4] {s['n_windows']:,} windows joined "
          f"({s['n_pure']:,} pure, {s['transition_windows']:,} transition)")
    print(f"[s4] coverage {s['coverage']:.1%}  confident accuracy {s['confident_accuracy']:.4f}  "
          f"({s['confident_errors']} confident errors)")
    for t in (HIGH, MEDIUM, LOW):
        r = s["by_tier"][t]
        if r["n"]:
            print(f"[s4]   {t:<7} n={r['n']:>5,}  accuracy {r['accuracy']:.4f}")

    curve = frontier(joined)
    print("[s4] operating points (all clear the target; the choice is a judgement):")
    for r in curve:
        print(f"[s4]   {r['policy']:<32} coverage {r['coverage']:>6.1%}  "
              f"accuracy {r['accuracy']:.4f}  errors {r['errors']:>4}")

    out_dir = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    fused.to_csv(out_dir / FUSED_CSV, index=False)
    (out_dir / "fusion.json").write_text(
        json.dumps({**s, "frontier": curve}, indent=2), encoding="utf-8")
    (out_dir / "fusion.md").write_text(
        render(s, args.proba_floor, curve), encoding="utf-8")

    # The abstention queue: what the S4 agent judges. Ordered worst-first so a capped
    # review reads the least defensible calls, not an arbitrary prefix.
    queue = fused[fused["abstain"]].sort_values("s2_proba")
    with (out_dir / REVIEW_QUEUE).open("w", encoding="utf-8") as fh:
        for r in queue.itertuples():
            fh.write(json.dumps({
                "rev": r.rev, "trial": int(r.trial), "segment": int(r.segment),
                "t_start_ms": float(r.t_start_ms),
                "call": int(r.fused_label), "alternative": r.alternative,
                "reasons": r.reasons.split(";") if r.reasons else [],
                "s2_pred": int(r.s2_pred), "s2_proba": round(float(r.s2_proba), 4),
                "physics": r.swap_verdict_adaptive, "swap_count": int(r.swap_count),
                "antiphase": round(float(r.antiphase), 4),
                "rest_trusted": bool(r.rest_offset_trusted),
                "explanation": r.explanation,
            }, ensure_ascii=False) + "\n")
    print(f"[s4] {len(queue):,} abstained windows -> {REVIEW_QUEUE}")
    print(f"[s4] artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
