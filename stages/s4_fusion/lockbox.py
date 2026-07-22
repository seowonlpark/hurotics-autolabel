# S4 lockbox evaluation: the ONE honest generalization number for the deployable FUSED model.
# the lockbox (rev8, rev13) has been sealed since Section 7 -- never trained on, never read by any S3
# plot or agent -- precisely so this number is unbiased. that only holds if you spend it ONCE,
# at the very end, on the model you are actually shipping. so:
#   - this scores the FUSED model (S2 + S3 + the fuser), not S2 alone. the fused call is the
#     deployable artifact; testing S2 in isolation would answer a question nobody ships.
#   - the S2 half is fit on EVERY non-lockbox rev (the real deployment fit, train.py's final
#     model), then predicts the sealed revs it has never seen. the S3 half computes its verdicts
#     on the sealed revs (physics is model-free, so it needs no fit).
#   - it REFUSES to run without an explicit --confirm token. opening the lockbox is a one-way
#     door: once these revs have set a number you acted on, they are spent as an unbiased test.
#     the guard makes that a deliberate act, never an accident of a stray `python -m`.
#
# STATUS: rev8 was OPENED once on 2026-07-22 -- the final number is recorded in DOMAIN_NOTES
# Section 12.5 (the fused label did not beat S2 out-of-sample; the confidence-gated system did, and
# is the deployable). rev8/rev13 are now SPENT (OPENED_REVS): main() will not re-score them even
# with --confirm, because the model was finalized knowing them so a second number is not blind.
# this module is KEPT as the reusable harness for a FUTURE sealed rev (a new model needs new
# held-out data, Section 12.5) -- not deleted, just guarded against a misleading re-open.
# see README / PLAN S4 / DOMAIN_NOTES Section 7 + Section 12.5.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from stages.s2_ml.dataset import STAND, WALK, DEFAULT_LOCKBOX_REVS, load_dataset
from stages.s2_ml.features import WindowSpec, build_windows, feature_columns
from stages.s2_ml.oof import CHAMPION_JSON, S2_OUT_DIR, champion_features
from stages.s2_ml.train import build_model, trainable
from stages.s3_physics.run import S3_OUT_DIR, build_anchor_table
from stages.s4_fusion.fuse import HIGH, LOW, MED
from stages.s4_fusion.run import JOIN_KEYS, S4_OUT_DIR, apply_fusion, calibration, per_rev, score

# opening the lockbox is deliberate: main() refuses unless --confirm matches this exactly.
CONFIRM_TOKEN = "OPEN-LOCKBOX"

RESULT_JSON = "lockbox_result.json"
RESULT_MD = "lockbox_result.md"

# the lockbox is TWO revs (both held out of training), but only one is an unbiased final test.
# rev13 was already scored on the Y-vs-X axis decision (DOMAIN_NOTES axis note: "This experiment
# scored the lockbox rev13, so rev13 is spent"), so its number is no longer blind -- it is
# reported for reference, never blended into the headline. rev8 is the clean, never-seen rev, so
# the honest generalization number is rev8 ALONE.
SEALED_REVS = ("rev8",)   # unspent -- the honest one-shot final test
SPENT_REVS = ("rev13",)   # already scored on the axis decision -- reference only, not the headline

# revs whose lockbox value has ALREADY been spent (opened once, number recorded in DOMAIN_NOTES
# Section 12.5). the harness stays here because it is the reusable tool for a FUTURE sealed rev,
# but a rev in this set must never be re-scored and called an unbiased test again: the model was
# finalized knowing it, so a second number is not blind. main() refuses to open a spent rev even
# with --confirm; to evaluate a new model, seal a genuinely NEW rev (add it to dataset's
# DEFAULT_LOCKBOX_REVS and to SEALED_REVS) that is not in this set.
OPENED_REVS = ("rev8", "rev13")


# score one subframe: fused vs S2-alone at full coverage, acting-on-confidence, tier calibration
def _score_block(g: pd.DataFrame) -> dict:
    true = g["true"].to_numpy(int)
    keep = ~g["abstain"].to_numpy(bool)
    return {
        "revs": sorted(g["rev"].unique().tolist()),
        "n_windows": int(len(g)),
        "s2_alone": score(true, g["s2_pred"].to_numpy(int)),
        "fused": score(true, g["fused_label"].to_numpy(int)),
        "acting_on_confidence": score(true, g["fused_label"].to_numpy(int), keep=keep),
        "calibration": calibration(g),
    }


# the aligned per-window table on the SEALED revs: the deployment-fit S2 (trained on every
# non-lockbox rev) predicting the lockbox, joined to the S3 adaptive verdict computed on the
# lockbox. this is the only place lockbox windows are ever fed to the model.
def lockbox_aligned(spec: WindowSpec | None = None, s2_dir: Path = S2_OUT_DIR,
                    s3_dir: Path = S3_OUT_DIR) -> pd.DataFrame:
    spec = spec or WindowSpec()
    trials = load_dataset() # includes the lockbox trials (split == "lockbox")
    windows = build_windows(trials, spec)
    feats = champion_features(feature_columns(windows), s2_dir / CHAMPION_JSON)

    # deployment fit: every non-lockbox, label-pure window (train.py's final refit)
    fit = trainable(windows, "train")
    model = build_model()
    model.fit(fit[feats].to_numpy(float), fit["label"].to_numpy(int))

    # predict the sealed revs the model has never seen
    lb = windows[(windows["split"] == "lockbox")
                 & (windows["label"].isin([STAND, WALK]))].copy()
    proba = model.predict_proba(lb[feats].to_numpy(float))
    lb["s2_pred"] = model.classes_[proba.argmax(1)]
    lb["s2_proba"] = proba.max(1)

    # S3 verdicts on the lockbox trials (physics is model-free -- no fit, nothing leaks by
    # computing it; it simply was not computed until this deliberate open)
    lb_trials = [t for t in trials if t.split == "lockbox"]
    anchors = build_anchor_table(lb_trials, spec)[JOIN_KEYS + ["swap_verdict_adaptive"]]

    m = (lb[JOIN_KEYS + ["label", "s2_pred", "s2_proba"]]
         .merge(anchors, on=JOIN_KEYS, how="inner")
         .rename(columns={"label": "true", "swap_verdict_adaptive": "s3"}))
    return m


# fuse and score. the HEADLINE is the sealed rev(s) only (rev8) -- the honest, never-seen number.
# the spent rev(s) (rev13) are scored separately and labeled reference-only, never blended in.
def evaluate_lockbox(spec: WindowSpec | None = None) -> dict:
    df = apply_fusion(lockbox_aligned(spec))
    sealed = df[df["rev"].isin(SEALED_REVS)]
    spent = df[df["rev"].isin(SPENT_REVS)]
    return {
        "held_out_of_training": list(DEFAULT_LOCKBOX_REVS),
        "sealed_revs": list(SEALED_REVS),
        "spent_revs": list(SPENT_REVS),
        # the honest generalization number -- rev8 alone
        "headline": _score_block(sealed) if len(sealed) else None,
        # rev13, already spent on the axis decision -- reference only, NOT the headline
        "reference_spent": _score_block(spent) if len(spent) else None,
        "per_rev": per_rev(df),
    }


# render one score block (fused vs S2, acting-on-confidence, tiers) under a heading
def _render_block(title: str, b: dict) -> list[str]:
    s2, f, a = b["s2_alone"], b["fused"], b["acting_on_confidence"]
    L = [f"## {title} ({b['revs']}, {b['n_windows']:,} windows)", "",
         "| model | macro-F1 | accuracy | stand-recall | walk-recall |",
         "|---|---|---|---|---|",
         f"| S2 alone | {s2['macro_f1']} | {s2['accuracy']} | {s2['stand_recall']} | {s2['walk_recall']} |",
         f"| **fused** | **{f['macro_f1']}** | {f['accuracy']} | {f['stand_recall']} | {f['walk_recall']} |",
         "",
         f"Acting on HIGH+MED (abstain on LOW): coverage **{a['coverage']}** at accuracy "
         f"**{a['accuracy']}** (stand-recall {a['stand_recall']}, walk-recall {a['walk_recall']}).", ""]
    L += ["| tier | windows | share | accuracy |", "|---|---|---|---|"]
    for tier in (HIGH, MED, LOW):
        c = b["calibration"].get(tier)
        if c:
            L.append(f"| {tier} | {c['n']:,} | {c['share']} | {c['accuracy']} |")
    L.append("")
    return L


# render the lockbox result: rev8 is the headline (honest test), rev13 is reference-only (spent)
def render(m: dict) -> str:
    L = [
        "# S4 lockbox result - the fused model on the sealed revs", "",
        "**One-shot generalization test.** The S2 half was fit on every non-lockbox rev and "
        "predicts the sealed revs unseen; the S3 half is model-free. This scores the **fused** "
        "model - the deployable artifact - not S2 alone.", "",
        f"The honest number is **rev8 alone** ({m['sealed_revs']}): the only lockbox rev never "
        f"seen. **rev13 was already spent** on the axis decision (DOMAIN_NOTES axis note), so it "
        "is reported below for reference only and is NOT the headline.", "",
    ]
    if m["headline"]:
        L += _render_block("Headline - honest generalization (rev8, never seen)", m["headline"])
    if m["reference_spent"]:
        L += _render_block("Reference only - rev13 (SPENT on the axis decision, not blind)",
                           m["reference_spent"])
    L += ["## Per-rev", "",
          "| rev | n | S2 F1 / st | fused F1 / st | acting F1 / acc / cov |",
          "|---|---|---|---|---|"]
    for r in m["per_rev"]:
        s, ff, aa = r["s2"], r["fused"], r["acting"]
        note = " (spent)" if r["rev"] in SPENT_REVS else ""
        L.append(f"| {r['rev']}{note} | {r['n']} | {s['macro_f1']} / {s['stand_recall']} | "
                 f"{ff['macro_f1']} / {ff['stand_recall']} | "
                 f"{aa['macro_f1']} / {aa['accuracy']} / {aa['coverage']} |")
    return "\n".join(L) + "\n"


# what the open WOULD do, without touching the lockbox -- the refusal path prints this so the
# operator sees exactly what is gated behind the confirm token
def _dry_run_notice(out_dir: Path) -> None:
    spent = all(r in OPENED_REVS for r in SEALED_REVS)
    if spent:
        print(f"[lockbox] SPENT - {list(SEALED_REVS)} was already opened once; the final number is "
              "recorded in DOMAIN_NOTES Section 12.5 (the fused label did not beat S2 out-of-"
              "sample; the confidence signal generalized - deployable = the confidence-gated system).")
        print(f"[lockbox] the result artifact is {out_dir / RESULT_MD}. This harness is kept for a "
              "FUTURE sealed rev only; --confirm will NOT re-score a spent rev.")
        return
    print("[lockbox] SEALED - refusing to open without confirmation.")
    print(f"[lockbox] the honest final number is {list(SEALED_REVS)} - never seen.")
    print("[lockbox] this would: fit S2 on every non-lockbox rev, predict the sealed revs, "
          "compute S3 verdicts on them, fuse, and score the FUSED model once.")
    print("[lockbox] to open - only when you are calling the fused model FINAL - run:")
    print(f"[lockbox]   python -m stages.s4_fusion.lockbox --confirm {CONFIRM_TOKEN}")
    print(f"[lockbox] results would be written to {out_dir / RESULT_JSON}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Open the lockbox and score the FUSED model on the sealed revs (one-way).")
    ap.add_argument("--out", type=Path, default=S4_OUT_DIR)
    ap.add_argument("--confirm", default="",
                    help=f"pass {CONFIRM_TOKEN} to actually open the lockbox")
    args = ap.parse_args()

    if args.confirm != CONFIRM_TOKEN:
        _dry_run_notice(args.out)
        return

    # spent-guard: rev8/rev13 were already opened (number recorded, Section 12.5). re-scoring a
    # spent rev is not a valid unbiased test -- refuse even with --confirm. the harness stays for
    # a future NEW sealed rev; only a rev outside OPENED_REVS may actually be opened here.
    fresh = [r for r in SEALED_REVS if r not in OPENED_REVS]
    if not fresh:
        print(f"[lockbox] SPENT - {list(SEALED_REVS)} already opened once; the number is recorded "
              "in DOMAIN_NOTES Section 12.5 (fused did not beat S2 out-of-sample; the confidence "
              "signal generalized). Re-scoring is NOT a valid unbiased test - the model was "
              "finalized knowing it.")
        print("[lockbox] to evaluate a future model, seal a genuinely NEW rev (add it to "
              "dataset.DEFAULT_LOCKBOX_REVS and SEALED_REVS, outside OPENED_REVS), then open that.")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    m = evaluate_lockbox()
    (args.out / RESULT_JSON).write_text(json.dumps(m, indent=2), encoding="utf-8")
    (args.out / RESULT_MD).write_text(render(m), encoding="utf-8")

    h = m["headline"]
    f, s = h["fused"], h["s2_alone"]
    print(f"[lockbox] OPENED - headline rev8 {h['revs']}, {h['n_windows']:,} windows (spent once).")
    print(f"[lockbox] FUSED macro-F1 {f['macro_f1']} (S2 alone {s['macro_f1']}), "
          f"stand-recall {f['stand_recall']} (S2 {s['stand_recall']})")
    a = h["acting_on_confidence"]
    print(f"[lockbox] acting on HIGH+MED: coverage {a['coverage']} at accuracy {a['accuracy']}")
    if m["reference_spent"]:
        rf = m["reference_spent"]["fused"]
        print(f"[lockbox] rev13 (SPENT, reference only): fused macro-F1 {rf['macro_f1']}")
    print(f"[lockbox] artifacts -> {args.out}")


if __name__ == "__main__":
    main()
