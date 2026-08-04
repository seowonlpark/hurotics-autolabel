# row-level eval of the labelling path, abstentions included; python -m stages.s2_ml.roweval

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stages.s2_ml.dataset import HUMAN_UNKNOWN, LABEL_COL, STAND, TRAIN_CLASSES, WALK, load_dataset
from stages.s2_ml.features import WindowSpec, build_windows, feature_columns
from stages.s2_ml import label as label_mod
from stages.s2_ml.label import PHYSICS_FLOOR_THRESHOLD, explain, score_frame
from stages.s3_physics.serve import STANDING, WALKING
from stages.s2_ml.locoeval import DEFAULT_THRESHOLDS
from stages.s2_ml import transitions as transitions_mod
from stages.report import add_report_flag
from freshness import stamp_inputs
from runslayout import REGEN, keep_dir_for
from stages.s2_ml.train import (
    CHAMPION_SPEC_PATH, PRESETS, build_model, load_spec, reference_stats, select_features,
    trainable,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# champion fitted on every training rev except `exclude_rev`, with its own reference stats
def fit_on(train_df: pd.DataFrame, feats: list[str], exclude_rev: str | None):
    sub = train_df if exclude_rev is None else train_df[train_df["rev"] != exclude_rev]
    model = build_model()
    model.fit(sub[feats].to_numpy(float), sub["label"].to_numpy(int))
    return model, reference_stats(sub, feats)


# run the shipping path over trials and join ground truth back on
def score_trials(model, meta: dict, trials, threshold: float) -> pd.DataFrame:
    out = []
    for tr in trials:
        frame = tr.frame.reset_index(drop=True)
        if frame.empty:
            continue
        spec = WindowSpec(window_s=meta["window_s"],
                          stride_s=meta["inference_stride_s"], fs_hz=meta["fs_hz"])
        scored = explain(score_frame(model, meta, frame), meta, threshold, spec)
        scored = scored.sort_values(["segment", "Time"]).reset_index(drop=True)
        truth = (frame.sort_values(["segment", "Time"])
                      .reset_index(drop=True)[LABEL_COL].to_numpy())
        scored["truth"] = truth
        scored["rev"] = tr.rev
        scored["trial"] = tr.trial
        out.append(scored)
    return pd.concat(out, ignore_index=True)


# the rows `label.py` would commit to at `thr`- threshold, band AND physics gate
def _committed(v: pd.DataFrame, conf: np.ndarray, thr: float) -> np.ndarray:
    return _committed_under(v, conf, thr,
                            label_mod.PHYSICS_CEILING, label_mod.PHYSICS_FLOOR)


# The four settings of `label.PHYSICS_CEILING` / `label.PHYSICS_FLOOR`
PHYSICS_POLICIES = {
    "off": (False, False),
    "ceiling": (True, False),
    "floor": (False, True),
    "both": (True, True),
}

# fine grid for the matched-coverage comparison; `DEFAULT_THRESHOLDS` is too coarse to match one
_MATCH_GRID = np.round(np.arange(0.50, 1.0001, 0.0025), 4)


# (agrees, contradicts) per row: the swap rule against the model's own guess
def _physics_masks(v: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if "physics" not in v.columns:
        empty = np.zeros(len(v), dtype=bool)
        return empty, empty
    phys = v["physics"].to_numpy(object)
    p = v["p_walk"].to_numpy(float)
    guess = np.where(p >= 0.5, WALK, STAND)
    cls = np.where(phys == WALKING, float(WALK),
                   np.where(phys == STANDING, float(STAND), np.nan))
    decisive = np.isfinite(cls)
    return decisive & (cls == guess), decisive & (cls != guess)


# `_committed`, but under an arbitrary physics policy rather than the shipped one
def _committed_under(v: pd.DataFrame, conf: np.ndarray, thr: float,
                     ceiling: bool, floor: bool) -> np.ndarray:
    band = (v["in_band"].to_numpy(bool) if "in_band" in v.columns
            else np.zeros(len(v), dtype=bool))
    agrees, contradicts = _physics_masks(v)
    eff = np.full(len(v), float(thr))
    if floor:
        eff[agrees] = PHYSICS_FLOOR_THRESHOLD
    if ceiling:
        eff[contradicts] = np.inf
    return (conf >= eff) & ~band


# the worst-scoring subject and its accuracy. The NAME is the actionable half: a bare minimum
# says a subject is bad, not which one to go and look at, and it is unrecoverable afterwards
def _worst_rev(correct: np.ndarray, revs: np.ndarray,
               keep: np.ndarray) -> tuple[float, str | None]:
    per = [(float(correct[keep & (revs == r)].mean()), str(r)) for r in pd.unique(revs)
           if (keep & (revs == r)).any()]
    # ties break on the rev name, so the column does not wander between runs
    return min(per) if per else (float("nan"), None)


# does the physics floor/ceiling beat simply moving the threshold?
def physics_gate_table(df: pd.DataFrame, threshold: float) -> dict:
    v = df[df["truth"].isin(TRAIN_CLASSES) & df["p_walk"].notna()]
    p = v["p_walk"].to_numpy(float)
    conf = np.maximum(p, 1 - p)
    correct = (np.where(p >= 0.5, WALK, STAND) == v["truth"].to_numpy())
    revs = v["rev"].to_numpy()
    agrees, contradicts = _physics_masks(v)

    def point(k: np.ndarray) -> dict:
        worst_acc, worst_rev = _worst_rev(correct, revs, k)
        return {"coverage": float(k.mean()), "n_committed": int(k.sum()),
                "selective_accuracy": float(correct[k].mean()) if k.any() else float("nan"),
                "worst_rev_accuracy": worst_acc,
                "worst_rev": worst_rev,
                "errors_kept": int((~correct[k]).sum())}

    # The OFF arm swept finely, so any gated coverage can be matched against it
    off_sweep = [{"threshold": float(t),
                  **point(_committed_under(v, conf, t, False, False))}
                 for t in _MATCH_GRID]

    # **Swept across thresholds, not one.** The physics objection is explicitly threshold-conditional
    by_threshold = []
    for thr in DEFAULT_THRESHOLDS:
        base_k = _committed_under(v, conf, thr, False, False)
        ce = int((~correct & base_k).sum())
        arms = []
        for name, (ceiling, floor) in PHYSICS_POLICIES.items():
            pt = point(_committed_under(v, conf, thr, ceiling, floor))
            entry = {"policy": name, "ceiling": ceiling, "floor": floor, **pt}
            if name != "off":
                # nearest OFF point by coverage; interpolating `errors_kept` would invent a count
                m = min(off_sweep, key=lambda r: abs(r["coverage"] - pt["coverage"]))
                entry["matched_off"] = m
                entry["errors_vs_matched_off"] = pt["errors_kept"] - m["errors_kept"]
                entry["coverage_gap"] = round(pt["coverage"] - m["coverage"], 6)
            arms.append(entry)
        by_threshold.append({
            "threshold": float(thr),
            # the ceiling's addressable set here: of the errors committed to, how many does it object to
            "committed_errors": ce,
            "committed_errors_contradicted": int((~correct & contradicts & base_k).sum()),
            "policies": arms,
        })

    shipped = min(by_threshold, key=lambda r: abs(r["threshold"] - threshold))
    return {
        "threshold": float(threshold),
        "floor_threshold": float(PHYSICS_FLOOR_THRESHOLD),
        "n_rows_scored": int(len(v)),
        "n_physics_decisive": int((agrees | contradicts).sum()),
        "n_physics_agrees": int(agrees.sum()),
        "n_physics_contradicts": int(contradicts.sum()),
        "committed_errors": shipped["committed_errors"],
        "committed_errors_contradicted": shipped["committed_errors_contradicted"],
        "policies": shipped["policies"],
        "by_threshold": by_threshold,
    }


# row-level coverage vs accuracy; only rows with a trainable ground truth count
def curve(df: pd.DataFrame) -> list[dict]:
    v = df[df["truth"].isin(TRAIN_CLASSES) & df["p_walk"].notna()]
    p = v["p_walk"].to_numpy(float)
    conf = np.maximum(p, 1 - p)
    correct = (np.where(p >= 0.5, WALK, STAND) == v["truth"].to_numpy())
    revs = v["rev"].to_numpy()
    rows = []
    for thr in DEFAULT_THRESHOLDS:
        k = _committed(v, conf, thr)
        worst_acc, worst_rev = _worst_rev(correct, revs, k)
        rows.append({
            "threshold": float(thr),
            "coverage": float(k.mean()),
            "n_labeled": int(k.sum()),
            "selective_accuracy": float(correct[k].mean()) if k.any() else float("nan"),
            "worst_rev_accuracy": worst_acc,
            "worst_rev": worst_rev,
            "errors_kept": int((~correct[k]).sum()),
        })
    return rows


# (coverage, accuracy) per held-out subject at the shipped threshold
def per_rev(df: pd.DataFrame, threshold: float) -> list[dict]:
    v = df[df["truth"].isin(TRAIN_CLASSES) & df["p_walk"].notna()]
    p = v["p_walk"].to_numpy(float)
    conf = np.maximum(p, 1 - p)
    correct = np.where(p >= 0.5, WALK, STAND) == v["truth"].to_numpy()
    k = _committed(v, conf, threshold)
    revs = v["rev"].to_numpy()

    truth = v["truth"].to_numpy()
    rows = []
    for r in sorted(pd.unique(revs)):
        m = revs == r
        km = k & m
        # per-class recall on committed rows: rev8's stand recall 0.5752 hides inside 0.9308 accuracy
        per_class = {}
        for cls, name in ((STAND, "stand"), (WALK, "walk")):
            in_cls = km & (truth == cls)
            per_class[f"{name}_committed"] = int(in_cls.sum())
            per_class[f"{name}_recall"] = (float(correct[in_cls].mean())
                                           if in_cls.any() else float("nan"))
        rows.append({
            "rev": str(r),
            "rows_scored": int(m.sum()),
            "rows_committed": int(km.sum()),
            "coverage": float(km.sum() / m.sum()) if m.any() else float("nan"),
            "accuracy": float(correct[km].mean()) if km.any() else float("nan"),
            "errors_kept": int((~correct[km]).sum()),
            **per_class,
        })
    return rows


# which way the committed errors go- the one cut pooled accuracy hides completely
def confusion(df: pd.DataFrame, threshold: float) -> dict:
    v = df[df["truth"].isin(TRAIN_CLASSES) & df["p_walk"].notna()]
    p = v["p_walk"].to_numpy(float)
    conf = np.maximum(p, 1 - p)
    guess = np.where(p >= 0.5, WALK, STAND)
    truth = v["truth"].to_numpy()
    k = _committed(v, conf, threshold)

    names = {STAND: "stand", WALK: "walk"}
    return {
        "threshold": float(threshold),
        "rows_committed": int(k.sum()),
        "rows_abstained": int((~k).sum()),
        "cells": [{"truth": names[t], "pred": names[g],
                   "rows": int(((truth == t) & (guess == g) & k).sum())}
                  for t in (STAND, WALK) for g in (STAND, WALK)],
    }


# is each ambiguity reason earning its place?
def reason_report(df: pd.DataFrame) -> pd.DataFrame:
    v = df[df["truth"].isin(TRAIN_CLASSES) & df["p_walk"].notna()].copy()
    v["would_be_correct"] = v["label"] == v["truth"]
    rows = []
    conf_rows = v[~v["ambiguous"]]
    rows.append({"reason": "(confident, kept)", "rows": len(conf_rows),
                 "accuracy_of_guess": conf_rows["would_be_correct"].mean()})
    for r, g in v[v["ambiguous"]].groupby("reason"):
        rows.append({"reason": r, "rows": len(g),
                     "accuracy_of_guess": g["would_be_correct"].mean()})
    return pd.DataFrame(rows).sort_values("rows", ascending=False)


# does the model abstain where a trained human also could not call it?
def unknown_agreement(df: pd.DataFrame) -> dict:
    v = df[df["p_walk"].notna()]
    unk = v["truth"] == HUMAN_UNKNOWN
    if not unk.any():
        return {}
    return {
        "rows_human_unknown": int(unk.sum()),
        "abstain_rate_on_human_unknown": float(v.loc[unk, "ambiguous"].mean()),
        "abstain_rate_elsewhere": float(v.loc[~unk, "ambiguous"].mean()),
    }


# the worst-subject cell, named where the artifact carries a name; older artifacts do not
def _worst_cell(r: dict) -> str:
    acc = f"{r['worst_rev_accuracy']:.4f}"
    return f"{acc} (`{r['worst_rev']}`)" if r.get("worst_rev") else acc


# the floor/ceiling section
def render_physics_gate(g: dict) -> list[str]:
    if not g:
        return []
    ce, cec = g["committed_errors"], g["committed_errors_contradicted"]
    share = f"{cec / ce:.1%}" if ce else "n/a"
    lines = [
        "", f"## Physics floor and ceiling, at threshold {g['threshold']:.2f}", "",
        "`label.PHYSICS_CEILING` abstains where the swap rule is decisive and contradicts "
        "the model; `label.PHYSICS_FLOOR` accepts a lower confidence "
        f"({g['floor_threshold']:.2f}) where it is decisive and agrees. **Both ship OFF.** "
        "This table is what a decision to change that has to argue against.", "",
        f"Of the **{ce:,}** errors the shipped policy commits to, the physics contradicts "
        f"**{cec:,}** ({share}). That is the ceiling's entire addressable set — no tuning "
        f"reaches an error the swap rule does not object to. The two read the same "
        f"1 Hz-filtered interleg angle, so they are wrong together (`caveats.md` §1.1c).", "",
        f"Physics is decisive on {g['n_physics_decisive']:,} of {g['n_rows_scored']:,} "
        f"scored rows ({g['n_physics_agrees']:,} agree, "
        f"{g['n_physics_contradicts']:,} contradict).", "",
        "| policy | coverage | selective acc | worst subject | errors kept | "
        "vs threshold-only at matched coverage |", "|---|---|---|---|---|---|",
    ]
    for r in g["policies"]:
        if r["policy"] == "off":
            verdict = "*(the baseline)*"
        else:
            d = r["errors_vs_matched_off"]
            m = r["matched_off"]
            verdict = (f"**{d:+,} errors** at coverage {m['coverage']:.4f} "
                       f"(thr {m['threshold']:.4f})")
            verdict += " — **worse**" if d > 0 else (" — better" if d < 0 else " — tied")
        lines.append(f"| `{r['policy']}` | {r['coverage']:.4f} | "
                     f"{r['selective_accuracy']:.4f} | {_worst_cell(r)} | "
                     f"{r['errors_kept']:,} | {verdict} |")
    lines += [
        "", "**How to read the last column.** A policy that changes coverage cannot be "
        "judged on accuracy at a fixed threshold — giving up the rows you are least sure "
        "of always raises accuracy. Each gated arm is therefore matched against the "
        "threshold-only policy at the same coverage, and compared on surviving errors. "
        "Positive means the gate spent coverage worse than the threshold would have; "
        "negative means it spent it better.", "",
    ]

    if g.get("by_threshold"):
        lines += [
            "### Across the threshold, because the objection is threshold-conditional", "",
            "The retraction in `anchors.py` is specific: physics contradicts ~12% of S2's "
            "high-confidence errors and **none at p >= 0.95**. A gate measured at one "
            "operating point cannot speak to that, and the addressable-set column below is "
            "the direct check — it is the ceiling's ceiling.", "",
            "| threshold | committed errors | physics objects to | ceiling vs thr-only | "
            "floor vs thr-only |", "|---|---|---|---|---|",
        ]
        for r in g["by_threshold"]:
            arms = {a["policy"]: a for a in r["policies"]}
            ce, cec = r["committed_errors"], r["committed_errors_contradicted"]
            share = f"{cec:,} ({cec / ce:.1%})" if ce else "—"
            def _d(name: str) -> str:
                a = arms.get(name)
                if not a or "errors_vs_matched_off" not in a:
                    return "—"
                return f"{a['errors_vs_matched_off']:+,}"
            lines.append(f"| {r['threshold']:.2f} | {ce:,} | {share} | "
                         f"{_d('ceiling')} | {_d('floor')} |")
        lines += ["", "Negative is better (fewer surviving errors at the same coverage).",
                  ""]
    return lines


def render(tag: str, c: list[dict], reasons: pd.DataFrame, unk: dict, revs: list[str],
           by_rev: list[dict], threshold: float, conf: dict,
           gate: dict | None = None) -> str:
    lines = [f"# Row-level evaluation - {tag}", "",
             f"subjects scored: {', '.join(revs)}", "",
             "| threshold | coverage | selective acc | worst subject | errors kept |",
             "|---|---|---|---|---|"]
    for r in c:
        lines.append(f"| {r['threshold']:.2f} | {r['coverage']:.4f} | "
                     f"{r['selective_accuracy']:.4f} | {_worst_cell(r)} | "
                     f"{r['errors_kept']:,} |")
    if by_rev:
        lines += ["", f"## Per subject, at the shipped threshold {threshold:.2f}", "",
                  "Each subject was scored by a model that never saw it. Read coverage and "
                  "accuracy together per row: a subject with high accuracy over a small "
                  "committed set is not one the pipeline handles well, it is one it mostly "
                  "declined to answer for.", "",
                  "The two recalls carry their own denominators because the classes are "
                  "wildly unbalanced on the committed set — a subject can hold a high "
                  "accuracy while missing most of its standing, and pooled accuracy is the "
                  "column that hides it.", "",
                  "| rev | rows scored | committed | coverage | errors kept | accuracy | "
                  "stand recall | walk recall |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in by_rev:
            acc = f"{r['accuracy']:.4f}" if r["rows_committed"] else "—"

            def _rec(name: str, row: dict = r) -> str:
                n = row.get(f"{name}_committed", 0)
                v = row.get(f"{name}_recall")
                return f"{v:.4f} (n={n:,})" if n and v == v else "—"

            lines.append(f"| `{r['rev']}` | {r['rows_scored']:,} | {r['rows_committed']:,} "
                         f"| {r['coverage']:.1%} | {r['errors_kept']:,} | {acc} "
                         f"| {_rec('stand')} | {_rec('walk')} |")
    if conf:
        cells = [x for x in conf["cells"] if x["rows"]]
        errs = [x for x in cells if x["truth"] != x["pred"]]
        lines += ["", f"## Direction of error, at the shipped threshold {threshold:.2f}", "",
                  f"{conf['rows_committed']:,} rows committed, "
                  f"{conf['rows_abstained']:,} abstained. **Rows, not windows** — the unit a "
                  f"caller receives, and the same unit as the coverage table above.", "",
                  "| truth → guess | rows |", "|---|---|"]
        for x in sorted(cells, key=lambda x: -x["rows"]):
            mark = "" if x["truth"] == x["pred"] else " ⚠"
            lines.append(f"| {x['truth']} → {x['pred']}{mark} | {x['rows']:,} |")
        if errs:
            top = max(errs, key=lambda x: x["rows"])
            n_err = sum(x["rows"] for x in errs)
            lines += ["", f"**{top['rows']:,} of {n_err:,} committed errors "
                          f"({top['rows'] / n_err:.0%}) are {top['truth']} called "
                          f"{top['pred']}.** The residual failure is one-directional, so "
                          f"watch {top['truth']} recall, not accuracy."]
    lines += ["", "## Ambiguity reasons", "",
              "`accuracy of guess` is how often the guess WOULD have been right had the row "
              "not been flagged. A reason earns its place by sitting well below the "
              "confident row.", "",
              "| reason | rows | accuracy of guess |", "|---|---|---|"]
    for _, r in reasons.iterrows():
        lines.append(f"| `{r['reason']}` | {r['rows']:,} | {r['accuracy_of_guess']:.4f} |")
    if unk:
        lines += ["", "## Agreement with human `-1` (§5.2)", "",
                  f"- rows a human marked unknown: {unk['rows_human_unknown']:,}",
                  f"- model abstains on those: **{unk['abstain_rate_on_human_unknown']:.1%}**",
                  f"- model abstains elsewhere: {unk['abstain_rate_elsewhere']:.1%}", "",
                  "The model never sees `-1` in training, so this is independent evidence."]
    lines += render_physics_gate(gate or {})
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REGEN / "s2_ml"))
    add_report_flag(ap)
    ap.add_argument("--lockbox", action="store_true",
                    help="SINGLE USE: fit on all training revs, score the sealed revs")
    args = ap.parse_args()

    # NOT a flag. `curve` below sweeps every threshold and the sweep is in the `.json`, so an
    # override would only re-pick which row of it the per-rev and confusion tables are cut at --
    # and this stage is THE accuracy claim, which the repo quotes at the declared operating point.
    # Read another point off the curve; do not re-run the stage to move the headline.
    threshold = PRESETS["balanced"]

    out_dir = (REPO_ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    trials = load_dataset()
    spec = WindowSpec()
    windows = build_windows([t for t in trials if t.split == "train"], spec)
    # the CHAMPION's feature set, not all 42 columns `build_windows` emits (wrong until 2026-08-04)
    feats = select_features(feature_columns(windows), load_spec()["drop_features"])
    train_df = trainable(windows, "train")

    base_meta = {
        "window_s": spec.window_s, "fs_hz": spec.fs_hz, "inference_stride_s": 0.25,
        "features": feats,
    }

    parts = []
    if args.lockbox:
        model, ref = fit_on(train_df, feats, None)
        meta = {**base_meta, "reference_stats": ref}
        held = [t for t in trials if t.split == "lockbox"]
        parts.append(score_trials(model, meta, held, threshold))
        tag = "LOCKBOX (single use)"
    else:
        for rev in sorted(train_df["rev"].unique()):
            model, ref = fit_on(train_df, feats, rev)
            meta = {**base_meta, "reference_stats": ref}
            held = [t for t in trials if t.split == "train" and t.rev == rev]
            print(f"[rowe] scoring held-out {rev} ({len(held)} trials)")
            parts.append(score_trials(model, meta, held, threshold))
        tag = "leave-one-rev-out"

    df = pd.concat(parts, ignore_index=True)
    c = curve(df)
    reasons = reason_report(df)
    unk = unknown_agreement(df)
    revs = sorted(df["rev"].unique())
    by_rev = per_rev(df, threshold)
    conf = confusion(df, threshold)

    print(f"\n[rowe] {tag}: {len(df):,} rows over {revs}")
    for r in c:
        if r["threshold"] in tuple(PRESETS.values()):
            print(f"[rowe]   thr {r['threshold']:.2f}: coverage {r['coverage']:.4f}  "
                  f"selective_acc {r['selective_accuracy']:.4f}  "
                  f"worst_subject {r['worst_rev_accuracy']:.4f} ({r.get('worst_rev') or '?'})")
    print(f"[rowe] per subject at threshold {threshold:.2f}:")
    for r in by_rev:
        print(f"[rowe]   {r['rev']:<6} scored={r['rows_scored']:>7,}  "
              f"committed={r['rows_committed']:>7,}  coverage {r['coverage']:>6.1%}  "
              f"errors {r['errors_kept']:>6,}  accuracy {r['accuracy']:.4f}  "
              f"stand_recall {r['stand_recall']:.4f}")
    errs = [x for x in conf["cells"] if x["truth"] != x["pred"] and x["rows"]]
    if errs:
        top = max(errs, key=lambda x: x["rows"])
        n_err = sum(x["rows"] for x in errs)
        print(f"[rowe] committed errors: {n_err:,} of {conf['rows_committed']:,} committed; "
              f"{top['rows']:,} ({top['rows'] / n_err:.0%}) are {top['truth']} "
              f"called {top['pred']}")
    print()
    print(reasons.to_string(index=False))
    if unk:
        print(f"\n[rowe] abstains on {unk['abstain_rate_on_human_unknown']:.1%} of human `-1` "
              f"rows vs {unk['abstain_rate_elsewhere']:.1%} elsewhere")

    gate = physics_gate_table(df, threshold)
    ce, cec = gate["committed_errors"], gate["committed_errors_contradicted"]
    print(f"\n[rowe] physics gate: of {ce:,} committed errors, physics contradicts "
          f"{cec:,} ({cec / ce:.1%})" if ce else "\n[rowe] physics gate: no committed errors")
    for r in gate["policies"]:
        if r["policy"] == "off":
            print(f"[rowe]   {r['policy']:<8} coverage {r['coverage']:.4f}  "
                  f"errors {r['errors_kept']:,}  (baseline)")
        else:
            print(f"[rowe]   {r['policy']:<8} coverage {r['coverage']:.4f}  "
                  f"errors {r['errors_kept']:,}  vs threshold-only at matched coverage: "
                  f"{r['errors_vs_matched_off']:+,}")

    # what `near_transition` is made of, computed on THIS pass so no refit can drift between tables
    print()
    t_spec = WindowSpec(window_s=base_meta["window_s"],
                        stride_s=base_meta["inference_stride_s"],
                        fs_hz=base_meta["fs_hz"])
    # The lockbox pass reads a split that is spent once (dataset.py), so its two reports are the
    # only ones that exist and no re-run legitimately replaces them: they go to the keep-half.
    # The LORO pass refits from the spec on demand, so it stays with the rest of the stage output.
    dest = keep_dir_for(out_dir) if args.lockbox else out_dir
    dest.mkdir(parents=True, exist_ok=True)

    t_stem = "transitions_lockbox" if args.lockbox else "transitions_loro"
    t_sum = transitions_mod.run(df, t_spec, dest, t_stem, tag, report=args.report)
    transitions_mod.print_summary(t_sum)

    stem = "roweval_lockbox" if args.lockbox else "roweval_loro"
    if args.report:
        (dest / f"{stem}.md").write_text(
            render(tag, c, reasons, unk, revs, by_rev, threshold, conf, gate),
            encoding="utf-8")
    (dest / f"{stem}.json").write_text(json.dumps(
        {"tag": tag, "threshold": threshold, "revs": revs,
         # recorded so a stale artifact is identifiable: which model produced this curve
         "n_features": len(feats), "features": feats,
         "curve": c,
         "per_rev": by_rev, "confusion": conf, "reasons": reasons.to_dict("records"),
         "human_unknown": unk, "physics_gate": gate}, indent=2),
        encoding="utf-8")
    # This is the accuracy claim the repo quotes, and it is re-fitted from the spec rather than
    # loaded, so a promotion between this run and a reader silently changes what it describes.
    # The stamp follows the report into `dest`: it is the record of what THAT file describes, and
    # for the lockbox it is the only warning a reader gets, since the read cannot be redone.
    stamp_inputs(dest, {"champion_spec": CHAMPION_SPEC_PATH}, stage=stem)

    ext = ".md" if args.report else ".json"
    print(f"\n[rowe] -> {dest / (stem + ext)}")
    print(f"[rowe] -> {dest / (t_stem + ext)}")


if __name__ == "__main__":
    main()
