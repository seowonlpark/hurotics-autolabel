"""Data audit: which whole files should not be trusted, and why.

    python -m stages.s2_ml.audit --out runs/s2_ml

Two questions, deliberately separate:

  1. **Is this labelled trial self-consistent?** Scores the human labels against the swap
     rule (DOMAIN_NOTES §10) - walking is the legs alternating, 0 swaps standing, >=2
     walking, zero fitted parameters. A trial where most windows contradict their own label
     is a labelling failure, not a hard example.
  2. **Is this recording usable at all?** Rest reference, channel liveness, coverage. Runs
     on unlabelled CSVs too, because the labelling path needs the same warning.

**The audit is model-free on purpose.** It would have been easy to flag files the
classifier disagrees with, and that is exactly how a corpus gets quietly fitted to its
model: the hard cases get deleted, every metric improves, and nothing has been learned.
The swap rule is an independent physical opinion with no trained parameters, so when it
contradicts a label the disagreement is evidence about the label. If this module ever
starts importing the classifier, that property is gone.

**It flags; it does not delete.** Confirmed exclusions live in `dataset.EXCLUDED_TRIALS`
as explicit, commented entries. Auto-dropping whatever trips a threshold would make the
corpus a moving target that silently shrinks whenever the threshold moves - the "no silent
mutation" rule in the README, applied to data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from stages.s2_ml.dataset import (
    FEATURES,
    LABEL_COL,
    STAND,
    WALK,
    Trial,
    load_dataset,
)
from stages.s2_ml.features import WindowSpec, rest_reference, swap_counts
from stages.s2_ml.rest import ANGLE_CHANNELS

REPO_ROOT = Path(__file__).resolve().parents[2]

# A trial is incoherent when MOST of its label-pure windows contradict the physics. Not a
# tuned number: 0.5 is "the file disagrees with itself more often than it agrees", the only
# non-arbitrary line available. The corpus sits at a median of 0.03 and a 95th percentile
# of 0.19, so nothing marginal is anywhere near it.
MAX_DISAGREE = 0.50

# Below this, a trial is too short for the fraction to mean anything.
MIN_WINDOWS = 20

# The swap rule's own AMBIGUOUS verdict: one leg passing the other happens both when you
# take a step and when you shift your weight, so exactly 1 swap votes for neither class.
AMBIGUOUS = -9


def physics_verdict(seg: pd.DataFrame, spec: WindowSpec, ileg_zero: float
                    ) -> tuple[np.ndarray, np.ndarray]:
    """(verdict, window start) per non-overlapping window, by the swap rule alone."""
    d = (seg[ANGLE_CHANNELS[0]].to_numpy(float)
         - seg[ANGLE_CHANNELS[1]].to_numpy(float) - ileg_zero)
    starts = np.arange(0, len(seg) - spec.n + 1, spec.n)
    sw = swap_counts(d, starts, spec.n)
    return np.where(sw == 0, STAND, np.where(sw >= 2, WALK, AMBIGUOUS)), starts


def label_coherence(trial: Trial, spec: WindowSpec | None = None) -> dict:
    """How often this trial's own labels contradict the physics."""
    spec = spec or WindowSpec()
    frame = trial.frame.reset_index(drop=True)
    if frame.empty:
        return {"rev": trial.rev, "trial": trial.trial, "windows": 0,
                "disagree": 0, "disagree_frac": float("nan"), "ambiguous_frac": float("nan"),
                "rest_trusted": False, "flag": False}

    _zeros, ileg_zero, trusted = rest_reference(frame, spec.fs_hz)
    agree = dis = amb = 0
    for _sid, seg in frame.groupby("segment", sort=True):
        seg = seg.reset_index(drop=True)
        if len(seg) < spec.n:
            continue
        verdict, starts = physics_verdict(seg, spec, ileg_zero)
        L = sliding_window_view(seg[LABEL_COL].to_numpy(), spec.n)[::spec.n]
        n_stand = (L == STAND).sum(1)
        n_walk = (L == WALK).sum(1)
        valid = n_stand + n_walk
        # Only label-PURE windows can testify: a window straddling a transition disagrees
        # with any single verdict by construction, which would read as incoherence.
        pure = (valid > 0) & ((n_stand == valid) | (n_walk == valid))
        lab = np.where(n_walk > n_stand, WALK, STAND)
        decided = pure & (verdict != AMBIGUOUS)
        agree += int((verdict[decided] == lab[decided]).sum())
        dis += int((verdict[decided] != lab[decided]).sum())
        amb += int((pure & (verdict == AMBIGUOUS)).sum())

    n = agree + dis
    frac = dis / n if n else float("nan")
    return {
        "rev": trial.rev, "trial": trial.trial, "windows": n,
        "disagree": dis, "disagree_frac": frac,
        "ambiguous_frac": amb / (n + amb) if (n + amb) else float("nan"),
        "rest_trusted": bool(trusted),
        "flag": bool(n >= MIN_WINDOWS and frac > MAX_DISAGREE),
    }


def file_health(frame: pd.DataFrame, spec: WindowSpec | None = None) -> dict:
    """Is this recording usable at all? Label-free, so it runs on any CSV.

    Every check answers a way a file can be broken without being empty: no measurable rest
    posture (so every interleg feature is anchored on a guess), a dead channel, or too
    little contiguous signal to fill a single window.
    """
    spec = spec or WindowSpec()
    if frame.empty:
        return {"rows": 0, "usable": False, "warnings": ["frame is empty"]}

    warnings: list[str] = []
    _zeros, _ileg, trusted = rest_reference(frame, spec.fs_hz)
    if not trusted:
        warnings.append("no rest span found: interleg zero is a fallback median, so every "
                        "interleg feature is weaker evidence on this file")

    for c in FEATURES:
        v = frame[c].to_numpy(float)
        if not np.isfinite(v).all():
            warnings.append(f"{c}: contains non-finite samples")
        if np.nanstd(v) < 1e-6:
            warnings.append(f"{c}: constant - channel is dead or unmapped")

    seg_len = frame.groupby("segment").size()
    scorable = int((seg_len >= spec.n).sum())
    if scorable == 0:
        warnings.append(f"no segment reaches one {spec.window_s}s window: nothing is scorable")
    short = int((seg_len < spec.n).sum())
    if short:
        warnings.append(f"{short} of {len(seg_len)} segments are shorter than one window")

    return {
        "rows": int(len(frame)),
        "segments": int(len(seg_len)),
        "scorable_segments": scorable,
        "rest_trusted": bool(trusted),
        "usable": scorable > 0,
        "warnings": warnings,
    }


def render(df: pd.DataFrame, flagged: pd.DataFrame) -> str:
    lines = ["# Label audit", "",
             "Human labels scored against the swap rule (DOMAIN_NOTES §10), which has no "
             "trained parameters. `disagree` is the share of label-pure windows whose "
             "physics verdict contradicts the annotation.", "",
             f"Flag rule: `disagree > {MAX_DISAGREE:.2f}` over at least {MIN_WINDOWS} "
             "windows - the file contradicts itself more often than it agrees.", ""]
    if len(flagged):
        lines += ["## Flagged", "",
                  "| rev | trial | windows | disagree | fraction |", "|---|---|---|---|---|"]
        for _, r in flagged.iterrows():
            lines.append(f"| {r['rev']} | {r['trial']} | {r['windows']:,} | "
                         f"{r['disagree']:,} | **{r['disagree_frac']:.4f}** |")
        lines += ["", "Add confirmed cases to `dataset.EXCLUDED_TRIALS` with the evidence. "
                  "This module never edits the corpus itself.", ""]
    else:
        lines += ["## Flagged", "", "None.", ""]

    lines += ["## All trials, worst first", "",
              "| rev | trial | windows | disagree fraction | swap-ambiguous | rest |",
              "|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| {r['rev']} | {r['trial']} | {r['windows']:,} | "
                     f"{r['disagree_frac']:.4f} | {r['ambiguous_frac']:.4f} | "
                     f"{'yes' if r['rest_trusted'] else 'NO'} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/s2_ml")
    ap.add_argument("--include-lockbox", action="store_true",
                    help="also audit sealed revs; label coherence is model-free, but this "
                         "still reads their labels")
    args = ap.parse_args()

    out_dir = (REPO_ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # The audit loads the RAW corpus: a trial already quarantined must still appear, or the
    # report silently stops justifying the exclusion it caused.
    kw = {"lockbox_revs": ()} if args.include_lockbox else {}
    trials = load_dataset(excluded=set(), **kw)
    rows = [label_coherence(t) for t in trials]
    df = pd.DataFrame([r for r in rows if r["windows"]]).sort_values(
        "disagree_frac", ascending=False).reset_index(drop=True)
    flagged = df[df["flag"]]

    print(f"[audit] {len(df)} trials scored against the swap rule")
    print(df.head(10).round(4).to_string(index=False))
    if len(flagged):
        print(f"\n[audit] FLAGGED {len(flagged)} trial(s) over {MAX_DISAGREE:.2f}:")
        for _, r in flagged.iterrows():
            print(f"[audit]   {r['rev']} trial {r['trial']}: "
                  f"{r['disagree']:,}/{r['windows']:,} windows contradict the label "
                  f"({r['disagree_frac']:.1%})")
        print("[audit] add confirmed cases to dataset.EXCLUDED_TRIALS; nothing was deleted")
    else:
        print(f"\n[audit] no trial exceeds {MAX_DISAGREE:.2f}")

    no_rest = df[~df["rest_trusted"]]
    if len(no_rest):
        print(f"\n[audit] {len(no_rest)} trial(s) never rest, so their interleg zero is a "
              f"fallback:")
        for _, r in no_rest.iterrows():
            print(f"[audit]   {r['rev']} trial {r['trial']}")

    (out_dir / "label_audit.md").write_text(render(df, flagged), encoding="utf-8")
    (out_dir / "label_audit.json").write_text(
        json.dumps({"max_disagree": MAX_DISAGREE, "min_windows": MIN_WINDOWS,
                    "trials": df.to_dict("records")}, indent=2), encoding="utf-8")
    print(f"\n[audit] -> {out_dir / 'label_audit.md'}")


if __name__ == "__main__":
    main()
