# audit the human LABELS against the physics verdict
#   python -m stages.s3_physics.label_audit
# the rule has no trained parameter and never sees a label, so when it contradicts one
# that is evidence about the label; it nominates, it does not exclude

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import binomtest

from stages.s2_ml.dataset import LABEL_COL, STAND, WALK, Trial, load_dataset
from stages.s2_ml.features import WindowSpec, amplitude_band
from stages.s3_physics.anchors import AMBIGUOUS, STANDING, WALKING, trial_anchors

REPO_ROOT = Path(__file__).resolve().parents[2]

# A trial is incoherent when MOST of its label-pure windows contradict the physics; not
# tuned: 0.5 is "the file disagrees with itself more often than it agrees", the only
# non-arbitrary line available; the corpus sits at a median near 0.03, so nothing marginal
# is anywhere near it
MAX_DISAGREE = 0.50

# Below this a trial is too short for the fraction to mean anything
MIN_WINDOWS = 20

# Family-wise error rate for the band-policy test; the conventional 0.05, Bonferroni'd
# across the trials tested- with ~17 testable trials, an uncorrected 0.05 would flag one
# by chance nearly every run, and a detector that cries wolf every run gets ignored
BAND_ALPHA = 0.05

VERDICT_CLASS = {STANDING: STAND, WALKING: WALK}

# What `coherence` carries out per label-pure window; wider than `band_policy` needs,
# because `nominate` reads the same frame- one notion of "the windows this trial was
# scored on", not two
WINDOW_COLUMNS = ["rev", "trial", "split", "segment", "t_start_ms", "label",
                  "ileg_minhalf", "swap_verdict_adaptive", "swap_count_adaptive",
                  "swap_window_s", "physics_class", "contradicts"]

# a flagged trial nominates at most this many windows, no two closer than the gap
# ergonomic, not statistical: a person needs a handful of clear cases SPREAD out, which
# is what separates a swapped channel from a misaligned label track
# the dropped count is always reported- a cap hiding its truncation reads as "all of them"
MAX_NOMINATIONS_PER_TRIAL = 12
MIN_NOMINATION_GAP_S = 5.0


# human label per window, on the SAME grid `trial_anchors` walks
def window_labels(trial: Trial, spec: WindowSpec) -> pd.DataFrame:
    frame = trial.frame.reset_index(drop=True)
    rows = []
    for seg_id, seg in frame.groupby("segment", sort=True):
        seg = seg.reset_index(drop=True)
        if len(seg) < spec.n:
            continue
        L = sliding_window_view(seg[LABEL_COL].to_numpy(), spec.n)[::spec.step]
        n_stand = (L == STAND).sum(1)
        n_walk = (L == WALK).sum(1)
        valid = n_stand + n_walk
        pure = (valid > 0) & ((n_stand == valid) | (n_walk == valid))
        starts = np.arange(0, len(seg) - spec.n + 1, spec.step)
        rows.append(pd.DataFrame({
            "rev": trial.rev, "trial": trial.trial, "segment": int(seg_id),
            "t_start_ms": seg["Time"].to_numpy(float)[starts],
            "label": np.where(n_walk > n_stand, WALK, STAND), "pure": pure,
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# how often this trial's labels contradict the physics verdict
def coherence(trial: Trial, spec: WindowSpec | None = None
              ) -> tuple[dict, pd.DataFrame]:
    spec = spec or WindowSpec()
    anchors = trial_anchors(trial, spec)
    labels = window_labels(trial, spec)
    empty = pd.DataFrame(columns=WINDOW_COLUMNS)
    if anchors.empty or labels.empty:
        return ({"rev": trial.rev, "trial": trial.trial, "windows": 0, "disagree": 0,
                 "disagree_frac": float("nan"), "abstain_frac": float("nan"),
                 "rest_trusted": False, "flag": False,
                 "t_first_s": float("nan"), "t_last_s": float("nan"),
                 "contradict_t_first_s": float("nan"),
                 "contradict_t_last_s": float("nan")}, empty)

    j = anchors.merge(labels, on=["rev", "trial", "segment", "t_start_ms"],
                      validate="one_to_one")
    pure = j[j["pure"]].copy()
    # Per-window contradiction, carried out of here rather than recomputed by the caller
    # `disagree_frac` below is the mean of this column, so the trial-level number a reader
    # sees in the report and the windows `nominate` hands to `inspect_window` cannot
    # disagree about which windows they are- the failure mode of the deleted S4 review
    # queue, which built its own notion of "offending window" beside this one
    pure["physics_class"] = pure["swap_verdict_adaptive"].map(VERDICT_CLASS)
    pure["contradicts"] = pure["physics_class"].notna() & (pure["physics_class"] != pure["label"])
    pure["split"] = trial.split

    decided = pure[pure["swap_verdict_adaptive"] != AMBIGUOUS]
    dis = int(pure["contradicts"].sum())
    n = len(decided)

    # WHERE the contradiction sits, not just how much of it there is; `render` already
    # says the reader's job is to tell a swapped channel (wrong everywhere) from a
    # misaligned label track (wrong in a stretch), and neither `disagree_frac` nor the
    # capped nomination list can answer that: 12 nominations spread 5 s apart look
    # identical for a trial that is wrong throughout and one wrong for 80 seconds; this is
    # the span the fraction was taken over, against the span the contradictions occupy
    t_c = pure["t_start_ms"] / 1000.0 + spec.window_s / 2.0
    bad_t = t_c[pure["contradicts"]]
    return ({
        "rev": trial.rev, "trial": trial.trial, "windows": n, "disagree": dis,
        "disagree_frac": dis / n if n else float("nan"),
        "abstain_frac": float((pure["swap_verdict_adaptive"] == AMBIGUOUS).mean())
        if len(pure) else float("nan"),
        "rest_trusted": bool(j["rest_offset_trusted"].iloc[0]),
        "flag": bool(n >= MIN_WINDOWS and n and dis / n > MAX_DISAGREE),
        "t_first_s": float(t_c.min()) if len(t_c) else float("nan"),
        "t_last_s": float(t_c.max()) if len(t_c) else float("nan"),
        "contradict_t_first_s": float(bad_t.min()) if len(bad_t) else float("nan"),
        "contradict_t_last_s": float(bad_t.max()) if len(bad_t) else float("nan"),
    }, pure[WINDOW_COLUMNS].copy())


# per-trial annotation convention inside the ambiguity band
def band_policy(windows: pd.DataFrame, broken: set[tuple[str, int]] | None = None
                ) -> tuple[pd.DataFrame, tuple[float, float], float]:
    broken = broken or set()
    clean = windows[(windows["split"] != "lockbox")
                    & ~windows.set_index(["rev", "trial"]).index.isin(broken)]
    lo, hi = amplitude_band(clean["ileg_minhalf"], clean["label"])
    inband = windows[windows["ileg_minhalf"].between(lo, hi)]
    corpus = float((inband["label"] == WALK).mean()) if len(inband) else float("nan")

    rows = []
    for (rev, trial), g in inband.groupby(["rev", "trial"], sort=False):
        rows.append({"rev": rev, "trial": trial, "band_windows": len(g),
                     "band_walk": int((g["label"] == WALK).sum())})
    out = pd.DataFrame(rows, columns=["rev", "trial", "band_windows", "band_walk"])
    if out.empty:
        return out.assign(band_walk_frac=[], band_p=[], band_flag=[]), (lo, hi), corpus

    out["band_walk_frac"] = out["band_walk"] / out["band_windows"]
    testable = out["band_windows"] >= MIN_WINDOWS
    alpha = BAND_ALPHA / max(int(testable.sum()), 1)
    out["band_p"] = [
        binomtest(int(r.band_walk), int(r.band_windows), corpus,
                  alternative="two-sided").pvalue
        if r.band_windows >= MIN_WINDOWS else float("nan")
        for r in out.itertuples()
    ]
    out["band_flag"] = out["band_p"].notna() & (out["band_p"] < alpha)
    return out, (lo, hi), corpus


# individual windows a human should adjudicate, with the command that shows them
def nominate(windows: pd.DataFrame, targets: set[tuple[str, int]],
             spec: WindowSpec | None = None) -> tuple[pd.DataFrame, int]:
    spec = spec or WindowSpec()
    bad = windows[windows["contradicts"] & windows.set_index(
        ["rev", "trial"]).index.isin(targets)].copy()
    if bad.empty:
        return pd.DataFrame(columns=["rev", "trial", "t_center_s", "cmd"]), 0

    # Centre of the window, which is what `inspect_window --t` wants: it prints a span
    # AROUND the timestamp, so handing it the start would show half the window plus the
    # seconds before it and let the reader adjudicate on the wrong stretch
    bad["t_center_s"] = bad["t_start_ms"] / 1000.0 + spec.window_s / 2.0
    bad["evidence"] = np.where(
        bad["physics_class"] == WALK,
        bad["swap_count_adaptive"].astype(float) - 1.0,
        1.0 / (1.0 + bad["ileg_minhalf"].clip(lower=0.0).astype(float)),
    )

    kept, dropped = [], 0
    for (rev, trial), g in bad.groupby(["rev", "trial"], sort=True):
        taken: list[float] = []
        for r in g.sort_values("evidence", ascending=False).itertuples():
            if len(taken) >= MAX_NOMINATIONS_PER_TRIAL:
                dropped += 1
                continue
            # Greedy spread: strongest first, but never within the gap of one already
            # taken; sorted-by-strength alone clusters on the worst few seconds
            if any(abs(r.t_center_s - t) < MIN_NOMINATION_GAP_S for t in taken):
                dropped += 1
                continue
            taken.append(r.t_center_s)
            kept.append({
                "rev": rev, "trial": int(trial), "segment": int(r.segment),
                "t_center_s": round(float(r.t_center_s), 2),
                "label": "walk" if r.label == WALK else "stand",
                "physics": r.swap_verdict_adaptive,
                "swap_count_adaptive": float(r.swap_count_adaptive),
                "swap_window_s": round(float(r.swap_window_s), 2),
                "ileg_minhalf": round(float(r.ileg_minhalf), 3),
                "cmd": f"python -m stages.s3_physics.inspect_window {rev} {int(trial)} "
                       f"--t {r.t_center_s:.2f}",
            })
    return pd.DataFrame(kept), dropped


# two detectors, two verdicts, one table
def render(df: pd.DataFrame, band: tuple[float, float], corpus: float,
           noms: pd.DataFrame, dropped: int) -> str:
    broken = df[df["flag"]]
    diverge = df[df["band_flag"].fillna(False)].sort_values("band_walk_frac")
    lines = [
        "# Label audit (physics vs annotation)", "",
        "Two independent trial-level failures, both scored against the swap rule (§10), "
        "which has no trained parameter and never sees a label.", "",
        f"- **`disagree`** — share of label-pure, physics-decided windows whose verdict "
        f"contradicts the annotation. Flags above **{MAX_DISAGREE:.2f}** over at least "
        f"{MIN_WINDOWS} windows: the file contradicts itself more often than it agrees. "
        f"That is broken data — a swapped channel or a misaligned label track.",
        f"- **`band walk`** — share of the trial's windows *inside the ambiguity band* "
        f"(**{band[0]:.2f}–{band[1]:.2f}°** interleg swing, where both human labels "
        f"genuinely occur) that it annotated `walk`. The corpus sits at "
        f"**{corpus:.3f}**; a trial far from it is not broken, it is annotated to a "
        f"different convention. Flagged by a two-sided binomial test at "
        f"{BAND_ALPHA:.2f} Bonferroni-corrected across trials with at least "
        f"{MIN_WINDOWS} band windows.", "",
        "The second is the one that costs accuracy quietly: the model learns the corpus "
        "convention, so a divergent trial reads as model error and no amount of retraining "
        "removes it.", "",
    ]
    lines += ["## Broken (`disagree`)", ""]
    if len(broken):
        lines += ["| rev | trial | windows | disagree | fraction |", "|---|---|---|---|---|"]
        for _, r in broken.iterrows():
            lines.append(f"| {r['rev']} | {r['trial']} | {r['windows']:,} | "
                         f"{r['disagree']:,} | **{r['disagree_frac']:.4f}** |")
        lines += ["", "Add confirmed cases to `dataset.EXCLUDED_TRIALS` with the evidence.", ""]
    else:
        lines += ["None.", ""]

    # The bridge to adjudication; a flag nobody can act on is a flag that gets ignored,
    # and "confirmed cases, with the evidence" above is unactionable without these
    lines += ["## Nominated windows — run these to adjudicate", ""]
    if len(noms):
        lines += [
            f"{len(noms)} window(s) across {noms['rev'].nunique()} rev(s), strongest "
            f"contradiction first, spread at least {MIN_NOMINATION_GAP_S:g} s apart so a "
            f"reader can tell a swapped channel (wrong everywhere) from a misaligned "
            f"label track (wrong in a stretch). Capped at "
            f"{MAX_NOMINATIONS_PER_TRIAL} per trial"
            + (f"; **{dropped:,} further contradicting window(s) not listed**." if dropped
               else "; nothing was dropped."), "",
            "A nomination is *not* a verdict. The swap rule can be wrong about a window, "
            "which is exactly why this hands you the raw trace instead of a decision.", "",
            "| rev | trial | t (s) | label | physics | swaps | interleg | command |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for _, r in noms.iterrows():
            lines.append(
                f"| {r['rev']} | {r['trial']} | {r['t_center_s']:.2f} | `{r['label']}` "
                f"| `{r['physics']}` | {r['swap_count_adaptive']:.0f} over "
                f"{r['swap_window_s']:.1f} s | {r['ileg_minhalf']:.2f}° "
                f"| `{r['cmd']}` |")
        lines += ["", "Machine-readable copy: `label_audit_windows.jsonl`.", ""]
    else:
        lines += ["None — no flagged trial has a contradicting window to show.", ""]

    lines += ["## Divergent convention (`band walk`)", ""]
    if len(diverge):
        lines += ["| rev | trial | band windows | band walk | vs corpus | p |",
                  "|---|---|---|---|---|---|"]
        for _, r in diverge.iterrows():
            lines.append(f"| {r['rev']} | {r['trial']} | {int(r['band_windows']):,} | "
                         f"**{r['band_walk_frac']:.3f}** | {r['band_walk_frac']-corpus:+.3f} "
                         f"| {r['band_p']:.2g} |")
        lines += ["",
                  "**Do not exclude these.** A self-consistent minority convention is not "
                  "bad data; it is the part of the residual error that is annotation "
                  "policy rather than model failure. Re-annotating them to the corpus "
                  "convention — or accepting them and reporting the ceiling — are both "
                  "defensible. Silently counting them as model error is not.", ""]
    else:
        lines += ["None — every trial annotates the band consistently with the corpus.", ""]

    lines += ["## All trials, worst first", "",
              "| rev | trial | windows | disagree | physics abstains | band n | band walk | rest |",
              "|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        bn = "—" if pd.isna(r["band_windows"]) else f"{int(r['band_windows']):,}"
        bw = "—" if pd.isna(r["band_walk_frac"]) else f"{r['band_walk_frac']:.3f}"
        mark = " ⚑" if bool(r["band_flag"]) else ""
        lines.append(f"| {r['rev']} | {r['trial']} | {r['windows']:,} | "
                     f"{r['disagree_frac']:.4f} | {r['abstain_frac']:.4f} | {bn} | "
                     f"{bw}{mark} | {'yes' if r['rest_trusted'] else 'NO'} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/s3_physics")
    ap.add_argument("--include-lockbox", action="store_true")
    args = ap.parse_args()

    out_dir = (REPO_ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the RAW corpus: a trial already quarantined must still appear, or the report
    # silently stops justifying the exclusion it caused
    kw = {"lockbox_revs": ()} if args.include_lockbox else {}
    trials = load_dataset(excluded=set(), **kw)

    scored = [coherence(t) for t in trials]
    rows = [r for r, _ in scored]
    windows = pd.concat([w for _, w in scored if len(w)], ignore_index=True)

    # The first detector runs first on purpose: its verdicts decide which trials are fit to
    # define the band the second detector measures against
    broken_keys = {(r["rev"], r["trial"]) for r in rows if r["flag"]}
    policy, band, corpus = band_policy(windows, broken_keys)
    df = (pd.DataFrame([r for r in rows if r["windows"]])
          .merge(policy, on=["rev", "trial"], how="left")
          .sort_values("disagree_frac", ascending=False).reset_index(drop=True))
    df["band_flag"] = df["band_flag"].fillna(False).astype(bool)
    broken, diverge = df[df["flag"]], df[df["band_flag"]]

    print(f"[audit] {len(df)} trials scored against the physics verdict")
    print(f"[audit] ambiguity band {band[0]:.2f}-{band[1]:.2f} deg holds "
          f"{len(windows[windows['ileg_minhalf'].between(*band)]):,} of {len(windows):,} "
          f"label-pure windows; the corpus annotates {corpus:.1%} of it walk")

    if len(broken):
        print(f"\n[audit] BROKEN: {len(broken)} trial(s) over {MAX_DISAGREE:.2f} disagreement:")
        for _, r in broken.iterrows():
            print(f"[audit]   {r['rev']} trial {r['trial']}: {r['disagree']:,}/"
                  f"{r['windows']:,} windows contradict the label ({r['disagree_frac']:.1%})")
        print("[audit]   -> add confirmed cases to dataset.EXCLUDED_TRIALS; nothing deleted")
    else:
        print(f"\n[audit] no trial exceeds {MAX_DISAGREE:.2f} disagreement")

    if len(diverge):
        print(f"\n[audit] DIVERGENT CONVENTION: {len(diverge)} trial(s) annotate the "
              f"ambiguity band unlike the corpus ({corpus:.2f}):")
        for _, r in diverge.sort_values("band_walk_frac").iterrows():
            print(f"[audit]   {r['rev']} trial {r['trial']}: {r['band_walk_frac']:.2f} walk "
                  f"over {int(r['band_windows'])} band windows (p={r['band_p']:.2g})")
        print("[audit]   -> NOT grounds for exclusion; this is annotation policy, not bad data")
    else:
        print("[audit] every trial annotates the ambiguity band consistently with the corpus")

    # Nominate for the BROKEN trials only; a divergent-convention trial is not a candidate
    # for window-by-window adjudication: its windows are labelled to a self-consistent
    # minority convention, so each one looks defensible in isolation and only the
    # trial-level rate says anything; sending a reader to inspect them would invite exactly
    # the exclusion the second detector exists to argue against
    noms, dropped = nominate(windows, broken_keys)
    if len(noms):
        print(f"\n[audit] {len(noms)} window(s) nominated for adjudication"
              + (f" ({dropped:,} further contradicting windows not listed)" if dropped else ""))
        for _, r in noms.head(3).iterrows():
            print(f"[audit]   {r['cmd']}   # label={r['label']} physics={r['physics']}")
        if len(noms) > 3:
            print(f"[audit]   ... {len(noms) - 3} more in label_audit_windows.jsonl")

    (out_dir / "label_audit.md").write_text(
        render(df, band, corpus, noms, dropped), encoding="utf-8")
    (out_dir / "label_audit.json").write_text(json.dumps(
        {"max_disagree": MAX_DISAGREE, "min_windows": MIN_WINDOWS,
         "band_alpha": BAND_ALPHA, "band": list(band), "band_corpus_walk_frac": corpus,
         "n_nominated": len(noms), "n_nominations_dropped": dropped,
         "trials": df.to_dict("records")}, indent=2, default=float), encoding="utf-8")
    # Written even when empty, and that is the point: a zero-line file says the audit ran
    # and found nothing to adjudicate, where a missing file says nobody looked
    (out_dir / "label_audit_windows.jsonl").write_text(
        "".join(json.dumps(r, default=float) + "\n" for r in noms.to_dict("records")),
        encoding="utf-8")
    print(f"\n[audit] -> {out_dir / 'label_audit.md'}")


if __name__ == "__main__":
    main()
