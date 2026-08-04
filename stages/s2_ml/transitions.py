"""How long is a transition, how many are there, and does the pipeline time them right?

`near_transition` is the single biggest abstention driver in the deliverable — 69,600 rows
at the shipped threshold, dominating 50 of the 78 swept corpus files, and the reason with
the weakest suppressed guess (0.6090) of any in `OPERATING_POINTS.md`. Nothing measured the
thing it fires on. `needtowrite.md` §5.4 gap 5 is that hole, and this closes it.

**Three questions, and the third is the one that can embarrass the pipeline.**

  *How many* — transitions between annotated `stand` and `walk` runs, by direction.
  *How long* — the annotator's own `-1` interval at each one. That is the honest answer to
    "how long is a transition": ground truth does not step instantaneously, it goes
    stand -> "I looked and cannot call it" -> walk, and the width of that middle run is a
    measurement of the boundary rather than a model of it.
  *Is the timing right* — the offset between each annotated boundary and the nearest state
    change the model actually predicted, plus whether `near_transition` fired there at all.

**The counting trap, stated because this repo already fell into it.** Run-length-encoding an
ALREADY-FILTERED label sequence turns one boundary into many: dropping the `-1` rows first
means every short unknown stretch inside a walk run closes and reopens it. Everything here
encodes the raw `truth` column with the unknowns still in it, and `-1` is read as the
transition's WIDTH rather than removed. `n_rejected` says exactly how many candidate
boundaries were dropped and why, so a number that shrinks is legible instead of alarming.

**This does not reconcile with the counts in `needtowrite.md` §5.4, and that is open.**
That note reports 185 candidates narrowed to 27 timeable, 14 of them on time. This module
measures 308 candidates, 267 timeable, over the leave-one-rev-out frame. Neither figure is
derivable from the other and the older pair predates the deleted S4 window grid, so no
attempt is made here to explain it away — the numbers below are what this definition, run
over this corpus, produces. Settle the discrepancy before either pair is quoted anywhere.

**It measures the serve path's own predictions**, not a reimplementation: `roweval` hands
this the frame it scored with the real `label.py`, so a transition is timed against the
guesses a caller receives. It is a library rather than a CLI for that reason — a second
leave-one-rev-out pass would let a refit drift between the coverage curve and this table
and show up as a timing effect.

**It publishes no policy and no accuracy pair.** It is a diagnostic: it says what the
biggest bucket of doubt in the deliverable is made of, and nothing downstream reads it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from stages.s2_ml.dataset import HUMAN_UNKNOWN, STAND, TIME_COL, WALK
from stages.s2_ml.features import WindowSpec

CLASS_NAME = {STAND: "stand", WALK: "walk"}

# A boundary is TIMEABLE when both sides hold at least one full window of their own class.
# Not a tuning knob: below it the model has no pure window on one side, so it cannot place
# a change there, and scoring its timing would be measuring the window length. Transitions
# that fail this are counted and reported, never silently dropped.
MIN_FLANK_WINDOWS = 1.0

# ...and when the boundary is at least one window from either end of the segment, for the
# same reason in the other direction: a predicted change needs room on both sides to exist.
EDGE_MARGIN_WINDOWS = 1.0

# The tolerances the offset is reported against. HALF a window is the tight one because
# that is exactly the radius `label.explain` marks `near_transition` over — a boundary timed
# better than this is one the flag covers — and a FULL window is the loose one, the point
# past which no window contains both the boundary and the row being judged.
TIGHT_TOL_WINDOWS = 0.5
LOOSE_TOL_WINDOWS = 1.0

NEAR_TRANSITION = "near_transition"


def _runs(a: np.ndarray) -> list[tuple]:
    """Run-length encode: [(value, start, stop_exclusive), ...]. NaN-free input only."""
    n = len(a)
    if n == 0:
        return []
    change = np.flatnonzero(a[1:] != a[:-1]) + 1
    starts = np.concatenate(([0], change))
    stops = np.concatenate((change, [n]))
    return [(a[s], int(s), int(e)) for s, e in zip(starts, stops)]


def _predicted_changes(t_ms: np.ndarray, guess: np.ndarray) -> np.ndarray:
    """Instants where the model's committed-or-not guess flips class, in ms.

    The guess, not the label: `label.explain` nulls `label` only where nothing covered the
    row, never for being unsure, so this is the model's opinion about where the boundary is
    regardless of whether it committed to either side. Timing an abstention's placement
    against a threshold it did not clear would just re-measure the threshold.

    Uncovered rows are skipped rather than filled, so a gap never manufactures a change.
    """
    cov = np.isfinite(guess)
    idx = np.flatnonzero(cov)
    if idx.size < 2:
        return np.empty(0)
    g = guess[idx]
    ch = np.flatnonzero(g[1:] != g[:-1])
    if ch.size == 0:
        return np.empty(0)
    # Midpoint of the two rows that straddle the flip: the change happened between them and
    # attributing it to either would bias every offset by half a sample.
    return (t_ms[idx[ch]] + t_ms[idx[ch + 1]]) / 2.0


def find_transitions(df: pd.DataFrame, spec: WindowSpec) -> tuple[pd.DataFrame, dict]:
    """Every annotated stand<->walk boundary, with its width and the model's timing.

    Returns (one row per boundary, rejection counts). Segments are the unit: a boundary
    cannot span a gap, because the two sides were not measured continuously and nothing
    about the interval between them is observed.
    """
    min_flank_ms = MIN_FLANK_WINDOWS * spec.window_s * 1000.0
    edge_ms = EDGE_MARGIN_WINDOWS * spec.window_s * 1000.0

    rows: list[dict] = []
    rejected = {"short_flank": 0, "near_segment_edge": 0}

    for (rev, trial, seg), g in df.groupby(["rev", "trial", "segment"], sort=True):
        g = g.sort_values(TIME_COL)
        t = g[TIME_COL].to_numpy(float)
        truth = g["truth"].to_numpy()
        guess = pd.to_numeric(g["label"], errors="coerce").to_numpy(float)
        reason = g["reason"].fillna("").astype(str).to_numpy(object)
        if len(t) < 2:
            continue

        runs = _runs(truth)
        preds = _predicted_changes(t, guess)

        for i, (v0, s0, e0) in enumerate(runs):
            if v0 not in (STAND, WALK):
                continue
            # At most ONE unknown run may sit between the two states, and its width is the
            # transition. Two or more non-trained runs in a row is not a boundary this can
            # read — some other annotation is interleaved — so it is skipped rather than
            # guessed at, and skipping is invisible here by design: it never became a
            # candidate, so it is not a rejection either.
            j = i + 1
            if j < len(runs) and runs[j][0] == HUMAN_UNKNOWN:
                j += 1
            if j >= len(runs):
                continue
            v1, s1, e1 = runs[j]
            if v1 not in (STAND, WALK) or v1 == v0:
                continue

            t_before, t_after = t[e0 - 1], t[s1]
            t_truth = (t_before + t_after) / 2.0
            flank_before = t_before - t[s0]
            flank_after = t[e1 - 1] - t_after

            if flank_before < min_flank_ms or flank_after < min_flank_ms:
                rejected["short_flank"] += 1
                continue
            if (t_truth - t[0]) < edge_ms or (t[-1] - t_truth) < edge_ms:
                rejected["near_segment_edge"] += 1
                continue

            # Nearest predicted change. `None` means the model never changed class anywhere
            # in this segment — a miss, and a different failure from a badly timed hit, so
            # it is never folded into the offset distribution as a large number.
            offset_s = None
            if preds.size:
                offset_s = float((preds[np.argmin(np.abs(preds - t_truth))] - t_truth) / 1000.0)

            near = (np.abs(t - t_truth) <= TIGHT_TOL_WINDOWS * spec.window_s * 1000.0)
            rows.append({
                "rev": rev, "trial": int(trial), "segment": int(seg),
                "direction": f"{CLASS_NAME[v0]}->{CLASS_NAME[v1]}",
                "t_truth_s": round(t_truth / 1000.0, 3),
                # The annotator's own unknown interval. 0.0 means the two runs are adjacent:
                # the boundary was called to the sample, with no admitted doubt.
                "width_s": round((t_after - t_before) / 1000.0, 3),
                "flank_before_s": round(flank_before / 1000.0, 3),
                "flank_after_s": round(flank_after / 1000.0, 3),
                "offset_s": None if offset_s is None else round(offset_s, 3),
                "flagged_near_transition": bool((reason[near] == NEAR_TRANSITION).any()),
            })

    return pd.DataFrame(rows), rejected


def near_transition_regions(df: pd.DataFrame, trans: pd.DataFrame,
                            spec: WindowSpec) -> dict:
    """Do the `near_transition` rows sit near an annotated transition?

    The converse question to `find_transitions`, and the one that decides whether the
    biggest abstention bucket in the deliverable is earning its rows. A contiguous stretch
    of flagged rows is one event however many rows it spans, so this counts REGIONS: 77,025
    rows is meaningless as a count of anything the flag claims to have found.

    Matched against EVERY annotated boundary, including the ones `find_transitions`
    rejected as untimeable. A flag beside a boundary too short to time is still correctly
    placed, and holding it to the timeable subset would manufacture false alarms out of the
    rejection rule.
    """
    tol_ms = LOOSE_TOL_WINDOWS * spec.window_s * 1000.0
    by_seg: dict[tuple, np.ndarray] = {}
    if len(trans):
        for k, g in trans.groupby(["rev", "trial", "segment"], sort=False):
            by_seg[k] = g["t_truth_s"].to_numpy(float) * 1000.0

    n_regions = n_unmatched = n_rows = 0
    for (rev, trial, seg), g in df.groupby(["rev", "trial", "segment"], sort=True):
        g = g.sort_values(TIME_COL)
        t = g[TIME_COL].to_numpy(float)
        flag = (g["reason"].fillna("").astype(str).to_numpy(object) == NEAR_TRANSITION)
        n_rows += int(flag.sum())
        if not flag.any():
            continue
        truths = by_seg.get((rev, trial, seg), np.empty(0))
        for v, s, e in _runs(flag.astype(np.int8)):
            if not v:
                continue
            n_regions += 1
            # The region's own span, widened by the tolerance. A long flagged stretch
            # should not need the boundary at its centre — it only needs to contain one.
            if not truths.size or not (
                    (truths >= t[s] - tol_ms) & (truths <= t[e - 1] + tol_ms)).any():
                n_unmatched += 1

    return {
        "rows_flagged": n_rows,
        "regions": n_regions,
        "regions_with_no_annotated_transition": n_unmatched,
        "unmatched_frac": (n_unmatched / n_regions) if n_regions else float("nan"),
        "tolerance_s": LOOSE_TOL_WINDOWS * spec.window_s,
    }


def summarize(trans: pd.DataFrame, rejected: dict, regions: dict,
              spec: WindowSpec) -> dict:
    """Aggregate the per-boundary table. Every share carries its denominator."""
    tight = TIGHT_TOL_WINDOWS * spec.window_s
    loose = LOOSE_TOL_WINDOWS * spec.window_s
    n = len(trans)
    timed = trans[trans["offset_s"].notna()] if n else trans
    off = timed["offset_s"].to_numpy(float) if len(timed) else np.empty(0)

    out = {
        "window_s": spec.window_s,
        "tight_tolerance_s": tight,
        "loose_tolerance_s": loose,
        "n_timeable": n,
        "n_rejected": rejected,
        "n_rejected_total": int(sum(rejected.values())),
        "by_direction": (trans["direction"].value_counts().to_dict() if n else {}),
        "width_s": {},
        "timing": {},
        "near_transition": regions,
    }
    if n:
        w = trans["width_s"].to_numpy(float)
        out["width_s"] = {
            "median": float(np.median(w)), "mean": float(w.mean()),
            "p90": float(np.percentile(w, 90)), "max": float(w.max()),
            "n_zero_width": int((w == 0).sum()),
        }
        out["flagged_near_transition"] = {
            "n": int(trans["flagged_near_transition"].sum()),
            "frac": float(trans["flagged_near_transition"].mean()),
        }
    if off.size:
        out["timing"] = {
            "n_timed": int(off.size),
            "n_no_predicted_change": int(n - off.size),
            "median_offset_s": float(np.median(off)),
            "median_abs_offset_s": float(np.median(np.abs(off))),
            "p90_abs_offset_s": float(np.percentile(np.abs(off), 90)),
            "n_within_tight": int((np.abs(off) <= tight).sum()),
            "frac_within_tight": float((np.abs(off) <= tight).mean()),
            "n_within_loose": int((np.abs(off) <= loose).sum()),
            "frac_within_loose": float((np.abs(off) <= loose).mean()),
            # Sign convention stated once, here and in the report: positive means the model
            # changed class LATE. Direction matters because a systematic lag is a filter
            # artifact with a fix, where symmetric scatter is just resolution.
            "n_late": int((off > 0).sum()),
            "n_early": int((off < 0).sum()),
        }
    return out


def per_rev(trans: pd.DataFrame, spec: WindowSpec) -> list[dict]:
    """The same numbers per subject: a pooled median hides one badly-timed subject."""
    if not len(trans):
        return []
    tight = TIGHT_TOL_WINDOWS * spec.window_s
    rows = []
    for rev, g in trans.groupby("rev", sort=True):
        off = g.loc[g["offset_s"].notna(), "offset_s"].to_numpy(float)
        rows.append({
            "rev": str(rev),
            "transitions": int(len(g)),
            "median_width_s": float(g["width_s"].median()),
            "median_abs_offset_s": float(np.median(np.abs(off))) if off.size else float("nan"),
            "frac_within_tight": float((np.abs(off) <= tight).mean()) if off.size else float("nan"),
            "frac_flagged": float(g["flagged_near_transition"].mean()),
        })
    return rows


def render(tag: str, s: dict, by_rev: list[dict], trans: pd.DataFrame) -> str:
    tight, loose = s["tight_tolerance_s"], s["loose_tolerance_s"]
    n, rej = s["n_timeable"], s["n_rejected"]
    lines = [
        f"# Transitions - {tag}", "",
        "What the deliverable's biggest abstention bucket is actually made of. "
        f"`near_transition` suppresses more rows than any other reason and its suppressed "
        f"guess is the least accurate of any; until now nothing measured the boundaries it "
        f"fires on (`needtowrite.md` §5.4, gap 5).", "",
        f"**{n} timeable transitions**, from "
        f"{n + s['n_rejected_total']} annotated stand/walk boundaries. Rejected: "
        + ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in rej.items()) + ".", "",
        f"A boundary is timeable when both sides hold at least "
        f"{MIN_FLANK_WINDOWS:g} full window ({s['window_s']:g} s) of their own class and it "
        f"sits at least {EDGE_MARGIN_WINDOWS:g} window from either end of its segment — "
        f"below that the model has no pure window to place a change from, and scoring its "
        f"timing would measure the window length. **The rejections are counted, not "
        f"dropped.** Everything here encodes the raw `truth` column with the unknowns still "
        f"in it, and reads `-1` as the boundary's width rather than removing it — filtering "
        f"first splits one boundary into many.", "",
    ]

    if s["by_direction"]:
        lines += ["## How many, and which way", "", "| direction | transitions |",
                  "|---|---|"]
        for k, v in sorted(s["by_direction"].items(), key=lambda x: -x[1]):
            lines.append(f"| `{k}` | {v} |")
        lines.append("")

    if s["width_s"]:
        w = s["width_s"]
        lines += [
            "## How long is a transition", "",
            "The annotator's own `-1` interval between the two runs — ground truth does not "
            "step instantaneously, and this is the width of its admitted doubt rather than a "
            "model of it. A width of 0 means the boundary was called to the sample.", "",
            f"| median | mean | p90 | max | called to the sample |",
            "|---|---|---|---|---|",
            f"| {w['median']:.2f} s | {w['mean']:.2f} s | {w['p90']:.2f} s | "
            f"{w['max']:.2f} s | {w['n_zero_width']} of {n} |", "",
        ]

    t = s.get("timing") or {}
    if t:
        lines += [
            "## Is the pipeline's timing right", "",
            "Offset from each annotated boundary to the nearest state change the model "
            "predicted. **Positive is late.** Measured on the guess, not the committed "
            "label: where the model thinks the boundary is does not depend on whether it "
            "cleared the threshold on either side.", "",
            f"- transitions with a predicted change in the segment: **{t['n_timed']} of {n}**"
            + (f" ({t['n_no_predicted_change']} segment(s) never change class at all)"
               if t["n_no_predicted_change"] else ""),
            f"- median offset: **{t['median_offset_s']:+.2f} s** "
            f"({t['n_late']} late, {t['n_early']} early)",
            f"- median |offset|: **{t['median_abs_offset_s']:.2f} s**, "
            f"p90 {t['p90_abs_offset_s']:.2f} s",
            f"- within ±{tight:g} s (the radius `near_transition` marks): "
            f"**{t['n_within_tight']} of {t['n_timed']}** ({t['frac_within_tight']:.0%})",
            f"- within ±{loose:g} s (one window): "
            f"**{t['n_within_loose']} of {t['n_timed']}** ({t['frac_within_loose']:.0%})", "",
        ]

    f = s.get("flagged_near_transition")
    r = s["near_transition"]
    if f:
        lines += [
            "## Does `near_transition` fire where the boundaries are", "",
            f"**{f['n']} of {n} ({f['frac']:.0%})** annotated transitions have a "
            f"`near_transition` row within ±{tight:g} s. That is the flag's recall.", "",
            f"Its precision, from the other side: the {r['rows_flagged']:,} flagged rows form "
            f"**{r['regions']:,} contiguous regions**, of which "
            f"**{r['regions_with_no_annotated_transition']:,} "
            f"({r['unmatched_frac']:.0%})** contain no annotated transition within "
            f"±{r['tolerance_s']:g} s of their span.", "",
            "Regions, not rows: a stretch of flagged rows is one event however wide it is, "
            "and a row count says nothing about how many boundaries the flag claims. An "
            "unmatched region is not necessarily wrong — the model may be changing class "
            "somewhere the annotation does not — but it is the flag firing where ground "
            "truth sees no boundary, which is what its 0.60 suppressed-guess accuracy has "
            "to be read against.", "",
        ]

    if by_rev:
        lines += ["## Per subject", "",
                  "| rev | transitions | median width | median &#124;offset&#124; | "
                  f"within ±{tight:g} s | flagged |", "|---|---|---|---|---|---|"]
        for x in by_rev:
            mo = "—" if pd.isna(x["median_abs_offset_s"]) else f"{x['median_abs_offset_s']:.2f} s"
            fw = "—" if pd.isna(x["frac_within_tight"]) else f"{x['frac_within_tight']:.0%}"
            lines.append(f"| `{x['rev']}` | {x['transitions']} | "
                         f"{x['median_width_s']:.2f} s | {mo} | {fw} | "
                         f"{x['frac_flagged']:.0%} |")
        lines.append("")

    if len(trans):
        lines += ["## Every timeable transition", "",
                  "| rev | trial | seg | t (s) | direction | width | offset | flagged |",
                  "|---|---|---|---|---|---|---|---|"]
        for _, x in trans.sort_values(["rev", "trial", "t_truth_s"]).iterrows():
            off = "—" if pd.isna(x["offset_s"]) else f"{x['offset_s']:+.2f} s"
            lines.append(f"| {x['rev']} | {x['trial']} | {x['segment']} | "
                         f"{x['t_truth_s']:.2f} | `{x['direction']}` | "
                         f"{x['width_s']:.2f} s | {off} | "
                         f"{'yes' if x['flagged_near_transition'] else 'NO'} |")
    return "\n".join(lines)


def run(df: pd.DataFrame, spec: WindowSpec, out_dir: Path, stem: str, tag: str) -> dict:
    """Measure, write `<stem>.md` + `<stem>.json`, return the summary for the caller."""
    trans, rejected = find_transitions(df, spec)
    regions = near_transition_regions(df, trans, spec)
    s = summarize(trans, rejected, regions, spec)
    by_rev = per_rev(trans, spec)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.md").write_text(render(tag, s, by_rev, trans), encoding="utf-8")
    (out_dir / f"{stem}.json").write_text(json.dumps(
        {"tag": tag, "summary": s, "per_rev": by_rev,
         "transitions": trans.to_dict("records")}, indent=2, default=float),
        encoding="utf-8")
    return s


def print_summary(s: dict) -> None:
    t = s.get("timing") or {}
    r = s["near_transition"]
    print(f"[trans] {s['n_timeable']} timeable transitions "
          f"({s['n_rejected_total']} rejected: {s['n_rejected']})")
    if s["width_s"]:
        print(f"[trans] annotator's unknown interval: median "
              f"{s['width_s']['median']:.2f}s  p90 {s['width_s']['p90']:.2f}s  "
              f"{s['width_s']['n_zero_width']} called to the sample")
    if t:
        print(f"[trans] timing: median offset {t['median_offset_s']:+.2f}s "
              f"({t['n_late']} late / {t['n_early']} early), "
              f"{t['frac_within_tight']:.0%} within +/-{s['tight_tolerance_s']:g}s")
    if s.get("flagged_near_transition"):
        print(f"[trans] near_transition recall: "
              f"{s['flagged_near_transition']['frac']:.0%} of transitions flagged")
    print(f"[trans] near_transition precision: {r['regions']:,} regions, "
          f"{r['regions_with_no_annotated_transition']:,} "
          f"({r['unmatched_frac']:.0%}) with no annotated transition")
