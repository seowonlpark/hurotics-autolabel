# how long transitions are and whether the pipeline times them; near_transition drives abstention

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from stages.s2_ml.dataset import CLASS_NAME, HUMAN_UNKNOWN, STAND, TIME_COL, WALK
from stages.s2_ml.features import WindowSpec

# TIMEABLE needs a full window of each class on both sides; failures are counted, never dropped
MIN_FLANK_WINDOWS = 1.0

# ...and one window of room from either segment end, for the same reason in the other direction
EDGE_MARGIN_WINDOWS = 1.0

# HALF a window is the radius `label.explain` marks `near_transition` over; a FULL one is the limit
TIGHT_TOL_WINDOWS = 0.5
LOOSE_TOL_WINDOWS = 1.0

NEAR_TRANSITION = "near_transition"


# run-length encode: [(value, start, stop_exclusive), ...]; NaN-free input only
def _runs(a: np.ndarray) -> list[tuple]:
    n = len(a)
    if n == 0:
        return []
    change = np.flatnonzero(a[1:] != a[:-1]) + 1
    starts = np.concatenate(([0], change))
    stops = np.concatenate((change, [n]))
    return [(a[s], int(s), int(e)) for s, e in zip(starts, stops)]


# instants where the model's committed-or-not guess flips class, in ms
def _predicted_changes(t_ms: np.ndarray, guess: np.ndarray) -> np.ndarray:
    cov = np.isfinite(guess)
    idx = np.flatnonzero(cov)
    if idx.size < 2:
        return np.empty(0)
    g = guess[idx]
    ch = np.flatnonzero(g[1:] != g[:-1])
    if ch.size == 0:
        return np.empty(0)
    # midpoint of the straddling rows: attributing the flip to either biases offsets half a sample
    return (t_ms[idx[ch]] + t_ms[idx[ch + 1]]) / 2.0


# every annotated stand<->walk boundary, with its width and the model's timing
def find_transitions(df: pd.DataFrame, spec: WindowSpec) -> tuple[pd.DataFrame, dict]:
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

            # nearest predicted change; None means it never changed class here- a miss, not a bad hit
            offset_s = None
            if preds.size:
                offset_s = float((preds[np.argmin(np.abs(preds - t_truth))] - t_truth) / 1000.0)

            near = (np.abs(t - t_truth) <= TIGHT_TOL_WINDOWS * spec.window_s * 1000.0)
            rows.append({
                "rev": rev, "trial": int(trial), "segment": int(seg),
                "direction": f"{CLASS_NAME[v0]}->{CLASS_NAME[v1]}",
                "t_truth_s": round(t_truth / 1000.0, 3),
                # the annotator's own unknown interval; 0.0 means the boundary was called to the sample
                "width_s": round((t_after - t_before) / 1000.0, 3),
                "flank_before_s": round(flank_before / 1000.0, 3),
                "flank_after_s": round(flank_after / 1000.0, 3),
                "offset_s": None if offset_s is None else round(offset_s, 3),
                "flagged_near_transition": bool((reason[near] == NEAR_TRANSITION).any()),
            })

    return pd.DataFrame(rows), rejected


# do the `near_transition` rows sit near an annotated transition?
def near_transition_regions(df: pd.DataFrame, trans: pd.DataFrame,
                            spec: WindowSpec) -> dict:
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
            # the region's span widened by the tolerance: a long stretch need only contain a boundary
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


# aggregate the per-boundary table; every share carries its denominator
def summarize(trans: pd.DataFrame, rejected: dict, regions: dict,
              spec: WindowSpec) -> dict:
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
            # positive is LATE; direction matters- a systematic lag is fixable, scatter is resolution
            "n_late": int((off > 0).sum()),
            "n_early": int((off < 0).sum()),
        }
    return out


# the same numbers per subject: a pooled median hides one badly-timed subject
def per_rev(trans: pd.DataFrame, spec: WindowSpec) -> list[dict]:
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
        f"fires on (`archive/needtowrite.md` §5.4, gap 5).", "",
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


# measure, write `<stem>.md` + `<stem>.json`, return the summary for the caller
def run(df: pd.DataFrame, spec: WindowSpec, out_dir: Path, stem: str, tag: str,
        report: bool = False) -> dict:
    trans, rejected = find_transitions(df, spec)
    regions = near_transition_regions(df, trans, spec)
    s = summarize(trans, rejected, regions, spec)
    by_rev = per_rev(trans, spec)

    out_dir.mkdir(parents=True, exist_ok=True)
    if report:
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
