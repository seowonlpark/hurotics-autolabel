# S4 curation: turn the confidence signal into a data-collection director.
# the pipeline's most honest remaining lever is not more macro-F1 in code -- it is better
# GROUND TRUTH (Section 12). this ranks the windows worth a human's attention into a worklist, and
# routes each: relabel it (physics contradicts the label), characterise it as a possible new
# class (structure the 2-class taxonomy misses -- governed, Section 11.2), or collect more of the
# condition (a systematically hard regime). reads artifacts only (fused table + S3 anchors);
# it directs the data effort, it never changes a label. see DOMAIN_NOTES Section 12.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from stages.s2_ml.dataset import STAND, WALK
from stages.s3_physics.anchors import STANDING, WALKING, AMBIGUOUS
from stages.s3_physics.run import ANCHORS_CSV, S3_OUT_DIR
from stages.s4_fusion.run import FUSED_CSV, S4_OUT_DIR, JOIN_KEYS

# grav_stab below this in a STAND-labeled window = the posture is moving, not quiet stance --
# the motion-contaminated "standing" that caps steady_confusion (Section 12). not fitted; grav_stab is
# a [0,1] steadiness score and 0.5 is its natural midpoint between still and sweeping.
GRAV_MOVING = 0.5

# routes, most-actionable first
RELABEL = "relabel_candidate" # physics contradicts the human label -> re-annotate
NEW_CLASS = "new_class_candidate" # structure the {stand,walk} taxonomy misses -> characterise (governed)
COLLECT = "collect_more" # a systematically hard condition -> collect more of it


# the route for one window, or None if it needs no attention. label is the human ground truth,
# s2 the learned prediction, s3 the physics verdict, conf the fusion confidence, grav steadiness.
# RELABEL requires BOTH independent models to contradict the label -- one model disagreeing is
# usually that model's own error (physics over-calling slow gait, Section 10.6), not a mislabel; two
# independent methods rarely share a failure, so their agreement against the label is the
# strongest signal available. still only a *candidate*: a human confirms, code never edits a
# label (non-negotiable #1; the "error_rate 1.00" against the label is disagreement by
# construction, not proof of wrongness -- Section 12.2).
def route_window(label: int, s2: int, s3: str, conf: str, grav: float) -> str | None:
    if label == STAND and s2 == WALK and s3 == WALKING:
        return RELABEL # both models say WALK, human says STAND -> strong mislabel candidate
    if label == WALK and s2 == STAND and s3 == STANDING and grav >= GRAV_MOVING:
        return RELABEL # both models say a genuinely still STAND, human says WALK
    if label == STAND and grav < GRAV_MOVING and s3 in (AMBIGUOUS, STANDING):
        return NEW_CLASS # "standing" that is moving but not clean gait -- ramp/repositioning (H2)
    if conf == "low":
        return COLLECT # single-model disagreement / slow-cadence under-call -> review + more data
    return None


# group consecutive flagged windows of one route into spans, per (rev, trial, segment)
def to_spans(df: pd.DataFrame) -> list[dict]:
    spans = []
    df = df.sort_values(JOIN_KEYS).reset_index(drop=True)
    cur = None
    for r in df.itertuples():
        if not isinstance(r.route, str): # None/NaN -> not flagged
            cur = None
            continue
        same = (cur and cur["rev"] == r.rev and cur["trial"] == r.trial
                and cur["segment"] == r.segment and cur["route"] == r.route)
        if not same:
            cur = {"rev": r.rev, "trial": int(r.trial), "segment": int(r.segment),
                   "route": r.route, "t_start_ms": r.t_start_ms, "t_end_ms": r.t_start_ms,
                   "n_windows": 0, "n_error": 0, "grav_sum": 0.0, "label": int(r.true)}
            spans.append(cur)
        cur["t_end_ms"] = r.t_start_ms
        cur["n_windows"] += 1
        cur["n_error"] += int(r.fused_label != r.true)
        cur["grav_sum"] += float(r.grav_stab)
    return spans


# priority = span length weighted by route (a label a human can flip is worth more than a
# request for more data), and on this labeled corpus we can show the flagged span really is
# error-dense (n_error) -- the director targeting real problems, not noise
def prioritise(spans: list[dict]) -> list[dict]:
    weight = {RELABEL: 3.0, NEW_CLASS: 2.0, COLLECT: 1.0}
    out = []
    for s in spans:
        out.append({
            "route": s["route"],
            "rev": s["rev"], "trial": s["trial"],
            "t_start_s": round(s["t_start_ms"] / 1000, 1),  # ms are trial-absolute
            "t_end_s": round(s["t_end_ms"] / 1000, 1),
            "n_windows": s["n_windows"],
            "human_label": "STAND" if s["label"] == STAND else "WALK",
            "grav_stab_mean": round(s["grav_sum"] / s["n_windows"], 3),
            "error_rate": round(s["n_error"] / s["n_windows"], 2),
            "priority": round(weight[s["route"]] * s["n_windows"], 1),
        })
    return sorted(out, key=lambda x: x["priority"], reverse=True)


# join the fused table with the S3 anchors, route every window, rank the spans
def build_queue(s4_dir: Path = S4_OUT_DIR, s3_dir: Path = S3_OUT_DIR) -> list[dict]:
    fused = pd.read_csv(s4_dir / FUSED_CSV)
    anchors = pd.read_csv(s3_dir / ANCHORS_CSV)[JOIN_KEYS + ["grav_stab"]]
    df = fused.merge(anchors, on=JOIN_KEYS, how="inner")
    df["route"] = [route_window(int(t), int(s2), s3, c, g)
                   for t, s2, s3, c, g in
                   zip(df["true"], df["s2_pred"], df["s3"], df["confidence"], df["grav_stab"])]
    return prioritise(to_spans(df))


# render the worklist as markdown, grouped by route
def render(queue: list[dict]) -> str:
    L = ["# S4 curation queue - where data investment pays off", "",
         "The confidence signal as a data-collection director (DOMAIN_NOTES Section 12). Each span is "
         "routed; `error_rate` is on the labeled corpus, showing the queue targets real errors.", ""]
    labels = {RELABEL: "Relabel candidates - physics contradicts the human label",
              NEW_CLASS: "New-class candidates - structure the stand/walk taxonomy misses (governed, Section 11.2)",
              COLLECT: "Collect-more - systematically hard conditions"}
    for route in (RELABEL, NEW_CLASS, COLLECT):
        rows = [q for q in queue if q["route"] == route]
        if not rows:
            continue
        n_win = sum(r["n_windows"] for r in rows)
        L += [f"## {labels[route]}  ({len(rows)} spans, {n_win} windows)", "",
              "| priority | rev/trial | t_start-t_end (s) | label | grav_stab | error_rate | windows |",
              "|---|---|---|---|---|---|---|"]
        for q in rows[:12]:
            L.append(f"| {q['priority']} | {q['rev']}_t{q['trial']} | "
                     f"{q['t_start_s']}-{q['t_end_s']} | {q['human_label']} | "
                     f"{q['grav_stab_mean']} | {q['error_rate']} | {q['n_windows']} |")
        L.append("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the S4 data-collection curation queue.")
    ap.add_argument("--out", type=Path, default=S4_OUT_DIR)
    args = ap.parse_args()
    queue = build_queue(args.out)
    (args.out / "curation_queue.jsonl").write_text(
        "\n".join(json.dumps(q, ensure_ascii=False) for q in queue) + "\n", encoding="utf-8")
    (args.out / "curation.md").write_text(render(queue), encoding="utf-8")
    from collections import Counter
    by = Counter(q["route"] for q in queue)
    tot = sum(q["n_windows"] for q in queue)
    print(f"[s4] curation queue: {len(queue)} spans, {tot} windows -> {args.out / 'curation.md'}")
    for route in (RELABEL, NEW_CLASS, COLLECT):
        rows = [q for q in queue if q["route"] == route]
        w = sum(q["n_windows"] for q in rows)
        err = sum(q["error_rate"] * q["n_windows"] for q in rows) / w if w else 0
        print(f"[s4]   {route:20} {by[route]:>3} spans, {w:>4} windows, mean error_rate {err:.2f}")


if __name__ == "__main__":
    main()
