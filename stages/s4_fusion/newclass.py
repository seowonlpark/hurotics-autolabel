# S4 new-class discovery core: assemble the evidence for classes the {stand, walk} taxonomy
# misses, WITHOUT deciding anything. the curation director (curate.py) already routed the
# NEW_CLASS windows -- STAND-labeled posture that is MOVING (grav_stab < 0.5) yet not clean
# gait: a ramp, a weight-shift, a repositioning, maybe a sit. this stage does not name those.
# it hands the agent a per-span physics profile + the S3 figure, the agent PROPOSES a class,
# and a deterministic gate (newclass agent) validates cluster mass before anything reaches a
# human. governed discovery (DOMAIN_NOTES Section 11.2): code never adds a class, never edits a label
# -- it can only assemble evidence and route a proposal to needs_human. reads artifacts only.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from stages.s2_ml.dataset import STAND
from stages.s3_physics.run import ANCHORS_CSV, PLOTS_SUBDIR, S3_OUT_DIR
from stages.s4_fusion.curate import NEW_CLASS, route_window, to_spans
from stages.s4_fusion.run import FUSED_CSV, JOIN_KEYS, S4_OUT_DIR

# the physics descriptors that characterise a candidate span -- what the body is doing that the
# 2-class taxonomy has no word for. means over the span; the agent reads these, never raw signal.
PROFILE_COLS = ["grav_stab", "antiphase", "periodicity", "interleg_offset", "swap_count"]

# cluster mass a proposed class must clear to be "supported" (still needs_human either way). a
# class is structure that RECURS: a one-off span is noise or a single mislabel, not a category.
MIN_SPANS = 4 # at least this many spans grouped under the proposed class
MIN_REVS = 2  # spanning at least this many subjects/days -- not one file's artifact

CANDIDATES_JSON = "new_class_candidates.json" # the evidence bundle the agent reads
CANDIDATES_MD = "new_class_candidates.md"


# the NEW_CLASS-routed windows, joined to their full physics profile. reuses the curation
# router so "what is a new-class candidate" is defined in exactly one place (curate.py).
def candidate_windows(s4_dir: Path = S4_OUT_DIR, s3_dir: Path = S3_OUT_DIR) -> pd.DataFrame:
    fused = pd.read_csv(s4_dir / FUSED_CSV)
    cols = JOIN_KEYS + [c for c in PROFILE_COLS if c != "swap_count"] + ["swap_count"]
    anchors = pd.read_csv(s3_dir / ANCHORS_CSV)[list(dict.fromkeys(cols))]
    df = fused.merge(anchors, on=JOIN_KEYS, how="inner")
    df["route"] = [route_window(int(t), int(s2), s3, c, g) for t, s2, s3, c, g in
                   zip(df["true"], df["s2_pred"], df["s3"], df["confidence"], df["grav_stab"])]
    return df[df["route"] == NEW_CLASS].copy()


# group the candidate windows into spans and attach each span's physics profile + S3 figure.
# to_spans (curate.py) is the shared span grouping; here we enrich each span with the aggregate
# descriptors the agent needs to tell one kind of "moving stand" from another.
def candidate_spans(df: pd.DataFrame) -> list[dict]:
    spans = to_spans(df)
    out = []
    for s in spans:
        win = df[(df["rev"] == s["rev"]) & (df["trial"] == s["trial"])
                 & (df["segment"] == s["segment"])
                 & (df["t_start_ms"] >= s["t_start_ms"]) & (df["t_start_ms"] <= s["t_end_ms"])]
        out.append({
            "rev": s["rev"], "trial": s["trial"], "segment": s["segment"],
            "t_start_s": round(s["t_start_ms"] / 1000, 1),
            "t_end_s": round(s["t_end_ms"] / 1000, 1),
            "n_windows": s["n_windows"],
            "human_label": "STAND" if s["label"] == STAND else "WALK",
            "profile": {c: round(float(win[c].mean()), 3) for c in PROFILE_COLS},
            "swap_verdicts": {k: int(v) for k, v in win["s3"].value_counts().items()},
            "plot": f"{PLOTS_SUBDIR}/trial_{s['rev']}_t{s['trial']}.png",
        })
    # busiest spans first: more windows = more evidence for the agent to characterise
    return sorted(out, key=lambda x: x["n_windows"], reverse=True)


# the corpus-level signature: how much candidate mass there is and its central physics. gives
# the agent (and the reader) the baseline a proposed class must stand out against.
def corpus_signature(df: pd.DataFrame, spans: list[dict]) -> dict:
    return {
        "n_spans": len(spans),
        "n_windows": int(len(df)),
        "n_revs": int(df["rev"].nunique()),
        "revs": sorted(df["rev"].unique().tolist()),
        "mean_profile": {c: round(float(df[c].mean()), 3) for c in PROFILE_COLS},
        "min_spans_for_support": MIN_SPANS,
        "min_revs_for_support": MIN_REVS,
    }


# validate ONE agent proposal against the candidate spans (used by the newclass agent). two
# gates, both deterministic: PROVENANCE -- every span_ref must overlap a real candidate span;
# CLUSTER MASS -- the matched spans must recur (>= MIN_SPANS spans across >= MIN_REVS revs). a
# proposal that clears both is "supported"; otherwise "insufficient_evidence". EITHER WAY it is
# needs_human: code proposes a class, it never adds one (Section 11.2, non-negotiable #1).
def validate_proposal(proposal: dict, spans: list[dict]) -> dict:
    reasons: list[str] = []
    refs = proposal.get("span_refs")
    if not proposal.get("class_name") or not proposal.get("statement"):
        reasons.append("no class_name/statement")
    if not isinstance(refs, list) or not refs:
        reasons.append("no span_refs")
        refs = []

    matched = [r for r in refs if _matches_a_span(r, spans)]
    n_unmatched = len(refs) - len(matched)
    if n_unmatched:
        reasons.append(f"{n_unmatched} span_refs match no candidate span")

    revs = {r.get("rev") for r in matched}
    supported = len(matched) >= MIN_SPANS and len(revs) >= MIN_REVS
    if refs and not supported:
        reasons.append(f"insufficient cluster mass ({len(matched)} spans / {len(revs)} revs; "
                       f"need >= {MIN_SPANS} / {MIN_REVS})")

    return {**proposal,
            "validation": {
                "ok": not reasons, # provenance clean AND cluster mass met
                "support": "supported" if supported else "insufficient_evidence",
                "n_matched_spans": len(matched), "n_revs": len(revs),
                "reasons": reasons,
            },
            "needs_human": True} # always: a human decides whether a class exists, never code


# does a proposal's span_ref {rev,trial,t_start_s,t_end_s} overlap a real candidate span?
def _matches_a_span(ref: dict, spans: list[dict]) -> bool:
    try:
        rev, trial = ref["rev"], int(ref["trial"])
        a, b = float(ref["t_start_s"]), float(ref["t_end_s"])
    except (KeyError, TypeError, ValueError):
        return False
    for s in spans:
        if s["rev"] == rev and s["trial"] == trial \
                and a <= s["t_end_s"] and b >= s["t_start_s"]: # time overlap
            return True
    return False


# render the evidence bundle as markdown for a human skim
def render(sig: dict, spans: list[dict]) -> str:
    L = ["# S4 new-class candidates - structure the stand/walk taxonomy misses", "",
         f"{sig['n_spans']} spans / {sig['n_windows']} windows across {sig['n_revs']} revs "
         f"({sig['revs']}). These are STAND-labeled windows the physics reads as MOVING "
         f"(grav_stab < 0.5) but not clean gait - ramps, weight-shifts, repositioning, maybe "
         f"sitting. Corpus mean profile: {sig['mean_profile']}.", "",
         f"A proposed class is *supported* only if it recurs: >= {MIN_SPANS} spans across "
         f">= {MIN_REVS} revs. Every proposal goes to a human - code never adds a class.", "",
         "| rev/trial | t_start-t_end (s) | label | windows | grav_stab | antiphase | "
         "periodicity | interleg_offset |", "|---|---|---|---|---|---|---|---|"]
    for s in spans[:20]:
        p = s["profile"]
        L.append(f"| {s['rev']}_t{s['trial']} | {s['t_start_s']}-{s['t_end_s']} | "
                 f"{s['human_label']} | {s['n_windows']} | {p['grav_stab']} | {p['antiphase']} | "
                 f"{p['periodicity']} | {p['interleg_offset']} |")
    return "\n".join(L) + "\n"


# build the evidence bundle (candidates + signature) and write it. returns the bundle dict.
def build_bundle(s4_dir: Path = S4_OUT_DIR, s3_dir: Path = S3_OUT_DIR) -> dict:
    df = candidate_windows(s4_dir, s3_dir)
    spans = candidate_spans(df)
    sig = corpus_signature(df, spans)
    bundle = {"corpus_signature": sig, "spans": spans}
    (s4_dir / CANDIDATES_JSON).write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    (s4_dir / CANDIDATES_MD).write_text(render(sig, spans), encoding="utf-8")
    return bundle


def main() -> None:
    ap = argparse.ArgumentParser(description="Assemble the S4 new-class discovery evidence bundle.")
    ap.add_argument("--out", type=Path, default=S4_OUT_DIR)
    args = ap.parse_args()
    bundle = build_bundle(args.out)
    sig = bundle["corpus_signature"]
    print(f"[s4] new-class candidates: {sig['n_spans']} spans, {sig['n_windows']} windows across "
          f"{sig['n_revs']} revs -> {args.out / CANDIDATES_MD}")
    print(f"[s4] corpus mean profile: {sig['mean_profile']}")


if __name__ == "__main__":
    main()
