# S3 label review: judge what label_audit flagged, recompute nothing

from __future__ import annotations

import json
from pathlib import Path

from agents.base import MODEL_SMART, AgentSpec, extract_json_array
from stages.s2_ml.dataset import EXCLUDED_TRIALS

REVIEW_FILENAME = "label_review.jsonl"

AUDIT_JSON = "label_audit.json"
AUDIT_WINDOWS = "label_audit_windows.jsonl"

# closed vocabulary
CAUSES = {
    "swapped_channel": (
        "the contradiction is uniform across the recording: a channel is swapped or "
        "mismapped for the whole trial"),
    "misaligned_label_track": (
        "the contradiction is confined to a stretch: the label track is offset against "
        "the signal, or spliced from the wrong recording"),
    "divergent_convention": (
        "the trial annotates the ambiguity band to a self-consistent minority convention "
        "— not broken data (§label_audit: do NOT exclude)"),
    "label_suspect": (
        "the annotation looks wrong here, but the pattern does not identify which of the "
        "above it is"),
    "physics_wrong": (
        "the swap rule is the unreliable party on this trial — untrusted rest offset, "
        "gait too slow for the span, or an amplitude regime the rule does not cover"),
    "needs_trace": (
        "the numbers do not separate the causes; a person must open the raw trace"),
}

SYSTEM_PROMPT = (
    "You are the S3 label review agent for an IMU locomotion pipeline. A deterministic "
    "audit has already scored every annotated trial against the swap rule — a physics "
    "rule with no trained parameter that never sees a label — and flagged the trials "
    "whose own annotations it disputes. Your job is to JUDGE those flags: name the "
    "likely CAUSE and say whether a person is needed. Do not recompute anything and do "
    "not open raw CSVs.\n\n"
    "Answer TWO independent questions per item. They are orthogonal: do not let one "
    "decide the other.\n\n"
    "1. `cause` — one of:\n"
    + "".join(f"  - \"{k}\": {v}\n" for k, v in CAUSES.items())
    + "\n2. `action` — is a person needed before this trial can be used as it is?\n"
    "  - \"none\": the pipeline already handles it end to end.\n"
    "  - \"human\": someone must decide (exclude it, re-annotate it, or accept it and "
    "report the ceiling).\n\n"
    "Do NOT infer `action` from `cause`. A trial can have an identified cause and still "
    "need nobody (it is already excluded), and one with no identified cause can still be "
    "inert.\n\n"
    "`action` evidence — read `handled` on the queue item:\n"
    "  Each item carries a machine-derived `handled` {value, why} computed from what the "
    "code actually does with this trial, not from prose. Reports state POLICY; `handled` "
    "states BEHAVIOUR. A report saying a trial should be excluded is not evidence that "
    "anything excludes it. Where the two disagree, follow `handled`.\n\n"
    "How to tell the causes apart — this is the judgement being asked for:\n"
    "  - `disagree_frac` is outright contradiction; `band_walk_frac` is convention inside "
    "a range where BOTH human labels genuinely occur. A trial flagged only on the second "
    "is NOT broken data, and recommending exclusion for it is the specific error this "
    "queue exists to prevent.\n"
    "  - `contradiction_spread` is machine-derived: the share of the trial's scored span "
    "that its contradicting windows cover. Near 1.0 with a high `disagree_frac` points at "
    "`swapped_channel` — wrong everywhere. A small span inside a long trial points at "
    "`misaligned_label_track` — wrong in a stretch. Say which one the number supports.\n"
    "  - `rest_trusted: false` means the interleg zero is a whole-recording median rather "
    "than a measured standing posture, so every physics verdict on that trial is SOFT. "
    "That is evidence for `physics_wrong`, and it weakens any other cause you assign.\n"
    "  - `windows` below ~20 makes a fraction unreliable whatever it reads.\n\n"
    "Rules:\n"
    "  - Judge ONLY from the provided evidence and the DOMAIN NOTES. You cannot see the "
    "raw interleg trace; `inspect_window` is what shows it and only a person runs it. If "
    "the numbers do not separate the causes, answer \"needs_trace\". That is a correct "
    "answer, not a failure — a confident cause the evidence does not support is worse "
    "than an admitted unknown.\n"
    "  - `sections`: every DOMAIN NOTES section you relied on, e.g. [\"10.2\"]. Required "
    "non-empty unless cause is \"needs_trace\".\n"
    "  - `confidence` is confidence in THESE TWO FIELDS, not in the underlying cause. "
    "Being certain the evidence is insufficient is high confidence.\n"
    "  - One sentence of rationale per item. Name the number that decided it. Be terse.\n\n"
    "Output ONLY a JSON array, one object per queue item, each exactly: "
    '{"ref": <the item ref>, "cause": <one of the causes above>, '
    '"action": "none"|"human", "sections": [<DOMAIN NOTES section strings>], '
    '"rationale": <one sentence naming the deciding number>, '
    '"confidence": <number 0.0-1.0>}. '
    "No text outside the JSON array."
)

# sonnet not haiku: reads a distribution, not a lookup; queue is single-digit
S3_LABEL_REVIEW_AGENT = AgentSpec(
    name="s3_label_review",
    system_prompt=SYSTEM_PROMPT,
    allowed_tools=["Read", "Grep"],
    model=MODEL_SMART,
    max_turns=15,
)


def _handled(value: bool, why: str) -> dict:
    return {"value": value, "why": why}


# flagged and excluded look identical here; EXCLUDED_TRIALS is the authority
def _handled_broken(rev: str, trial: int) -> dict:
    if (rev, trial) in EXCLUDED_TRIALS:
        return _handled(True,
                        "already in dataset.EXCLUDED_TRIALS, so load_dataset drops it from "
                        "training and from every evaluation; it still appears in this "
                        "audit because the audit deliberately loads the raw corpus")
    return _handled(False,
                    "nothing excludes it — dataset.EXCLUDED_TRIALS does not list it, so "
                    "load_dataset feeds it to training and scores against it, and only a "
                    "person editing that set by hand changes that")


# nothing consumes band_flag, and the band policy it argues about ships OFF
def _handled_divergent(rev: str, trial: int) -> dict:
    if (rev, trial) in EXCLUDED_TRIALS:
        return _handled(True,
                        "the trial is in dataset.EXCLUDED_TRIALS for an unrelated reason, "
                        "so its annotation convention reaches neither training nor "
                        "evaluation")
    return _handled(False,
                    "no code reads band_flag, and label.BAND_ABSTAINS is False, so the "
                    "serve path commits to this trial's band rows like any other — the "
                    "divergent convention enters both training and the accuracy it is "
                    "scored against, and nothing anywhere subtracts it")


# separates swapped_channel (wrong everywhere) from misaligned_label_track (wrong in a stretch)
def _spread(rec: dict) -> float | None:
    span = (rec.get("t_last_s") or 0) - (rec.get("t_first_s") or 0)
    bad = (rec.get("contradict_t_last_s") or 0) - (rec.get("contradict_t_first_s") or 0)
    if not span or span <= 0:
        return None
    return round(min(bad / span, 1.0), 4)


# audit artifacts -> review queue
def build_queue(audit_dir: Path) -> tuple[list[dict], dict]:
    audit = json.loads((audit_dir / AUDIT_JSON).read_text(encoding="utf-8"))
    trials = audit["trials"]

    noms_path = audit_dir / AUDIT_WINDOWS
    noms: dict[tuple[str, int], list[dict]] = {}
    if noms_path.exists():
        for line in noms_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            n = json.loads(line)
            noms.setdefault((n["rev"], int(n["trial"])), []).append(n)

    queue: list[dict] = []
    for r in trials:
        rev, trial = r["rev"], int(r["trial"])

        if r.get("flag"):
            windows = noms.get((rev, trial), [])
            queue.append({
                "ref": f"broken:{rev}:t{trial}",
                "type": "outright_contradiction",
                "rev": rev, "trial": trial,
                "evidence": {
                    "windows_decided": r["windows"],
                    "windows_contradicting": r["disagree"],
                    "disagree_frac": r["disagree_frac"],
                    "physics_abstain_frac": r["abstain_frac"],
                    "rest_trusted": r["rest_trusted"],
                    "scored_span_s": [r.get("t_first_s"), r.get("t_last_s")],
                    "contradicting_span_s": [r.get("contradict_t_first_s"),
                                             r.get("contradict_t_last_s")],
                    "contradiction_spread": _spread(r),
                },
                "nominated_windows": windows,
                "n_nominations_dropped": audit.get("n_nominations_dropped", 0),
                "handled": _handled_broken(rev, trial),
            })

        if r.get("band_flag"):
            queue.append({
                "ref": f"convention:{rev}:t{trial}",
                "type": "divergent_band_convention",
                "rev": rev, "trial": trial,
                "evidence": {
                    "band_deg": audit["band"],
                    "band_windows": r.get("band_windows"),
                    "band_walk_frac": r.get("band_walk_frac"),
                    "corpus_band_walk_frac": audit["band_corpus_walk_frac"],
                    "delta_vs_corpus": (
                        round(r["band_walk_frac"] - audit["band_corpus_walk_frac"], 4)
                        if r.get("band_walk_frac") is not None else None),
                    "binomial_p": r.get("band_p"),
                    "disagree_frac": r["disagree_frac"],
                    "rest_trusted": r["rest_trusted"],
                },
                "handled": _handled_divergent(rev, trial),
            })

    summary = {
        "n_broken": sum(1 for x in queue if x["type"] == "outright_contradiction"),
        "n_divergent": sum(1 for x in queue if x["type"] == "divergent_band_convention"),
        "n_trials_scored": len(trials),
        "band_deg": audit["band"],
        "corpus_band_walk_frac": audit["band_corpus_walk_frac"],
        "max_disagree": audit["max_disagree"],
    }
    return queue, summary


def build_prompt(queue: list[dict], summary: dict) -> str:
    return (
        "Judge this label-audit queue.\n\n"
        f"Context: {summary['n_trials_scored']} annotated trials were scored against the "
        f"swap rule. {summary['n_broken']} exceeded the outright-contradiction threshold "
        f"of {summary['max_disagree']:.2f}; {summary['n_divergent']} annotate the "
        f"ambiguity band ({summary['band_deg'][0]:.2f}-{summary['band_deg'][1]:.2f} deg "
        f"interleg swing) significantly unlike the corpus, which sits at "
        f"{summary['corpus_band_walk_frac']:.3f} walk. Trials absent from this queue were "
        f"scored and passed both detectors.\n\n"
        "QUEUE — return exactly one verdict per item, matched by `ref`:\n"
        f"{json.dumps(queue, indent=2, ensure_ascii=False, default=float)}\n"
    )


# thin alias over base.extract_json_array- one tolerant parse for every agent
def parse_review(final_text: str) -> list[dict] | None:
    return extract_json_array(final_text)


_REVIEW_KEYS = ("cause", "action", "sections", "rationale", "confidence")

# agent names the cause; this table decides what the pipeline calls it
_DISPOSITION = {
    "swapped_channel": "mislabel_candidate",
    "misaligned_label_track": "mislabel_candidate",
    "label_suspect": "mislabel_candidate",
    "divergent_convention": "annotation_policy",
    "physics_wrong": "physics_limitation",
    "needs_trace": "needs_human",
}

# agent may nominate a label change, never make one
_GROUND_TRUTH_CAUSES = frozenset(
    {"swapped_channel", "misaligned_label_track", "label_suspect"})


# combine two
def collapse(cause: str | None, action: str | None,
             handled: bool = False) -> tuple[str, str]:
    if cause in _GROUND_TRUTH_CAUSES:
        if not handled:
            return _DISPOSITION[cause], "human"
        return _DISPOSITION[cause], action if action in ("none", "human") else "none"
    if cause == "needs_trace":
        return "needs_human", "human"
    if cause in _DISPOSITION and action in ("none", "human"):
        return _DISPOSITION[cause], action
    # unrecognized pair: judge nothing, escalate; same conservatism as a parse failure
    return "needs_human", "human"


# final
def write_review(out_dir: Path, queue: list[dict], decisions: list[dict] | None,
                 final_text: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_ref = {d.get("ref"): d for d in decisions} if decisions else {}
    out = out_dir / REVIEW_FILENAME
    with out.open("w", encoding="utf-8") as fh:
        for item in queue:
            d = by_ref.get(item["ref"])
            if d is None:
                review = {
                    "cause": None, "action_agent": None, "action": "human",
                    "sections": [], "disposition": "needs_human",
                    "rationale": "no parseable agent verdict for this item",
                    "confidence": 0.0, "unparsed": True,
                }
            else:
                review = {k: d.get(k) for k in _REVIEW_KEYS}
                review["action_agent"] = review.get("action")
                review["disposition"], review["action"] = collapse(
                    review.get("cause"), review.get("action"),
                    bool(item["handled"]["value"]))
            fh.write(json.dumps({**item, "review": review},
                                ensure_ascii=False, default=float) + "\n")
    (out_dir / "label_review_raw.txt").write_text(final_text, encoding="utf-8")
    return out
