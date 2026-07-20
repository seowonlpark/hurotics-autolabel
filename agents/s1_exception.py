"""S1 exception agent: triage the deterministic layer's exception queue.

The clean stage MEASURES and quarantines deterministically. This agent JUDGES the
result: for each exception, is it a KNOWN failure mode (already in DOMAIN_NOTES), a
NOVEL one worth surfacing, or one that needs a human before the pipeline can
proceed? It never touches data and never recomputes — it reads the run's exception
records + DOMAIN_NOTES and returns a verdict as text; this module's deterministic
wrapper writes the review. (Non-negotiable: code does the work, agents judge it.)

Queue = genuine exceptions only (whole-file quarantines, gyro axis anomalies, yaw
drift flags). Routine gyro-abstentions are designed-normal behaviour, so they are
summarized as context, not triaged item by item.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from agents.base import MODEL_CHEAP, AgentSpec
from stages.s2_ml.transform import SAGITTAL_DEG_AXIS_BY_VARIANT

REVIEW_FILENAME = "exceptions_review.jsonl"

# Deg axes any known hardware revision treats as sagittal — i.e. the only ones the
# feature path can read. Derived from the consumer, not restated, so a new variant
# mapping cannot make this stale (§6.2).
_SAGITTAL_CANDIDATE_AXES = frozenset(SAGITTAL_DEG_AXIS_BY_VARIANT.values())

# The raw->rev* feature path (`transform.py:raw_to_features`) loops L and R only.
_DATA_PATH_SIDES = frozenset({"L", "R"})

SYSTEM_PROMPT = (
    "You are the S1 exception triage agent for an IMU locomotion pipeline. The "
    "deterministic clean stage has already measured the data and flagged exceptions. "
    "Your job is to JUDGE them — not to recompute, not to open raw signals.\n\n"
    "Answer TWO independent questions per item. They are orthogonal: do not let one "
    "decide the other.\n\n"
    "1. `explained` — does DOMAIN NOTES account for this evidence?\n"
    "  - \"yes\": a finding covers it and the evidence agrees with that finding.\n"
    "  - \"no\": no finding covers it. Say what is unexplained. This is the case we "
    "most want surfaced.\n"
    "  - \"contradicts\": a finding covers it and the evidence DISAGREES with the "
    "finding. Name the section and the disagreement. This is also a find — the note "
    "may be wrong — but never act on it.\n\n"
    "2. `action` — is a person needed before this data can be used?\n"
    "  - \"none\": the pipeline already handles it end to end.\n"
    "  - \"human\": someone must act or decide (re-export, repair, amend a note).\n\n"
    "Do NOT infer `action` from `explained`. A documented cause can still need a "
    "person (a broken clock is §2.6-explained and still needs a re-export). An "
    "unexplained anomaly can be inert. Judge them separately; deterministic code "
    "collapses the pair into the final disposition.\n\n"
    "`action` evidence — read `handled` on the queue item:\n"
    "  Each item carries a machine-derived `handled` {value, why} computed from the "
    "actual downstream consumers, not from prose. It is the authority on whether the "
    "pipeline handles this item. DOMAIN NOTES states POLICY; `handled` states what the "
    "code does. A note saying an anomaly is 'recorded' or that features 'must consult' "
    "a flag is NOT evidence that anything consumes it. Where the two disagree, follow "
    "`handled` and set explained=\"contradicts\".\n\n"
    "Rules:\n"
    "  - Judge ONLY from the provided evidence and the DOMAIN NOTES. Do NOT open raw "
    "CSVs to 'eyeball' signals — the notes warn repeatedly that the eyeball is not "
    "truth and whole-file statistics mislead.\n"
    "  - `sections`: every DOMAIN NOTES section you relied on, e.g. [\"4.1b\"]. List "
    "all that apply, not just one. Required non-empty when explained is \"yes\" or "
    "\"contradicts\"; MUST be [] when explained is \"no\".\n"
    "  - `confidence` is confidence in THESE TWO FIELDS, not in the underlying cause. "
    "Being certain that something is unexplained is high confidence, however deep the "
    "mystery.\n"
    "  - One sentence of rationale per item. Be terse.\n\n"
    "Output ONLY a JSON array, one object per queue item, each exactly: "
    '{"ref": <the item ref>, "explained": "yes"|"no"|"contradicts", '
    '"action": "none"|"human", "sections": [<DOMAIN NOTES section strings>], '
    '"rationale": <one sentence>, "confidence": <number 0.0-1.0>}. '
    "No text outside the JSON array."
)

S1_EXCEPTION_AGENT = AgentSpec(
    name="s1_exception",
    system_prompt=SYSTEM_PROMPT,
    allowed_tools=["Read", "Grep"],
    model=MODEL_CHEAP,
    max_turns=15,
)


def _handled(value: bool, why: str) -> dict:
    return {"value": value, "why": why}


def _handled_quarantine() -> dict:
    """A quarantined file is EXCLUDED, which is not the same as repaired."""
    return _handled(False,
                    "the file is kept out of data/clean so downstream is safe, but the "
                    "pipeline cannot repair it — recovery needs a person (§2.6)")


def _handled_axis_anomaly(side: str, conflicts: list[str]) -> dict:
    """Is this permutation conflict on a channel the feature path actually reads?

    `transform.py` resolves the sagittal Deg axis per VARIANT, then checks it against
    this file's own record (`check_axis_trust`). A conflict on a channel it reads is
    now FATAL, not silent — but fatal is not handled: the file cannot be featurized
    until a person resolves it. A conflict on a channel it never reads stays inert.
    That distinction is code, not prose; the notes' "recorded, not reordered" says only
    that S1 did not mutate, never that a consumer honours the record.
    """
    if side not in _DATA_PATH_SIDES:
        return _handled(True,
                        f"the raw->rev* feature path reads L/R only, never {side}; no "
                        f"consumer reads this side's gyro axes")
    reachable = sorted(set(conflicts) & _SAGITTAL_CANDIDATE_AXES)
    if not reachable:
        return _handled(True,
                        f"conflict is on Deg {sorted(conflicts)}, which no known "
                        f"variant treats as sagittal, so the feature path never reads it")
    return _handled(False,
                    f"conflict touches Deg {reachable}, which the feature path reads "
                    f"as sagittal on some variant — transform.py's check_axis_trust "
                    f"hard-fails this file rather than reading the documented column, "
                    f"so nothing is corrupted, but nothing is featurized either until "
                    f"someone resolves the axis")


def _handled_drift(channel: str) -> dict:
    """No code consumes `drift_contaminated`; §4.2's "must consult" is policy only."""
    if channel.endswith("_Deg_Z"):
        return _handled(True,
                        "nothing in the pipeline reads the drift flag, but nothing "
                        "reads Deg_Z either — the feature path uses the sagittal Deg "
                        "axis and its gyro rate, so this flag is inert, not honoured")
    return _handled(False,
                    f"{channel} is not the yaw-like Deg_Z axis §4.2 predicts, and no "
                    f"code consumes the drift flag, so nothing would exclude it")


def build_queue(clean_run_dir: Path) -> tuple[list[dict], dict]:
    """Extract genuine exceptions from a clean run's artifacts.

    Routine gyro-abstentions are counted, not queued — they are designed-normal
    (static files falling back to the documented convention), not exceptions.
    """
    queue: list[dict] = []

    q_path = clean_run_dir / "quarantine.jsonl"
    if q_path.exists():
        for line in q_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            queue.append({
                "ref": f"quarantine:{Path(r['file']).name}",
                "type": "whole_file_quarantine",
                "file": r["file"],
                "reason": r["reason"],
                "handled": _handled_quarantine(),
            })

    abstain_files = 0
    o_path = clean_run_dir / "observations.jsonl"
    if o_path.exists():
        for line in o_path.read_text(encoding="utf-8").splitlines():
            o = json.loads(line)
            ct = o["channel_trust"]
            if o.get("kind") == "channel_trust_abstained":
                abstain_files += 1
            for side in ct.get("anomalies", []):
                rec = ct["sides"][side]
                queue.append({
                    "ref": f"anomaly:{Path(o['path']).name}:{side}",
                    "type": "gyro_axis_anomaly",
                    "file": o["path"],
                    "side": side,
                    # `conflicts_with_documented` is WHY this is an anomaly — without
                    # it the agent is judging a permutation with no stated grievance.
                    "detail": {k: rec[k] for k in (
                        "gyro_axis_by_deg_axis", "conflicts_with_documented",
                        "is_bijection", "unit", "r")},
                    "handled": _handled_axis_anomaly(
                        side, rec["conflicts_with_documented"]),
                })
            for ch in o.get("drift_contaminated", []):
                queue.append({
                    "ref": f"drift:{Path(o['path']).name}:{ch}",
                    "type": "yaw_drift",
                    "file": o["path"],
                    "channel": ch,
                    "handled": _handled_drift(ch),
                })

    summary = {
        "n_quarantine": sum(1 for x in queue if x["type"] == "whole_file_quarantine"),
        "n_axis_anomaly": sum(1 for x in queue if x["type"] == "gyro_axis_anomaly"),
        "n_yaw_drift": sum(1 for x in queue if x["type"] == "yaw_drift"),
        "n_routine_abstain_files": abstain_files,
    }
    return queue, summary


def build_prompt(queue: list[dict], summary: dict) -> str:
    return (
        "Triage this S1 exception queue.\n\n"
        f"Context: the clean stage also produced {summary['n_routine_abstain_files']} "
        "files with routine gyro-abstentions (static files falling back to the "
        "documented convention). These are designed-normal and are NOT in the queue; "
        "flag only if that COUNT itself looks anomalous for this corpus.\n\n"
        "QUEUE — return exactly one verdict per item, matched by `ref`:\n"
        f"{json.dumps(queue, indent=2, ensure_ascii=False)}\n"
    )


def parse_review(final_text: str) -> list[dict] | None:
    """Extract the JSON array from the agent's final text; tolerant of fences/prose."""
    if not final_text:
        return None
    m = re.search(r"\[.*\]", final_text, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, list) else None


_REVIEW_KEYS = ("explained", "action", "sections", "rationale", "confidence")


def collapse(explained: str | None, action: str | None) -> tuple[str, str]:
    """Collapse the agent's two orthogonal judgements into (disposition, action).

    The agent answers two questions that live on different axes; this is the only place
    the pipeline decides how they combine, so two runs cannot disposition the same item
    differently. `explained` names the bucket, `action` rides along and is never lost:

        explained    action        disposition
        yes          none       -> known_expected
        yes          human      -> needs_human
        no           (kept)     -> novel
        contradicts  -> human   -> novel

    Unexplained wins the label because "surface it" is the point of the bucket, and it
    costs nothing: `action` still carries whether the pipeline is blocked, so a novel
    item that also needs a person is not demoted to a queue of routine repairs. A
    contradicted note is a find too — but it is never acted on (§ DOMAIN_NOTES header),
    so its action is forced, not read.
    """
    if explained == "contradicts":
        return "novel", "human"
    if explained == "no":
        return "novel", action if action in ("none", "human") else "human"
    if explained == "yes" and action in ("none", "human"):
        return ("known_expected" if action == "none" else "needs_human"), action
    # Unrecognized pair: judge nothing, escalate. Same conservatism as a parse failure.
    return "needs_human", "human"


def write_review(out_dir: Path, queue: list[dict], decisions: list[dict] | None,
                 final_text: str) -> Path:
    """Write one review row per queue item, agent verdict merged in.

    The agent's two judgements are recorded as given; `disposition` is DERIVED here by
    `collapse`, never taken from the model — the taxonomy is the pipeline's, not a thing
    each run re-decides. A queue item with no parseable verdict is conservatively marked
    needs_human, so a parsing failure never silently drops an exception. Raw text is kept
    for audit.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    by_ref = {d.get("ref"): d for d in decisions} if decisions else {}
    out = out_dir / REVIEW_FILENAME
    with out.open("w", encoding="utf-8") as fh:
        for item in queue:
            d = by_ref.get(item["ref"])
            if d is None:
                review = {
                    "explained": None, "action": "human", "sections": [],
                    "disposition": "needs_human",
                    "rationale": "no parseable agent verdict for this item",
                    "confidence": 0.0, "unparsed": True,
                }
            else:
                review = {k: d.get(k) for k in _REVIEW_KEYS}
                review["disposition"], review["action"] = collapse(
                    review.get("explained"), review.get("action"))
            fh.write(json.dumps({**item, "review": review}, ensure_ascii=False) + "\n")
    if decisions is None:
        (out_dir / "exceptions_review_raw.txt").write_text(final_text, encoding="utf-8")
    return out
