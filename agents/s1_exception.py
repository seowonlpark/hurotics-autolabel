# S1 exception agent: triage the clean stage's exception queue (read-only)
# clean measures/quarantines; this agent judges each exception -- known, novel, or
# needs-human -- from the records + DOMAIN_NOTES. code writes the review. see README.

from __future__ import annotations

import json
from pathlib import Path

from agents.base import MODEL_CHEAP, AgentSpec, extract_json_array
from stages.s2_ml.transform import SAGITTAL_DEG_AXIS

REVIEW_FILENAME = "exceptions_review.jsonl"

# the only Deg axis the feature path reads (sagittal/Y, Section 6.3); derived from the consumer
# so it can't go stale -- a conflict on any other axis never reaches the feature path
_SAGITTAL_CANDIDATE_AXES = frozenset({SAGITTAL_DEG_AXIS})

# the raw->rev* feature path (transform.raw_to_features) loops L and R only
_DATA_PATH_SIDES = frozenset({"L", "R"})

SYSTEM_PROMPT = (
    "You are the S1 exception triage agent for an IMU locomotion pipeline. The "
    "deterministic clean stage has already measured the data and flagged exceptions. "
    "JUDGE them - do not recompute or open raw CSVs to 'eyeball' signals; the notes "
    "warn the eyeball is not truth and whole-file statistics mislead.\n\n"
    "Answer TWO orthogonal questions per item. Judge them separately; do not infer one "
    "from the other. Deterministic code collapses the pair into the final disposition."
    "\n\n"
    "1. `explained` - does DOMAIN NOTES account for this evidence?\n"
    "  - \"yes\": a finding covers it and the evidence agrees.\n"
    "  - \"no\": no finding covers it - say what is unexplained. The case we most want "
    "surfaced.\n"
    "  - \"contradicts\": a finding covers it but the MEASURED EVIDENCE disagrees with "
    "what that finding says the signal should look like. Name the section and the "
    "disagreement - the note may be wrong. Never act on it.\n\n"
    "2. `action` - is a person needed before this data can be used?\n"
    "  - \"none\": the pipeline already handles it end to end.\n"
    "  - \"human\": someone must act or decide (re-export, repair, amend a note).\n"
    "  A documented cause can still need a person (a broken clock is explained and "
    "still needs re-export); an unexplained anomaly can be inert.\n\n"
    "Authority for `action`: each item carries a machine-derived `handled` {value, "
    "why}, computed from actual downstream consumers. DOMAIN NOTES states POLICY; "
    "`handled` states what the code does. A note that an anomaly is 'recorded' or that "
    "features 'must consult' a flag is NOT evidence anything consumes it. On "
    "disagreement, `handled` wins - but it settles `action` ONLY. A stale claim about "
    "handling does not make the finding's account of the signal wrong, so it is not "
    "\"contradicts\"; say so in the rationale and leave `explained` on the evidence."
    "\n\n"
    "Fields:\n"
    "  - `sections`: every DOMAIN NOTES section relied on, e.g. [\"4.1b\"] - all that "
    "apply, not just one. Non-empty when explained is \"yes\"/\"contradicts\"; [] when "
    "\"no\".\n"
    "  - `confidence`: confidence in these two fields, NOT the underlying cause - "
    "certainty that something is unexplained is high confidence.\n"
    "  - One terse sentence of rationale.\n\n"
    "Output ONLY a JSON array, one object per item, each exactly: "
    '{"ref": <ref>, "explained": "yes"|"no"|"contradicts", "action": "none"|"human", '
    '"sections": [<section strings>], "rationale": <one sentence>, '
    '"confidence": <0.0-1.0>}. No text outside the JSON array.'
)

S1_EXCEPTION_AGENT = AgentSpec(
    name="s1_exception",
    system_prompt=SYSTEM_PROMPT,
    allowed_tools=["Read", "Grep"],
    model=MODEL_CHEAP,
    max_turns=15,
    # schema/units/quarantine/drift/axis + label contamination + eyeball-not-truth
    domain_sections=("1", "2", "3", "4", "5", "6", "11"),
)


# handled record: what the code actually does with this exception
def _handled(value: bool, why: str) -> dict:
    return {"value": value, "why": why}


# handled helper: a quarantined file is excluded, not repaired
def _handled_quarantine() -> dict:
    return _handled(False,
                    "the file is kept out of data/clean so downstream is safe, but the "
                    "pipeline cannot repair it - recovery needs a person (Section 2.6)")


# handled helper: is this axis conflict on a channel the feature path actually reads?
# a conflict on a read channel is fatal (not handled -- needs a person); otherwise inert
def _handled_axis_anomaly(side: str, conflicts: list[str]) -> dict:
    if side not in _DATA_PATH_SIDES:
        return _handled(True,
                        f"the raw->rev* feature path reads L/R only, never {side}; no "
                        f"consumer reads this side's gyro axes")
    reachable = sorted(set(conflicts) & _SAGITTAL_CANDIDATE_AXES)
    if not reachable:
        return _handled(True,
                        f"conflict is on Deg {sorted(conflicts)}, not the sagittal Y "
                        f"plane the feature path reads (Section 6.3), so no consumer reads it")
    return _handled(False,
                    f"conflict touches Deg {reachable}, the sagittal Y plane the feature "
                    f"path reads - transform.py's check_axis_trust hard-fails this file "
                    f"rather than reading a corrupted axis, so nothing is corrupted, but "
                    f"nothing is featurized either until someone resolves the axis")


# handled helper: no code consumes drift_contaminated; Section 4.2's "must consult" is policy only
def _handled_drift(channel: str) -> dict:
    if channel.endswith("_Deg_Z"):
        return _handled(True,
                        "nothing in the pipeline reads the drift flag, but nothing "
                        "reads Deg_Z either - the feature path uses the sagittal Deg "
                        "axis and its gyro rate, so this flag is inert, not honoured")
    return _handled(False,
                    f"{channel} is not the yaw-like Deg_Z axis Section 4.2 predicts, and no "
                    f"code consumes the drift flag, so nothing would exclude it")


# extract genuine exceptions from a clean run's artifacts
# routine gyro-abstentions are counted, not queued -- they're designed-normal
def build_queue(clean_run_dir: Path) -> tuple[list[dict], dict]:
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
                    # conflicts_with_documented is WHY this is an anomaly
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


# assemble the triage prompt: routine-abstain count as context + the queue
def build_prompt(queue: list[dict], summary: dict) -> str:
    return (
        "Triage this S1 exception queue.\n\n"
        f"Context: the clean stage also produced {summary['n_routine_abstain_files']} "
        "files with routine gyro-abstentions (static files falling back to the "
        "documented convention). These are designed-normal and are NOT in the queue; "
        "flag only if that COUNT itself looks anomalous for this corpus.\n\n"
        "QUEUE - return exactly one verdict per item, matched by `ref`:\n"
        f"{json.dumps(queue, indent=2, ensure_ascii=False)}\n"
    )


# pull the JSON array out of the agent's final text; tolerant of fences/prose
def parse_review(final_text: str) -> list[dict] | None:
    return extract_json_array(final_text)


_REVIEW_KEYS = ("explained", "action", "sections", "rationale", "confidence")


# collapse the agent's two orthogonal judgements into (disposition, action) -- the only
# place they combine, so two runs can't disposition the same item differently:
#   yes/none -> known_expected;  yes/human -> needs_human;
#   no -> novel (action rides along);  contradicts -> novel, action forced to human
def collapse(explained: str | None, action: str | None) -> tuple[str, str]:
    if explained == "contradicts":
        return "novel", "human"
    if explained == "no":
        return "novel", action if action in ("none", "human") else "human"
    if explained == "yes" and action in ("none", "human"):
        return ("known_expected" if action == "none" else "needs_human"), action
    # unrecognized pair: judge nothing, escalate
    return "needs_human", "human"


# write one review row per queue item, agent verdict merged in; disposition is derived
# here by collapse(), never taken from the model. no verdict => needs_human, never dropped
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
