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

REVIEW_FILENAME = "exceptions_review.jsonl"

SYSTEM_PROMPT = (
    "You are the S1 exception triage agent for an IMU locomotion pipeline. The "
    "deterministic clean stage has already measured the data and flagged exceptions. "
    "Your job is to JUDGE them — not to recompute, not to open raw signals.\n\n"
    "Give each queue item exactly one disposition, chosen by OUTCOME:\n"
    "  - known_expected: explained by a DOMAIN NOTES finding AND already handled by "
    "the pipeline, so NO human action is needed (e.g. a recorded gyro axis anomaly, a "
    "per-file yaw-drift flag that feature code will honour). Cite the section.\n"
    "  - needs_human: a person must act or decide before this data can be used — even "
    "if the cause is documented. A quarantined file the pipeline cannot fix (e.g. a "
    "broken clock that must be re-exported) is needs_human, NOT known_expected. Cite "
    "the section if the cause is known.\n"
    "  - novel: NOT explained by any DOMAIN NOTES finding — say what is unexplained. "
    "This is the case we most want surfaced.\n\n"
    "Rules:\n"
    "  - Judge ONLY from the provided evidence and the DOMAIN NOTES. Do NOT open raw "
    "CSVs to 'eyeball' signals — the notes warn repeatedly that the eyeball is not "
    "truth and whole-file statistics mislead.\n"
    "  - If evidence contradicts a DOMAIN NOTES finding, do not act on it: mark "
    "needs_human and say so.\n"
    "  - One sentence of rationale per item. Be terse.\n\n"
    "Output ONLY a JSON array, one object per queue item, each exactly: "
    '{"ref": <the item ref>, "disposition": "known_expected"|"novel"|"needs_human", '
    '"section": <DOMAIN NOTES section string or null>, "rationale": <one sentence>, '
    '"confidence": <number 0.0-1.0>}. No text outside the JSON array.'
)

S1_EXCEPTION_AGENT = AgentSpec(
    name="s1_exception",
    system_prompt=SYSTEM_PROMPT,
    allowed_tools=["Read", "Grep"],
    model=MODEL_CHEAP,
    max_turns=15,
)


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
                    "detail": {k: rec[k] for k in (
                        "gyro_axis_by_deg_axis", "is_bijection", "unit", "r")},
                })
            for ch in o.get("drift_contaminated", []):
                queue.append({
                    "ref": f"drift:{Path(o['path']).name}:{ch}",
                    "type": "yaw_drift",
                    "file": o["path"],
                    "channel": ch,
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


_REVIEW_KEYS = ("disposition", "section", "rationale", "confidence")


def write_review(out_dir: Path, queue: list[dict], decisions: list[dict] | None,
                 final_text: str) -> Path:
    """Write one review row per queue item, agent verdict merged in.

    A queue item with no parseable verdict is conservatively marked needs_human, so a
    parsing failure never silently drops an exception. Raw text is kept for audit.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    by_ref = {d.get("ref"): d for d in decisions} if decisions else {}
    out = out_dir / REVIEW_FILENAME
    with out.open("w", encoding="utf-8") as fh:
        for item in queue:
            d = by_ref.get(item["ref"])
            if d is None:
                review = {
                    "disposition": "needs_human", "section": None,
                    "rationale": "no parseable agent verdict for this item",
                    "confidence": 0.0, "unparsed": True,
                }
            else:
                review = {k: d.get(k) for k in _REVIEW_KEYS}
            fh.write(json.dumps({**item, "review": review}, ensure_ascii=False) + "\n")
    if decisions is None:
        (out_dir / "exceptions_review_raw.txt").write_text(final_text, encoding="utf-8")
    return out
