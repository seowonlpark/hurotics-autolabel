# S3 hypothesis agent: read the plots, write hypotheses grounded in real windows (read-only)
# the deterministic core already computed the anchors, drew the figures, and gave every
# anchor a rate-invariance verdict. this agent does the one thing code cannot: look at the
# physics and say what it means -- but only ever about a NAMED window, and never leaning on
# an anchor the audit called a grid artifact without saying so. it proposes hypotheses as
# JSON; code validates the provenance gate and writes hypotheses.jsonl. the agent does not
# label windows and does not train -- rule discovery, not fitting (§11.3). see README / PLAN.

from __future__ import annotations

import json
import re
from pathlib import Path

from agents.base import MODEL_SMART, AgentSpec
from stages.s3_physics.anchors import ANCHOR_NAMES

HYPOTHESES_FILENAME = "hypotheses.jsonl"

CONFIDENCE_LEVELS = ("low", "medium", "high")

SYSTEM_PROMPT = (
    "You are the S3 physics analyst for an IMU locomotion corpus. Deterministic code has "
    "already computed the anchor features, drawn a figure per trial, and given every anchor "
    "a rate-invariance verdict. You cannot see the raw data — the FIGURES are your evidence. "
    "Your job is to read them and write HYPOTHESES about the body and the labels.\n\n"
    "Read the plots with the Read tool (they are PNGs; Read renders them — use the absolute "
    "paths given in the prompt). Start with the rate-invariance figure (rate_audit.png) and "
    "the trials ranked highest for physics-vs-label disagreement — those are where the "
    "interesting physics is. Read more if you need them.\n\n"
    "Each trial figure has three stacked panels on one time axis (seconds):\n"
    "  1. the two leg angles, with the HUMAN label drawn as the background band\n"
    "  2. the interleg signal L_ang − R_ang with the ±1° swap bands, and the swap rule's "
    "stride-adaptive verdict as coloured squares along the top\n"
    "  3. the anchor timeline: antiphase, periodicity, grav_stab (left axis), gait_hz (right)\n\n"
    "The swap rule is validated and zero-parameter (DOMAIN NOTES §10): 0 swaps = STANDING, "
    "1 = AMBIGUOUS, ≥2 = WALKING. Two corrections are already applied so you are not re-deriving "
    "them (DOMAIN NOTES §10.4–§10.6): the interleg signal is recentred on each file's rest zero "
    "(a per-subject offset otherwise reads real gait as STANDING), and the swap window is sized to "
    "the local stride period so slow gait a fixed 2 s window can't resolve is not misread. The "
    "verdict squares and the disagreement ranking both use this corrected call. Where the swap "
    "track disagrees with the label band is "
    "exactly what you are here to explain — slow walking the rule abstains on, a standing "
    "bout that is really a split stance, a labeled STANDING stretch that is a moving ramp.\n\n"
    "HARD RULES — a hypothesis that breaks either is rejected by code before it is recorded:\n"
    "  1. **Window-level provenance.** Every hypothesis must point at real windows: give "
    "`evidence` as a list of {rev, trial, t_start_s, t_end_s, anchors, note}. No window, no "
    "hypothesis. 'Walking is often antiphase' is not a hypothesis; 'in rev2 t1 from 388–450s "
    "the swap rule abstains while the label says WALK, and gait_hz halves — a slow-cadence "
    "tail' is.\n"
    "  2. **Name the anchors you lean on, and respect their verdict.** List them in "
    "`relies_on`. An anchor the audit called `rate_dependent` (its value tracks the sampling "
    "grid, not the body) may be discussed, but you must say so — never build a load-bearing "
    "claim on it as if it were a body property.\n\n"
    "How to think (DOMAIN NOTES §11.3, §11.4):\n"
    "  - You are doing rule DISCOVERY, not per-window labeling. The swap rule came from "
    "asking *what walking is*, not from fitting. Look for structure the human labels miss.\n"
    "  - A metric that moves does not confirm the story you tell about it. State what would "
    "FALSIFY each hypothesis in its `caveat`, and what it does NOT claim.\n"
    "  - If a labeled region's physics contradicts its label, set `label_audit` true — that "
    "routes it to the human label-review path. Precedent: labeled STANDING that included "
    "accel/decel ramps, not plateau-only ground truth.\n"
    "  - Prefer few, well-evidenced hypotheses over many thin ones. Confidence is one of "
    f"{list(CONFIDENCE_LEVELS)}; earn 'high'.\n\n"
    "Output ONLY a JSON array of hypothesis objects, each with these fields:\n"
    '  {"name": snake_case, "statement": one sentence, "confidence": one of '
    f'{list(CONFIDENCE_LEVELS)}, '
    '"evidence": [{"rev","trial","t_start_s","t_end_s","anchors":[...],"note"}], '
    '"relies_on": [anchor/feature names], "label_audit": true|false, '
    '"caveat": what would falsify it / what it does not claim}\n'
    "No prose outside the JSON array."
)

S3_HYPOTHESIS_AGENT = AgentSpec(
    name="s3_physics",
    system_prompt=SYSTEM_PROMPT,
    allowed_tools=["Read", "Grep"], # Read renders the PNGs; Grep spot-checks anchors.csv
    model=MODEL_SMART,
    max_turns=30, # reading many figures costs turns
)


# assemble the prompt: the rate-invariance verdicts, the disagreement ranking (which plots to
# read first), and the full plot inventory
def build_prompt(audit: dict, disagreement: list[dict], plot_paths: list[str]) -> str:
    verdicts = {a: audit["anchors"][a]["verdict"] for a in ANCHOR_NAMES}
    top = disagreement[:8]
    return (
        "Write physics hypotheses for this corpus.\n\n"
        f"RATE-INVARIANCE VERDICTS (from {audit['rate_hz']:.0f}→{audit['decimated_hz']:.0f} Hz "
        f"decimation; anchors called rate_dependent track the grid, not the body):\n"
        f"{json.dumps(verdicts, indent=2)}\n\n"
        "TRIALS RANKED BY PHYSICS-vs-LABEL DISAGREEMENT (read these figures first — "
        "`walk_labeled_not_walking` = fraction of WALK windows the swap rule does not call "
        "walking; `stand_labeled_is_walking` = fraction of STAND windows it calls walking). "
        "`reasons` histograms the physics CAUSE per window: `stand_reads_walking` = a STAND "
        "label the physics reads as gait (label-audit candidate); `in_phase_not_gait` = WALK "
        "label but the legs move in phase (physics likely right); `single_hump`/`sub_threshold` "
        "= WALK label the rule under-called (a residual physics limit, weaker evidence of a label "
        "problem):\n"
        f"{json.dumps(top, indent=2)}\n\n"
        f"ALL TRIAL FIGURES ({len(plot_paths)}):\n{json.dumps(plot_paths, indent=2)}\n\n"
        "Read runs/s3_physics/plots/rate_audit.png and the top disagreement figures, then "
        "write your hypotheses as a JSON array. Every hypothesis needs window-level evidence."
    )


# pull the JSON array out of the agent's reply; tolerant of fences and prose
def parse_hypotheses(final_text: str) -> list[dict] | None:
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


# one evidence item carries window-level provenance: a rev, a trial, a real time window, and
# at least one anchor/feature it points at
def _valid_evidence(ev: object) -> bool:
    if not isinstance(ev, dict):
        return False
    if not ev.get("rev") or ev.get("trial") is None:
        return False
    try:
        t0, t1 = float(ev["t_start_s"]), float(ev["t_end_s"])
    except (KeyError, TypeError, ValueError):
        return False
    if not t1 > t0:
        return False
    return bool(ev.get("anchors"))


# enforce the PLAN S3 gate on one hypothesis: window-level provenance, valid confidence, and
# every relied-on anchor carries its rate-invariance verdict. returns the verdict-annotated
# hypothesis; ok=False means it fails the gate (recorded, but flagged, never silently kept).
def validate_hypothesis(h: dict, verdicts: dict[str, str]) -> dict:
    reasons: list[str] = []
    warnings: list[str] = []

    if not isinstance(h, dict) or not h.get("statement"):
        reasons.append("no statement")
    if h.get("confidence") not in CONFIDENCE_LEVELS:
        reasons.append(f"confidence not in {list(CONFIDENCE_LEVELS)}")

    evidence = h.get("evidence") or []
    if not isinstance(evidence, list) or not evidence:
        reasons.append("no evidence windows")
    elif not all(_valid_evidence(e) for e in evidence):
        reasons.append("an evidence item lacks rev/trial/time-window/anchors provenance")

    # attach the rate-invariance verdict for every anchor the hypothesis leans on; a claim
    # resting on a rate_dependent anchor is allowed but flagged unless it says so
    relied = [a for a in (h.get("relies_on") or []) if a in verdicts]
    rate_map = {a: verdicts[a] for a in relied}
    depended_dependent = [a for a in relied if verdicts[a] == "rate_dependent"]
    if depended_dependent and not h.get("label_audit"):
        text = (h.get("statement", "") + " " + h.get("caveat", "")).lower()
        if not any(w in text for w in ("rate", "grid", "sampl", "decimat")):
            warnings.append(f"leans on rate_dependent anchor(s) {depended_dependent} "
                            "without acknowledging the verdict")

    return {**h, "rate_invariance": rate_map,
            "validation": {"ok": not reasons, "reasons": reasons, "warnings": warnings}}


# validate all, write one JSONL line per hypothesis (verdict-annotated) + a header line; the
# raw agent text is kept alongside for audit. returns (path, n_ok, n_flagged).
def write_hypotheses(out_dir: Path, hypotheses: list[dict] | None, verdicts: dict[str, str],
                     final_text: str) -> tuple[Path, int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / HYPOTHESES_FILENAME
    (out_dir / "hypotheses_raw.txt").write_text(final_text or "", encoding="utf-8")

    if not hypotheses:
        path.write_text("", encoding="utf-8")
        return path, 0, 0

    validated = [validate_hypothesis(h, verdicts) for h in hypotheses]
    n_ok = sum(1 for v in validated if v["validation"]["ok"])
    with path.open("w", encoding="utf-8") as fh:
        for v in validated:
            fh.write(json.dumps(v, ensure_ascii=False) + "\n")
    return path, n_ok, len(validated) - n_ok
