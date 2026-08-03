"""S4 fusion agent: judge WHY the pipeline could not confidently call a window.

The deterministic layer already knows *that* a window is uncertain and *which* signal
dissented — `fuse.py` emits reason codes for both. What it cannot say is what is actually
happening there. That is the judgement, and it is the whole reason this stage has an agent:

    reason code (code)          ->  cause (agent)
    physics_contradicts             the human label is wrong
    physics_abstains                a real state transition mid-window
    model_split                     genuinely slow gait the window under-reads
                                    a weight shift that mimics a step
                                    a data-quality problem

`label_suspect` is the highest-value verdict here and the reason the queue includes
confidently-wrong windows, not only abstentions. If ground truth is wrong, our "error" is
not one — and the precedent is on record: labeled STANDING that included the accel/decel
ramps while the physics read the ramp as walking. Raising the accuracy ceiling means
finding those, not tuning against them.

The agent never touches data values, never recomputes an anchor, and never changes a
call. It reads the queue rows and DOMAIN_NOTES, and returns a verdict per row; the
deterministic wrapper below writes the review and derives the disposition.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from agents.base import MODEL_SMART, AgentSpec, extract_json_array
from stages.s2_ml.dataset import STAND, TRAIN_CLASSES, WALK

REVIEW_FILENAME = "fusion_review.jsonl"

# The causes an agent may assign. A CLOSED vocabulary, checked below: an open-ended
# "what's wrong" field would drift into a new category per run, and §11.2 records exactly
# that failure — a whole class ("walking with no rhythm") invented to explain an artifact,
# which was then nearly promoted into the schema.
CAUSES = (
    "transition",         # the window straddles a real state change; benign
    "label_suspect",      # the ground truth looks wrong — the label-audit direction
    "slow_gait",          # real walking the window is too short or too still to read
    "weight_shift",       # standing, but with motion that mimics a step
    "data_quality",       # drift, untrusted rest zero, degenerate signal
    "ambiguous",          # genuinely undecidable from the evidence; the honest default
)

CLASS_NAME = {STAND: "stand", WALK: "walk"}

SYSTEM_PROMPT = (
    "You are the S4 review agent for an IMU stand/walk labeling pipeline. Deterministic "
    "code has already fused two independent opinions per 2 s window — a Random Forest "
    "(S2) and the zero-parameter swap rule (S3) — and flagged the windows it could not "
    "confidently and correctly call. Your job is to JUDGE WHY. You do not relabel "
    "anything and you do not recompute anything.\n\n"
    "The pipeline labels exactly two classes, stand and walk. Do NOT propose new classes; "
    "if a window looks like neither, that is `ambiguous`, not a new category. DOMAIN "
    "NOTES §11.2 records a category invented to explain an artifact — do not repeat it.\n\n"
    "Answer TWO independent questions per row. They are orthogonal: do not let one "
    "decide the other.\n\n"
    "1. `cause` — what is actually going on in this window? One of:\n"
    "  - \"transition\": it straddles a genuine stand<->walk change. Expected and benign. "
    "Evidence: the neighbouring windows carry different labels or calls.\n"
    "  - \"label_suspect\": the human ground truth looks WRONG. This is the most valuable "
    "verdict you can return. Evidence: the physics asserts a class with real support "
    "(swap_count >= 2 and antiphase clearly positive) against a contrary label, over a "
    "run of adjacent windows rather than one. Precedent: labeled STANDING that included "
    "the acceleration/deceleration ramps.\n"
    "  - \"slow_gait\": real walking whose stride the window under-reads. Evidence: "
    "physics abstained or the span had to be grown, with some periodicity present.\n"
    "  - \"weight_shift\": standing, with one leg crossing that mimics a step. Evidence: "
    "swap_count 1, low periodicity, weak antiphase.\n"
    "  - \"data_quality\": the window cannot be trusted — rest zero untrusted, drift, or "
    "a degenerate signal.\n"
    "  - \"ambiguous\": the evidence genuinely does not decide it. The honest default; "
    "prefer it over a confident guess.\n\n"
    "2. `action` — is a person needed?\n"
    "  - \"none\": the abstention is correct behaviour and nothing needs doing.\n"
    "  - \"human\": someone should look — a suspected mislabel, or a data problem.\n\n"
    "A benign transition needs no person even though the pipeline abstained. A "
    "`label_suspect` always needs one, because only a human may change ground truth.\n\n"
    "Also return `likely_label`: what YOU think the window most likely is — \"stand\", "
    "\"walk\", or \"unknown\" when the evidence does not decide. This is the pipeline's "
    "answer to \"what does it think it might be instead\", so `unknown` is a real answer, "
    "not a cop-out.\n\n"
    "Rules:\n"
    "  - Judge ONLY from the evidence on each row and from DOMAIN NOTES. Do NOT open raw "
    "CSVs to eyeball signals: the notes warn repeatedly that the eyeball is not truth "
    "(§11 item 4 — a bout called '2.5 s' was 6,246 ms) and that whole-file statistics "
    "mislead about a state occupying a small share of the file.\n"
    "  - One window is not a pattern. Prefer `label_suspect` when neighbouring windows "
    "agree with it; say so in the rationale when it rests on a single window.\n"
    "  - `sections`: DOMAIN NOTES sections you relied on, e.g. [\"10.7\"]. May be [].\n"
    "  - `confidence` is confidence in YOUR TWO FIELDS, not in the underlying physics.\n"
    "  - One sentence of rationale. Be terse and concrete: cite the numbers you used.\n\n"
    "Output ONLY a JSON array, one object per row, each exactly: "
    '{"ref": <the row ref>, "cause": <one of the six>, "action": "none"|"human", '
    '"likely_label": "stand"|"walk"|"unknown", "sections": [<strings>], '
    '"rationale": <one sentence>, "confidence": <number 0.0-1.0>}. '
    "No text outside the JSON array."
)

S4_FUSION_AGENT = AgentSpec(
    name="s4_fusion",
    system_prompt=SYSTEM_PROMPT,
    allowed_tools=["Read", "Grep"],
    # Sonnet, not haiku: this is the judgement stage, and the label-audit verdict is the
    # one finding worth paying for. §8 also measured the cheap model silently dropping a
    # verdict and stopping populating a field once the prompt grew.
    model=MODEL_SMART,
    max_turns=15,
)


def _neighbour_context(df: pd.DataFrame, row: pd.Series, flagged: pd.Series) -> dict:
    """What the windows either side of this one look like.

    `transition` and `label_suspect` are distinguished almost entirely by this: a single
    dissenting window between two agreeing ones is a boundary artifact, while a RUN of
    them against a constant label is a mislabel candidate. Neighbours are taken within
    the same segment only — across a gap they are not neighbours in time (§3.1).

    `flagged` must be the SAME mask the queue was selected with (abstained OR confidently
    wrong), not `abstain` alone. Counting only abstentions reported a run of 0 on every
    confident error — zero for precisely the rows where a run is the evidence that
    separates a mislabelled stretch from a one-window artifact.
    """
    sib = df[(df["rev"] == row["rev"]) & (df["trial"] == row["trial"])
             & (df["segment"] == row["segment"])].sort_values("t_start_ms")
    idx = sib.index.get_indexer([row.name])[0]
    if idx < 0:
        return {}
    before = sib.iloc[idx - 1] if idx > 0 else None
    after = sib.iloc[idx + 1] if idx + 1 < len(sib) else None

    def brief(r):
        if r is None:
            return None
        return {"label": None if pd.isna(pd.to_numeric(r["label"], errors="coerce"))
                else CLASS_NAME.get(int(float(r["label"]))),
                "call": CLASS_NAME.get(int(r["fused_label"])),
                "physics": r["swap_verdict_adaptive"]}

    flags = flagged.reindex(sib.index).fillna(False).to_numpy()
    run = 1
    for j in range(idx - 1, -1, -1):
        if flags[j]:
            run += 1
        else:
            break
    for j in range(idx + 1, len(sib)):
        if flags[j]:
            run += 1
        else:
            break
    return {"before": brief(before), "after": brief(after),
            "consecutive_flagged_windows": int(run)}


def build_queue(fused_csv: Path, max_items: int = 40) -> tuple[list[dict], dict]:
    """The rows worth a human-grade judgement, worst first.

    Two kinds, and both matter:
      - **abstained** windows: the pipeline declined to call them. Why?
      - **confidently wrong** windows: it called them and was wrong against the label.
        Only these can surface a mislabel, and they are the ones that cap the accuracy
        number, so they are queued FIRST.

    Capped at `max_items` because every row costs tokens and the queue is long. The cap
    is reported in the summary — a silent truncation would read as "the agent reviewed
    everything" when it did not.

    The cap is split BETWEEN the two kinds rather than applied to a single ranked list.
    Ranked together, the 110 confident errors filled all 40 slots and no abstention was
    ever reviewed — which silently deleted half the stage's purpose. A quota is the fix:
    each kind answers a different question, so neither may crowd the other out.
    """
    df = pd.read_csv(fused_csv)
    truth = pd.to_numeric(df["label"], errors="coerce")
    scoreable = truth.isin(TRAIN_CLASSES)

    wrong = df[scoreable & ~df["abstain"] & (df["fused_label"] != truth)]
    abstained = df[df["abstain"]]
    # The same mask the queue is built from, so neighbour runs measure the right thing.
    flagged = (scoreable & ~df["abstain"] & (df["fused_label"] != truth)) | df["abstain"]

    half = max_items // 2
    quota = {
        # Confident errors get the larger share when abstentions are scarce, and vice
        # versa: the split adapts rather than wasting slots on an empty category.
        "confident_error": min(len(wrong), max(half, max_items - len(abstained))),
        "abstained": min(len(abstained), max(max_items - half, max_items - len(wrong))),
    }

    rows: list[dict] = []
    for kind, sub in (("confident_error", wrong), ("abstained", abstained)):
        # Least-defensible first: a confident error at HIGH probability is the worst case
        # (loudly wrong), an abstention at LOW probability the most expected.
        sub = sub.sort_values("s2_proba", ascending=(kind == "abstained"))
        sub = sub.head(quota[kind])
        for _i, r in sub.iterrows():
            t = pd.to_numeric(r["label"], errors="coerce")
            rows.append({
                "ref": f"{kind}:{r['rev']}:t{int(r['trial'])}:s{int(r['segment'])}"
                       f":{int(r['t_start_ms'])}",
                "kind": kind,
                "where": {"rev": r["rev"], "trial": int(r["trial"]),
                          "segment": int(r["segment"]),
                          "t_start_s": round(float(r["t_start_ms"]) / 1000.0, 2)},
                "human_label": None if pd.isna(t) else CLASS_NAME.get(int(t), str(r["label"])),
                "pipeline_call": CLASS_NAME.get(int(r["fused_label"])),
                "confidence": r["confidence"],
                "reasons": r["reasons"].split(";") if isinstance(r["reasons"], str) and r["reasons"] else [],
                "model": {"pred": CLASS_NAME.get(int(r["s2_pred"])),
                          "probability": round(float(r["s2_proba"]), 3)},
                "physics": {"verdict": r["swap_verdict_adaptive"],
                            "swap_count": int(r["swap_count"]),
                            "antiphase": round(float(r["antiphase"]), 3),
                            "periodicity": round(float(r["periodicity_adaptive"]), 3),
                            "span_s": round(float(r["swap_window_s"]), 1),
                            "rest_zero_trusted": bool(r["rest_offset_trusted"])},
                "label_composition": {
                    "purity": round(float(r["purity"]), 3),
                    "human_unknown_fraction": round(float(r["unknown_frac"]), 3)},
                "neighbours": _neighbour_context(df, r, flagged),
            })

    summary = {
        "n_confident_errors": int(len(wrong)),
        "n_abstained": int(len(abstained)),
        "n_queued": len(rows),
        "queued_confident_errors": int(quota["confident_error"]),
        "queued_abstained": int(quota["abstained"]),
        "n_dropped_by_cap": int(len(wrong) + len(abstained) - len(rows)),
        "cap": max_items,
    }
    return rows, summary


def build_prompt(queue: list[dict], summary: dict) -> str:
    return (
        "Judge why the pipeline could not confidently and correctly call these windows.\n\n"
        f"Context: this run produced {summary['n_confident_errors']} confident errors and "
        f"{summary['n_abstained']} abstentions. The {summary['n_queued']} least defensible "
        f"are below (confident errors first); {summary['n_dropped_by_cap']} were left out "
        f"by the review cap.\n\n"
        "`kind: confident_error` means the pipeline claimed this window and disagreed with "
        "the human label — either the model is wrong or the LABEL is. `kind: abstained` "
        "means it declined to claim it.\n\n"
        "ROWS — return exactly one verdict per row, matched by `ref`:\n"
        f"{json.dumps(queue, indent=2, ensure_ascii=False)}\n"
    )


def parse_review(final_text: str) -> list[dict] | None:
    return extract_json_array(final_text)


_REVIEW_KEYS = ("cause", "action", "likely_label", "sections", "rationale", "confidence")


def collapse(cause: str | None, action: str | None) -> tuple[str, str]:
    """Derive (disposition, action) from the agent's two orthogonal judgements.

    The taxonomy is the pipeline's, decided once here, so two runs cannot disposition the
    same window differently:

        cause                       action      disposition
        label_suspect            -> human    -> label_audit
        data_quality             -> human    -> data_issue
        transition               (kept)      -> expected
        slow_gait / weight_shift (kept)      -> expected
        ambiguous                (kept)      -> unresolved
        unrecognized             -> human    -> unresolved

    `label_suspect` forces `human` because only a person may change ground truth — the
    agent may nominate a mislabel, never enact one. `data_quality` forces it too: a
    recording problem is not something the pipeline can decide its way out of.
    """
    if cause == "label_suspect":
        return "label_audit", "human"
    if cause == "data_quality":
        return "data_issue", "human"
    if cause in ("transition", "slow_gait", "weight_shift"):
        return "expected", action if action in ("none", "human") else "none"
    if cause == "ambiguous":
        return "unresolved", action if action in ("none", "human") else "none"
    return "unresolved", "human"  # unrecognized cause: judge nothing, escalate


def write_review(out_dir: Path, queue: list[dict], decisions: list[dict] | None,
                 final_text: str) -> Path:
    """One review row per queued window, agent verdict merged in.

    A window with no parseable verdict is marked `unresolved` / `human` rather than
    dropped: a parse failure must never quietly shrink the queue.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    by_ref = {d.get("ref"): d for d in decisions} if decisions else {}
    out = out_dir / REVIEW_FILENAME
    with out.open("w", encoding="utf-8") as fh:
        for item in queue:
            d = by_ref.get(item["ref"])
            if d is None:
                review = {
                    "cause": None, "action": "human", "likely_label": "unknown",
                    "sections": [], "disposition": "unresolved",
                    "rationale": "no parseable agent verdict for this window",
                    "confidence": 0.0, "unparsed": True,
                }
            else:
                review = {k: d.get(k) for k in _REVIEW_KEYS}
                if review.get("cause") not in CAUSES:
                    # An out-of-vocabulary cause is recorded, not accepted — the closed
                    # taxonomy is the point, and collapse() will escalate it.
                    review["cause_rejected"] = review.get("cause")
                    review["cause"] = None
                review["disposition"], review["action"] = collapse(
                    review.get("cause"), review.get("action"))
            fh.write(json.dumps({**item, "review": review}, ensure_ascii=False) + "\n")
    if decisions is None:
        (out_dir / "fusion_review_raw.txt").write_text(final_text, encoding="utf-8")
    return out
