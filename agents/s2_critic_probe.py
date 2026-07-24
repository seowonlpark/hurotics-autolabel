# adversarial probe for the S2 critic: feed it three proposals it must NOT approve (verbatim repeat,
# cosmetic repeat, false-premise) and fail if any is approved. writes to a throwaway dir, not the ledger.

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from agents.base import run_agent
from agents import s2_critic
from stages.s2_ml.experiment import ledger, load_champion

REPO_ROOT = Path(__file__).resolve().parent.parent
S2_RUN_DIR = REPO_ROOT / "runs" / "s2_ml"
PROBE_RUN_DIR = REPO_ROOT / "runs" / "_critic_probe"


# each probe: the proposal + the verdicts that count as the critic doing its job
# approve is a failure for all three by construction
PROBES = [
    {
        "id": "verbatim_repeat",
        "proposal": {
            "name": "wider_window_4s",
            "rationale": "A 4 s window should capture slow gait cycles that a 2 s window "
                         "cuts in half, improving recall on slow walkers.",
            "drop_features": [],
            "window_s": 4.0,
            "model_params": {},
        },
        "must_be_in": {"reject"},
        "why": "exact spec of a rejected ledger entry (wider_window_4s, 0.8150)",
    },
    {
        "id": "cosmetic_repeat",
        "proposal": {
            "name": "longer_window_for_slow_gait",
            "rationale": "This population runs to 0.13 Hz (DOMAIN NOTES Section 9); a 2 s window "
                         "cannot resolve a stride that slow, so widening to 4 s should let "
                         "the frequency features see the fundamental and lift rev14.",
            "drop_features": [],
            "window_s": 4.0,
            "model_params": {},
        },
        "must_be_in": {"reject"},
        "why": "window_s=4.0 again under a new name - a repeat of wider_window_4s by spec",
    },
    {
        "id": "false_premise",
        "proposal": {
            "name": "drop_dom_hz_wrong_band",
            "rationale": "features.py computes the dominant-frequency features over the "
                         "healthy-adult (0.5, 3.0) Hz gait band, which excludes this "
                         "population's slow gait down to 0.13 Hz, so L/R_angvel_dom_hz "
                         "clip slow walkers to the band floor and mislead the classifier.",
            "drop_features": ["L_angvel_dom_hz", "R_angvel_dom_hz"],
            "window_s": None,
            "model_params": {},
        },
        "must_be_in": {"reject", "revise"},
        "why": "premise is false - GAIT_BAND_HZ is (0.13, 3.0), not (0.5, 3.0)",
    },
]


# run one probe through the real critic, return whether it bit
async def run_probe(probe: dict, rows: list[dict], report_md: str,
                    champion: dict | None, run_dir: Path) -> dict:
    prompt = s2_critic.build_prompt(probe["proposal"], rows, report_md, champion)
    res = await run_agent(s2_critic.S2_CRITIC_AGENT, prompt, run_dir)
    review = s2_critic.parse_review(res.final_text) or {}
    verdict = review.get("verdict", "revise") # unparsed => revise, per write_review
    bit = verdict in probe["must_be_in"]
    return {
        "id": probe["id"],
        "verdict": verdict,
        "expected": sorted(probe["must_be_in"]),
        "bit": bit,
        "repeat_of": review.get("repeat_of"),
        "reasons": review.get("reasons", []),
        "cost_usd": res.cost_usd,
        "why": probe["why"],
    }


# run the selected probes, print a bit/miss line each, fail if any was approved
async def main_async(args) -> None:
    rows = ledger(S2_RUN_DIR)
    if not rows:
        raise SystemExit(f"no ledger at {S2_RUN_DIR / 'experiments.jsonl'}")
    report_md = (S2_RUN_DIR / "locoeval.md").read_text(encoding="utf-8")
    champion = load_champion(S2_RUN_DIR)

    probes = PROBES if args.id is None else [p for p in PROBES if p["id"] == args.id]
    if not probes:
        raise SystemExit(f"no probe named {args.id!r}; have {[p['id'] for p in PROBES]}")

    print(f"Feeding the real critic {len(probes)} proposal(s) it must not approve.\n")
    results, total = [], 0.0
    for probe in probes:
        r = await run_probe(probe, rows, report_md, champion, PROBE_RUN_DIR)
        results.append(r)
        total += r["cost_usd"]
        status = "BIT" if r["bit"] else "MISSED"
        print(f"[{status}] {r['id']:18} verdict={r['verdict']:8} "
              f"(need {'/'.join(r['expected'])})  repeat_of={r['repeat_of']}")
        print(f"        target: {r['why']}")
        for reason in r["reasons"][:3]:
            print(f"        - {reason}")
        print()

    (PROBE_RUN_DIR / "probe_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    missed = [r["id"] for r in results if not r["bit"]]
    print(f"cost: ${total:.4f} across {len(results)} probe(s)")
    if missed:
        raise SystemExit(
            f"CRITIC DID NOT BITE on: {missed}. It approved a proposal it should have "
            f"stopped - the filter is not doing its job and the prompt needs work.")
    print("CRITIC BITES: every adversarial proposal was rejected or sent back to revise. "
          "The filter is demonstrated, not merely asserted.")


def main() -> None:
    # console guard (cp949 can't encode em-dash/sec , DOMAIN NOTES Section 8) + load the API key
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    load_dotenv(REPO_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Stress-test the S2 critic.")
    parser.add_argument("--id", help="run a single probe by id")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
