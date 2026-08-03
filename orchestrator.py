"""Stage orchestrator. Deliberately dumb: sequence, gate, log. No intelligence here.

Usage:
    python orchestrator.py --phase 2      # S1 exception triage
    python orchestrator.py --phase 3      # S2 champion/challenger cycle
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from agents.base import run_agent
from agents.s1_exception import (
    S1_EXCEPTION_AGENT,
    build_prompt,
    build_queue,
    parse_review,
    write_review,
)
from agents import s2_critic, s2_experimenter

REPO_ROOT = Path(__file__).resolve().parent
RUNS_DIR = REPO_ROOT / "runs"
CLEAN_RUN_DIR = RUNS_DIR / "s1_clean"  # where `python -m stages.s1_clean.clean` writes its ledgers
S2_RUN_DIR = RUNS_DIR / "s2_ml"        # champion.json + experiments.jsonl live here


def git_sha() -> str:
    """runs/ is gitignored, so each run records the commit that produced it."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def new_run_dir() -> Path:
    """runs/YYYY-MM-DD_runN — never overwrite a previous run."""
    today = date.today().isoformat()
    n = 1
    while (RUNS_DIR / f"{today}_run{n}").exists():
        n += 1
    run_dir = RUNS_DIR / f"{today}_run{n}"
    run_dir.mkdir(parents=True)
    (run_dir / "run_meta.json").write_text(
        json.dumps(
            {"started": datetime.now(timezone.utc).isoformat(), "git_sha": git_sha()},
            indent=2,
        ),
        encoding="utf-8",
    )
    return run_dir


async def phase2(run_dir: Path) -> None:
    """S1 exception triage: the agent judges the clean stage's exception queue.

    Deterministic code builds the queue and writes the review; the agent only judges.
    """
    if not CLEAN_RUN_DIR.exists():
        raise FileNotFoundError(
            f"No clean run at {CLEAN_RUN_DIR}. Run `python -m stages.s1_clean.clean "
            f"--out runs/s1_clean` first."
        )

    queue, summary = build_queue(CLEAN_RUN_DIR)
    print(f"[s1-exc] queue: {summary}")

    if not queue:
        write_review(run_dir, queue, [], "")
        print("[s1-exc] empty queue — nothing to triage")
        return

    result = await run_agent(S1_EXCEPTION_AGENT, build_prompt(queue, summary), run_dir)
    decisions = parse_review(result.final_text)
    out = write_review(run_dir, queue, decisions, result.final_text)

    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    tally: dict[str, int] = {}
    for r in rows:
        tally[r["review"]["disposition"]] = tally.get(r["review"]["disposition"], 0) + 1
    print(f"[s1-exc] {len(queue)} triaged -> {out.name}: {tally}; "
          f"turns={result.num_turns} cost=${result.cost_usd:.4f}")
    if decisions is None:
        print("[s1-exc] WARNING: agent output did not parse — all items marked needs_human")


async def phase3(run_dir: Path) -> None:
    """S2 champion/challenger: experimenter proposes, critic reviews, code decides.

    The order matters. The critic runs BEFORE training, so a repeat or an incoherent
    proposal costs an API call rather than a full leave-one-rev-out sweep. And promotion
    is never an agent's call — `decide()` applies the margin rule to the measured
    macro-F1 after the run, whatever either agent believed.
    """
    from stages.s2_ml.experiment import (
        ExperimentSpec, decide, ledger, load_champion, proposals,
        record, record_proposal, run_experiment,
    )
    from stages.s2_ml.dataset import load_dataset
    from stages.s2_ml.features import WindowSpec, build_windows, feature_columns

    champion = load_champion(S2_RUN_DIR)
    if champion is None:
        raise FileNotFoundError(
            f"No champion in {S2_RUN_DIR}. Establish a baseline first "
            f"(`python -m stages.s2_ml.train --out runs/s2_ml`)."
        )
    report_md = (S2_RUN_DIR / "locoeval.md").read_text(encoding="utf-8")
    rows = ledger(S2_RUN_DIR)
    prior = proposals(S2_RUN_DIR)

    trials = load_dataset()
    feats = feature_columns(build_windows(trials, WindowSpec()))

    # 1. Propose.
    ex_prompt = s2_experimenter.build_prompt(report_md, rows, champion, feats)
    if prior:
        ex_prompt += ("\nPROPOSALS ALREADY RAISED (some never ran — do not repeat "
                      f"these either):\n{json.dumps([p['proposal'] for p in prior], indent=2)}\n")
    ex = await run_agent(s2_experimenter.S2_EXPERIMENTER_AGENT, ex_prompt, run_dir)
    proposal = s2_experimenter.parse_proposal(ex.final_text)
    s2_experimenter.write_proposal(run_dir, proposal, ex.final_text)
    if proposal is None:
        print("[s2-exp] proposal did not parse — nothing run")
        record_proposal(S2_RUN_DIR, {"unparsed": ex.final_text[:500]},
                        {"verdict": "revise"}, ran=False, note="proposal did not parse")
        return
    print(f"[s2-exp] proposed '{proposal.get('name')}': {proposal.get('rationale')}")

    # 2. Critique, before spending a training run.
    cr_prompt = s2_critic.build_prompt(proposal, rows, report_md, champion)
    cr = await run_agent(s2_critic.S2_CRITIC_AGENT, cr_prompt, run_dir)
    review = s2_critic.parse_review(cr.final_text)
    s2_critic.write_review(run_dir, review, cr.final_text)
    verdict = (review or {}).get("verdict", "revise")
    print(f"[s2-critic] {verdict}: {'; '.join((review or {}).get('reasons', [])[:3])}")

    if verdict != "approve":
        record_proposal(S2_RUN_DIR, proposal, review or {}, ran=False,
                        note=f"critic said {verdict}")
        print(f"[s2] not run (critic: {verdict}); champion unchanged")
        return

    # 3. Run it. Deterministic from here on.
    try:
        spec = ExperimentSpec(
            name=proposal["name"], rationale=proposal["rationale"],
            drop_features=proposal.get("drop_features") or [],
            window_s=proposal.get("window_s"),
            model_params=proposal.get("model_params") or {},
        )
        result = run_experiment(spec, trials, taxonomy=True)
    except (KeyError, ValueError) as exc:
        record_proposal(S2_RUN_DIR, proposal, review or {}, ran=False,
                        note=f"invalid spec: {exc}")
        print(f"[s2] spec rejected before training: {exc}")
        return

    # 4. Gate. The metric decides, not the agents.
    promote, why = decide(result, champion)
    record(S2_RUN_DIR, result, promote, why, critic=review)
    record_proposal(S2_RUN_DIR, proposal, review or {}, ran=True, note=why)

    print(f"[s2] macro-F1 {result.macro_f1:.4f} vs champion {champion['macro_f1']:.4f}")
    print(f"[s2] {'PROMOTED' if promote else 'rejected'}: {why}")
    print(f"[s2] cost: experimenter ${ex.cost_usd:.4f} + critic ${cr.cost_usd:.4f}")


PHASES = {2: phase2, 3: phase3}


def main() -> None:
    # Agent text is prose: em-dashes, arrows, the odd Greek letter. The Windows console
    # defaults to cp949 here and raises UnicodeEncodeError on the first one — which
    # would crash the run AFTER the API call was paid for, purely on a print. Replace
    # unencodable characters instead; garbled output beats a lost run.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    load_dotenv(REPO_ROOT / ".env")  # SDK reads ANTHROPIC_API_KEY from the environment

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, required=True, choices=sorted(PHASES))
    args = parser.parse_args()

    run_dir = new_run_dir()
    print(f"[run] {run_dir}")
    asyncio.run(PHASES[args.phase](run_dir))
    print(f"[run] artifacts in {run_dir}")


if __name__ == "__main__":
    main()
