"""Stage orchestrator. Deliberately dumb: sequence, gate, log. No intelligence here.

Usage:
    python orchestrator.py --phase 0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
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

REPO_ROOT = Path(__file__).resolve().parent
RUNS_DIR = REPO_ROOT / "runs"
CLEAN_RUN_DIR = RUNS_DIR / "s1_clean"  # where `python -m stages.s1_clean.clean` writes its ledgers


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


PHASES = {2: phase2}


def main() -> None:
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
