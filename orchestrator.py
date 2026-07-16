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

from agents.base import run_agent
from agents.smoke import SMOKE_AGENT, SMOKE_PROMPT

REPO_ROOT = Path(__file__).resolve().parent
RUNS_DIR = REPO_ROOT / "runs"


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


async def phase0(run_dir: Path) -> None:
    """Prove the skeleton: agent runs, hook logs, cost recorded."""
    result = await run_agent(SMOKE_AGENT, SMOKE_PROMPT, run_dir)
    print(result.final_text)
    print(f"\n[smoke] turns={result.num_turns} cost=${result.cost_usd:.4f}")


PHASES = {0: phase0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, required=True, choices=sorted(PHASES))
    args = parser.parse_args()

    run_dir = new_run_dir()
    print(f"[run] {run_dir}")
    asyncio.run(PHASES[args.phase](run_dir))
    print(f"[run] artifacts in {run_dir}")


if __name__ == "__main__":
    main()
