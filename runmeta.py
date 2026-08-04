# run provenance shared across the pipeline: the single definition of git_sha(), stamped by the
# orchestrator and the s2 ledger/champion records so every recorded artifact carries its commit

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


# current HEAD short sha, or 'unknown' outside a git checkout
def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"
