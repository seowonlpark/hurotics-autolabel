# clean-room end-to-end runner: fresh clone + data/raw -> full pipeline -> S4 fusion report, gated, no
# manual intervention. each stage runs as its own subprocess and must produce its gate artifact before
# the next starts. the lockbox is SPENT, so this never invokes lockbox.py (Section 12.5).

# modes: (default) deterministic spine only; --with-agents adds paid agent steps; --s2-cycle[s N]
# runs champion/challenger cycles. see --help.

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parent
RAW_DIR = REPO_ROOT / "data" / "raw"
RUNS = REPO_ROOT / "runs"
PY = sys.executable  # the venv's python, so subprocesses use the same interpreter

# the champion is defined by its spec + git_sha. runs/ is gitignored, so a fresh clone has no
# champion.json; the forest is deterministic, so re-running the spec reproduces the recorded champion.
# the spec lives in the git-tracked champion_spec.json, which every promotion rewrites automatically.


# one pipeline step. exactly one of cmd / fn is set
@dataclass
class Step:
    key: str                       # short id, also the --only selector
    title: str                     # one-line human label
    cmd: list[str] | None = None   # subprocess argv (relative to REPO_ROOT)
    fn: Callable[[], None] | None = None  # in-process action (champion seed only)
    gate: Path | None = None       # artifact that must exist after; None => trust exit code
    agent: bool = False            # paid agent step (needs the API key)
    opt_in: bool = False           # only runs when explicitly requested (s2_cycle)


# seed champion.json deterministically when a fresh clone has none. idempotent: if a champion
# already exists (e.g. a prior run, or one promoted by --s2-cycle) it is left untouched
def seed_champion() -> None:
    from stages.s2_ml.experiment import (
        decide, load_champion, load_champion_spec, record, run_experiment,
    )

    out = RUNS / "s2_ml"
    if load_champion(out) is not None:
        print("  champion.json already present -- leaving it untouched")
        return
    spec = load_champion_spec()  # the git-tracked seed, kept current by every promotion
    print(f"  seeding champion '{spec.name}' ({len(spec.drop_features)} features dropped)...")
    result = run_experiment(spec, taxonomy=True)
    promote, why = decide(result, None)  # champion=None => establishes the baseline
    record(out, result, promote, why)
    print(f"  champion seeded: LORO macro-F1 {result.macro_f1:.4f}")


# the ordered pipeline. deterministic cores are free; agent steps are marked agent=True. order matters:
# OOF and the champion model are refit after any champion change, and S4 fuse joins the OOF with S3.
def build_steps() -> list[Step]:
    return [
        Step("s1_census", "S1 census (measure the corpus)",
             cmd=[PY, "-m", "stages.s1_clean.run", "--out", "runs/s1_census"],
             gate=RUNS / "s1_census" / "census.md"),
        Step("s1_clean", "S1 clean (resample to canonical 100 Hz, quarantine)",
             cmd=[PY, "-m", "stages.s1_clean.clean", "--out", "runs/s1_clean"],
             gate=RUNS / "s1_clean" / "clean_report.md"),
        Step("s1_exception", "S1 exception agent (triage the exception queue)",
             cmd=[PY, "orchestrator.py", "s1_exception"], agent=True),
        Step("s2_train", "S2 train + locoeval (champion spec, leave-one-rev-out)",
             cmd=[PY, "-m", "stages.s2_ml.train", "--out", "runs/s2_ml", "--taxonomy"],
             gate=RUNS / "s2_ml" / "locoeval.md"),
        Step("s2_champion", "S2 seed champion (deterministic, from champion_spec.json)",
             fn=seed_champion, gate=RUNS / "s2_ml" / "champion.json"),
        Step("s2_cycle", "S2 champion/challenger cycle (experimenter + critic)",
             cmd=[PY, "orchestrator.py", "s2_cycle"], agent=True, opt_in=True),
        Step("s2_refit", "S2 refit champion artifacts (after the cycle may have promoted)",
             cmd=[PY, "-m", "stages.s2_ml.train", "--out", "runs/s2_ml", "--taxonomy",
                  "--skip-if-current"],  # a no-op when the cycle promoted nothing
             gate=RUNS / "s2_ml" / "champion.joblib", opt_in=True),
        Step("s2_oof", "S2 out-of-fold predictions (fusion input)",
             cmd=[PY, "-m", "stages.s2_ml.oof", "--out", "runs/s2_ml"],
             gate=RUNS / "s2_ml" / "oof_champion.csv"),
        Step("s3_core", "S3 physics core (anchors, rate audit, plots)",
             cmd=[PY, "-m", "stages.s3_physics.run"],
             gate=RUNS / "s3_physics" / "anchors.csv"),
        Step("s3_physics", "S3 hypothesis agent (reads plots, provenance-gated)",
             cmd=[PY, "orchestrator.py", "s3_physics"], agent=True),
        Step("s4_fuse", "S4 fusion core (fuse S2+S3 -> call + confidence)",
             cmd=[PY, "-m", "stages.s4_fusion.run"],
             gate=RUNS / "s4_fusion" / "fusion_report.md"),
        Step("s4_fusion", "S4 fusion agent (judge the disagreement cases)",
             cmd=[PY, "orchestrator.py", "s4_fusion"], agent=True),
        Step("s4_newclass", "S4 new-class discovery (governed, needs_human)",
             cmd=[PY, "orchestrator.py", "s4_newclass"], agent=True),
    ]


# which steps to run given the flags. n_cycles is how many champion/challenger rounds to run
# (0 = none); each round is a fresh subprocess that sees the ledger + champion the last one left,
# and one s2_refit at the end brings the deployment artifacts to the final champion
def select(steps: list[Step], with_agents: bool, n_cycles: int) -> list[Step]:
    out = []
    for s in steps:
        if s.opt_in:
            if n_cycles <= 0:
                continue
            if s.key == "s2_cycle":  # expand into one step per round
                for i in range(1, n_cycles + 1):
                    title = s.title if n_cycles == 1 else f"{s.title} [round {i}/{n_cycles}]"
                    out.append(replace(s, title=title))
            elif s.key == "s2_refit":  # once, after the last round
                out.append(s)
            continue
        if s.agent and not with_agents:
            continue
        out.append(s)
    return out


# fail fast with a clear message before spending any time
def preflight(selected: list[Step]) -> None:
    raw_csvs = list(RAW_DIR.rglob("*.csv")) if RAW_DIR.exists() else []
    if not raw_csvs:
        sys.exit(
            f"[preflight] no raw data in {RAW_DIR}\n"
            f"            drop the device logs into data/raw/<YYYYMMDD>/*.csv first "
            f"(raw is never in git)."
        )
    print(f"[preflight] {len(raw_csvs)} raw csv files under data/raw/")

    if any(s.agent for s in selected):
        has_key = bool(os.environ.get("ANTHROPIC_API_KEY")) or (REPO_ROOT / ".env").exists()
        if not has_key:
            sys.exit(
                "[preflight] agent steps requested but no API key found\n"
                "            set ANTHROPIC_API_KEY or create .env (see .env.example)."
            )
        print("[preflight] agent steps enabled (this spends API credit)")


# run one step; returns True on success
def run_step(step: Step, dry_run: bool) -> bool:
    tag = "[agent]" if step.agent else "[det]  "
    print(f"\n=== {tag} {step.key}: {step.title}")
    if dry_run:
        print("  (dry-run) " + (" ".join(step.cmd) if step.cmd else f"in-process {step.key}"))
        return True

    start = time.time()
    if step.cmd is not None:
        import subprocess
        rc = subprocess.run(step.cmd, cwd=str(REPO_ROOT)).returncode
        if rc != 0:
            print(f"  FAILED: exit code {rc}")
            return False
    else:
        try:
            step.fn()  # type: ignore[misc]
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            return False

    if step.gate is not None and not step.gate.exists():
        print(f"  FAILED: expected artifact not written -> {step.gate.relative_to(REPO_ROOT)}")
        return False

    took = time.time() - start
    where = f" -> {step.gate.relative_to(REPO_ROOT)}" if step.gate else ""
    print(f"  ok ({took:.1f}s){where}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the full H-CARE pipeline end to end (clean-room).")
    ap.add_argument("--with-agents", action="store_true",
                    help="also run the paid agent steps (needs the API key)")
    ap.add_argument("--s2-cycle", action="store_true",
                    help="also run one champion/challenger cycle (paid, nondeterministic)")
    ap.add_argument("--s2-cycles", type=int, metavar="N",
                    help="run N champion/challenger cycles back to back (implies --s2-cycle); "
                         "each round sees the ledger and champion the previous one left")
    ap.add_argument("--list", action="store_true",
                    help="print the selected steps and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="print each command without running it")
    args = ap.parse_args()

    # --s2-cycles N wins if given; otherwise --s2-cycle means one round, absent means none
    n_cycles = args.s2_cycles if args.s2_cycles is not None else (1 if args.s2_cycle else 0)
    if n_cycles < 0:
        ap.error("--s2-cycles must be >= 0")

    # agent text carries characters the cp949 console can't encode; never let a stray print
    # crash a run (DOMAIN_NOTES Section 8)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    steps = build_steps()
    selected = select(steps, args.with_agents, n_cycles)

    if args.list:
        print("selected steps:")
        for s in selected:
            print(f"  {'[agent]' if s.agent else '[det]  '} {s.key:<14} {s.title}")
        return

    preflight(selected)

    n_agents = sum(s.agent for s in selected)
    print(f"[run] {len(selected)} steps ({n_agents} agent) "
          f"{'[DRY RUN]' if args.dry_run else ''}")

    started = time.time()
    for step in selected:
        if not run_step(step, args.dry_run):
            sys.exit(f"\n[run] STOPPED at {step.key}. fix the failure and re-run "
                     f"(completed stages are reused).")

    report = RUNS / "s4_fusion" / "fusion_report.md"
    print(f"\n[run] done: {len(selected)} steps in {time.time() - started:.1f}s")
    if report.exists():
        print(f"[run] report -> {report.relative_to(REPO_ROOT)}")
    if n_agents:
        print("[run] agent costs are in each run's runs/<date>_runN/costs.json")


if __name__ == "__main__":
    main()
