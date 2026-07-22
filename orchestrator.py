# stage orchestrator -- deliberately dumb: sequence, gate, log; no intelligence here
# invoked by STEP NAME matching the agent file, e.g. `python orchestrator.py s4_newclass`.
# steps: s1_exception, s2_cycle, s3_physics, s4_fusion, s4_newclass. see README.

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
CLEAN_RUN_DIR = RUNS_DIR / "s1_clean" # where the clean stage writes its ledgers
S2_RUN_DIR = RUNS_DIR / "s2_ml" # champion.json + experiments.jsonl live here


# current HEAD short sha; runs/ is gitignored, so each run records its commit
def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


# runs/YYYY-MM-DD_runN -- never overwrite a previous run
def new_run_dir() -> Path:
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


# S1 exception triage: code builds the queue and writes the review, the agent only judges
async def run_s1_exception(run_dir: Path) -> None:
    if not CLEAN_RUN_DIR.exists():
        raise FileNotFoundError(
            f"No clean run at {CLEAN_RUN_DIR}. Run `python -m stages.s1_clean.clean "
            f"--out runs/s1_clean` first."
        )

    queue, summary = build_queue(CLEAN_RUN_DIR)
    print(f"[s1-exc] queue: {summary}")

    if not queue:
        write_review(run_dir, queue, [], "")
        print("[s1-exc] empty queue - nothing to triage")
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
        print("[s1-exc] WARNING: agent output did not parse - all items marked needs_human")


# S2 champion/challenger: experimenter proposes, critic reviews before any training,
# code decides. promotion is never an agent's call -- decide() gates on measured macro-F1
async def run_s2_cycle(run_dir: Path) -> None:
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

    # 1. propose
    ex_prompt = s2_experimenter.build_prompt(report_md, rows, champion, feats)
    if prior:
        ex_prompt += ("\nPROPOSALS ALREADY RAISED (some never ran - do not repeat "
                      f"these either):\n{json.dumps([p['proposal'] for p in prior], indent=2)}\n")
    ex = await run_agent(s2_experimenter.S2_EXPERIMENTER_AGENT, ex_prompt, run_dir)
    proposal = s2_experimenter.parse_proposal(ex.final_text)
    s2_experimenter.write_proposal(run_dir, proposal, ex.final_text)
    if proposal is None:
        print("[s2-exp] proposal did not parse - nothing run")
        record_proposal(S2_RUN_DIR, {"unparsed": ex.final_text[:500]},
                        {"verdict": "revise"}, ran=False, note="proposal did not parse")
        return
    print(f"[s2-exp] proposed '{proposal.get('name')}': {proposal.get('rationale')}")

    # 2. critique, before spending a training run
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

    # 3. run it -- deterministic from here on
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

    # 4. gate -- the metric decides, not the agents
    promote, why = decide(result, champion)
    record(S2_RUN_DIR, result, promote, why, critic=review)
    record_proposal(S2_RUN_DIR, proposal, review or {}, ran=True, note=why)

    print(f"[s2] macro-F1 {result.macro_f1:.4f} vs champion {champion['macro_f1']:.4f}")
    print(f"[s2] {'PROMOTED' if promote else 'rejected'}: {why}")
    print(f"[s2] cost: experimenter ${ex.cost_usd:.4f} + critic ${cr.cost_usd:.4f}")


# S3 physics: deterministic core computes anchors + audit + plots, then the hypothesis agent
# reads the figures and writes hypotheses. code enforces the provenance gate before recording
# -- an agent's claim is kept only if it points at a real window (PLAN S3).
async def run_s3_physics(run_dir: Path) -> None:
    from stages.s3_physics.run import S3_OUT_DIR, run as run_s3_core
    from agents import s3_physics

    # 1. deterministic core -- regenerate the anchor table, audit and figures
    audit = run_s3_core(S3_OUT_DIR)
    verdicts = {a: v["verdict"] for a, v in audit["anchors"].items()}
    print(f"[s3] rate-invariance verdicts: {verdicts}")

    plots_dir = S3_OUT_DIR / "plots"
    plot_paths = sorted(str(p.resolve()) for p in plots_dir.glob("*.png"))
    disagreement = [{**d, "plot": str((S3_OUT_DIR / d["plot"]).resolve())}
                    for d in audit["disagreement"]]

    # 2. the agent reads the figures and proposes hypotheses (read-only)
    prompt = s3_physics.build_prompt(audit, disagreement, plot_paths)
    res = await run_agent(s3_physics.S3_HYPOTHESIS_AGENT, prompt, run_dir)

    # 3. gate -- code keeps a hypothesis only if it carries window-level provenance
    hyps = s3_physics.parse_hypotheses(res.final_text)
    path, n_ok, n_flagged = s3_physics.write_hypotheses(run_dir, hyps, verdicts, res.final_text)

    if hyps is None:
        print("[s3] hypotheses did not parse - nothing recorded")
    else:
        print(f"[s3] {n_ok} hypotheses passed the provenance gate, {n_flagged} flagged "
              f"-> {path.name}")
    print(f"[s3] cost: ${res.cost_usd:.4f}, turns={res.num_turns}")


# S4 fusion: the deterministic fuser combines S2 + S3 into one call + a confidence, then the
# agent characterises the disagreement (LOW-confidence) cases. code makes every call; the agent
# only judges what the abstentions are made of (PLAN S5).
async def run_s4_fusion(run_dir: Path) -> None:
    from stages.s4_fusion.run import S4_OUT_DIR, run as run_s4_core
    from stages.s3_physics.run import S3_OUT_DIR
    from agents import s4_fusion

    # 1. deterministic core -- regenerate the fused table, metrics, disagreement ranking
    metrics = run_s4_core(S4_OUT_DIR)
    a = metrics["acting_on_confidence"]
    print(f"[s4] fused macro-F1 {metrics['fused']['macro_f1']} (S2 alone "
          f"{metrics['s2_alone']['macro_f1']}); acting on HIGH+MED: coverage {a['coverage']} "
          f"at accuracy {a['accuracy']}")

    # 2. the agent judges the disagreements (read-only; physics figures are S3's, lockbox-sealed)
    prompt = s4_fusion.build_prompt(metrics, S3_OUT_DIR / "plots")
    res = await run_agent(s4_fusion.S4_FUSION_AGENT, prompt, run_dir)

    # 3. gate -- code keeps a finding only if it carries window-level provenance
    findings = s4_fusion.parse_review(res.final_text)
    path, n_ok, n_flagged = s4_fusion.write_review(run_dir, findings, res.final_text)
    if findings is None:
        print("[s4] review did not parse - nothing recorded")
    else:
        print(f"[s4] {n_ok} findings passed the provenance gate, {n_flagged} flagged "
              f"-> {path.name}")
    print(f"[s4] cost: ${res.cost_usd:.4f}, turns={res.num_turns}")


# S4 new-class discovery: the deterministic core assembles the evidence bundle (the NEW_CLASS
# curation spans + their physics profiles), then the agent PROPOSES classes the {stand, walk}
# taxonomy misses. governed (DOMAIN_NOTES Section 11.2): code validates cluster mass + provenance and
# routes every proposal to needs_human -- it never adds a class, never touches the model, the
# swap rule, the fuser, or a label. read-only, orthogonal to the deployed algorithm.
async def run_s4_newclass(run_dir: Path) -> None:
    from stages.s4_fusion.newclass import CANDIDATES_JSON, build_bundle
    from stages.s4_fusion.run import FUSED_CSV, S4_OUT_DIR
    from stages.s3_physics.run import S3_OUT_DIR
    from agents import s4_newclass

    if not (S4_OUT_DIR / FUSED_CSV).exists():
        raise FileNotFoundError(
            f"No fused table at {S4_OUT_DIR / FUSED_CSV}. Run `python -m stages.s4_fusion.run` "
            f"(or `orchestrator.py s4_fusion`) first - new-class discovery reads the fusion's "
            f"curation spans.")

    # 1. deterministic core -- assemble the candidate evidence bundle
    bundle = build_bundle(S4_OUT_DIR, S3_OUT_DIR)
    sig = bundle["corpus_signature"]
    print(f"[s4-nc] {sig['n_spans']} candidate spans / {sig['n_windows']} windows across "
          f"{sig['n_revs']} revs -> {S4_OUT_DIR / CANDIDATES_JSON}")
    if not bundle["spans"]:
        s4_newclass.write_proposals(run_dir, [], bundle["spans"], "")
        print("[s4-nc] no candidate spans - nothing to propose")
        return

    # 2. the agent proposes classes (read-only; physics figures are S3's, lockbox-sealed)
    prompt = s4_newclass.build_prompt(bundle, S3_OUT_DIR / "plots")
    res = await run_agent(s4_newclass.S4_NEWCLASS_AGENT, prompt, run_dir)

    # 3. gate -- code keeps cluster mass + provenance; every proposal is needs_human either way
    proposals = s4_newclass.parse_proposals(res.final_text)
    path, n_ok, n_weak = s4_newclass.write_proposals(run_dir, proposals, bundle["spans"],
                                                     res.final_text)
    if proposals is None:
        print("[s4-nc] proposals did not parse - nothing recorded")
    else:
        print(f"[s4-nc] {n_ok} proposals supported (cluster mass met), {n_weak} insufficient "
              f"-> {path.name}; all routed to needs_human")
    print(f"[s4-nc] cost: ${res.cost_usd:.4f}, turns={res.num_turns}")


# run steps, keyed by the NAME you type -- each name matches its agent file (agents/<name>.py,
# except s2_cycle which runs the experimenter+critic pair). the deployable pipeline is one stage
# per pass; a stage with more than one agent (s2, s4) gets one named step per job.
STEPS = {
    "s1_exception": run_s1_exception, # agents/s1_exception.py
    "s2_cycle": run_s2_cycle,         # agents/s2_experimenter.py + s2_critic.py
    "s3_physics": run_s3_physics,     # agents/s3_physics.py
    "s4_fusion": run_s4_fusion,       # agents/s4_fusion.py
    "s4_newclass": run_s4_newclass,   # agents/s4_newclass.py
}


def main() -> None:
    # agent text carries em-dashes/arrows/Greek that the Windows cp949 console can't encode;
    # replace unencodable chars so a stray print can't crash a run already paid for (Section 8)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    load_dotenv(REPO_ROOT / ".env") # SDK reads ANTHROPIC_API_KEY from the environment

    parser = argparse.ArgumentParser(description="Run one pipeline step by name.")
    parser.add_argument("step", choices=list(STEPS),
                        help="which step to run (matches agents/<name>.py)")
    args = parser.parse_args()

    run_dir = new_run_dir()
    print(f"[run] {args.step} -> {run_dir}")
    asyncio.run(STEPS[args.step](run_dir))
    print(f"[run] artifacts in {run_dir}")


if __name__ == "__main__":
    main()
