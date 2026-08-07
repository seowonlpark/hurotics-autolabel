from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

# runs/regen is regenerated in place, so a file sitting there need not match any commit; each run
# stamps the sha that produced it. One shared definition, with the S2 experiment ledger
from runmeta import git_sha as _git_sha
# the runs/ tree, defined once: REGEN rebuilds from this command, KEEP never does
from runslayout import AGENT_RUNS, BREAKDOWN_MD, KEEP_S2, LABELED_RAW, REGEN
from stages.console import use_replacement_encoding

REPO_ROOT = Path(__file__).resolve().parent
RAW_DIR = REPO_ROOT / "data" / "raw"
PY = sys.executable  # the venv's python, so subprocesses use the same interpreter


# a relative scale, since runtime is a fact about the corpus: the ORDER is the claim, not the clock
RUNTIMES = {
    "short":      "seconds- reads little or no data",
    "medium":     "up to a minute or so- one pass over the corpus, or one agent turn",
    "long":       "several minutes- fits a model, or rebuilds features corpus-wide",
    "super long": "tens of minutes- refits per held-out rev, or a full fit inside a cycle",
}


# one pipeline step
@dataclass(frozen=True)
class Step:
    key: str                       # short id, the --from handle and the progress label
    cmd: list[str] | None = None   # subprocess argv
    fn: Callable[[], None] | None = None   # OR run in-process; exactly one of the two
    # Artifact that must exist after; None => trust the exit code. Always the machine-read `.json`
    # twin, never the `.md`: the JSON is what the next stage and `breakdown` actually parse, so it
    # is the artifact whose absence really breaks the run. Gating on the rendered report made the
    # human-facing copy load-bearing, which is backwards -- a report exists to be read, not to be
    # depended on, and it should stay deletable without the pipeline concluding the stage never ran.
    gate: Path | None = None
    agent: bool = False            # paid agent step (needs the API key)
    desc: str = ""                 # one line, for --keys; what this step answers, not how
    runtime: str = ""              # one of RUNTIMES; relative scale, not a measurement


# runs/keep/agent_runs/YYYY-MM-DD_runN; never overwrite a previous run. Under keep/ because the
# reviews inside are paid and non-deterministic: re-running the agent does not reproduce them.
def _new_run_dir() -> Path:
    today = date.today().isoformat()
    n = 1
    while (AGENT_RUNS / f"{today}_run{n}").exists():
        n += 1
    run_dir = AGENT_RUNS / f"{today}_run{n}"
    run_dir.mkdir(parents=True)
    (run_dir / "run_meta.json").write_text(
        json.dumps({"started": datetime.now(timezone.utc).isoformat(),
                    "git_sha": _git_sha()}, indent=2),
        encoding="utf-8",
    )
    return run_dir


# disposition counts from a review ledger
def _tally(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)["review"]["disposition"]
        out[d] = out.get(d, 0) + 1
    return out


# S1 exception triage: the agent judges the clean stage's exception queue
async def _s1_exception(run_dir: Path) -> None:
    from agents.base import run_agent
    from agents.s1_exception import (
        S1_EXCEPTION_AGENT, build_prompt, build_queue, parse_review, write_review,
    )

    clean_dir = REGEN / "s1_clean"
    if not clean_dir.exists():
        raise FileNotFoundError(
            f"No clean run at {clean_dir}. Run `python -m stages.s1_clean.clean` first."
        )

    queue, summary = build_queue(clean_dir)
    print(f"[s1-exc] queue: {summary}")
    if not queue:
        write_review(run_dir, queue, [], "")
        print("[s1-exc] empty queue — nothing to triage")
        return

    result = await run_agent(S1_EXCEPTION_AGENT, build_prompt(queue, summary), run_dir)
    decisions = parse_review(result.final_text)
    out = write_review(run_dir, queue, decisions, result.final_text)
    print(f"[s1-exc] {len(queue)} triaged -> {out.name}: {_tally(out)}; "
          f"turns={result.num_turns} cost=${result.cost_usd:.4f}")
    if decisions is None:
        print("[s1-exc] WARNING: agent output did not parse — all items marked needs_human")


# S3 label review: the agent judges what the physics-vs-annotation audit flagged
async def _s3_label_review(run_dir: Path) -> None:
    from agents.base import run_agent
    from agents.s3_label_review import (
        S3_LABEL_REVIEW_AGENT, build_prompt, build_queue, parse_review, write_review,
    )

    audit_dir = REGEN / "s3_physics"
    if not (audit_dir / "label_audit.json").exists():
        raise FileNotFoundError(
            f"No label audit at {audit_dir}. Run "
            f"`python -m stages.s3_physics.label_audit` first."
        )

    queue, summary = build_queue(audit_dir)
    print(f"[s3-rev] queue: {summary}")
    if not queue:
        write_review(run_dir, queue, [], "")
        print("[s3-rev] empty queue — every trial passed both detectors")
        return

    result = await run_agent(S3_LABEL_REVIEW_AGENT, build_prompt(queue, summary), run_dir)
    decisions = parse_review(result.final_text)
    out = write_review(run_dir, queue, decisions, result.final_text)
    print(f"[s3-rev] {len(queue)} judged -> {out.name}: {_tally(out)}; "
          f"turns={result.num_turns} cost=${result.cost_usd:.4f}")
    if decisions is None:
        print("[s3-rev] WARNING: agent output did not parse — all items marked needs_human")


# one bounded retry: a critic unsatisfied after a rewrite disagrees about the idea, not the spec
MAX_PROPOSE_ATTEMPTS = 2


# first attempt keeps the canonical filename; a revision gets a suffix
def _attempt_name(base: str, attempt: int) -> str:
    if attempt == 1:
        return base
    stem, _, ext = base.rpartition(".")
    return f"{stem}_rev{attempt - 1}.{ext}"


# S2 champion/challenger: the experimenter proposes, the critic reviews, code decides
async def _s2_cycle(run_dir: Path) -> None:
    from agents import s2_critic, s2_experimenter
    from agents.base import run_agent
    from stages.s2_ml.dataset import load_dataset
    from stages.s2_ml.experiment import (
        ExperimentSpec, champion_config, comparable, corpus_fingerprint, decide,
        ledger_by_basis, load_champion, load_champion_spec, proposals, record,
        record_proposal, run_experiment, seed,
    )
    from stages.s2_ml.features import build_windows, feature_columns
    from stages.s2_ml.train import trainable

    s2_dir = REGEN / "s2_ml"
    report = s2_dir / "locoeval.json"
    if not report.exists():
        raise FileNotFoundError(
            f"No champion report at {report}. Run `python -m stages.s2_ml.train` first."
        )

    trials = load_dataset()

    # built once and read twice: the feature names the experimenter is shown, and the fingerprint
    # of the ground they are measured over
    windows = build_windows(trials, champion_config(load_champion_spec())[0])
    feats = feature_columns(windows)
    here = corpus_fingerprint(trainable(windows, "train"))

    # An ABSENT champion is seeded, and so is one measured over ground this corpus no longer is:
    # `decide()` would refuse to compare against it and every cycle from here would cost a fit and
    # settle nothing. Re-measuring the TRACKED spec is the re-baseline that refusal asks for, and
    # it is not a judgement call -- the fingerprints either match or they do not -- so it happens
    # here rather than through a flag someone has to know to pass.
    champion = load_champion(s2_dir)
    if champion is not None:
        same_ground, why = comparable(here, champion.get("corpus"))
        if not same_ground:
            print(f"[s2] incumbent was measured over other ground ({why}); "
                  f"re-seeding from champion_spec.json before challenging it")
            champion = None
    if champion is None:
        seed(s2_dir, trials)
        champion = load_champion(s2_dir)

    report_md = report.read_text(encoding="utf-8")
    # inherited rows go in a labelled block: already-tried ideas, but not findings about these features
    rows, prior_basis = ledger_by_basis(s2_dir, champion)
    prior = proposals(s2_dir)

    base_prompt = s2_experimenter.build_prompt(report_md, rows, champion, feats)
    if prior_basis:
        old = [{"name": e["spec"]["name"], "spec": e["spec"],
                "macro_f1": round(e["macro_f1"], 4), "promoted": e["promoted"],
                "outcome": e["decision_reason"]} for e in prior_basis]
        base_prompt += (
            "\nEARLIER-BASIS LEDGER - measured on a predecessor of this pipeline, a different "
            "model over a different feature set. Their macro-F1 values are NOT comparable to "
            "anything above and their outcomes are NOT findings about the current features: "
            "the zeroing-family drop recorded as a promotion there was re-measured on this "
            "basis as a regression. Use them to avoid re-treading an idea, and say how yours "
            "differs if you build on one. Never cite one as evidence a change will work:\n"
            f"{json.dumps(old, indent=2)}\n")
    if prior:
        base_prompt += ("\nPROPOSALS ALREADY RAISED (some never ran - do not repeat these "
                        f"either):\n{json.dumps([p['proposal'] for p in prior], indent=2)}\n")

    proposal = review = None
    ex_cost = cr_cost = 0.0
    revise_reasons: list[str] | None = None
    for attempt in range(1, MAX_PROPOSE_ATTEMPTS + 1):
        prompt = base_prompt
        if revise_reasons is not None:
            prompt += s2_experimenter.revision_block(proposal, revise_reasons)
        ex = await run_agent(s2_experimenter.S2_EXPERIMENTER_AGENT, prompt, run_dir)
        ex_cost += ex.cost_usd
        proposal = s2_experimenter.parse_proposal(ex.final_text)
        s2_experimenter.write_proposal(
            run_dir, proposal, ex.final_text,
            name=_attempt_name(s2_experimenter.PROPOSAL_FILENAME, attempt))
        if proposal is None:
            print("[s2-exp] proposal did not parse - nothing run")
            record_proposal(s2_dir, {"unparsed": ex.final_text[:500]}, {"verdict": "revise"},
                            ran=False, note="proposal did not parse")
            return
        print(f"[s2-exp] {'proposed' if attempt == 1 else 'revised'} "
              f"'{proposal.get('name')}': {proposal.get('rationale')}")

        # Critique before spending a training run, which is the cycle's whole cost asymmetry
        cr = await run_agent(s2_critic.S2_CRITIC_AGENT,
                             s2_critic.build_prompt(proposal, rows, report_md, champion),
                             run_dir)
        cr_cost += cr.cost_usd
        review = s2_critic.parse_review(cr.final_text)
        s2_critic.write_review(run_dir, review, cr.final_text,
                               name=_attempt_name(s2_critic.REVIEW_FILENAME, attempt))
        verdict = (review or {}).get("verdict", "revise")
        reasons = (review or {}).get("reasons", [])
        print(f"[s2-critic] {verdict}: {'; '.join(reasons[:3])}")

        if verdict == "approve":
            break
        if verdict == "revise" and attempt < MAX_PROPOSE_ATTEMPTS:
            revise_reasons = reasons or ["(the critic gave no specific reason)"]
            print(f"[s2] critic asked to revise; one bounded retry "
                  f"(attempt {attempt + 1}/{MAX_PROPOSE_ATTEMPTS})")
            continue
        # reject, or revise with no attempts left: terminal, champion untouched
        record_proposal(s2_dir, proposal, review or {}, ran=False,
                        note=f"critic said {verdict} after {attempt} attempt(s)")
        print(f"[s2] not run (critic: {verdict}); champion unchanged")
        return

    # Deterministic from here; neither agent sees what happens next
    try:
        spec = ExperimentSpec(
            name=proposal["name"], rationale=proposal["rationale"],
            # NOT `or []`: null means "keep the champion's drops", [] means drop nothing
            drop_features=proposal.get("drop_features"),
            window_s=proposal.get("window_s"), stride_s=proposal.get("stride_s"),
            model_params=proposal.get("model_params") or {},
        )
        result = run_experiment(spec, trials)
    except (KeyError, ValueError) as exc:
        record_proposal(s2_dir, proposal, review or {}, ran=False, note=f"invalid spec: {exc}")
        print(f"[s2] spec rejected before training: {exc}")
        return

    promote, why = decide(result, champion)
    record(s2_dir, result, promote, why, critic=review)
    record_proposal(s2_dir, proposal, review or {}, ran=True, note=why)

    print(f"[s2] macro-F1 {result.macro_f1:.4f} vs champion {champion['macro_f1']:.4f}")
    print(f"[s2] {'PROMOTED' if promote else 'rejected'}: {why}")
    print(f"[s2] cost: experimenter ${ex_cost:.4f} + critic ${cr_cost:.4f}")

    # refit immediately, or the rest of the run describes the champion that just lost
    if promote:
        print("[s2] promoted - refitting the champion so downstream artifacts match it")
        subprocess.run([PY, "-m", "stages.s2_ml.train"], cwd=REPO_ROOT, check=True)
        print("[s2] COMMIT stages/s2_ml/champion_spec.json: the tracked declaration moved")


# wrap one agent review so a Step can call it: fresh run dir, then await it
def _agent_step(coro) -> Callable[[], None]:
    def run() -> None:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")  # the SDK reads ANTHROPIC_API_KEY from the environment
        run_dir = _new_run_dir()
        print(f"  run dir: {run_dir.relative_to(REPO_ROOT)}")
        asyncio.run(coro(run_dir))
    run.__qualname__ = coro.__name__   # so --dry-run names the review, not the closure
    return run


# the ordered pipeline; order is load-bearing once: s2_roweval refits per rev, so it follows s2_train
def build_steps() -> list[Step]:
    return [
        # first and unconditional: it reads no data, so a broken vectorization stops the run in seconds
        Step("verify_features",
             [PY, "-m", "stages.s2_ml.verify_features"],
             desc="vectorized feature path vs its scalar reference", runtime="short"),
        # same class, same reason: it guards `breakdown`'s staleness flag, whose failure mode is silence
        Step("verify_freshness",
             [PY, "-m", "freshness", "--self-test"],
             desc="the staleness checker still fires (self-test)", runtime="short"),
        # before anything reads data/raw: every stage below now reads the cache, so this is the
        # one place that still proves the cache and the CSV are the same frame. It warms the
        # cache as a side effect, which is why the census below it is no longer a full parse.
        Step("verify_rawread",
             [PY, "-m", "stages.s1_clean.verify_rawread"],
             desc="the raw-read cache returns the CSV exactly", runtime="long"),
        Step("s1_census",
             [PY, "-m", "stages.s1_clean.run"],
             gate=REGEN / "s1_census" / "manifest.jsonl",
             desc="inventory data/raw: files, sessions, channels", runtime="medium"),
        Step("s1_clean",
             [PY, "-m", "stages.s1_clean.clean"],
             gate=REGEN / "s1_clean" / "segments.jsonl",
             desc="raw -> per-file channel trust; the cleaned frame is dropped", runtime="long"),
        Step("s1_exception",
             fn=_agent_step(_s1_exception), agent=True,
             desc="triage the clean stage's exception queue", runtime="medium"),
        # the bridge's MATH before anything trains on it; differential, so no corpus change stales it
        Step("verify_transform",
             [PY, "-m", "stages.s2_ml.verify_transform"],
             desc="lpf_view columns rebuilt from raw vs the vendor export", runtime="long"),
        Step("s2_train",
             [PY, "-m", "stages.s2_ml.train"],
             gate=REGEN / "s2_ml" / "locoeval.json",
             desc="fit the champion and LOCO-evaluate it", runtime="long"),
        # the champion/challenger cycle; the artifact is the ledger entry, not a promotion
        Step("s2_experiment",
             fn=_agent_step(_s2_cycle), agent=True,
             gate=KEEP_S2 / "experiments.jsonl",
             desc="champion/challenger cycle; code decides promotion", runtime="super long"),
        # what a caller actually GETS; differential like the check above, so no golden number can stale
        Step("verify_serve",
             [PY, "-m", "stages.s2_ml.verify_serve"],
             desc="one recording labelled both ways, row for row", runtime="super long"),
        # the accuracy claim: the real `label.py` per held-out rev, and the curve OPERATING_POINTS quotes
        Step("s2_roweval",
             [PY, "-m", "stages.s2_ml.roweval"],
             gate=REGEN / "s2_ml" / "roweval_loro.json",
             desc="THE accuracy claim: real label.py per held-out rev",
             runtime="super long"),
        # the same claim for the raw route, measured rather than inferred; refuses the lockbox in code
        Step("s2_raweval",
             [PY, "-m", "stages.s2_ml.raweval"],
             gate=REGEN / "s2_ml" / "raweval.json",
             desc="the same claim on the raw device route", runtime="super long"),
        # S3 pointed at the ANNOTATIONS, and it names the windows a human should adjudicate
        Step("s3_label_audit",
             [PY, "-m", "stages.s3_physics.label_audit"],
             gate=REGEN / "s3_physics" / "label_audit.json",
             desc="physics vs annotations: which trials contradict themselves",
             runtime="medium"),
        # ...and the consumer for those nominations: a cause per flagged trial, and no number at all
        Step("s3_label_review",
             fn=_agent_step(_s3_label_review), agent=True,
             desc="assign a cause to what the audit flagged", runtime="medium"),
        # is an anchor describing the body or the sampling grid? `gyro_energy` is expected to FAIL
        Step("s3_rate_audit",
             [PY, "-m", "stages.s3_physics.rate_audit"],
             gate=REGEN / "s3_physics" / "rate_audit.json",
             desc="is an anchor describing the body or the sampling grid?", runtime="medium"),
        # the serve path's file-level bounds against their own controls; an unfired bound is no evidence
        Step("s3_plausibility",
             [PY, "-m", "stages.s3_physics.plausibility"],
             gate=REGEN / "s3_physics" / "plausibility.json",
             desc="file-level sanity bounds, checked against injected faults",
             runtime="medium"),
        # the serve path over the WHOLE corpus, not the annotated slice: coverage, refusals and
        # the preset sweep, with no ground truth anywhere in it. It reads the champion, so it has
        # to follow s2_experiment's refit -- run before that promotion and every file in
        # labeled_raw/ describes a model the rest of the run has already replaced. The gate is a
        # `.csv` rather than the usual `.json` because `label_summary.csv` IS the machine-read
        # artifact here; `preset_sweep.json` is the wrong choice, since the sweep is legitimately
        # skipped whenever an abstention gate is on and a missing file would then read as failure.
        Step("s2_label_all",
             [PY, "-m", "stages.s2_ml.label_all"],
             gate=LABELED_RAW / "label_summary.csv",
             desc="label every raw file: corpus coverage, refusals, preset sweep",
             runtime="super long"),
        # last, reading every stage above; deterministic, so an unpaid run still gets its one page
        Step("breakdown",
             [PY, "-m", "stages.breakdown"],
             gate=BREAKDOWN_MD,
             desc="one page over every stage above", runtime="short"),
    ]


def select(steps: list[Step], with_agents: bool, start_at: str | None) -> list[Step]:
    out = [s for s in steps if with_agents or not s.agent]
    if start_at is not None:
        keys = [s.key for s in out]
        if start_at not in keys:
            # naming an opt-in step without its flag is not a typo- say which flag brings it back
            missing = next((s for s in steps if s.key == start_at), None)
            if missing is not None:
                sys.exit(
                    f"[run] --from {start_at}: that step exists but is not selected — it "
                    f"needs --with-agents.\n"
                    f"      python run_pipeline.py --from {start_at} --with-agents"
                )
            sys.exit(f"[run] --from {start_at}: no such step.\n"
                     f"      python run_pipeline.py --keys  lists them")
        out = out[keys.index(start_at):]
    return out


# the --from menu: ALL keys in run order, since filtering by an unpassed flag hides what you came for
def print_keys(steps: list[Step]) -> None:
    # a bucket missing from RUNTIMES prints blank and reads as "instant", so a typo stops it here
    bad = [s.key for s in steps if s.runtime not in RUNTIMES]
    if bad:
        sys.exit(f"[run] --keys: unknown runtime bucket on {', '.join(bad)}; "
                 f"expected one of {', '.join(RUNTIMES)}")

    kw = max(len(s.key) for s in steps)
    rw = max(len(s.runtime) for s in steps)
    print(f"{len(steps)} steps, in run order. `--from KEY` starts at one and runs "
          f"everything after it.\n")
    for s in steps:
        tag = " (agent)" if s.agent else ""
        print(f"  {s.key:<{kw}}  {s.runtime:<{rw}}  {s.desc}{tag}")
    print("\nruntime is a relative scale, not a measurement- every step scales with how much "
          "raw data\nyou have, so these hold their ORDER as the corpus grows, not their "
          "wall-clock:")
    for name, meaning in RUNTIMES.items():
        print(f"  {name:<{rw}}  {meaning}")
    print("\n(agent) steps need --with-agents, and spend API credit.")
    print("--dry-run prints the exact command each one runs.")


# fail fast with a clear message, before spending any time or any credit
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
    print(f"\n=== {tag} {step.key}", flush=True)
    if dry_run:
        print("  (dry-run) " + (" ".join(step.cmd) if step.cmd
                                else f"in-process: {step.fn.__qualname__}"))
        return True

    start = time.time()
    if step.fn is not None:
        # in-process, so an exception is the failure signal; caught here to stop with a resume hint
        try:
            step.fn()
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            return False
    else:
        rc = subprocess.run(step.cmd, cwd=str(REPO_ROOT)).returncode
        if rc != 0:
            print(f"  FAILED: exit code {rc}")
            return False

    if step.gate is not None and not step.gate.exists():
        print(f"  FAILED: expected artifact not written -> {step.gate.relative_to(REPO_ROOT)}")
        return False

    took = time.time() - start
    where = f" -> {step.gate.relative_to(REPO_ROOT)}" if step.gate else ""
    print(f"  ok ({took:.1f}s){where}")
    return True


# run the pipeline end to end, data/raw -> runs/breakdown.md; an unrecognized flag is an error
USAGE = "usage: run_pipeline.py [--with-agents] [--keys] [--from KEY] [--dry-run]"

_FLAGS = {"--with-agents": "with_agents", "--keys": "keys", "--dry-run": "dry_run"}


# the parsed command line; plain attributes, same names argparse produced
class _Args:

    def __init__(self) -> None:
        self.with_agents = False
        self.keys = False
        self.dry_run = False
        self.start_at: str | None = None


# four flags, by hand; anything unrecognized is an error rather than ignored
def _parse_args(argv: list[str]) -> _Args:
    args = _Args()
    it = iter(argv)
    for tok in it:
        if tok in _FLAGS:
            setattr(args, _FLAGS[tok], True)
        elif tok == "--from" or tok.startswith("--from="):
            # both spellings: half-supporting one is how a resume silently becomes a full re-run
            value = tok[len("--from="):] if "=" in tok else next(it, None)
            if not value:
                sys.exit(f"{USAGE}\nrun_pipeline.py: error: --from needs a step key "
                         f"(--keys lists them)")
            args.start_at = value
        else:
            sys.exit(f"{USAGE}\nrun_pipeline.py: error: unrecognized argument: {tok}")
    return args


def main() -> None:
    args = _parse_args(sys.argv[1:])

    # stage text carries characters cp949 cannot encode; a stray print must not crash a paid run
    use_replacement_encoding()

    steps = build_steps()

    # before select() and preflight: it must work on a checkout with no data and no key
    if args.keys:
        print_keys(steps)
        return

    selected = select(steps, args.with_agents, args.start_at)

    # a dry run reads nothing and spends nothing, so it must not require raw data or a key
    if not args.dry_run:
        preflight(selected)

    n_agents = sum(s.agent for s in selected)
    print(f"[run] {len(selected)} steps ({n_agents} agent) "
          f"{'[DRY RUN]' if args.dry_run else ''}")

    started = time.time()
    for step in selected:
        if not run_step(step, args.dry_run):
            sys.exit(f"\n[run] STOPPED at {step.key}. Fix the failure, then re-run with "
                     f"`--from {step.key}` to resume, or without it to redo the spine.")

    print(f"\n[run] done: {len(selected)} steps in {time.time() - started:.1f}s")

    # the staleness check lives in `breakdown` now: a hardcoded second copy could only cover less
    report = BREAKDOWN_MD
    if report.exists():
        print(f"[run] report -> {report.relative_to(REPO_ROOT)}")
    if n_agents:
        print("[run] agent costs are in each run's "
              "runs/keep/agent_runs/<date>_runN/costs.json")


if __name__ == "__main__":
    main()
