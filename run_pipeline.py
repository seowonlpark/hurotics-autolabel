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

# runs/ is gitignored, so each run records the commit that produced it; one definition,
# shared with the S2 experiment ledger, which makes the same provenance claim
from runmeta import git_sha as _git_sha
from stages.console import use_replacement_encoding

REPO_ROOT = Path(__file__).resolve().parent
RAW_DIR = REPO_ROOT / "data" / "raw"
RUNS = REPO_ROOT / "runs"
PY = sys.executable  # the venv's python, so subprocesses use the same interpreter


# one pipeline step
@dataclass(frozen=True)
class Step:
    key: str                       # short id, the --from handle and the progress label
    cmd: list[str] | None = None   # subprocess argv
    fn: Callable[[], None] | None = None   # OR run in-process; exactly one of the two
    gate: Path | None = None       # artifact that must exist after; None => trust the exit code
    agent: bool = False            # paid agent step (needs the API key)


# runs/YYYY-MM-DD_runN; never overwrite a previous run
def _new_run_dir() -> Path:
    today = date.today().isoformat()
    n = 1
    while (RUNS / f"{today}_run{n}").exists():
        n += 1
    run_dir = RUNS / f"{today}_run{n}"
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

    clean_dir = RUNS / "s1_clean"
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

    audit_dir = RUNS / "s3_physics"
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


# One bounded retry when the critic asks for a revision; two is the whole budget: a critic
# that still is not satisfied after one rewrite is disagreeing about the idea, not the spec,
# and another round buys a third phrasing of the same argument
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
        ExperimentSpec, champion_config, decide, ledger_by_basis, load_champion,
        load_champion_spec, proposals, record, record_proposal, run_experiment, seed,
    )
    from stages.s2_ml.features import build_windows, feature_columns

    s2_dir = RUNS / "s2_ml"
    report = s2_dir / "locoeval.md"
    if not report.exists():
        raise FileNotFoundError(
            f"No champion report at {report}. Run `python -m stages.s2_ml.train` first."
        )

    trials = load_dataset()

    # An absent champion.json is seeded rather than fatal: the incumbent's number must have
    # been produced by this gate's own code on this corpus, or the first challenger is
    # correctly refused as incomparable and the cycle can never start; a champion that IS
    # present but was measured over different ground is a different case and is left alone
    # `decide()` refuses it loudly and says how to re-baseline, and silently re-seeding there
    # would erase exactly the signal that the corpus moved under the metric
    champion = load_champion(s2_dir)
    if champion is None:
        seed(s2_dir, trials)
        champion = load_champion(s2_dir)

    report_md = report.read_text(encoding="utf-8")
    # Only rows measured on THIS basis are the ledger the agents reason from; the sixteen
    # inherited from the sibling repo go in as a separate, labelled block: their ideas still
    # count as already-tried, but their outcomes are not findings about the current features
    # the zeroing-family drop they record as a promotion is a regression here
    rows, prior_basis = ledger_by_basis(s2_dir, champion)
    prior = proposals(s2_dir)
    feats = feature_columns(build_windows(trials, champion_config(load_champion_spec())[0]))

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
            # NOT `or []`: that would collapse a null- "keep the champion's drops"- into an
            # empty list, which means drop nothing; the two are different proposals and the
            # difference is invisible in the ledger once made
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

    # A promotion rewrites `champion_spec.json`, and every artifact downstream of here
    # `champion.joblib`, the reference stats `label.py` serves from, roweval's accuracy claim
    #- still describes the model that just lost; refitting immediately is what keeps the
    # rest of this run about one champion instead of two; it is also the loudest possible
    # place for a promoted spec that `train.py` refuses to build: the declaration check runs
    # here, seconds after the promotion, not on someone's next pull
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


# the ordered pipeline; order is load-bearing in one place: s2_roweval re-fits per held-out
# rev and runs the real label.py, so it must follow s2_train, and it is the accuracy claim
# S4 fusion was deleted 2026-08-04: it published a pair no customer CSV ever went through
# S3 now does three things at three strengths: audits the ANNOTATIONS (strong), bounds a
# FILE against the model (moderate), gates WINDOWS against it (weak, ships OFF)
# the agents judge a deterministic stage's queue; none of them ever sees a number
def build_steps() -> list[Step]:
    return [
        # First, and unconditional; it reads no data at all- it holds `features.swap_counts`
        # (vectorized, all windows at once) to `rest.swap_count` (scalar, one window) on random
        # signals, so it is the one check that costs ~2 s and cannot be invalidated by a change
        # of corpus; running it before S1 means a broken vectorization stops the run in seconds
        # rather than after the minutes it takes to clean
        Step("verify_features",
             [PY, "-m", "stages.s2_ml.verify_features"]),
        # Same class as the check above and the same reason for sitting here: it reads no
        # data, costs milliseconds, and holds a mechanism to its own reference rather than to
        # a corpus; what it protects is `breakdown`'s staleness flag, whose failure mode is
        # silence- a check that has quietly stopped firing looks exactly like a pipeline with
        # nothing wrong, and every complaint it can make is exercised against a case built to
        # trip it (`freshness.py`, negative control at the bottom of the file)
        Step("verify_freshness",
             [PY, "-m", "freshness", "--self-test"]),
        Step("s1_census",
             [PY, "-m", "stages.s1_clean.run"],
             gate=RUNS / "s1_census" / "census.md"),
        Step("s1_clean",
             [PY, "-m", "stages.s1_clean.clean"],
             gate=RUNS / "s1_clean" / "clean_report.md"),
        Step("s1_exception",
             fn=_agent_step(_s1_exception), agent=True),
        # The bridge's MATH, before anything trains on it: the four `lpf_view` columns rebuilt
        # from raw and compared to HUROTICS' own export on every paired recording; needs no
        # model, so it runs here rather than after `s2_train`- a broken bridge should stop the
        # run before it spends minutes fitting 400 trees on features it cannot reproduce
        # On the spine since 2026-08-04, no longer a `--verify` opt-in: it is differential,
        # not a stored number- each trial carries its own MATLAB reference, so a change of
        # corpus changes its coverage and never its claim, and it fails loudly rather than
        # vacuously when no pairs are left. It cost minutes and got skipped; that is the
        # whole failure mode this repo has already lived through once
        Step("verify_transform",
             [PY, "-m", "stages.s2_ml.verify_transform"]),
        Step("s2_train",
             [PY, "-m", "stages.s2_ml.train"],
             gate=RUNS / "s2_ml" / "locoeval.md"),
        # The champion/challenger cycle, restored 2026-08-04; it needs a champion report to
        # reason from and a refit to follow a promotion, so it sits directly after `s2_train`
        # and before everything that consumes the champion; the gate is `experiments.jsonl`
        # rather than a promotion: a cycle whose challenger lost did its job, and the ledger
        # entry recording WHY is the artifact- it is what stops the next cycle spending
        # another fit on the same idea
        Step("s2_experiment",
             fn=_agent_step(_s2_cycle), agent=True,
             gate=RUNS / "s2_ml" / "experiments.jsonl"),
        # what a caller actually GETS, a different claim from the one above: gap
        # segmentation, decimation, the rest reference and the ensemble all sit between
        # "the serve path works" and "works on 86% of the corpus" are different claims
        # Also on the spine since 2026-08-04: it compares two code paths against each other
        # over whatever data is present, so there is no golden number in it to go stale
        Step("verify_serve",
             [PY, "-m", "stages.s2_ml.verify_serve"]),
        # The accuracy claim; runs the real `label.py` per held-out rev over ~1.27M rows and
        # writes the curve, the per-subject table and the reason validation that
        # `OPERATING_POINTS.md` quotes; it was never a pipeline step before 2026-08-04, which
        # meant a full run produced a fusion report nobody shipped and never regenerated the
        # numbers the deliverable is actually sold on
        Step("s2_roweval",
             [PY, "-m", "stages.s2_ml.roweval"],
             gate=RUNS / "s2_ml" / "roweval_loro.md"),
        # The same claim for the route a caller actually uses; `s2_roweval` scores the
        # annotated `lpf_view` export and `verify_serve` shows the raw route matches it, so
        # the raw path's accuracy was an inference off two artifacts rather than a
        # measurement; this labels the raw device CSVs that have a human annotation and
        # scores them against it; refuses the lockbox in code, so it is repeatable
        # unlike `roweval --lockbox`, it may run on every pipeline pass
        Step("s2_raweval",
             [PY, "-m", "stages.s2_ml.raweval"],
             gate=RUNS / "s2_ml" / "raweval.md"),
        # S3's own opinion, pointed at the ANNOTATIONS rather than at the classifier; it calls
        # `anchors.trial_anchors` in process, so it needs no anchors artifact; it also emits
        # `label_audit_windows.jsonl`- the specific windows a human should adjudicate with
        # `inspect_window`, which is what makes a trial-level flag actionable instead of
        # leaving a reader to binary-search 60 windows by hand
        Step("s3_label_audit",
             [PY, "-m", "stages.s3_physics.label_audit"],
             gate=RUNS / "s3_physics" / "label_audit.md"),
        # ...and the consumer for those nominations; the audit deliberately stops at "go
        # look"- the swap rule can be wrong about a window, so it hands over the raw trace
        # instead of a decision- which left its flags sitting unadjudicated; this assigns
        # each flagged trial a cause and says whether a person is needed; it reads
        # `label_audit.json`, so it follows the audit; it publishes no number, so it can
        # never become a second proof competing with `s2_roweval`
        Step("s3_label_review",
             fn=_agent_step(_s3_label_review), agent=True),
        # Is an anchor describing the body, or the sampling grid? Restored 2026-08-04 with
        # its own CLI: it was deleted with S4 because its DRIVER (`s3_physics/run.py`) also
        # built the fusion join's anchor table, but the audit itself answers a question that
        # has nothing to do with fusion; is the cost of not asking it- an experiment
        # once read clustering that partitioned by acquisition rate and was measuring the
        # clock; `gyro_energy` is the negative control and is expected to FAIL
        Step("s3_rate_audit",
             [PY, "-m", "stages.s3_physics.rate_audit"],
             gate=RUNS / "s3_physics" / "rate_audit.md"),
        # the serve path's file-level sanity bounds, run against their own controls
        # --control injects synthetic channel faults and reports which bounds catch them
        # a bound that has never fired is not evidence the data is clean
        Step("s3_plausibility",
             [PY, "-m", "stages.s3_physics.plausibility", "--calibrate", "--control"],
             gate=RUNS / "s3_physics" / "plausibility.json"),
        # Last, and it reads every stage above rather than producing anything of its own
        # Deterministic, so it runs on a free spine too: the run that most needs one page
        # saying what happened is the one nobody paid for an agent to review
        Step("breakdown",
             [PY, "-m", "stages.breakdown"],
             gate=RUNS / "breakdown.md"),
    ]


def select(steps: list[Step], with_agents: bool, start_at: str | None) -> list[Step]:
    out = [s for s in steps if with_agents or not s.agent]
    if start_at is not None:
        keys = [s.key for s in out]
        if start_at not in keys:
            # Naming an opt-in step without its flag is the common mistake, and it is not a
            # typo- say which flag brings it back rather than listing the rest
            missing = next((s for s in steps if s.key == start_at), None)
            if missing is not None:
                sys.exit(
                    f"[run] --from {start_at}: that step exists but is not selected — it "
                    f"needs --with-agents.\n"
                    f"      python run_pipeline.py --from {start_at} --with-agents"
                )
            sys.exit(f"[run] --from {start_at}: no such step. "
                     f"Selected: {', '.join(keys)}")
        out = out[keys.index(start_at):]
    return out


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
        # In-process, so an exception is the failure signal rather than an exit code; caught
        # here for the same reason a non-zero rc is: one step failing must stop the run with a
        # resume hint, not unwind through the runner and lose which step it was
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


# run the H-CARE pipeline end to end, from data/raw to runs/breakdown.md
# no argparse: --dry-run already lists every step and the exact command it runs
# an unrecognized flag must still be an error, not a silent no-op
#   --with-agents  also run the paid agent steps (needs the API key)
#   --from KEY     resume at this step; --dry-run lists the keys
#   --dry-run      print each command without running it
# `--verify` was removed 2026-08-04 and its two steps folded into the spine; a run that
# still passes it fails by the rule above rather than quietly skipping the checks again
USAGE = "usage: run_pipeline.py [--with-agents] [--from KEY] [--dry-run]"

_FLAGS = {"--with-agents": "with_agents", "--dry-run": "dry_run"}


# the parsed command line; plain attributes, same names argparse produced
class _Args:

    def __init__(self) -> None:
        self.with_agents = False
        self.dry_run = False
        self.start_at: str | None = None


# three flags, by hand; anything unrecognized exits 2 rather than being ignored
def _parse_args(argv: list[str]) -> _Args:
    args = _Args()
    it = iter(argv)
    for tok in it:
        if tok in _FLAGS:
            setattr(args, _FLAGS[tok], True)
        elif tok == "--from" or tok.startswith("--from="):
            # `--from KEY` and `--from=KEY` both, because half-supporting one spelling is how a
            # resume silently becomes a full re-run
            value = tok[len("--from="):] if "=" in tok else next(it, None)
            if not value:
                sys.exit(f"{USAGE}\nrun_pipeline.py: error: --from needs a step key "
                         f"(--dry-run lists them)")
            args.start_at = value
        else:
            sys.exit(f"{USAGE}\nrun_pipeline.py: error: unrecognized argument: {tok}")
    return args


def main() -> None:
    args = _parse_args(sys.argv[1:])

    # Stage and agent text carries characters the cp949 console cannot encode; never let a stray
    # print crash a run that has already been paid for; see stages/console.py
    use_replacement_encoding()

    selected = select(build_steps(), args.with_agents, args.start_at)

    # A dry run reads nothing and spends nothing, so it must not require raw data or a key
    # it is now also the way to see the step keys `--from` accepts, which has to work on a
    # checkout with no data in it yet
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

    # The staleness check used to live here, against `runs/s4_fusion` by name; it moved out with
    # the stage rather than being repointed: `breakdown` runs last and already calls the same
    # `check_all` over every `runs/*` directory carrying an `_inputs.json`, so a hardcoded second
    # copy could only ever cover less than the generic one and go stale the same way this one did
    report = RUNS / "breakdown.md"
    if report.exists():
        print(f"[run] report -> {report.relative_to(REPO_ROOT)}")
    if n_agents:
        print("[run] agent costs are in each run's runs/<date>_runN/costs.json")


if __name__ == "__main__":
    main()
