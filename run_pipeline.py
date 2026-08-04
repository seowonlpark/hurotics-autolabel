from __future__ import annotations

import argparse
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

# runs/ is gitignored, so each run records the commit that produced it. One definition,
# shared with the S2 experiment ledger, which makes the same provenance claim.
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
    verify: bool = False           # slow correctness check, only with --verify


def _new_run_dir() -> Path:
    """runs/YYYY-MM-DD_runN — never overwrite a previous run."""
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


def _tally(path: Path) -> dict[str, int]:
    """Disposition counts from a review ledger."""
    out: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)["review"]["disposition"]
        out[d] = out.get(d, 0) + 1
    return out


async def _s1_exception(run_dir: Path) -> None:
    """S1 exception triage: the agent judges the clean stage's exception queue."""
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


async def _s3_label_review(run_dir: Path) -> None:
    """S3 label review: the agent judges what the physics-vs-annotation audit flagged.

    The audit nominates and refuses to conclude ("a nomination is not a verdict"), which
    left its flags unadjudicated — a flagged trial meant opening `inspect_window` by hand
    twelve times, so nobody did. This is the consumer that turns the flag into a ranked
    queue with a named cause.
    """
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


# One bounded retry when the critic asks for a revision. Two is the whole budget: a critic
# that still is not satisfied after one rewrite is disagreeing about the idea, not the spec,
# and another round buys a third phrasing of the same argument.
MAX_PROPOSE_ATTEMPTS = 2


def _attempt_name(base: str, attempt: int) -> str:
    """First attempt keeps the canonical filename; a revision gets a suffix.

    Every proposal and critique the cycle produced stays on disk. The draft the critic sent
    back is frequently the more informative artifact, and overwriting it would leave a run
    log that only ever shows agents agreeing.
    """
    if attempt == 1:
        return base
    stem, _, ext = base.rpartition(".")
    return f"{stem}_rev{attempt - 1}.{ext}"


async def _s2_cycle(run_dir: Path) -> None:
    """S2 champion/challenger: the experimenter proposes, the critic reviews, code decides.

    Promotion is never an agent's call. `experiment.decide()` gates on measured
    leave-one-rev-out macro-F1, and every outcome — including a proposal the critic stopped
    before it cost a fit — is logged where the next cycle will read it.
    """
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
    # correctly refused as incomparable and the cycle can never start. A champion that IS
    # present but was measured over different ground is a different case and is left alone —
    # `decide()` refuses it loudly and says how to re-baseline, and silently re-seeding there
    # would erase exactly the signal that the corpus moved under the metric.
    champion = load_champion(s2_dir)
    if champion is None:
        seed(s2_dir, trials)
        champion = load_champion(s2_dir)

    report_md = report.read_text(encoding="utf-8")
    # Only rows measured on THIS basis are the ledger the agents reason from. The sixteen
    # inherited from the sibling repo go in as a separate, labelled block: their ideas still
    # count as already-tried, but their outcomes are not findings about the current features —
    # the zeroing-family drop they record as a promotion is a regression here.
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

        # Critique before spending a training run, which is the cycle's whole cost asymmetry.
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

    # Deterministic from here. Neither agent sees what happens next.
    try:
        spec = ExperimentSpec(
            name=proposal["name"], rationale=proposal["rationale"],
            # NOT `or []`: that would collapse a null — "keep the champion's drops" — into an
            # empty list, which means drop nothing. The two are different proposals and the
            # difference is invisible in the ledger once made.
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

    # A promotion rewrites `champion_spec.json`, and every artifact downstream of here —
    # `champion.joblib`, the reference stats `label.py` serves from, roweval's accuracy claim
    # — still describes the model that just lost. Refitting immediately is what keeps the
    # rest of this run about one champion instead of two. It is also the loudest possible
    # place for a promoted spec that `train.py` refuses to build: the declaration check runs
    # here, seconds after the promotion, not on someone's next pull.
    if promote:
        print("[s2] promoted - refitting the champion so downstream artifacts match it")
        subprocess.run([PY, "-m", "stages.s2_ml.train"], cwd=REPO_ROOT, check=True)
        print("[s2] COMMIT stages/s2_ml/champion_spec.json: the tracked declaration moved")


def _agent_step(coro) -> Callable[[], None]:
    """Wrap one agent review so a Step can call it: fresh run dir, then await it."""
    def run() -> None:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")  # the SDK reads ANTHROPIC_API_KEY from the environment
        run_dir = _new_run_dir()
        print(f"  run dir: {run_dir.relative_to(REPO_ROOT)}")
        asyncio.run(coro(run_dir))
    run.__qualname__ = coro.__name__   # so --dry-run names the review, not the closure
    return run


# The ordered pipeline. Order is load-bearing in one place: `s2_roweval` re-fits the champion
# per held-out rev and runs the real `label.py` over it, so it must follow `s2_train` — and it is
# the step that produces the accuracy claim, so a run that skips it has trained a model and
# measured nothing a caller can act on.
#
# The S4 fusion stage sat at the end here until 2026-08-04 and is deleted. It joined S2's
# out-of-fold predictions to S3's anchors and published a (coverage, accuracy) pair on the 2 s
# window grid — a policy **nothing served**: `label.py` never imported it, used a different unit,
# a different abstention rule and a different reason vocabulary. Two accuracy proofs for one
# product, and the more prominent one described the path no customer CSV ever took. Its one
# distinguishing policy, the amplitude band, was then measured at row level and lost to simply
# raising the threshold (`OPERATING_POINTS.md`). `stages/s2_ml/oof.py` and
# `stages/s3_physics/run.py` went with it: their only consumer was that join.
#
# `rate_audit.py` went with them and came BACK the same day, which is the more useful lesson:
# it was deleted for being run.py's neighbour rather than for failing its own test. Deleting a
# driver is not a reason to delete what the driver called, and the check it performs — does an
# anchor describe the body or the sampling grid — never depended on anything fusing.
#
# What S3 does now is three things at three different strengths, and the ordering matters
# because they are not equally defensible: it audits the ANNOTATIONS (strong — the swap rule is
# genuinely independent of them), it bounds a FILE against the model (moderate), and it can gate
# individual WINDOWS against the model (weak, ships OFF, and `s2_roweval` measures it rather
# than arguing about it). `stages/s3_physics/anchors.py` carries the retraction that sizes them.
#
# The `s4_review` agent went with it too, and it was the expensive one — ~$1.10 a run against
# ~$0.09 for `s1_exception`. It judged why fusion could not call a window, so with no fusion
# there is nothing for it to judge.
#
# `s3_label_review` (2026-08-04) is the part of it worth keeping, repointed. The dead agent's
# most valuable verdict was `label_suspect` — if the ground truth is wrong then the pipeline's
# "error" is not one — and that question never depended on fusing anything; it died because
# its QUEUE was a corpus-level join no customer CSV went through. `label_audit` produces the
# same kind of queue against the annotations instead of against the classifier, and had been
# nominating windows that nothing consumed. Both are cheap, and both judge a deterministic
# stage's exception queue rather than producing a number of their own.
#
# `s2_experiment` (2026-08-04) is the third paid step and the one that is NOT of that kind:
# its agents propose and criticise a change to the model itself. It is still not allowed a
# number — `experiment.decide()` gates on measured macro-F1 and neither agent sees the result
# it produced. What it buys is that the champion can change again at all: without it,
# `champion_spec.json` moves only when a human edits it, and the rule for whether a change was
# an improvement lives nowhere.
def build_steps() -> list[Step]:
    return [
        # First, and unconditional. It reads no data at all — it holds `features.swap_counts`
        # (vectorized, all windows at once) to `rest.swap_count` (scalar, one window) on random
        # signals, so it is the one check that costs ~2 s and cannot be invalidated by a change
        # of corpus. Running it before S1 means a broken vectorization stops the run in seconds
        # rather than after the minutes it takes to clean.
        Step("verify_features",
             [PY, "-m", "stages.s2_ml.verify_features"]),
        # Same class as the check above and the same reason for sitting here: it reads no
        # data, costs milliseconds, and holds a mechanism to its own reference rather than to
        # a corpus. What it protects is `breakdown`'s staleness flag, whose failure mode is
        # silence — a check that has quietly stopped firing looks exactly like a pipeline with
        # nothing wrong, and every complaint it can make is exercised against a case built to
        # trip it (`freshness.py`, negative control at the bottom of the file).
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
        # from raw and compared to HUROTICS' own export on every paired recording. Needs no
        # model, so it runs here rather than after `s2_train` — a broken bridge should stop the
        # run before it spends minutes fitting 400 trees on features it cannot reproduce.
        Step("verify_transform",
             [PY, "-m", "stages.s2_ml.verify_transform"], verify=True),
        Step("s2_train",
             [PY, "-m", "stages.s2_ml.train"],
             gate=RUNS / "s2_ml" / "locoeval.md"),
        # The champion/challenger cycle, restored 2026-08-04. It needs a champion report to
        # reason from and a refit to follow a promotion, so it sits directly after `s2_train`
        # and before everything that consumes the champion. The gate is `experiments.jsonl`
        # rather than a promotion: a cycle whose challenger lost did its job, and the ledger
        # entry recording WHY is the artifact — it is what stops the next cycle spending
        # another fit on the same idea.
        Step("s2_experiment",
             fn=_agent_step(_s2_cycle), agent=True,
             gate=RUNS / "s2_ml" / "experiments.jsonl"),
        # What a caller actually GETS, which is a different claim from the one above: between
        # the four columns and the answer sit gap segmentation, FIR decimation, the rest
        # reference, 42 features and the ensemble, and comparing columns exercises none of it.
        # Labels the same recording both ways and diffs row for row, then sweeps `data/raw` and
        # reports what abstains and why — "the serve path works" and "the serve path works on
        # 86% of the corpus" are different claims and only one of them is true. Needs the
        # champion, so it follows `s2_train`.
        Step("verify_serve",
             [PY, "-m", "stages.s2_ml.verify_serve"], verify=True),
        # The accuracy claim. Runs the real `label.py` per held-out rev over ~1.27M rows and
        # writes the curve, the per-subject table and the reason validation that
        # `OPERATING_POINTS.md` quotes. It was never a pipeline step before 2026-08-04, which
        # meant a full run produced a fusion report nobody shipped and never regenerated the
        # numbers the deliverable is actually sold on.
        Step("s2_roweval",
             [PY, "-m", "stages.s2_ml.roweval"],
             gate=RUNS / "s2_ml" / "roweval_loro.md"),
        # The same claim for the route a caller actually uses. `s2_roweval` scores the
        # annotated `lpf_view` export and `verify_serve` shows the raw route matches it, so
        # the raw path's accuracy was an inference off two artifacts rather than a
        # measurement. This labels the raw device CSVs that have a human annotation and
        # scores them against it. Refuses the lockbox in code, so it is repeatable —
        # unlike `roweval --lockbox`, it may run on every pipeline pass.
        Step("s2_raweval",
             [PY, "-m", "stages.s2_ml.raweval"],
             gate=RUNS / "s2_ml" / "raweval.md"),
        # S3's own opinion, pointed at the ANNOTATIONS rather than at the classifier. It calls
        # `anchors.trial_anchors` in process, so it needs no anchors artifact. It also emits
        # `label_audit_windows.jsonl` — the specific windows a human should adjudicate with
        # `inspect_window`, which is what makes a trial-level flag actionable instead of
        # leaving a reader to binary-search 60 windows by hand.
        Step("s3_label_audit",
             [PY, "-m", "stages.s3_physics.label_audit"],
             gate=RUNS / "s3_physics" / "label_audit.md"),
        # ...and the consumer for those nominations. The audit deliberately stops at "go
        # look" — the swap rule can be wrong about a window, so it hands over the raw trace
        # instead of a decision — which left its flags sitting unadjudicated. This assigns
        # each flagged trial a cause and says whether a person is needed. It reads
        # `label_audit.json`, so it follows the audit; it publishes no number, so it can
        # never become a second proof competing with `s2_roweval`.
        Step("s3_label_review",
             fn=_agent_step(_s3_label_review), agent=True),
        # Is an anchor describing the body, or the sampling grid? Restored 2026-08-04 with
        # its own CLI: it was deleted with S4 because its DRIVER (`s3_physics/run.py`) also
        # built the fusion join's anchor table, but the audit itself answers a question that
        # has nothing to do with fusion. §2.3 is the cost of not asking it — an experiment
        # once read clustering that partitioned by acquisition rate and was measuring the
        # clock. `gyro_energy` is the negative control and is expected to FAIL.
        Step("s3_rate_audit",
             [PY, "-m", "stages.s3_physics.rate_audit"],
             gate=RUNS / "s3_physics" / "rate_audit.md"),
        # The serve path's file-level sanity bounds, run against their own controls rather
        # than against the corpus. `--calibrate` regenerates the numbers the bounds are
        # sited against; `--control` injects three synthetic channel faults into a clean
        # recording and reports which bounds catch them. The second half is the point: a
        # bound that has never fired on real data is not evidence that the data is clean,
        # so the controls are what show it can fire at all (§11.1, same argument as
        # `gyro_energy`). Needs the champion, so it follows `s2_train`.
        Step("s3_plausibility",
             [PY, "-m", "stages.s3_physics.plausibility", "--calibrate", "--control"],
             gate=RUNS / "s3_physics" / "plausibility.json"),
        # Last, and it reads every stage above rather than producing anything of its own.
        # Deterministic, so it runs on a free spine too: the run that most needs one page
        # saying what happened is the one nobody paid for an agent to review.
        Step("breakdown",
             [PY, "-m", "stages.breakdown"],
             gate=RUNS / "breakdown.md"),
    ]


def select(steps: list[Step], with_agents: bool, with_verify: bool,
           start_at: str | None) -> list[Step]:
    out = [s for s in steps
           if (with_agents or not s.agent) and (with_verify or not s.verify)]
    if start_at is not None:
        keys = [s.key for s in out]
        if start_at not in keys:
            # Naming an opt-in step without its flag is the common mistake, and it is not a
            # typo — say which flag brings it back rather than listing the rest.
            missing = next((s for s in steps if s.key == start_at), None)
            if missing is not None:
                flag = "--with-agents" if missing.agent else "--verify"
                sys.exit(
                    f"[run] --from {start_at}: that step exists but is not selected — it "
                    f"needs {flag}.\n"
                    f"      python run_pipeline.py --from {start_at} {flag}"
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
        # In-process, so an exception is the failure signal rather than an exit code. Caught
        # here for the same reason a non-zero rc is: one step failing must stop the run with a
        # resume hint, not unwind through the runner and lose which step it was.
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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the H-CARE pipeline end to end, from data/raw to runs/breakdown.md.")
    ap.add_argument("--with-agents", action="store_true",
                    help="also run the paid agent step (needs the API key)")
    ap.add_argument("--verify", action="store_true",
                    help="also run the slow raw-path verifications (verify_transform, "
                         "verify_serve) — run them after any change to transform.py or the "
                         "serve dispatch")
    ap.add_argument("--from", dest="start_at", metavar="KEY",
                    help="resume at this step, skipping everything before it (see --dry-run)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print each command without running it")
    args = ap.parse_args()

    # Stage and agent text carries characters the cp949 console cannot encode; never let a stray
    # print crash a run that has already been paid for. See stages/console.py.
    use_replacement_encoding()

    selected = select(build_steps(), args.with_agents, args.verify, args.start_at)

    # A dry run reads nothing and spends nothing, so it must not require raw data or a key —
    # it is now also the way to see the step keys `--from` accepts, which has to work on a
    # checkout with no data in it yet.
    if not args.dry_run:
        preflight(selected)

    n_agents = sum(s.agent for s in selected)
    n_verify = sum(s.verify for s in selected)
    print(f"[run] {len(selected)} steps ({n_agents} agent, {n_verify} verify) "
          f"{'[DRY RUN]' if args.dry_run else ''}")

    started = time.time()
    for step in selected:
        if not run_step(step, args.dry_run):
            sys.exit(f"\n[run] STOPPED at {step.key}. Fix the failure, then re-run with "
                     f"`--from {step.key}` to resume, or without it to redo the spine.")

    print(f"\n[run] done: {len(selected)} steps in {time.time() - started:.1f}s")

    # The staleness check used to live here, against `runs/s4_fusion` by name. It moved out with
    # the stage rather than being repointed: `breakdown` runs last and already calls the same
    # `check_all` over every `runs/*` directory carrying an `_inputs.json`, so a hardcoded second
    # copy could only ever cover less than the generic one and go stale the same way this one did.
    report = RUNS / "breakdown.md"
    if report.exists():
        print(f"[run] report -> {report.relative_to(REPO_ROOT)}")
    if n_agents:
        print("[run] agent costs are in each run's runs/<date>_runN/costs.json")


if __name__ == "__main__":
    main()
