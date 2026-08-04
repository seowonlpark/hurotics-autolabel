# Runbook — every path, every command, every output

Where the data comes from, what each command does to it, and exactly which files land
where. Derived by reading the code on 2026-08-04, not from the other docs — where this
disagrees with `README.md`, §7 says so.

`README.md` says *what to run*; this file says *what it touches*. Read
`DOMAIN_NOTES.md` before either.

---

## 1. The single entry point

```powershell
python run_pipeline.py                # deterministic spine only  (13 steps)
python run_pipeline.py --with-agents  # + 3 paid agent steps      (16 steps)
python run_pipeline.py --from s2_train          # resume at a step key
python run_pipeline.py --dry-run                # print argv per step, run nothing
```

Flags are hand-parsed in `run_pipeline.py` (`_parse_args`) — anything unrecognized exits
non-zero with the usage line rather than being silently ignored. `--from` accepts both
`--from KEY` and `--from=KEY`, because half-supporting one spelling is how a resume quietly
becomes a full re-run. Naming an agent step without `--with-agents` names the flag it needs
rather than listing the rest.

**`--verify` was removed on 2026-08-04** and its two steps — `verify_transform` (6) and
`verify_serve` (9) — folded into the spine, so `--with-agents` is now the only opt-in.
Both are differential checks rather than stored evidence: `verify_transform` compares a
rebuild against HUROTICS' own export *for that same trial*, and `verify_serve` compares two
code paths against each other, so both regenerate from whatever corpus is present and
neither can go stale the way a recorded number does. What they defend — train/serve skew in
`transform.py` — is a property of the code, which is why the flag stopped earning the chance
to forget it. Passing `--verify` now fails the run rather than quietly skipping the checks,
which is what happened the last time the wiring was lost (§7).

**Preflight** (`preflight`) — skipped entirely under `--dry-run`, so a dry run works on a
checkout with no data and no key:

- `data/raw/**/*.csv` must be non-empty, else exit.
- If any agent step is selected: `ANTHROPIC_API_KEY` in the environment **or** a `.env`
  file must exist.

**Per-step contract** (`run_step`): exit 0 (or, for in-process steps, no exception) **and**
the gate artifact must exist afterwards. A stage that exits 0 without writing anything
stops the run rather than letting the next stage read a file from a previous run and
quietly report last week's numbers. Failure prints `--from <key>` and exits. A step with
`gate=None` is trusted on its exit code alone.

---

## 2. Input paths — what must exist before anything runs

| path | what | in git? |
|---|---|---|
| `data/raw/<YYYYMMDD[_n]>/*.csv` | unlabeled device logs, verbatim. **Session date comes from the folder name**, pattern `^(\d{8})(?:_(\d+))?$` (`s1_clean/config.py`) | no |
| `data/labeled/rev*/csv/annotated_loco_*_trial_*.csv` | the golden annotated corpus; glob in `s2_ml/dataset.py`. `rev8` is the lockbox, `("rev13", 4)` is excluded | no |
| `data/clean/<session>/*.parquet` + `channel_trust.json` | S1 output, S2/S3 input | no |
| `stages/s2_ml/champion_spec.json` | the tracked champion declaration; `train.py` asserts the code matches it | **yes** |
| `.env` | `ANTHROPIC_API_KEY`, loaded by `_agent_step` before each agent run | no |

A `Label` column appearing anywhere under `data/raw/` is a contamination event, not a
schema variant — every accuracy number computed afterwards is worthless.

---

## 3. The 16 steps, in order

Command → gate → everything it writes.

| # | key | command | gate | outputs |
|---|---|---|---|---|
| 1 | `verify_features` | `python -m stages.s2_ml.verify_features` | — | nothing; stdout + exit code |
| 2 | `verify_freshness` | `python -m freshness --self-test` | — | nothing; stdout + exit code |
| 3 | `s1_census` | `python -m stages.s1_clean.run` | `runs/s1_census/census.md` | `runs/s1_census/`: `census.md`, `manifest.jsonl` |
| 4 | `s1_clean` | `python -m stages.s1_clean.clean` | `runs/s1_clean/clean_report.md` | `data/clean/<session>/*.parquet` + per-file `channel_trust.json`; `runs/s1_clean/`: `clean_report.md`, `segments.jsonl`, `observations.jsonl`, `quarantine.jsonl` |
| 5 | `s1_exception` 🔑 | in-process `_s1_exception` | — | `runs/<date>_runN/exceptions_review.jsonl` (+ `exceptions_review_raw.txt` when the output does not parse) |
| 6 | `verify_transform` ⏱ | `python -m stages.s2_ml.verify_transform` | — | nothing; stdout, `sys.exit(1)` on mismatch |
| 7 | `s2_train` | `python -m stages.s2_ml.train` | `runs/s2_ml/locoeval.md` | `runs/s2_ml/`: `locoeval.md`, `locoeval.json`, `model_meta.json`, `champion.joblib`, `_inputs.json` |
| 8 | `s2_experiment` 🔑 | in-process `_s2_cycle` | `runs/s2_ml/experiments.jsonl` | `runs/<date>_runN/`: `proposal.json`, `critic_review.json` (+ `_rev1` variants on a retry); `runs/s2_ml/`: `experiments.jsonl`, `proposals.jsonl`, `champion.json`. **On promotion**: re-runs `python -m stages.s2_ml.train` and rewrites `stages/s2_ml/champion_spec.json` |
| 9 | `verify_serve` ⏱ | `python -m stages.s2_ml.verify_serve` | — | nothing; stdout, `sys.exit(1)` on failure |
| 10 | `s2_roweval` | `python -m stages.s2_ml.roweval` | `runs/s2_ml/roweval_loro.md` | `runs/s2_ml/`: `roweval_loro.md`, `roweval_loro.json`, `transitions_loro.md`, `transitions_loro.json` |
| 11 | `s2_raweval` | `python -m stages.s2_ml.raweval` | `runs/s2_ml/raweval.md` | `runs/s2_ml/`: `raweval.md`, `raweval.json` (fits per-rev models into temp dirs) |
| 12 | `s3_label_audit` | `python -m stages.s3_physics.label_audit` | `runs/s3_physics/label_audit.md` | `runs/s3_physics/`: `label_audit.md`, `label_audit.json`, `label_audit_windows.jsonl` |
| 13 | `s3_label_review` 🔑 | in-process `_s3_label_review` | — | `runs/<date>_runN/`: `label_review.jsonl`, `label_review_raw.txt` |
| 14 | `s3_rate_audit` | `python -m stages.s3_physics.rate_audit` | `runs/s3_physics/rate_audit.md` | `runs/s3_physics/`: `rate_audit.md`, `rate_audit.json` |
| 15 | `s3_plausibility` | `python -m stages.s3_physics.plausibility --calibrate --control` | `runs/s3_physics/plausibility.json` | `runs/s3_physics/plausibility.json` |
| 16 | `breakdown` | `python -m stages.breakdown` | `runs/breakdown.md` | `runs/breakdown.md`, `runs/breakdown.json` |

🔑 = `--with-agents` only · ⏱ = `--verify` only

### Why the order is what it is

Steps 1 and 2 read no data at all and cost milliseconds, so they run first and
unconditionally: a broken vectorization or a staleness checker that has quietly stopped
firing should stop the run in seconds, not after the minutes it takes to clean.

Two later constraints are load-bearing. `s2_roweval` re-fits per held-out rev and runs the
real `label.py`, so it must follow `s2_train` — it is the accuracy claim this repo quotes.
`s2_experiment` sits between them because it needs a champion report to reason from and a
refit to follow a promotion, and everything downstream must describe one champion rather
than two.

`breakdown` is last and reads every stage above rather than producing anything of its own.
It is deterministic, so it runs on the free spine too: the run that most needs one page
saying what happened is the one nobody paid for an agent to review.

---

## 4. Agent run directories

Every agent step calls `_new_run_dir()` → `runs/YYYY-MM-DD_runN`, incrementing `N` until
unused. **Never overwritten.** Each holds:

| file | written by |
|---|---|
| `run_meta.json` | `_new_run_dir` — UTC start time + `git_sha()` from `runmeta.py`. `runs/` is gitignored, so each run records the commit that produced it |
| `system_prompt.txt` | `agents/base.py` — verbatim what the agent was told |
| `costs.json` | `agents/base.py` — per-agent spend |
| `run_log.jsonl` | a `PostToolUse` hook, one line per tool call — **absent when the agent used no tools** |
| the review itself | `exceptions_review.jsonl` / `proposal.json` + `critic_review.json` / `label_review.jsonl` |

The S2 cycle allows exactly one revision (`MAX_PROPOSE_ATTEMPTS = 2`); a second attempt's
files get a `_rev1` suffix via `_attempt_name`. A critic verdict of `reject` — or `revise`
with no attempts left — is terminal: the champion is untouched and the outcome still lands
in `proposals.jsonl`, which is what stops the next cycle spending a fit on the same idea.

---

## 5. Commands outside the spine

### The deliverable

```powershell
python -m stages.s2_ml.label FILE [--out PATH] [--model-dir DIR]
                                  [--preset high_coverage|balanced|high_precision]
                                  [--threshold F] [--full] [--no-verdict]
```

Default output is `<input-stem>_labelled.csv` beside the input. Takes either shape of file
— a raw device log or an `lpf_view` file — dispatched on the header's family marker
resolved by name. It **abstains rather than guesses**, with a stated reason and no output
file.

```powershell
python -m stages.s2_ml.label_all [--raw data/raw] [--out labeled_raw]
                                 [--summary-only] [--limit N]
                                 + the same threshold/shape flags
```

→ `labeled_raw/`: per-session labelled CSVs, `label_summary.csv`, `abstentions.jsonl`,
`plausibility.jsonl`, `label_report.md`, `preset_sweep.json`, `_inputs.json`. One file's
abstention never ends the sweep.

### Pre-flight and adjudication

```powershell
python -m stages.s1_clean.validate FILE [FILE...] [--window-s 2.0]      # stdout only
python -m stages.s3_physics.inspect_window REV TRIAL --t SECONDS \
                                           [--span-s 12.0] [--every-ms 100]   # stdout only
```

### Overrides on spine stages

```powershell
python -m stages.s1_clean.run   --raw data/raw --out runs/s1_census
python -m stages.s1_clean.clean --raw data/raw --out runs/s1_clean
python -m stages.s2_ml.train    --out runs/s2_ml [--window-s F] [--drop FEATURE ...]
python -m stages.s2_ml.roweval  --out runs/s2_ml [--threshold F] [--lockbox]
python -m stages.s2_ml.raweval  --out runs/s2_ml [--threshold F] [--no-baseline]
python -m stages.s3_physics.label_audit  --out runs/s3_physics [--include-lockbox]
python -m stages.s3_physics.rate_audit   --out runs/s3_physics [--include-lockbox]
python -m stages.s3_physics.plausibility --calibrate --control --out runs/s3_physics
python -m stages.s2_ml.experiment [--seed] [--show] [--no-taxonomy] [--out runs/s2_ml]
python -m stages.breakdown --out runs
python -m freshness [--self-test] [--check DIR...]
```

### Ablations

`--window-s` or `--drop` on `train.py` marks the run an experiment rather than a champion
refit: the spec assertion is skipped and the reason is printed. The artifacts still land,
which is the point — an ablation is only worth anything if its `locoeval` can be read next
to the champion's. That is how the ablation dirs were made:

```powershell
python -m stages.s2_ml.train --out runs/s2_ml_abl_zeroing --drop <feature ...>
python -m stages.s2_ml.train --out runs/s2_ml_abl_moments --drop <feature ...>
python -m stages.s2_ml.train --out runs/s2_ml_abl_bottom8 --drop <feature ...>
```

Each writes the same four files (`locoeval.md`, `locoeval.json`, `model_meta.json`,
`champion.joblib`) into its own directory; `breakdown` reads them back. `runs/s2_ml_rf42`
and `runs/s2_ml_seedsweep.json` are the same kind of artifact.

`roweval --lockbox` writes `roweval_lockbox.md/.json` and `transitions_lockbox.md/.json`
instead of the `_loro` pair. **It is single-use** — `rev8` is spent, read twice on
2026-08-03, and must not be read a third time without a new sealed subject. `raweval`
refuses the lockbox in code rather than by remembering a flag, so it may run on every pass.

---

## 6. Staleness plumbing

`freshness.stamp_inputs()` writes `_inputs.json` — the sha256 of every artifact a stage
consumed — into that stage's output directory. Two callers today:

- `train.py` stamps `champion_spec.json` into `runs/s2_ml/`
- `label_all.py` stamps `champion.joblib` and `model_meta.json` into `labeled_raw/`

`breakdown` globs every `runs/*/_inputs.json` and runs `check_all` over all of them. The
staleness check used to live in `run_pipeline.py` against `runs/s4_fusion` by name; it
moved out with that stage rather than being repointed, because a hardcoded second copy
could only ever cover less than the generic one and go stale the same way.

`runs/` is overwritten in place, so re-running one stage by hand leaves the downstream
reports describing inputs that no longer exist. **This check reports; it never deletes.**

---

## 7. Every file in `runs/` has a producer

Both mismatches this file found on 2026-08-04 were fixed at the source rather than
described here, so §3 and §5 now account for every artifact in `runs/` and `labeled_raw/`.
If you find one they do not, that is a defect in one of them.

- `README.md` documented the freshness self-test as `python -m stages.s2_ml.verify_freshness`,
  a module that does not exist. It now says `python -m freshness --self-test`, which is what
  `run_pipeline.py` actually invokes.
- `runs/s3_physics/physics.md` was written by `s3_physics/run.py`, deleted with the S4
  fusion stage on 2026-08-04, and nothing had produced it since. Its rate-invariance table
  was byte-identical to the live `rate_audit.md`, and its rest-trust line was already in
  `DOMAIN_NOTES.md` §10.4 — the only content unique to it was the per-window swap tally.
  That tally now lives in `DOMAIN_NOTES.md` §10.5, next to the argument it is evidence for,
  and the file is deleted. A frozen snapshot sitting among stage reports reads as current
  output; a frozen table inside the section that reasons from it does not.
