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
python run_pipeline.py --keys                   # the step keys, one line each
python run_pipeline.py --from s2_train          # resume at a step key
python run_pipeline.py --dry-run                # print argv per step, run nothing
```

**`--keys` is the menu for `--from`** — every key in run order, what the step answers, and
roughly what it costs you to wait for. It reads nothing and spends nothing, so it works on a
checkout with no data and no key, and it lists *all* the steps rather than the selected
ones: finding out an agent step exists is the thing you came to look up.

```
16 steps, in run order. `--from KEY` starts at one and runs everything after it.

  verify_features   short       vectorized feature path vs its scalar reference
  verify_freshness  short       the staleness checker still fires (self-test)
  s1_census         medium      inventory data/raw: files, sessions, channels
  s1_clean          long        raw -> data/clean parquet + per-file channel trust
  s1_exception      medium      triage the clean stage's exception queue (agent)
  verify_transform  long        lpf_view columns rebuilt from raw vs the vendor export
  s2_train          long        fit the champion and LOCO-evaluate it
  s2_experiment     super long  champion/challenger cycle; code decides promotion (agent)
  verify_serve      super long  one recording labelled both ways, row for row
  s2_roweval        super long  THE accuracy claim: real label.py per held-out rev
  s2_raweval        super long  the same claim on the raw device route
  s3_label_audit    medium      physics vs annotations: which trials contradict themselves
  s3_label_review   medium      assign a cause to what the audit flagged (agent)
  s3_rate_audit     medium      is an anchor describing the body or the sampling grid?
  s3_plausibility   medium      file-level sanity bounds, checked against injected faults
  breakdown         short       one page over every stage above
```

| bucket | roughly | why a step lands there |
|---|---|---|
| `short` | seconds | reads little or no data |
| `medium` | up to a minute or so | one pass over the corpus, or one agent turn |
| `long` | several minutes | fits a model, or rebuilds features corpus-wide |
| `super long` | tens of minutes | refits per held-out rev, or a full fit inside a cycle |

**The bucket is a relative scale, not a measurement.** Every step scales with how much raw
data is in `data/raw`, so a fixed wall-clock number would be one machine on one corpus
quoted as a property of the pipeline — the stale-evidence failure this repo keeps finding.
What the buckets claim is their *order*, which survives a corpus that grows.

So `--from s2_train` is an example, not the only key. `Step.desc` and `Step.runtime` live in
`run_pipeline.py` next to the step they describe, which makes `--keys` the live copy —
prefer it if it ever disagrees with the block above, and note `--keys` refuses to print at
all if a step carries a bucket that is not one of the four. There is no way to run a
*single* step through the runner: `--from` always runs to the end, so call the module
directly (§3) when you want one stage alone.

Flags are hand-parsed in `run_pipeline.py` (`_parse_args`) — anything unrecognized exits
non-zero with the usage line rather than being silently ignored. `--from` accepts both
`--from KEY` and `--from=KEY`, because half-supporting one spelling is how a resume quietly
becomes a full re-run. Naming an agent step without `--with-agents` names the flag it needs
rather than listing the rest. `--with-agents` is the only opt-in: every deterministic step,
verifications included, runs on the free spine.

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
| 3 | `s1_census` | `python -m stages.s1_clean.run` | `runs/s1_census/manifest.jsonl` | `runs/s1_census/`: `census.md`, `manifest.jsonl` |
| 4 | `s1_clean` | `python -m stages.s1_clean.clean` | `runs/s1_clean/segments.jsonl` | `data/clean/<session>/<stem>.channel_trust.json` — **the only persisted per-file product**; the canonical-grid frame is measured and dropped; `runs/s1_clean/`: `clean_report.md`, `segments.jsonl`, `observations.jsonl`, `quarantine.jsonl` |
| 5 | `s1_exception` (agent) | in-process `_s1_exception` | — | `runs/<date>_runN/exceptions_review.jsonl` (+ `exceptions_review_raw.txt` when the output does not parse) |
| 6 | `verify_transform` | `python -m stages.s2_ml.verify_transform` | — | nothing; stdout, `sys.exit(1)` on mismatch |
| 7 | `s2_train` | `python -m stages.s2_ml.train` | `runs/s2_ml/locoeval.json` | `runs/s2_ml/`: `locoeval.md`, `locoeval.json`, `model_meta.json`, `_inputs.train.json`, and `champion.joblib` **on a refit only** — an experiment run (§5) writes metrics without the estimator |
| 8 | `s2_experiment` (agent) | in-process `_s2_cycle` | `runs/s2_ml/experiments.jsonl` | `runs/<date>_runN/`: `proposal.json`, `critic_review.json` (+ `_rev1` variants on a retry); `runs/s2_ml/`: `experiments.jsonl`, `proposals.jsonl`, `champion.json`. **On promotion**: re-runs `python -m stages.s2_ml.train` and rewrites `stages/s2_ml/champion_spec.json` |
| 9 | `verify_serve` | `python -m stages.s2_ml.verify_serve` | — | nothing; stdout, `sys.exit(1)` on failure |
| 10 | `s2_roweval` | `python -m stages.s2_ml.roweval` | `runs/s2_ml/roweval_loro.json` | `runs/s2_ml/`: `roweval_loro.md`, `roweval_loro.json`, `transitions_loro.md`, `transitions_loro.json` |
| 11 | `s2_raweval` | `python -m stages.s2_ml.raweval` | `runs/s2_ml/raweval.json` | `runs/s2_ml/`: `raweval.md`, `raweval.json` (fits per-rev models into temp dirs) |
| 12 | `s3_label_audit` | `python -m stages.s3_physics.label_audit` | `runs/s3_physics/label_audit.json` | `runs/s3_physics/`: `label_audit.md`, `label_audit.json`, `label_audit_windows.jsonl` |
| 13 | `s3_label_review` (agent) | in-process `_s3_label_review` | — | `runs/<date>_runN/`: `label_review.jsonl`, `label_review_raw.txt` |
| 14 | `s3_rate_audit` | `python -m stages.s3_physics.rate_audit` | `runs/s3_physics/rate_audit.json` | `runs/s3_physics/`: `rate_audit.md`, `rate_audit.json` |
| 15 | `s3_plausibility` | `python -m stages.s3_physics.plausibility --calibrate --control` | `runs/s3_physics/plausibility.json` | `runs/s3_physics/plausibility.json` |
| 16 | `breakdown` | `python -m stages.breakdown` | `runs/breakdown.md` | `runs/breakdown.md` |

(agent) = `--with-agents` only · every other step runs on the free spine

### Gates are the `.json`, never the `.md`

Every gate names the machine-read artifact, because that is the one whose absence actually
breaks the run: `breakdown` and the next stage parse the `.json`, and no code anywhere
parses a `.md` under `runs/`. Gating on the rendered report had it backwards — it made the
human-facing copy load-bearing, so deleting a report you had finished reading would make
the pipeline conclude the stage never ran.

**Each `.md` is a pure render of its `.json` twin**, verified function by function: the
render takes exactly the values the twin serializes, so a report holds no number its twin
does not. `runs/breakdown.md` is the exception and the one to read — it is assembled from
every stage's `.json`, not rendered from one of them.

The per-stage reports were **deleted on 2026-08-04** for that reason, `breakdown.md` aside.
They are not suppressed: each stage still writes its `.md`, so any of them comes back the
next time that stage runs. Two artifacts were retired rather than deleted:

- `runs/breakdown.json` — the machine-readable twin of the page, which nothing ever read.
  `breakdown.py` no longer writes it; `build()` still returns the structure.
- `labeled_raw/plausibility.jsonl` — deleted, but `label_all.py` still writes it. It is the
  only per-file plausibility record for the labelled corpus, so the writer stays.

One report cannot be rebuilt by re-running its stage: `roweval_lockbox.md` would need
`roweval --lockbox`, and rev8 is spent (§5). Its numbers survive in the tracked
`roweval_lockbox.json`, and `roweval.render()` will re-render them from it — but do **not**
re-run the stage to get the page back.

### Why the order is what it is

Steps 1 and 2 read no data at all and cost milliseconds, so they run first: a broken
vectorization or a staleness checker that has quietly stopped firing should stop the run in
seconds, not after the minutes it takes to clean.

The other two verifiers are slow because they read the whole corpus, and each sits at the
earliest point it can. `verify_transform` needs no model, so it goes before `s2_train` — a
bridge that cannot reproduce its own features should stop the run before it spends minutes
fitting 400 trees on them. `verify_serve` needs the champion, so it cannot move earlier than
9. Neither is hoisted to the front despite being a check: cheap-and-corpus-free runs first,
expensive runs where its inputs exist.

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
| `run_meta.json` | `_new_run_dir` — UTC start time + `git_sha()` from `runmeta.py`. Much of `runs/` is regenerated in place or unversioned, so each run records the commit that produced it |
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

Each writes `locoeval.md`, `locoeval.json`, `model_meta.json` and `_inputs.train.json` into
its own directory; `breakdown` reads them back. `runs/s2_ml_rf42` and
`runs/s2_ml_seedsweep.json` are the same kind of artifact.

**An experiment does not write `champion.joblib`.** An ablation exists to have its
`locoeval` read next to the champion's, and that is all anything reads: `breakdown` opens
`locoeval.json` and nothing else, and no serve path points at an ablation dir. What each
ablation *was* survives in `model_meta.json`, which carries `dropped_features` alongside
the surviving `features`, so the `breakdown` table and the RF-vs-ET measurement stay
sourced without it. The four estimators already on disk were **deleted on 2026-08-04**
(~95 MB); re-running the command above reproduces one if it is ever genuinely wanted.

Only `runs/s2_ml/champion.joblib` is load-bearing — `label.py` loads it, so it is what
stands between `data/raw` and the labelled CSVs — and a refit still writes it. The rule is
"persist the estimator when something loads it", not "persist it when it is cheap to".

`roweval --lockbox` writes `roweval_lockbox.md/.json` and `transitions_lockbox.md/.json`
instead of the `_loro` pair. **It is single-use** — `rev8` is spent, read twice on
2026-08-03, and must not be read a third time without a new sealed subject. `raweval`
refuses the lockbox in code rather than by remembering a flag, so it may run on every pass.

---

## 6. Staleness plumbing

`freshness.stamp_inputs()` writes the sha256 of every artifact a stage consumed into that
stage's output directory. The stamp is **per stage, not per directory** —
`_inputs.<stage>.json`, with the bare `_inputs.json` still read for stamps written before
the split. Callers today:

| stage | stamp | declares |
|---|---|---|
| `train.py` | `runs/<out>/_inputs.train.json` | `champion_spec.json` |
| `roweval.py` | `runs/s2_ml/_inputs.roweval_loro.json` (or `_lockbox`) | `champion_spec.json` |
| `raweval.py` | `runs/s2_ml/_inputs.raweval.json` | `champion_spec.json` |
| `plausibility.py` | `runs/s3_physics/_inputs.plausibility.json` | `champion.joblib` + `model_meta.json`, **only under `--control`** |
| `rate_audit.py` | `runs/s3_physics/_inputs.rate_audit.json` | nothing — model-free by construction |
| `s1_clean/run.py`, `clean.py` | `runs/s1_*/_inputs.s1_*.json` | nothing — the only upstream is the `data/raw` tree |
| `label_all.py` | `labeled_raw/_inputs.json` | `champion.joblib` + `model_meta.json` |

**Why per stage.** `runs/s2_ml` holds three reports with three separate lifetimes —
`train`'s locoeval, `roweval`'s curve, `raweval`'s raw-path read. One shared stamp let
whichever stage ran last refresh the record for all of them, which is worse than no check:
a promotion between `s2_train` and `s2_roweval` would leave `locoeval.md` describing the
old spec and the stamp reporting clean. The self-test builds exactly that case.

**An empty stamp is a declaration, not an omission.** A stage that reads nothing hashable
still stamps, writing `{}`. That keeps two different things apart: *checked, nothing to
declare* versus *never declared anything*, which used to render identically.

`runs/s2_ml/_inputs.json` is the one pre-split stamp still on disk. It is accurate today
and is still read, and `train.py` removes it on its next run, having written
`_inputs.train.json` in its place. Left alone it would become a stamp with no writer —
nothing refreshes it, so the next spec change would make it complain about a report that
had in fact been rebuilt. `labeled_raw/_inputs.json` keeps the bare name on purpose:
`label_all.py` is the only stage writing there, so there is nothing to collide with.

`breakdown` runs `check_all` over every `runs/*` stage directory — not only the ones
already carrying a stamp, which was the old behaviour and meant the check ran on exactly
the dirs that could pass and skipped the ones that could not. A directory with no stamp now
raises one grouped `unchecked` flag naming all of them. Agent run dirs (`*_run*`) are
excluded on purpose: they are never overwritten (§4), so nothing can go stale under them.

The staleness check used to live in `run_pipeline.py` against `runs/s4_fusion` by name; it
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
