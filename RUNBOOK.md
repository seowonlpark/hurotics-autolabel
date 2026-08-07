# Runbook — every path, every command, every output

Where the data comes from, what each command does to it, and exactly which files land
where. Derived by reading the code on 2026-08-04, not from the other docs — where this
disagrees with `README.md`, §7 says so.

`README.md` says *what to run*; this file says *what it touches*. Read
`DOMAIN_NOTES.md` before either.

---

## 1. The single entry point

```powershell
python run_pipeline.py                # deterministic spine only  (15 steps)
python run_pipeline.py --with-agents  # + 3 paid agent steps      (18 steps)
python run_pipeline.py --keys                   # the step keys, one line each
python run_pipeline.py --from s2_train          # resume at a step key
python run_pipeline.py --dry-run                # print argv per step, run nothing
```

**`--keys` is the menu for `--from`** — every key in run order, what the step answers, and
roughly what it costs you to wait for. It reads nothing and spends nothing, so it works on a
checkout with no data and no key, and it lists *all* the steps rather than the selected
ones: finding out an agent step exists is the thing you came to look up.

```
18 steps, in run order. `--from KEY` starts at one and runs everything after it.

  verify_features   short       vectorized feature path vs its scalar reference
  verify_freshness  short       the staleness checker still fires (self-test)
  verify_rawread    long        the raw-read cache returns the CSV exactly
  s1_census         medium      inventory data/raw: files, sessions, channels
  s1_clean          long        raw -> per-file channel trust; the cleaned frame is dropped
  s1_exception      medium      triage the clean stage's exception queue (agent)
  verify_transform  long        lpf_view columns rebuilt from raw vs the vendor export
  s2_train          long        fit the champion and LOCO-evaluate it
  s2_experiment     super long  champion/challenger cycle; code decides promotion (agent)
  verify_serve      super long  one recording labelled both ways, row for row
  s2_roweval        super long  THE accuracy claim: real label.py per held-out rev
  s3_label_audit    medium      physics vs annotations: which trials contradict themselves
  s3_label_review   medium      assign a cause to what the audit flagged (agent)
  s3_rate_audit     medium      is an anchor describing the body or the sampling grid?
  s3_plausibility   medium      file-level sanity bounds, checked against injected faults
  s2_label_all      super long  label every raw file: corpus coverage, refusals, preset sweep
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
| `data/labeled/rev*/csv/annotated_loco_*_trial_*.csv` | the golden annotated corpus. **Nothing globs for it** — every path is declared row by row in `data/corpus.json` and read by `s2_ml/corpus.py`, which is also where the lockbox subject and the quarantine live (§1.6). A file on disk that the manifest does not declare is not in the corpus | no |
| `data/clean/<session>/<stem>.channel_trust.json` | S1 output, and the only thing it persists per file — the canonical-grid frame is measured and dropped. Read by `s2_ml/transform.py`, which refuses to label a raw log without it. **No cleaned frame is persisted**, and no stage reads one | no |
| `data/cache/<session>/<stem>.parquet` | a **transcode** of the CSV body, written by `s1_clean/rawread.py` — `read_csv(index_col=False)` with the `Unnamed` columns dropped, and nothing else. Not a stage output and not evidence: no resampling, no filtering, no column selection, so there is no code version to track. Keyed on the source's own `(size, mtime_ns)`, stored in the parquet footer; a stale key is a miss. Delete it freely, or set `RAWREAD_CACHE=0` to bypass it | no |
| `data/corpus.json` | the manifest above — subject, session, split, raw pairing and quarantine reason, one row per recording. `s2_ml/corpus.py` validates it on load and an unknown split, a missing file, a duplicate `(subject, session)` or an empty training set is fatal, so nothing downstream runs without it | **yes** |
| `stages/s2_ml/champion_spec.json` | the tracked champion declaration; `train.py` asserts the code matches it | **yes** |
| `.env` | `ANTHROPIC_API_KEY`, loaded by `_agent_step` before each agent run | no |

A `Label` column appearing anywhere under `data/raw/` is a contamination event, not a
schema variant — every accuracy number computed afterwards is worthless.

### Output paths — `runs/regen` vs `runs/keep`

Everything the pipeline writes lands under `runs/`, split by whether a full run can rebuild
it. `runs/README.md` is the index; `runslayout.py` is the single definition every stage
imports, so no path under `runs/` is spelled out twice in code.

| path | rebuilt by a full `python run_pipeline.py`? | holds |
|---|---|---|
| `runs/regen/{s1_census,s1_clean,s2_ml,s3_physics}` | **yes** — `rm -rf runs/regen` is safe | every stage's working output; the fitted model and every metric refitted from `champion_spec.json` |
| `runs/keep/s2_ml` | **no** | `experiments.jsonl`, `proposals.jsonl`, `roweval_lockbox.json`, and every superseded read beside it (`roweval_lockbox.<date>.json`) |
| `runs/keep/ablations` | **no** | `s2_ml_abl_*`, `s2_ml_rf42`, `s2_ml_seedsweep.json`, `s2_ml_featseedsweep.json`, `s2_ml_rowseedsweep.json` |
| `runs/keep/agent_runs` | **no** | one dated dir per paid agent run (§4) |
| `runs/breakdown.md` | **yes** | the deliverable (§5) |

The split cuts *through* S2 rather than around it: `train.py`'s `locoeval` refits on demand,
but the experiment ledger is append-only history and the lockbox is a single-use read. So
`experiment.py` and `roweval.py` each write to two directories — `--out` for the regenerable
half, `runslayout.keep_dir_for(--out)` for the other. Point `--out` at a scratch directory
and both halves follow it there, so a throwaway run cannot write into the real `runs/keep`.

---

## 3. The 17 steps, in order

Command → gate → everything it writes.

| # | key | command | gate | outputs |
|---|---|---|---|---|
| 1 | `verify_features` | `python -m stages.s2_ml.verify_features` | — | nothing; stdout + exit code |
| 2 | `verify_freshness` | `python -m freshness --self-test` | — | nothing; stdout + exit code |
| 3 | `verify_rawread` | `python -m stages.s1_clean.verify_rawread` | — | nothing; stdout, `sys.exit(1)` on mismatch. Reads every raw CSV twice — once directly, once through `data/cache` — and asserts the two frames are identical (names, order, dtypes, values, NaN). **Warms the cache**, so every stage below it stops parsing CSV |
| 4 | `s1_census` | `python -m stages.s1_clean.run` | `runs/regen/s1_census/manifest.jsonl` | `runs/regen/s1_census/`: `manifest.jsonl`, `_inputs.s1_census.json`; `census.md` **only with `--report`** |
| 5 | `s1_clean` | `python -m stages.s1_clean.clean` | `runs/regen/s1_clean/segments.jsonl` | `data/clean/<session>/<stem>.channel_trust.json` — **the only persisted per-file product**; the canonical-grid frame is measured and dropped; `runs/regen/s1_clean/`: `segments.jsonl`, `observations.jsonl`, `quarantine.jsonl`, `_inputs.s1_clean.json`; `clean_report.md` **only with `--report`** |
| 6 | `s1_exception` (agent) | in-process `_s1_exception` | — | `runs/keep/agent_runs/<date>_runN/exceptions_review.jsonl` (+ `exceptions_review_raw.txt` when the output does not parse) |
| 7 | `verify_transform` | `python -m stages.s2_ml.verify_transform` | — | nothing; stdout, `sys.exit(1)` on mismatch |
| 8 | `s2_train` | `python -m stages.s2_ml.train` | `runs/regen/s2_ml/locoeval.json` | `runs/regen/s2_ml/`: `locoeval.json`, `model_meta.json`, `_inputs.train.json`; `locoeval.md` **only with `--report`**; `champion.joblib` **on a refit only** — an experiment run (§5) writes metrics without the estimator |
| 9 | `s2_experiment` (agent) | in-process `_s2_cycle` | `runs/keep/s2_ml/experiments.jsonl` | `runs/keep/agent_runs/<date>_runN/`: `proposal.json`, `critic_review.json` (+ `_rev1` variants on a retry); `runs/regen/s2_ml/`: `experiments.jsonl`, `proposals.jsonl`, `champion.json`. **On promotion**: re-runs `python -m stages.s2_ml.train` and rewrites `stages/s2_ml/champion_spec.json` |
| 10 | `verify_serve` | `python -m stages.s2_ml.verify_serve` | — | nothing; stdout, `sys.exit(1)` on failure |
| 11 | `s2_roweval` | `python -m stages.s2_ml.roweval` | `runs/regen/s2_ml/roweval_loro.json` | `runs/regen/s2_ml/`: `roweval_loro.json`, `transitions_loro.json`, `_inputs.roweval_loro.json`; the two `.md` **only with `--report`** |
| 12 | `s3_label_audit` | `python -m stages.s3_physics.label_audit` | `runs/regen/s3_physics/label_audit.json` | `runs/regen/s3_physics/`: `label_audit.json`, `label_audit_windows.jsonl`, `_inputs.label_audit.json`; `label_audit.md` **only with `--report`** |
| 13 | `s3_label_review` (agent) | in-process `_s3_label_review` | — | `runs/keep/agent_runs/<date>_runN/`: `label_review.jsonl`, `label_review_raw.txt` |
| 14 | `s3_rate_audit` | `python -m stages.s3_physics.rate_audit` | `runs/regen/s3_physics/rate_audit.json` | `runs/regen/s3_physics/`: `rate_audit.json`, `_inputs.rate_audit.json`; `rate_audit.md` **only with `--report`** |
| 15 | `s3_plausibility` | `python -m stages.s3_physics.plausibility` | `runs/regen/s3_physics/plausibility.json` | `runs/regen/s3_physics/`: `plausibility.json`, `_inputs.plausibility.json` |
| 16 | `s2_label_all` | `python -m stages.s2_ml.label_all` | `labeled_raw/label_summary.csv` | `labeled_raw/`: `label_summary.csv`, `abstentions.jsonl`, `plausibility.jsonl`, `preset_sweep.json`, `_inputs.json`, and `<session>/<stem>_labelled.csv` per labelled file — **a few GB, gitignored**; `label_report.md` **only with `--report`** |
| 17 | `breakdown` | `python -m stages.breakdown` | `runs/breakdown.md` | `runs/breakdown.md` |

(agent) = `--with-agents` only · every other step runs on the free spine

### Gates are the `.json`, never the `.md`

Every gate names the machine-read artifact, because that is the one whose absence actually
breaks the run: `breakdown` and the next stage parse the `.json`, and no code anywhere
parses a `.md` under `runs/`. Gating on the rendered report had it backwards — it made the
human-facing copy load-bearing, so deleting a report you had finished reading would make
the pipeline conclude the stage never ran.

`s2_label_all` is the one gate that is not a `.json`, and it is the same rule rather than an
exception: `label_summary.csv` *is* the machine-read artifact there — §5 of `breakdown`
parses it. `preset_sweep.json` would have been the wrong choice despite the extension,
because the sweep is legitimately skipped whenever an abstention gate is on, and a
deliberate skip would then be indistinguishable from a stage that failed.

**Each `.md` is a pure render of its `.json` twin**, verified function by function: the
render takes exactly the values the twin serializes, so a report holds no number its twin
does not. `runs/breakdown.md` is the exception and the one to read — it is assembled from
every stage's `.json`, not rendered from one of them.

### `--report` — the `.md` is opt-in, the `.json` is not

The per-stage reports were **deleted on 2026-08-04** and are **no longer written by
default**. Every stage that renders one takes `--report`:

```powershell
python -m stages.s1_clean.run          --report   # census.md
python -m stages.s1_clean.clean        --report   # clean_report.md
python -m stages.s2_ml.train           --report   # locoeval.md
python -m stages.s2_ml.roweval         --report   # roweval_*.md + transitions_*.md
python -m stages.s2_ml.label_all       --report   # label_report.md
python -m stages.s3_physics.label_audit --report  # label_audit.md
python -m stages.s3_physics.rate_audit  --report  # rate_audit.md
```

The flag is registered from one place — `stages/report.py` — so the rule cannot drift
between stages, and the reason it is safe lives there next to it. The `.json` is always
written either way; `--report` only adds the render beside it. Each stage's stdout names
whichever artifact it actually wrote, so a default run does not print a path to a file that
is not there.

`runs/breakdown.md` is deliberately **not** behind the flag. It is the one page written to
be read, and it is assembled from every stage's `.json` rather than rendered from one of
them — the deliverable, not a redundant second copy of a stage's own output.

Two artifacts were retired rather than made optional:

- `runs/breakdown.json` — the machine-readable twin of the page, which nothing ever read.
  `breakdown.py` no longer writes it; `build()` still returns the structure.
- `labeled_raw/plausibility.jsonl` — deleted, but `label_all.py` still writes it. It is the
  only per-file plausibility record for the labelled corpus, so the writer stays.

One report cannot be rebuilt by re-running its stage: `roweval_lockbox.md` would need
`roweval --lockbox`, and rev8 is spent (§5). Its numbers survive in the tracked
`runs/keep/s2_ml/roweval_lockbox.json` — which is why that file is in `keep/` — and
`roweval.render()` will re-render them from it. Do **not** re-run the stage to get the page
back. The page present today came from the 2026-08-07 read, which was spent on a schema
problem and not on the page; `runs/keep/s2_ml/roweval_lockbox.2026-08-03.json` holds the
earlier read and has no page at all, which is the shape this rule is about.

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

`s2_label_all` is late for the same reason `verify_serve` is: it reads the champion, so it
cannot run before `s2_experiment`'s refit. Run it earlier and every file under
`labeled_raw/` would describe a model the rest of the run has already replaced — and because
its stamp declares `champion.joblib`, the staleness check would then flag the whole directory
on every promotion. It sits immediately before `breakdown`, which is its only consumer.

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

Every agent step calls `_new_run_dir()` → `runs/keep/agent_runs/YYYY-MM-DD_runN`, incrementing `N` until
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

**The incumbent re-baselines itself.** Before proposing anything, `_s2_cycle` fingerprints
the corpus in front of it (`corpus_fingerprint`) and compares it to the one the recorded
champion was measured over. If they differ — a rev added, a class set changed, a window
count moved — it re-measures `champion_spec.json` on the current ground and records *that*
as the incumbent, saying so on stdout. Without it, `decide()` refuses to compare across
bases (correctly) and every subsequent cycle would spend a full fit to settle nothing. It is
not a judgement call and was never a good use of a flag: the fingerprints match or they do
not.

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

This one is **on the spine** as step 16 (`s2_label_all`), run with no flags. The invocations
above are the overrides — `--limit N` for a smoke test, `--summary-only` when you want the
sweep and the summary without the few GB of per-session CSVs, `--out` to write somewhere
that is not the tracked directory. The flagless spine run writes the CSVs, because they are
the deliverable and not merely evidence for the page.

### Pre-flight and adjudication

```powershell
python -m stages.s1_clean.validate FILE [FILE...] [--window-s 2.0]      # stdout only
python -m stages.s3_physics.inspect_window REV TRIAL --t SECONDS \
                                           [--span-s 12.0] [--every-ms 100]   # stdout only
```

### Overrides on spine stages

Every spine stage runs correctly with **no arguments**. What is left below is the whole set,
and each one changes *what is being asked*, not how carefully it is answered:

```powershell
python -m stages.s1_clean.run   [--raw data/raw] [--out runs/regen/s1_census]
python -m stages.s1_clean.clean [--raw data/raw] [--out runs/regen/s1_clean]
python -m stages.s2_ml.train    [--out DIR] [--drop FEATURE ...]   # --drop = an ablation
python -m stages.s2_ml.roweval  [--out DIR] [--lockbox]            # --lockbox = SINGLE USE
python -m stages.s3_physics.label_audit  [--out DIR] [--include-lockbox]
python -m stages.s3_physics.rate_audit   [--out DIR] [--include-lockbox]
python -m stages.s3_physics.plausibility [--out DIR]
python -m stages.s2_ml.experiment [--out DIR]   # prints the ledger; a cycle needs the agents
python -m stages.s2_ml.verify_serve             # takes nothing at all
python -m stages.breakdown [--out runs]
python -m freshness [--self-test]
```

`--out` and `--raw` are paths, `--report` (above) adds a render, and the three that remain —
`--drop`, `--lockbox`, `--include-lockbox` — each name a *different measurement*, which is
why they survive: an ablation, the sealed split, the sealed split.

**What was cut on 2026-08-04, and where it went.** Every one of these was a knob that either
weakened an artifact while keeping its filename, or duplicated a decision the pipeline is
already able to make for itself:

| gone | why | what to do instead |
| --- | --- | --- |
| `train --window-s F` | the window is a field of `ExperimentSpec`; the CLI copy wrote an unlabelled experiment into the champion's own directory | propose it — the ledger records the window *with* its rationale and outcome (§5) |
| `roweval --threshold F` | this stage **is** the accuracy claim, and it already sweeps every threshold into `curve` | read another row of `curve` in `roweval_loro.json`; the headline stays at the declared preset |
| `plausibility --calibrate/--control` | the spine always passed both, and either alone wrote a half `plausibility.json` under the same name — a bound sited against the corpus but never fired at a fault is a number with no evidence | nothing; both always run |
| `experiment --seed` | a re-baseline is not a judgement call: `_s2_cycle` now seeds whenever the incumbent's corpus fingerprint does not match the ground in front of it | nothing; it is automatic, and it prints when it fires |
| `experiment --no-taxonomy` | bought speed by dropping the tiebreaker `decide()` used on a macro-F1 tie. Removing the flag was right, but the default it left behind turned the tiebreaker off permanently — after it, no challenger was ever scored with a taxonomy, so the branch could not fire. The whole row-level taxonomy went on 2026-08-07 (DOMAIN_NOTES §7) | nothing; a macro-F1 tie keeps the incumbent |
| `experiment --show` | it was the only thing this CLI could do that did not need the agents | it is the default now: `python -m stages.s2_ml.experiment` |
| `verify_serve --threshold/--skip-pairs/--skip-sweep` | a differential check with one question; skipping half of it turns a PASS into a PASS about half | nothing; it runs whole |
| `freshness --check DIR...` | naming the dirs by hand is exactly how a stage escapes the staleness check — the same hole `checkable_dirs()` was written to close | nothing; it checks every checkable dir |

### The 38 features

`features.feature_names()` is the single source of truth for the column order. Everything
below is derived from the four cleaned channels in `dataset.FEATURES` — `L_ang_LPF`,
`R_ang_LPF`, `L_angvel_LPF`, `R_angvel_LPF`, thigh angle and angular velocity per leg — over
a 2 s window at 100 Hz, non-overlapping, **never spanning a segment gap**. A window whose
labels are not pure is a `transition`, not a majority vote (`PURITY_MIN = 1.0`).

`feature_names()` emits **42**. The champion drops four for parsimony and trains on **38**.
Read the 38 as *16 per-leg quantities mirrored L/R, plus 6 bilateral* — the pairs are not
independently droppable, since removing one side leaves the model blind on that leg. Ranks
below are the champion's `model_meta.json:feature_importance`, 1–38 of 38.

**Amplitude — how far and how fast the limb moved.** Plain window statistics on each
channel.

| feature | what it is | rank |
| --- | --- | --- |
| `L/R_ang_LPF_std`, `_ptp` | spread and peak-to-peak of thigh angle | 4, 7, 8, 11 |
| `L/R_angvel_LPF_std`, `_ptp`, `_absmean` | same for angular velocity | 10, 12–15, 17 |
| `L/R_angvel_LPF_mean` | signed mean angular velocity — net drift over the window | 34, 38 |

The angle channels get **only** `std` and `ptp`. `mean`/`absmean` on an angle is the
static-offset family, deliberately absent ([features.py:203](stages/s2_ml/features.py#L203)):
it encodes mounting and zeroing convention, not gait, and does not transfer across revisions.

**Bilateral coupling — the strongest signal in the model.** Level walking swings the legs
antiphase; standing does not.

| feature | what it is | rank |
| --- | --- | --- |
| `ang_LR_corr` | L-vs-R angle correlation over the window — strongly negative in gait | **1** |
| `angvel_LR_corr` | same on angular velocity | 16 |
| `angvel_LR_lag_s` | inter-leg timing offset from the cross-correlation peak, clipped to ±`MAX_LAG_S` = 1 s; level gait sits about half a cycle out of phase | 32 |
| `ileg_minhalf`, `ileg_minquarter` | peak-to-peak of the inter-leg difference `L − R − ileg_zero`, taken as the **minimum over 2 / 4 sub-windows** — the whole-window ptp reads high on a single weight shift, so the min is what distinguishes sustained stepping from one shuffle | **2**, 5 |
| `ileg_swaps` | sign changes of that difference with a 1° deadband (`SWAP_DELTA_DEG`) — roughly one per step; a sign change needs both endpoints inside the window | **3** |

Four of the top five are bilateral. This model recognizes gait mainly by the two legs
alternating, not by either leg's own motion.

**Spectral and periodicity — per side, from the angular-velocity channel.**

| feature | what it is | rank |
| --- | --- | --- |
| `L/R_angvel_dom_hz` | dominant frequency inside the gait band | 24, 35 |
| `L/R_angvel_band_frac` | gait band's share of non-DC power | 22, 30 |
| `L/R_angvel_hf_ratio` | share above the gait band, `(3, 15)` Hz | 26, 29 |
| `L/R_cycle_s` | stride period from the first autocorrelation peak in `CYCLE_BAND_S` = `(0.6, 2.5)` s | 28, 33 |

`GAIT_BAND_HZ = (0.13, 3.0)` is set for **this** population — cadence 16–102 steps/min — not
the healthy-adult `(0.5, 3.0)`. That distinction is load-bearing: a genuinely slow walker
under the wrong band reads near-zero exactly like standing for a whole bout. `cycle_s` comes
from autocorrelation rather than the periodogram because it needs sub-bin resolution; the
FFT grid is only `1/window_s` = 0.5 Hz. `hf_ratio` is mostly emptied by the 1 Hz low-pass
upstream, and is kept anyway because dropping it costs worst-subject accuracy
([features.py:115](stages/s2_ml/features.py#L115)).

**Shape and asymmetry — per side, angular velocity.**

| feature | what it is | rank |
| --- | --- | --- |
| `L/R_angvel_posfrac` | fraction of the window with positive angular velocity | 21, 23 |
| `L/R_angacc_rms` | RMS angular acceleration, the differentiated channel | 20, 25 |
| `L/R_angvel_skew` | third standardized central moment | 27, 31 |
| `L/R_angvel_kurt` | fourth moment, excess | *dropped* |
| `L/R_angvel_peakratio` | `max / |min|`, swing-vs-stance asymmetry | *dropped* |

The last two are the champion's drop list — noise-sensitive by construction, ranked 39–42 of
42, and worth 0.0003 macro-F1, i.e. nothing measurable. See `champion_spec.json`.

**Rest-referenced posture — per side, angle.** The only features that consult a per-file
calibration.

| feature | what it is | rank |
| --- | --- | --- |
| `L/R_ang_p95_rest` | 95th percentile of thigh angle **minus this subject's own standing zero** — how far the thigh lifts above their own posture | 18, 19 |
| `L/R_ang_med_rest` | window median minus that zero | 36, 37 |
| `L/R_ang_p95_p05` | p95 − p05, the one member of the family that needs no zero | 6, 9 |

The zero comes from `rest_reference()`: the median over a 3 s resting span
(`REST_ANCHOR_S`), taken from the file's opening if that opening is genuinely rest, else the
best resting span anywhere in the file. If neither exists it falls back to whole-recording
medians and sets `rest_trusted = False`. **Per file, never corpus-wide** — that is the whole
point, since each subject is zeroed differently at cuff-fitting time.

`ang_med_rest` and `angvel_LPF_mean` — ranks 34 and 36–38, the four lowest — are the
"zeroing family," and dropping them is the largest measured regression in the ablation table
below. Low importance is not droppability: ExtraTrees dilutes importance across correlated
features, so a low score means *redundant or weak* and the table cannot tell you which. Only
an ablation can.

### Ablations

`--drop` on `train.py` marks the run an experiment rather than a champion refit: the spec
assertion is skipped and the reason is printed. The artifacts still land,
which is the point — an ablation is only worth anything if its `locoeval` can be read next
to the champion's. That is how the ablation dirs were made:

```powershell
python -m stages.s2_ml.train --out runs/keep/ablations/s2_ml_abl_<name> --drop <feature ...>
```

One directory per ablation, named for what it removes: `zeroing` and `moments` drop a
family, `bottom8` drops the eight lowest-importance features, and the `top10` / `top14` /
`top19` / `top27` family drops everything *outside* the top K to measure how the curve pays
for each block of features. No count is given here on purpose — `breakdown` globs
`s2_ml_abl_*` and tabulates whatever is on disk, so the page cannot fall behind the
directory.

Each writes `locoeval.json`, `model_meta.json` and `_inputs.train.json` into its own
directory, plus `locoeval.md` under `--report`; `breakdown` reads the `.json` back and
nothing else. `s2_ml_rf42`, `s2_ml_seedsweep.json` and `s2_ml_featseedsweep.json` are the
same kind of artifact: the last is the 5-seeds-*within*-each-feature-set sweep that sets the
resolution band every row of the ablation table has to be read against.

`s2_ml_rowseedsweep.json` is the same measurement one unit down, and it exists because the
one above could not stand in for it:

```powershell
python -m stages.s2_ml.rowseedsweep          # ~35 s per seed, 5 by default
```

Five seeds through the **row** path — the same features, folds, threshold and ambiguity gate
that produce the shipped `coverage 84.66% at 0.9901`, with `random_state` as the only thing
that varies. The window sweep measures 5,984 label-pure windows; the headline is 1,244,292
scored rows behind a gate, so its band had to be measured, not inferred. Seed 0 reproduces the headline
exactly, which is what makes the other four a band rather than a different experiment.
`breakdown` prints it directly under the row-level curve.

**It does not re-run the claim.** `roweval.py` stays single-seed for the same reason its
threshold is not a flag: the shipped number is what seed 0 measured. This file is what you
read a *change* to that number against — and at `spread 0.0004` on selective accuracy, most
single-seed deltas anyone will ever quote are inside it.

**An experiment does not write `champion.joblib`.** An ablation exists to have its
`locoeval` read next to the champion's, and that is all anything reads: `breakdown` opens
`locoeval.json` and nothing else, and no serve path points at an ablation dir. What each
ablation *was* survives in `model_meta.json`, which carries `dropped_features` alongside
the surviving `features`, so the `breakdown` table and the RF-vs-ET measurement stay
sourced without it. The four estimators already on disk were **deleted on 2026-08-04**
(~95 MB); re-running the command above reproduces one if it is ever genuinely wanted.

Only `runs/regen/s2_ml/champion.joblib` is load-bearing — `label.py` loads it, so it is what
stands between `data/raw` and the labelled CSVs — and a refit still writes it. The rule is
"persist the estimator when something loads it", not "persist it when it is cheap to".

`roweval --lockbox` writes `roweval_lockbox.md/.json` and `transitions_lockbox.md/.json`
instead of the `_loro` pair. **It is single-use** — `rev8` is spent, read twice on 2026-08-03
and once on 2026-08-07, and must not be read a fourth time without a new sealed subject. The
flag requires `--read-note`: a sealed read has to be argued for in a string that ships inside
the artifact, because a bare flag is too cheap for what it spends. The `raweval` stage,
which also refused the lockbox in code, was removed on 2026-08-07 (`caveats.md` §1.7).

---

## 6. Staleness plumbing

`freshness.stamp_inputs()` writes the sha256 of every artifact a stage consumed into that
stage's output directory. The stamp is **per stage, not per directory** —
`_inputs.<stage>.json`, with the bare `_inputs.json` still read for stamps written before
the split. A stage with no artifact upstream calls `freshness.declare_no_inputs()` instead,
which stamps an empty record — "checked, nothing upstream" rather than "nobody can tell".
Callers today:

| stage | stamp | declares |
|---|---|---|
| `train.py` | `runs/<out>/_inputs.train.json` | `champion_spec.json` |
| `roweval.py` | `runs/regen/s2_ml/_inputs.roweval_loro.json`, or `runs/keep/s2_ml/_inputs.roweval_lockbox.json` under `--lockbox` | `champion_spec.json` |
| `plausibility.py` | `runs/regen/s3_physics/_inputs.plausibility.json` | `champion.joblib` + `model_meta.json` — the controls score the injected faults through the fitted champion |
| `rate_audit.py` | `runs/regen/s3_physics/_inputs.rate_audit.json` | nothing — model-free by construction |
| `label_audit.py` | `runs/regen/s3_physics/_inputs.label_audit.json` | nothing — reads the annotations and the anchors, never the champion |
| `s1_clean/run.py`, `clean.py` | `runs/regen/s1_*/_inputs.s1_*.json` | nothing — the only upstream is the `data/raw` tree |
| `label_all.py` | `labeled_raw/_inputs.json` | `champion.joblib` + `model_meta.json` |

**Why per stage.** `runs/regen/s2_ml` holds reports with separate lifetimes —
`train`'s locoeval and `roweval`'s curve. One shared stamp let
whichever stage ran last refresh the record for all of them, which is worse than no check:
a promotion between `s2_train` and `s2_roweval` would leave `locoeval.md` describing the
old spec and the stamp reporting clean. The self-test builds exactly that case.

**An empty stamp is a declaration, not an omission.** A stage that reads nothing hashable
still stamps, writing `{}`. That keeps two different things apart: *checked, nothing to
declare* versus *never declared anything*, which used to render identically.

`runs/regen/s2_ml/_inputs.json` is the one pre-split stamp still on disk. It is accurate today
and is still read, and `train.py` removes it on its next run, having written
`_inputs.train.json` in its place. Left alone it would become a stamp with no writer —
nothing refreshes it, so the next spec change would make it complain about a report that
had in fact been rebuilt. `labeled_raw/_inputs.json` keeps the bare name on purpose:
`label_all.py` is the only stage writing there, so there is nothing to collide with.

**What gets checked** is `runslayout.checkable_dirs()` — every `runs/regen/*`, plus
`runs/keep/s2_ml`, plus `labeled_raw/`. `breakdown` and `python -m freshness` both read that
one list, so the page and the command cannot disagree about what was covered. Directories
are included **whether or not they already carry a stamp**: selecting on `_inputs.json` ran
the check on exactly the dirs that could pass and skipped the ones that could not, so a
stage that never declared its inputs read as clean. A directory with no stamp now raises one
grouped `unchecked` flag naming all of them, repo-relative — `s2_ml` alone stopped being
unique when `runs/` split, and the reader needs to know which half to go re-run.

**`runs/keep/s2_ml` is in the list and the rest of `keep/` is not**, and the asymmetry is
the point. `agent_runs` and `ablations` are frozen — nothing rewrites them, so nothing can
go stale under them. `keep/s2_ml` holds the spent lockbox, and its stamp is the *only*
warning that a single-use read describes an older `champion_spec.json`. There is no re-run
that would catch it later, so the check fires there or nowhere.

**That gap used to be unclosable, and it is the reason the lockbox was re-read [2026-08-07].**
The 08-03 artifact carried no stamp and no model identity — `curve`, `threshold`, `revs`,
`reasons`, `human_unknown`, and nothing that said which model produced them. `n_features` and
`features` were added 2026-08-04, so every later read is identifiable and that one was not.
The git history was the only evidence and it did not reassure: at `dcb2b7b`, the commit that
versioned the file, `model_meta.json` recorded **42 features** against the champion's
`extratrees400_drop_moments38` at **38**, and the on-disk copy matched that commit's curve
exactly. So the headline was *probably* measured on a model that never shipped, and the
argument for quoting it anyway was the spec's five-seed noise band on LORO — an argument, not
a measurement.

It is a measurement now. The 08-07 read is stamped, carries `schema_version`, `n_features:
38`, the full feature list, and a `read_note` saying why it was spent. **0.9308 → 0.9269**,
which is 13× the 0.0003 the same correction moved the development row — inside the noise
band the old argument appealed to, but not by the margin that argument assumed. Quote
`0.9269`, and see `caveats.md` §3.2 and §3.2b before quoting it at all.

The staleness check used to live in `run_pipeline.py` against `runs/s4_fusion` by name; it
moved out with that stage rather than being repointed, because a hardcoded second copy
could only ever cover less than the generic one and go stale the same way.

`runs/regen/` is overwritten in place, so re-running one stage by hand leaves the downstream
reports describing inputs that no longer exist. **This check reports; it never deletes.**

---

## 7. Every file in `runs/` has a producer

Both mismatches this file found on 2026-08-04 were fixed at the source rather than
described here, so §3 and §5 now account for every artifact in `runs/` and `labeled_raw/`.
If you find one they do not, that is a defect in one of them.

- `README.md` documented the freshness self-test as `python -m stages.s2_ml.verify_freshness`,
  a module that does not exist. It now says `python -m freshness --self-test`, which is what
  `run_pipeline.py` actually invokes.
- `runs/s3_physics/physics.md` — the path as it was, before `runs/` split — was written by
  `s3_physics/run.py`, deleted with the S4 fusion stage on 2026-08-04, and nothing had
  produced it since. Its rate-invariance table
  was byte-identical to the live `rate_audit.md`, and its rest-trust line was already in
  `DOMAIN_NOTES.md` §10.4 — the only content unique to it was the per-window swap tally.
  That tally now lives in `DOMAIN_NOTES.md` §10.5, next to the argument it is evidence for,
  and the file is deleted. A frozen snapshot sitting among stage reports reads as current
  output; a frozen table inside the section that reasons from it does not.
