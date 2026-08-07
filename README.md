# h-medi data distillation

**English** · [한국어](README.ko.md)

A staged pipeline that distils raw **h-medi** IMU device logs down to a per-row stand/walk
call with a stated confidence and, where it is unsure, a stated reason —
**deterministic code does the work and Claude agents handle judgment at defined points**,
with every decision logged and reproducible.

Built on the [Claude Agent SDK](https://docs.claude.com/en/docs/agent-sdk/overview) (Python).

> **On the names.** The corpus, the device and the subject of every measurement in this repo
> is **h-medi**. The working directory and the git remote are both `hurotics-autolabel`,
> which predates the pipeline and names the vendor, not the data. HUROTICS
> is the vendor. Anything in `archive/` or in a `runs/keep/agent_runs/*/system_prompt.txt`
> that says *H-CARE* is a frozen record of what was written at the time and is deliberately
> not rewritten — see [What "h-care" still means here](#what-h-care-still-means-here).

---

## What "distillation" means here, concretely

Each stage throws away more than it keeps, and the discard is the point — every column that
survives is one a downstream number can be traced back to.

| from | to | thrown away, on purpose |
|---|---|---|
| a raw device log | the **45-column contract**, `Time` … `Total Gait Length` | the variant-specific tail past index 44, which is a firmware era, not a measurement |
| the 45 columns | the **measured** channels only, on a canonical **100 Hz** grid | everything the firmware *computed* — Cadence, Stride Length, GCP, admittance, PID state |
| a cleaned frame | one **`channel_trust.json`** per file | the frame itself. S1 measures the frame and drops it (see [Data layout](#data-layout)) |
| the measured superset | **4 rotational channels** — `L/R_ang_LPF`, `L/R_angvel_LPF` | every channel encoding linear translation, which reads ~0 on a treadmill while the person is plainly walking (`DOMAIN_NOTES` §5.7) |
| 4 channels × 2 s windows | **38 window features** (`extratrees400_drop_moments38`) | four moment features the ablation ledger could not justify |
| 38 features per window | **one committed call per row**, or an abstention with a named reason | the rows the ensemble was not sure enough about — 15.3% of them at the shipped threshold |

The last row is the one that makes this a distillation rather than a classifier: **coverage
is a deliberate loss**, argued in `OPERATING_POINTS.md`, and the discarded rows leave with
their reason attached rather than silently.

---

## Every command in this repo

Everything else is a module the runner invokes — `python run_pipeline.py --keys` is the
index of those, one line each, and they are documented under [Running it](#running-it) for
when you are iterating on one stage by hand.

| command | what it is for | when |
|---|---|---|
| `python run_pipeline.py` | **the pipeline.** `data/raw` → `runs/breakdown.md`, every stage gated on its output artifact. `--with-agents` adds the paid reviews, `--from KEY` resumes | after new data lands, or any change to a stage |
| `python -m stages.s2_ml.label FILE` | **the deliverable.** One recording in, one row out per row in, carrying `Label` / `guess` / `confidence` / `ambiguous` / `reason`. Takes a raw device log or an `lpf_view` file | labelling one recording |
| `python -m stages.s2_ml.label_all` | the deliverable over the whole of `data/raw` in one sweep, same code path per file. Also step 16 of the pipeline, so a full run produces it | labelling the corpus off a champion you did not just fit |
| `python -m stages.s1_clean.validate FILE` | **is this recording usable at all?** Label-free, model-free pre-flight: rate, gaps, segment length, channel presence. Answers before you spend anything | a new recording arrives and you want to know if it can be scored |
| `python -m stages.s3_physics.inspect_window REV TRIAL --t SECONDS` | **adjudicate one suspect window by eye.** Prints the raw interleg trace and the ±1° crossings the swap rule counted, around one timestamp | a review flagged a window as a suspected mislabel and a person has to settle it |

The heading carries no count on purpose. It said "Five." once, drifted, and the drift is
recorded in `archive/needtowrite.md` §4 as a defect worth not repeating.

---

## Read these first, in this order

| file | what it is |
|---|---|
| **`DOMAIN_NOTES.md`** | Everything the h-medi corpus taught us the hard way. Injected into every agent's prompt. **Read before touching any data.** |
| **`caveats.md`** | What this pipeline is shaky about: thin constants, accepted imperfections, unverified paths, deliberate omissions. Read before trusting a number. |
| **`OPERATING_POINTS.md`** | The abstention threshold: what each preset costs and buys, and why the default is 0.85. |
| this file | How to run it. [`README.ko.md`](README.ko.md) is the Korean twin of this page. |
| **`RUNBOOK.md`** | What each command *touches*: every input path, every step's gate, every file it writes. Derived by reading the code, not from the other docs. Read it when this file says what to run and you need to know what landed where. |
| **`runs/README.md`** | The `runs/` tree itself — what is safe to delete, what can never be rebuilt, and why. |

`DOMAIN_NOTES.md` is not background reading — it is the reason this pipeline is shaped the
way it is. Every entry carries a provenance tag: **[measured]** (reproducible by re-running
a stage), **[reported]** (a human said so, unverified), **[decided]** (a design choice),
**[open]** (known unknown). If you only read one thing, read §1.3 — the column prefix is a
lie, and positional indexing silently swaps a firmware step counter for locomotion state on
part of the corpus without ever raising an error.

---

## Setup

Requires **Python 3.10+**.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope Process RemoteSigned
pip install -r requirements.txt
Copy-Item .env.example .env
```

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

The API key comes from the [Console](https://platform.claude.com). The SDK reads it from
the process environment and `run_pipeline.py` calls `load_dotenv()`, so `.env` is enough.

> **Set a monthly spend cap in the Console.** That rail cannot be enforced from inside this
> repo — nothing here can stop an agent step from costing what it costs.

**You must activate the venv in every new terminal.** Forgetting is the single most common
source of `ModuleNotFoundError` here.

---

## Data layout

```
data/raw/<YYYYMMDD[_n]>/*.csv       unlabeled h-medi device logs, exactly as the device wrote them
data/labeled/rev*/csv/*.csv         the golden annotated corpus, never mixed into raw
data/clean/<session>/
    <stem>.channel_trust.json       S1 output, and the ONLY thing it persists per file: the
                                    measured gyro unit and Deg->Gyro permutation for that
                                    recording, plus which axes abstained
```

**S1's product is a verdict, not a copy of the corpus.** Clean resamples each file onto the
canonical 100 Hz grid, keeps measured columns only and normalizes gyro to deg/s — then
*measures the frame and drops it*. Nothing downstream reads a cleaned frame: `label.py`
rebuilds features from the raw log at label time and consults that file's
`channel_trust.json` to know which axis it may read, and S2 trains from `data/labeled`.
A persisted second copy of the signal would be a second thing to keep in step with raw.

**Session date comes from the FOLDER, not from the filename or any column** (§manifest).
Filenames carry a date that is sometimes the export date and sometimes wrong; the directory
is the only metadata this corpus trusts.

**A `Label` column appearing under `data/raw/` is a contamination event, not a schema
variant.** Raw is unlabeled by definition — a `Label` there means annotated data has been
copied into the input tree, and every accuracy number computed afterwards is worthless.

**Quarantine is a ledger, not a deletion.** A file S1 cannot clean is recorded in
`runs/regen/s1_clean/quarantine.jsonl` with the reason and stays on disk. Nothing in this
pipeline deletes raw data.

### What is and is not in git

`data/` in its entirety, `.env`, `tracker/`, and anything new under `archive/` are
gitignored. **No h-medi recording leaves the machine via git.**

`runs/` and `labeled_raw/` are filtered on **size**, not on worth: what stays out is the
heavy binaries — fitted models, per-row scores, and the few-GB per-session labelled CSVs,
all of which `train.py` and `label_all.py` rebuild on demand. Everything else there is
small and is evidence for a claim made in code or in `OPERATING_POINTS.md`, so it travels
with the claim. See `.gitignore`, which explains itself.

**That is a different question from `runs/regen` vs `runs/keep`,** which splits on whether a
full run can rebuild a file at all — `rm -rf runs/regen` is safe and sometimes the right
move; nothing under `runs/keep` can be recovered at any price.
`runs/keep/s2_ml/roweval_lockbox.json` is the rev8 read, with the superseded 2026-08-03 one
kept beside it as `roweval_lockbox.2026-08-03.json` — a re-read replaces the artifact but
never the record,
`runs/keep/s2_ml/experiments.jsonl` is the only record of how the feature set got to where
it is, and `runs/keep/ablations/` holds the LOCO evaluations behind `champion_spec.json`'s
rejected lines. `runs/README.md` is the index; `runslayout.py` is the single definition of
every path under `runs/`, imported rather than restated by each stage.

---

## Running it

### The whole thing

```powershell
python run_pipeline.py                  # the deterministic spine, verifications included
python run_pipeline.py --with-agents    # + the paid agent reviews (spends API credit)
python run_pipeline.py --keys           # every step key, with one line on what it answers
python run_pipeline.py --dry-run        # what would run, in order, without running it
```

Each step is its own subprocess and must produce its **gate artifact** before the next
starts — a stage that exits 0 without writing anything stops the run rather than letting
the next stage read a file from a previous run and quietly report last week's numbers. A
failure prints `--from <key>` to resume at that step.

`--dry-run` reads nothing and spends nothing, so it works on a checkout with no data in it
yet. It is also the way to see the step keys `--from` accepts.

The individual commands below are what the runner invokes; run them by hand when iterating
on one stage. If you do, note that `runs/regen/` is overwritten in place, so re-running one
stage alone leaves the downstream reports describing inputs that no longer exist.
`freshness.py` stamps the sha256 of every artifact a stage consumed into
`_inputs.<stage>.json` in that stage's output directory — **one stamp per stage, not per
directory**, so `train`, `roweval` and `raweval` sharing `runs/regen/s2_ml` cannot vouch for
each other. `stages.breakdown` checks every directory `runslayout.checkable_dirs()` names at
the end of every run, stamped or not: a stage that never declared its inputs raises an
`unchecked` flag rather than reading as clean. **It reports; it does not delete.**

`RUNBOOK.md` §3 has the whole thing as one table — every step, its gate, and every file it
writes.

### The `.md` is opt-in; the `.json` is not

Every stage writes its machine-readable artifact on every run and renders the human-facing
report **only under `--report`**:

```powershell
python -m stages.s1_clean.run --report      # adds census.md beside manifest.jsonl
python -m stages.s2_ml.train  --report      # adds locoeval.md beside locoeval.json
```

The flag is registered from one place, `stages/report.py`, so it cannot drift between
stages, and each stage's stdout names whichever artifact it actually wrote. **A gate always
names the machine-read artifact, never the `.md`** — nothing in this repo parses a `.md`
under `runs/`, so gating on a rendered page would have made deleting a report you had
finished reading look like a stage that never ran. Each `.md` is a pure render of its twin
and holds no number the twin does not.

`runs/breakdown.md` is the deliberate exception to `--report`: it is assembled from every
stage's `.json` rather than rendered from one of them, so it is the one page written to be
read, and it is never behind a flag.

### The checks that read no data

```powershell
python -m stages.s2_ml.verify_features    # vectorized swap counts == the scalar one
python -m freshness --self-test           # the staleness checker's own negative control
```

Both run first and unconditionally, because they cost milliseconds and cannot be
invalidated by a change of corpus. The second exists because a check that has quietly
stopped firing looks exactly like a pipeline with nothing wrong.

### S1 — census, then clean

```powershell
python -m stages.s1_clean.run      # measure the corpus, judge nothing -> runs\regen\s1_census
python -m stages.s1_clean.clean    # resample onto the canonical grid -> runs\regen\s1_clean
```

The census produces `manifest.jsonl` (one row per file: session, variant, measured rate,
jitter, gaps, label codes), and `census.md` under `--report`. The clean stage writes the
per-file `channel_trust.json` into `data/clean/`, plus `segments.jsonl`,
`observations.jsonl` and `quarantine.jsonl` into its run directory — and `clean_report.md`
under `--report`. Both default their `--out` to the right place; pass it only to send a run
somewhere scratch.

### S2 — train, then measure what a caller receives

```powershell
python -m stages.s2_ml.train      # champion + leave-one-rev-out window CV -> locoeval.json
python -m stages.s2_ml.roweval    # the ROW-level curve, per-subject table, reason validation
python -m stages.s2_ml.raweval    # the same accuracy, measured on the RAW device route
```

`train.py` scores **windows**, which is the unit the model learns on. A caller labels a CSV
and gets **rows**, and the two differ — rows are scored by averaging every window that
covers them, which changes both the accuracy and the confidence ordering the threshold is
set from. `roweval` runs the real `label.py` path rather than a reimplementation, and is
the number this repo quotes. It also writes `transitions_loro.json`, which measures what the
largest abstention bucket is actually made of.

Under `--lockbox`, `roweval` writes to `runs/keep/s2_ml` instead: that read is spent once
and no re-run legitimately replaces it, so its report and its input stamp live on the side
of the tree nothing rebuilds. The LORO pass refits from the spec on demand and stays in
`runs/regen/s2_ml` with the rest of the stage. **`roweval_lockbox.md` cannot be recovered by
re-running the stage — do not try.** Its numbers survive in the tracked
`runs/keep/s2_ml/roweval_lockbox.json`, and `roweval.render()` re-renders the page from it.

`raweval` closes the last link in the chain: `roweval` measures the `lpf_view` export,
`verify_serve` shows the raw route agrees with that export row for row but drops `Label`
before comparing — so it proves *equivalence*, never *correctness*. `raweval` scores the
raw route directly against human labels, with the lockbox refused in code rather than by
remembering a flag.

### Verifying the raw path

```powershell
python -m stages.s2_ml.verify_transform   # the four columns, rebuilt from raw
python -m stages.s2_ml.verify_serve       # the same recording labelled BOTH ways, row for row
```

Run both after touching `transform.py`. **An unexercised bridge cannot be wrong out loud,
and it was:** the sagittal axis for the majority variant was wrong until the serve path was
built (`caveats.md` §5). Both are slow, and neither is optional — they run on the spine, not
behind a flag, because slow is not a reason to skip a check.

### S3 — the model-free second opinion

```powershell
python -m stages.s3_physics.label_audit   # which trials contradict their own labels
python -m stages.s3_physics.rate_audit    # body or clock? gyro_energy must FAIL
python -m stages.s3_physics.plausibility  # file-level bounds: calibrate, then fire at faults
```

S3 does three things at three different strengths and the ordering is deliberate, because
they are not equally defensible: it audits the **annotations** (strong — the swap rule has
no trained parameter and never sees a label), it bounds a **file** against the model
(moderate), and it can gate individual **windows** against the model (weak, ships OFF, and
`roweval` measures it rather than arguing about it). `anchors.py` carries the retraction
that sizes them.

`rate_audit` asks whether an anchor describes the body or the sampling grid; `gyro_energy`
is its negative control and is *expected* to fail. `plausibility` injects synthetic channel
faults into a clean recording and reports which bounds catch them — on every run, not under
a flag, because a bound that has never fired is not evidence that the data is clean.

### The breakdown — every stage on one page

```powershell
python -m stages.breakdown     # -> runs/breakdown.md
```

**The last step of every run**, and the only one that reads all three stages plus the
corpus sweep. It **measures nothing** — every number is copied from the artifact that owns
it and the artifact is named beside it, so it cannot disagree with a stage. Read
`breakdown.md` to see the whole pipeline at once; go to the stage that owns a number to
change it. There is no `breakdown.json` — it was the machine-readable twin nothing ever
read, and `build()` still returns the structure if a reader ever turns up.

Two things it does that no stage report can. It **states its own gaps**: an artifact that
was not there is printed in §B with the command that produces it, because a stage that did
not run must not read as a stage with nothing to say. And it raises **§A flags** from
mechanical rules — the lockbox below target, a subject spread this wide, corpus files
clustering on one variant or one run of sessions, and *whether the checked-in prose still
quotes the live headline*. That last one exists because `README.md` and `caveats.md` both
sat at a stale pair for a run that measured something else; both told the reader to prefer
the report, and both were still wrong to a reader who did not. [`README.ko.md`](README.ko.md)
is checked by the same rule, so a translation cannot quietly outlive the number it
translates.

Flags are not verdicts. Each names what to look at and none of them conclude anything.

### Labelling a recording

```powershell
python -m stages.s2_ml.label data\raw\20251024\00321_63_2025_10_24_15_32_0.csv
python -m stages.s2_ml.label some_annotated_trial.csv --preset high_precision
python -m stages.s2_ml.label_all                      # every raw file -> labeled_raw\
python -m stages.s2_ml.label_all --summary-only       # sweep only, writes no per-file CSVs
```

`label_all` also runs as step 16 of `run_pipeline.py`, after the champion is final, so a
full run leaves `labeled_raw/` describing the model everything else in that run describes.
Call it by hand when you want a different threshold, a subset, or a scratch `--out`.

Takes **either** shape of file, dispatched on the header's family marker resolved by name:
a raw device log, or an `lpf_view` file. A raw log is bridged to the four `lpf_view`
channels first, and the run prints which hardware revision was resolved, which two channels
were actually read, and which `channel_trust.json` was consulted.

It **abstains rather than guesses**, with a stated reason and no output file: an unmapped
hardware revision, a file whose measured permutation contradicts the axis about to be read,
a gyro that is not natively deg/s, a final timestamp that cannot carry the filter, or a
missing trust record.

`label_all` is `label.py` in a loop — every file goes through the same `label_csv` entry
point, so a batch run cannot drift from what an operator gets by hand. One file's
abstention never ends the sweep: each is caught, recorded in `abstentions.jsonl`, and the
run continues. Every raw file is accounted for exactly once, asserted rather than hoped.

### The output contract — `Label` vs `guess`

Each labelled CSV **opens with the annotated corpus's own six columns** — `Time`, the four
`*_LPF` features, `Label` — then appends the verdict columns:

| column | meaning |
|---|---|
| `Label` | the committed call: `0` stand, `10` walk, `-1` scored but under the threshold, `255` no window covered the row |
| `guess` | the same call *before* the threshold: `0` or `10` on every row that was scored at all, `255` where nothing was. Never `-1` |
| `confidence` | `max(p, 1-p)` from the ensemble; empty on a `255` row, which was never scored. **Not a calibrated probability** — an ordering, and the only thing it is fit for is comparison against a threshold. Nothing downstream may do arithmetic with it. See `caveats.md` §2.6 |
| `ambiguous` | `True` for every `-1` and `255` row — the plain boolean, if you don't want to decode `Label` |
| `reason` | which doubt: `near_transition`, `weight_shift_or_step`, `model_split`, `posture_shift`, `low_excursion_gait`, `out_of_distribution`, `uncovered`. Empty on a confident row |

> `Label` answers "may I use this row", `guess` answers "what did it think" — and `-1`
> destroys the second to give the first. **`Label != guess` selects exactly the rows that
> were scored and abstained**, which is the review queue with the model's own opinion
> attached rather than stripped out.
>
> The two abstention codes stay apart deliberately: `-1` is the machine analogue of an
> annotator's own "I looked and cannot call it", `255` is `config.LABEL_UNKNOWN_MACHINE` —
> nothing was measured well enough to judge. A reader that lumps them cannot tell an unsure
> model from an absent signal. `--no-verdict` drops everything after `Label`, leaving a file
> indistinguishable in shape from an annotated one, for a reader that needs exactly that;
> `--full` keeps the entire frame. `label` and `label_all` take the same two flags and
> default to the same shape.

One further reason code exists in `label.REASONS` — `amplitude_ambiguous` — and **it cannot
appear in shipped output**, because it is written only when `BAND_ABSTAINS` is on and that
flag defaults to `False`. It stays switchable deliberately: the table that beats the band
scores it against exactly the labels the band exists to distrust, so leaving it off is a
judgement and not a settled loss (`OPERATING_POINTS.md`). A `physics_contradicts` code used
to sit beside it behind a `PHYSICS_CEILING` flag; both are **deleted**, because that one was
settled — the ceiling is worth −11 errors at the shipped point and has nothing at all to
catch at p ≥ 0.95. `roweval` still sweeps a hypothetical ceiling so the retraction stays
measured rather than remembered.

**The call is S2's alone.** The per-row state comes from the ExtraTrees ensemble; the
physics diagnostics are used only to *name* the doubt. So `ambiguous` means "the ensemble
was unsure", never "two opinions disagreed".

---

## What S1 actually does, and why

**Resolves columns by name, never by position.** The `NN_` prefix is a per-file position,
not an identifier. `loco` sits at index 47 in **80 of the 91 raw files** — and in variant
`0fda484e`, the other **11**, index 47 is `Step` [measured 2026-08-04]. `Step` itself sits
at 46 or 47 depending on variant. So `df.iloc[:, 47]` reads a firmware step counter as
locomotion state on 12% of the corpus and never raises. Both are outdated columns (legacy
algorithm output / firmware counter) and pruned by clean. See `DOMAIN_NOTES` §1.3.

**The minority is what makes it dangerous.** A positional read that broke everywhere would
be found on the first plot; one that is right on 80 files and wrong on 11 produces a result
that looks fine and is not.

**Segments at gaps.** Gaps land anywhere. A file is a bag of continuous runs, and the
**segment**, not the file, is the unit of analysis. Nothing is ever resampled across a gap
— that would invent data that was never measured.

**Normalizes rate to 100 Hz.** The corpus has two eras: an earlier ~100 Hz era and a later
500 Hz era. Downsampling uses `scipy.signal.decimate(..., ftype='fir')`, never `[::5]` —
naive decimation folds everything above 50 Hz into the gait band as a full-amplitude fake
signal. See `DOMAIN_NOTES` §2.5 for the measured proof. The odd rates (99.3789 / 99.688 /
99.961 Hz) are *timestamp quantization* (10 + 2⁻ᵏ ms), not different devices — they are
grid-corrected, not discarded.

**Keeps measured channels only.** The device *measures* IMU channels and load cells; it
*computes* Cadence, Stride Length, GCP, admittance, PID state. Computed columns are the
firmware's opinion, not observation. Dropping them collapses the schema variants into one
and removes firmware-version signal from the feature set. The exception mechanism
(`KEEP_EXCEPTIONS` in `stages/s1_clean/config.py`) is **currently empty** — canonical ==
measured, no caveat. The one former exception, `Hip_Deg_L/R`, was cut once measured: 0.991
correlated with the `Deg_Y` already kept, its residual carrying nothing but the firmware's
zeroing convention, and dead on part of the corpus. See `DOMAIN_NOTES` §9.

Note that "measured" means *not app-layer-computed*. The `Deg` channels are the IMU's own
on-sensor fusion output, not a transducer reading — kept, but see `DOMAIN_NOTES` §4.6
before treating them as ground truth.

---

## Architecture

Three stages. The **filesystem is the only interface** between them — no agent-to-agent
messaging, no message queues. `run_pipeline.py` is a dumb sequencer: it shells out to each
stage in order and checks that the gate artifact appeared. It never imports a stage to
"help" it — a sequencer that could reshape a stage's output would be a fourth stage nobody
documented.

| stage | deterministic core | agent role |
|---|---|---|
| **S1 clean** | schema census, rate normalization, gap segmentation, channel trust | `s1_exception` — triage the exception queue |
| **S2 ml** | windowing + features, train + locoeval, row-level labelling with abstention, row and raw evaluation, the corpus sweep | `s2_experiment` — propose a challenger; `locoeval` decides, not the agent |
| **S3 physics** | swap-rule anchors, rate-invariance audit, annotation audit, file-level plausibility | `s3_label_review` — judge what the annotation audit flagged |

### The four non-negotiables

1. **Code does the work; agents judge the work.** Agents never touch data values and never
   crunch numbers themselves.
2. **No silent mutation.** Failures are logged to the quarantine ledger, decisions get
   written rationale, low confidence escalates to `needs_human`.
3. **"Best" is defined by `locoeval`, not by an agent's opinion.** An agent may propose a
   declarative `ExperimentSpec`; `experiment.py` runs it, scores it, applies the promotion
   rule, and logs every outcome — rejections included — to `runs/keep/s2_ml/experiments.jsonl`.
   The champion changes only via a logged, metric-justified promotion.
4. **Every stage closes with a `DOMAIN_NOTES.md` update.** Discoveries become permanent,
   not conversational.

Agents are never authorized to delete raw data, edit `DOMAIN_NOTES.md` without human
review, or retrain the champion. Each writes into its own `runs/keep/agent_runs/<date>_runN/` containing
`run_meta.json` (the commit that produced it), `costs.json` (per-agent spend), and
`system_prompt.txt` (exactly what the agent was told). A `run_log.jsonl` appears alongside
via a `PostToolUse` hook — one line per tool call, so a run whose agent used no tools writes
none.

---

## Status

| stage | state |
|---|---|
| **S1** clean | complete — schema/rate/gaps, gyro unit+axis trust, yaw-drift trust, degenerate-time-base rejection, quarantine ledger; gate passes |
| **S1** exception agent | complete — triages the exception queue into known_expected / novel / needs_human with grounded rationale |
| **S2** ml | complete — windowing/features, leave-one-rev-out CV, per-row labelling with confidence and ambiguity reasons |
| **S2** row + raw evaluation | complete — `roweval` on the annotated export, `raweval` on the raw device route, both leave-one-rev-out |
| **S2** experiment agent | complete — proposes challengers against a logged promotion rule; the ledger records rejections too |
| **S3** physics | complete — swap-rule anchors, rate-invariance verdict recorded for all four anchors |
| **S3** label audit | complete — two trial-level detectors, both model-free |
| **S3** label review agent | complete — assigns a cause to each flagged trial; nominates, never enacts |
| **S3** plausibility | complete — file-level bounds with synthetic-fault controls |
| **S2** corpus sweep | complete — `label_all` on the spine after the champion is final, so `labeled_raw/` describes the run's own model |
| breakdown | complete — final step of every run, reads every stage, states its own gaps, raises mechanical flags |
| handoff docs | complete — `RUNBOOK.md` (every path, command and output, read off the code), `runs/README.md` (what is safe to delete), `archive/README.md` (what was retired and why), [`README.ko.md`](README.ko.md) (the Korean twin of this page) |
| hardening | not started |

---

## What this pipeline is judged on

> **Label stand/walk on any h-medi recording at ≥95% accuracy over the rows it commits to,
> and abstain rather than guess on the rest.** Abstention is a feature: an ambiguous row
> gets a call, a confidence, the reason it is uncertain, and the alternative it was
> weighing.

At the shipped threshold of 0.85, on development subjects held out one at a time:
**coverage 84.66% at accuracy 0.9901, worst subject `rev5` at 0.9525.**

That pair is copied from **`runs/regen/s2_ml/roweval_loro.json`** and **re-running
`s2_roweval` invalidates it.** `stages/breakdown.py` checks mechanically that this file
still quotes the live pair and flags it the moment the champion or the threshold moves —
if the two disagree, believe `runs/regen/s2_ml/roweval_loro.json` and never this table
(`python -m stages.s2_ml.roweval --report` renders it as a page).

> **These are development-subject numbers.** The sealed lockbox subject `rev8` scores
> **0.9269 on the rows it commits to**, at 75.6% coverage — the target is missed there, and
> its stand recall is 0.5622 against a walk recall of 1.0000. Read `caveats.md` §3.2 before
> quoting anything above, along with what is thin, what is unverified, and what was
> deliberately left out.

`rev8` is **spent** — read twice on 2026-08-03 and once on 2026-08-07, all three logged in
`caveats.md` §3.2. It must not be read a fourth time without a new sealed subject.

---

## What "h-care" still means here

Nothing about the data. It was a wrong product name carried in prose, corrected on
2026-08-05; the corpus was always h-medi. The working directory was `h-care-champion-5219cd5`
until 2026-08-07, when it was renamed to `hurotics-autolabel` to match the remote — a
checkout-level move, not an edit. Two places still contain the string, and each is
deliberate:

| where | why it stays |
|---|---|
| `runs/keep/agent_runs/*/system_prompt.txt`, `archive/runs/*/system_prompt.txt` | **Evidence, not documentation.** These files exist to record *exactly what the agent was told*. Editing one would make the record lie about a run that already happened, and every verdict in the same directory was reached under the text as written. |
| `labeled_raw/preset_sweep.json`, `runs/keep/agent_runs/*/run_log.jsonl` | Generated artifacts holding absolute paths, so they carry the directory name as it stood when they were written. Regenerated on the next run; nothing reads the string. |

`archive/needtowrite.md` **was** rewritten, against the general rule that the archive is
left as found — on the precedent that file sets itself for the `rev2_view` → `lpf_view`
rename: the archive exists to be rewritten *from*, and transcribing a name the code no
longer uses carries the defect forward into whatever is written next.
