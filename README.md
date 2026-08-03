# h-care-agents

A staged pipeline for turning raw H-CARE IMU device logs into a locomotion classification system —
cleaning, ML experimentation, physics-based analysis, and reporting — where **deterministic code
does the work and Claude agents handle judgment at defined points**, with every decision logged and
reproducible.

Built on the [Claude Agent SDK](https://docs.claude.com/en/docs/agent-sdk/overview) (Python).

---

## Read these first, in this order

| file | what it is |
|---|---|
| **`DOMAIN_NOTES.md`** | Everything the corpus taught us the hard way. Injected into every agent's prompt. **Read before touching any data.** |
| **`caveats.md`** | What this pipeline is shaky about: thin constants, accepted imperfections, unverified paths, deliberate omissions. Read before trusting a number. |
| **`PLAN.md`** | Architecture, stage contracts, phase gates, cost rails. |
| this file | How to run it. |

`DOMAIN_NOTES.md` is not background reading — it is the reason this pipeline is shaped the way it
is. Every entry carries a provenance tag: **[measured]** (reproducible by rerunning S1),
**[reported]** (a human said so, unverified), **[decided]** (a design choice), **[open]** (known
unknown). If you only read one thing, read §1.3 — the column prefix is a lie, and positional
indexing silently corrupts most of the corpus without ever raising an error.

---

## Setup

Requires Python 3.10+.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # PowerShell; if blocked: Set-ExecutionPolicy -Scope Process RemoteSigned
pip install -r requirements.txt
Copy-Item .env.example .env         # then put your real key in it
```

```bash
python3 -m venv .venv && source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
cp .env.example .env
```

Get an API key from the [Console](https://platform.claude.com). The SDK reads it from the process
environment; `orchestrator.py` calls `load_dotenv()` so `.env` is enough. **Set a monthly spend cap
in the Console** — that rail cannot be enforced from inside this repo.

You must activate the venv in every new terminal. Forgetting is the single most common source of
`ModuleNotFoundError` here.

---

## Data layout

```
data/
├── raw/<YYYYMMDD[_n]>/*.csv   unlabeled device logs. Session date comes from the FOLDER.
├── labeled/                   golden data. Never mixed into raw/.
└── clean/<session>/*.parquet  S1 output: canonical 100 Hz, 30 measured + 2 documented-exception columns, gyro normalized to deg/s
```

Whole-file rejects are recorded in each run's `quarantine.jsonl` ledger; the raw file
is never copied or moved.

`data/`, `runs/` and `.env` are gitignored. Nothing from HUROTICS leaves the machine via git.

A `Label` column appearing under `data/raw/` is a **contamination event**, not a schema variant.

---

## Running it

### S1 — census (measure the corpus, judge nothing)

```powershell
python -m stages.s1_clean.run --out runs\s1_census
```

Produces `census.md` (human-readable), `variants.json` (every distinct header + the stable prefix
per family), `manifest.jsonl` (one row per file: session, variant, measured rate, jitter, gaps,
label codes).

### S1 — clean (resample onto the canonical grid)

```powershell
python -m stages.s1_clean.clean --out runs\s1_clean
```

Produces `data/clean/**.parquet` (gyro normalized to deg/s) with a per-file
`channel_trust.json` sidecar, plus per-run `segments.jsonl`, `observations.jsonl`,
`quarantine.jsonl`, and `clean_report.md`.

### S2 → S3 → S4 (the labeling chain)

```powershell
python -m stages.s2_ml.train   --out runs\s2_ml       # train + leave-one-rev-out locoeval
python -m stages.s2_ml.oof     --out runs\s2_ml       # out-of-fold predictions + probabilities
python -m stages.s3_physics.run --out runs\s3_physics # swap-rule anchors + rate audit
python -m stages.s4_fusion.run  --out runs\s4_fusion  # call + confidence + reason
```

Run in that order — S4 inner-joins S2's `oof_champion.csv` to S3's `anchors.csv` on
`(rev, trial, segment, t_start_ms)` and refuses to proceed if the two stages windowed
differently. `runs/s4_fusion/fusion.md` is the number the pipeline is judged on.

### Labelling a file (the deliverable)

```powershell
python -m stages.s2_ml.label data\raw\20251024\00321_63_2025_10_24_15_32_0.csv   # raw device log
python -m stages.s2_ml.label some_annotated_trial.csv --preset high_precision    # rev2 view
```

Takes **either** shape of file, dispatched on the header's family marker resolved by name: a
raw device log, or the derived rev2 view. A raw log is bridged to the four rev2 channels
first (`transform.raw_csv_to_features`), and the run prints which hardware revision was
resolved, which two channels were actually read, and which `channel_trust.json` was
consulted. Output is the caller's own rows with `state` / `confidence` / `ambiguous` /
`reason` / `alternative` appended — rows in, rows out.

It **abstains rather than guesses**, with a stated reason and no output file: an unmapped
hardware revision, a file whose measured permutation contradicts the axis about to be read,
a gyro that is not natively deg/s, a final timestamp that cannot carry the filter, or a
missing trust record. 78 of the 91 files under `data/raw` are servable today; the other 13
abstain, correctly.

### Verifying the raw path

```powershell
python -m stages.s2_ml.verify_transform   # the four columns, rebuilt from raw to ~1e-13
python -m stages.s2_ml.verify_serve       # the same recording labelled BOTH ways, row for row
```

`verify_transform` checks the bridge's math on every paired recording (18/18, ≤7.3e-13).
`verify_serve` checks what a caller actually gets: it labels the raw log and its rev2 export
and asserts identical state, confidence and reason on every row (536,590 rows, 0
disagreements), then sweeps `data/raw` reporting what is servable and what abstains, by
reason. Run both after touching `transform.py` — an unexercised bridge cannot be wrong out
loud, and it was: the sagittal axis for the majority variant was wrong until the serve path
was built (`caveats.md` §5).

### Agents

```powershell
python orchestrator.py --phase 2      # S1 exception triage over the latest clean run
python orchestrator.py --phase 4      # S4 review: why a window could not be called
```

The S4 review agent (`agents/s4_fusion.py`) reads `runs/s4_fusion/fused.csv` and judges
the windows the pipeline could not confidently *and* correctly call — abstentions, and
confident errors against the human label, quota'd so neither crowds the other out. It
assigns a cause from a closed vocabulary (`transition` / `label_suspect` / `slow_gait` /
`weight_shift` / `data_quality` / `ambiguous`) plus whether a person is needed, and
`collapse()` derives the disposition deterministically.

`label_suspect` is the verdict worth having: if ground truth is wrong, the pipeline's
"error" is not one. Those are printed as mislabel candidates and always routed to a
human — an agent may nominate a label change, never enact one.

The S1 exception agent (`agents/s1_exception.py`) reads the clean stage's exception queue
(`quarantine.jsonl` + `observations.jsonl`), judges each item — `known_expected` / `novel` /
`needs_human`, grounded in `DOMAIN_NOTES` — and the deterministic wrapper writes
`exceptions_review.jsonl`. It is read-only: the agent judges, code does the work.

Every run gets `runs/YYYY-MM-DD_runN/` containing `run_meta.json` (commit SHA — `runs/` is
gitignored, so each run records the commit that produced it), `run_log.jsonl` (every tool call, via
a PostToolUse hook) and `costs.json` (per-agent spend from the SDK's ResultMessage).

---

## What S1 actually does, and why

**Resolves columns by name, never by position.** The `NN_` prefix is a per-file position, not an
identifier. `loco` sits at index 47 — but in one header variant, index 47 is `Step`. Both are
outdated columns (legacy algorithm output / firmware counter) and pruned by clean. See `DOMAIN_NOTES` §1.3.

**Segments at gaps.** Gaps land anywhere. A file is a bag of continuous runs, and the **segment**,
not the file, is the unit of analysis. Nothing is ever resampled across a gap — that would invent
data that was never measured.

**Normalizes rate to 100 Hz.** The corpus has two eras: an earlier ~100 Hz era and a later 500 Hz era.
Downsampling uses `scipy.signal.decimate(..., ftype='fir')`, never `[::5]` — naive decimation folds
everything above 50 Hz into the gait band as a full-amplitude fake signal. See `DOMAIN_NOTES` §2.5
for the measured proof. The odd rates (99.3789 / 99.688 / 99.961 Hz) are *timestamp quantization*
(10 + 2⁻ᵏ ms), not different devices — they are grid-corrected, not discarded.

**Keeps measured channels only.** The device *measures* IMU channels and load cells; it *computes*
Cadence, Stride Length, GCP, admittance, PID state. Computed columns are the firmware's opinion, not
observation. Dropping them collapses the schema variants into 1 and removes firmware-version signal
from the feature set. The exception mechanism (`KEEP_EXCEPTIONS` in `stages/s1_clean/config.py`) is
**currently empty** — canonical == measured, no caveat. The one former exception, `Hip_Deg_L/R`, was
cut once measured: 0.991 correlated with the `Deg_Y` already kept, its residual carrying nothing but
the firmware's zeroing convention, and dead on part of the corpus. See `DOMAIN_NOTES` §9.

Note that "measured" means *not app-layer-computed*. The `Deg` channels are the IMU's own on-sensor
fusion output, not a transducer reading — kept, but see `DOMAIN_NOTES` §4.6 before treating them as
ground truth.

---

## Architecture

Four stages. The **filesystem is the only interface** between them — no agent-to-agent messaging,
no message queues. `orchestrator.py` is a dumb sequencer; all intelligence lives in the stages.

| stage | deterministic core | agent role |
|---|---|---|
| **S1 clean** | schema census, rate normalization, gap segmentation, channel trust | exception queue only |
| **S2 ml** | windowing + features, train + locoeval, champion/challenger, OOF artifact | propose → critic reviews → metric-gated promotion |
| **S3 physics** | swap-rule anchors, rate-invariance audit | — (deterministic, label-free) |
| **S4 fusion** | join S2 + S3 → call, confidence, reason | judge the abstention queue |

Non-negotiables:

1. **Code does the work; agents judge the work.** Agents never touch data values and never crunch
   numbers themselves.
2. **No silent mutation.** Failures are logged to the quarantine ledger, decisions get written
   rationale, low confidence escalates to `needs_human`.
3. **"Best" is defined by locoeval, not by an agent's opinion.** Champion changes only via a
   logged, metric-justified promotion.
4. **Every phase closes with a `DOMAIN_NOTES.md` update.** Discoveries become permanent, not
   conversational.

Agents are never authorized to delete raw data, edit `DOMAIN_NOTES.md` without human review, or
change the champion outside the S2 promotion path.

---

## Status

| phase | state |
|---|---|
| 0 — skeleton | done |
| 1 — S1 deterministic core | complete — schema/rate/gaps, gyro unit+axis trust, yaw-drift trust, degenerate-time-base rejection, quarantine ledger; gate passes (every raw file accounted) |
| 2 — S1 exception agent | complete — `agents/s1_exception.py` triages the exception queue into known_expected / novel / needs_human with grounded rationale; verified on the real corpus and signed off (gate closed) |
| 3 — S2 loop | running — windowing/features, leave-one-rev-out CV, champion/challenger with a metric-gated promotion rule, OOF artifact |
| 4 — S3 physics | complete — swap-rule anchors on the shared window grid; rate-invariance verdict recorded for all four anchors (gate closed) |
| 5 — S4 fusion | running — call + confidence + reason per window, abstention queue for review |
| 6 — hardening + handoff | not started |

Current corpus counts (files, clean vs quarantined, usable segments/minutes, subjects) live in the
latest `runs/*/clean_report.md` and `census.md`, regenerated every run. A 2026-05 batch is
quarantined for a broken time base (`DOMAIN_NOTES` §2.6).

---

## What this pipeline is judged on

**Label stand/walk on any recording at ≥95% accuracy over the windows it claims, and
abstain rather than guess on the rest.** Abstention is a feature: an ambiguous window
gets a call, a confidence tier, the reason it is uncertain, and the alternative it was
weighing.

Current position, leave-one-rev-out over 4,812 label-pure windows:

| | |
|---|---|
| **coverage 95.1% at confident accuracy 0.9760** | the headline, always reported as a pair |
| high tier | 4,140 windows, accuracy 0.9853 |
| medium | 437 windows, 0.8879 |
| low (abstained) | 235 windows, 0.5702 |

Coverage and accuracy are quoted together because either alone is meaningless — abstain
on all but the easiest window and accuracy reads 1.000. The full operating curve
regenerates every run in `runs/s4_fusion/fusion.md`.

The abstained set is *supposed* to score badly: that gap between 0.9760 and 0.5702 is what
makes the confidence signal informative rather than decorative.

**Read [`caveats.md`](caveats.md) before trusting any of these numbers.** It records what
is thin, what is unverified, what was deliberately left out, and the lockbox that has
never been opened.
