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

### Agents

```powershell
python orchestrator.py --phase 0
```

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
observation. Dropping them collapses 5 schema variants into 1 and removes firmware-version signal
from the feature set. Documented exceptions live in `KEEP_EXCEPTIONS` in `stages/s1_clean/config.py`,
each with its reason.

---

## Architecture

Four stages. The **filesystem is the only interface** between them — no agent-to-agent messaging,
no message queues. `orchestrator.py` is a dumb sequencer; all intelligence lives in the stages.

| stage | deterministic core | agent role |
|---|---|---|
| **S1 clean** | schema census, rate normalization, gap segmentation, channel trust | exception queue only |
| **S2 ml** | train + locoeval, champion/challenger | propose → critic reviews → metric-gated promotion |
| **S3 physics** | anchor features, plots | read plots, write hypotheses with provenance |
| **S4 report** | — (Read/Grep only) | cross-reference, label audit, flag anomalies |

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
| 1 — S1 deterministic core | deterministic core complete — schema/rate/gaps, gyro unit+axis trust, yaw-drift trust, quarantine ledger; gate passes (every raw file accounted). Exception agent (Phase 2) not yet built |
| 2 — S1 exception agent | not started |
| 3 — S2 loop | not started |
| 4 — S3 physics | not started |
| 5 — S4 report | not started |
| 6 — hardening + handoff | not started |

Current corpus counts (files, clean vs quarantined, usable segments/minutes, subjects) live in the
latest `runs/*/clean_report.md` and `census.md`, regenerated every run. A 2026-05 batch is
quarantined for a broken time base (`DOMAIN_NOTES` §2.6).

See `PLAN.md` for each phase's gate. Sacrifice order if time runs short: Phase 4 first, then
Phase 5. Never Phases 1–3 — they are the handoff-critical spine.
