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

A `Label` column appearing under `data/raw/` is a **contamination event**: quarantine and flag it.

---

## Running it

### S1 — census (measure the corpus, judge nothing)

```powershell
python -m stages.s1_clean.run --out runs\s1_census
```

Produces `census.md` (human-readable: files, families, and any unregistered column name — the
"schema changed" alarm) and `manifest.jsonl` (one row per file: session, family, measured rate,
jitter, gaps, label codes). Columns are resolved by name, never by header shape.

### S1 — clean (resample onto the canonical grid)

```powershell
python -m stages.s1_clean.clean --out runs\s1_clean
```

Produces `data/clean/**.parquet` (gyro normalized to deg/s) with a per-file
`channel_trust.json` sidecar, plus per-run `segments.jsonl`, `observations.jsonl`,
`quarantine.jsonl`, and `clean_report.md`.

### S2 — train and evaluate (deterministic, no agent)

```powershell
python -m stages.s2_ml.train --out runs\s2_ml               # leave-one-rev-out CV
python -m stages.s2_ml.train --out runs\s2_ml --taxonomy    # + row-level error taxonomy
python -m stages.s2_ml.verify_transform                     # standing raw->rev2 bridge guard
python -m stages.s2_ml.profile_incumbent                    # legacy `loco` under the same taxonomy
```

Trains on the labeled `rev*` trials, grouped and split by `rev` (one subject, one day), with the
lockbox revs sealed from the start. `--taxonomy` scores at row level via dense inference — see
`DOMAIN_NOTES` §7 for why a windowed classifier cannot be scored by the taxonomy directly.

### Agents

```powershell
python orchestrator.py --phase 2      # S1 exception triage over the latest clean run
python orchestrator.py --phase 3      # one S2 champion/challenger cycle
```

The S1 exception agent (`agents/s1_exception.py`) reads the clean stage's exception queue
(`quarantine.jsonl` + `observations.jsonl`) and answers two orthogonal questions per item —
`explained` (yes / no / contradicts) and `action` (none / human), grounded in `DOMAIN_NOTES`.
Deterministic code collapses that pair into `known_expected` / `novel` / `needs_human` and writes
`exceptions_review.jsonl`. It is read-only: the agent judges, code does the work.

Phase 3 runs one S2 cycle: `agents/s2_experimenter.py` proposes a single declarative
`ExperimentSpec` (features to drop, window, stride, whitelisted hyperparameters — never code),
`agents/s2_critic.py` reviews it **before** any training with the experiment ledger in view, and
`experiment.decide()` gates promotion on the measured metric. **Neither agent can promote
anything.** Every measured outcome appends to `experiments.jsonl`; proposals killed before training
land in `proposals.jsonl`, so the next cycle can see that an idea was already raised and refused.

Agent system prompts are written to `runs/<run>/system_prompt.txt` and passed to the SDK by path,
not on the command line — `DOMAIN_NOTES` outgrew the Windows `CreateProcess` limit in a single day
(`DOMAIN_NOTES` §8). The file doubles as an audit record of exactly what each agent was told.

Every run gets `runs/YYYY-MM-DD_runN/` containing `run_meta.json` (commit SHA — `runs/` is
gitignored, so each run records the commit that produced it), `run_log.jsonl` (every tool call, via
a PostToolUse hook) and `costs.json` (per-agent spend from the SDK's ResultMessage).

---

## What S1 actually does, and why

**Resolves columns by name, never by position.** The `NN_` prefix is a per-file position, not an
identifier. `loco` sits at index 47 — but in some headers, index 47 is `Step`. Both are
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
observation. Dropping them collapses every header shape into one canonical schema and removes
firmware-version signal from the feature set. The exception mechanism (`KEEP_EXCEPTIONS` in `stages/s1_clean/config.py`) is
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
| 1 — S1 deterministic core | complete — schema/rate/gaps, gyro unit+axis trust, yaw-drift trust, degenerate-time-base rejection, quarantine ledger; gate passes (every raw file accounted) |
| 2 — S1 exception agent | complete — `agents/s1_exception.py` triages the exception queue into known_expected / novel / needs_human with grounded rationale; verified on the real corpus and signed off (gate closed) |
| 3 — S2 loop | running end to end — dataset/transform/features/train/locoeval/taxonomy built; champion `drop_offset_only` at LORO macro-F1 **0.8862**, promoted from an agent proposal (2 promotions, 5 rejections ledgered). Gate open: replay-reconstructibility unverified, the critic has never rejected a proposal, lockbox still sealed |
| 4 — S3 physics | not started |
| 5 — S4 report | not started |
| 6 — hardening + handoff | not started |

Current corpus counts (files, clean vs quarantined, usable segments/minutes, subjects) live in the
latest `runs/*/clean_report.md` and `census.md`, regenerated every run. A 2026-05 batch is
quarantined for a broken time base (`DOMAIN_NOTES` §2.6).

See `PLAN.md` for each phase's gate. Sacrifice order if time runs short: Phase 4 first, then
Phase 5. Never Phases 1–3 — they are the handoff-critical spine.
