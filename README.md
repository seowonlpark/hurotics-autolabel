# hurotics-autolabel

**English** | [한국어](README.ko.md)

A pipeline that reads raw wearable-sensor logs and, for every moment of a recording, decides whether
the wearer is **standing** or **walking** - and, crucially, how **confident** it is in that call. It
is built so that ordinary code does all the number-crunching and AI is used only for judgment calls,
with every decision written down and reproducible.

This page gets you from a fresh copy of the project to a finished report. For how the pipeline works
inside, see [`PIPELINE.md`](PIPELINE.md); for why it is built the way it is, see
[`BUILDLOG.md`](BUILDLOG.md).

---

## Background (for non-technical readers)

Two different kinds of "AI" appear in this project, and they do different jobs.

**Machine learning (ML)** is a program that learns patterns from labeled examples. Here it is shown
many windows of sensor data that a person has already marked as standing or walking, and it learns to
label new windows on its own. ML is very good at generalizing from lots of examples, but it is a black
box: it cannot easily explain itself, and it can be confidently wrong.

**Agentic AI** is a language model (the kind behind chat assistants) that can read files and reason
about them. Here it is used only for **judgment** that a fixed rule cannot make well - things like
"this file looks odd, is it a known problem or something new?" or "does the physics agree with the
human label here?". It is good at reasoning over messy context, but it is never allowed to touch the
data or compute a result.

**Why both, and why this shape.** The guiding rule is *code does the work; agents judge the work*.
Deterministic code and ML handle everything measurable - cleaning the data, training the classifier,
scoring it. The AI agents only weigh in at a few defined points, and they can change nothing on their
own. This matters because it keeps the whole system honest: every number can be reproduced by
re-running the code, every agent decision is logged with its reasoning, and whenever the machine
genuinely cannot decide, it says so and hands the case to a human instead of guessing. Splitting the
work into stages (clean -> learn -> physics -> combine) lets each part be improved and checked on its
own.

---

## Setup

You need **Python 3.10 or newer**. All commands below are run in a terminal from the project folder.

**1. Open the project.** Open the `hurotics-autolabel` folder in VS Code (File -> Open Folder), then
open a terminal inside it (Terminal -> New Terminal).

**2. Create and activate a virtual environment** (an isolated place for this project's Python
packages):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activate script, run this once, then try again:
`Set-ExecutionPolicy -Scope Process RemoteSigned`.

(On macOS or Linux the activate line is `source .venv/bin/activate` instead.)

You must activate the environment in every new terminal. Forgetting to is the most common cause of
"module not found" errors.

**3. Install the packages:**

```powershell
pip install -r requirements.txt
```

**4. Set up your API key (only needed for the AI-agent steps).** The agents call the Claude API, which
is pay-as-you-go and requires a key.

- Get a key from the Claude Console: **https://platform.claude.com** (create an account, then
  API keys -> Create key).
- In the Console, **set a monthly spend cap** so costs can never run away. This is a safety rail that
  cannot be set from inside the project.
- Copy the example config and paste your key into it:

```powershell
Copy-Item .env.example .env
```

  Then open `.env` and put your key after `ANTHROPIC_API_KEY=`.

**Cost note.** The whole deterministic pipeline (cleaning, training, scoring, the final report) runs
**for free** - no key, no charges. Only the four AI-agent steps spend credit, and a full run of them is
roughly a few dollars. You can do everything except the agent steps without a key.

---

## Get your data in

Two kinds of data go in two different places. Put files in with your file explorer or VS Code; nothing
here is edited by hand.

**Unlabeled data you want the pipeline to label** goes in `data/raw/`, one folder per recording
session, named by date:

```
data/raw/20260114/some_recording.csv
data/raw/20260114/another_recording.csv
```

**Labeled data you want to train the classifier on** goes in `data/labeled/`, organized by **rev** -
one rev is one subject recorded on one day - with the filename pattern
`annotated_loco_<rev>_trial_<n>.csv`:

```
data/labeled/rev2/annotated_loco_rev2_trial_1.csv
data/labeled/rev3/annotated_loco_rev3_trial_1.csv
```

Keep labeled and unlabeled data separate - never put labeled files under `data/raw/`.

**The lockbox (held-out test set).** To trust a score, some labeled revs must be set aside *before
training* and never looked at until the very end - this is the lockbox. It is how you find out whether
the model really generalizes to a new person, rather than just memorizing the people it trained on.

You choose which revs are the lockbox by editing one line in
[`stages/s2_ml/dataset.py`](stages/s2_ml/dataset.py):

```python
DEFAULT_LOCKBOX_REVS = ("rev8", "rev13")   # the revs held out until the final test
```

Put one or two whole revs (subjects) here that the model will never train on. The pipeline enforces
this in code. Note that a lockbox is **single-use**: once you open it for the final number, it is
spent, and testing again honestly needs a fresh held-out rev.

---

## Run it (one command)

```powershell
python run_pipeline.py
```

That runs the full deterministic pipeline - clean the data, train and score the classifier, run the
physics, and produce the final fused report - **for free**, in order, stopping with a clear message if
anything is missing.

To also run the four AI-agent steps (this spends API credit):

```powershell
python run_pipeline.py --with-agents
```

Useful extras: `--list` prints the steps without running them, and `--dry-run` shows each command it
would run.

---

## What you get

When the run finishes, the results live under `runs/s4_fusion/`:

- **`fusion_report.md`** - the human-readable summary: how the combined call performed, and how much
  of the data it can label at each confidence level.
- **`fused_windows.csv`** - one row per window, with the call (standing / walking) and its confidence
  tier: **high**, **medium**, or **low**.

The point of the project is that last column. A **high** or **medium** call is one you can act on; a
**low** call is the system honestly saying "I am not sure here" - an abstention, not a guess. In a
real device, you would act on the confident calls and hold on the low-confidence ones.

To write the calls back onto your original files (one CSV per recording, with the label and confidence
added as columns), run `python -m stages.s4_fusion.export` - the results land in `results/`.

---

## Running a single stage

The one command above is usually all you need. For finer control, each stage can be run on its own:

| command | what it does |
|---|---|
| `python -m stages.s1_clean.clean --out runs/s1_clean` | clean + resample the raw data |
| `python -m stages.s2_ml.train --out runs/s2_ml --taxonomy` | train + score the classifier |
| `python -m stages.s3_physics.run` | compute the physics anchors + plots |
| `python -m stages.s4_fusion.run` | combine into the fused call + confidence |
| `python orchestrator.py s1_exception` | AI: triage unusual files (spends credit) |
| `python orchestrator.py s3_physics` | AI: read the plots, propose rules (spends credit) |
| `python orchestrator.py s4_fusion` | AI: review the disagreements (spends credit) |

See [`PIPELINE.md`](PIPELINE.md) section 8 for the complete list.

---

## Where to learn more

- [`PIPELINE.md`](PIPELINE.md) - how each part works, and the exact commands and outputs.
- [`BUILDLOG.md`](BUILDLOG.md) - why the pipeline is built the way it is, decision by decision.
- Questions: **seowonlpark@gmail.com**.

Before feeding in a new batch of CSVs, you can check they are well-formed with the companion validator:
**https://github.com/seowonlpark/hurotics-imu-csv-validation**.

---

## Troubleshooting

- **"ModuleNotFoundError" / packages missing** - the virtual environment is not active. Run
  `.venv\Scripts\Activate.ps1` (you must do this in every new terminal), then re-run.
- **An agent step complains about the API key** - your `.env` is missing or has no key. Copy
  `.env.example` to `.env` and paste your key in. (The non-agent steps do not need a key.)
- **"no raw data" at startup** - `data/raw/` is empty. Add your recordings under
  `data/raw/<date>/` first.
- **PowerShell won't run the activate script** - run
  `Set-ExecutionPolicy -Scope Process RemoteSigned` once, then activate again.
