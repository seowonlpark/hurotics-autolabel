# understand.md — why every part of this pipeline is the way it is

This file answers one question for every file and every function in the repo: **why does
this exist, and why in this form?** It is not an API reference. Where a choice was made
against an alternative, the alternative and the reason it lost are recorded here.

Reading order: `DOMAIN_NOTES.md` (what the corpus taught us) → `caveats.md` (what is
shaky) → `OPERATING_POINTS.md` (the threshold) → *how to run it* → this file (why any of it
is shaped this way).

**There is no `README.md` right now.** It was deleted on 2026-08-04 and `needtowrite.md`
holds both the spec for its replacement and the prose that existed nowhere else. Until it
returns, "how to run it" is `run_pipeline.py`'s own `--dry-run` and the per-stage commands
in `OPERATING_POINTS.md` § Reproducing. `stages/breakdown.py::PROSE_CLAIMS` still lists
`README.md`, so every run raises a `gap` flag naming this — deliberately, so the hole
cannot go quiet.

A `§` reference points into `DOMAIN_NOTES.md` unless another file is named.

---

## 0. What is being built, and what it is judged on

**Label stand/walk on any recording at ≥95% accuracy over the windows it claims, and
abstain rather than guess on the rest.**

Two halves of that sentence drive nearly every design decision below.

- *"any recording"* is why the pipeline resolves columns by name, detects sampling rate
  instead of assuming it, refuses unmapped hardware revisions, and carries a raw→`lpf_view`
  bridge at all. A model that only runs on files someone already pre-processed is not a
  deliverable.
- *"abstain rather than guess"* is why abstention is a first-class output with a reason
  and an alternative, why coverage and accuracy are never quoted apart, and why five
  separate modules refuse a file rather than return a number they cannot defend.

The target is **met on development subjects and missed on the one genuinely unseen
subject** (`rev8`, 0.9308 on committed rows). That gap is not hidden — `caveats.md` §3.2
is the most important paragraph in the repo, and every headline number in this document
is bounded by it.

---

## 1. The five principles, and the evidence for each

Everything downstream is a consequence of these. They are not style preferences; each was
paid for.

### 1.1 Code does the work; agents judge the work

Agents never touch a data value, never recompute a statistic, and never change a label.
They read records that deterministic code produced and return a verdict as text, which
deterministic code then writes down.

*Why:* §11.3 — *"Every rule asserted was wrong; every rule the tests checked survived or
died honestly."* An agent reading graphs found real structure (slow walking, split stance)
that no descriptor encoded. The same agent, asked to label windows, invented a class
("walking with no rhythm", §11.2) to explain its own artifact. So agents are used for rule
*discovery* and *triage*, never for per-window labelling.

### 1.2 The filesystem is the only interface between stages

No agent-to-agent messaging, no queues, no shared memory. S2 writes `champion.joblib` and
`model_meta.json`; `roweval.py` and `label_audit.py` each read them back from disk.

*Why:* every stage boundary becomes inspectable and re-runnable without the stages either
side. `label.load_champion` is the rule in miniature — it returns the model **and** its meta
together, because loading one without the other is how train/serve skew starts, and the meta
is a file precisely so the pairing can be checked rather than assumed.

### 1.3 No silent mutation, ever

A file that cannot be trusted is recorded in a ledger and left in place. A channel whose
axis is mislabelled is *recorded*, not reordered. A file with a broken clock is refused,
not repaired with a synthetic one.

*Why:* §2.6 records the decision explicitly (Lu, 2026-07-20): a uniform-rate clock would
*fabricate* unmeasured time. The pipeline's whole claim is that its numbers come from
measurements; one invented column destroys that claim everywhere, retroactively, and
nothing downstream would notice.

### 1.4 "Recorded" is not "handled" — a note is not a consumer

The single most expensive lesson in the repo, learned twice.

§4.1b said the gyro axis "must be resolved from `channel_trust.json`". §4.2 said any
yaw-derived feature "must consult the per-file drift flag". Both read as descriptions of
pipeline behaviour. **Neither had a single consumer.** S1 measured the answer and nothing
opened the file.

The consequence was not hypothetical. `transform.py`'s sagittal-axis lookup mapped the
majority variant (62 of 91 raw files) to the wrong plane, and *stayed wrong* because
nothing ever ran that path against a real file — `caveats.md` §5. An unexercised path
cannot be wrong out loud.

The rules that fell out of this, and which the code now enforces:

- A guard takes its evidence as a **required argument**, with an explicit named opt-out
  (`transform.TRUST_UNCHECKED`), so skipping it is a decision at the call site.
- The S1 exception agent is handed a machine-derived `handled {value, why}` field computed
  from the *actual consumers*, and is told in its system prompt that where the notes and
  `handled` disagree, `handled` wins.
- Claims about behaviour get tagged `[decided]` vs `[measured]`, and unenforced policy is
  labelled as unenforced in place.

### 1.5 Every constant is either measured, or declared a judgement

There is no third category. A number is one of:

| kind | example | rule |
|---|---|---|
| a measured floor | `SWAP_DELTA_DEG = 1.0` (5× standing noise, 0.76–0.88° on three files) | must not be tuned; tuning converts the one label-free opinion into a second fitted model |
| a judgement on a curve | `DEFAULT_PROBA_FLOOR = 0.70`, `balanced = 0.85` | the whole curve is re-measured every run and the shipped point marked on it |
| derived from the corpus at runtime | the amplitude band, `feat_p01`/`feat_p99` | never hard-coded; moves when the data moves |

And the corollary, which `caveats.md` §1.1 records as a scar: **a constant justified
against one measurement silently expires when the thing underneath it changes.** The
asymmetric physics veto was right 0.788 of the time when adopted and 0.566 after S2
absorbed the interleg features. Whole-corpus accuracy still looked fine (+0.2 points),
which is exactly what made it survive.

---

## 2. Repository-level choices

| choice | why |
|---|---|
| three stages, S1→S3, one directory each | each stage is runnable alone and its artifacts are readable without the next stage. It was four until 2026-08-04; S4 is deleted (see the banner) |
| `data/`, `runs/`, `.env` gitignored | nothing from HUROTICS leaves the machine via git |
| `runs/YYYY-MM-DD_runN/` never overwritten, each with `run_meta.json` carrying the git SHA | `runs/` is gitignored, so the run must record the commit that produced it or the artifact is unattributable |
| clean output is **parquet** | storage cost is a file-format problem, not a column-count problem — ~5–10× smaller, dtypes preserved. This is what lets the canonical file keep the honest superset instead of pruning to save space |
| `DOMAIN_NOTES.md` injected into every agent prompt **as a file**, not argv | it crossed the ~32 KB Windows `CreateProcess` limit and every agent died with a *misleading* `CLINotFoundError` naming a healthy binary (§8). A path is O(1) on the command line, so institutional memory can grow without a ceiling |
| `stages/console.py` called first in every `main()`, UTF-8 in written `.md` | the console is cp949 here, so any prose character raises `UnicodeEncodeError` on the write. Policing every string down to ASCII is not enforceable and would degrade the reports, which are read as Markdown where the typography is correct and wanted — so the *encoder* is relaxed instead, once, in one place. Garbled output beats a lost run, and beats a command that cannot print its own `--help` (§12) |

---

## 3. The two drivers

### 3.0 `run_pipeline.py` — `data/raw` → the breakdown report, gated, no manual intervention

Each stage runs as **its own subprocess** and must produce its **gate artifact** before the
next starts. That is the whole design: a stage that exits 0 without writing anything stops
the run, instead of leaving the next stage to read a file from a previous run and quietly
report last week's numbers.

| item | why |
|---|---|
| `Step.gate` | trust an exit code only where there is no artifact to check. Exit 0 is a claim; a written file is evidence |
| order is load-bearing, and said so | *(was: S4's join of S2's OOF to S3's anchors)*. With S4 deleted the one ordering constraint left is `s2_roweval` after `s2_train` — it re-fits the champion per held-out rev and runs the real `label.py`, and it is the step that produces the accuracy claim, so a run that skips it has trained a model and measured nothing a caller can act on |
| three modes: default / `--with-agents` / `--verify` | the deterministic spine is free; the agent steps cost credit and the verifications are slow. Opt in to each separately, so the cheap path is the default path |
| `preflight()` | fails **before spending any time or any credit** — no raw data, or agent steps requested with no API key |
| `--from KEY` | resume after a fixed failure without redoing the spine, and the failure message names the exact flag to use |
| `PY = sys.executable` | subprocesses use the same interpreter, so a venv that is active for the parent is active for every child. §README notes forgetting the venv is the single most common source of `ModuleNotFoundError` here |
| stale check at the end, **reported not fatal** | a full run leaves every stage consistent by construction; the check catches the case that matters — a stage re-run *alone* afterwards. The artifacts are real, they just no longer agree, and *the runner is the wrong place to decide that is unacceptable* |

### 3.1 `freshness.py` — does this output still describe its inputs?

The failure it exists for is silent. `runs/` is gitignored and every stage overwrites
`runs/<stage>/` in place, so **re-running one stage alone — the normal way to iterate —
leaves every downstream artifact describing a model that no longer exists.** `breakdown.md`
can quote a headline accuracy for a champion that has already been replaced, and nothing
notices: the files are all present, all parseable, and **wrong together**.

`breakdown.py` is the consumer that makes this bite: it copies every number from the
artifact that owns it, so a stale artifact becomes a stale claim on the one page a reader
trusts. `check_all` runs there first and raises the staleness as a flag *above* the numbers
it invalidates, rather than beside them.

| function | why |
|---|---|
| `artifact_id()` | **content hash, not mtime.** A fresh checkout, a file copy or a touch all move mtime without changing what the stage read, and any of those would cry wolf. Hashing a ~20 MB csv costs milliseconds against stage runtimes in minutes. Returns `None` for a missing file — a legitimate answer, not an error |
| `stamp_inputs()` | records what a stage consumed **next to what it produced** |
| `check_inputs()` | **an absent stamp is reported, not passed.** An unstamped directory is exactly the state that hides this bug, so "cannot tell" and "stale" are reported the same way — and the caller decides how loudly to fail |

Two callers stamp today: `label_all.py` (the champion behind every labelled CSV) and
`train.py` (`champion_spec.json`, because a promotion rewrites it and until a refit follows,
`locoeval`, `model_meta` and `champion.joblib` all describe the model that just lost).
`breakdown.py` checks `labeled_raw/` unconditionally and `runs/*` only where a stamp
already exists — a stage that consumes nothing has nothing to go stale against, and
demanding a stamp from each would be noise.

### 3.1b `runmeta.py` — one definition of "which commit produced this"

`runs/` is gitignored, so an artifact that cannot name its commit cannot be traced to the
code that wrote it. `git_sha()` is that name, and it lives in exactly one module because
the alternative is observable: `run_pipeline.py` (`run_meta.json`), `experiment.py` (the
ledger and `champion.json`) and `breakdown.py` (the report footer) all stamp it, and until
2026-08-04 `breakdown.py` carried a private copy — **two answers to one question**, which is
the same defect as two accuracy proofs, at a smaller scale. It returns `"unknown"` outside a
checkout rather than raising: provenance is worth recording, never worth failing a run over.

### 3.2 The two agent reviews — steps in the runner, not a second driver

They had their own CLI until 2026-08-03 (`orchestrator.py --stage s1|s4`). That was a
**second entry point for the same job**: a duplicate copy of the dotenv and console setup,
a subprocess hop, and a second answer to "how do I run something in this repo". Nothing but
`run_pipeline.py` ever invoked it. The two review functions moved into the runner and the
file was deleted; `--from s1_exception` / `--from s4_review` runs either one alone, which is
what the separate CLI had been for.

They are the only steps that run **in-process** rather than as a subprocess. The
deterministic steps stay isolated because a stage that rebuilds a 30 MB model is worth a
process boundary; an agent call is a network wait, and isolating it bought nothing.

| function | why it exists / why this form |
|---|---|
| `Step.fn` alongside `Step.cmd` | exactly one of the two. `run_step` catches an exception from `fn` the way it checks a non-zero exit code from `cmd`, so an in-process failure still stops the run with a resume hint instead of unwinding through the runner and losing which step it was |
| `_git_sha()` | `runs/` is gitignored, so a run that does not record its commit cannot be reproduced. Swallows exceptions and returns `"unknown"` — a missing SHA must not kill a run that is otherwise fine |
| `_new_run_dir()` | monotonic `runN` suffix, never overwrites. A pipeline whose reruns clobber the previous evidence cannot support a claim like "this changed nothing", which `caveats.md` §5.1 makes |
| `_s1_exception()` | builds the S1 exception queue deterministically, runs the agent, writes the review. Raises a *directed* error naming the command to run if there is no clean run — the failure belongs to the producer, not the consumer |
| `_agent_step()` | wraps one review as a `Step.fn`: `load_dotenv`, fresh run dir, then `asyncio.run`. Sets `__qualname__` from the coroutine so `--dry-run` names the review rather than a closure |
| SDK imports are **inside** the functions | a deterministic run neither needs nor loads the agent SDK, so the free path stays free |

**Both reviews print the dropped-by-cap count.** A capped review that says nothing about
the cap reads as "the agent reviewed everything". `_s4_review` additionally prints mislabel
candidates individually, because that verdict is the one worth acting on and burying it in
a tally would waste it.

---

## 4. `agents/` — the judgement layer

### 4.1 `agents/base.py` — shared machinery, nothing else talks to the SDK

| item | why |
|---|---|
| `MODEL_CHEAP` / `MODEL_SMART` | route triage to haiku, judgement to sonnet. §8 measured the cheap model, under a grown prompt, returning 22 verdicts for 23 items and silently dropping a field — so the stage whose verdict is load-bearing (`label_suspect`) pays for sonnet |
| `DEFAULT_MAX_TURNS = 25` | a stuck agent must fail fast, not spin up a bill |
| `AgentSpec` / `AgentResult` | `AgentResult` is deliberately small — name, text, cost, turns. The runner is not given a channel through which an agent could return data |
| `_load_domain_notes()` | **raises** if the notes are missing. No agent runs without institutional memory; a fresh subagent context has no other channel for hard-won facts |
| `_build_system_prompt()` | wraps the notes in explicit BEGIN/END markers with "do not re-derive, do not silently contradict" |
| `_make_audit_hook()` | PostToolUse hook, observe-only, never blocks. Honestly documents its own gap: hooks may not fire if the agent hits `max_turns` |
| `extract_json_array()` | greedy first `[` to last `]`. Returns `None` rather than raising — **every caller already needs a conservative path for an unjudged item, and a parse failure is just that case at scale** |
| `_record_cost()` | appends to a per-run ledger with a running total; spend is an artifact of the run, not a line in a terminal that scrolls away |
| `run_agent()` | writes the system prompt to `runs/<run>/system_prompt.txt` and passes it by path (§1.4 above / §8). The file doubles as the audit record of exactly what the agent was told |

### 4.2 `agents/s1_exception.py` — triage the clean stage's exceptions

**Two orthogonal questions, not one taxonomy.** The original design collapsed
`known_expected` / `novel` / `needs_human` into a single choice, and those three collide
on two different axes with no precedence. The agent now answers:

1. `explained` — does DOMAIN NOTES account for this? `yes` / `no` / **`contradicts`**
2. `action` — is a person needed? `none` / `human`

and `collapse()` derives the disposition. The prompt says explicitly: *do not infer
`action` from `explained`.* A broken clock is §2.6-explained **and** still needs a
re-export; an unexplained anomaly can be inert.

| function | why |
|---|---|
| `_SAGITTAL_CANDIDATE_AXES` | derived from `transform.SAGITTAL_DEG_AXIS_BY_VARIANT.values()`, not restated. A new variant mapping cannot make this stale |
| `_handled_quarantine()` | returns `handled=False` — **excluded is not repaired.** Downstream is safe; the file is still lost until a person re-exports it |
| `_handled_axis_anomaly()` | the whole §1.4 lesson in one function: is the conflict on a channel the feature path *actually reads*? `L`/`R` only, and only on an axis some variant treats as sagittal. Note it returns `False` (not handled) even though `check_axis_trust` hard-fails — *fatal is not handled*, the file still cannot be featurized |
| `_handled_drift()` | states plainly that nothing consumes `drift_contaminated`; the flag is **inert, not honoured**, and only costs nothing because no feature reads `Deg_Z` at all |
| `build_queue()` | genuine exceptions only. Routine gyro-abstentions are *counted and summarized*, not triaged item by item — they are designed-normal behaviour, and queuing them would bury the two real anomalies in noise |
| `collapse()` | the taxonomy is the pipeline's, decided in one place, so two runs cannot disposition the same item differently. `contradicts` → `novel` + forced `human`: a contradicted note is a find, and is never acted on |
| `write_review()` | an item with no parseable verdict is written as `needs_human`, never dropped. Raw text is kept when parsing fails |

### 4.3 `agents/s2_experimenter.py` — propose one challenger, as data

Read-only, and it **never writes code or trains anything**. It emits one declarative
`ExperimentSpec` — `name`, a one-sentence `rationale`, `drop_features`, window/stride,
model params — from the champion report plus the ledger, and deterministic code runs it.

| decision | why |
|---|---|
| a fixed spec vocabulary, not free-form | an agent that could write the experiment could write the result. The spec is the entire channel, and `ALLOWED_MODEL_PARAMS` / `WINDOW_S_RANGE` are imported from `experiment.py` rather than restated, so the prompt cannot describe a knob the runner does not have |
| `drop_features: null` ≠ `[]` | the prompt spells this out: `null` keeps the champion's drops, `[]` re-adds every removed feature and makes the proposal two changes at once. A one-change rule that a default silently violates is not a rule |
| **one change per turn** | the ledger is the record of *why* a feature set moved; a two-change proposal that wins tells you nothing about which half won |

### 4.4 `agents/s2_critic.py` — is this worth a cycle?

Reviews the proposal **before it runs** — `approve` / `reject` / `revise` — with the ledger
in hand so a repeat is caught before it costs a fit. It **does not decide promotion**;
`experiment.decide()` does that afterwards on the metric. The division is §1.1 exactly: the
critic spends or saves compute, code spends or withholds the champion title.

`run_pipeline.MAX_PROPOSE_ATTEMPTS = 2` bounds the propose→critique loop. Every draft and
critique stays on disk under `_attempt_name()`, because *the draft the critic sent back is
frequently the more informative artifact* and overwriting it leaves a run log that only
ever shows agents agreeing.

### 4.5 `agents/s3_label_review.py` — adjudicate what the label audit nominated

`label_audit.py` measures and stops at "go look" on purpose — *a nomination is not a
verdict* — which left its flags unread, because a flagged trial meant opening
`inspect_window` twelve times by hand. This is the consumer that turns a flag into a ranked
queue with a named cause from a closed vocabulary.

| decision | why |
|---|---|
| never recomputes, never opens a raw signal | it reads the audit's own numbers and DOMAIN NOTES. A judge that re-derives the evidence is a second measurement competing with the first |
| **publishes no coverage or accuracy** | that is what makes it safe where the deleted S4 review agent was not. S4 judged a corpus-level join no customer CSV went through; this queue is the audit's, pointed at the *annotations* — and the swap rule has no trained parameter and never sees a label, so contradicting one is evidence about the label |
| every bad-data cause forced to `action=human` by `collapse` | `dataset.EXCLUDED_TRIALS` is edited by hand with the evidence beside it. **An agent may nominate a label change, never make one** |

## 5. `stages/s1_clean/` — measure the corpus, judge nothing

### 5.1 `config.py` — everything tunable, nothing buried

The file exists so that no constant lives inside logic. Every entry carries its reason
inline. The load-bearing ones:

| constant | why this value |
|---|---|
| `COLUMN_PREFIX_PATTERN` | the `NN_` prefix is a per-file **position**, not an id. `loco` sits at index 47 in most files; in variant `0fda484e` index 47 is `Step`. `df.iloc[:, 47]` therefore blends a step counter into a locomotion state on the majority of files and **never raises** (§1.3) |
| `FAMILY_MARKERS` | the corpus holds two **shapes** sharing only `Time`. "The stable prefix across everything" is a meaningless question, so family is decided by a marker column and contracts are per family. Both names describe a representation, not a product — `lpf_view` was `rev2_view` until 2026-08-04, which named one of the eight annotated revisions and did not cover the files this pipeline itself writes (DOMAIN_NOTES §6.1) |
| `LABEL_UNKNOWN_MACHINE=255` / `LABEL_UNKNOWN_HUMAN=-1` | two unknowns, **opposite in kind**. `255` = nothing was measured. `-1` = data is fine, a trained human looked and could not call it. Merging them destroys the single most valuable signal in the corpus (§5.1/§5.2) |
| `CANONICAL_HZ = 100.0` | incoming golden data is 100 Hz, most files already sit there, 500 Hz decimates down cleanly. **Nothing is upsampled** — no bandwidth is invented |
| `RATE_TOLERANCE = 0.05` | wide enough to fold 99.3789 / 99.688 / 99.961 Hz into the 100 Hz family. Those are *timestamp quantization* (10 + 2⁻ᵏ ms), not different devices (§2.2). Grid-correct them; do not discard them |
| `MIN_SEGMENT_S = 1.0` | **a duration, not a sample count.** It was `MIN_SEGMENT_SAMPLES = 100`, documented as "1 s at canonical rate" — true only in the ~100 Hz era. At 500 Hz it meant 0.2 s, and a 238-row segment reached `data/clean` and the report's usable minutes while being too short to yield a single 2 s window. *A threshold in samples is a threshold whose meaning changes with the rate era* |
| `DECIMATE_FILTER = "fir"` | taking every 5th sample of 500 Hz folds everything above 50 Hz into the gait band at **full amplitude** — measured, §2.5 |
| `TRUST_R_FLOOR = 0.9` | below this, `d(Deg_A)/dt` carries no signal and the fit is noise. Applied **per axis, not per side**: `Deg_Z` drifts rather than oscillates, so its derivative is noise even mid-walk, and scoring the side as a whole laundered that noise into 25 confident-looking anomalies where there are 2 |
| `DOCUMENTED_GYRO_PERMUTATION` | X→X, Y→Z, Z→Y, identical on every variant that answers. **Explicitly annotated as NOT sagittality** — that conflation is what made the old `sagittal_gyro_axis` field wrong on the majority variant |
| `KEEP_MEASURED` (30 columns) | the line is **measured vs computed**, not useful vs useless. Dropping computed columns collapses every schema variant into one shape *and* removes the firmware-era confound from the feature set |
| `KEEP_EXCEPTIONS = {}` | empty, and that is the finding. `Hip_Deg_L/R` failed every premise of its own exception: 0.991 correlated with `Deg_Y`, residual = the firmware's zeroing convention (the exact signal the rule strips), zero-variance on 12/180 (file, side) pairs — twice frozen at a *nonzero* constant no `!= 0` guard catches — and the dataset it bridged to is not in the repo. **An empty exception dict is a stronger invariant than a populated one** |

### 5.2 `census.py` — the only place that decides what a column *is*

| function | why |
|---|---|
| `strip_prefix()` | strips the positional prefix and nothing else. Whitespace normalized because some trials carry stray spaces |
| `read_header()` | first row only; trailing empty fields tolerated and dropped |
| `family_of()` | marker-column lookup. Duck-typing on "does this file have the columns I want" would misroute a half-written file instead of rejecting it |
| `fingerprint()` | sha1 of the **stripped names**, not positions or widths. Two files with the same columns in the same order get the same id regardless of prefix numbering |
| `resolve()` | maps a header to roles by name. **Absence is recorded as a fact, not an error** — presence is per-file, not per-corpus |
| `build_registry()` | one pass over headers only, cheap enough to run on every file |
| `stable_prefix()` | longest identical leading run **within a family**. Measured, not assumed — currently 45 columns, and computing it across families would be meaningless |

`Variant` no longer carries `raw_columns` — see §11.

### 5.3 `manifest.py` — numbers only, no verdicts

| function | why |
|---|---|
| `session_of()` | session identity comes from the **directory**, the only metadata trusted here |
| `measure_time()` | reports **two** rate estimates (`hz_from_median_dt`, `hz_from_span`) plus whether they agree. They legitimately disagree on a file with jitter and gaps, and *hiding that behind one number is how a rate confound goes unnoticed*. Guards `median_dt > 0` and returns `None` rather than dividing by zero |
| `measure_labels()` | codes, counts, segment structure, and separate flags for `255` and `-1`. No judgement |
| `profile_file()` | **never raises on bad data** — records `read_error` and returns the row. A census that dies on file 40 of 91 measures nothing |

### 5.4 `resample.py` — the two rules this module exists to enforce

1. Never resample across a gap.
2. Never downsample without anti-aliasing.

| function | why |
|---|---|
| `Segment` | carries `usable` + `reason` rather than being filtered away. An unusable segment is dropped from the *output* and always survives in the *table* |
| `segment_at_gaps()` | `dt > GAP_FACTOR × median(dt)`. Gaps land anywhere; a file is a bag of continuous runs |
| `measure_hz()` | returns `nan` for a degenerate time base instead of dividing by zero — the caller drops that segment |
| `rate_family()` | snaps to 100/200/500 Hz, or `None`. `None` is a refusal, not a default |
| `_interp_to_grid()` | linear for continuous channels, **nearest for categorical**. Averaging a class code produces a label that was never annotated |
| `resample_segment()` | for `factor > 1`: build a uniform grid at the source rate first (`decimate` assumes uniform spacing), *then* FIR-decimate. Records its own `method` string so the report can tally how each segment was handled. For `factor == 1`: interpolate onto the exact canonical grid, correcting timestamp quantization |
| `resample_file()` | stacks usable segments, returns the full segment table alongside |

**Every grid is built from real timestamps** (`np.arange(t[0], t[-1], …)`) and every value
interpolated against the measured time vector — never reconstructed from a nominal rate.
Audited: a synthetic 494 Hz segment places a true-t=160000 ms event at **160000.0 ms**,
against **+1943 ms** under an `int(t*fs)@500` reconstruction (§7).

### 5.5 `channel_trust.py` — detect, do not assert

Gyro is reliable (it *is* `d(angle)/dt`, r = 0.999) but its **unit is inconsistent within
one file** (B is rad/s, L/R is deg/s — a silent 57.3×) and the axis **labels are permuted**
relative to the Deg labels.

The module regresses `d(Deg_A)/dt` against every gyro axis, for every Deg axis A. The
strongest correlate is A's counterpart (recovering the permutation); the slope reveals the
unit (~1 → deg/s, ~1/57.3 → rad/s).

**It does not know what "sagittal" means and must not pretend to.** It measures a
correspondence *internal* to the file. Which plane is sagittal is a hardware-revision fact
with no in-file signature, resolved by variant lookup in `s2_ml/transform.py`.

| function | why |
|---|---|
| `_classify_unit()` | nearest unit in **log space** — 1.0 and 1/57.3 are ~1.76 decades apart, so linear distance would be dominated by the larger value |
| `_pooled()` | derivative computed **within each gap-free segment**, never across a gap |
| `detect_side()` | r-floor **per Deg axis**: an axis with no signal abstains rather than dissenting. `is_bijection` is only claimable when all three axes answered — two axes pointing at one gyro axis is a degenerate detection, not an exotic device. Only axes that *answered* may contradict the documented map |
| `detect_drift()` | duration-weighted `|corr(Deg, Time)|` over segments ≥ 5 s. **Flags, never drops** — drift is file-specific, not universal, and this is a feature-time exclusion signal, not a quarantine trigger |
| `detect_and_normalize()` | normalization is **unit-only** (a per-side scalar). Axes are recorded, not reordered; sign is left intact and carried in `r` so polarity is never silently flipped |

### 5.6 `clean.py` — the stage that produces `data/clean/`

| function | why |
|---|---|
| `quarantine_record()` | the raw file is **never moved or copied**. Raw is source-of-truth; the quarantine is a *ledger*. `needs_human` is always set — a whole-file reject is exactly the "the pipeline cannot proceed" case |
| `select_columns()` | returns `(kept, missing)`. Missing is a fact about the variant, reported, and then treated as fatal for that file |
| `clean_one()` | order matters: resolve names → select columns → **guard `median(dt) > 0`** → resample → normalize gyro. The clock guard sits before the resampler because `np.interp` on a non-monotonic time base corrupts silently |
| `main()` | asserts the partition: `written + quarantined == len(paths)`. **Assert it, don't hope.** A file that is neither cleaned nor quarantined is a file nobody knows about |

`clean_report.md` deliberately reports rollups (normalized sides, abstentions, anomalies,
drift-flagged channels) rather than restating them in prose anywhere. §0 makes this a rule:
live tallies belong in run artifacts, not in the notes.

### 5.7 `run.py` — schema census

Writes `variants.json`, `manifest.jsonl`, `census.md`. The one part worth calling out is
the **"names that change position within a family"** section: it computes, per family, every
name occupying more than one index, and prints *"Positional indexing is unsafe here.
Resolve by name."* The §1.3 hazard is regenerated from the data every run rather than
remembered.

### 5.8 `validate.py` — the label-free gate in front of every consumer

Answers "was anything measured properly", which is a property of the file, so it lives in
S1 and not in S2. It needs no model, no label, and no training set.

**Every check is a way a file can be broken without being empty.** An empty file fails
loudly on its own; a file with a flat channel or one that never rests produces *confident
numbers built on nothing*.

| function | why |
|---|---|
| `check_columns()` | resolves by name (§1.3) |
| `check_time()` | non-finite, non-positive median `dt`, and backward steps are reported separately — three different diseases with three different remedies |
| `health()` | `errors` mean nothing is scorable; `warnings` mean the numbers are weaker than they look **and the reason travels with them**. `rest_reference` is imported *lazily* so the gate does not depend at module scope on the stage it guards |
| `health_of_csv()` | reads, grids, judges — the CLI form |

---

## 6. `stages/s2_ml/` — the model, and the deliverable

### 6.1 `dataset.py` — load labeled trials, split by rev

| item | why |
|---|---|
| reuses S1's `resample_file` | rev* trials log at ~494 Hz with jitter. Two disciplines are not optional here either: canonical grid, and never window across a gap. Continuous channels FIR-decimate; `Label` is nearest-sampled by role |
| `DEFAULT_LOCKBOX_REVS = ("rev8",)` | **rev13 was sealed alongside rev8 and deliberately released.** The first lockbox run showed rev13's confidence signal was flat from threshold 0.50 to 0.95, and *a failure that cannot be looked at cannot be fixed*. rev8 stays sealed and is now spent |
| `EXCLUDED_TRIALS = {("rev13", 4)}` | 12,691 rows all annotated `stand`, containing two runs under one label — 7.0 s at 2.1 deg/s, then **119.9 s at 45.3 deg/s** with 71° of interleg swing, against that subject's own labelled walking at 35–50 deg/s. **The evidence is internal to the file** and would stand with no model at all. Excluding data because a model dislikes it is how a corpus gets quietly fitted to its model |
| `load_trial()` restores integer `Label` | nearest-sampling returns float; a float class code silently breaks `isin` comparisons downstream |
| `census()` / `main()` accounting line | prints raw rows → dropped rows with the percentage and the reason (§3.2 startup-burst fragments). Rows are **counted, not silently discarded** |

### 6.2 `rest.py` — the per-recording calibration primitive

§7 is blunt: *per-file calibration, never corpus-wide*. A fixed global threshold scores
`walk_rec = 0.000` on rev8 — it calls every walking window standing — while per-file
calibration scores 1.000 on the same file. **More data yields a better global constant, and
global is the disease.** §10.2 is what makes the cure possible: recordings begin at rest,
so every file carries a standing reference measured on the same person, sensor and mounting
minutes earlier.

| function | why |
|---|---|
| `swap_count()` | hysteresis state machine: commits past `+delta`, then past `−delta`. Commits alternate by construction, so swaps = commits − 1 |
| `interleg()` | `L_ang − R_ang`. A difference between two legs on one body, so mounting offset and amplitude scale largely cancel |
| `is_rest()` | **parameter-free on purpose** — it is the swap rule's own STANDING verdict, locally centred. Testing rest with a *different* criterion than the rule uses would let a span count as rest for calibration and as motion for scoring. `min_n` is required, not defaulted: the verdict is length-dependent and a short span passes far too easily |
| `rest_span_frame()` | searched **within a segment only**; quarter-span hops so rest straddling a block boundary is still found |

**The module holds the primitive; the rule lives in S3.** That split is not cosmetic: S3
imports S2, so defining the primitive in S3 would close an import cycle the moment anything
in S2 needs a rest zero.

The rest *zero* is deliberately **not** derived here — `features.rest_reference` is the
single implementation, because it must return the per-side postures and the interleg offset
from the **same chosen span**. Two searchers could centre the angle features on one rest and
the interleg features on another. A second implementation did once live here, had drifted
into a different preference order, and was removed (`caveats.md` §5).

### 6.3 `features.py` — the model-agnostic boundary

Feature selection lives here, not in the clean layer: the clean layer keeps the honest
superset, this module decides what a model sees.

**Why the interleg block exists.** Per-trial `L_angvel` standard deviation spans 10.6–52.8
deg/s for walking and 1.3–44.0 for standing. Those ranges overlap almost entirely, so **no
absolute amplitude threshold separates the classes across subjects** — which is how an
amplitude-led feature set fails, confidently calling small-amplitude gait "standing". §10
says the same thing from the physics side.

| function | why |
|---|---|
| `amplitude_band()` | the interval where the two **annotated** classes overlap. Both edges come from the annotation and one label-free descriptor; **no model output enters either**, which is what lets `label_audit` use it without forfeiting its model-free property. `BAND_TAIL_PCT = 1.0`, not 5: at 5% the band narrows and 11 of 41 trials contain no band window at all, leaving the per-trial statistic undefined for a quarter of the corpus |
| `WindowSpec` | seconds in, samples out, in one place. `DEFAULT_STRIDE_S = 2.0` for training — **non-overlapping**, because overlapping windows manufacture near-duplicate rows that inflate the sample count and flatter any metric computed on them |
| `swap_counts()` | vectorized `rest.swap_count`. The subtlety a plain prefix sum gets wrong is "both endpoints inside the window": the reference enters every window *uncommitted*, so the change carried by the window's first committed sample must be subtracted. Held to exact agreement by `verify_features.py` |
| `_spectral()` | resolution is 1/window_s, so at 2 s the low edge of the band is unresolvable. **The number is still computed** — §9's tradeoff is made visible rather than hidden |
| `_hf_ratio()` | the transform low-passes at 1 Hz, so most of this band is gone before windowing. Kept only because dropping the whole spectral block measurably lowers worst-subject accuracy |
| `_cycle_s()` | stride period from autocorrelation — **sub-bin by construction**, unlike `_spectral`'s 0.5 Hz grid at 2 s |
| `_min_over_parts()` | the smallest interleg peak-to-peak among k sub-windows, **not** the whole-window ptp. A single weight shift makes one large excursion and reads high on ptp while only one part of the window moves. Continuous gait moves in *every* part — this is what separates walking from a standing subject who shifts weight or turns, the dominant residual error on this corpus |
| `feature_names()` | the single source of truth for column order. The static-offset family (angle means, interleg median) is **deliberately absent**: §4.6 shows it carries the subject's zeroing bias rather than gait, and keeping it measurably hurt held-out subjects |
| `segment_features()` | `zeros` and `ileg_zero` are **required, not defaulted** — substituting one silently is exactly the train/serve skew this pipeline exists to avoid |
| `rest_reference()` | returns `(zeros, ileg_zero, trusted)`. Untrusted is **returned, not raised**: such a recording still has to yield features, and the flag travels with them so `label.py` can say so instead of quietly guessing |
| `label_windows()` | `-1` excluded before voting, its share reported. A window not pure over {stand, walk} is `TRANSITION`, **never a coin-flip majority vote** |
| `windows_of_trial()` | windows drawn inside one segment only |

Whole-array computation is not a micro-optimization: dense per-row inference needs a window
every 25 samples over million-row recordings, which a per-window Python loop cannot deliver.

**Accepted imperfection, recorded rather than fixed:** `-1` exclusion is per **row**, not per
window, so a window can be "label-pure" on a minority of its rows — 94 of 5,984 contain
`-1`, 22 are more than half, worst is 93.5%. Decision (Lu): leave it. ~1.6% label noise, and
gating on `unknown_frac` would trade measured windows for a threshold nobody has evidence
for. `unknown_frac` rides on every window so the check is available if per-rev accuracy ever
splits along it.

### 6.4 `train.py` — fit the champion, score it honestly

Two guarantees enforced in code rather than in comments:

1. **The lockbox is never touched** — asserted, so a refactor that widens the split fails
   loudly instead of producing an optimistic number nobody can trust again.
2. **Validation is leave-one-rev-out.** A rev is one subject on one day, so a held-out rev
   is the closest honest stand-in for "a new person on a new day". Random k-fold would
   split one subject's trials across train and test and report a flattering, meaningless
   score.

| item | why |
|---|---|
| ExtraTrees, not RandomForest | identical features, folds and params, **over 5 seeds**: accuracy 0.9625 ±0.0003 vs 0.9568 ±0.0008, macro-F1 0.9194 vs 0.9138, **seed ranges disjoint** — the gap is the model, not the draw. At threshold 0.85 the two commit at the *same* accuracy (0.9887 vs 0.9889, ranges overlapping) while ExtraTrees commits to **87.6% of windows against 81.9%**: equal precision over more of the data, which is exactly what the abstention layer consumes |
| …and the two honesties about that comparison | **RandomForest is genuinely better on balanced accuracy** (0.9246 vs 0.8995, disjoint) — it recovers more of the smaller class, paid for out of overall accuracy. And at a *single* seed RF appeared to hold a worst-subject edge, which is why per-seed numbers are quoted at all: over 5 seeds that edge dissolves into noise. **This is §2.6's lesson applied prospectively** — an effect that does not survive strengthening the test is not that effect |
| not gradient boosting | marginally higher raw accuracy, badly overconfident — **the worse failure here**, because an overconfident model does not abstain when it should. Stated with its limit: that measurement is inherited from the 23-feature stage and has **not** been re-measured at 42 features |
| `champion_spec.json` | the champion's identity as data next to the code that fits it — name, rationale, params, window spec, `drop_features: []`. The repo holds **one settled champion** (README non-negotiable 3), so the spec is a declaration, not a search space |
| `class_weight="balanced"` | the corpus is ~86% walking |
| `DEFAULT_INFERENCE_STRIDE_S = 0.25` | inference slides the same window at a fraction of its length so a row's probability is an average over several overlapping views; training stays non-overlapping |
| `trainable()` | selects **positively**, on membership of the trained classes. Excluding `TRANSITION`/`None` by name instead let a third label state through silently — 84 all-unknown windows reached `to_numpy(int)` and crashed it, a loud failure that would have been quiet contamination had the codes been numeric |
| `reference_stats()` | the constants the ambiguity reasons are stated against, measured on **training windows only**, so no magic numbers live in the labelling path and every one is reproducible by re-running this. The amplitude band is recorded here so S4 reads it off the champion instead of recomputing it — **one definition**, and it is load-bearing in two stages |
| joblib dump wrapped in `try` | metrics are the deliverable; the model artifact is convenience |

### 6.5 `locoeval.py` — the blind measure layer

Emits objective numbers and **no opinion**. Every judgement lives above it. Keeping that
separation is what stops a model being adopted because a narrative sounded convincing.

- Headline is **macro-F1**: at ~86% walking, "predict walk always" scores >0.8 accuracy
  while being useless.
- Second headline is the **selective curve**, because once a classifier may abstain, a lone
  accuracy number is meaningless without the coverage it was bought at.
- `worst_rev_accuracy` is reported beside pooled accuracy at every threshold: pooled
  accuracy averages a new subject with subjects the model has effectively seen, and the two
  **disagree by several points at every threshold**.

`evaluate()` recurses into itself per group for `per_rev_macro_f1` — one definition of the
metric, so a per-rev number and a pooled number cannot be computed differently.

### 6.6 `roweval.py` — the number you can believe

`train.py` scores **windows**, the unit the model learns on. A caller labels a CSV and gets
**rows**, scored by averaging every covering window — which changes both the accuracy and,
more importantly, the confidence ordering the threshold is set from. *Publishing window
numbers and shipping row behaviour would be measuring one thing and selling another.*

It runs the real `label.py` path, not a reimplementation.

| function | why |
|---|---|
| `fit_on()` | reference stats come from the **same subset** as the model. Computing them once over all revs would leak the held-out subject into the ambiguity explanations |
| `curve()` | only rows with a trainable ground truth count |
| `reason_report()` | the check every reason code must pass: **the accuracy the guess *would* have had if it had not been flagged.** A reason that fires on rows the model would have got right is noise dressed as an explanation |
| `unknown_agreement()` | the model never sees `-1`, so agreement there is independent evidence that the confidence signal tracks genuine ambiguity rather than its own miscalibration. Measured: abstains on 56.1% of human-`-1` rows vs 15.5% elsewhere |
| `--lockbox` | single use, and now spent |

### 6.7 `transform.py` — the raw→`lpf_view` bridge

The model trains on the `lpf_view` family but must **run on raw device CSVs**. This is the
bridge, and it reproduces HUROTICS' MATLAB exactly — to ~1e-13 on every paired recording
the corpus holds.

Two things here are easy to get wrong, and both are stated at the top of the file:

1. **The filter is CAUSAL.** First-order IIR, one pole, fc = 1 Hz, phase lag by
   construction. `filtfilt` would be "better" signal processing and **wrong** — it would
   shift the features relative to the labels the model learned.
2. **The sagittal axis is a DEVICE property, not a signal property.** Signal-only detection
   was measured at *below chance* (gait-band energy 16%, antiphase×amplitude 16%, raw
   amplitude 5%, against a 33% baseline). Worse than chance is systematic: the
   highest-amplitude Deg axis is reliably *not* sagittal, because `Deg_Z`'s variance is
   dominated by yaw drift. **Unknown variant ⇒ abstain.**

| item | why |
|---|---|
| `SAGITTAL_DEG_AXIS_BY_VARIANT` stores only the **Deg** axis | the gyro axis follows from the permutation S1 measures. Storing both would invite them to drift apart |
| the per-variant *shape* is kept although all three measured variants read `Deg_Y` | sagittality has no in-file signature, so "three revisions agree" is a fact about three revisions, **not a licence to default a fourth** |
| five typed exceptions | `UnknownVariantError`, `NotRawDeviceError`, `GyroUnitError`, `DegenerateClockError`, `AxisConflictError`. Each names a different remedy. A bare `KeyError` would tell the operator nothing about what to do |
| `GyroUnitError` refuses instead of converting | converting rad/s → deg/s here is defensible physics and is still refused: **no paired rad/s recording exists**, so the conversion has no ground truth behind it, and an unverified transform of the classifier's input is the exact silent skew this module prevents |
| `TRUST_UNCHECKED` is a named sentinel, not `None` | skipping the check becomes a decision at the call site rather than the silent default that let the §1.4 gap sit open |
| `matlab_dt()` vs `safe_dt()` | both exist, and `serve_dt` chooses `matlab_dt` **deliberately**: the training features were produced by a MATLAB that filters a whole trial with its *final* inter-sample interval. Substituting the honest median moves features by up to **20.1 deg**. Serve reproduces the upstream quirk — and guards the case where the quirk is fatal (a tied final timestamp gives `alpha = 0`, the recursion collapses to `y[n] = y[n-1]`, and the output freezes flat for the entire trial with nothing raised) |
| `load_raw_frame()` is the **single reader** for verification and serve | two copies drifting apart would mean "verified to 1e-13" describes a read production never performs |
| ragged / duplicate-name files raise | `df[name]` on a duplicated name returns a *frame*, not a column, so every read below it is ambiguous |
| `raw_csv_to_features()` returns both frames | the deliverable is rows-in/rows-out; re-reading a 132k-row CSV to recover the caller's rows would be the only reason to open the file twice |
| `provenance` dict | the axis a raw file resolves to is the one assumption this path cannot verify from inside the file, so it is stated every time |

### 6.8 `label.py` — the deliverable

Everything else in S2 exists to make this row honest.

| design point | why |
|---|---|
| **takes either shape of file**, dispatched on family resolved by name | the raw path is the one a device actually produces. Until it existed, the pipeline could only label files that had already been through HUROTICS' MATLAB |
| `SERVE_REFUSALS` caught at the CLI boundary | a refusal prints its reason and the remedy instead of a traceback. An operator holding an unservable recording needs to read **why** |
| refusal produces **no output file** | a labelled file the pipeline cannot defend is worse than none, because a controller would act on it |
| **rows in, rows out** | probabilities computed on the 100 Hz grid, mapped back to the input file's own rows by nearest time. A caller gets their CSV with columns appended, not a resampled one they must re-join |
| `uncovered` rows get **no guess at all** | extrapolating a class into time that was never observed is the invention §3.1 forbids |
| on the raw path the four derived channels ride along in the output | without them the file states a verdict over columns nobody can see — and the sagittal-axis bug would have been invisible in the deliverable as well as in the code |

| function | why |
|---|---|
| `load_champion()` | returns `(model, meta)` together — **loading one without the other is how train/serve skew starts.** `lru_cache` because `verify_serve` labels 36 files in one process and unpickling 400 trees each time dominates its runtime |
| `read_source()` | dispatch on **family marker**, not on "does this file happen to have the columns I want" |
| `_accumulate()` | difference-array sum/count per row: O(windows + rows) instead of O(windows × window length), which is what makes a 0.25 s stride over million-row recordings affordable |
| `score_frame()` | a row's probability is the **mean over every covering window**. That is what makes a boundary abstain on its own — rows near a state change are covered by windows that straddle it, so their probabilities pull apart and the mean lands mid-scale. **No separate boundary rule and no smoothing constant to tune** |
| `explain()` | reasons ordered **most specific first**, so a row near a boundary is reported as a boundary rather than as whatever else is also true there. `posture_shift` outranks everything except `uncovered`: "not upright at all" outranks any stand-vs-walk story told about it. `model_split` is the catch-all |
| untrusted-rest note appended to every row | said once, on every row, rather than silently |
| `describe_source()` | printed for **every** run, not only the interesting ones |

Each surviving ambiguity reason is validated by the accuracy of the guess it suppressed
(`OPERATING_POINTS.md`): `near_transition` 0.6090 and `low_excursion_gait` 0.7108 against
0.9901 confident. A reason that could not discriminate was removed.

### 6.9 `label_all.py` — the corpus sweep

**`label.py` in a loop and nothing more.** Every file goes through `label_csv`, the same
entry point the single-file CLI calls, so a batch run cannot drift from what an operator
gets labelling one recording by hand — *a separate batch implementation of the same
pipeline is exactly the skew this repo keeps refusing to introduce elsewhere.*

| decision | why |
|---|---|
| **an abstention is a result, not a crash** | over 91 files some refusals *will* fire, and one must not take the other 90 down with it |
| `SystemExit` caught alongside the declared refusals | `label_csv` uses it for health-gate rejections, which are the same *kind* of answer — the file is unservable for a stated reason. It is a `BaseException`, so a bare `except Exception` would let it end the sweep |
| `UNEXPECTED:` prefix + **non-zero exit** | a bug is not an abstention. The file was refused by a defect, not by a rule, so a batch run must not pass while quietly skipping it |
| partition asserted | every raw file lands in exactly one bucket — the same gate `s1_clean/clean.py` enforces. *A batch report that silently drops a file it choked on is worse than no report, because the count still looks complete* |
| session directories mirrored from `data/raw` | session identity comes from the directory, and flattening it would lose the only metadata this corpus trusts |
| `--summary-only` | the per-file CSVs carry every input column plus nine appended ones, so a full corpus run is a few GB. The question "what is servable and how much of it is confident" should not cost that |
| the report says coverage only | no label is read here. **What fraction of rows clear the threshold is a fact about the corpus; whether those calls are right is measured in `OPERATING_POINTS.md`** — the pair is not quietly half-quoted |

### 6.10 The three verifiers

| module | what it defends |
|---|---|
| `verify_features.py` | `swap_counts` (vectorized) vs `rest.swap_count` (scalar) on random signals **including the degenerate cases** — flat, all-committed, window shorter than the hysteresis. That rewrite is where a silent error hides: exact on most windows, off by one on the rest, which no accuracy number would ever reveal as a bug rather than as noise |
| `verify_transform.py` | the bridge's **math** — four columns rebuilt from raw to float roundoff on every pair. Reads through `load_raw_frame`, the same reader serve uses. Passes `excluded=set()` on purpose: `EXCLUDED_TRIALS` quarantines trials whose *labels* are wrong and this check never reads a label |
| `verify_serve.py` | what a caller **actually gets** — the same recording labelled both ways, row for row. Between the four columns and the answer sit gap segmentation, FIR decimation, the rest reference, 42 features and a 400-tree ensemble; none of that is exercised by comparing columns. Also sweeps `data/raw` and reports what abstains, by reason, because *"the serve path works" and "the serve path works on 86% of the corpus" are different claims and only one is true* |

`verify_serve._comparable()` drops `Label` structurally: **the lockbox guarantee expressed
as code**, so the function cannot return a column anything downstream could score against.

**All three are wired into `run_pipeline.py`, at two different prices.** `verify_features`
is **unconditional and runs first**: it reads no data, costs ~2 s, and holds the vectorized
feature path to its scalar reference — which every downstream number rides on — so a broken
vectorization stops the run in seconds instead of after the minutes it takes to clean. The
other two read the whole corpus and are `--verify` opt-ins: `verify_transform` before
`s2_train` (a bridge that cannot reproduce its features should stop the run before fitting
400 trees on them), `verify_serve` after it (it needs the champion).

That wiring was **lost and restored on 2026-08-04**. The `--verify` flag did not survive the
`orchestrator.py` → `run_pipeline.py` rewrite, and both scripts were deleted from the working
tree at the same time, while this section, `caveats.md` §3.1 and DOMAIN_NOTES §6.2 all went
on citing their numbers as re-runnable. An unrun verification is `§1.4` in its purest form —
a check that exists and is not a consumer of anything — and a *deleted* one that four
documents still quote is the same failure with the evidence removed.

`verify_transform.find_pairs` accepts 5 s of end-time slack while comparing rows by index —
§7's "never select by index" hazard. It was **flagged rather than changed**, then measured:
every annotated trial matches zero or one raw candidate, each at `max|Δt| = 0`. The slack
buys nothing and risks nothing *on this corpus*; it is still the wrong rule to carry
elsewhere, so it stays listed in `caveats.md` §3.4.

### 6.11 `predict.py` — dense-stride inference, and why there are two strides

Turns a windowed classifier into per-row predictions: slide the window at a small stride,
assign each prediction to the rows around its **centre**. Resolution becomes the stride
rather than the window, and centring (not leading-edge) keeps transitions unbiased.

The one number worth arguing about is `DEFAULT_INFERENCE_STRIDE_S = 0.1`, and it is
**deliberately not** `train.DEFAULT_INFERENCE_STRIDE_S = 0.25`. Two strides, two jobs: 250 ms
is the serve path's, priced for a customer's whole recording; 100 ms is the measurement
path's, chosen to sit finer than `FLICKER_MAX_MS = 200` so a flicker is *expressible at
all*. At 250 ms the error taxonomy's flicker bucket would read zero by construction — see
DOMAIN_NOTES §, where that hazard is recorded as a live warning against the serve stride.

### 6.12 `taxonomy.py` — score our errors on the incumbent's yardstick

A verbatim port of the sibling repo's `locoeval/diagnose.py` bucketing — precedence
`correct > omission/swallowed/edge_omission > flicker > late > early > steady_confusion` —
with the thresholds **copied, not tuned**, because they are what make the two repos'
numbers comparable. `FLICKER_MAX_MS`, `LAG_MAX_MS` and `SUSTAINED_FRACTION` are constants
of the comparison, not knobs of this pipeline; tuning one silently would end the
comparability it exists for.

It takes dense per-row predictions, never window labels. **This module was deleted with the
champion/challenger loop and came back with it on 2026-08-04**; DOMAIN_NOTES carried "no
longer in this repo / nothing computes it" for the gap in between.

### 6.13 `experiment.py` — the only path to champion

An agent proposes an `ExperimentSpec`; this runs it, scores it with `locoeval`, applies the
rule, and logs **every** outcome — rejections included — to `experiments.jsonl`. Promotion
is never an agent's call.

| decision | why |
|---|---|
| `decide()` gates on leave-one-rev-out macro-F1 past `PROMOTION_MARGIN` | one metric, stated in advance, computed by code. The agents never see a promotion switch |
| `comparable()` **refuses** to subtract across corpora | a macro-F1 delta is meaningless across a changed class set, corpus or window count. Fail-passive: an unverifiable comparison keeps the incumbent rather than promoting on it. The comment names the real incident — a two-class incumbent scoring six-class challengers for a whole import |
| `STEADY_CONFUSION_MARGIN = 0.02`, the tiebreaker | on a macro-F1 tie, prefer lower `steady_confusion`: *a sustained wrong call becomes a sustained wrong ACTION on a powered device, whereas omissions fail passive.* This is the one place the error taxonomy is load-bearing rather than descriptive |
| one LORO pass fits both windows and rows | the fold holding a rev out is the same model that scores that rev's OOF windows and its dense rows, so it is fit once for both. `oof` is filled by mask, so fold order cannot matter |
| `champion_spec.json` tracked in git, `champion.json` not | `runs/` is gitignored; the spec is the champion's identity as *data*, committed alongside the claims it supports. `train.py` stamps it via `freshness.stamp_inputs`, because a promotion rewrites it and until a refit follows, everything in `runs/s2_ml/` describes the model that just lost |
| estimator and `trainable` imported from `train.py` | this file arrived from a sibling repo whose champion was a RandomForest over a `label != TRANSITION` filter. Both are wrong here, **and both are the kind of wrong that still produces a number** |

### 6.14 `raweval.py` — the accuracy of the route a caller actually uses

`roweval` scores the annotated `lpf_view` export. `verify_serve` shows the raw device route
agrees with that export row for row — but it drops `Label` before comparing, so it proves
*equivalence* and never touches *correctness*. The raw path's accuracy was therefore only
ever available transitively: raw ≡ lpf_view, lpf_view is 0.99, so raw is 0.99. Sound, and
still a chain. This measures the endpoint directly — raw CSV in, `label_csv` out, joined
against the human annotation of the same recording.

| decision | why |
|---|---|
| `_drop_lockbox()` refuses rev8 **in code** | rev8 has four paired recordings. §7 spends the lockbox once and it is spent; a "direct" number that quietly re-read it would be worth less than the transitive one it replaced. Refusing by flag would make that a thing to remember |
| each rev labelled by a champion refit without it | reference statistics included, since those enter the abstention reasons. The shipped `champion.joblib` saw every training rev, so pointing it at rev7's raw file would report memorization |
| reported **per variant** | the axis map is per variant, so a mis-mapped sagittal channel shows up as one row sitting apart from the others — a check the transitive argument could not perform at all |

The narrowness is stated rather than smoothed over: 3 development subjects, 13 CSVs, no
lockbox. It is a direct measurement of a smaller population, **not** a second opinion on the
row-level curve.

### 6.15 `transitions.py` — what the biggest abstention bucket is made of

`near_transition` suppresses more rows than any other reason and has the weakest suppressed
guess (0.6090) of any in `OPERATING_POINTS.md`, and until 2026-08-04 nothing measured the
thing it fires on. Three questions, and the third can embarrass the pipeline: how many
transitions, how long is one, and **does the pipeline time them right?**

| decision | why |
|---|---|
| "how long" = the annotator's own `-1` interval | ground truth does not step instantaneously — it goes stand → "I looked and cannot call it" → walk. The width of that middle run is a *measurement* of the boundary rather than a model of it |
| a boundary is *timeable* only with a full window either side, and **rejections are counted, not dropped** | below that the model has no pure window to place a change from, and scoring its timing would measure the window length. Counting the rejects is the fix for the original defect: `transition_report` computed runs on a sequence it had just filtered, which is how 185 "on-time" boundaries turned out to be mostly an artefact of deletion |
| timing measured on the **guess**, not the committed label | where the model thinks a boundary is does not depend on whether it cleared the threshold on either side |
| `near_transition_regions` counts REGIONS, not rows | a contiguous stretch of flagged rows is one event however wide it is. 77,025 rows is meaningless as a count of anything the flag claims to have found; 384 regions is not |

It reports the flag's recall (218 of 267, 82%) **and** its precision from the other side
(167 of 384 regions, 43%, contain no annotated transition within ±2 s) — the pair, never
half of it.

---

## 7. `stages/s3_physics/` — the model-free view

### 7.1 `anchors.py` — the swap rule

> Walking is the legs alternating. Not how far they swing — *whether they swap*.
> 0 swaps → STANDING. ≥2 → WALKING. Exactly 1 → AMBIGUOUS.

`1` is genuinely ambiguous, not a fudge: one leg passing the other happens both when you
take a step and when you shift your weight. **The rule abstains there rather than guessing.**

**Zero fitted parameters — with a stated boundary.** `delta = 1°` is a sensor noise floor
(5× measured standing noise came out 0.76–0.88° on three files independently), and `1 swap`
is not a chosen threshold: standing measures 0 and walking measures 2 on every file across
a **4× amplitude range**, so 1 is the only integer between them. §10.7 says plainly where
that claim stops being literally true — `6.0` s, `0.35`, `8.0°` and `0.5` on the slow-gait
path *are* chosen constants. They were picked to make a physically-stated distinction, not
tuned against a score, but "zero fitted parameters" is a statement about the **core rule**,
not about all of S3.

**The independence claim is retracted.** This docstring used to say the rule "reads a
different quantity" than the classifier. That stopped being true when S2 absorbed
`ileg_swaps` / `ileg_minhalf` / `ileg_minquarter` — both now read the same 1 Hz-filtered
interleg angle. Physics contradicts only ~12% of S2's high-confidence errors, and at
p ≥ 0.95 it catches none. Removing the interleg block from S2 to restore separation was
tried and does not help: **the correlation is in the signal, not the feature list**
(`caveats.md` §1.1c).

| function | why |
|---|---|
| `_corr1()` | a reshape over S2's vectorized `_corr`, **not** a reimplementation — the point is that S2 and S3 cannot compute correlation differently, including the convention that a constant series yields 0.0 rather than NaN. A sibling repo's second copy drifted, and its grow gate calls `_corr` without importing it at all |
| `stride_period()` | the first autocorrelation local max **after the acf dips below zero**. Skipping the lag-0 shoulder is essential rather than tidy: on slow gait a plain argmax grabs the decay shoulder, returns a period of a few samples, and sizes the adaptive span far too short — re-creating the problem it exists to fix |
| `_periodicity()` | rhythm **strength**, amplitude-independent, within-window only. Read it as "steady rhythm here", never "this subject has a rhythm": §11.2 records a whole invented category that came from reading a per-window periodicity dip as a property of the walker, when it was a cadence *change* at a protocol event, in both trials |
| `_grows_to_walking()` | **three** gates, not a swap count. Over 6 s, a person who shifts their weight twice produces two crossings indistinguishable from a slow walker by count alone. Amplitude (both halves swing ≥ 8°) and antiphase (legs oppose at ≥ 0.5) are what separate them |
| `_adaptive_span()` | single-sourced, so the swap count and the adaptive periodicity read the **same** span. Two spans would let a verdict and its own supporting evidence describe different stretches of time. Capped at 6.0 s because an unbounded span eventually swallows a whole bout and averages two states into one verdict |
| `_adaptive_cell()` | returns the swap count **over the span that produced the verdict**, because the fixed-window count describes a different stretch of time and a reader shown only that one sees `WALKING` beside `swap_count 0` and concludes the rule was violated |
| `window_anchors()` | `interleg_center` is subtracted **before the swap count only**. For the descriptors the offset *is* the posture — `interleg_offset` separates feet-together from split stance at +17°/−22° — so removing it there would delete the signal |
| `ileg_ptp` **alongside** `ileg_minhalf`, not replacing it | the two answer different questions and confusing them cost a wrong conclusion. `ileg_minhalf` asks *"did BOTH halves swing"* — the right question for the §10.7 grow gate and the **wrong** one for *"how far did the legs separate"*. It collapses when the motion sits in one half of the window, which is exactly what a start/stop ramp looks like, so reading it as amplitude **under-reads the ramps hardest** — and the ramps are the known label-failure mode |
| `gyro_energy` | defined the way that **fails** the rate audit, on purpose — the audit's negative control, since an audit that has never rejected anything is not evidence that the others passed (§11.1). The audit itself went with S4 (it existed to certify anchor *features*, and anchors are no longer a feature source); this descriptor is kept as the record of why the check mattered |
| `trial_anchors()` | walks the identical window grid `features.windows_of_trial` walks. That exactness was built for a join that no longer exists; what it buys now is that `label_audit` scores **the same windows S2 does**, so a disagreement is about the labels and not about the windowing. Uses **S2's** `rest_reference`, not a second opinion about where rest is |

### 7.2 `label_audit.py` — point the model-free opinion at the *annotations*

**Model-free is the whole point.** Flagging the files the classifier dislikes would delete
exactly the hard cases, improve every metric, and teach nothing. The property to protect is
that it never imports a model, a probability or a prediction — importing `s2_ml.dataset` and
`s2_ml.features` for trial loading and the window grid is fine and necessary, *so that it
scores the same windows S2 does*.

**Two detectors, because one trial-level failure is not the only one:**

| detector | what it catches |
|---|---|
| `disagree_frac` | labels that contradict the physics **outright** — a swapped channel, a misaligned label track. Blunt by design: 0.50 is "the file disagrees with itself more often than it agrees", the only non-arbitrary line available, and the corpus median is near 0.03 |
| `band_walk_frac` | the failure one cannot see: a trial whose labels are locally consistent but drawn to a **different convention** than the corpus. Both classes genuinely occur in the band, so a trial can call it `stand` or `walk` without ever contradicting the physics — and trials do, from 0.38 to 1.00. This one number rank-correlates with a trial's confident-error rate at **−0.66 (p=0.004, n=17)** |

`band_policy()` uses a two-sided binomial test **Bonferroni-corrected** across the trials
actually tested, not a threshold on the fraction: a trial with 6 band windows at 0.50 is
unremarkable and one with 150 at 0.76 is not, and only the test knows the difference. With
~17 testable trials an uncorrected 0.05 would flag one by chance nearly every run, *and a
detector that cries wolf every run gets ignored.*

**A band flag is not grounds for exclusion**, and the report says so in bold. A trial
annotated to a self-consistent minority convention is not broken data — it is a measurement
of how much residual error is annotation policy rather than model failure.

`main()` loads with `excluded=set()`: a trial already quarantined must still appear, or the
report silently stops justifying the exclusion it caused.

Open, and stated: the `disagree` line has separated exactly **one** case from 42 (rev13/4 at
0.95, next worst 0.30), so it is validated against a single positive example and its
false-positive rate is unmeasured.

### 7.3 `inspect_window.py` — hand a suspect window to a person

A mislabel nomination is worthless until someone checks it against the signal. This is a
**reader**: it changes nothing and deliberately cannot write a label.

**Who nominates changed on 2026-08-04.** It used to be the `s4_review` agent, writing
`runs/s4_fusion/review_queue.jsonl`; both went with S4, leaving this tool requiring a
`--t SECONDS` that nothing in the repo produced. `label_audit.nominate` replaced it: the
audit already knows which windows contradict their label, so the nominations now come from
the same computation as the flag rather than from a separate agent's notion of a suspect
window. It emits ready-to-paste commands into `label_audit_windows.jsonl`.

It also reads with `excluded=set()`, which is load-bearing rather than defensive.
`find_trials` applies `EXCLUDED_TRIALS` by default, so with the default this tool could not
open the one trial in the corpus that has been excluded — while that exclusion's entry asks
for exactly the internal, model-free evidence this module prints. **A reader that hides the
quarantine cannot audit the quarantine**, and re-checking an old exclusion is as legitimate
as justifying a new one.

**It prints numbers, not a plot, and that is the point.** §11 item 4 records a bout
eyeballed as "2.5 s" that measured 6,246 ms. The columns are the quantities the swap rule
actually uses, at the resolution it uses them, so the reader adjudicates on the same
evidence the rule did rather than on the shape of a curve.

`crossings()` mirrors `rest.swap_count`'s state machine rather than re-deriving it, so
jitter around one threshold produces one event, not many. It uses the **same** rest zero S2
and S3 use — measuring a fresh one here would adjudicate the window against an origin
neither stage used.

### 7.4 `rate_audit.py` — does an anchor describe the body, or the sampling grid?

The gate is *no anchor feature without a rate-invariance verdict*, and §2.3 is why:
experiment id=69 read clustering that partitioned by "acquisition rate" across
99.4/99.7/100.0/500.0 Hz — which §2.2 later showed was timestamp quantization. **The
geometry was measuring the clock.**

**Deleted with S4 on 2026-08-04 and restored the same day**, which is the more useful
lesson of the two. It was removed for being `run.py`'s neighbour, not for failing its own
test: `run.py` built the fusion join's anchor table *and* drove this audit, so deleting the
driver took the audit with it. Deleting a driver is not a reason to delete what the driver
called. It has its own `main()` now and does not resurrect `anchors.csv`, the 1.4 MB
per-window table that genuinely did exist only for the join.

| item | why |
|---|---|
| gated to WALKING windows via the swap rule | keeps the audit **label-free**. Standing has no cadence to compare, and using ground truth to select the audit set would make a physics check depend on the annotations it is supposed to be independent of |
| `AUDIT_PAD_SAMPLES = 128` | `decimate(..., ftype='fir')` builds a 41-tap filter and `zero_phase` runs it through `filtfilt`, whose padlen is 123. On a bare 200-sample window **over half the window is the filter's boundary behaviour** — that artefact alone moved `antiphase` by 0.1734 and condemned the literal gait signature as `rate_dependent`. With real context either side it moves 0.0005 |
| windows without full context on both sides are **skipped, and counted** | a half-padded window carries the artefact on one edge only, which is harder to reason about than not measuring it |
| `ANCHOR_METRIC` splits abs vs rel | an absolute delta on a ratio-scale magnitude is meaningless; a relative delta on a correlation blows up whenever it passes through zero |
| never raises on a moved anchor | it **reports**. `gyro_energy` failing is the expected outcome, and `main()` warns when it *doesn't* — a sweep that rejects nothing is not evidence that everything passed (§11.1) |

### 7.5 `serve.py` — the swap rule on a file that has no labels

`label_audit` points the rule at the annotations, the one thing it is still independent of.
This points it at a raw recording, so the serve path can have a physics opinion at all.

**It returns windows, not rows.** `segment_verdicts` mirrors `features.segment_features`
exactly — same grid walk, same `(values, starts)` shape — and stops there. Mapping windows
onto rows stays `label.py`'s `_accumulate`, because a row's physics has to be averaged over
the same covering windows, by the same difference array, as that row's probability. A
second window-to-row mapping here could drift from the one the probability uses, and then a
gate would compare two quantities that describe different rows while looking like they
describe the same one. `label.score_frame` asserts the two grids match element for element
rather than trusting that they do.

`adaptive_verdict` was split out of `anchors._adaptive_cell` so this path can get a verdict
without paying for the periodicity autocorrelation it does not use. **The split is a
factoring, not a second implementation** — a physics gate on the serve path computing its
verdict differently from the physics reported on the corpus would be two rules wearing one
name, the precise skew that made S4 indefensible.

### 7.6 `plausibility.py` — file-level sanity bounds on the *output*

`label_all` refuses files for INPUT reasons. Nothing checked the output, so a recording
with miswired channels passed every gate, produced confident labels, and was wrong end to
end with no artifact saying so.

**File level, not window level, and that distinction is the whole justification.** The
per-window objection (§7.1's retraction — physics contradicts ~12% of S2's high-confidence
errors, none at p ≥ 0.95) is about which *window* is right. It says nothing about "this
recording commits to nothing", which is a statement about the file and corresponds to a
diagnosable fault.

The bounds are sited outside a measured range rather than tuned, because the faults are
absent from the corpus by construction — a recording with miswired channels never reached
the annotation stage. So the module carries **synthetic controls** instead, and they are
what justify believing it:

| injected fault | model walk | physics walk | committed | caught by |
|---|---|---|---|---|
| (unmodified) | 0.9935 | 0.9919 | 0.9981 | — plausible — |
| `R := L` duplicated leg | 0.0000 | 0.0000 | 0.0064 | `commitment_collapse` |
| `R := const` dead sensor | 0.0000 | 0.9894 | 0.0066 | `commitment_collapse` + `physics_loud` |
| `L <-> R` swapped legs | 0.9935 | 0.9919 | 0.9981 | **nothing** |

Three things that table is load-bearing for. **`commitment_collapse` is the workhorse**,
and that was not the expected answer — both destructive faults are caught because the model
stops committing (0.0065 against a corpus minimum of 0.1487 over 78 real files), not
because the physics contradicts it. **The swapped-leg fault is invisible**, and that is
arithmetic: swapping the legs negates `L - R`, and `swap_count` thresholds symmetrically,
so the rule cannot see it — `caveats.md` §3.4 keeps that listed and this does not fix it.
And **`physics_silent` / `physics_loud` have never fired on real data**, which is stated
rather than left for a reader to assume otherwise (§11.1).

The instructive near-miss: rev13 t4 looks like the perfect `physics_loud` case — 0.00
annotated walk against 0.9429 physics walk — but at serve time the bound compares physics
against the *model*, and the model calls it 0.9916 walk. Model and physics agree; the
annotation is the outlier. A serve-path check has no annotation to compare against, so it
**cannot** catch that trial, and `label_audit` is what does. The two divide the work rather
than overlapping: **`label_audit` distrusts the labels, `plausibility` distrusts the file.**

---

## 8. The documentation files, and why they are separate

| file | job | why separate |
|---|---|---|
| `DOMAIN_NOTES.md` | what the corpus taught us | injected into every agent prompt. Every entry is *a finding with a reason*, provenance-tagged `[measured]` / `[reported]` / `[decided]` / `[open]`. **Records findings, not tallies** — live counts belong in run artifacts, because a count in prose goes stale silently while a finding does not |
| `caveats.md` | what the pipeline is **shaky** about | a reader who trusts a number needs a single place that says what is thin, unverified or deliberately omitted. Mixing it into DOMAIN_NOTES would let a caveat read as a finding |
| `OPERATING_POINTS.md` | the threshold, and its price | one knob, one file, with the full curve and the reason 0.85 is default (**the lowest threshold at which every held-out development subject independently clears 95%**) |
| `README.md` | how to run it | deliberately last in the reading order. **Deleted 2026-08-04 and not yet rewritten** — `needtowrite.md` holds the spec and the un-regenerable prose, and `breakdown.py` flags its absence every run |
| this file | why any of it is shaped this way | the justification layer. A reader who disagrees with a decision should be able to find the evidence it rests on and attack that, rather than re-deriving the decision |

Two conventions worth preserving:

- **Retractions stay in place.** §4.1's noise claim, §5.5's subject claim, §6.2's axis table
  and `anchors.py`'s independence claim are all retracted *in the position where the wrong
  claim lived*, with the reason. A quietly corrected note teaches nothing.
- **Every stage closes with a DOMAIN_NOTES update.** Discoveries become permanent, not
  conversational.

---

## 9. What is deliberately NOT built

Excluded on purpose, not by oversight — the experimental work lives in a sibling repo.

- **S4 fusion — built, measured, and deleted (2026-08-04).** It joined S2's out-of-fold
  predictions to S3's anchors and published a (coverage, accuracy) pair on the 2 s window
  grid. `label.py` never imported a line of it: different unit, different abstention rule, a
  reason vocabulary sharing one code out of five. **Two accuracy proofs for one product, and
  the authoritative-looking one described a path no customer CSV ever took** — §1.4's lesson
  arriving as a whole stage rather than a field. Its distinguishing policy, abstaining on the
  annotation-ambiguity band, was then measured at row level and lost outright to raising the
  threshold (`OPERATING_POINTS.md`). `stages/s2_ml/oof.py`, `stages/s3_physics/run.py` and
  `rate_audit.py` went with it — their only consumer was that join. What survives of S3 is
  `anchors.py` as a library and `label_audit.py` as its consumer: the swap rule pointed at
  the annotations, the one question it was ever independent for.

- new-class discovery and the discriminator registry
- the stairs-ascent rule — measured to be **ascent-only**: `lift_rest` separates stairs-up
  from walking, but stairs-down sits *below* walking and overlaps decline, so no threshold
  over those descriptors can name it
- the 11-class taxonomy — **data and code are not interchangeable between the two repos.**
  The same `rev2/trial_1.csv` reads `[-1, 0, 10]` here and `[-1, 10, 20]` there; mixing them
  relabels every row silently, with no error raised
- hypothesis generation from plots — S3 here is deterministic and has no agent
- the champion/challenger agent loop — this repo holds **one settled champion**;
  experimentation lives in the sibling repo

---

## 10. The honest limits

Stated here so this document cannot be read as a defence.

1. **The target is missed on the only genuinely unseen subject.** rev8: 0.9308 at coverage
   76.4%, entirely one class — walk recall 1.0000, stand recall 0.5752, all 2,197 confident
   errors standing called walking. And accuracy there is **non-monotonic** in the threshold
   (0.8997 / 0.9308 / 0.9184 at 0.70/0.85/0.95), so the confidence ordering does not transfer
   and `worst subject` in `OPERATING_POINTS.md` is an optimistic floor.
2. **Nothing predicts which new subject will degrade.** Five candidate signals were built and
   tested; the strongest correlation (`pred_walk_frac`, ρ = +0.75) is **class balance wearing
   a difficulty measure's clothes** and is worthless. On a corpus this imbalanced, any flag
   built from prediction statistics measures the class mix, not the difficulty.
3. **The confidence layer cannot close the gap**, and this is provable rather than
   measurable for recalibration: Platt/temperature/isotonic are *monotone* maps on `p`, so
   accuracy at fixed coverage is invariant under them, and rev8's defect is an **ordering**
   defect. Nine alternatives were measured at matched coverage; the one that looked best
   **inverted** when the perturbation it was attributed to got three times stronger.
4. **Seven revs is a small basis** for a threshold, a probability floor and tier boundaries.
5. **Confident accuracy — 0.9904 on the shipped row path — is a lower bound on model
   quality and an upper bound on label quality, and the two cannot be separated by
   measuring harder.** `label_audit.py` measures the second half directly and model-free:
   **8 of 41 trials annotate the ambiguity band unlike the corpus**, one at 0.32 walk
   against a corpus rate of 0.87 (p=2.8e-09). That is annotation policy, not bad data, and
   no threshold can recover a convention the corpus does not share. Separating model error
   from label error needs a person on the queue.
6. **86%, not 100%, of raw files are servable.** `e5f2660f` and `86069795` have no paired
   recording, so the newest two hardware batches abstain — correct behaviour, and still a
   gap until one paired file exists.
7. **The drift flag has no consumer.** `drift_contaminated` is inert, and only costs nothing
   because no feature reads `Deg_Z` at all. **The first yaw-derived feature must add the
   consumer.**

---

## 11. Changes made while writing this document

Everything below was found by reading the code against the notes. Nothing here changes a
label, a confidence, or any measured number — the verification suite and the end-to-end
serve path were re-run after the edits (see §12).

### Deleted

| what | why |
|---|---|
| `roweval.reason_report`: `v["human_unknown_near"] = False` | written, never read. A column that exists only to be assigned is the smallest version of "recorded ≠ handled" |
| `label.explain`: `change &= scored["segment"] == scored["segment"]` | **a tautology.** An array is always equal to itself, so the line was a no-op that read as a segment-boundary guard. Removed, and replaced with a comment stating what the code actually does: a class difference across a segment boundary *does* mark both sides `near_transition`, which is the conservative call because neither side has a full window of measured context. Behaviour unchanged |
| `census.Variant.raw_columns` | zero readers — `to_dict()` never emitted it and `fingerprint()` takes the header separately. Same standing as `peek.py` and `oof._folds_match_logo`, removed in an earlier pass for the same reason |
| `orchestrator.py`, whole file | a **second driver for one pipeline**. `run_pipeline.py` shelled out to it twice and nothing else ever called it, so it bought a duplicate dotenv/console setup and a second answer to "how do I run this". Its two review functions moved into the runner as `Step.fn`; `--from s1_exception` / `--from s4_review` replaces `--stage s1` / `--stage s4`. See §3.2 |
| `dataset.main()` and `features.main()` with their `__main__` blocks | census and window-summary printers with **no `ArgumentParser`** — the only two entry points in the repo like that, wired into nothing and documented nowhere. `census()` and `build_windows()` are untouched; only the unreachable CLI shell went |
| `runs/s2_ml/label_audit.{json,md}` | orphaned output of the deleted `stages/s2_ml/audit.py`, and a *contradictory* duplicate: it reported rev13/4 at 0.9508 against the live S3 audit's 0.9516, for the same question. Two answers to one question is worse than none |
| `runs/s2_ml_baseline/`, `runs/baseline_all23/champion.joblib` | a run whose macro-F1 was 0.5 — chance — kept by accident, and a 7.5 MB model that cannot be loaded against the current 42-feature set. `baseline_all23`'s two small metric files were **kept**: that 23-feature / 0.8881 number is recorded nowhere else and is not regenerable |

### Fixed

| what | why it mattered |
|---|---|
| `python -m stages.s2_ml.label --help` crashed | `--full`'s help string carries an em-dash; on the cp949 console argparse's own help write raised `UnicodeEncodeError`. **The repo's headline deliverable could not print its usage.** `inspect_window` had the identical bug through `description=__doc__` |
| three copies of the `reconfigure(errors="replace")` guard | the fix above would have made a fourth. Now one function, `stages/console.py`, called first in each `main()` — the same anti-duplication argument `features.py` makes about `_corr` having drifted once already |
| `--phase N` on the driver | numbered project milestones, not stages, and the two schemes had drifted apart: milestone 4 was *S3 physics* while `--phase 4` ran the *S4* review. The README status table was re-keyed by stage for the same reason |

### Revised (stale claims)

| where | was | now |
|---|---|---|
| `transform.py` docstring | "verified … against **19** paired recordings" | 18 — `caveats.md` §3.4 records that enumerating them yields 18 and the "19" was never reproducible |
| `transform.serve_dt` docstring | "on the **17** verifiable pairs" | count removed. 17 reconciles with neither 18 nor 19, and I will not mint a number I did not measure. The 20.1 deg figure it carries is unchanged |
| `dataset.find_trials` docstring | "`stages.s2_ml.audit` proposes exclusions" | `stages.s3_physics.label_audit` — the named module no longer exists |
| `roweval.py` docstring | "rev8 **and rev13** were sealed" | rev8 only, and it is spent. rev13 was deliberately released to development (`dataset.DEFAULT_LOCKBOX_REVS` says why) |
| `locoeval.py` docstring | "the corpus is ~**82%** walking" | ~86%, matching `caveats.md` §3.0 |
| `fuse.py` + `caveats.md` §1.1d | "same failure as `moving_stand` in `label.py` (§2.6)" | `moving_stand` is gone from `label.py`, and `caveats.md` §2.6 is now the nine-confidence-signals entry, so the cross-reference pointed at the wrong section. Rewritten to state the shared *check* — measure the accuracy of the guess the reason suppresses — and to name where each is run |
| `anchors.py` docstring | "moved walk-recall 0.69 → 0.86" | 0.7092 → 0.9384 with AMBIGUOUS 24.0% → 5.3%, the measurement §10.5 actually records |
| `inspect_window.inspect` | `load_trial(paths[0], "dev")` | the trial's real split via `assign_split`. `Trial.split` is a three-value field and a fourth value would make the record lie about a lockbox trial |

### Found and deliberately left alone

- **`validate.Health.rest_trusted` has no reader.** It is not deleted because the same fact
  already reaches the caller as a *warning* string that `label.py` prints; the field is the
  structured form of information that is genuinely surfaced. Recorded here so it is a known
  redundancy rather than a discovery.
- **`caveats.md` §3.5's disagreement threshold rests on one positive example.** Real, open,
  and already stated there — not something a documentation pass can close.
- **The `near_transition` marker crosses segment boundaries** (see the deletion above).
  Adding a real guard would change measured coverage and therefore every number in
  `OPERATING_POINTS.md`, so it is documented rather than silently changed. If it is ever
  wanted, it is a behaviour change that must be re-measured, not a bug fix.
---

## 12. Verification after this pass

| check | result |
|---|---|
| `compileall` over `stages/`, `agents/`, `run_pipeline.py`, `freshness.py` | clean |
| import every module under `stages/` + `agents/`, plus both root scripts | 40/40 |
| `--help` on all six documented commands | 6/6 respond; two were crashing before this pass (§12) |
| `run_pipeline.py --dry-run [--verify] [--with-agents]` | 9 / 11 / 14 steps — the free spine, `+2 verify`, `+3 agent`. `--from verify_serve` without `--verify` names the flag that brings it back. There is no `--list`: `--dry-run` is how you see the step keys, and it works on a checkout with no data |
| `python -m stages.s2_ml.verify_features` | 2,400 windows, 0 mismatches; `_min_over_parts` 0; `WindowSpec` 0 |
| `python -m stages.s2_ml.train` re-run to add the band | macro-F1 CV **0.9193053789761384**, bit-identical to the previous champion; `reference_stats` gained `band_lo`/`band_hi` and changed nothing else |
| `python -m stages.s3_physics.label_audit` | band **2.62–20.34°** measured independently of `train.py` and agreeing with it; 1 trial broken, 8 divergent |
| `census.build_registry` + `stable_prefix` on a raw sample | 45-column stable prefix, variants resolve as documented |
| `python -m stages.s2_ml.label` on a raw device log, end to end | serves: variant `0fda484e`, `Deg_Y → Gyro_Z`, trust record consulted, 5,470 rows labelled with the reason breakdown intact |

`verify_transform` and `verify_serve` **were** re-run, on 2026-08-04, after being restored
from `HEAD` (§6.10) — the previous pass skipped them as "the expensive ones", which is how
their deletion went unnoticed. Both reproduce their documented numbers exactly:

| check | result |
|---|---|
| `python -m stages.s2_ml.verify_transform` | 18 pairs, 0 abstained, worst 7.25e-13 (tolerance 1e-9) — PASS |
| `python -m stages.s2_ml.verify_serve` | 18 pairs / 536,590 rows, **0 disagreeing verdicts**, max \|Δconfidence\| = 0 — PASS |
| the same run's corpus sweep | 78 of 91 servable (86%); `fb5ea2c2` ×62, `0fda484e` ×11, `4bfd6ab2` ×5, all `Deg_Y`; 13 abstain, **every one `UnknownVariantError`** |

Run them after any change to `transform.py` or the serve dispatch, or run the whole spine
with `python run_pipeline.py --verify`.

**A note on how this document was written.** The tree moved while it was being written:
`freshness.py`, `run_pipeline.py`, `label_all.py` and `champion_spec.json` appeared, S3
began emitting `ileg_ptp` and `train.py` gained the 5-seed model comparison. This file was reconciled against each of those rather than
describing the tree as it was at the start — a justification document that describes a
previous version of the code is the same failure it spends §1.4 warning about. If it
disagrees with the code, **the code is right and this file is stale**; fix it here.
