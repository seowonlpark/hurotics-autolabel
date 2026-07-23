# Build Log

This log records decisions made throughout the developement of this pipeline and the evidence 
behind each one, so a reader can understand *why* the structure looks the way it does.

**How to read this.**
Entries are grouped by pipeline stage (S1, S2, S3, S4), followed by cross-cutting principles (X) 
and agents (A). Every entry has a stable ID (for example `[S1-3]`) that the technical reference, 
`PIPELINE.md`, cites when it explains a design choice. The `[SX-0]` entry in each stage describes 
that stage as it is currently built; the entries after it are the decisions that shaped it.

---

## S1 - Cleaning

### [S1-0] Current structure and function

Turns raw device logs into trustworthy table per recording; it is fully deterministic

The stage has two entry points:
(1) **census** (`stages/s1_clean/run.py`) - measures the corpus and writes a human-readable 
    `census.md` plus a per-file  `manifest.jsonl`, judging nothing
(2) **clean** (`stages/s1_clean/clean.py`) - produces the canonical output

Function
    Clean resamples every recording onto a **100 Hz** grid
    Keeps only the channels the device actually *measures* (30 cols)
        [1]         `Time`
        [2-28]      L/R/B inertial channels
        [29-30]     Two load cells 
    Normalizes each gyro channel to deg/s
    Records per-file trust in a `channel_trust.json` sidecar
    Recordings that cannot be cleaned are written to a `quarantine.jsonl` ledger

Notes    
    Raw file itself is never moved or altered
    Every run also emits
    `segments.jsonl`, `observations.jsonl`, and a `clean_report.md` summary. All tunable constants
    live in one file (`stages/s1_clean/config.py`) rather than being buried in logic.

Downstream of the deterministic core, an exception agent triages anything the checks flag (see `[A-1]`).

### [S1-1] Columns are resolved by name, never by position

Raw headers carry a positional prefix (ex. `47_loco`), but that number is a per-file position,
not a stable identifier (ex. @index 47 the column is `loco` in some files and `Step` in others; 
reading a col by position  (`df.iloc[:, 47]`) would blend loco state and step counter into one)

solution: stage strips the numeric prefix and resolves every col by name against a fixed role map,
so a shifted header is caught rather than absorbed.

### [S1-2] Keep measured channels, drop computed ones

**measured vs computed** decides what survives.
- *measured*: inertial channels (angle, gyro, accel) + load cells -> kept
- *computed*: Cadence, gait-cycle %, admittance, PID, `loco` -> dropped (firmware opinion, not observation)

2nd payoff: every col that shifts position between headers is a computed one, so dropping them
collapses 5 header shapes into 1 canonical schema and strips the firmware-version confound.

### [S1-3] Two rates, and why decimation must be anti-aliased

two rates:

1) ~100 Hz
2) 500 Hz

downsampling 500 Hz must NOT use naive slicing (`[::5]`).
why: naive decimation folds everything >50 Hz into the gait band as a full-amplitude fake signal
(ex. real 120 Hz -> fake 20 Hz, indistinguishable from motion).
solution: anti-aliasing FIR (`scipy.signal.decimate(..., ftype='fir')`) - removes it before it aliases.

### [S1-4] The "odd" sampling rates are clock quantization, not different devices

the odd rates (99.38 / 99.69 / 99.96 Hz) aren't different hardware: the device clock quantizes its
timestep to `10 + 2^-k` ms, which yields exactly those rates from the same 100 Hz devices.
treating them as distinct rates would invent device families that don't exist.
solution: group them within a tolerance and grid-correct to 100 Hz, rather than discard.

### [S1-5] Segment at gaps; never resample across one

gaps fall anywhere, so the unit of analysis is the **segment** (a gap-free run), not the file.
rule: a gap is any timestep above 5x the local median; split there and resample each segment on
its own. never interpolate across a gap - that invents data the device never measured. segments
under 1 s are recorded but not used.

### [S1-6] Gyro units and axes are normalized per file, by measurement

gyro is reliable (it's the derivative of its angle channel), but its units + axes are inconsistent
*within a single file*:

1) trunk gyro is rad/s, thigh gyros are deg/s
2) axes permuted: `d(Deg_Y)/dt` tracks `Gyro_Z`, not `Gyro_Y` (X->X, Y->Z, Z->Y, device-wide)

problem: mixing trunk + thigh gyro unnormalized is a silent 57.3x unit error.
solution: resolve per file by measurement - regress each angle-derivative vs every gyro axis, read
the axis + unit off the slope, rescale to deg/s.

NOT sagittality: which axis is the walking plane has no in-file signature -> fixed in S2, not guessed.
trust floor: per axis - a too-static axis abstains rather than vote noise (`Deg_Z` drifts, so its
derivative is noise even mid-walk).

### [S1-7] A broken time base is quarantined, not repaired

a degenerate time base (ex. the clock flatlines when the final timestamps tie) -> no rate can be
established.

solution: quarantine, don't salvage - a synthesized clock would silently corrupt every
rate-dependent measurement built on it.
handling: logged in the ledger, raw untouched, escalated to a human. the honest fix is a re-export
from source.

### [S1-8] `Hip_Deg` was cut, and the empty exception list is itself the finding

canonical means measured, with no exceptions. `Hip_Deg_L/R` (once kept as a bridge to an external
dataset) was cut when every premise for keeping it failed:

1) 0.991 correlated with the `Deg_Y` already kept
2) residual carried only the firmware's zeroing convention (what measured-only strips)
3) frozen at a constant on part of the corpus

the now-empty exception list is itself the finding, not an oversight.

---

## S2 - ML

### [S2-0] Current structure and function

S2 learns a locomotion classifier (a random forest) from the labeled trials and gates every change
on a measured metric. fully deterministic core; agents only judge.

pipeline (deterministic core):

- `dataset` - load labeled trials onto the canonical grid, group by rev (one subject, one day)
- `transform` - reconstruct the derived angle view from raw, w/ a standing verify guard
- `features` - window each recording
- `train` - leave-one-rev-out CV, refit a final model
- `locoeval` - blind scoring
- `taxonomy` - break errors into named types (dense inference)
- `experiment` - run a spec -> apply the promotion rule -> log it

champion: not a model file - a small `champion.json` + the commit it was made at (see `[S2-4]`).
agents: an experimenter (`[A-2]`) + a critic (`[A-3]`); neither can promote, only the metric gate can.
outputs: `champion.json`, the append-only `experiments.jsonl` ledger, `proposals.jsonl`,
`locoeval.md`, `taxonomy.json`, and `oof_champion.csv` (the fusion stage's input).

### [S2-1] Leave-one-rev-out, because subject leakage is the confound

trials are grouped by rev (one subject, one day); CV holds out a whole rev at a time.

why: a score then measures generalization to an unseen subject/session - the number that matters
     when a wearable meets a new wearer. a random split would put the same subject in train + test, let
     the model memorize its idiosyncrasies, and inflate the score.

note: the labeled pool is small (in development, 5 subjects, one dominant), so holding out by rev
is also the honest bound on how far the evidence reaches.

### [S2-2] The raw-to-derived transform is reconstructed exactly, and guarded

labeled trials are annotated against a derived view (a low-pass angle + its angular velocity), not
the raw channels. the pipeline reconstructs that view from raw exactly, using the same causal
first-order filter the device uses, w/ a standing guard (`verify_transform`) that re-derives it and
fails loudly on any drift.

why causal, not zero-phase: a zero-phase filter would shift features in time vs the labels, quietly
misaligning every windowed feature against the annotation it describes.

### [S2-3] A challenger is a declarative spec, never code

an agent that wants to change the model writes no code and touches no data. it proposes a
declarative spec from a fixed vocabulary: features to drop, window + stride, a whitelist of model
params. nothing outside the vocabulary is expressible -> reviewable before it runs, reproducible
after.

withheld on purpose: the random seed + CPU parallelism - reproducibility and machine resources are
the pipeline's to fix, not a proposal's to vary.

### [S2-4] The champion is its spec plus a commit, not a model blob

the forest is deterministic, so re-running a logged spec reproduces the same model. the champion's
identity is therefore its spec + the git commit it was made at, in `champion.json`.

no model blob, no git tag: either would be a second, drift-prone record that could disagree w/ the
spec.

payoff: the pipeline can re-seed the champion from scratch on a fresh checkout - re-run the spec,
get the model back.

### [S2-5] Promotion is metric-gated, with a fail-passive tiebreaker

the only path to champion is the promotion rule.

primary (macro-F1): a challenger must beat the champion by a margin, not merely exceed it - CV over
a handful of revs is noisy, and a razor-thin win would ratchet the champion on noise.

tiebreaker (on a statistical tie): promote if it lowers `steady_confusion` (a wrong call sustained
across a whole bout).

why: on a powered device a sustained wrong call is a sustained wrong action, while passive failures
(no assistance) are merely unhelpful - so at equal accuracy, prefer the model that fails passively.

every outcome is logged, promotions + rejections alike, so a rejected idea isn't re-proposed.

### [S2-6] The current champion drops a per-subject offset family

the champion (in development) drops the static-offset angle family: the between-leg angle offset +
the per-side angle means.

why: on the development corpus these encode a per-subject zeroing bias, not gait, so a forest
occasionally builds rules on them that fit one subject's stance and fail on a held-out one.
dropping them raised leave-one-rev-out macro-F1 to ~0.90 on that corpus.

note: the number is development evidence, not a guarantee for another corpus. what carries over is
the reasoning - drop features that encode subject identity, not motion.

### [S2-7] The row-level taxonomy needs dense inference

the headline metric is per-window, but the error taxonomy is defined in milliseconds (flickers,
late edges, short omissions). a model predicting once every ~2 s can't emit an error shorter than
its window -> those buckets would read zero by construction, not by merit.

solution: run the model densely (short stride), assign each prediction to the rows around its
center.

why center, not leading edge: leading-edge assignment would push every transition half a window
late and manufacture "late" errors at exactly the threshold being tested.

### [S2-8] Per-file calibration was built and rejected

a per-subject calibration (recenter each recording's angle on its resting posture, rescale its
amplitude) was built, then removed after it lost to the plain champion:

1) recentering is redundant once the offset family is dropped (`[S2-6]`)
2) amplitude rescaling regressed the score
3) no label-free scale recovered the loss

the forest already carries a balanced class weight, which supplies the per-class adjustment a fixed
threshold would. recorded so it isn't retried as if untried.

### [S2-9] Standing is under-recalled on unseen subjects - the real ceiling

the model's characteristic failure on held-out subjects: it under-recalls standing and over-calls
walking, and the errors sit in the calm middle of standing bouts, not at the boundaries.

diagnosis: a class-balance / decision-threshold behavior, not a missing feature. it points at the
data (better labels on motion-contaminated standing, more subjects) more than at anything left to
win in the model.

this is the honest ceiling the later stages work around rather than erase.

---

## S3 - Physics

### [S3-0] Current structure and function

S3 computes model-free physics features ("anchors") per window, audits each for rate-invariance,
renders a figure per trial, and ranks where the physics disagrees with the human label. its
centerpiece is the swap rule (`[S3-1]`).

deterministic core (`stages/s3_physics/`):

- `anchors.py` - per-window physics (the swap rule + the audited anchors)
- `rate_audit.py` - the rate-invariance test
- `plots.py` - one figure per trial (the agent's evidence)
- `run.py` - entry point

outputs: `anchors.csv`, `rate_audit.json`, a `disagreement.json` ranking, and `plots/`.

then a hypothesis agent (`[A-5]`) reads the figures under a provenance gate enforced in code.

note: the physics needs no training data -> an independent baseline the model must beat, not a copy.
because its figures feed an agent, S3 seals the held-out revs the same as training (`[S3-5]`).

### [S3-1] The swap rule: zero-parameter walk-versus-stand from leg alternation

walking is the legs alternating - not how far each leg swings, but whether they take turns.

rule: count how many times the between-leg angle difference commits past +1 deg then past -1 deg
within a window. 0 = standing, 1 = ambiguous (a single weight shift), >=2 = walking (a full stride).

zero fitted parameters:

1) the 1 deg is a sensor noise floor
2) the thresholds aren't tuned but forced - standing measures 0 and walking measures 2 across a
   wide amplitude range, and 1 is the only integer between them

needs no labels -> a genuine independent check on the learned classifier, not a restatement of it.

### [S3-2] Every anchor earns a rate-invariance verdict

an anchor is trustworthy only if it describes the body, not the sampling grid.

test: recompute it after decimating the window to half rate; see if it moves (absolute change for
bounded correlations/scores, relative change for magnitudes). run only on walking windows -
auditing a rhythm anchor on standing just measures decimation noise.

verdicts (development corpus): `periodicity` + `grav_stab` invariant; `antiphase` + `gyro_energy`
rate-dependent.

note: `gyro_energy` is defined the failing way on purpose (an unnormalized sum that halves when the
sample count halves), so the audit visibly re-derives a known failure. a rate-dependent anchor is
never used silently - the agent must acknowledge the verdict of anything it leans on.

### [S3-3] Rest-anchor centering, so a stance offset does not read as standing

the swap count is taken on the between-leg difference, but each subject's standing posture puts a
constant offset on it -> a large enough offset biases the count so real gait reads as standing.

fix: each recording begins at rest, giving a per-file zero measured on the same person + mounting
minutes earlier. subtract that rest zero before counting swaps.

result (development corpus): improved walking + standing recall together, not one traded for the other.

### [S3-4] A stride-adaptive window, so slow gait does not silently abstain

a fixed 2 s window holds fewer than 2 strides once cadence is slow, so the swap rule abstains -
exactly the regime of the assistive/rehab population the device exists for. physics that looks
solid on healthy lab cadence would quietly fail on the intended users.

fix: size the window to ~2 detected strides (from the local autocorrelation period, capped), w/ a
fallback that grows the span when the detector under-fires on slow/large strides. this makes the
swap count cadence-invariant by construction and carries the rule from lab to clinic cadence.

note: standing has no detectable period and keeps the short base window.

### [S3-5] The held-out set is sealed from any stage that feeds an agent, not only from training

sealing a held-out set from training isn't enough - it must also be hidden from anything that feeds
an agent whose output reaches a person.

so: S3 computes its anchors, plots, audit, and disagreement ranking on train + validation only; the
sealed recordings never appear in a figure.

why: an agent forming hypotheses off held-out plots would spend that set's independence just as
surely as training on it. the point of a sealed set is that no part of the system's judgment was
shaped by it until the very end.

---

## S4 - Fusion

### [S4-0] Current structure and function

S4 combines the S2 learned label + the S3 physics verdict into one call + a confidence - the signal
the existing device algorithm lacks, and the reason the project exists.

deterministic core (`stages/s4_fusion/`, `fuse.py` + `run.py`): join the model's out-of-fold
predictions w/ the S3 anchors, apply a zero-parameter fusion policy, calibrate the confidence tiers,
rank the disagreements.

outputs: `fused_windows.csv`, `fusion.json` (metrics), `fusion_report.md`, `disagreements.json`.

built on that core:

- a judgement agent (`[A-6]`) characterizes the disagreement cases
- a curation director (`curate.py`) turns the confidence signal into data-collection actions (`[S4-4]`)
- governed discovery (`newclass.py`, `[A-7]`) proposes classes the two-label taxonomy misses

the one honest test on the sealed set is kept behind an explicit confirmation gate (`[S4-5]`).

### [S4-1] Late fusion, to keep the confidence signal a controller can act on

the physics view can reach the label two ways:

1) early fusion - as one more feature inside the model
2) late fusion - as an adjudicator applied after the model

the pipeline uses late fusion: the physics verdict adjudicates the model's call through a
zero-parameter policy and, in doing so, produces a confidence tier the model alone doesn't give.

why: it preserves the swap rule's interpretable, parameter-free guarantee and yields a HIGH/MED/LOW
signal a controller can act on. that confidence output is the point of the whole project - the
algorithm this improves on can't say when it's unsure.

### [S4-2] The fusion policy is read off the measured agreement, not assumed

the rule combining the two views is set by their measured cross-tabulation, not intuition.

- agree -> HIGH (about 98% correct on the development corpus)
- disagree -> LOW, and the system abstains (either view's own label is ~a coin flip there)

rejected (both tested, both lost to plain agreement):

1) let physics override the model whenever it says standing
2) trust whichever side reports higher confidence

so the tier is set by whether the two views agree, never by whose confidence is louder.

### [S4-3] The tiers are calibrated, and LOW is a deliberate abstention

the tiers are calibrated so accuracy rises monotonically LOW -> MED -> HIGH, and LOW is an
abstention - a machine "I cannot call this," mirroring the human "looked and could not call it"
already in the corpus.

use: a safety-critical controller acts on HIGH + MED, holds on LOW.

result (development corpus): acting only on the confident tiers covered most of the data at markedly
higher accuracy than the model at full coverage. abstaining exactly where the two independent views
disagree is the safe move - those windows genuinely sit between the two states.

### [S4-4] The curation director, and the two-vote relabel gate

the same confidence signal becomes a data-collection director: each flagged window is routed to
relabel, characterize-as-new-class, or collect-more-of-a-hard-condition.

two-vote relabel gate: both the model AND the physics must contradict the human label before a
window is a relabel candidate - a single view disagreeing is usually that view's own error, not a
wrong label.

note: code never edits a label; it only nominates candidates for a human to confirm. this is the
pipeline directing the data effort - the highest-leverage move once the ceiling is data, not the
model (`[S2-9]`).

### [S4-5] The sealed set is opened once, and it changed what ships

the held-out set is opened exactly once, at the very end, behind an explicit confirmation gate - and
its result decided the deployable artifact.

finding (development corpus): the raw fused label did NOT beat the model out-of-sample - the physics
veto flipped a few true standing windows to walking, so the fusion's training-time advantage didn't
replicate. but the confidence signal DID generalize - acting on the confident tiers stayed above the
model, w/ a well-behaved low tier.

so: the deployable is the confidence-gated system (act on HIGH/MED, abstain on LOW), not the raw relabel.

single-use by design: once opened, the set is spent - no number can honestly be optimized toward it after.

### [S4-6] New-class discovery is allowed, but governed

the pipeline can propose structure the two-label taxonomy misses - but only under governance, never
as an automatic change.

flow: from the windows the curation director marks as possible new classes, a deterministic core
assembles a physics-profile bundle and an agent proposes candidate classes.

code then validates each proposal on two axes:

1) provenance - it must point at real windows
2) mass - it must recur across enough material + enough subjects to be more than an anecdote

every survivor routes to human adjudication regardless of the agent's confidence. it never adds a
class and never edits a label - read-only, orthogonal to the deployed classifier: discovery that
surfaces work for a human, not a taxonomy the system rewrites for itself.

---

## Cross-cutting principles

### [X-1] Code does the work; agents judge the work

every number is produced by deterministic code, and every change to data is made by deterministic
code. agents never read raw values to compute a result and never mutate anything - they judge, at
defined points, material the code already prepared.

payoff: the pipeline stays auditable - a result reproduces by re-running the code, and an agent's
contribution is always a recorded judgment, never a hidden calculation.

### [X-2] The filesystem is the only interface between stages

stages communicate only through artifacts on disk - no agent-to-agent messaging, no shared
in-memory state. each stage reads the previous stage's files and writes its own.

payoff: every intermediate is inspectable, a run can stop and resume at a stage boundary, and the
orchestrator stays a simple sequencer with no intelligence of its own.

### [X-3] No silent mutation, and needs_human is a deliverable

nothing is dropped quietly:

- a file that fails a check goes to a quarantine ledger, raw left untouched
- an agent's decision carries a written rationale
- low confidence escalates to an explicit needs_human flag

reaching a point where the machine can't honestly proceed without a person is a valuable, explicit
output - the pipeline surfaces where human judgment is required rather than papering over it with a guess.

### [X-4] Assert nothing; measure, then state

a rule enters the pipeline only when a test upheld it, and a rejected idea is logged so it isn't
retried as if untried. a plausible explanation is never promoted into the permanent record on the
strength of a good metric - the measurement licenses the change, not the story.

why it matters: several times a change improved the score while its stated reason turned out wrong.
keep what measurement supports, and hold explanations to the same standard as the numbers.

### [X-5] Institutional memory is corpus-agnostic

DOMAIN_NOTES is injected into every agent, and this is a data-pipeline tool: the dataset fed into it
can change. So the notes separate what generalizes from what does not. Durable and kept as instruction:
the labeling conventions, the methodology and its failure modes, the design decisions, and the reasoning
behind each finding. Corpus-bound and framed as *reference-corpus observations* rather than invariants:
counts, dates, per-file measurements and per-split scores - on new data these must be re-measured, and
the live values regenerate every run in the `runs/*` artifacts. Corpus-specific identifiers are
anonymized: labeled-export units carry role-mapped aliases (`revA`, `revB`, ... rather than their
concrete names), example filenames become descriptive phrases, and subject indices become roles ("the
dominant subject"). Named individuals and vendor/brand names are kept out of the permanent record (a
decision is attributed to its role, for example "the data owner", not a person). The tradeoff accepted:
the anonymized labels no longer string-match the code and run artifacts, so readers cross-reference by
role, not index.

why it matters: an agent reading the notes as universal fact would carry one dataset's specifics onto
data where they do not hold. Framing the specifics as reference-corpus findings keeps the reasoning
reusable while flagging the numbers as illustrations to re-verify.

---

## Agents

### [A-0] Scope and limit

the agents share one small framework (`agents/base.py`, `run_agent`). each agent is a fixed spec -
a role prompt, an allowed-tool list, a model, and a turn cap - into which the shared institutional
notes are injected on every call, so no agent runs without the corpus's hard-won context. every
tool call is logged and each run's cost is recorded.

two limits hold for all of them:

1) effectively read-only - the only tools any agent gets are Read + Grep, so none can change
   data, promote a model, or edit a label. a deterministic wrapper parses the output and
   enforces the real gate in code.
2) bounded - a turn cap makes a stuck agent fail fast rather than spin.

most agents run on a sonnet-class model; the one cheap tagging job runs on haiku-class.

### [A-1] s1_exception - clean-stage exception triage

reads the S1 quarantine + observation ledgers and answers two orthogonal questions per flagged item:
does a known finding explain it, and does it need a person. code collapses that pair into
`known_expected` / `novel` / `needs_human`, so the disposition is the pipeline's to define, not
re-decided each run. read-only; the wrapper writes the review.

(haiku-class, Read + Grep, cap 15.)

### [A-2] s2_experimenter - proposes one experiment

proposes a single declarative `ExperimentSpec` per cycle - features to drop, window + stride,
whitelisted params - grounded in the notes and the ledger. writes no code, never trains, never
touches data; it only raises a proposal the deterministic core can run and reproduce.

(sonnet-class, Read + Grep, cap 15.)

### [A-3] s2_critic - reviews before a training run is spent

reviews a proposal *before* any training run is spent, with the full ledger in view, and returns
approve or revise. it can't promote anything - it only filters proposals ahead of the metric gate.

why the ledger: a re-proposed idea sounds just as reasonable the second time, and only the ledger
reveals it was already tried and refused.

(sonnet-class, Read + Grep, cap 10.)

### [A-4] s2_critic_probe - a verification harness for the critic

not a pipeline agent but a test of one. it feeds the live critic proposals it must reject - a
verbatim ledger repeat, a cosmetic rename of the same change, and a claim on a false premise - and
checks that it does. it exists so "the critic actually bites" is demonstrated on demand, not merely
asserted.

(reuses the critic's spec.)

### [A-5] s3_physics - reads the figures, proposes rules

reads the S3 figures directly (the model sees the plots) and proposes hypotheses as structured
output, each tied to the specific window it rests on.

gate (in code): a hypothesis is kept only if it names a real window, and every anchor it leans on is
annotated with its rate-invariance verdict - so a claim resting on a rate-dependent anchor can't
pass unflagged. its job is rule discovery, not per-window labeling.

(sonnet-class, Read + Grep, cap 30.)

### [A-6] s4_fusion - judges the disagreement cases

characterizes each low-confidence (disagreement) case from the fuser as a label problem, a model
error the physics caught, a physics error, or genuine ambiguity - always with window-level
provenance - and flags likely label problems for review. read-only: it judges what the abstentions
are made of, never re-labels or changes the fusion.

(sonnet-class, Read + Grep, cap 30.)

### [A-7] s4_newclass - proposes classes the taxonomy misses, under governance

proposes classes the two-label taxonomy misses, from the spans the curation director marked.
governed: code validates that each proposal points at real spans and recurs with enough mass across
enough subjects, and routes every survivor to human adjudication regardless of the agent's
confidence. it never adds a class and never edits a label - discovery that hands work to a person,
not an automatic taxonomy change.

(sonnet-class, Read + Grep, cap 30.)

---

## Appendix A - Environment and tooling gotchas

These are development/runtime-environment issues for whoever *runs* the pipeline, not knowledge an
inference-time agent needs. They were moved here out of the agent-injected DOMAIN_NOTES. `Section` refs
below point at DOMAIN_NOTES sections.

- **NumPy 2.0:** the `.ptp()` ndarray method was removed. Use `np.ptp(array, axis=...)`.
- **Windows command-line limit - this PREDICTION CAME TRUE [measured, 2026-07-20].** Every agent
  began failing with `CLINotFoundError: Claude Code not found at: ...\_bundled\claude.exe` - pointing
  at a binary that was present, 253 MB, and ran fine standalone. The message is a guess: the SDK
  catches any `FileNotFoundError` during spawn and blames the CLI. The real cause was **WinError 206
  `ERROR_FILENAME_EXCED_RANGE`** - on `CreateProcess` that means *the command line is too long*, not
  the path.
  Cause: `run_agent` injects all of DOMAIN_NOTES into the system prompt, and the SDK spent it on
  argv (`--system-prompt <text>`). This file grew **30,029 -> 54,551 chars in one day (+82%)**,
  putting the system prompt at **56,765** against a ~32,767 `CreateProcess` cap. It worked in the
  morning and not by evening; nothing in our code changed, only the size of this file.
  **Fix:** `run_agent` writes the prompt to `runs/<run>/system_prompt.txt` and passes
  `system_prompt={"type": "file", "path": ...}`, which the SDK forwards as `--system-prompt-file`.
  A path is O(1) on the command line, so institutional memory can now grow without a ceiling -
  and the file doubles as an audit record of exactly what each agent was told.
  **Diagnostic lesson:** an SDK error naming a missing file may be masking any spawn failure.
  Unwrap `__cause__` before believing it - the stated path here was correct and healthy.
- **Injecting all of DOMAIN_NOTES into every agent has a soft cost too [measured, 2026-07-20].**
  On the same run, the cheap (haiku-class) exception agent returned 22 verdicts for 23 queue items
  and stopped populating the `section` field (still citing sections in prose), while cost doubled
  $0.052 -> $0.111. Nothing else changed but prompt size. The deterministic wrapper caught the
  missing verdict and marked it `needs_human` - no silent drop - but the trend is clear: the
  whole-file injection does not scale indefinitely. When it bites again, inject the relevant
  sections per agent rather than the entire file.
- PowerShell 5.1 does not accept `&&` as a statement separator.
- **cp949 is the default encoding on this machine, and it breaks BOTH directions on agent text
  [measured, 2026-07-20 - hit three times in one day].** This box runs a Korean-locale Windows
  console, so Python's default encoding is cp949, not UTF-8. Agent-authored text (proposal
  rationales, critic reviews, decision reasons) routinely contains em-dashes and `sec `, none of which
  cp949 can represent.
  - **Writing/printing -> `UnicodeEncodeError`.** A run died on a `print` *after* the API call was
    already billed. `orchestrator.py` now reconfigures stdout/stderr with `errors="replace"`.
  - **Reading -> `UnicodeDecodeError`.** `json.load(open(path))` on `experiments.jsonl` fails on the
    first non-cp949 byte. Every read of a pipeline artifact must pass `encoding="utf-8"` explicitly;
    the default is not UTF-8 here and never will be.

  **[open] - the fix is incomplete.** The `orchestrator.py` guard covers *pipeline runs only*.
  Ad-hoc inspection one-liners have no such protection and this bug caught one within an hour of
  being documented, reading a champion spec that contained an em-dash. Rules for anything touching
  a JSON/JSONL artifact, script or pipeline:
  1. read with `io.open(p, encoding="utf-8")` - never bare `open()`;
  2. write with `encoding="utf-8"` and `ensure_ascii=False`;
  3. for throwaway scripts that print artifact text, set `PYTHONIOENCODING=utf-8`.

  Worth fixing properly: a tiny `read_jsonl` / `read_json` helper that hard-codes the encoding, so
  the correct call is the shortest one to type. Every occurrence so far has been someone (including
  Claude) reaching for the stdlib default under time pressure.
- **SDK stream buffer - the third stdio trap [measured, 2026-07-22].** The first live S3 run
  (`s3_physics`) reached the agent, then died: `Failed to decode JSON: JSON message exceeded maximum
  buffer size of 1048576 bytes. The SDK transport (`subprocess_cli.py`) reads the CLI's stdout as
  newline-delimited JSON with a **1 MB per-message buffer** (`_DEFAULT_MAX_BUFFER_SIZE`). S3 is the
  first agent to Read PNGs, and the CLI streams a `Read`-of-image back **inline as base64** in one
  message; a turn that batches several figures (parallel Read) past 1 MB is a fatal decode mid-run.
  **Fix:** `run_agent` sets `ClaudeAgentOptions(max_buffer_size=32 MB)` - a supported option, not a
  monkey-patch. Chose to raise the buffer rather than shrink the plots: the agent needs the figure
  detail (swap strips, anchor timelines) to do its job. Same family as the two traps above - a stdio
  limit that only bites once the payload (prompt size, then image size) crosses it.

---

## Appendix B - DOMAIN_NOTES changelog

The dated development timeline, moved out of the agent-injected DOMAIN_NOTES (an agent deciding now
does not need it). `Section` refs point at DOMAIN_NOTES sections; rev labels are the anonymized
aliases (`[X-5]`).

| Date | Phase | Added |
|---|---|---|
| 2026-07-16 | 0 | v1 seeded: channel trust, rate confound, label semantics, eval rules, NumPy gotcha |
| 2026-07-16 | 1 | v2 from the real corpus: 5 variants / 45-col contract / position-is-a-lie; two rate eras; quantization tiers; anti-aliasing proof; segments + startup burst; -1 vs 255; revA as lossy family; provenance tags |
| 2026-07-16 | 1 | v3 merging the physics/rule-discovery track. **RETRACTED Section 4.1** (angvel is reliable, r=0.999 - the noise claim was never verified and the "confirming" measurement was taken over a 93.7%-walking file). **NEW:** Section 4.1b gyro units inconsistent within a file (B=rad/s, L/R=deg/s; Y<->Z swap); Section 5.5 session != subject, subject unknown for 88/95 (needs_human); Section 10 the swap rule + validated descriptors + richer states; Section 11 methodology warnings, broken tests, retractions. **UPGRADED:** Section 4.5 loco severed with evidence; Section 6.2 raw->revA mapping largely resolved; Section 7 select-by-time, per-file calibration, abstain-don't-force; Section 9 window length now an open tradeoff, gait band 0.13 Hz not 0.5-3.0 |
| 2026-07-20 | 1 | Corpus refreshed (more raw files + new subjects; trial-batch markers removed). Section 2.6 NEW: a 2026-05 batch has a degenerate time base (non-monotonic, duplicated timestamps, median dt=0) - quarantined, the pipeline's first real quarantine; `measure_hz`/`clean_one` now guard `median(dt)>0`. **Records made count-free** (per the data owner): live tallies live in the run artifacts, not this file. Gate still holds (partition asserted). |
| 2026-07-20 | 1 | S1 hardening. **MEASURED:** Section 4.1b confirmed on all 88 files (gyro units normalized to deg/s in the clean layer, per-file `channel_trust.json`, detect-don't-assert with static-file abstention); Section 4.2 yaw is `Deg_Z`, drift is file-specific (13 chans / 11 files), flagged not dropped; Section 7 `resample.py` proven time-safe by drift test. **REVERSED Section 5.5:** filename field 2 IS the subject (the dominant subject recurs across ~5 months) - subject known for every file, `needs_human` RESOLVED (data owner confirmed the trial-batch markers are trial batches, not separate people). **DESIGN:** quarantine is a `quarantine.jsonl` ledger, raw file never moved. Docs reconciled (angvel "untrusted"->reliable Section 6.1; 30-vs-32 cols; PLAN/README status). |
| 2026-07-20 | 2 | Phase 2: S1 exception agent (`agents/s1_exception.py`). Read-only agent triages the clean stage's genuine exceptions (quarantines + gyro axis anomalies + yaw-drift flags; routine abstentions summarized, not triaged) into `known_expected` / `novel` / `needs_human`, each grounded in a DOMAIN_NOTES section; deterministic wrapper writes `exceptions_review.jsonl`. Degenerate-time-base quarantines -> `needs_human` (re-export); handled anomalies/drift -> `known_expected`. Deleted the Phase-0 smoke agent; fixed the shipped-but-uncalled `load_dotenv()`. |
| 2026-07-20 | 2->3 | Phase 2 gate CLOSED: the data owner signed off on the exception review (7 `needs_human` = Section 2.6 broken-clock batch -> re-export; 16 `known_expected`; 0 novel). Dead code removed (`Resolution.has`, unreachable `legacy_algo` interp-role). Count-free sweep extended past the prose into code comments, the generated `clean_report.md`, and surviving inventory counts (schema-variant / subject / startup-burst tallies); named-file example stats and this changelog keep their numbers. Phase 3 (S2 loop) starting. |
| 2026-07-20 | 3 | Provenance audit (4 flags raised on the 33-column canonical set, all checked against the corpus). **CORRECTED Section 4.1b:** the Deg<->Gyro Y<->Z crossing is **device-wide, not a trunk defect** - `d(Deg_Y)/dt`->`Gyro_Z` unanimously on L (62 files) and R (65), and 45/47 on B; `L_Gyro_Y` is no more sagittal than `B_Gyro_Y` (median \|r\| 0.16 vs 0.99). The earlier trunk-only framing would have sent a fix to one side of a three-side convention. Post-clean slopes 0.987/0.984/0.986 confirm the unit fix landed. **Section 4.3 narrowed** to placement risk only. **NEW Section 4.6:** `Deg` is on-sensor *fusion*, not a transducer reading - three-tier provenance (transducer / on-sensor fusion / app-layer compute); explains Section 4.2 yaw drift mechanistically and demotes Section 4.1's r=0.999 from corroboration to near-tautology. **CUT `Hip_Deg_L/R`** - `KEEP_EXCEPTIONS` is now empty: 0.991 redundant with `Deg_Y`, residual = firmware zeroing convention, zero-variance on 12/180 (file,side) pairs (twice frozen nonzero), and the open dataset it bridged to is not in the repo (Section 5.7) - nothing read it. **Section 9 count reconciliation:** 30 / 32 / 33 all correct, different questions; raw-side clean output is now **31** (30 + `segment`). |
| 2026-07-20 | 3 | **"Recorded" != "handled" - audited both places this file claimed handling it did not have.** Section 4.1b's "resolve it from `channel_trust.json`" and Section 4.2's "must consult the per-file drift flag" both read as descriptions of pipeline behaviour; **neither had a consumer.** `transform.py` resolved axes from `SAGITTAL_DEG_AXIS_BY_VARIANT` + `DOCUMENTED_GYRO_PERMUTATION` and never opened the per-file record, so a file whose *permutation* breaks on its sagittal axis would have been read on the documented column silently - inert to date only because both known anomalies are B-side and the feature path reads L/R. **FIXED:** `transform.check_axis_trust` hard-fails such a file (same idiom as `UnknownVariantError` - refuse, never guess); `trust` is now a **required** argument on `raw_to_features` with an explicit `TRUST_UNCHECKED` opt-out, so skipping the check is a decision at the call site. Bridge still exact: 19/19 pairs at 7.4e-13. Drift flag left **unenforced and now labelled so** - no yaw feature exists to exclude. **S1 exception agent reworked** for the same root cause: `known_expected`/`novel`/`needs_human` collided on two different axes with no precedence, so the agent now answers two orthogonal questions - `explained` (yes/no/**contradicts**) and `action` (none/human) - and `collapse()` derives the disposition deterministically, so the taxonomy is the pipeline's and not re-decided per run. A contradicted note lands in `novel` (it is a find) with `action` forced to human (never acted on), which was the case the old contradiction rule buried. Queue items now carry machine-derived `handled {value, why}` computed from the real consumers, plus the `conflicts_with_documented` field that was the missing grievance; `confidence` pinned to the disposition, not the cause. Same partition on the current corpus (7 + 2 + 14). |
| 2026-07-20 | 3 | **S2 champion/challenger loop live.** Declarative `ExperimentSpec` vocabulary + whitelisted, bounds-checked hyperparameters (Section 9); `experiment.decide()` is the only path to champion - `PROMOTION_MARGIN = 0.005` primary, `steady_confusion` fail-passive tiebreaker on a tie (Section 7). Experimenter proposes -> critic reviews **before** training, seeing the **ledger** and not just the proposal (a re-proposed idea always sounds as reasonable the second time); ideas killed pre-training go to `proposals.jsonl`, measured ones to `experiments.jsonl`. Unparseable critic review => `revise`, never `approve`. **MEASURED:** the incumbent `loco` algorithm profiled through our own taxonomy - its dominant bucket is `steady_confusion` (0.519) like ours, not `swallowed`; its distinguishing failure is `edge_omission` 0.411, so the "fails complementarily" recollection is directionally right and mechanistically wrong (Section 7). **Confound found in our own machinery:** `stride_s` was tied to `window_s`, halving the training set whenever the window changed (Section 9). Ledger to date (7 entries, 2 promotions): `baseline_v1` 0.8730 promoted as seed; `drop_absolute_angle` 0.8758, `wider_window_4s` 0.8150, `window_4s_stride_2s` 0.8188, `drop_bad_band_frac` 0.8711, `regularize_min_samples_leaf10` 0.8181 rejected; **`drop_offset_only` 0.8862 promoted (+0.0132) - the first agent-proposed champion**, dropping `ang_LR_offset` on the Section 10.1 grounds that `interleg_offset` measures AUC 0.50 for walk/stand. **Caveat [measured]:** its rationale predicted a `steady_confusion` reduction and that bucket moved only 0.888 -> 0.884, so the feature was dead weight but not for the reason given - the same self-refuting pattern as `drop_absolute_angle`. A defensible metric does not validate the mechanism claimed for it. ~$0.21/cycle. |
| 2026-07-21 | 3 | **RESOLVED the sagittal-axis question and DELETED revI (Section 6.3/Section 6.4).** The data provider confirmed every export uses the sagittal/**Y** plane; the per-variant "convention" was a raw-axis naming inconsistency, and `csv2mat.m` proves the MATLAB selects no axis (it LPFs all three Deg axes and saves all three - selection is a downstream step not in the repo). **Implemented always-Y:** `transform.raw_to_features` always reads `Deg_Y`+`Gyro_Z`; `SAGITTAL_DEG_AXIS_BY_VARIANT` deleted (and with it the "unknown variant => break" failure); `verify_transform` rewritten to verify Y-reproduction and flag revH as an expected `OVERRIDE->Y` (11 Y-pairs exact to 7e-13, 0 broken); `s1_exception` candidate-axis set narrowed to `{Y}`. Signal-only recovery of the exporter's *label* was re-confirmed impossible six ways (amplitude 7.7%, antiphase 24-40%, energy 11-28%, gravity identical across variants - they differ by rotation *about* vertical) - but moot, since we fix the axis rather than detect it. **Validated Y is better even for the lone X-exporter revH** (spent from the lockbox for this test): under the Y-trained model revH-from-Y scores acc 0.948 / macro-F1 0.764 vs X's 0.913 / 0.708, because `Deg_Y` carries 3-4x the gait swing (gait-RMS 1080 vs 298, L) AND is the axis training used; revH's labeled cols being corrected to Y at source; `revG` stays sealed. **revI DELETED** (`dataset.EXCLUDED_REVS`): unverifiable (no raw, `fb5ea2c2` assignment never measured, gait 2-4x low), and removing it raised champion LORO macro-F1 0.8862->0.8932 / acc 0.9334->0.9381. Supersedes the earlier same-day "per-rev calibration / rebuild blocked / revI not an axis bug" framing - the always-Y fix needed neither a rebuild nor calibration, and the revI "not an axis bug" claim was circular (its variant was never measured). |
| 2026-07-21 | 3 | **RETIRED header-variant tracking (code + docs), per the data owner - it drove no decision.** Once S1 drops computed columns every header shape collapses into one canonical schema (Section 9) and columns are read by name (Section 1.3), so the variant fingerprints (`fb5ea2c2` etc.) were reporting-only - nothing branched on them. Removed `Variant`/`fingerprint`/`build_registry`/`stable_prefix` and the `variant_id` field from `census.py`; dropped `variants.json` and the variant tables from `run.py`'s `census.md` (now a name-based schema report: files, families, unregistered-column alarm); dropped `variant_id` from `manifest.py` and `verify_transform.py`. Reworded stale comments in `config.py`/`clean.py`/`channel_trust.py`/`transform.py`/`s1_exception.py` (one referenced the already-deleted `SAGITTAL_AXIS_BY_VARIANT`). Docs: Section 1.1 rewritten as "one canonical schema"; hash IDs and the per-variant tables in Section 2.1/Section 4.1b/Section 6.2 removed, load-bearing findings (resolve-by-name, 45-col contract, device-wide permutation, always-Y) kept. No data-path change: `verify_transform` still exact, S1 gate unchanged. |
| 2026-07-21 | 3 | **Phase-3 gate: two blockers closed, one denominator reconciled.** (1) **Critic bites** - `agents/s2_critic_probe.py` fed the live critic 3 proposals it must not approve (a verbatim ledger repeat, a cosmetic `window_s=4` repeat, and a false `GAIT_BAND_HZ` premise); all 3 rejected, the critic verifying the premise against `features.py` and citing ledger entries by name ($0.23). (2) **Reconstructibility demonstrated** - `stages/s2_ml/replay.py --name drop_angvel_dom_hz` replays the at-HEAD champion **EXACT** (<=1e-9), so a same-sha ledger entry is a faithful revert unit. (3) **Incumbent reconciled** (Section 7): the naive `0.519`-vs-`0.888` `steady_confusion` gap was cross-recording (7/8 `loco` pairs are the sealed revH lockbox); on the one shared non-lockbox recording, same rows, ours is **LOWER** (0.016 vs 0.086) - directional, pending lockbox. Current champion `drop_angvel_dom_hz` macro-F1 0.8886. |
| 2026-07-21 | 3 | **Per-file calibration MEASURED and SCRAPPED as an S2 lever (Section 7).** Built per-file posture (recentre `ang` on the at-rest opening, Section 10.2) + amplitude (rescale `angvel` by a robust per-file scale) calibration onto the corpus-global reference, gate-wired behind a default-off `ExperimentSpec.calibrate`. Against champion `drop_static_offset_family` (0.8977): **posture is a no-op** (+0.0000 - the champion already drops the five posture features; keep-and-calibrate 0.8904 loses to drop 0.8977) and **amplitude regresses** (-0.026) with no label-free scale rescuing it (MAD/p75/p90/p95/p99 monotonic toward but never past baseline, best -0.013). The leave-one-rev-out RandomForest already handles the per-file gain a fixed threshold couldn't, so Section 7's revG `0->1.000` is threshold-only and does not transfer. **Code reverted** - `stages/s2_ml/calibrate.py` removed, `experiment.py`/`replay.py` wiring backed out, the orphaned `calibrate_posture_amplitude` ledger entry dropped (kept `replay.py` exact-replay green); finding preserved in Section 7. Global-features version stands. |
| 2026-07-21 | 4 | **S3 physics stage built - Phase 4 deterministic core (Section 10.4 NEW).** Ported the swap rule (Section 10) + `ileg_minhalf`/`interleg_offset` (Section 10.1) + five anchors into `stages/s3_physics/` on the canonical 2 s windows; faithful to Section 10 on non-lockbox files (revA_t1 walk-rec 0.994, revA_t3 0.931). **Rate-invariance audit** (`rate_audit.py`; 100->50 Hz, walking-gated; absolute delta for bounded anchors, relative for ratio-scale): `periodicity`/`grav_stab`/`gait_hz` **invariant**; `antiphase` and `gyro_energy` **rate_dependent** - `gyro_energy` re-derives the Section 2.3 rate-experiment failure, defined as the summed-square that fails on purpose so the audit demonstrates itself. Per-trial plots + a physics-vs-label disagreement ranking feed the hypothesis agent (`agents/s3_physics.py`, Read/Grep); code enforces the provenance gate (no window-level evidence => rejected) and attaches every relied-on anchor's verdict. Wired as `orchestrator.py s3_physics`; `features.iter_windows` refactored out as the single-sourced window iterator. **Lockbox sealed in the stage** (`load_analysis_trials`, train+val only) after it was first - wrongly - computed over revG/revH; mistake + fix in tracker POSTDAY4 Section 3. **Stride-adaptive window** flagged product-critical (Section 10.4). Agent not yet run live. |
| 2026-07-22 | 4 | **`gait_hz` anchor removed end-to-end (Section 10.8).** Sanity check of the S3 physics core found `gait_hz` (the fifth audited anchor) already dropped from `ANCHOR_NAMES`/`window_anchors` but still referenced downstream, which would crash a fresh run at plot time. Completed the removal: the `plots.py` panel-3 cadence axis, the dead `rate_audit.ANCHOR_METRIC`/`ANCHOR_FLOOR` keys, and the `agents/s3_physics.py` prompt example; reconciled the docs (`DOMAIN_NOTES` four-anchor table + Section 10.6/Section 12.3 refs, `PLAN.md`, `USE.md`, `README.md`) to four anchors. Verified: modules compile, the anchor table carries no `gait_hz`, every audited anchor still has a metric, a synthetic trial renders its plot without error. Rationale for the drop itself is Section 10.8 (degenerate at the 2 s window, walk-vs-stand AUC 0.487, a false-pass audit). Separately swept em dashes out of all project Python source to stop cp949 console-encoding errors. |
| 2026-07-22 | 5 | **S4 fusion stage built + agent run live (Section 12, Section 12.1 NEW).** Restructured S4 from a read-only report to a **fusion**: `stages/s4_fusion/` (`fuse`, `run`) + `stages/s2_ml/oof.py` (champion out-of-fold predictions as a joinable artifact). Policy read off the measured S2xS3 contingency, not assumed - agreement 98% correct -> HIGH; S3-WALKING vetoes toward WALK, S3-STANDING ignored against S2; two intuitive rules measured and **rejected** (physics-overrides-standing scored *below* S2; higher-confidence-wins fails because S2 is confident-wrong in disagreement). Fused macro-F1 **0.921** (S2 0.898); acting on HIGH+MED covers **94%** at accuracy **0.969**, stand-recall **0.897** - standing recovered by *abstention*, not a cleverer label; LOW/abstain = a machine `-1` (Section 5.2). Judgement agent (`agents/s4_fusion.py`, Read/Grep, provenance-gated) run live (`runs/2026-07-22_run3`, 6/6 findings, $1.41): validated the fuser, flagged a label problem, and claimed a brief-stop weakness whose *mechanism it got wrong* - verified as a window-resolution floor, not adaptive-window bridging (Section 12.1). `s4_fusion` step wired. **Row-level: `steady_confusion` 0.823 -> 0.785, row-acc -> 0.942 - helps the hazardous bucket but does not solve it; the residual is label contamination (Section 12.2).** Built `curate.py` - the confidence signal as a **data-collection director**: routes flagged windows to `relabel_candidate` (error_rate 1.00, a precise mislabel detector) / `new_class_candidate` / `collect_more`. New-class idea evaluated: taxonomy is too coarse (18% non-quiet standing) but a static "sit" is absent - governed proposal loop required (Section 11.2/Section 12.2). Evaluated train+val; lockbox sealed. |
| 2026-07-22 | 5 | **Anchor early-fusion MEASURED and NOT promoted (Section 12.3 NEW).** Tested pulling the rate-invariant, not-already-present S3 anchors (`grav_stab`, `periodicity`) directly into `features.window_features` (early fusion) vs leaving them in the S4 swap policy (late fusion, Section 12). Measured in champion config (`drop_static_offset_family` 0.8977), LORO, with a control run reproducing the champion **exactly** (0.8977/18-feat, harness clean): `grav_stab` **+0.0006** (below the 0.005 margin, and 0.72-collinear with `L/R_ang_LPF_ptp`/`std` already present), `periodicity` **-0.0056** (regresses), both **-0.0010** (`steady_confusion` 0.823->0.838 worse). **None clears the gate - not promoted, `features.py` reverted, champion unchanged.** `grav_stab` is a monotone transform of `std` features the tree already splits on and already earns its keep in the S4 abstain tier; early fusion relocates signal, does not add it. Section 11.4 in the other direction - the gate correctly refused a good-sounding change. New op-doc `USE.md` documents the anchor-promotion procedure + the two eligibility gates (rate-invariance, non-redundancy). |
| 2026-07-22 | 5 | **Rest-quality guard added to the swap-rule zero (Section 10.5.1, Section 10.2, Section 12.4 NEW).** `rest_offset()` trusted the opening 3 s was rest on faith; measured it is **not** - 6 of 8 revD files open *mid-gait* (36-44deg swing, 4-5 swaps), unique among revs, so the "recordings begin at rest" label (Section 10.2) is not universal. `anchors.rest_anchor()` now verifies: trust the opening span only if the swap rule calls it STANDING (0 swaps centered) -> else the stillest true-rest span anywhere -> else whole-file median flagged `rest_offset_trusted=False`, surfaced on every anchor row + the per-trial plot (`[!] rest zero untrusted`). A *single* 3 s swing window is a poisoned zero (per-window median wanders **+/-4-5deg**, spread to 20deg); the whole-file median is stable (averages strides). **Correctness fix with a measured-null metric impact:** corpus walk-recall 0.691->0.692, stand 0.927->0.925; revD bit-identical - the offset trap only bites when offset ~= swing amplitude (revA_t6/t7), and revD's swings are large. Value is latent insurance + honesty. S4 does **not** consume the flag by decision (Section 12.4). Label-free, lockbox untouched (Section 10.4). Runs alongside the Section 10.7 span-grow work, both uncommitted. |
| 2026-07-22 | 5 | **LOCKBOX OPENED - final number, revG spent (Section 12.5 NEW).** Built `stages/s4_fusion/lockbox.py` (guarded one-way open, `--confirm OPEN-LOCKBOX`; scores the FUSED model - S2 fit on every non-lockbox rev predicting revG unseen, S3 verdicts on revG, fuse) and corrected it to headline **revG alone** (revH already spent on the axis decision, reported reference-only). Opened once. **The raw fused label did NOT beat S2 alone out-of-sample** (revG 0.798 vs 0.822; revH identical) - the S3 WALKING-veto flipped ~4 true stands to walk, so the train+val fusion advantage (0.921>0.898) did not replicate. **But the confidence signal generalized and is the real product**: acting-on-confidence (abstain LOW) scored macro-F1 0.845 / acc 0.898 / stand-recall 0.659 at 94% coverage, above S2, with a calibrated LOW tier (0.27 acc). Generalization gap is stand-recall on unseen subjects (0.56/0.38 vs ~0.90 LORO) = the Section 12.2 data ceiling, not a code defect. Deployable artifact is the confidence-GATED system, not the raw relabel. 2-class model FINAL at these numbers; no sealed rev left to spend. |
| 2026-07-22 | 6 | **Governed new-class discovery built + run live (Section 11.2, Section 12.2).** `stages/s4_fusion/newclass.py` (evidence core: 28 `new_class_candidate` spans / 347 windows across 6 revs into per-span physics profiles) + `agents/s4_newclass.py` (proposer, Read/Grep, provenance + cluster-mass gate: >=4 spans across >=2 revs) + `orchestrator.py s4_newclass`. Read-only, orthogonal to the deployed algorithm - never touches the model/rule/fuser/labels; every proposal is `needs_human`. Run live (`runs/2026-07-22_run4`, $0.99, 15 turns): 2 proposals, both cluster-mass supported - `bilateral_transition_maneuver` (10 spans/4 revs, medium: brief in-phase bilateral swing bracketing STAND/WALK, candidate sit-to-stand/squat/turn, with the honest caveat that gravity-referenced thigh sensors cannot distinguish a body turn from a sit-to-stand) and `standing_shifting` (8 spans/2 revs, low: restless stance, the Section 10.3 hypothesis recurring with mass). Both to `needs_human`. Also added `curate.py` two-vote relabel (BOTH S2 and physics must contradict a label - corrected the earlier overstated "error_rate 1.00 = precise mislabel detector"; it is disagreement by construction, Section 12.2). |
