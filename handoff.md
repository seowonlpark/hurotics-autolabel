# h-care-champion — handoff

**Purpose of this document.** A complete description of what was built, what every major
choice was and why, what the evidence for each is, and what is honestly still weak. Written
to be handed to someone (or another model) who has never seen the repo and needs to build a
presentation from it.

Everything here is traceable to a file in the repo. Where a number is quoted, the artifact
that owns it is named. Where something is uncertain, it says so — the project's central
discipline is that a stated weakness beats a tidy number, and a presentation that hides the
weaknesses would misrepresent the work.

**Generated 2026-08-04, against commit `7d0176c`.**

---

## Part 1 — What was built

### 1.1 The deliverable in one sentence

A staged pipeline that takes a raw H-CARE IMU device log and returns **one row of output per
row of input**, each carrying a stand/walk call, a confidence, and — where the model is
unsure — an explicit abstention with a named reason and the alternative it was weighing.

*(The repo never names the device type. The raw schema carries load cells, `L/R_Ref_Force`
described as "controller setpoints — commanded, not measured", and admittance/PID state, so
it is a powered assistive device with thigh-mounted IMUs. Confirm the exact product name with
Lu/HUROTICS before putting it on a slide.)*

```
python -m stages.s2_ml.label data/raw/20251024/00321_63_2025_10_24_15_32_0.csv
```

Output columns, on top of the input's own six:

| column | meaning |
|---|---|
| `Label` | the **committed** call: `0` stand, `10` walk, `-1` scored but below threshold, `255` no window covered the row |
| `guess` | the same call *before* the threshold — `0`/`10` on every scored row, never `-1` |
| `confidence` | `max(p, 1−p)` from the ensemble |
| `ambiguous` | `True` for every `-1` and `255` |
| `reason` | which doubt: `near_transition`, `weight_shift_or_step`, `model_split`, `posture_shift`, `low_excursion_gait`, `out_of_distribution`, `uncovered` |

The `Label`/`guess` split is a deliberate design point worth presenting: **`Label` answers
"may I use this row", `guess` answers "what did it think".** `Label != guess` selects
exactly the rows that were scored and abstained — the review queue, with the model's opinion
attached rather than stripped out.

### 1.2 What it is judged against

> Label stand/walk on any recording at **≥95% accuracy over the rows it commits to**, and
> abstain rather than guess on the rest.

Abstention is treated as a feature, not a failure. That framing comes from the data: the
human annotators themselves have a `-1` "I looked and cannot call it" code, and the model is
being asked to reproduce that judgement, not to paper over it.

### 1.3 Headline results

**Development subjects** (leave-one-rev-out over 7 held-out subjects, 1,244,292 scored rows,
`runs/regen/s2_ml/roweval_loro.json`):

| preset | threshold | coverage | accuracy on committed | worst subject |
|---|---|---|---|---|
| `high_coverage` | 0.70 | 92.75% | 0.9784 | 0.9099 |
| **`balanced` (shipped)** | **0.85** | **84.66%** | **0.9901** | **0.9525** |
| `high_precision` | 0.95 | 67.83% | 0.9980 | 0.9884 |

**Window-level LORO macro-F1: 0.9196**, accuracy 0.9624, over 5,984 label-pure windows
(`runs/regen/s2_ml/locoeval.json`).

**The sealed lockbox subject `rev8`: 0.9308 at 76.4% coverage. The target is met on
development subjects and missed on the one genuinely unseen subject.**

That last line must be in the presentation. It is the most important number in the repo and
the project treats it that way — `caveats.md` §3.2 is labelled "the most important entry in
this file", and `breakdown.py` raises a mechanical `risk` flag for it on every run.

### 1.4 The corpus

Three populations, and conflating them is the easiest way to misread every number:

| population | count | ground truth? |
|---|---|---|
| raw device logs (`data/raw`) | 91 | **no** |
| label-pure 2 s windows (from `data/labeled`) | 5,984 | **yes** |
| rows labelled by the corpus sweep | 2,594,823 | **no** |

- **Labelled set:** 43 annotated trials across 8 revisions in `data/labeled/`.
- **Subjects:** development `rev2, rev3, rev4, rev5, rev6, rev7, rev13`; lockbox `rev8`.
  A `rev` is one subject on one day.
- **Class balance: ~86% walking.** A "predict walk always" model scores ~86%. This is why
  macro-F1 is the headline metric and pooled accuracy is explicitly distrusted.
- **One trial quarantined:** `rev13/trial_4` — 12,691 rows all annotated `stand`, containing
  119.9 s at 45.3 deg/s with 71° of interleg swing. rev13's own labelled *walking* measures
  35–50 deg/s. The second run is walking. The evidence is internal to the file, so the
  exclusion stands with no model involved.
- **Only 7 distinct development subjects.** Stated repeatedly as the binding limitation.

---

## Part 2 — Architecture

### 2.1 Three stages, filesystem as the only interface

```
data/raw ──► S1 clean ──► S2 ml ──► S3 physics ──► runs/breakdown.md
```

| stage | deterministic core | agent role |
|---|---|---|
| **S1 clean** | schema census, rate normalization, gap segmentation, per-file channel trust | `s1_exception` — triage the exception queue |
| **S2 ml** | windowing + features, train + LORO eval, row labelling with abstention, row/raw evaluation, corpus sweep | `s2_experimenter` + `s2_critic` — propose a challenger; **code decides** |
| **S3 physics** | swap-rule anchors, rate-invariance audit, annotation audit, file-level plausibility | `s3_label_review` — judge what the annotation audit flagged |

`run_pipeline.py` is a **dumb sequencer**: 17 steps, each its own subprocess, each gated on
its output artifact existing. It never imports a stage to "help" it — a sequencer that could
reshape a stage's output would be a fourth stage nobody documented.

**Every step must produce its gate artifact before the next starts.** A stage that exits 0
without writing anything stops the run, rather than letting the next stage read a file from a
previous run and quietly report last week's numbers.

### 2.2 The four non-negotiables

1. **Code does the work; agents judge the work.** Agents never touch data values and never
   compute numbers themselves.
2. **No silent mutation.** Failures go to a quarantine ledger; decisions get written
   rationale; low confidence escalates to `needs_human`.
3. **"Best" is defined by `locoeval`, not by an agent's opinion.** An agent proposes a
   declarative `ExperimentSpec`; `experiment.py` runs it, scores it, applies the promotion
   rule, and logs every outcome — rejections included.
4. **Every stage closes with a `DOMAIN_NOTES.md` update.** Discoveries become permanent, not
   conversational.

### 2.3 The document set

| file | role |
|---|---|
| `DOMAIN_NOTES.md` | institutional memory, 1,050 lines. **Injected into every agent's system prompt.** Every entry tagged `[measured]` / `[reported]` / `[decided]` / `[open]` |
| `caveats.md` | everything known to be weak, unverified, or deliberately omitted |
| `OPERATING_POINTS.md` | the abstention threshold: what each preset costs and buys |
| `RUNBOOK.md` | every path, command and output, derived by reading the code |
| `runs/breakdown.md` | generated every run; one page over every stage |
| `runs/README.md` | what is safe to delete, what can never be rebuilt |

---

## Part 3 — S1 clean: what it does and why

**S1's product is a verdict, not a copy of the corpus.** It resamples each file onto a
canonical 100 Hz grid, keeps measured columns only, normalizes gyro units — then *measures
the frame and drops it*. Nothing downstream reads a cleaned frame. The only per-file artifact
it persists is `channel_trust.json`: the measured gyro unit, the Deg→Gyro permutation, and
which axes abstained.

Rationale: a persisted second copy of the signal would be a second thing to keep in step with
raw.

### 3.1 Resolve columns by NAME, never by position

**This is the single most important finding in the repo and the README says to read it
first.** The `NN_` numeric prefix on each column is a per-file position, not an identifier:

- `loco` sits at index 47 in **80 of the 91 raw files** — and in variant `0fda484e`, the
  other **11**, index 47 is `Step`.
- `Step` itself sits at 46 or 47 depending on variant; `Cadence` at 45 or 46.

So `df.iloc[:, 47]` reads a firmware step counter as locomotion state on 12% of the corpus
and **never raises**.

**The minority is what makes it dangerous.** A positional read that broke everywhere would be
found on the first plot; one that is right on 80 files and wrong on 11 produces a result that
looks fine and is not.

### 3.2 Segment at gaps; never resample across one

Gaps land anywhere. A file is a bag of continuous runs, and **the segment, not the file, is
the unit of analysis**. Resampling across a gap would invent data that was never measured.

Current run: 91 files → 141 segments, 94 usable, 481.5 usable minutes. Of the 47 dropped, 35
are one systematic artifact (a ≤12-row export preamble before the first real gap) — recorded
as *one* finding, not 35 independent data problems.

### 3.3 Normalize rate to 100 Hz, with anti-aliasing

The corpus has two eras: ~100 Hz through 2026-01, 500 Hz from 2026-05. Downsampling uses
`scipy.signal.decimate(..., ftype='fir')`, **never `[::5]`**. Measured proof on a test signal
carrying 1.5 Hz gait plus a 120 Hz component:

```
SOURCE 500Hz : 1.5Hz=1.000  120Hz=0.500
FIR decimate : 1.5Hz=1.000  20Hz(alias)=0.000   <- correct
naive [::5]  : 1.5Hz=1.000  20Hz(alias)=0.500   <- fake 20 Hz at full amplitude
```

The alias lands at |120−100| = 20 Hz at full strength, indistinguishable from real gait.

The odd rates (99.3789 / 99.688 / 99.961 Hz) turned out to be **timestamp quantization**, not
different devices — the device clock counts in binary sub-millisecond ticks, so a nominal
10 ms interval lands on `10 + 1/16`, `10 + 1/32`, `10 + 1/256` ms. They are grid-corrected,
not discarded.

*Presentation-worthy consequence:* an earlier experiment on this data reported clustering
that "partitioned by acquisition rate." It was almost certainly reading timestamp counter
granularity, not acquisition rate — the geometry was measuring the clock.

### 3.4 Keep measured channels only

The device *measures* IMU channels and load cells; it *computes* Cadence, Stride Length, GCP,
admittance, PID state. Computed columns are the firmware's opinion, not observation. Dropping
them:

- **(a)** collapses the schema variants into one — every column that churns position between
  variants is a computed one;
- **(b)** removes firmware-version signal from the feature set — training on computed columns
  partly learns *which firmware produced the file*, the era confound baked in.

The exception mechanism (`KEEP_EXCEPTIONS`) is **currently empty**: canonical == measured, no
caveat. An empty exception list is a stronger invariant than a populated one.

The one former exception, `Hip_Deg_L/R`, was cut once every premise was checked: 0.991
correlated with the `Deg_Y` already kept; its residual carries nothing but the firmware's
zeroing convention; zero-variance on 12 of 180 (file, side) pairs, twice **frozen at a nonzero
constant** (`−7.42`, `+9.58`) which no `!= 0` guard would catch; and the open-source dataset it
was a bridge to is not in the repo — the bridge had no far side.

### 3.5 Channel trust: detect, don't assert

Two real hazards, both measured corpus-wide:

- **Units are inconsistent within a single file.** `B_Gyro_*` is rad/s; `L/R_Gyro_*` is deg/s.
  Same naming convention, same file. Any feature mixing them is off by **57.3×**.
- **The Deg↔Gyro axis naming crosses, device-wide.** `d(Deg_Y)/dt` tracks `Gyro_Z` on every
  side (median |r| 0.986 L, 0.987 R), not `Gyro_Y`. This is a naming convention of the device,
  not a bug in one IMU — an earlier framing called it a trunk defect, which would have sent a
  fix to one side of a three-side convention.

Both are **detected per file** and written to `channel_trust.json`. Files too static to answer
**abstain** and fall back to the documented convention, recording that they did.

*Methodological point worth presenting:* the first cut of the permutation detector scored each
side once and reported all three axes — **25 anomalies**, nearly all of the form
`{X→X, Y→Z, Z→X}`. `Deg_Z` is the yaw-like axis that drifts rather than oscillates, so its
derivative is often noise even mid-walk and its argmax is meaningless. Applying the r-floor
**per axis**, so a silent axis abstains instead of dissenting, returns exactly the **2**
genuine anomalies.

---

## Part 4 — S2 ml: the model

### 4.1 The training asset, and why only four channels

The only labelled data is `data/labeled/rev*/annotated_loco_rev*_trial_*.csv` — the `lpf_view`
family, six columns: `Time`, `L_ang_LPF`, `R_ang_LPF`, `L_angvel_LPF`, `R_angvel_LPF`, `Label`.
Thigh angle and angular velocity, per leg, low-passed at 1 Hz.

**Why only these four (rotational) features:** some trials are walked on a **treadmill**. Any
channel encoding *linear translation* — net displacement, stride velocity, stride length —
reads ~0 on a treadmill even while the person is plainly walking, so it actively misleads a
classifier. **Joint angle and angular velocity are rotational: the limb swings the same whether
or not the ground moves underneath.**

This is a good presentation beat — it is a domain constraint that dictated the feature basis,
not a modelling preference.

**The raw↔labelled bridge.** The four columns are derivable from any raw device log:
`L_ang_LPF ≈ lowpass(L_Deg_Y)`, `L_angvel_LPF ≈ lowpass(L_Gyro[sagittal])`. So the model
**trains on annotated exports and runs on a fresh raw CSV** by recomputing the same four
features. Verified to **≤7.3e-13** on all 18 paired recordings (`verify_transform`).

### 4.2 Windowing

| property | value | why |
|---|---|---|
| length | **2.0 s** = 200 samples @ 100 Hz | ~2 gait cycles — the shortest span that can show a leg swap |
| training stride | **2.0 s (non-overlapping)** | overlapping windows manufacture near-duplicate rows and leak between CV folds |
| inference stride | **0.25 s** | a row is labelled by whichever windows cover it |
| bounded by | **segment**, never a file | a window may never span a gap |
| label rule | vote over rows, human `-1` dropped, **purity must be 1.0** | anything less is a `transition`, not a majority vote |

The training/inference stride asymmetry is deliberate and worth explaining: training on
overlapping windows flatters every metric, while inference *needs* overlap so that a row's
probability is the mean over ~8 covering windows. That averaging is what sharpens the
confidence ordering the threshold selects on — and is why row-level evaluation exists rather
than quoting window numbers.

### 4.3 The 38 features

`features.feature_names()` emits **42**; the champion drops 4 and trains on **38**. Read them
as *16 per-leg quantities mirrored L/R, plus 6 bilateral*.

**Bilateral coupling — the strongest signal in the model.**

| feature | what it is | importance rank |
|---|---|---|
| `ang_LR_corr` | L-vs-R angle correlation — strongly negative in gait | **1** (0.1114) |
| `ileg_minhalf` | ptp of `L − R − ileg_zero`, min over 2 sub-windows | **2** (0.0918) |
| `ileg_swaps` | sign changes of that difference, 1° deadband | **3** (0.0523) |
| `ileg_minquarter` | same, min over 4 sub-windows | **5** (0.0470) |
| `angvel_LR_corr` | same on angular velocity | 16 |
| `angvel_LR_lag_s` | inter-leg timing offset from cross-correlation peak | 32 |

**Four of the top five are bilateral. This model recognizes gait mainly by the two legs
alternating, not by either leg's own motion.** That is the single clearest thing to say about
what the model learned.

The `min over sub-windows` construction is the subtle part: the whole-window ptp reads high on
one weight shift, so taking the *minimum* over halves/quarters is what distinguishes sustained
stepping from a single shuffle.

**Amplitude (12):** `L/R_ang_LPF_std`, `_ptp`; `L/R_angvel_LPF_std`, `_ptp`, `_absmean`,
`_mean`. The angle channels get **only** `std` and `ptp` — `mean`/`absmean` on an angle is the
static-offset family, deliberately absent because it encodes mounting and zeroing convention,
not gait, and does not transfer across revisions.

**Spectral / periodicity (8):** `dom_hz`, `band_frac`, `hf_ratio`, `cycle_s` per side.
`GAIT_BAND_HZ = (0.13, 3.0)` is set for **this** population — cadence 16–102 steps/min — not
the healthy-adult `(0.5, 3.0)`. Load-bearing: a genuinely slow walker under the wrong band
reads near-zero exactly like standing for a whole bout.

**Shape (4 kept, 4 dropped):** `posfrac`, `angacc_rms`, `skew` per side kept; `kurt` and
`peakratio` dropped.

**Rest-referenced posture (6)** — the only features that consult a per-file calibration:
`L/R_ang_p95_rest`, `ang_med_rest` (both minus this subject's own standing zero), and
`ang_p95_p05` (the one member needing no zero).

The zero comes from `rest_reference()`: the median over a 3 s resting span, taken from the
file's opening if that opening is genuinely rest, else the best resting span anywhere in the
file, else a whole-recording median with `rest_trusted = False`. **Per file, never
corpus-wide** — each subject is zeroed differently at cuff-fitting time. Currently 1.4% of
windows fall back.

### 4.4 The estimator

**`ExtraTreesClassifier(n_estimators=400, class_weight='balanced', random_state=0)`.**

Chosen against RandomForest on **identical features, folds and params over 5 seeds**:

| model | seeds | macro-F1 | coverage @0.85 | selective acc @0.85 |
|---|---|---|---|---|
| ExtraTrees | 5 | 0.9194 | **0.8760** | 0.9887 |
| RandomForest | 5 | 0.9138 | 0.8189 | 0.9889 |

**The deciding line is coverage at equal precision: same quality of answer, more answers.**
That is the right way to compare two abstaining classifiers and is a clean slide.

### 4.5 Abstention: why 0.85

Pooled accuracy clears the 95% target from threshold 0.50 onward, **so pooled accuracy is the
wrong number to set a threshold by** — it averages a new subject together with six the model
has effectively seen.

**0.85 is the lowest threshold at which every held-out development subject independently
clears 95%.**

| threshold | pooled | worst development subject |
|---|---|---|
| 0.80 | 0.9866 | 0.9383 |
| **0.85** | **0.9901** | **0.9525** |

The threshold is explicitly framed as *a choice on a curve, not an optimum*: maximising
coverage subject to the ≥95% target always drives the threshold to its minimum, producing a
policy that ignores the model's probability entirely.

### 4.6 The reason codes, and how each earned its place

Every ambiguity reason is validated by **the accuracy the guess would have had if it had not
been flagged**. A reason earns its place by sitting well below the confident set:

| reason | rows | accuracy of the suppressed guess |
|---|---|---|
| (confident, kept) | 1,053,405 | **0.9901** |
| `near_transition` | 69,600 | 0.6090 |
| `low_excursion_gait` | 6,342 | 0.7108 |
| `model_split` | 42,576 | 0.8312 |
| `out_of_distribution` | 14,914 | 0.8427 |
| `weight_shift_or_step` | 45,757 | 0.8767 |
| `posture_shift` | 11,698 | 0.9367 |

**Independent check:** the model abstains on **55.8%** of rows a human marked `-1` versus
15.3% elsewhere. `-1` never enters training, so the agreement is not circular.

Two reason codes have been **removed for failing this test**, and both stories are worth
telling:

- `verdict_from_grown_span` fired on **75.7% of all windows**, and those windows were **more**
  accurate (0.9815) than the ones it left clean (0.9682). An audit trail that says "be careful"
  about most of the answer, and preferentially about the correct part of it, is worse than
  silence.
- `moving_stand` failed the same way earlier.

**The check every reason code now has to pass** is: measure the accuracy of the guess the
reason is suppressing.

---

## Part 5 — S3 physics: the swap rule, and an honest retraction

### 5.1 The rule

> **Walking is the legs alternating.** Not how far they swing — *whether they swap*.
>
> Count how many times `L_ang − R_ang` commits past `+1°` and then past `−1°` within the
> window. **0 swaps → STANDING. ≥2 swaps → WALKING. Exactly 1 → AMBIGUOUS.**

**Zero fitted parameters** (for the core rule): `1°` is a sensor noise floor — 5× the measured
standing noise came out 0.76–0.88° on three files independently. `1 swap` is not a chosen
threshold: standing measures **0** and walking measures **2** on every file across a **4×
amplitude range** (rev8 swings 22°, rev2 swings 49°), so 1 is the only integer between them.

Performance standalone, on held-out `rev8_t3`: stand recall 0.966, walk recall 1.000, coverage
0.981, macro-F1 0.988.

The rule counts about a **per-file rest zero**, not about zero: mounting and cuff position give
every recording a per-subject interleg offset, and a large enough one holds the signal entirely
to one side of ±delta through real gait — every crossing suppressed, walking read as 0 swaps.

### 5.2 The retraction — S2 absorbed the physics

**This is the most intellectually interesting result in the project and belongs in the
presentation.**

S3 was originally built as an *independent* second opinion that could veto or corroborate S2.
Then `ileg_swaps` and `ileg_minhalf` were added as S2 features. Measured consequence:

| S2 windows | physics contradicts |
|---|---|
| correct | 1.7% |
| stand called walk (the dominant error) | 31.4% |
| **errors at `s2_proba ≥ 0.95`** | **12.0%** |

So the guard is informative in aggregate — an 18× enrichment — but **it fails precisely where
it is needed**: when the model is confidently wrong, physics usually agrees with it.

**Two opinions over one signal are not two opinions.** Both read the same 1 Hz-filtered
interleg angle, so they are wrong together.

Two routes to restore independence were built and measured:

- **Route A — take the interleg features away from S2.** Costs accuracy, buys 3 points on the
  guard, leaves it useless. The correlation is not about shared features: both opinions read
  the same filtered interleg motion, and where that signal is ambiguous they are wrong together
  however S2 is fed.
- **Route B — give S3 a channel S2 has never seen** (raw accelerometer magnitude; the 1 Hz
  low-pass removes every footfall impulse before S2 looks). Genuinely informative (AUC 0.88)
  and genuinely independent (fires on 6% of correct windows, 28% of errors). **And it still
  does not work:** at `p ≥ 0.95`, neither guard flags a single error while accelerometer still
  fires on 4% of correct windows — a false-alarm generator, not a safety net.

**The conclusion is about the errors, not the guards.** A confident error here is not a window
where one view dissents; it is a window that looks like the other class to *every* measurement
available. No second opinion over the same recording can catch it.

**When neither route worked, the whole S4 fusion stage was deleted (2026-08-04) rather than
kept as an unearned second opinion.** One deliverable, one proof.

### 5.3 Where the swap rule survives, and why

The rule is not gone — it is pointed at the thing it is still independent of:

| consumer | pointed at | worth |
|---|---|---|
| `label_audit.py` | the **annotations** | **strong** — the rule never sees a label, so a disagreement is evidence about the labels |
| `plausibility.py` | a whole **file**, against the model | **moderate** — a file-level contradiction is a diagnosable fault |
| `label.py` physics gate | single **windows**, against the model | **weak, ships OFF** |

That progression — strong / moderate / weak-and-off — is a clean way to present "we kept the
part that survived scrutiny and switched off the part that didn't."

### 5.4 Negative controls

Three, and they matter because **an audit that has never rejected anything is not evidence that
the others passed**:

- **`gyro_energy` must FAIL the rate-invariance audit.** It sums over samples, so halving the
  rate halves it (median relative Δ 0.4994). It fired, so the three anchors that passed
  (`periodicity` 0.0273, `antiphase` 0.0005, `grav_stab` 0.0001) mean something.
- **`plausibility.py` injects synthetic channel faults** into a clean recording on every run,
  not behind a flag, and reports which bounds catch them.
- **`freshness --self-test`** runs first and unconditionally, because a staleness checker that
  has quietly stopped firing looks exactly like a pipeline with nothing wrong.

*The rate audit also had to be fixed before it could be trusted:* decimating a bare 200-sample
window applies a 41-tap FIR whose edge transient covers over half the window. That artefact
alone moved `antiphase` by 0.1734 and condemned it `rate_dependent` — **the literal gait
signature, rejected on a filter edge.** Decimating with 1.28 s of real context either side
drops the move to 0.0005. The control that proves the fix did not just blunt the test:
`gyro_energy` is unchanged at 0.4994 and still fails.

---

## Part 6 — The evaluation regime

This is where the project is most defensible and is probably the strongest presentation
material after the domain findings.

### 6.1 Windows vs rows — two units, and only one ships

`train.py` scores **windows**, the unit the model learns on. A caller labels a CSV and gets
**rows**. The two differ: rows are scored by averaging every window that covers them, which
changes both the accuracy and the confidence ordering the threshold is set from.

`roweval.py` runs the **real `label.py` path**, not a reimplementation, per held-out rev. That
is the number this repo quotes.

`raweval.py` closes the last link: `roweval` measures the annotated export, `verify_serve`
shows the raw route agrees with that export row for row but drops `Label` before comparing — so
it proves *equivalence*, never *correctness*. `raweval` scores the raw route directly against
human labels, with the lockbox refused **in code** rather than by remembering a flag.

### 6.2 Grouped cross-validation

`LeaveOneGroupOut(rev)`. A `rev` is one subject on one day, so a held-out rev is a genuine
"new subject/session" — the honest deployment bar. Each subject is scored by a model that
never saw it, **including its reference statistics**, so nothing about the held-out person
entered the thresholds its rows were judged against.

### 6.3 The lockbox

`rev8` was held out of feature selection, model selection *and* threshold selection, and opened
**exactly twice** (both reads logged, both honest — once while rev13 was still sealed, once
after freezing). **No decision in the repo was made using either result.**

**It is now spent and must not be read again.** The discipline is enforced against real
temptation: when `roweval` was found to be fitting 42 features where the champion declares 38,
the development row was re-measured (it moved 0.0003) and **the lockbox row was deliberately
left un-re-measured**, with the provenance mismatch stated rather than tidied away. The note in
`OPERATING_POINTS.md` is explicit that *"the numbers in the table match" is precisely the kind
of reason that makes re-reading feel harmless.*

### 6.4 Coverage and accuracy are a pair

Quoted together everywhere. Either alone is meaningless — abstain on all but the easiest window
and accuracy reads 1.000. **Treat any future summary that quotes one without the other as
broken.**

The per-subject table makes the point concretely: pooled 84.66% coverage is an average over
subjects it claims 98.8% of (`rev13`) and subjects it claims 76.5% of (`rev2`).

### 6.5 Watch stand recall, not accuracy

**90% of committed errors are standing called walking** (9,364 of 10,391). The residual failure
is one-directional.

| truth → guess | rows |
|---|---|
| walk → walk | 945,747 |
| stand → stand | 97,267 |
| **stand → walk ⚠** | **9,364** |
| walk → stand ⚠ | 1,027 |

The per-subject recall columns expose what pooled accuracy hides. `rev5` holds **0.9525
accuracy while getting a third of its standing wrong**: stand recall 0.6647 against walk recall
1.0000. That is the same shape as the lockbox (0.5752 vs 1.0000) — **on a development
subject**, visible all along under a pooled number that averaged it out.

**"Stand recall is the canary and pooled accuracy is the anaesthetic"** is the repo's own line
and it is a good slide title.

---

## Part 7 — The agent layer

Built on the Claude Agent SDK. Four agents, each read-only over artifacts, each unable to
change anything by itself.

| agent | model | job |
|---|---|---|
| `s1_exception` | haiku | triage the clean stage's exception queue into explained/action |
| `s2_experimenter` | sonnet | propose ONE declarative `ExperimentSpec` |
| `s2_critic` | sonnet | review that proposal **before** any training run |
| `s3_label_review` | sonnet | assign a cause to each flagged trial; nominates, never enacts |

**Total logged spend: $2.67** across the 5 live runs in `runs/keep/agent_runs/`, plus **$4.50**
across 5 archived S4-era runs in `archive/runs/` — **$7.16 all told.** Quote $2.67 if you mean
the current pipeline, $7.16 if you mean the whole project.

### 7.1 The cost asymmetry is the design

The critic reviews **before** a fit is paid for. Six proposals were stopped by the critic and
never cost a training run at all:

`balanced_class_weight`, `drop_offset_family_and_band_frac`, `more_trees_1000`,
`drop_angvel_ptp`, `drop_offset_family_and_angvel_mean`, `drop_angvel_lr_corr`.

### 7.2 Promotion is code, not opinion

`experiment.decide()` gates on measured macro-F1 with a **`PROMOTION_MARGIN = 0.005`** — a
margin, not `>`, because LORO over a handful of revs is noisy and a +0.001 win would ratchet on
noise. Tie-break is on `steady_confusion` share: at equal accuracy, prefer the model that fails
passively.

It also **refuses to compare across corpora**: `comparable()` checks a corpus fingerprint
(window count, class set, rev set) and fails *passive* if either side is missing one. Macro-F1
averages per-class F1, so it does not survive a change of class set — subtracting across one is
exactly the error the guard exists to prevent.

The ledger records rejections too. The current basis:

| experiment | macro-F1 | features | outcome |
|---|---|---|---|
| `extratrees400_drop_moments38` | 0.9196 | 38 | promoted (incumbent seed) |
| `min_leaf2_extratrees` | 0.9211 | 42 | **rejected** — +0.0016, below the 0.005 margin |

A further 16 entries sit on an earlier basis and are **excluded from that table**, because a
macro-F1 column that mixes two bases invites exactly the subtraction `decide()` refuses to make.

### 7.3 What agents may never do

Delete raw data, edit `DOMAIN_NOTES.md` without human review, or retrain the champion. Each run
writes `run_meta.json` (the commit), `costs.json` (per-agent spend), `system_prompt.txt`
(exactly what the agent was told), and a `run_log.jsonl` of every tool call via a `PostToolUse`
hook.

### 7.4 A real operational finding

`DOMAIN_NOTES.md` is injected into every agent's system prompt. It grew **30,029 → 54,551 chars
in one day (+82%)**, putting the system prompt at 56,765 against a ~32,767 `CreateProcess` cap
on Windows. Every agent began failing with `CLINotFoundError: Claude Code not found at …` —
pointing at a binary that was present, 253 MB, and ran fine standalone.

**The real cause was `WinError 206 ERROR_FILENAME_EXCED_RANGE` — the command line was too long,
not the path.** The SDK catches any `FileNotFoundError` during spawn and blames the CLI.

Fix: write the prompt to a file and pass `system_prompt={"type": "file", ...}`. A path is O(1)
on the command line, and the file doubles as an audit record.

There is a **soft cost** too, measured on the same run: the cheap exception agent returned 22
verdicts for 23 queue items and stopped populating a required field, while cost doubled
$0.052 → $0.111. Nothing changed but prompt size. The deterministic wrapper caught the missing
verdict and marked it `needs_human` — no silent drop.

---

## Part 8 — The methodological spine (this is the actual thesis)

If the presentation has one argument beyond "we built a classifier", this is it. The repo is
organised around a small number of methodological rules, each of which was *learned by being
burned*.

### 8.1 Provenance tags on every claim

`[measured]` (reproducible by re-running a named stage) / `[reported]` (a human said so) /
`[decided]` (a design choice) / `[open]` (known unknown). A `[measured]` tag is only worth what
the measurement isolated.

### 8.2 Retract in place, never edit quietly

Wrong claims are struck through with the correction and the reason beside them. Examples that
are all real and all documented:

- **`angvel` is not noisy.** An entry claimed the angular-velocity channels "ring at ±40–80
  deg/s during static postures", then "confirmed" it by measuring |angvel| p99 ≈ 105 deg/s on a
  file that is **93.7% walking**. That measured walking and called it rest. The channel was
  never noisy; the measurement was wrong.
- **"Walking with no rhythm" does not exist.** Those windows are identical to normal walking on
  every descriptor except a periodicity measure that fails at *cadence changes*, not arrhythmia.
  They occur at ~69 s and ~196 s in both trials independently — a protocol event, probably a
  turn. The model invented a category to explain its own artifact and nearly asked for it to be
  defined.
- **`gyro_energy` bimodality was two subjects**, unimodal within each — not two behaviours.
- **The physics-independence claim** (§5.2) was the load-bearing premise of a whole stage, and
  was retracted in place rather than quietly softened.

### 8.3 An unexercised path cannot be wrong out loud

**The flagship failure.** `SAGITTAL_DEG_AXIS_BY_VARIANT` mapped variant `fb5ea2c2` — the
**majority variant, 62 of 91 raw files** — to `Deg_X`. It should have been `Deg_Y`.

Reproducing the labelled columns from raw, per axis, max absolute error:

| variant | pairs | `Deg_Y` → `Gyro_Z` | `Deg_X` → `Gyro_X` | `Deg_Z` → `Gyro_Y` |
|---|---|---|---|---|
| `fb5ea2c2` | 7 | **≤2.4e-13** | 208–368 | 193–460 |
| `0fda484e` | 10 | **≤7.3e-13** | 177–232 | 99–350 |
| `4bfd6ab2` | 1 | **6.8e-13** | 381 | 247 |

68% of the corpus would have been served the frontal plane instead of the sagittal one,
**silently, with every downstream number still looking plausible.**

**Two things let it stand, and both are the lesson:**

1. **It was unreachable.** `verify_transform` was its only caller, and re-running it is exactly
   what exposes the failure. The bug was one command away the whole time and nothing ran that
   command — until the serve path was built and *the first thing the path did when pointed at
   real files was fail*.
2. **It was self-justifying in prose.** The source comment and the domain note both argued at
   length that `fb5ea2c2` is "not an anomalous permutation, simply a revision whose sagittal
   plane is `Deg_X`" — which **reads as evidence and is an explanation of a number nobody
   re-derived**.

The fix is structural: `verify_transform` and `verify_serve` now run **on the pipeline spine,
not behind a flag**, because slow is not a reason to skip a check.

A related finding: those two evidence scripts had at one point been deleted from the working
tree silently, without an entry in any of the deletion tables the repo keeps. **An evidence
script that can be deleted without a doc noticing is a claim with no owner.**

### 8.4 "Recorded" ≠ "handled"

Two places claimed a per-file record "must be consulted"; **neither had a consumer.** S1
measured the answer and nothing read it. When judging whether an anomaly is *handled*, verify a
consumer exists — a note saying a record is "recorded" is not evidence that anything consults
it. (The yaw drift flag is still unenforced, and is now **labelled** as unenforced rather than
described as behaviour.)

### 8.5 An effect that does not scale with the mechanism it is attributed to is not that effect

Nine alternative confidence signals were built and scored against `max(p, 1−p)`. The best,
`conf − λ·std` over a subject-holdout ensemble, looked like a genuine win: **+0.0197 on the
worst subject at coverage 0.95**, exactly the trade a subject-transfer fix should make.

**It inverts under a stronger perturbation.** Leaving one of six subjects out barely moves the
model (`p_std` mean 0.0160). Rebuilding the ensemble on 3-subject subsets — all C(6,3)=20 —
nearly tripled the spread (mean 0.0428), so a real subject-shift signal should get *louder*. At
λ=10 the worst-subject effect goes from **+0.0197 to −0.0065**. The 5-subject gain was noise on
seven subjects.

Also proved rather than measured: **recalibration cannot be the answer.** Platt/temperature/
isotonic are *monotone* maps on `p`, so they preserve row order and accuracy at fixed *coverage*
is invariant under them. The lockbox defect is an *ordering* defect, so no recalibration touches
it.

### 8.6 Density needs mass

Mode-finding declared files "unimodal — one behaviour" when a 5 s stand was 0.4% of the file.
*A stop is not a mode; it is a stretch of time.* Hit **twice**, the second time one message
after invoking it as a lesson.

### 8.7 Tests that were themselves broken

- `cluster_stability` scores a **continuum at 0.974** vs real clusters at 1.000. It measures
  whether k-means cuts repeatably (a gradient does, deterministically), not whether there is
  anything to cut.
- An anchor-independence check became **tautological** once the anchor was defined by the
  descriptor it was tested against.
- **GMM + BIC counts Gaussians, not modes.** It selects k=8 with evenly-spaced means on a
  2-state signal.
- A "physics also contradicts the label on 100% of those windows" claim was **struck as
  arithmetic, not corroboration**: with two classes, `model != label` and `physics != label`
  *forces* `physics == model`.

### 8.8 The report measures nothing

`runs/breakdown.md` is assembled from every stage's JSON. **Every number is copied from the
artifact that owns it and the artifact is named beside it, so it cannot disagree with a stage.**
Re-run the stage, not the report, to change a number.

It does two things no stage report can:

- **It states its own gaps.** A missing artifact is printed in §B with the command that produces
  it, because a stage that did not run must not read as a stage with nothing to say.
- **It raises mechanical flags**, including *whether the checked-in prose still quotes the live
  headline*. That exists because `README.md` and `caveats.md` both sat at a stale pair for a run
  that measured something else; both told the reader to prefer the report, and both were still
  wrong to a reader who did not.

Current run raises 5 flags: `unchecked` (un-stamped artifacts), `risk` ×2 (lockbox below target;
per-subject spread 0.8553–0.9687), `concentration` (12 low-coverage files clustering on 4
sessions), `gap` (13 files unservable).

**Flags are not verdicts. Each names what to look at and none of them conclude anything.**

### 8.9 Delete rather than keep an unearned claim

S4 fusion was deleted because it measured a policy `label.py` never ran, so the number was true
of nothing anyone shipped. The rate audit's `decimate_window` was deleted because its own
docstring said it was unused and wrong. Dead code removed in the same pass is listed with the
reason for each.

The counterpart discipline: `runs/keep/` holds everything that **cannot** be rebuilt — the spent
lockbox read, the append-only experiment ledger, the ablations behind each `rejected` line — and
`runs/regen/` is safe to `rm -rf` at any time. `rm -rf runs/regen` is described as *the fastest
way to prove the pipeline still builds from nothing*.

---

## Part 9 — Honest limitations

Put these on a slide. The project's credibility rests on stating them.

1. **The 95% target is missed on the one genuinely unseen subject.** rev8: 0.9308 at 76.4%
   coverage. Every development number is an **upper bound, not an estimate**.

2. **On rev8, accuracy is non-monotonic in the threshold** — 0.8997 → 0.9308 → 0.9184 at
   0.70/0.85/0.95. Abstention is supposed to buy accuracy monotonically. It does not there,
   which means the threshold chosen on development subjects is not transferable and the
   worst-subject column is an **optimistic floor**.

3. **Seven subjects is a small basis** for a threshold, a probability floor, and every tier
   boundary. Per-rev accuracy ranges roughly 0.91–0.99, so a single unusual subject moves these
   numbers more than any tuning does.

4. **Nothing predicts which new subject will degrade.** Five candidate warning signals were
   built and all fail. The most instructive: *predicted walk fraction* has the **strongest
   correlation of the five (ρ = +0.75) and is worthless** — subjects predicted mostly walking
   score better, because the only failure mode is standing called walking, so a recording with
   little standing has few chances to fail. **On a corpus this imbalanced, any flag built from
   prediction statistics is measuring the class mix, not the difficulty.**

5. **`stand` is not one behaviour.** It absorbs quiet standing, weight shifts, turning, sitting
   and transfers because the taxonomy has nowhere else to put them. A third class would address
   it; that is deliberately out of scope.

6. **A confident-accuracy figure is a lower bound on model quality and an upper bound on label
   quality, and the two cannot be separated by measuring harder.** Of 112 confident errors on
   development subjects, **87 (77.7%)** are windows where the physics also contradicts the
   annotation. But since S2 absorbed the physics (§5.2), that is a weaker statement than it
   sounds — a shared blind spot produces it as readily as a bad label. **None of these 87 has
   been adjudicated by a human.**

7. **13 of 91 raw files cannot be served at all** — variants `e5f2660f` (11) and `86069795` (2)
   have no measured sagittal-axis mapping. Signal-only axis detection scores **below chance**
   (16% / 16% / 5% against a 33% random baseline), so guessing is not available. What unblocks
   it is **one paired raw+annotated recording per variant** — not more raw data.

8. **12 of 78 labelled files commit to under 50% of their rows**, and they cluster: 4 of 12
   sessions on variant `fb5ea2c2` hold every one, the other 8 hold none. The same hardware
   records fine outside 20251230–20260109. **That is a session-level cause, not 12 hard
   recordings** — and nobody has looked at one by eye yet.

9. **1.6% label noise is accepted.** `label_windows` drops `-1` rows and votes over the rest, so
   94 of 5,984 trainable windows contain `-1`, 22 are more than half `-1`, worst is 93.5%.
   Accepted deliberately; `unknown_frac` rides on every window so the effect is checkable.

10. **The serve path reproduces an upstream bug on purpose.** `serve_dt` filters with the *final*
    inter-sample interval because the vendor's `timestamp.m` does, and the training features carry
    that quirk. Using the honest median instead moves features by up to **20.1 deg**. If the
    vendor fixes it, this must be re-measured, not preserved out of habit.

11. **Two transition-timing measurements do not reconcile** and the discrepancy is flagged
    `[open]` rather than papered over: an older count of 27 timeable transitions / 14 on-time
    against a newer 308 candidates / 267 timeable / 170 within ±1 s. Neither is derivable from
    the other.

12. **Single-seed macro-F1 differences below ~0.004 are not interpretable at this corpus size.**
    An earlier version of the champion spec compared seed-0 runs against a ±0.0003 band — which
    is the *standard error of a 5-seed mean*, not the spread of one draw. That error was found
    and corrected on 2026-08-04, and several "rejected" ablation lines were downgraded from
    "regression" to "no measured difference" as a result.

---

## Part 10 — New finding (2026-08-04, this session): the feature set is ~35% redundant

Not yet in `runs/keep/`. Measured in scratch; reproducible from the repo's own
`cross_validate` / `evaluate` / `selective_curve`. **Include only if you want a "what's next"
slide** — it is a live result, not a shipped change.

### 10.1 What was measured

The 38 features contain heavy near-duplication, measured on the training windows:

```
0.992  L_ang_LPF_std        L_ang_p95_p05     0.989  L_angvel_absmean  L_angvel_std
0.988  L_ang_LPF_ptp        L_ang_p95_p05     0.977  L_angvel_ptp      L_angvel_std
0.979  L_ang_LPF_ptp        L_ang_LPF_std     0.899  L_angacc_rms      L_angvel_ptp
```

Six angle-amplitude features do the work of two; eight angvel-magnitude features do the work of
two. Dropping one representative of each pair (plus `hf_ratio`, which correlates 0.94–0.97 with
its own complement `band_frac`, and `angvel_LR_lag_s`) gives **25 features**.

5 seeds, LORO:

| variant | n | macro-F1 | Δ | cov@.85 | worst-rev@.85 |
|---|---|---|---|---|---|
| champion (baseline) | 38 | 0.9186 ±0.0006 | — | 0.8791 | 0.9671 |
| **redundancy-pruned** | **25** | **0.9280 ±0.0006** | **+0.0094** | 0.8797 | **0.9711** |
| top-27 (importance-ranked) | 27 | 0.9176 ±0.0009 | −0.0009 | 0.8812 | 0.9699 |

**The redundancy axis and the importance axis separate cleanly.** Every ablation in the repo
truncates by *importance*, which is the wrong axis when features are 0.99-correlated —
importance splits among clones, so they all rank low and get cut together.

### 10.2 Why it is NOT being proposed as a promotion

Two adversarial checks were run, and both weaken it:

- **The mechanism is `max_features`, not the features.** ExtraTrees defaults to
  `max_features='sqrt'` — ~6 of 38 columns per split. Duplicates crowd that draw. The gap
  collapses as the splitter sees more columns: **+0.0091 at `sqrt`, +0.0040 at 0.5, +0.0019 at
  `None`** (i.e. gone). It is a tuning finding wearing a feature-selection costume.
  Incidentally, `max_features=0.5` on the untouched 38 buys +0.0026 by itself.
- **One held-out subject carries it.** `rev2` gains +0.0354 (10/10 seeds). Excluding rev2, the
  mean-over-revs delta is **+0.0020 ±0.0019** — indistinguishable from zero. `rev3` and `rev4`
  return delta *exactly* 0.0000 with zero seed variance; those folds are saturated.

**What does hold:** the 13 features cost nothing on any rev (worst is rev7 at −0.0013), so
**parsimony is free** — which is exactly the argument `champion_spec.json` used to drop 4
moments ("four features that buy nothing measurable are four fewer things to explain and to keep
trustworthy across a new subject"). It is a maintainability case, not a performance one, and
should be argued that way.

**A gate gap this exposes:** `decide()` would have promoted this on pooled macro-F1. It has no
per-group concentration check, over 7 groups ranging 89 to 1,751 windows. One fold can buy a
promotion.

---

## Part 11 — Suggested presentation structure

### A narrative that works

1. **The task and the honest bar** — stand/walk per row, ≥95% on what you commit to, abstain
   otherwise. Show the output contract (`Label` vs `guess`).
2. **Why the data is the hard part** — lead with the positional-indexing trap (`loco` at index
   47 in 80 files, `Step` in 11, never raises). This immediately establishes why the pipeline is
   shaped the way it is.
3. **The feature basis is a domain constraint, not a preference** — treadmill trials make every
   linear channel lie, so only rotational features are admissible.
4. **What the model learned** — four of the top five features are bilateral. Gait is the legs
   alternating.
5. **Results, with coverage and accuracy always paired** — the preset table, then the per-subject
   table, then stand recall as the canary.
6. **The lockbox** — 0.9308 vs 0.9901, and the fact that it was not re-read even when it would
   have tidied the docs.
7. **The retraction story** — S2 absorbed the physics, both routes to independence failed, S4 was
   deleted. This is the intellectual high point.
8. **The methodological spine** — unexercised paths, provenance tags, negative controls, reason
   codes that must earn their place.
9. **Limitations, stated plainly.**
10. **What's next.**

### Numbers to memorise

- 0.9901 accuracy at 84.66% coverage, worst dev subject 0.9525 — shipped, threshold 0.85
- 0.9308 at 76.4% — lockbox rev8, target missed
- macro-F1 0.9196, 5,984 windows, 38 features, ExtraTrees 400
- 90% of errors are stand→walk; corpus is 86% walk
- 7 development subjects, 1 lockbox, 91 raw files, 78 servable, 43 annotated trials
- $2.67 agent spend on the current pipeline ($7.16 including archived S4-era runs)

### Things NOT to claim

- **Do not quote accuracy without coverage.** Ever.
- **Do not call the lockbox number a small gap.** It is the finding.
- **Do not call S3 an independent second opinion.** That claim is formally retracted.
- **Do not quote "zero fitted parameters" about all of S3** — it is true of the core swap rule;
  the slow-gait path has four chosen constants (6.0 s, 0.35, 8.0°, 0.5).
- **Do not present any number from the corpus sweep as accuracy.** `labeled_raw/` has no ground
  truth; it reports coverage only.
- **Do not quote single-seed macro-F1 differences below ~0.004** as meaningful.
- **Do not claim rare-class (stairs, terrain) performance.** The corpus has essentially only
  standing and walking; any such claim is overclaiming.

### The one-line summary if you only get one

> A stand/walk classifier that reaches 99% accuracy on the 85% of rows it commits to — and a
> pipeline built so that every one of those numbers can be re-derived, every claim carries its
> provenance, and the places it fails are written down more carefully than the places it works.
