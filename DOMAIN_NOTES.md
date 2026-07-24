# DOMAIN NOTES - IMU locomotion pipeline

**This file is injected into every agent's system prompt. It is the pipeline's institutional memory.**

**This is a data-pipeline tool, not a study of one dataset.** The dataset fed into it can change, and
much of what follows is grounded in one *reference corpus* - the data the pipeline was developed and
validated against. Read those entries as *observations from that corpus*, not as invariants that hold
for whatever data you feed in next. Concrete identifiers, counts, dates, per-file measurements and
per-split scores describe the reference corpus; on new data they must be re-measured, and the live
values regenerate every run in the `runs/*` artifacts. Labeled-export units are referred to here by
anonymized labels (`revA`, `revB`, ...); the code and run artifacts identify them by their own
concrete names, so map by role (for example "the sealed lockbox rev", "the deleted weak rev") rather
than by index when cross-referencing. What *does* generalize - and is the durable
value of this file - is the **labeling conventions** (Section 5), the **methodology and its failure
modes** (Section 11), the **design decisions** and the **reasoning** behind each finding. Keep those;
treat the numbers as illustrations.

Rules for this file:
- Every entry is a *finding with a reason*, not an instruction without justification.
- Agents may propose additions (via their rationale logs). Only a human commits them.
- If you are an agent reading this: these facts are established *for the reference corpus*. Do not
  re-derive them, do not contradict them silently. If your evidence contradicts an entry - or if you
  are working on different data where an entry may not hold - say so explicitly and flag
  `needs_human` rather than acting on it.
- **[measured]** = derived from the reference corpus, reproducible by rerunning S1.
  **[reported]** = came from a human, not independently verified.
  **[decided]** = a design choice, not a fact.
  **[open]** = known unknown.

---

## 0. The corpus

**[measured]** `data/raw/` holds one family (`raw_device`) of device logs across many dated session
folders. **This file records findings, not tallies** - live counts (files, folders, clean vs
quarantined, usable minutes, rate mix) regenerate every run in `runs/*/clean_report.md` and
`census.md`; read those for current numbers. In the reference corpus one export batch was quarantined
for a degenerate time base (Section 2.6) - the failure mode and its handling are the finding; the
specific batch is not.

Session date comes from the folder name. **Subject is filename field 2** (Section 5.5) - the
cross-validation group key, *not* session. In the reference corpus one subject dominated and the
others appeared in only one or two sessions each; a corpus with a different subject balance is
expected, and cross-subject claims are always bounded by the live subject count (Section 5.5).

Labeled data lives in `data/labeled/`, never `data/raw/`. A `Label` column appearing under
`data/raw/` is a contamination event: quarantine and flag it.

---

## 1. Schema

### 1.1 One family, one canonical schema **[measured]**

The corpus carries several raw header shapes (differing column widths), all one `raw_device`
family. They **differ only in the tail** - the wider widths are all *computed* columns the clean
layer drops (Section 9) - so the head is a single stable contract and the differences never reach a
canonical file. **We do not track or branch on header shape.** Columns are read by name (Section 1.3),
which makes the shapes interchangeable; there is no per-header lookup anywhere in the pipeline.
`census.md` reports file/family counts and any unregistered column name (the real "schema changed"
alarm) - nothing enumerates header variants, because nothing consumes them.

### 1.2 The contract is 45 columns **[measured]**

`Time` through `Total Gait Length` is identical across **every** header seen. Everything past index
44 is a computed column. The 45-column prefix is the real contract - the wider widths are not.

### 1.3 The numeric prefix is a LIE - resolve by name, never by position **[measured]**

The `NN_` prefix is a per-file position, not a stable identifier:

- `Step` sits at index **46 or 47** depending on the header
- `Cadence` sits at index **45 or 46**
- `loco` sits at index 47 in most files - but in some headers **index 47 is `Step`**

`df.iloc[:, 47]` therefore blends a step counter into a locomotion state, on the majority of files,
and never raises. The column that *looks* positionally stable (`loco`) is precisely the trap.

Note both are **outdated columns anyway**: `loco` is the legacy rule-based algorithm's output (the
thing this project replaces - severed, Section 4.5) and `Step` is a firmware-computed counter. Neither is a
trusted signal; both are dropped from the canonical measured set. The finding here is about the
*positional-indexing danger*, not about the columns' value.

**Policy:** strip the prefix, match on the name. Column presence is per-file; absence is recorded
as a fact, not an error.

---

## 2. Sampling rate

### 2.1 Two rate eras, confounded with schema AND session date **[measured]**

| | earlier era (through ~2026-01) | later era (from ~2026-05) |
|---|---|---|
| rate | ~100 Hz | 500 Hz |

Rate, header shape and date change together. Resampling removes the *rate* difference; it does not
remove the era. A model can still learn "era" as a shortcut. Group by **subject** (Section 5.5) for
cross-validation to defend subject leakage - but note era is a *separate* confound: a subject who
spans both eras (as the dominant subject did in the reference corpus) is not neutralized by subject
grouping, which does nothing about era-specific firmware artifacts.

### 2.2 The "odd" rates are timestamp quantization, not different rates **[measured]**

Measured rates cluster at 100.0 Hz and 500.0 Hz, plus a few "odd" values just below 100 Hz
(99.3789, 99.688, 99.961). Those odd ones are exact quantization tiers, not different devices:

| measured | dt (ms) | = |
|---|---|---|
| 99.3789 Hz | 10.062500 | 10 + 1/16 |
| 99.688 Hz | 10.031250 | 10 + 1/32 |
| 99.961 Hz | 10.003906 | 10 + 1/256 |

The device clock counts in binary sub-millisecond ticks, so a nominal 10 ms interval lands on the
nearest representable value. **These are 100 Hz devices.** Do not delete them; grid-correct them.

### 2.3 This reframes an earlier per-subject rate experiment **[measured, unconfirmed against source]**

An earlier per-subject experiment reported clustering partitioning by "acquisition rate" across
99.4 / 99.7 / 100.0 / 500.0 Hz - exactly the tier ladder above. The geometry was likely reading
**timestamp counter granularity**, not acquisition rate. Still a confound, still disqualifying for
`gyro_energy` as a body-defined anchor, but the mechanism differs from what was assumed. Not yet
confirmed against that experiment's source.

### 2.4 Canonical grid is 100 Hz **[decided]**

Incoming golden data is 100 Hz **[reported]**, so 100 Hz is canonical: most files already sit
there, the 500 Hz era is decimated down, and off-grid rates are grid-corrected. Nothing is
upsampled; no bandwidth is invented. (The per-method segment counts each run are in `clean_report.md`.)

### 2.5 Never downsample without anti-aliasing **[measured]**

Taking every 5th sample of 500 Hz folds everything above 50 Hz into the gait band. Verified on a
test signal carrying 1.5 Hz gait + a 120 Hz component:

```
SOURCE 500Hz : 1.5Hz=1.000  120Hz=0.500
FIR decimate : 1.5Hz=1.000  20Hz(alias)=0.000   <- correct
naive [::5]  : 1.5Hz=1.000  20Hz(alias)=0.500   <- fake 20 Hz at full amplitude
```

The alias lands at |120-100| = 20 Hz, at full strength, indistinguishable from real signal.
Use `scipy.signal.decimate(x, 5, ftype='fir')`.

### 2.6 A corrupted `Time` column - quarantined **[measured]**
In the reference corpus a few files from two sessions had a **destroyed `Time` column**: it collapses
to a small, non-monotonic range (median `dt` = 0, a large fraction of `dt` negative). No forward
cadence exists, and `np.interp` would silently corrupt the resample. This is the failure mode to
recognize on any data; the specific files are incidental.

**It is a per-file `Time` corruption, not a bad header and not clock jitter.** Healthy sibling
files of the *same* header shape and era carry a clean monotonic 500 Hz clock - so the
acquisition is fine; only these files' timestamps were overwritten. The *data* channels look
intact and in row order (row-to-row continuity matches a healthy file). There is **no recoverable
surrogate clock** in the file (the large monotone counter some carry is a normal column, present in
the healthy files too).

**Decision (2026-07-20, data owner): do NOT reconstruct a synthetic clock.** A uniform-rate clock
would *fabricate* unmeasured time, which the pipeline refuses to do. These files stay quarantined as
`needs_human` (re-export from the source is the only honest fix). The non-corrupted files proceed
downstream normally - `data/clean/` only ever holds files that passed, so later stages never see
the corrupted ones.

S1 catches this up front (`clean_one` guards `median(dt) > 0`; `measure_hz` returns `nan` rather
than dividing by zero) and routes it to the `quarantine.jsonl` ledger with reason
`degenerate time base`. The raw file is left in place, never moved.

**Update (2026-07-22): part of the batch was removed from the corpus by the data owner (pulled, not
re-exported), so a current clean run quarantines just the remaining corrupted file. It remains
`needs_human`; re-export from source is still its only honest fix. The finding stands - removal
disposed of the batch, it did not fix the timestamps.**

---

## 3. Gaps and segments

### 3.1 The segment is the unit of analysis, not the file **[measured]**

A file is a bag of continuous runs; the **segment**, not the file, is the unit of analysis (usable
segment count and minutes are in `clean_report.md`). Windows must never straddle a gap; resampling
across one invents data that was never measured.

### 3.2 There is a ~10-sample startup burst **[measured]**

Many files open with a segment of **exactly 10 rows**, then a gap, then the real trial. The 500 Hz
era shows two tiny leading segments (2-8 rows each). Mechanical, not random.
`MIN_SEGMENT_SAMPLES = 100` currently trims it as a side effect of a size filter - the right
outcome for an incidental reason.

### 3.3 Gap position is otherwise unpredictable **[reported + measured]**

Apart from the startup burst, gaps land anywhere. One reference-corpus file shatters into 6
segments, 4 unusable. The ~1264 ms gap in nine 500 Hz files is systematic (all within 14 ms of each
other) but its cause is **unknown** - suspected to be an error **[reported]**. Policy: segment
conservatively, never interpolate across, flag the pattern.

---

## 4. Channel trust

### 4.1 Angular velocity is RELIABLE - it is the derivative **[measured]**

`angvel_LPF` **is** `d(angle)/dt`: r = **0.999**, slope **0.98**. Use it directly; no need to re-derive
velocity from the angle channels. (An earlier "it rings during static postures" claim was
**retracted**: the "confirming" measurement took |angvel| p99 over a 93.7%-walking file and called it
rest - a `[measured]` tag is only worth what the measurement isolated; Section 11 item 1, *density needs
mass*.)

### 4.1b Gyro units and axis names are both inconsistent within a file - fixed in the clean layer **[measured]**

- **Units: `B_Gyro_*` is rad/s, `L/R_Gyro_*` is deg/s** (same file, same naming) - any feature mixing
  trunk and thigh gyro without conversion is off by **57.3x**. The clean layer normalizes every gyro
  channel to **deg/s**, detected per file (regress `d(Deg_Y)/dt` on each gyro axis, slope -> unit; static
  files, whose derivative carries no signal, **abstain** to the documented convention and record that
  they did - Section 11.1). Verified post-clean: L/R/B slopes 0.987/0.984/0.986, the 57.3x gone.
- **Axis names: the Deg<->Gyro crossing is DEVICE-WIDE, a full permutation X->X, Y->Z, Z->Y**
  (`DOCUMENTED_GYRO_PERMUTATION`), identical on all three sides and every header - not a trunk defect
  (median |r| of `d(Deg_Y)/dt` vs `Gyro_Z` ~= 0.96-0.99, vs `Gyro_Y` ~= 0.15-0.32). So **`*_Gyro_Y` is
  NOT the sagittal rate**; the channel matching `*_Deg_Y` is **`*_Gyro_Z`**. Resolve it from the file's
  `channel_trust.json` (`gyro_axis_by_deg_axis`), never from the column name - `transform.check_axis_trust`
  enforces this on the read path (refuses a file whose measured permutation breaks on the axis it is
  about to read).
- **The permutation is NOT sagittality** (conflating them was a real bug). It says which gyro axis
  measures which *angle* axis - internal to a file, measurable because `Deg` is the reference - not which
  axis is the sagittal plane (no in-file signature, Section 6.2). The pipeline fixes the plane to Y
  (Section 6.3) rather than detecting it.
- **Detected, not universal:** two reference-corpus files map `B_Deg_Y -> B_Gyro_Y` (one sign-flipped) -
  flagged as an anomaly, not blindly swap-corrected (a blanket Y<->Z swap would corrupt exactly those).
- `Deg_Y` needs **no sign normalization** (raw L vs R is already antiphase in 84% of files) and is the
  sagittal (flexion) channel the swap rule reads. Axes are **recorded, not reordered** - no silent
  mutation; per-run rollups live in `clean_report.md` / `channel_trust.json`.

### 4.2 Yaw is drift-contaminated - but per-file, not wholesale **[measured]**
Original **[reported]**: in treadmill data yaw correlated with session time at r ~= -0.95, measuring
elapsed time not orientation.

Measured on the raw corpus (clean-layer drift test, `|corr(Deg, Time)|` duration-weighted over
segments >= 5 s): **`Deg_Z` is the yaw-like axis** on every side - median |r| ~= 0.33 vs the sagittal
`Deg_Y` at ~= 0.08. But the strong drift signature is **file-specific, not universal**: only a
minority of files cross |r| >= 0.9, and always on `*_Deg_Z` (the flagged channels each run are in
`clean_report.md` / `channel_trust.json`). So yaw is **flagged per
channel per file** in `channel_trust.json` (`drift` section), **not dropped wholesale**. It is a
feature-time exclusion signal; the raw superset is kept.

**Any yaw-derived feature must consult the per-file drift flag - [decided], and currently
UNENFORCED [measured].** No code reads `drift_contaminated`; the flag is inert, and it costs nothing
only because no feature reads `Deg_Z` at all (S2 trains on the four rotational features, sourced from
the sagittal `Deg` axis and its gyro rate). **The first yaw-derived feature must add the consumer** -
writing one against this section and assuming the exclusion already happens would silently train on
drift.

### 4.3 The trunk (B) IMU was dropped from revA for a real reason **[reported]**
Belly/trunk placement was inconsistent between subjects, and may have been treadmill-mounted in some
trials. The raw family still carries `B_Deg_*` / `B_Gyro_*` / `B_Acc_*`; treat trunk channels as
suspect until placement consistency is established per session.

**This flag is about PLACEMENT only.** It is not an axis-labeling flag and it is not a unit flag:
the Deg<->Gyro name crossing is device-wide (Section 4.1b), and the rad/s units are fixed in the clean layer.
The two `B_Deg_Y->B_Gyro_Y` files are an axis anomaly, tracked in Section 4.1b, not evidence of bad trunk
placement. Keep the two risks separate - conflating them makes the trunk look doubly untrustworthy
and lets the L/R axis crossing hide.

### 4.4 `Time` is metadata, not a feature **[decided]**
Used for dt / rate / segmentation only. Never fed to a model.

### 4.5 `loco` is severed and stays severed **[measured]**
Its "standing" class contains a decile **as periodic at the gait frequency as median walking**. A
label that fails inspection cannot validate anything. (It is also an outdated algorithm's output
**[reported]**, but that is the weaker reason - the strong one is that it is demonstrably wrong.)

Not ground truth, not a feature. Must be dropped **by name** - in some headers, dropping index
47 would delete the step counter instead.

### 4.6 `Deg` is a FUSED ESTIMATE, not a transducer reading **[decided, 2026-07-20]**

The measured-vs-computed line in Section 9 is real but its label is too coarse. An IMU has exactly two
transducers - a rate gyro and an accelerometer. **Angle is not among them.** `L/R/B_Deg_*` is the
sensor's onboard fusion output (gyro integration corrected by the gravity vector, Kalman or
complementary). It is *computed*; it just happens on the sensor die instead of in app-layer firmware.

The honest partition is three-way, not two-way:

| tier | channels | trust |
|---|---|---|
| **transducer** | `*_Gyro_*`, `*_Acc_*`, `L LC` / `R LC` | raw observation |
| **on-sensor fusion** | `*_Deg_*` | computed, but vendor-fixed and firmware-*version*-stable |
| **app-layer compute** | `Cadence`, `Step`, `Stride Length`, `Hip_ROM`, `GCP`, `Hip_Deg_*`, `loco`, admittance / PID state | the firmware's opinion; churns with firmware version |

**This does not change what S1 keeps.** The drop rule targets the third tier, and that is still
correct - tier 3 is what churns column position between headers and bakes in the era confound (Section 9).
`Deg` stays.

**What it changes is how `Deg` may be described.** Do not call it ground-truth measurement:
- It is the reason **Section 4.2 yaw drift exists at all.** Drift is what dead-reckoned fusion does with no
  magnetometer to correct heading. Section 4.2 reports drift as an empirical oddity; this is its mechanism,
  and it predicts the finding: `Deg_Z` (heading, unobservable from gravity) drifts, while `Deg_Y`
  (pitch, continuously corrected by the gravity vector) does not.
- It means `Gyro` is the *more primitive* channel, not the derived one. Section 4.1's `Gyro == d(Deg)/dt`
  is true, but the causality runs the other way: `Deg` was integrated **from** `Gyro`. That is why
  the correlation is so tight (r = 0.999) - it is near-tautological, not independent corroboration.
- Its fusion parameters are a device property. Two revisions may fuse differently even where every
  column name matches.

---

## 5. Labels

### 5.1 Two unknowns, opposite in kind - never merge them **[reported]**

- **`255` = machine unknown.** Data error. Nothing was measured properly.
- **`-1` = human unknown.** Data is fine; a trained human looked and could not call it.

### 5.2 `-1` is training-poison and evaluation-gold **[decided]**
It is a hand-drawn map of where confidence *should* be low - the exact signal the current
rule-based algorithm lacks. Exclude from training targets (ambiguous label = noisy target); retain
and report separately in evaluation as the natural test set for a confidence signal.

### 5.3 Label encoding **[measured]**
A representative labeled trial: `int64`, no nulls, three codes - `10` (93.7%), `0` (5.6%),
`-1` (0.8%), 12 segments. `-1` appears at most but not all transitions, consistent with its
meaning: absence means the labeler *could* tell. (Class shares vary by trial; the encoding is the
finding, not the proportions.)

### 5.4 Class imbalance makes accuracy meaningless **[measured]**
93.7% walking. A "predict 10 always" model scores 93.7%. Macro-F1 is the headline metric.

### 5.5 Subject = filename field 2, and it is KNOWN for every file **[measured + data-owner-confirmed, 2026-07-20]**
**Corrected - this reverses the earlier entry.** An earlier version claimed filename field 2 "tracks
the date block (device / firmware / protocol), not a subject," and concluded subject was
unrecoverable (a `needs_human`). **Both are wrong.**

Field 2 is the **subject tag** (data-owner-confirmed, corroborated by the data): its value is constant
within every session-day folder, and one value recurs across **many folders spanning many months** in
the reference corpus. A date/firmware/protocol block cannot span that long; a person tested repeatedly
can.

In the reference corpus one subject dominated; the others appeared in only one or two sessions each,
and at least one subject had only a degenerate-time-base file (Section 2.6) so it contributed no
usable data. Some trial-batch markers once seen in filenames were **trial batches of one subject,
not two people** (data-owner-confirmed) - since removed from the filenames; they were trial indices,
never wearer ids. (Per-subject file counts, if needed, come from parsing field 2 across `data/raw/`.)

**Consequences:**
- Cross-validation groups by **field 2 (subject)**. This defends subject leakage fully; session-day
  grouping is subsumed (one subject per day).
- Subject identity is **not** a `needs_human`. **RESOLVED.**
- Real remaining limitation (not a blocker): **few distinct subjects**, one dominant, several tested
  only once or twice. Cross-subject generalization is bounded by subject *count*, not unknown
  identity. State that limit honestly; do not overclaim breadth.

### 5.6 Corpus class coverage bounds what can be claimed **[reported]**
The reference corpus contains essentially only STANDING and WALKING. Rare-class separability (stairs,
varied terrain) is **untestable** on such data. Any claim about rare-class performance is overclaiming
unless the fed-in corpus actually carries those classes with mass (Section 11.1) - check coverage
before making one.

### 5.7 The labeled family IS the training asset; features are rotational for a reason **[decided + reported, 2026-07-20]**
The supervised task is **stand (`0`) vs walk (`10`)**, `-1` excluded from targets (Section 5.2). The **only**
labeled data is the `data/labeled/rev*/csv/annotated_loco_rev*_trial_*.csv` set - the derived revA
family (Section 6.1), whose four features are `L/R_ang_LPF` (low-pass sagittal angle) and `L/R_angvel_LPF`
(low-pass angular velocity). The S1 clean corpus (raw IMU superset) carries **no labels**; the two
never share a file. The open-source gait dataset config.py calls the "primary asset" is **not in the
repo** - until it is, rev* is the training asset.

**Why only those four (rotational) features - not the fuller raw set (data-owner-reported):** some
trials are walked on a **treadmill**. Any channel encoding *linear translation* (net displacement / stride velocity /
stride length / total gait length) reads ~0 on a treadmill even while the person is plainly walking,
so it actively misleads a classifier. **Joint angle + angular velocity are rotational** - the limb
swings the same whether or not the ground moves underneath - so they are treadmill-robust. (Those
linear channels are exactly the *computed* columns S1 already prunes; the kept raw `Acc` still holds
oscillatory gait content, but whether to *use* it is a feature-layer call per Section 9, not a clean-layer
one. See Section 4.3: trunk placement itself was treadmill-suspect.)

**The raw<->rev\* bridge (why this still classifies "loco state from a CSV"):** the four rev* features
are **derivable from any raw device CSV** - `L_ang_LPF ~= lowpass(L_Deg_Y)`, `L_angvel_LPF ~=
lowpass(L_Gyro[sagittal])`, both kept by S1. So the model trains on rev* labels and **runs on a fresh
raw CSV** by recomputing the same four features. Feature extraction is the bridge; the deliverable is
CSV -> four rotational features -> per-window loco state.

---

## 6. Products / families

### 6.1 revA is a separate, lossy family **[measured]**
`Time, L_ang_LPF, R_ang_LPF, L_angvel_LPF, R_angvel_LPF, Label` - 6 columns, sharing only `Time`
with the raw device family. It discards the trunk IMU, **all accelerometers** (the gravity
reference, hence all axis/calibration checks), load cells and GCP; and it keeps both
angular-velocity channels - which are **reliable**, being `d(angle)/dt` (Section 4.1), not the "untrusted"
they were once called.

**Do not make revA the storage format.** Canonical storage is the name-resolved raw superset. A
wide honest table can always be projected down; a narrow one can never be recovered.

**The labeled rev\* set is MIXED-RATE across trials [measured, 2026-07-20].** Not one family rate:
by median-dt, `revH` is 100 Hz, `revI` ~99.4 Hz, `revA` `trial_1/2` are 500 Hz while `trial_3/4`
are 200 Hz. (No contradiction with Section 7's "revA samples at 494 Hz" - that is the *span/n* estimate on
one file; median-dt reads 500. The two legitimately disagree, which is exactly why `manifest.py`
reports both.) Consequence: **never assume a rate for this family** - detect per trial and normalize.
`stages/s2_ml/dataset.py` puts every trial on the 100 Hz grid via S1's `resample_file`, so windowing
downstream sees one rate. Dropped in the process: only the Section 3.2 startup-burst fragments (~0.011% of
rows), and they are counted, not silently discarded.

### 6.2 The raw->revA feature transform is RESOLVED EXACTLY **[measured]**
The four labeled features are reproducible from any raw device CSV to float roundoff (~1e-13, verified
on 19 paired recordings). The transform is a first-order **causal** IIR (single pole), per channel:

    a    = 2*pi*dt*fc / (2*pi*dt*fc + 1)
    y[0] = x[0]
    y[n] = a*x[n] + (1-a)*y[n-1]

with **fc = 1 Hz** for both angle and angular velocity. It is **causal, not zero-phase** - it adds lag.
Do NOT substitute `filtfilt`: that shifts features relative to labels.

- `*_angvel_LPF` is **`LPF(Gyro)`** (filtered raw gyro), **not** `d(*_ang_LPF)/dt`; the two only
  resemble each other because `Gyro == d(Deg)/dt` (Section 4.1).
- `*_ang_LPF` is `LPF` of the sagittal `Deg` axis - fixed to Y by Section 6.3.

**Signal-only axis detection does not work [measured - every rule scored below a 33% chance baseline].**
The highest-amplitude Deg axis is reliably *not* sagittal (`Deg_Z` variance is dominated by yaw drift,
Section 4.2), and L/R antiphase is necessary but not sufficient (every planar-swing projection is
antiphase). Do not re-attempt it - the fix was to stop needing it (Section 6.3). The labeled set is
nonetheless genuinely sagittal: all revs show L/R antiphase in walking (median r -0.43 to -0.86).

**Upstream MATLAB defects to guard against on the next export [measured - the current labels are
clean]:** `timestamp.m` filters using only the *last* inter-sample dt (if the final two timestamps tie,
`a=0` and the output freezes flat for the whole trial silently - the Section 2.6 failure mode; use
`median(diff(time))`), and `csv2mat.m` does naive 5x decimation with no anti-aliasing (Section 2.5).
Neither was active for the current labels (paired row counts identical).

### 6.3 The pipeline always reads the sagittal/Y plane **[decided + implemented]**
**Policy (in code): `transform.raw_to_features` always reads `Deg_Y` + `Gyro_Z`** (the Y->Z permutation
of Section 4.1b), for every file, whatever its header. The data provider confirmed the exporter always
selects the sagittal plane (which they call Y), so there is no axis to detect and no per-header lookup
to miss. Anomalies are **flagged, not guessed**: `check_axis_trust` refuses a file whose measured
permutation breaks on Y (Section 4.1b); a drift check (dominant frequency ~= DC on the axis read) is the
candidate guard for "this `Deg_Y` isn't gait."

Y is also the *better* axis, not just the convention - proven on the one revision (revH) whose labeled
export used `Deg_X`, re-derived on each axis and scored under the Y-trained model:

| revH fed as | accuracy | macro-F1 |
|---|---|---|
| `Deg_X` (its exporter axis) | 0.913 | 0.708 |
| **`Deg_Y` (policy)** | **0.948** | **0.764** |

because `Deg_Y` carries 3-4x the gait swing (the sensor is rotated so most sagittal flexion lands on Y)
and every training rev is Y. **Lockbox note:** this scored revH, so **revH is spent**; `revG` stays
sealed as the final test.

### 6.4 revI is excluded **[decided]**
`revI` is out of the pipeline (`dataset.EXCLUDED_REVS`; CSV kept on disk). It is derived-only (no raw in
the repo, so its header was never measurable) and held out weak (macro-F1 0.647, gait 2-4x lower
amplitude); "genuine quiet gait vs wrong-axis export" is undecidable without its raw, and removing it
raised champion LORO macro-F1 0.8862 -> 0.8932. Re-derivable on the Y plane (Section 6.3) if the raw ever
arrives. (Full rationale in BUILDLOG Appendix B.)

---

## 7. Evaluation

- Headline metric: **macro-F1**.
- Error taxonomy is a strictly precedence-ordered MECE partition. **Corrected against the
  implementation [measured, 2026-07-20]** - ported verbatim from the upstream `locoeval/diagnose.py`
  into `stages/s2_ml/taxonomy.py`:

      correct
        > omission - a gt segment pred never reaches, split by what pred did instead:
           ; swallowed     - pred flanks it with the SAME label both sides (bout absorbed, no trace)
           ; omission      - pred flanks it with two DIFFERENT labels (transitioned, skipped the class)
           ; edge_omission - touches a recording boundary, so one flank does not exist
          > flicker - a pred run shorter than 200 ms flanked by equal labels
            > late  - pred still shows the old label after a gt transition
              > early - pred already shows the new label before a gt transition
                > steady_confusion - pred stays in another class for the WHOLE gt segment

  The earlier line here was wrong twice: **there is no `remainder` bucket** (`steady_confusion`
  absorbs whatever precedence leaves, so the partition closes without a catch-all), and
  **`omission` splits three ways** - `swallowed` is the one worth watching, since a fully
  absorbed bout leaves no trace at all. Thresholds (the three actually wired): `FLICKER_MAX_MS=200`,
  `LAG_MAX_MS=1000`, `SUSTAINED_FRACTION=0.5`. (The port's `MIN_EVENTS_FOR_STATISTIC`/`WEAK_CLASS_F1`
  low-sample/weak-class tagging was never implemented here and has been dropped, 2026-07-22.)
- **The taxonomy is ROW-level (~10 ms) and a windowed classifier cannot be scored by it
  directly [decided].** A model predicting once per 2 s is piecewise-constant over that
  span, so it *cannot emit* a run shorter than `FLICKER_MAX_MS` - flicker would read zero
  by construction, not by merit, and lag would quantize to whole windows. Inference
  therefore slides the window at a small stride (`stages/s2_ml/predict.py`, 100 ms default,
  finer than the flicker threshold) and assigns each prediction to the rows around its
  **centre** - leading-edge assignment would shift every predicted transition half a window
  late and manufacture `late` rows. Training is unaffected; this is inference-side only.
  Scoring at row level is also what makes our classifier directly comparable to the
  incumbent algorithm under its own metrics.
- **The promotion gate: macro-F1 primary, error *type* as tiebreaker [decided, 2026-07-20].**
  Implemented in `stages/s2_ml/experiment.py: decide()` - the only path to champion.
  - Primary: `macro_f1` must beat the incumbent by `PROMOTION_MARGIN = 0.005`. A plain `>` would
    ratchet the champion on noise and call it progress; leave-one-rev-out over seven revs is not
    precise to +/-0.001.
  - Secondary, firing **only on a statistical tie** (`|delta| < margin`): a `steady_confusion` share
    lower by `STEADY_CONFUSION_MARGIN = 0.02` promotes. **At equal accuracy, prefer the model that
    fails passively.** `steady_confusion` is a sustained wrong call over a whole bout, which on a
    powered device is a sustained wrong *action* (stairs read as sitting); `swallowed`/`omission`
    withhold assistance, which is unhelpful rather than hazardous. **Wrong action beats no action as
    a hazard** - an earlier framing here had this inverted.
  - `champion.json` therefore carries `taxonomy`, and the current incumbent (`drop_angvel_dom_hz`,
    macro-F1 **0.8886**) does. **The tiebreaker has now DECIDED its first promotion (2026-07-21):**
    `drop_angvel_dom_hz` tied on macro-F1 (+0.0024, **below** the 0.005 margin) but cut
    `steady_confusion` 0.884 -> 0.818 (>= 0.02), so it promoted on the fail-passive criterion - the
    first champion change the *secondary* criterion carried, not the primary margin. This is the
    tiebreaker working exactly as designed: a metric-neutral change that moves error out of the
    hazardous bucket is preferred.
- **The incumbent `loco` algorithm's failure profile, measured [measured - supersedes the [reported]
  recollection].** Run through the *same* taxonomy against the *same* ground truth
  (`profile_incumbent.py`), `loco`'s dominant bucket is **also `steady_confusion`**, not the *swallowing*
  the recollection assumed; its distinguishing failure is `edge_omission` (declining to commit near
  recording boundaries). This does not resurrect `loco` (Section 4.5 severed it) - measuring how a
  predictor *fails* is a different question from trusting what it says. **Caution [measured]:** the naive
  share-of-errors comparison (`loco` 0.519 vs ours 0.888 `steady_confusion`) is scored on almost-disjoint
  recordings (7/8 `loco` pairs are the sealed lockbox) - a Section 11 different-denominator trap. On the
  one shared non-lockbox recording, same rows, ours is **lower** (0.016 vs 0.086; row accuracy 0.966 vs
  0.769). Directional (one recording), pending lockbox-open.
- The measure layer is **blind**: objective numbers only, no opinion. All judgment lives in diagnose.
- Ground truth already locates transitions exactly. Do not implement cross-correlation lag search.
- UNKNOWN is excluded consistently across per-trial and corpus-level metrics.
- Group by **subject (filename field 2)** for cross-validation (Section 5.5): subject is known for every
  file, so this defends subject leakage directly. Session-day grouping is subsumed - one subject per
  day. Caveat: in the reference corpus the subject pool was small and one subject dominated (see the
  census for the live count), so cross-subject claims are bounded by how few distinct subjects exist,
  not by unknown identity.
- **The labeled family (rev\*) groups by `rev`, not by field 2** - those filenames have no subject
  field. A `rev` is **one subject on one day** (identity + date unknown; trials within a rev share
  both, data-owner-confirmed, 2026-07-20). So the group unit there is the rev, and a held-out rev is a genuine
  "new subject/session" - the honest deployment bar for "classify loco state from an unseen CSV."
- **Lockbox holdout [decided].** Split the labeled set **before any training**, grouped by rev so no
  trial leaks: *train* (fit) / *validation* (the champion-challenger loop gates promotions here) /
  *lockbox* (1-2 whole revs, sealed, opened **exactly once** at the very end for the honest number).
  The loop must never see the lockbox, or the final macro-F1 is a number it optimized toward, not a
  generalization estimate. Candidate lockbox spans both regimes - a balanced rev (`revG` ~70/27) plus
  a walk-heavy one (`revH`).
- **Always select by time, never by index.** revA samples at **494 Hz, not 500** - jitter
  accumulates to **4.2 s of drift by t=320 s**, so `int(t*fs)` points 4.2 seconds past the event.
  **AUDITED (2026-07-20) [measured]:** `stages/s1_clean/resample.py` is time-safe - every grid is
  built from real timestamps (`np.arange(t[0], t[-1], ...)`) and every value interpolated against the
  measured time vector, never reconstructed from a rate. A synthetic 494 Hz segment places a
  true-t=160000 ms event at output **160000.0 ms (err 0.00)**, vs **+1943 ms** under the
  `int(t*fs)@500` bug. `manifest.py` / `census.py` never invert rate to an index either.
- **Per-file calibration, never corpus-wide** (for threshold rules / S3 physics). A fixed threshold
  scores `walk_rec = 0.000` on one rev (calls every walking window standing); per-file calibration
  scores 1.000 on the same file. Global constants are the disease - one rev wants 0.426 where another
  wants 0.605; the answer is not needing one. **But this is a THRESHOLD property - it does NOT transfer
  to the S2 ML champion [measured]:** built as an S2 challenger, per-file posture + amplitude calibration
  *lost* to the global-features champion (posture recentring is a no-op - the champion already drops
  those features - and amplitude rescaling *removes* generalizable signal the leave-one-rev-out
  RandomForest uses). Calibration code was reverted.
- **Abstain rather than force.** Coverage of 73-98% is acceptable; abstained windows genuinely
  contain both states, mirroring the human `-1` label. Note the tension: the abstained windows are
  the transitions, which is where a controller most needs an answer.

---

## 8. Environment gotchas

*Moved to BUILDLOG Appendix A - development/runtime-environment issues (NumPy, Windows CLI cap, cp949, SDK buffer) for whoever runs the pipeline, not inference-time agent knowledge.*

---

## 9. Standing decisions

- Sample rate is 100 Hz (10 ms). **Window length is now an open tradeoff, and data warrants
  revisiting it [measured]:** 2 s gives transition precision; 4 s is needed for slow gait (a 0.22 Hz
  stride yields ~0.9 swaps per 2 s window and abstains). Wider windows cost wider abstention at
  transitions.
- **Band-limited features must not assume a healthy-adult gait band.** `(0.5, 3.0) Hz` was wrong for
  the reference population, whose cadence ran down to **0.13 Hz** (16-102 steps/min) **[measured]**.
  Assistive/rehab populations walk slow and variable; measure the band on the data at hand rather than
  assuming a textbook one.
- Random Forest is the starting model. The windowing/feature-extraction layer is the durable,
  model-agnostic boundary - **feature selection lives there, not in the clean layer.**
- **An agent proposes a declarative `ExperimentSpec`, never code [decided, 2026-07-20].** The
  vocabulary is: features to drop, `window_s`, `stride_s`, and whitelisted model hyperparameters
  (`ALLOWED_MODEL_PARAMS`, each bounds-checked before any training - an agent dict otherwise reaches
  the estimator constructor verbatim). `random_state` and `n_jobs` are **deliberately absent**:
  reproducibility and machine resources are the pipeline's to decide, not a proposal's. A spec is
  reviewable before it runs and reproducible after, and re-running a logged spec reproduces the
  model exactly - which is what makes revert trivial.
- **`window_s` and `stride_s` must stay independent [measured, 2026-07-20 - a confound in our own
  machinery].** `ExperimentSpec` originally tied stride to window, so changing `window_s` silently
  halved the training set (5,226 -> 2,477 windows) and the comparison could not separate "longer
  window" from "half the data". Re-tested cleanly at 4 s/2 s (4,935 windows): -0.0542 vs the
  confounded -0.0581. The conclusion survived - **4 s windows really are worse**, and `revI`
  collapses 0.6105 -> 0.3536 - but it survived by luck, not by design.
- **The canonical file keeps MEASURED channels only.** The line is measurement vs computation, not
  useful vs useless. The device *measures* IMU channels (L/R/B x Deg/Gyro/Acc) and load cells; it
  *computes* Cadence, Stride Length, Hip_ROM, GCP, Adaptability, admittance, PID state. Computed
  columns are the firmware's opinion, not observation - the same category as `loco`, and there is
  nothing to trust-check in a number the firmware derived. Two consequences beyond storage:
  **(a)** every column that churns position between headers (`Step` 46/47, `Cadence` 45/46, the
  whole 83/79/91 tail) is a computed one, so dropping them **collapses every header shape into one**;
  **(b)** computed values depend on firmware version, so training on them partly learns which
  firmware produced the file - the era confound baked into the feature set.
  Canonical = **30 measured columns**, defined as `KEEP_MEASURED` in `stages/s1_clean/config.py`.
  (The rule targets *app-layer* compute; on-sensor fusion is a separate tier - see Section 4.6.)
- **The measured contract is ALWAYS 30 columns** - `KEEP_MEASURED` is fixed, identical across every
  header, no per-file variation (the historical 32/33 counts were the since-removed `Hip_Deg` pair).
  The clean layer then appends a `segment` index (`resample.py`, Section 3.1) as **row metadata - not one
  of the 30**, in no `KEEP_*` tuple - so a parquet holds those 30 plus `segment`. `Label` is not among
  them: raw carries no labels (Section 0), so `KEEP_IF_PRESENT` fires only on the labeled family.
- **`KEEP_EXCEPTIONS` is EMPTY - the measured-only rule has no exceptions [decided].** `Hip_Deg_L/R`
  were once kept as a bridge to an open gait dataset's `Hip_Flex_L/R`; removed, because they are
  0.991-redundant with `Deg_Y` (the sagittal angle S1 already keeps, re-zeroed), their residual *is* the
  firmware zeroing convention the measured-only rule exists to strip, and nothing downstream read them
  (the bridge dataset is not in the repo, Section 5.7). Recoverable by name from `data/raw/` if it ever
  arrives - same standing as `loco`/`L/R_Ref_Force`. An empty `KEEP_EXCEPTIONS` is a stronger invariant
  than a populated one: canonical == measured, no caveat. (Removal autopsy in BUILDLOG Appendix B.)
- **The 30-column clean parquet and the modelling track are disjoint - kept that way on purpose
  [confirmed, 2026-07-22].** `data/clean/**.parquet` (30 measured cols) is read only by S1 itself
  (`clean.py`, `peek.py`, the S1 exception agent); **S2/S3/S4 never open it.** They read the separate
  revA view (`data/labeled/`, `Time` + 4 rotational features + `Label`), which shares only `Time`
  with the raw family (`FAMILY_MARKERS`). So end to end the model consumes exactly **4 channels** -
  L/R sagittal angle + angular velocity - while the clean layer deliberately keeps the full measured
  superset (all 3 IMUs, every Acc axis, both load cells). This is the Section 9 principle in force, not
  waste: **the clean layer is model-agnostic so a future feature (Acc for a gravity anchor, `B` for
  trunk lean, load cells for stance) isn't foreclosed by today's 4-feature model.** Narrowing the
  parquet toward what's modelled was considered and declined - it would bake the current feature
  choice into the cleaning layer, which is exactly the coupling this boundary exists to prevent.
- `L_Ref_Force` / `R_Ref_Force` are excluded as controller setpoints (commanded, not measured).
  **[open]** - not yet confirmed with the firmware side.
- Storage cost is a file-format problem, not a column-count problem: clean output is **parquet**.
- HMM is a post-processing smoothing layer, not a standalone model. Its transition penalty trades
  off against transition lag - the same problem the existing rule-based algorithm has. Tune it
  deliberately; do not adopt naively.

---

## 10. The swap rule (S3 physics baseline) **[measured]**

**Walking is the legs alternating.** Not how far they swing - *whether they swap*.

> Count how many times `L_ang - R_ang` commits past `+1deg` and then past `-1deg` within a 2 s window.
> **0 swaps -> STANDING. >=2 swaps -> WALKING. Exactly 1 -> AMBIGUOUS.**

`1` is genuinely ambiguous, not a fudge: one leg passing the other happens both when you take a step
and when you shift your weight.

| file | stand_rec | walk_rec | coverage | macro-F1 |
|---|---|---|---|---|
| revA_t1 | 0.930 | 0.989 | 0.830 | 0.928 |
| revA_t3 | 1.000 | 0.930 | 0.729 | 0.776 |
| **revG_t3** (held out) | **0.966** | **1.000** | **0.981** | **0.988** |

**Zero fitted parameters.** `delta = 1deg` is a sensor noise floor - 5x the measured standing noise
came out 0.76-0.88deg on all three files independently. `1 swap` is not a chosen threshold: standing
measures **0** and walking measures **2** on every file across a **4x amplitude range** (revG swings
22deg, revA swings 49deg), so 1 is the only integer between them.

Known weakness: slow walking (0.22 Hz stride) yields ~0.9 swaps per 2 s window and abstains. A
**window-length** problem, not a rule problem (Section 9).

### 10.1 Validated descriptors **[measured]**

| descriptor | status |
|---|---|
| `ileg_minhalf` - min of `ptp(L-R)` over the window's two halves | AUC 0.967 / 0.972 on two independent trials; medians 0.2-0.5 (standing) vs 39-41 (walking). **Needs per-file calibration.** |
| `interleg_offset` - median of `L-R` | **Posture only** (AUC 0.50 for walk/stand). Separates feet-together from split-stance: baseline -3.3deg, splits at **+17deg** and **-22deg** - opposite legs leading. |

### 10.2 Recordings begin at rest **[measured - most files, NOT universal]**

An **external** label - it comes from how sessions are run, not from any algorithm. It gives every
file a standing reference measured on the same person, sensor and mounting minutes earlier. **This
is the mechanism that makes per-file calibration possible** (Section 7), and it is the
calibration-as-data-harvest insight arriving from the physics side.

**[open]** many files also open with a segment of exactly 10 rows (Section 3.2). Possibly the same files -
worth checking.

**[open - protocol] revD breaks the assumption.** 6 of 8 revD files open *mid-gait* (a 36-44deg interleg
swing, 4-5 swaps), not at rest - unique among all revs (Section 10.5.1). The rest-quality guard now detects
this and falls back to the whole-file median, flagged `rest_offset_trusted=False`, so no S3 number is
affected. But it means the "begins at rest" label is a per-session property that does **not** hold for
revD - worth raising with whoever owns the acquisition protocol (different operator/session?). Not an
algorithm bug; a data-provenance fact to confirm.

### 10.3 States richer than the human labels **[measured]**

Human labels are `0` / `10` / `-1`. The physics distinguishes more: `WALKING`; `WALKING_SLOW`
(cadence < 0.6x the file's own median); `STANDING_FEET_TOGETHER`; `STANDING_SPLIT_L` / `_R` (0 swaps,
offset +17deg/-22deg from baseline - stopped mid-stride, one leg leading); `STANDING_SHIFTING`
(**unvalidated - rests on 3 and 8 windows**).

In revA_t1 the last **61.9 s** labeled WALKING is walking at a third the cadence, and the two
STANDING bouts are **different postures**. The labels are coarser than the signal.

### 10.4 Ported to S3 (2026-07-21) **[measured]**

The swap rule and descriptors are now a pipeline stage (`stages/s3_physics/`), computed on the
canonical 2 s windows the rest of the pipeline uses. Faithful to Section 10 on the non-lockbox files:
revA_t1 walk-recall **0.994**, revA_t3 **0.931** (Section 10 reported 0.989 / 0.930). The **lockbox stays
sealed** - S3 runs on train+val only. An agent reading a sealed rev's plots would spend its
independence as surely as training on it would, so the seal is not only the training path (see
tracker POSTDAY4 Section 3).

**Four anchors, each with a rate-invariance verdict** (`rate_audit.json`; 100->50 Hz decimation
through the S1 anti-aliasing filter, judged on walking windows so a silent standing window cannot
manufacture noise - the Section 11.1 trap). A fifth, `gait_hz`, was dropped: degenerate at the 2 s window
and its "invariant" verdict was a false pass (Section 10.8).

| anchor | what | verdict |
|---|---|---|
| `grav_stab` | steadiness of the gravity-referenced tilt (->1 standing, Section 4.6) | **invariant** |
| `periodicity` | rhythm strength (normalized-autocorrelation peak), amplitude-free | **invariant** |
| `antiphase` | -corr(L_ang, R_ang) | **rate_dependent** - moves even in walking windows; the Section 6.2 "necessary but not sufficient" caution, now measured |
| `gyro_energy` | summed `angvel2` over the window | **rate_dependent** - re-derives the Section 2.3 rate-experiment failure from scratch; the sum scales with sample count, so it is a claim about the *grid*, not the body (defined the failing way on purpose) |

**Window length is the product-critical open item.** At a fixed 2 s window the swap rule abstains on
slow gait - one leg swing does not complete an alternation, so `swap_count` reads 0-1 (Section 9/Section 10's known
weakness, now inherited by the stage). This is **not cosmetic**: the deployment population
(assistive / rehab) walks slowly and variably, which is exactly where the fixed window fails, so a
**stride-adaptive window** - length scaled to the detected stride period - is what carries the physics
from lab cadence to clinic. **Now built and measured - see Section 10.6.**

### 10.5 Center on the per-file rest zero before counting swaps **[measured, 2026-07-22]**

The raw swap rule counts crossings of `L_ang - R_ang` past +/-`SWAP_DELTA_DEG`, silently assuming the
interleg signal is zero-centered. It is not: some files carry a large per-subject DC offset on `L-R`
(the `ang_LR_offset` zeroing bias the S2 champion also drops, Section 4.6). When the offset exceeds the
swing's negative reach, `L-R` never commits past -delta, `swap_count` reads 0, and rhythmic antiphase
gait is called STANDING (worst case in the corpus: a +11deg offset made the rule miss ~all WALK windows
on two trials).

**Fix: subtract the per-file rest zero (Section 10.2) before counting swaps** - `anchors.rest_offset()`,
the median of `L-R` over the opening rest. Corpus-wide (non-lockbox) this is a **Pareto gain: walk-recall
0.626 -> 0.691, stand-recall 0.903 -> 0.927**, both up. Only the swap count is recentred; the posture
descriptors (`interleg_offset`, `ileg_minhalf`) stay on the raw signal - the offset *is* the posture.
Per-window centering is **wrong** (it removes within-file posture, so sway/split-stance false-fire as
walking; stand-recall collapses to 0.798). Label-free, lockbox untouched.

### 10.5.1 Verify the rest zero is actually rest **[measured, 2026-07-22]**

The opening span is not always rest - some files open mid-gait, where a single 3 s swing window is a
poisoned zero (per-window median wanders +/-4-5deg). The whole-file median, averaged over many strides,
is stable. Guard: `anchors.rest_anchor()` returns `(offset, trusted)` -
1. trust the opening span only if the **swap rule's own STANDING verdict** holds on it (0 swaps,
   locally centered) - reusing the validated rule keeps it parameter-free (common case);
2. else the **stillest true-rest span** anywhere (smallest-`ptp` 0-swap span, scanned within-segment);
3. else **whole-file median, `trusted=False`** - best available, flagged as resting on the symmetry
   assumption, not a measured rest.

The flag rides every row as `rest_offset_trusted` and annotates the plot, so the S3 agent discounts
those verdicts. **Correctness fix with a null metric impact on this corpus** (the offset trap only bites
when the offset is comparable to the swing amplitude, and the mid-gait files here have large swings) -
its value is latent insurance plus honesty, not a recall gain. Label-free, lockbox-safe.

### 10.6 Stride-adaptive window - no fixed window calls slow gait and standing both **[measured, 2026-07-22]**

The swap rule needs >=2 strides in a window; once the stride period nears the window length it holds <2
and under-calls walking. The deployment population walks slowly (measured strides to ~2.9 s / 0.33 Hz),
so this is central, not an edge case. The tradeoff is fundamental (rest-centered rule, non-lockbox):

| window | walk-recall | stand-recall |
|---|---|---|
| 2 s | 0.691 | 0.927 |
| 4 s | 0.946 | 0.780 |
| 6 s | 0.983 | 0.738 |

Longer windows see more strides (walk up) but absorb transitions/sway (stand down); no fixed window
wins both. **Fix: size the swap window per output cell to ~2 detected strides.**
`anchors.stride_period()` reads the local period from the interleg autocorrelation - the first local max
*after* the acf dips below zero (a plain argmax grabs the lag-0 shoulder). Periodic -> window = 2xperiod
(capped 6 s); non-periodic (no peak above `ADAPTIVE_PERIODICITY_FLOOR = 0.35`) -> base 2 s, so a standing
cell is never lengthened. Result on the 2 s grid: **walk-recall 0.855, stand-recall 0.915** - most of
the long-window walk gain for almost none of the stand cost. Emitted as `swap_verdict_adaptive` +
`swap_window_s` alongside the untouched Section 10 `swap_verdict`. Lab-validated, **not** deployment-solved.

### 10.7 Span-grow fallback for slow/large gait the detector still misses **[measured, 2026-07-22]**

`stride_period()` itself fails on some slow gait (returns `None`, leaving the cell at base 2 s with
`swap_count == 1` - one hump, though the interleg swings a median 40deg: a span problem, not a signal
absence). **Fix: when the adaptive verdict is not WALKING, re-count over the full max span and upgrade
only if it genuinely alternates** - `swap_count >= 2` over 6 s, *both* halves past
`GROW_MIN_PTP_DEG = 8deg`, *and* raw legs anti-correlated past `GROW_MIN_ANTIPHASE = 0.5`. The antiphase
gate is load-bearing: two standing weight-shifts also give swap_count 2 over 6 s, but their legs are not
antiphase (the Section 10.2/Section 11 discriminator at span level). Result on the 2 s grid:

| variant | walk-recall | stand-recall |
|---|---|---|
| Section 10.6 adaptive | 0.855 | 0.915 |
| + Section 10.7 grow-fallback | **0.921** | 0.868 |

A ~10:1 trade (+~268 WALK recovered vs +27 pure-STAND false-WALK, all routed to the disagreement audit).
`adaptive_swap_at` keeps the untouched Section 10.6 behaviour for callers passing only the interleg signal.
Lab-validated, not deployment-solved.

### 10.8 `gait_hz` dropped - degenerate at the 2 s window **[measured, 2026-07-22]**

`gait_hz` (dominant gait-band FFT peak) was a fifth anchor. At a 2 s window the FFT resolution is 0.5 Hz,
above the gait band's 0.13 Hz low edge, so it pinned at 0.5 Hz for >80% of windows (walk-vs-stand AUC
**0.487** - no information) and its rate-audit "invariant" verdict was a **degenerate pass** (both rates
collapse to the same bin - a Section 11.1 trap inside the anchor set). Removed entirely, not demoted. The
reintroduction path is to recompute cadence on the Section 10.7 grown span (up to 6 s -> 0.167 Hz
resolution) where it would carry real information; not done speculatively. `GAIT_BAND_HZ` stays (used by
`stride_period`/`_periodicity`).

### 10.9 Stride-adaptive periodicity - the fixed-window anchor is blind to slow gait **[added 2026-07-24, not yet corpus-validated]**

The `periodicity` anchor needs >=2 cycles in-window (`hi = n // 2`), so at the fixed 2 s / 200-sample
window it only resolves rhythm at >=1 Hz (stride period <=1 s). But the deployment population strides to
~2.9 s / 0.33 Hz (Section 10.6), which cannot fit twice in 2 s. With <2 cycles in-window the `max` over
`[lo, n/2]` lands on the short-lag autocorrelation *shoulder*, not a resolved stride peak (unlike
`stride_period`, `_periodicity` does not skip the shoulder) - so for slow gait the anchor reports a middling
smoothness proxy that does **not** separate rhythmic slow gait from a non-rhythmic smooth sweep. It is not
near-zero, it is *uninformative*: on a synthetic 0.4 Hz stride vs a linear ramp in a 2 s window it reads
**0.53 vs 0.51** (indistinguishable).
**Fix (mirrors Section 10.6): also measure periodicity over the same stride-adaptive span the swap count uses**
(up to the 6 s cap -> resolves down to ~0.33 Hz strides; on the same stride-vs-ramp pair, with 8 s of
context the adaptive value separates them **0.65 vs 0.51**). Emitted as `periodicity_adaptive` alongside the
**untouched** fixed-window `periodicity` anchor: the rate audit and the S4 new-class profile keep reading
the fixed-window value, whose **invariant** verdict is measured (Section 10.4 / audit table) - so this change
adds a descriptor and disturbs no validated number. For standing / non-periodic cells the adaptive span is
the base window, so `periodicity_adaptive == periodicity` there; it only widens for periodic cells.
Code-side (`_adaptive_cell`, single span shared with the swap count); **not yet measured on the corpus** -
the walk/stand separability of `periodicity_adaptive` vs the fixed anchor, and its own rate-invariance,
still need a run before any claim rests on it.

---

## 11. Methodology warnings **[measured - each was hit in practice]**

1. **Density needs mass.** Mode-finding declared files "unimodal - one behaviour" when a 5 s stand
   was 0.4% of the file. Hit **twice**, the second time one message after invoking it as a lesson.
   *A stop is not a mode; it is a stretch of time.* This is also what broke Section 4.1 (whole-file p99 on
   a 93.7%-walking file).
2. **Windows must not straddle edges.** A "32x cadence-invariant" claim measured whole phases, not
   per-window. A bout analysis deleted UNKNOWNs *then* computed runs, silently merging across gaps.
3. **Index != time.** See Section 7.
4. **The eyeball is not truth.** Claude called a 6.0 s standing bout "blurred, should be 2.5 s"; the
   labels said **6,246 ms**. Same error as Section 3.1 - measuring the static plateau and treating it as
   ground truth.

### 11.1 Tests that were themselves broken **[measured]**

- `cluster_stability` scores a **continuum at 0.974** vs real clusters at 1.000. It measures whether
  k-means cuts repeatably (a gradient does, deterministically), not whether there is anything to cut.
- An anchor-independence check became **tautological** once the anchor was defined by the descriptor
  it was tested against.
- A 0.994 check was **near-trivial**: predicting "high motion" from other motion channels.
- **GMM + BIC counts Gaussians, not modes.** It selects k=8 with evenly-spaced means on a 2-state
  signal. A bimodal density with skewed modes needs >2 Gaussians and still has exactly 2 states.

### 11.2 Retracted claims **[measured]**

- **"Walking with no rhythm" does not exist.** Those windows are identical to normal walking on every
  descriptor except a `periodicity` measure that fails at **cadence changes**, not arrhythmia. They
  occur at ~69 s and ~196 s in *both* trials independently - a protocol event, probably a turn.
  Claude invented a category to explain its own artifact and nearly asked for it to be defined.
- **`gyro_energy` bimodality (modes 22 / 61) was two subjects**, unimodal within each (one at 43,
  the other at 64) - not two behaviours.
- **`0.605` (HI/p75 ratio) was `30/49.6` from one file counted twice.** revG's true ratio is 0.426.

### 11.3 Agent proposes, test disposes **[decided]**

> *"Every rule asserted was wrong; every rule the tests checked survived or died honestly."*

**Agent = rule discovery, not per-window labeling.** Reading graphs found slow walking and split
stance that no descriptor encoded. The swap rule came from asking *what walking is*, not from
fitting. This belongs in the S3 agent's system prompt verbatim.

### 11.4 A metric that moves does not confirm the mechanism claimed for it **[measured, 2026-07-20 - hit twice]**

**The S2 gate measures the number, and only the number. It does not test the story told about it.**
Both S2 experiments that carried a mechanistic hypothesis had that hypothesis refuted while the
headline metric behaved defensibly:

| experiment | predicted | measured |
|---|---|---|
| `drop_absolute_angle` | revI improves, `steady_confusion` falls | revI **fell** 0.6105 -> 0.5216 and `steady_confusion` **rose** 0.888 -> 0.911 - both backwards; rejected on margin anyway |
| `drop_offset_only` | `steady_confusion` falls (it was 88.8% of error rows) | `steady_confusion` **flat** at 0.888 -> 0.884, while macro-F1 rose +0.0132; **promoted** |

`drop_offset_only` was right that `ang_LR_offset` is dead weight - Section 10.1 measures its descriptor at
AUC 0.50 - but wrong about *why removing it helps*. The gate correctly promoted it; a reader who
took the rationale at face value would have learned something false about where our errors come
from.

**Rules that follow:**
1. **A promotion licenses the spec, not its rationale.** The ledger entry is evidence that the
   configuration scores better. It is not evidence for the causal story attached to it.
2. **Never promote an agent's rationale into this file on the strength of its macro-F1.** A finding
   enters DOMAIN_NOTES only when something measured *that specific claim*.
3. **If a proposal names a mechanism, check that mechanism in the result** - the taxonomy fractions
   are already recorded per experiment, so this is free. A hypothesis that predicts a bucket should
   be scored against that bucket, whatever the headline does.
4. This is Section 11.3 pointed at ourselves: the *test* disposed correctly both times. The prose is what
   went unchecked, because nothing in the loop reads it.

---

## 12. Fusion - the confidence signal is agreement, not either model's own confidence **[measured, 2026-07-22]**

S4 combines the S2 learned label and the S3 physics verdict into one call **and a confidence** -
the signal the incumbent lacks and the reason the project exists. Every rule below is read off the
measured S2xS3 contingency (leave-one-rev-out out-of-fold S2 intersect adaptive swap verdict, non-lockbox,
4,812 windows), not assumed.

| S2 pred | S3 verdict | n | true STAND | true WALK | |
|---|---|---|---|---|---|
| STAND | STANDING | 652 | **93%** | 7% | agree -> STAND |
| WALK | WALKING | 3326 | 1% | **99%** | agree -> WALK |
| STAND | AMBIGUOUS | 76 | 74% | 26% | keep S2 |
| WALK | AMBIGUOUS | 419 | 9% | 91% | keep S2 |
| STAND | WALKING | 142 | 22% | **78%** | disagree |
| WALK | STANDING | 158 | 21% | **79%** | disagree |

**What the contingency dictates:**
- **Agreement is 98% correct -> HIGH confidence.** Disagreement is where the risk concentrates.
- **S3's WALKING verdict is a reliable WALK signal** (78-99%); its STANDING verdict is **not** -
  both disagreement cells run ~79% WALK, because S3 under-calls slow gait as "standing". So the
  fused label is **STAND only when S2 says STAND *and* S3 does not say WALKING; every other cell is
  WALK.** Physics vetoes toward WALK; its STANDING call is ignored against S2.
- **Two intuitive rules were measured and REJECTED.** (1) "Trust physics on standing" (override
  S2=WALK when S3=STANDING) scores macro-F1 0.894 - *below* S2 alone (0.898) - because that cell is
  79% WALK. (2) "Take whichever model is more confident" fails because S2 is **confident-wrong
  exactly in disagreement**: when S3 says STANDING, S2's median probability is 0.97 but its accuracy
  is 0.90. Self-reported confidence cannot see its own overconfidence; the independent physics view
  can. So the tier is set by **agreement**, never by whose confidence is louder - and agreement also
  beats an S2-probability abstention at matched coverage (keep-agree 0.833/0.979 vs proba>=0.8
  0.840/0.973).

**Result.** Fused macro-F1 **0.921** (S2 alone 0.898). The confidence is calibrated and monotonic:
HIGH 0.83 share / **0.98** acc; MED 0.10 / 0.88; LOW 0.06 / 0.79. A controller that **abstains on
LOW** (disagreement) covers **94%** of windows at accuracy **0.969**, macro-F1 **0.943**, and
stand-recall **0.897** - *above* S2's 0.865. **Standing is recovered by abstention, not by a
cleverer label**: the fused label alone drops stand-recall to 0.826, and the abstention brings it
back by declining exactly the windows where standing was being confused. That LOW/abstain output is
a **machine `-1`** - the same "looked and couldn't call it" the human `-1` marks (Section 5.2), reproduced
from model disagreement. Measured train+val only; the lockbox opens once, at the end.

The unrecoverable standing (true STAND that both models call WALK) has **low `grav_stab`** - it does
not look like standing to the physics either, i.e. it is motion-contaminated "standing" (ramps,
weight-shifts), a **label** problem (S3 hypothesis `stand_labeled_bouts_are_dynamic_repositioning`),
not a feature gap. No anchor separates it because there is nothing physical to separate.

### 12.1 Brief stops are a window-resolution floor, absorbed by the gate **[measured, 2026-07-22]**

The first live `s4_fusion` agent validated the fuser and flagged a real weakness: brief (<2 s) STAND
pauses inside walking bouts get labeled WALK. Its *mechanism* was wrong (it blamed the adaptive window;
those windows sit at the base 2 s, and the adaptive rule only widens) - Section 11.4 again, on the fusion
agent. The real cause is the opposite of slow gait: a 2 s window is too *coarse* to isolate a sub-2 s
stop, straddling the neighbouring strides and counting them. **Handled, not fixed:** all such
truly-STAND windows are LOW-confidence, so the confidence-gated deployable abstains on every one, and
the lockbox confirmed it generalizes (LOW-tier accuracy 0.27 on the unseen rev, Section 12.5). A shorter
window trades resolution for noise pipeline-wide and cannot be validated now (lockbox spent).

### 12.2 The ceiling is data; the fusion directs the data effort **[measured, 2026-07-22]**

Row-level, the fusion cuts the hazardous bucket (`steady_confusion` 0.823 -> 0.785, row accuracy 0.925
-> 0.942) but does not solve it: the residual is **label contamination** - the missed standing has low
`grav_stab`, so it does not look like standing to the physics either (motion-contaminated "standing").
That is a **ground-truth ceiling**, so the dominant remaining lever is *data* (better labels, more
subjects), not more macro-F1 in code. `stages/s4_fusion/curate.py` turns the model into a
data-collection director, routing each flagged window:

| route | signal | meaning |
|---|---|---|
| `relabel_candidate` | **both** models contradict the label (STAND labeled, S2=WALK *and* swap=WALKING) | strong mislabel *candidate* - two independent methods rarely share an error |
| `new_class_candidate` | STAND labeled but moving (`grav_stab` < 0.5) | structure the {stand,walk} taxonomy misses (dynamic repositioning) |
| `collect_more` | single-model disagreement / slow-cadence under-call | a physics error or under-served condition |

**Non-negotiable: code never auto-fixes a label.** The "physics contradicts the label 100% of the time,
just flip it" shortcut is a *trap* - that 1.00 is tautological (the route is defined as disagreement and
the fused label follows physics), and "physics contradicts label" can be a mislabel, a **physics error**
(slow-gait/brief-stop, Section 10.6/Section 12.1), or a transition. Requiring **both** independent models to
contradict the label is the strongest signal, but still only a *candidate a human confirms*.

**New classes are governed, not invented (the Section 11.2 trap).** The taxonomy is too coarse (18% of
"standing" is non-quiet), but the intuitive static "sit" is absent - the excess is *moving* standing. So
`agents/s4_newclass.py` proposes over `new_class_candidate` spans and `validate_proposal` gates on
**provenance** (every `span_ref` overlaps a real candidate span) AND **cluster mass** (>= 4 spans across
>= 2 revs) before `supported`, stamping `needs_human` on every proposal regardless. Read-only, orthogonal
to the deployed model/rule/fuser/labels. Rest-relative leg angle (removes the per-subject zeroing bias,
Section 4.6) is the honest feature for this, not raw absolute angle.

The lockbox eval (`stages/s4_fusion/lockbox.py`) scores the **fused** model and needs a deliberate open
(`--confirm OPEN-LOCKBOX`); the new-class agent changes nothing in the model, so either order is safe.
Only *retraining on new labels* would require a fresh sealed rev.

### 12.3 Anchors belong in the S4 policy, not the S2 feature set **[measured, 2026-07-22]**

Pulling validated anchors straight into S2's features (**early fusion**) was tested against leaving them
in the S4 swap policy (**late fusion**). The two eligible anchors (rate-invariant, not already an S2
proxy) both failed the gate: `grav_stab` **+0.0006** (below the 0.005 margin, and 0.72-collinear with
`std` features already present - it is a monotone transform of them), `periodicity` **-0.0056**
(regresses, Section 11.2: it moves at cadence *changes*, not arrhythmia). **Rule: an anchor belongs in
the S4 policy, not the S2 feature set, unless it is rate-invariant AND carries information no existing
feature does - measure collinearity first (the Section 4.6 discipline).** Section 11.4 in the other
direction: the gate refused a good-sounding change.

### 12.4 S4 does not special-case an untrusted rest zero **[decided, 2026-07-22]**

S3 exposes `rest_offset_trusted` (Section 10.5.1). The S4 policy does **not** condition on it: measured,
the untrusted-rest windows are classified identically either way (their large swings commit past
+/-`delta` regardless of center error), so a trust-conditioned rule would discriminate on noise for zero
gain (Section 11.1). The flag stays an **agent-facing** annotation, not a fusion input. Revisit only if a
future file is untrusted **and** small-swing.

### 12.5 Lockbox OPENED - the confidence signal generalized, the fused label did not **[measured, 2026-07-22, FINAL - revG spent]**

`revG` (the last sealed rev; revH already spent on the axis decision) was opened once. This is the final
unbiased number and **cannot be re-run**.

| model (revG, 198 windows) | macro-F1 | accuracy | stand-recall | walk-recall |
|---|---|---|---|---|
| S2 alone | 0.822 | 0.874 | 0.635 | 0.959 |
| fused (raw label) | 0.798 | 0.864 | **0.558** | 0.973 |
| **acting-on-confidence** (abstain LOW) | **0.845** | **0.898** | **0.659** | 0.972 |

Tier calibration held out-of-sample: HIGH 0.92 / MED 0.58 / **LOW 0.27**. Across all revs the gated
deployable ties-or-beats S2 (full per-rev breakdown in the run artifacts).

**Three findings:**
1. **The raw fused label did NOT beat S2 alone out-of-sample** (revG 0.798 vs 0.822; revH a no-op) - the
   S3 WALKING-veto flipped ~4 true stands to walk, exactly the risk Section 12 flagged. On 198 windows this
   *fails to confirm* fusion; it does not decisively refute it, but the burden was on fusion.
2. **The confidence signal - the reason this project exists - DID generalize.** Abstaining on LOW beat
   both S2-alone and full-coverage fusion (macro-F1 0.845, stand-recall 0.659, 94% coverage) with a
   calibrated LOW tier that concentrates errors. **The deployable artifact is the confidence-GATED
   system, not the raw fused relabel** - standing is recovered by abstention, not a cleverer label.
3. **The generalization gap is stand-recall on unseen subjects** (0.56/0.38 held out vs ~0.90 LORO;
   walk-recall stays 0.96-1.0) - the Section 12.2 label-contamination data ceiling, not a code defect. n=2
   held-out subjects: wide error bars, read alongside the LORO number, not instead of it.

**What this closes:** the 2-class fused model is final at these numbers. Any improvement now produces a
*different* model needing a *new* sealed rev - there is none left in this corpus. Do not tune against
revG/revH; they are spent.

---

## Changelog

*Moved to BUILDLOG Appendix B - the dated development timeline. An agent deciding now does not need it; see BUILDLOG for how each section here was arrived at.*
