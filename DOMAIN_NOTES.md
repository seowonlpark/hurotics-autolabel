# DOMAIN NOTES — H-CARE IMU locomotion pipeline

**This file is injected into every agent's system prompt. It is the pipeline's institutional memory.**

Rules for this file:
- Every entry is a *finding with a reason*, not an instruction without justification.
- Agents may propose additions (via their rationale logs). Only a human commits them.
- If you are an agent reading this: these facts are established. Do not re-derive them, do not
  contradict them silently. If your evidence contradicts an entry, say so explicitly and flag
  `needs_human` rather than acting on it.
- **[measured]** = derived from the real corpus, reproducible by rerunning S1.
  **[reported]** = came from a human, not independently verified.
  **[decided]** = a design choice, not a fact.
  **[open]** = known unknown.

---

## 0. The corpus

**[measured]** `data/raw/` holds one family (`raw_device`) of device logs across many dated session
folders (2025-10 onward). **This file records findings, not tallies** — live counts (files, folders,
clean vs quarantined, usable minutes, rate mix) regenerate every run in `runs/*/clean_report.md` and
`census.md`; read those for current numbers. A 2026-05 export batch is quarantined for a degenerate
time base (§2.6).

Session date comes from the folder name. **Subject is filename field 2** (§5.5) — the
cross-validation group key, *not* session. One subject (`69`) dominates the corpus; the others
appear in only one or two sessions each.

Labeled data lives in `data/labeled/`, never `data/raw/`. A `Label` column appearing under
`data/raw/` is a contamination event, not a variant: quarantine and flag it.

---

## 1. Schema

### 1.1 Multiple header variants, one family **[measured]**

The corpus carries several distinct header shapes (differing column widths), all one `raw_device`
family. They **differ only in the tail**; the head is near-stable and the variants track firmware
eras. The current variant list — IDs, column widths, per-variant file counts, eras — regenerates
each run in `census.md` / `variants.json`; that is the source of truth, not a table here.

### 1.2 The contract is 45 columns **[measured]**

`Time` through `Total Gait Length` is identical across **every** variant. Everything past index 44
is variant-specific. The 45-column prefix is the real contract — the wider widths are not.

### 1.3 The numeric prefix is a LIE — resolve by name, never by position **[measured]**

The `NN_` prefix is a per-file position, not a stable identifier:

- `Step` sits at index **46 or 47** depending on variant
- `Cadence` sits at index **45 or 46**
- `loco` sits at index 47 in most files — but in one header variant (`0fda484e`), **index 47 is
  `Step`**

`df.iloc[:, 47]` therefore blends a step counter into a locomotion state, on the majority of files,
and never raises. The column that *looks* positionally stable (`loco`) is precisely the trap.

Note both are **outdated columns anyway**: `loco` is the legacy rule-based algorithm's output (the
thing this project replaces — severed, §4.5) and `Step` is a firmware-computed counter. Neither is a
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
| schema | `fb5ea2c2` / `0fda484e` / `4bfd6ab2` | `e5f2660f` / `86069795` |

Rate, schema and date change together. Resampling removes the *rate* difference; it does not
remove the era. A model can still learn "era" as a shortcut. Group by **subject** (§5.5) for
cross-validation to defend subject leakage — but note era is a *separate* confound: subject `69`
spans both eras, so subject grouping does not neutralize era-specific firmware artifacts.

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

### 2.3 This reframes the id=69 rate experiment **[measured, unconfirmed against source]**

id=69 reported clustering partitioning by "acquisition rate" across 99.4 / 99.7 / 100.0 / 500.0 Hz
— exactly the tier ladder above. The geometry was likely reading **timestamp counter granularity**,
not acquisition rate. Still a confound, still disqualifying for `gyro_energy` as a body-defined
anchor, but the mechanism differs from what was assumed. Not yet confirmed against the id=69 source.

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

The alias lands at |120−100| = 20 Hz, at full strength, indistinguishable from real signal.
Use `scipy.signal.decimate(x, 5, ftype='fir')`.

### 2.6 A 2026-05 batch has a corrupted `Time` column — quarantined **[measured]**
A few files from two 2026-05 sessions (subject `70` on 2026-05-15, some subject-`100` files on
2026-05-19) have a **destroyed `Time` column**: it collapses to a small, non-monotonic range
(median `dt` = 0, a large fraction of `dt` negative). No forward cadence exists, and `np.interp`
would silently corrupt the resample.

**It is a per-file `Time` corruption, not a bad variant and not clock jitter.** Healthy sibling
files of the *same* variant (`e5f2660f`) and era carry a clean monotonic 500 Hz clock — so the
acquisition is fine; only these files' timestamps were overwritten. The *data* channels look
intact and in row order (row-to-row continuity matches a healthy file). There is **no recoverable
surrogate clock** in the file (the large monotone counter some carry is a normal column, present in
the healthy files too).

**Decision (2026-07-20, Lu): do NOT reconstruct a synthetic clock.** A uniform-rate clock would
*fabricate* unmeasured time, which the pipeline refuses to do. These files stay quarantined as
`needs_human` (re-export from the source is the only honest fix). The non-corrupted files proceed
downstream normally — `data/clean/` only ever holds files that passed, so later stages never see
the corrupted ones.

S1 catches this up front (`clean_one` guards `median(dt) > 0`; `measure_hz` returns `nan` rather
than dividing by zero) and routes it to the `quarantine.jsonl` ledger with reason
`degenerate time base`. The raw file is left in place, never moved.

---

## 3. Gaps and segments

### 3.1 The segment is the unit of analysis, not the file **[measured]**

A file is a bag of continuous runs; the **segment**, not the file, is the unit of analysis (usable
segment count and minutes are in `clean_report.md`). Windows must never straddle a gap; resampling
across one invents data that was never measured.

### 3.2 There is a ~10-sample startup burst **[measured]**

Many files open with a segment of **exactly 10 rows**, then a gap, then the real trial. The 500 Hz
era shows two tiny leading segments (2–8 rows each). Mechanical, not random.
`MIN_SEGMENT_SAMPLES = 100` currently trims it as a side effect of a size filter — the right
outcome for an incidental reason.

### 3.3 Gap position is otherwise unpredictable **[reported + measured]**

Apart from the startup burst, gaps land anywhere. `00038_69_2026_1_8_15_11_0.csv` shatters into 6
segments, 4 unusable. The ~1264 ms gap in nine 500 Hz files is systematic (all within 14 ms of each
other) but its cause is **unknown** — suspected to be an error **[reported]**. Policy: segment
conservatively, never interpolate across, flag the pattern.

---

## 4. Channel trust

### 4.1 Angular velocity is RELIABLE — it is the derivative **[measured]**

`angvel_LPF` **is** `d(angle)/dt`: r = **0.999**, slope **0.98**. Use it directly. There is no need
to re-derive velocity from the angle channels.

**RETRACTED (2026-07-16):** an earlier entry claimed `L/R_angvel_LPF` "ring at ±40–80 deg/s during
static postures" and instructed gradient-derivation instead. That claim was **[reported]**, never
verified, and was then "confirmed" by measuring |angvel| p99 ≈ 105 deg/s on
`annotated_loco_rev2_trial_1.csv` — **computed over the whole file, 93.7% of which is walking.**
That measured walking and called it rest. The channel was never noisy; the measurement was wrong.

**Lesson, kept deliberately:** a `[measured]` tag is only worth what the measurement isolated.
Statistics over a whole file say nothing about a state that occupies 5% of it. See §11 item 1:
*density needs mass.*

### 4.1b Gyro units are inconsistent WITHIN a single file **[measured — confirmed across the clean corpus]**

- **`B_Gyro_*` is rad/s. `L_Gyro_*` / `R_Gyro_*` is deg/s.** Same naming convention, same file.
  Any feature mixing trunk and thigh gyro without conversion is off by **57.3×**. The clean-layer
  trust check reproduces this from the sagittal regression slope: **0.98** on L/R (deg/s) vs
  **0.017** (≈1/57.3) on B (rad/s).
- **Axis: the Deg↔Gyro name crossing is DEVICE-WIDE, not a trunk defect
  [measured, 2026-07-20 — corrects an earlier trunk-only framing].** `d(Deg_Y)/dt` tracks `Gyro_Z`
  on **every** side, not just B. Over the clean corpus, `argmax|r|` of `corr(d(Deg_Y)/dt, Gyro_axis)`
  on files confident enough to answer (|r| ≥ 0.9):

  | side | files that answer | axis picked | median \|r\| vs `Gyro_Y` | vs `Gyro_Z` |
  |---|---|---|---|---|
  | L | 62 | **Z, unanimously** | 0.161 | **0.986** |
  | R | 65 | **Z, unanimously** | 0.147 | **0.987** |
  | B | 47 | Z ×45, Y ×2 | 0.316 | **0.961** |

  So there is no "the legs are fine, the trunk is swapped." **`L_Gyro_Y` is no more sagittal than
  `B_Gyro_Y` is** — the column *name* and the physical axis disagree on all three sides identically.
  Treat this as a naming convention of the device, not a bug in one IMU.

  **Consequence for anyone reading the parquet: `*_Gyro_Y` is NOT the sagittal rate.** The channel
  that matches `*_Deg_Y` (and hence `*_ang_LPF`) is `*_Gyro_Z`. Resolve it from the file's
  `channel_trust.json` (`sides.<side>.sagittal_gyro_axis`), never from the column name.

  Still **not universal**, which is why it is detected and not tabled: two files
  (`00001_69_…1_14_10_4_0`, `00038_69_…1_15_11_28`) map `B_Deg_Y → B_Gyro_Y` at |r| ≈ 0.96–0.99 (one
  sign-flipped). A blanket Y↔Z swap would corrupt exactly those — and, per the table above, would
  have to be applied to L and R too, which the earlier trunk-only framing would have missed.

- **It is a full PERMUTATION, and it is variant-invariant [measured, 2026-07-20].** Extending the
  regression to every Deg axis (not just `Deg_Y`) gives the same map on every variant that answers:

  | | `Deg_X` → | `Deg_Y` → | `Deg_Z` → |
  |---|---|---|---|
  | `fb5ea2c2` | X ×68 | Z ×102 | Y ×70 |
  | `0fda484e` | X ×15 | Z ×20 | Y ×10 |
  | `4bfd6ab2` | X ×4 | Z ×5 | Y ×3 |

  **X→X, Y→Z, Z→Y**, identically, with no firmware-era dependence. `DOCUMENTED_GYRO_PERMUTATION`.

- **The permutation is NOT sagittality, and conflating them was a real bug [measured, 2026-07-20].**
  This map says which gyro axis measures which angle axis — a correspondence *internal* to a file,
  measurable because `Deg` is the reference. It does **not** say which axis is the sagittal plane;
  that is a hardware-revision fact with no in-file signature (§6.2 measured every signal-only rule
  for it at below chance) and lives in `SAGITTAL_AXIS_BY_VARIANT`.

  `channel_trust.json` used to publish a field called **`sagittal_gyro_axis`**, which actually held
  *"the gyro axis matching `Deg_Y`"*. On `fb5ea2c2` — **the majority variant** — sagittal is `Deg_X`,
  so the field read `Z` and was wrong. Never a data-path bug (`transform.py` uses the variant
  lookup and hard-fails on unknown variants), but the S1 exception agent triaged against it.
  Replaced by `gyro_axis_by_deg_axis` + `conflicts_with_documented`. **§6.2's table and this
  permutation never disagreed** — every entry there obeys X→X / Y→Z, so its gyro column is
  derivable from its Deg column.

- **Apply the r-floor PER AXIS, not per side [measured, 2026-07-20 — 11.1 one level down].**
  First cut of the permutation detector scored the side once and then reported all three axes:
  **25 anomalies**, nearly all of the form `{X→X, Y→Z, Z→X}` — the two real axes right, `Deg_Z`
  garbage. `Deg_Z` is the yaw-like axis that *drifts rather than oscillates* (§4.2), so
  `d(Deg_Z)/dt` is often noise even mid-walk, and its argmax is meaningless. Flooring per axis, so a
  silent axis abstains instead of dissenting, returns exactly the **2** hand-named anomalies above.
  *Density needs mass applies to each axis separately, not to the file.*
- `Deg_Y` needs **no sign normalization** — raw L vs R is already anti-phase in 84% of files.

`Deg_Y` is the sagittal (flexion) channel. This is the channel the swap rule reads.

**RESOLVED (2026-07-20):** unit normalization now lives in the clean layer
(`stages/s1_clean/channel_trust.py`). Every gyro channel is normalized to **deg/s**; the sagittal
axis + unit are **detected per file** (regress `d(Deg_Y)/dt` against each gyro axis, slope → unit,
argmax|r| → axis) and written to a per-file `channel_trust.json`. Static files, whose derivative
carries no signal, **abstain** and fall back to the documented convention (r-floor 0.9), recording
that they did (11.1, density needs mass). Axes are **recorded, not reordered** — no silent mutation.
Every clean file's B-side is normalized rad/s→deg/s; static sides abstain; a couple of files show a
confident `B_Deg_Y→B_Gyro_Y` axis anomaly (named above). Per-run rollups live in `clean_report.md`
and the per-file `channel_trust.json`.

**Verified on the clean output [measured, 2026-07-20]:** re-regressing `Gyro_Z` on `d(Deg_Y)/dt`
*after* cleaning gives a median slope of **0.987 / 0.984 / 0.986** for L / R / B. The 57.3× is gone —
B reads deg/s in the parquet like everything else. Note what this check does and does not cover: it
confirms the **unit** fix landed, and it re-confirms the axis crossing (it had to be run against
`Gyro_Z`, not `Gyro_Y`, to produce a slope near 1 at all).

### 4.2 Yaw is drift-contaminated — but per-file, not wholesale **[measured]**
Original **[reported]**: in treadmill data yaw correlated with session time at r ≈ −0.95, measuring
elapsed time not orientation.

Measured on the raw corpus (clean-layer drift test, `|corr(Deg, Time)|` duration-weighted over
segments ≥ 5 s): **`Deg_Z` is the yaw-like axis** on every side — median |r| ≈ 0.33 vs the sagittal
`Deg_Y` at ≈ 0.08. But the strong drift signature is **file-specific, not universal**: only a
minority of files cross |r| ≥ 0.9, and always on `*_Deg_Z` (the flagged channels each run are in
`clean_report.md` / `channel_trust.json`). So yaw is **flagged per
channel per file** in `channel_trust.json` (`drift` section), **not dropped wholesale**. It is a
feature-time exclusion signal; the raw superset is kept. Any yaw-derived feature must consult the
per-file drift flag.

### 4.3 The trunk (B) IMU was dropped from rev2 for a real reason **[reported]**
Belly/trunk placement was inconsistent between subjects, and may have been treadmill-mounted in some
trials. The raw family still carries `B_Deg_*` / `B_Gyro_*` / `B_Acc_*`; treat trunk channels as
suspect until placement consistency is established per session.

**This flag is about PLACEMENT only.** It is not an axis-labeling flag and it is not a unit flag:
the Deg↔Gyro name crossing is device-wide (§4.1b), and the rad/s units are fixed in the clean layer.
The two `B_Deg_Y→B_Gyro_Y` files are an axis anomaly, tracked in §4.1b, not evidence of bad trunk
placement. Keep the two risks separate — conflating them makes the trunk look doubly untrustworthy
and lets the L/R axis crossing hide.

### 4.4 `Time` is metadata, not a feature **[decided]**
Used for dt / rate / segmentation only. Never fed to a model.

### 4.5 `loco` is severed and stays severed **[measured]**
Its "standing" class contains a decile **as periodic at the gait frequency as median walking**. A
label that fails inspection cannot validate anything. (It is also an outdated algorithm's output
**[reported]**, but that is the weaker reason — the strong one is that it is demonstrably wrong.)

Not ground truth, not a feature. Must be dropped **by name** — in `0fda484e` files, dropping index
47 would delete the step counter instead.

### 4.6 `Deg` is a FUSED ESTIMATE, not a transducer reading **[decided, 2026-07-20]**

The measured-vs-computed line in §9 is real but its label is too coarse. An IMU has exactly two
transducers — a rate gyro and an accelerometer. **Angle is not among them.** `L/R/B_Deg_*` is the
sensor's onboard fusion output (gyro integration corrected by the gravity vector, Kalman or
complementary). It is *computed*; it just happens on the sensor die instead of in app-layer firmware.

The honest partition is three-way, not two-way:

| tier | channels | trust |
|---|---|---|
| **transducer** | `*_Gyro_*`, `*_Acc_*`, `L LC` / `R LC` | raw observation |
| **on-sensor fusion** | `*_Deg_*` | computed, but vendor-fixed and firmware-*version*-stable |
| **app-layer compute** | `Cadence`, `Step`, `Stride Length`, `Hip_ROM`, `GCP`, `Hip_Deg_*`, `loco`, admittance / PID state | the firmware's opinion; churns with firmware version |

**This does not change what S1 keeps.** The drop rule targets the third tier, and that is still
correct — tier 3 is what churns column position between variants and bakes in the era confound (§9).
`Deg` stays.

**What it changes is how `Deg` may be described.** Do not call it ground-truth measurement:
- It is the reason **§4.2 yaw drift exists at all.** Drift is what dead-reckoned fusion does with no
  magnetometer to correct heading. §4.2 reports drift as an empirical oddity; this is its mechanism,
  and it predicts the finding: `Deg_Z` (heading, unobservable from gravity) drifts, while `Deg_Y`
  (pitch, continuously corrected by the gravity vector) does not.
- It means `Gyro` is the *more primitive* channel, not the derived one. §4.1's `Gyro == d(Deg)/dt`
  is true, but the causality runs the other way: `Deg` was integrated **from** `Gyro`. That is why
  the correlation is so tight (r = 0.999) — it is near-tautological, not independent corroboration.
- Its fusion parameters are a device property. Two revisions may fuse differently even where every
  column name matches.

---

## 5. Labels

### 5.1 Two unknowns, opposite in kind — never merge them **[reported]**

- **`255` = machine unknown.** Data error. Nothing was measured properly.
- **`-1` = human unknown.** Data is fine; a trained human looked and could not call it.

### 5.2 `-1` is training-poison and evaluation-gold **[decided]**
It is a hand-drawn map of where confidence *should* be low — the exact signal the current
rule-based algorithm lacks. Exclude from training targets (ambiguous label = noisy target); retain
and report separately in evaluation as the natural test set for a confidence signal.

### 5.3 rev2 label encoding **[measured]**
`annotated_loco_rev2_trial_1.csv`: `int64`, no nulls, three codes — `10` (93.7%), `0` (5.6%),
`-1` (0.8%), 12 segments. `-1` appears at most but not all transitions, consistent with its
meaning: absence means the labeler *could* tell.

### 5.4 Class imbalance makes accuracy meaningless **[measured]**
93.7% walking. A "predict 10 always" model scores 93.7%. Macro-F1 is the headline metric.

### 5.5 Subject = filename field 2, and it is KNOWN for every file **[measured + confirmed with Lu, 2026-07-20]**
**Corrected — this reverses the earlier entry.** An earlier version claimed filename field 2 "tracks
the date block (device / firmware / protocol), not a subject," and concluded subject was
unrecoverable (a `needs_human`). **Both are wrong.**

Field 2 is the **subject tag** (Lu, corroborated by the data): its value is constant within every
session-day folder, and one value (`69`) recurs across **many folders spanning many months**. A
date/firmware/protocol block cannot span that long; a person tested repeatedly can.

One subject (`69`) dominates the corpus; the others appear in only one or two sessions each, and at
least one subject has only a degenerate-time-base file (§2.6) so it contributes no usable data. The
`sub1` / `sub2` markers once on `20260114` were **trial batches of subject 69, not two people**
(Lu) — since removed from the filenames; they were trial indices, never wearer ids. (Per-subject
file counts, if needed, come from parsing field 2 across `data/raw/`.)

**Consequences:**
- Cross-validation groups by **field 2 (subject)**. This defends subject leakage fully; session-day
  grouping is subsumed (one subject per day).
- Subject identity is **not** a `needs_human`. **RESOLVED.**
- Real remaining limitation (not a blocker): **few distinct subjects**, one dominant, several tested
  only once or twice. Cross-subject generalization is bounded by subject *count*, not unknown
  identity. State that limit honestly; do not overclaim breadth.

### 5.6 Current corpus class coverage **[reported]**
h-medi contains essentially only STANDING and WALKING. Rare-class separability (stairs, varied
terrain) is **untestable** on current data. Any claim about rare-class performance is overclaiming.

### 5.7 The labeled family IS the training asset; features are rotational for a reason **[decided + reported, 2026-07-20]**
The supervised task is **stand (`0`) vs walk (`10`)**, `-1` excluded from targets (§5.2). The **only**
labeled data is the `data/labeled/rev*/csv/annotated_loco_rev*_trial_*.csv` set — the derived rev2
family (§6.1), whose four features are `L/R_ang_LPF` (low-pass sagittal angle) and `L/R_angvel_LPF`
(low-pass angular velocity). The S1 clean corpus (raw IMU superset) carries **no labels**; the two
never share a file. The open-source gait dataset config.py calls the "primary asset" is **not in the
repo** — until it is, rev* is the training asset.

**Why only those four (rotational) features — not the fuller raw set (Lu):** some trials are walked on
a **treadmill**. Any channel encoding *linear translation* (net displacement / stride velocity /
stride length / total gait length) reads ~0 on a treadmill even while the person is plainly walking,
so it actively misleads a classifier. **Joint angle + angular velocity are rotational** — the limb
swings the same whether or not the ground moves underneath — so they are treadmill-robust. (Those
linear channels are exactly the *computed* columns S1 already prunes; the kept raw `Acc` still holds
oscillatory gait content, but whether to *use* it is a feature-layer call per §9, not a clean-layer
one. See §4.3: trunk placement itself was treadmill-suspect.)

**The raw↔rev\* bridge (why this still classifies "loco state from a CSV"):** the four rev* features
are **derivable from any raw device CSV** — `L_ang_LPF ≈ lowpass(L_Deg_Y)`, `L_angvel_LPF ≈
lowpass(L_Gyro[sagittal])`, both kept by S1. So the model trains on rev* labels and **runs on a fresh
raw CSV** by recomputing the same four features. Feature extraction is the bridge; the deliverable is
CSV → four rotational features → per-window loco state.

---

## 6. Products / families

### 6.1 rev2 is a separate, lossy family **[measured]**
`Time, L_ang_LPF, R_ang_LPF, L_angvel_LPF, R_angvel_LPF, Label` — 6 columns, sharing only `Time`
with the raw device family. It discards the trunk IMU, **all accelerometers** (the gravity
reference, hence all axis/calibration checks), load cells and GCP; and it keeps both
angular-velocity channels — which are **reliable**, being `d(angle)/dt` (§4.1), not the "untrusted"
they were once called.

**Do not make rev2 the storage format.** Canonical storage is the name-resolved raw superset. A
wide honest table can always be projected down; a narrow one can never be recovered.

**The labeled rev\* set is MIXED-RATE across trials [measured, 2026-07-20].** Not one family rate:
by median-dt, `rev13` is 100 Hz, `rev14` ~99.4 Hz, `rev2` `trial_1/2` are 500 Hz while `trial_3/4`
are 200 Hz. (No contradiction with §7's "rev2 samples at 494 Hz" — that is the *span/n* estimate on
one file; median-dt reads 500. The two legitimately disagree, which is exactly why `manifest.py`
reports both.) Consequence: **never assume a rate for this family** — detect per trial and normalize.
`stages/s2_ml/dataset.py` puts every trial on the 100 Hz grid via S1's `resample_file`, so windowing
downstream sees one rate. Dropped in the process: only the §3.2 startup-burst fragments (~0.011% of
rows), and they are counted, not silently discarded.

### 6.2 The raw→rev2 mapping is RESOLVED EXACTLY **[measured, 2026-07-20 — supersedes the earlier approximation]**
Established by reproducing the labeled columns from raw to **~1e-13 (float roundoff)** on **19 paired
recordings** — annotated trials and raw files with identical `Time` vectors and row counts, found by
matching `t[0]`/`t[-1]`/`n`. Source: HUROTICS MATLAB LPF FILES (`LPF.m`, `timestamp.m`, `csv2mat.m`).

**The transform.** First-order *causal* IIR (single pole), applied per channel:

    a     = 2*pi*dt*fc / (2*pi*dt*fc + 1)
    y[0]  = x[0]
    y[n]  = a*x[n] + (1-a)*y[n-1]

with **fc = 1 Hz** for BOTH angle and angular velocity (`csv2mat.m` sets `f_ang = 1; f_angvel = 1;`
commented "for locomotion classification"; `f_angvel = 10` is the GCP variant — do not use it here).
It is **causal, not zero-phase** — it introduces lag. Do NOT substitute `filtfilt`: that would shift
features relative to labels.

**Two earlier claims here were WRONG:**
- `*_angvel_LPF` is **not** `d(*_ang_LPF)/dt`. It is `LPF(Gyro)` — filtered gyro, straight from the
  raw gyro channel (`csv2mat.m` reads angle from cols 2:4/11:13 and angvel from cols 5:7/14:16).
  The two merely *resemble* each other because `Gyro == d(Deg)/dt` (§4.1).
- `*_ang_LPF` is **not** always `LPF(Deg_Y)`. The source axis is **device-revision dependent.**

**The sagittal axis is a DEVICE property, not a signal property — resolve it by schema variant:**

| variant | `*_ang_LPF` ← | `*_angvel_LPF` ← | revs | raw files |
|---|---|---|---|---|
| `fb5ea2c2` | `Deg_X` | `Gyro_X` | rev13, rev14 | 62 |
| `0fda484e` | `Deg_Y` | `Gyro_Z` | rev7, rev8 | 11 |
| `4bfd6ab2` | `Deg_Y` | `Gyro_Z` | rev4 | 5 |

**Every row of this table obeys the §4.1b permutation** (X→X, Y→Z, Z→Y): `fb5ea2c2` pairs `Deg_X`
with `Gyro_X`, the others pair `Deg_Y` with `Gyro_Z`. So the **`*_angvel_LPF` column above is
derivable from the `*_ang_LPF` column** and carries no independent information — the only per-variant
fact here is *which Deg axis the exporter treated as sagittal*.

`fb5ea2c2` is therefore **not** an anomalous *permutation* — it is wired like every other variant.
It is simply a revision whose sagittal plane is `Deg_X`. (An earlier version of this entry called it
"the anomaly family" by pointing at `SAGITTAL_DEG_AXIS`/`DOCUMENTED_SAGITTAL_GYRO_AXIS`, constants
that conflated the permutation with sagittality and have since been **deleted** — see §4.1b.) The
genuine §4.1b anomalies are the two files whose *permutation* breaks, `B_Deg_Y → B_Gyro_Y`; one of
them, `00001_69_…1_14_10_4_0.csv`, happens to be the raw file paired with `rev13_trial_1`, which is
why S1 and the Phase-2 exception agent both flagged it. That coincidence is what made the two
questions look like one.
Coverage: **78 of 97 raw files (80%)**; unmapped are `e5f2660f` (mostly the §2.6 quarantine family)
and `86069795`. Unknown variant ⇒ **abstain**, do not guess.

**Signal-only axis detection DOES NOT WORK [measured — all rules scored below chance].** Over the 19
ground-truth pairs: variant lookup **100%**, gait-band energy 16%, antiphase×amplitude 16%, raw
amplitude 5% — against a 33% random baseline. Worse than chance is systematic, not noise: the
highest-amplitude Deg axis is reliably *not* sagittal, because `Deg_Z`'s variance is dominated by yaw
drift (§4.2). L/R antiphase is **necessary but not sufficient** — every projection of a planar leg
swing is antiphase, so it cannot discriminate. Do not re-attempt these; extend the lookup instead.

**The labeled features ARE all genuinely sagittal [measured].** Independent of any raw file, every
one of the 9 revs shows L/R antiphase during walking (median r −0.43 to −0.86), including the revs
with no raw counterpart (rev2/3/5/6). So the exporter picked a sagittal channel every time and the
training set is physically consistent, even though the *named* source axis differs by revision.

**Two defects in the upstream MATLAB [measured] — neither corrupted the current labeled set:**
- `timestamp.m` returns only the **last** inter-sample interval: its loop overwrites `del_t` every
  pass (the commented-out variant that accumulates `time_list` was the intent). With
  `time_temp = time{i,j}` set once at `k==1`, one Δt filters the whole trial. If the final two
  timestamps ever tie, `a = 0`, the recursion becomes `y[n] = y[n-1]`, and the output **freezes flat
  for the entire trial, silently** — the §2.6 failure mode exactly. Checked all 44 trials:
  last-dt/median-dt ∈ [0.90, 1.28], no zeros, so current features are sound. Fix: return the vector,
  or use `median(diff(time))`, never `diff(end)`.
- `csv2mat.m:35` does `data_table(1:5:end,:)` — **naive 5× decimation, no anti-aliasing**, precisely
  what §2.5 forbids. The annotated set was NOT produced with it active (paired row counts are
  identical), so labels are clean; but it will alias the next export that runs through it.

---

## 7. Evaluation

- Headline metric: **macro-F1**.
- Error taxonomy is a strictly precedence-ordered MECE partition:
  `correct → omission → flicker → late → early → steady_confusion → remainder`.
- The measure layer is **blind**: objective numbers only, no opinion. All judgment lives in diagnose.
- Ground truth already locates transitions exactly. Do not implement cross-correlation lag search.
- UNKNOWN is excluded consistently across per-trial and corpus-level metrics.
- Group by **subject (filename field 2)** for cross-validation (§5.5): subject is known for every
  file, so this defends subject leakage directly. Session-day grouping is subsumed — one subject per
  day. Caveat: the subject pool is small and `69` dominates (see the census for the live count), so
  cross-subject claims are bounded by how few distinct subjects exist, not by unknown identity.
- **The labeled family (rev\*) groups by `rev`, not by field 2** — those filenames have no subject
  field. A `rev` is **one subject on one day** (identity + date unknown; trials within a rev share
  both, Lu, 2026-07-20). So the group unit there is the rev, and a held-out rev is a genuine
  "new subject/session" — the honest deployment bar for "classify loco state from an unseen CSV."
- **Lockbox holdout [decided].** Split the labeled set **before any training**, grouped by rev so no
  trial leaks: *train* (fit) / *validation* (the champion–challenger loop gates promotions here) /
  *lockbox* (1–2 whole revs, sealed, opened **exactly once** at the very end for the honest number).
  The loop must never see the lockbox, or the final macro-F1 is a number it optimized toward, not a
  generalization estimate. Candidate lockbox spans both regimes — a balanced rev (`rev8` ~70/27) plus
  a walk-heavy one (`rev13`).
- **Always select by time, never by index.** rev2 samples at **494 Hz, not 500** — jitter
  accumulates to **4.2 s of drift by t=320 s**, so `int(t*fs)` points 4.2 seconds past the event.
  **AUDITED (2026-07-20) [measured]:** `stages/s1_clean/resample.py` is time-safe — every grid is
  built from real timestamps (`np.arange(t[0], t[-1], …)`) and every value interpolated against the
  measured time vector, never reconstructed from a rate. A synthetic 494 Hz segment places a
  true-t=160000 ms event at output **160000.0 ms (err 0.00)**, vs **+1943 ms** under the
  `int(t*fs)@500` bug. `manifest.py` / `census.py` never invert rate to an index either.
- **Per-file calibration, never corpus-wide.** A fixed threshold scores `walk_rec = 0.000` on rev8
  (it calls every walking window standing); per-file calibration scores 1.000/1.000 on the same
  file. Do not fit constants to more data — more data yields a better *global* constant, and global
  is the disease. rev8 wants 0.426 where rev2 wants 0.605; the answer is not needing one.
- **Abstain rather than force.** Coverage of 73–98% is acceptable; abstained windows genuinely
  contain both states, mirroring the human `-1` label. Note the tension: the abstained windows are
  the transitions, which is where a controller most needs an answer.

---

## 8. Environment gotchas

- **NumPy 2.0:** the `.ptp()` ndarray method was removed. Use `np.ptp(array, axis=...)`.
- Windows: subagents with very long prompts can hit the 8191-char command-line limit. Define
  subagents as filesystem files rather than inline prompts.
- PowerShell 5.1 does not accept `&&` as a statement separator.

---

## 9. Standing decisions

- Sample rate is 100 Hz (10 ms). **Window length is now an open tradeoff, and data warrants
  revisiting it [measured]:** 2 s gives transition precision; 4 s is needed for slow gait (a 0.22 Hz
  stride yields ~0.9 swaps per 2 s window and abstains). Wider windows cost wider abstention at
  transitions.
- **Band-limited features must not assume a healthy-adult gait band.** `(0.5, 3.0) Hz` is wrong for
  this population, which runs to **0.13 Hz** (cadence 16–102 steps/min) **[measured]**.
- Random Forest is the starting model. The windowing/feature-extraction layer is the durable,
  model-agnostic boundary — **feature selection lives there, not in the clean layer.**
- **The canonical file keeps MEASURED channels only.** The line is measurement vs computation, not
  useful vs useless. The device *measures* IMU channels (L/R/B x Deg/Gyro/Acc) and load cells; it
  *computes* Cadence, Stride Length, Hip_ROM, GCP, Adaptability, admittance, PID state. Computed
  columns are the firmware's opinion, not observation — the same category as `loco`, and there is
  nothing to trust-check in a number the firmware derived. Two consequences beyond storage:
  **(a)** every column that churns position between variants (`Step` 46/47, `Cadence` 45/46, the
  whole 83/79/91 tail) is a computed one, so dropping them **collapses the schema variants into one**;
  **(b)** computed values depend on firmware version, so training on them partly learns which
  firmware produced the file — the era confound baked into the feature set.
  Canonical = **30 measured columns**, defined as `KEEP_MEASURED` in `stages/s1_clean/config.py`.
  (The rule targets *app-layer* compute; on-sensor fusion is a separate tier — see §4.6.)
- **The column count is 30, 32 or 33 depending on which question is asked. All three are right
  [measured, 2026-07-20]** — recorded because the drift looked like a bug and is not:
  | count | is | where it comes from |
  |---|---|---|
  | **30** | `KEEP_MEASURED` | `Time` + 27 IMU (3 sides × Deg/Gyro/Acc × XYZ) + 2 load cells |
  | 32 | 30 + the `Hip_Deg` pair | **historical — no longer produced**, see below |
  | **33** | what is on disk | 32 + `segment`, which the clean layer *adds* |

  `segment` is pipeline metadata (§3.1), not a raw column, which is why it appears in no `KEEP_*`
  tuple. `Label` is **not** among these: raw carries no labels (§0), so `KEEP_IF_PRESENT` only fires
  on the labeled family. **After the `Hip_Deg` removal below, raw-side clean output is 31 columns**
  (30 measured + `segment`).
- **`KEEP_EXCEPTIONS` is now EMPTY — the measured-only rule has no exceptions
  [decided, 2026-07-20 — reverses the entry below it].** `Hip_Deg_L` / `Hip_Deg_R` were retained as
  a bridge to the open-source gait dataset's `Hip_Flex_L/R`. Removed, because every premise of the
  exception failed when checked:
  - **It is redundant.** `corr(Hip_Deg_<side>, <side>_Deg_Y)` = **0.991** median across the clean
    corpus (vs ~0.02–0.21 against X and Z). It is the sagittal angle S1 already keeps, re-zeroed.
  - **But not a clean function of it**, which is worse than being redundant. Affine fits give slope
    ~0.96–0.99 with a per-file offset of −75 to −88°, and max residual **4° to 152°**. The
    unmodeled remainder *is* the firmware's zeroing convention — the exact firmware-version signal
    the measured-only rule exists to strip (§9 consequence **b**). The old entry flagged that
    convention as `[open]`; the resolution is that we do not need it.
  - **It is dead on part of the corpus.** Zero-variance on **12 of 180** (file, side) pairs — six
    files, four flat at `0.0`, and twice **frozen at a nonzero constant** (`-7.42`, `+9.58`), which
    no `!= 0` sanity guard would catch.
  - **The bridge had no far side.** §5.7: the open dataset is *not in the repo*. Nothing downstream
    ever read the column — S2 trains on the four rotational rev* features only.

  Recoverable by name from `data/raw/` if that dataset ever arrives — the same standing as `loco`
  and `L/R_Ref_Force`. **The exception mechanism stays in place; it just holds nothing.** An empty
  `KEEP_EXCEPTIONS` is a stronger invariant than a populated one: canonical == measured, no caveat.
- `L_Ref_Force` / `R_Ref_Force` are excluded as controller setpoints (commanded, not measured).
  **[open]** — not yet confirmed with the firmware side.
- Storage cost is a file-format problem, not a column-count problem: clean output is **parquet**.
- HMM is a post-processing smoothing layer, not a standalone model. Its transition penalty trades
  off against transition lag — the same problem the existing rule-based algorithm has. Tune it
  deliberately; do not adopt naively.

---

## 10. The swap rule (S3 physics baseline) **[measured]**

**Walking is the legs alternating.** Not how far they swing — *whether they swap*.

> Count how many times `L_ang − R_ang` commits past `+1°` and then past `−1°` within a 2 s window.
> **0 swaps → STANDING. ≥2 swaps → WALKING. Exactly 1 → AMBIGUOUS.**

`1` is genuinely ambiguous, not a fudge: one leg passing the other happens both when you take a step
and when you shift your weight.

| file | stand_rec | walk_rec | coverage | macro-F1 |
|---|---|---|---|---|
| rev2_t1 | 0.930 | 0.989 | 0.830 | 0.928 |
| rev2_t3 | 1.000 | 0.930 | 0.729 | 0.776 |
| **rev8_t3** (held out) | **0.966** | **1.000** | **0.981** | **0.988** |

**Zero fitted parameters.** `delta = 1°` is a sensor noise floor — 5× the measured standing noise
came out 0.76–0.88° on all three files independently. `1 swap` is not a chosen threshold: standing
measures **0** and walking measures **2** on every file across a **4× amplitude range** (rev8 swings
22°, rev2 swings 49°), so 1 is the only integer between them.

Known weakness: slow walking (0.22 Hz stride) yields ~0.9 swaps per 2 s window and abstains. A
**window-length** problem, not a rule problem (§9).

### 10.1 Validated descriptors **[measured]**

| descriptor | status |
|---|---|
| `ileg_minhalf` — min of `ptp(L−R)` over the window's two halves | AUC 0.967 / 0.972 on two independent trials; medians 0.2–0.5 (standing) vs 39–41 (walking). **Needs per-file calibration.** |
| `interleg_offset` — median of `L−R` | **Posture only** (AUC 0.50 for walk/stand). Separates feet-together from split-stance: baseline −3.3°, splits at **+17°** and **−22°** — opposite legs leading. |

### 10.2 Recordings begin at rest **[measured — nearly all files]**

An **external** label — it comes from how sessions are run, not from any algorithm. It gives every
file a standing reference measured on the same person, sensor and mounting minutes earlier. **This
is the mechanism that makes per-file calibration possible** (§7), and it is the
calibration-as-data-harvest insight arriving from the physics side.

**[open]** many files also open with a segment of exactly 10 rows (§3.2). Possibly the same files —
worth checking.

### 10.3 States richer than the human labels **[measured]**

Human labels are `0` / `10` / `-1`. The physics distinguishes more: `WALKING`; `WALKING_SLOW`
(cadence < 0.6× the file's own median); `STANDING_FEET_TOGETHER`; `STANDING_SPLIT_L` / `_R` (0 swaps,
offset +17°/−22° from baseline — stopped mid-stride, one leg leading); `STANDING_SHIFTING`
(**unvalidated — rests on 3 and 8 windows**).

In rev2_t1 the last **61.9 s** labeled WALKING is walking at a third the cadence, and the two
STANDING bouts are **different postures**. The labels are coarser than the signal.

---

## 11. Methodology warnings **[measured — each was hit in practice]**

1. **Density needs mass.** Mode-finding declared files "unimodal — one behaviour" when a 5 s stand
   was 0.4% of the file. Hit **twice**, the second time one message after invoking it as a lesson.
   *A stop is not a mode; it is a stretch of time.* This is also what broke §4.1 (whole-file p99 on
   a 93.7%-walking file).
2. **Windows must not straddle edges.** A "32× cadence-invariant" claim measured whole phases, not
   per-window. A bout analysis deleted UNKNOWNs *then* computed runs, silently merging across gaps.
3. **Index ≠ time.** See §7.
4. **The eyeball is not truth.** Claude called a 6.0 s standing bout "blurred, should be 2.5 s"; the
   labels said **6,246 ms**. Same error as §3.1 — measuring the static plateau and treating it as
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
  occur at ~69 s and ~196 s in *both* trials independently — a protocol event, probably a turn.
  Claude invented a category to explain its own artifact and nearly asked for it to be defined.
- **`gyro_energy` bimodality (modes 22 / 61) was two subjects**, unimodal within each (sub1 at 43,
  sub2 at 64) — not two behaviours.
- **`0.605` (HI/p75 ratio) was `30/49.6` from one file counted twice.** rev8's true ratio is 0.426.

### 11.3 Agent proposes, test disposes **[decided]**

> *"Every rule asserted was wrong; every rule the tests checked survived or died honestly."*

**Agent = rule discovery, not per-window labeling.** Reading graphs found slow walking and split
stance that no descriptor encoded. The swap rule came from asking *what walking is*, not from
fitting. This belongs in the S3 agent's system prompt verbatim.

---

## Changelog

| Date | Phase | Added |
|---|---|---|
| 2026-07-20 | 3 | Provenance audit (4 flags raised on the 33-column canonical set, all checked against the corpus). **CORRECTED §4.1b:** the Deg↔Gyro Y↔Z crossing is **device-wide, not a trunk defect** — `d(Deg_Y)/dt`→`Gyro_Z` unanimously on L (62 files) and R (65), and 45/47 on B; `L_Gyro_Y` is no more sagittal than `B_Gyro_Y` (median \|r\| 0.16 vs 0.99). The earlier trunk-only framing would have sent a fix to one side of a three-side convention. Post-clean slopes 0.987/0.984/0.986 confirm the unit fix landed. **§4.3 narrowed** to placement risk only. **NEW §4.6:** `Deg` is on-sensor *fusion*, not a transducer reading — three-tier provenance (transducer / on-sensor fusion / app-layer compute); explains §4.2 yaw drift mechanistically and demotes §4.1's r=0.999 from corroboration to near-tautology. **CUT `Hip_Deg_L/R`** — `KEEP_EXCEPTIONS` is now empty: 0.991 redundant with `Deg_Y`, residual = firmware zeroing convention, zero-variance on 12/180 (file,side) pairs (twice frozen nonzero), and the open dataset it bridged to is not in the repo (§5.7) — nothing read it. **§9 count reconciliation:** 30 / 32 / 33 all correct, different questions; raw-side clean output is now **31** (30 + `segment`). |
| 2026-07-16 | 0 | v1 seeded: channel trust, rate confound, label semantics, eval rules, NumPy gotcha |
| 2026-07-16 | 1 | v2 from the real corpus: 5 variants / 45-col contract / position-is-a-lie; two rate eras; quantization tiers; anti-aliasing proof; segments + startup burst; -1 vs 255; rev2 as lossy family; provenance tags |
| 2026-07-20 | 2→3 | Phase 2 gate CLOSED: Lu signed off on the exception review (7 `needs_human` = §2.6 broken-clock batch → re-export; 16 `known_expected`; 0 novel). Dead code removed (`Resolution.has`, unreachable `legacy_algo` interp-role). Count-free sweep extended past the prose into code comments, the generated `clean_report.md`, and surviving inventory counts (schema-variant / subject / startup-burst tallies); named-file example stats and this changelog keep their numbers. Phase 3 (S2 loop) starting. |
| 2026-07-20 | 2 | Phase 2: S1 exception agent (`agents/s1_exception.py`). Read-only agent triages the clean stage's genuine exceptions (quarantines + gyro axis anomalies + yaw-drift flags; routine abstentions summarized, not triaged) into `known_expected` / `novel` / `needs_human`, each grounded in a DOMAIN_NOTES section; deterministic wrapper writes `exceptions_review.jsonl`. Degenerate-time-base quarantines → `needs_human` (re-export); handled anomalies/drift → `known_expected`. Deleted the Phase-0 smoke agent; fixed the shipped-but-uncalled `load_dotenv()`. |
| 2026-07-20 | 1 | Corpus refreshed (more raw files + new subjects `70`/`92`; `sub1/sub2` markers removed). §2.6 NEW: a 2026-05 batch has a degenerate time base (non-monotonic, duplicated timestamps, median dt=0) — quarantined, the pipeline's first real quarantine; `measure_hz`/`clean_one` now guard `median(dt)>0`. **Records made count-free** (per Lu): live tallies live in the run artifacts, not this file. Gate still holds (partition asserted). |
| 2026-07-20 | 1 | S1 hardening. **MEASURED:** §4.1b confirmed on all 88 files (gyro units normalized to deg/s in the clean layer, per-file `channel_trust.json`, detect-don't-assert with static-file abstention); §4.2 yaw is `Deg_Z`, drift is file-specific (13 chans / 11 files), flagged not dropped; §7 `resample.py` proven time-safe by drift test. **REVERSED §5.5:** filename field 2 IS the subject (`69` recurs across 5 months) — subject known for every file, `needs_human` RESOLVED (Lu confirmed `sub1/sub2` are trial batches). **DESIGN:** quarantine is a `quarantine.jsonl` ledger, raw file never moved. Docs reconciled (angvel "untrusted"→reliable §6.1; 30-vs-32 cols; PLAN/README status). |
| 2026-07-16 | 1 | v3 merging the physics/rule-discovery track. **RETRACTED §4.1** (angvel is reliable, r=0.999 — the noise claim was never verified and the "confirming" measurement was taken over a 93.7%-walking file). **NEW:** §4.1b gyro units inconsistent within a file (B=rad/s, L/R=deg/s; Y↔Z swap); §5.5 session ≠ subject, subject unknown for 88/95 (needs_human); §10 the swap rule + validated descriptors + richer states; §11 methodology warnings, broken tests, retractions. **UPGRADED:** §4.5 loco severed with evidence; §6.2 raw→rev2 mapping largely resolved; §7 select-by-time, per-file calibration, abstain-don't-force; §9 window length now an open tradeoff, gait band 0.13 Hz not 0.5–3.0 |