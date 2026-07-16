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

## 0. The corpus (as of 2026-07-16)

**[measured]** `data/raw/` holds **88 CSV files** across **20 session folders**, 2025-10-21 to
2026-06-01. All 88 are one family (`raw_device`). Session date comes from the folder name and is
the only metadata that is trusted; it is also the correct group key for cross-validation.

Labeled data lives in `data/labeled/`, never `data/raw/`. A `Label` column appearing under
`data/raw/` is a contamination event, not a variant: quarantine and flag it.

---

## 1. Schema

### 1.1 Five header variants, one family **[measured]**

| variant | cols | files | era |
|---|---|---|---|
| `fb5ea2c2` | 67 | 62 | 2025-12 → 2026-01 |
| `0fda484e` | 67 | 11 | 2025-10 → 2025-12 |
| `e5f2660f` | 91 | 10 | 2026-05 |
| `4bfd6ab2` | 83 | 3 | 2025-10, 2025-12 |
| `86069795` | 79 | 2 | 2026-06 |

Variants track firmware eras. They differ in the tail; the head is near-stable.

### 1.2 The contract is 45 columns **[measured]**

`Time` through `Total Gait Length` is identical across all five variants. Everything past index 44
is variant-specific. The 45-column prefix is the real contract — not 67, not 83.

### 1.3 The numeric prefix is a LIE — resolve by name, never by position **[measured]**

The `NN_` prefix is a per-file position, not a stable identifier:

- `Step` sits at index **46 or 47** depending on variant
- `Cadence` sits at index **45 or 46**
- `loco` sits at index 47 — and in the 11 files of `0fda484e`, **index 47 is `Step`**

`df.iloc[:, 47]` therefore blends a step counter into a locomotion state across 73 files and never
raises. The column that *looks* positionally stable (`loco`) is precisely the trap.

**Policy:** strip the prefix, match on the name. Column presence is per-file; absence is recorded
as a fact, not an error.

---

## 2. Sampling rate

### 2.1 Two rate eras, confounded with schema AND session date **[measured]**

| | through 2026-01-28 | from 2026-05-20 |
|---|---|---|
| rate | ~100 Hz | 500 Hz |
| schema | `fb5ea2c2` / `0fda484e` / `4bfd6ab2` | `e5f2660f` / `86069795` |
| files | 76 | 12 |

Rate, schema and date change together. Resampling removes the *rate* difference; it does not
remove the era. A model can still learn "era" as a shortcut. Group by session for cross-validation.

### 2.2 The "odd" rates are timestamp quantization, not different rates **[measured]**

Measured across the corpus: 100.0 Hz (70 files), 500.0 Hz (12), 99.688 (3), 99.961 (2),
99.3789 (1). The odd ones are exact:

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

Incoming golden data is 100 Hz **[reported]**, so 100 Hz is canonical: 70 files already there,
12 decimated down, 6 grid-corrected. Nothing is upsampled; no bandwidth is invented.

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

---

## 3. Gaps and segments

### 3.1 The segment is the unit of analysis, not the file **[measured]**

88 files → **134 segments**, 91 usable, 43 dropped, **477.4 minutes usable**. Windows must never
straddle a gap; resampling across one invents data that was never measured.

### 3.2 There is a ~10-sample startup burst **[measured]**

26 files open with a segment of **exactly 10 rows**, then a gap, then the real trial. The 500 Hz
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

### 4.1 Angular velocity / gyro channels are untrusted **[reported, rev2 only]**
In rev2, `L_angvel_LPF` / `R_angvel_LPF` ring at ±40–80 deg/s during static postures where true
velocity is ~0. Measured on `annotated_loco_rev2_trial_1.csv`: |angvel| p99 ≈ 105 deg/s.

**[open]** The raw-family analogue is presumably `L_Gyro_*` / `R_Gyro_*`, but this has **not been
verified**. Do not apply the rev2 policy to raw gyro channels without testing them first.

### 4.2 Yaw is drift-contaminated **[reported]**
In treadmill data, yaw correlated with session time at r ≈ −0.95 — measuring elapsed time, not
orientation. Excluded. Any new yaw-derived feature must first pass a session-time correlation test.

### 4.3 The trunk (B) IMU was dropped from rev2 for a real reason **[reported]**
Belly/trunk placement was inconsistent between subjects, and may have been treadmill-mounted in some
trials. The raw family still carries `B_Deg_*` / `B_Gyro_*` / `B_Acc_*`; treat trunk channels as
suspect until placement consistency is established per session.

### 4.4 `Time` is metadata, not a feature **[decided]**
Used for dt / rate / segmentation only. Never fed to a model.

### 4.5 `loco` is an outdated algorithm's output **[reported]**
Not ground truth, not a feature. Recorded for provenance only. Must be dropped **by name** — in
`0fda484e` files, dropping index 47 would delete the step counter instead.

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

### 5.5 Current corpus class coverage **[reported]**
h-medi contains essentially only STANDING and WALKING. Rare-class separability (stairs, varied
terrain) is **untestable** on current data. Any claim about rare-class performance is overclaiming.

---

## 6. Products / families

### 6.1 rev2 is a separate, lossy family **[measured]**
`Time, L_ang_LPF, R_ang_LPF, L_angvel_LPF, R_angvel_LPF, Label` — 6 columns, sharing only `Time`
with the raw device family. It discards the trunk IMU, **all accelerometers** (the gravity
reference, hence all axis/calibration checks), load cells and GCP; and it keeps both untrusted
angular-velocity channels.

**Do not make rev2 the storage format.** Canonical storage is the name-resolved raw superset. A
wide honest table can always be projected down; a narrow one can never be recovered.

### 6.2 The raw→rev2 mapping is unresolved **[open]**
`R_ang_LPF` opens at 85.43 and raw `R_Deg_Y` opens at 85.69 — suggestive that
`*_ang_LPF ≈ LPF(*_Deg_Y)`, but those are different trials, so it is a **hypothesis, not a mapping**.
LPF parameters unknown **[reported]**. Superseded in priority by incoming raw-format labeled data
**[reported]**, which removes the need for a bridge.

---

## 7. Evaluation

- Headline metric: **macro-F1**.
- Error taxonomy is a strictly precedence-ordered MECE partition:
  `correct → omission → flicker → late → early → steady_confusion → remainder`.
- The measure layer is **blind**: objective numbers only, no opinion. All judgment lives in diagnose.
- Ground truth already locates transitions exactly. Do not implement cross-correlation lag search.
- UNKNOWN is excluded consistently across per-trial and corpus-level metrics.
- Group by **session** for cross-validation. Era leakage is the confound to defend against.

---

## 8. Environment gotchas

- **NumPy 2.0:** the `.ptp()` ndarray method was removed. Use `np.ptp(array, axis=...)`.
- Windows: subagents with very long prompts can hit the 8191-char command-line limit. Define
  subagents as filesystem files rather than inline prompts.
- PowerShell 5.1 does not accept `&&` as a statement separator.

---

## 9. Standing decisions

- Window size held at 100 Hz / 10 ms unless performance data warrants revisiting.
- Random Forest is the starting model. The windowing/feature-extraction layer is the durable,
  model-agnostic boundary — **feature selection lives there, not in the clean layer.**
- **The canonical file keeps MEASURED channels only.** The line is measurement vs computation, not
  useful vs useless. The device *measures* IMU channels (L/R/B x Deg/Gyro/Acc) and load cells; it
  *computes* Cadence, Stride Length, Hip_ROM, GCP, Adaptability, admittance, PID state. Computed
  columns are the firmware's opinion, not observation — the same category as `loco`, and there is
  nothing to trust-check in a number the firmware derived. Two consequences beyond storage:
  **(a)** every column that churns position between variants (`Step` 46/47, `Cadence` 45/46, the
  whole 83/79/91 tail) is a computed one, so dropping them **collapses 5 schema variants into 1**;
  **(b)** computed values depend on firmware version, so training on them partly learns which
  firmware produced the file — the era confound baked into the feature set.
  Canonical = **30 measured columns**, defined as `KEEP_MEASURED` in `stages/s1_clean/config.py`.
- **Documented exceptions to measured-only** (`KEEP_EXCEPTIONS`): `Hip_Deg_L` / `Hip_Deg_R` are
  computed but retained, because the open-source gait dataset's `Hip_Flex_L/R` is their direct
  analogue and that dataset is the primary real training asset. Deriving hip angle from thigh IMUs
  instead would require the firmware's convention, which is unresolved. **[open]**
- `L_Ref_Force` / `R_Ref_Force` are excluded as controller setpoints (commanded, not measured).
  **[open]** — not yet confirmed with the firmware side.
- Storage cost is a file-format problem, not a column-count problem: clean output is **parquet**.
- HMM is a post-processing smoothing layer, not a standalone model. Its transition penalty trades
  off against transition lag — the same problem the existing rule-based algorithm has. Tune it
  deliberately; do not adopt naively.

---

## Changelog

| Date | Phase | Added |
|---|---|---|
| 2026-07-16 | 0 | v1 seeded: channel trust, rate confound, label semantics, eval rules, NumPy gotcha |
| 2026-07-16 | 1 | v2 from the real corpus: 5 variants / 45-col contract / position-is-a-lie; two rate eras; quantization tiers; anti-aliasing proof; segments + startup burst; -1 vs 255; rev2 as lossy family; provenance tags |