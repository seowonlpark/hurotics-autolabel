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

### 4.1b Gyro units are inconsistent WITHIN a single file **[measured — confirmed on all 88 raw files]**

- **`B_Gyro_*` is rad/s. `L_Gyro_*` / `R_Gyro_*` is deg/s.** Same naming convention, same file.
  Any feature mixing trunk and thigh gyro without conversion is off by **57.3×**. The clean-layer
  trust check reproduces this from the sagittal regression slope: **0.98** on L/R (deg/s) vs
  **0.017** (≈1/57.3) on B (rad/s).
- **Axis: `d(Deg_Y)/dt` tracks `Gyro_Z`, not `Gyro_Y`** — on 83–85 of 88 files per side, but this is
  **not universal**. Two files (`…10_4_0_sub1`, `00038…1_15_11_28`) map `B_Deg_Y → B_Gyro_Y` at
  r≈0.96–0.99 (one sign-flipped). So the axis is **detected per file, never asserted from a table**:
  a blanket Y↔Z swap would corrupt exactly those.
- `Deg_Y` needs **no sign normalization** — raw L vs R is already anti-phase in 84% of files.

`Deg_Y` is the sagittal (flexion) channel. This is the channel the swap rule reads.

**RESOLVED (2026-07-20):** unit normalization now lives in the clean layer
(`stages/s1_clean/channel_trust.py`). Every gyro channel is normalized to **deg/s**; the sagittal
axis + unit are **detected per file** (regress `d(Deg_Y)/dt` against each gyro axis, slope → unit,
argmax|r| → axis) and written to a per-file `channel_trust.json`. Static files, whose derivative
carries no signal, **abstain** and fall back to the documented convention (r-floor 0.9), recording
that they did (11.1, density needs mass). Axes are **recorded, not reordered** — no silent mutation.
On the 88-file corpus: 88 B-sides normalized rad/s→deg/s, 91 side-abstentions, **2** confident axis
anomalies (the `B_Deg_Y→B_Gyro_Y` files above).

### 4.2 Yaw is drift-contaminated — but per-file, not wholesale **[measured]**
Original **[reported]**: in treadmill data yaw correlated with session time at r ≈ −0.95, measuring
elapsed time not orientation.

Measured on the raw corpus (clean-layer drift test, `|corr(Deg, Time)|` duration-weighted over
segments ≥ 5 s): **`Deg_Z` is the yaw-like axis** on every side — median |r| ≈ 0.33 vs the sagittal
`Deg_Y` at ≈ 0.08. But the strong drift signature is **file-specific, not universal**: only **13
channels across 11 of 88 files** cross |r| ≥ 0.9 (all of them `*_Deg_Z`). So yaw is **flagged per
channel per file** in `channel_trust.json` (`drift` section), **not dropped wholesale**. It is a
feature-time exclusion signal; the raw superset is kept. Any yaw-derived feature must consult the
per-file drift flag.

### 4.3 The trunk (B) IMU was dropped from rev2 for a real reason **[reported]**
Belly/trunk placement was inconsistent between subjects, and may have been treadmill-mounted in some
trials. The raw family still carries `B_Deg_*` / `B_Gyro_*` / `B_Acc_*`; treat trunk channels as
suspect until placement consistency is established per session.

### 4.4 `Time` is metadata, not a feature **[decided]**
Used for dt / rate / segmentation only. Never fed to a model.

### 4.5 `loco` is severed and stays severed **[measured]**
Its "standing" class contains a decile **as periodic at the gait frequency as median walking**. A
label that fails inspection cannot validate anything. (It is also an outdated algorithm's output
**[reported]**, but that is the weaker reason — the strong one is that it is demonstrably wrong.)

Not ground truth, not a feature. Must be dropped **by name** — in `0fda484e` files, dropping index
47 would delete the step counter instead.

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

### 5.5 Session ≠ subject, and subject is mostly UNKNOWN **[measured]**
`20260114` contains both `sub1` and `sub2`, **both under `id=69`**. Filename field 2 is **not a
subject** — it tracks the date block (device / firmware / protocol). Only **7 of 95** files mark
the wearer at all.

So grouping by session is **not sufficient** for cross-validation: subject leakage is the confound
that inflates scores, and subject is unrecoverable for 88 of 95 files.

This is a genuine **`needs_human`** finding — exactly the "AI cannot proceed without human input"
case the PL asked to surface. Someone may be able to reconstruct wearer identity from session
records; until then, no honest claim about cross-subject generalization is possible.

**[reconcile]** The `95` in this section (and the `7 of 95` above) is a **stale/undefined count** —
the current raw corpus is **88 files** (§0), and no 95-file set is defined anywhere. Treat the
*ratio* (almost all files subject-unmarked), not the absolute counts, as the finding until the
95-file universe is reconciled against the present corpus. Do not silently rewrite 95→88: the
earlier snapshot may have included files since removed.

### 5.6 Current corpus class coverage **[reported]**
h-medi contains essentially only STANDING and WALKING. Rare-class separability (stairs, varied
terrain) is **untestable** on current data. Any claim about rare-class performance is overclaiming.

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

### 6.2 The raw→rev2 mapping is largely resolved **[measured]**
`*_ang_LPF ≈ LPF(*_Deg_Y)`, and `*_angvel_LPF = d(*_ang_LPF)/dt` (r=0.999, §4.1). Supported by:
`Deg_Y` is the sagittal channel, already anti-phase L vs R in 84% of files (§4.1b); and `R_ang_LPF`
opens at 85.43 against raw `R_Deg_Y` at 85.69.

**[open]** LPF parameters remain unknown. Incoming raw-format labeled data **[reported]** removes
the need for a bridge, so this is no longer blocking.

---

## 7. Evaluation

- Headline metric: **macro-F1**.
- Error taxonomy is a strictly precedence-ordered MECE partition:
  `correct → omission → flicker → late → early → steady_confusion → remainder`.
- The measure layer is **blind**: objective numbers only, no opinion. All judgment lives in diagnose.
- Ground truth already locates transitions exactly. Do not implement cross-correlation lag search.
- UNKNOWN is excluded consistently across per-trial and corpus-level metrics.
- Group by **session** for cross-validation — but see §5.5: session ≠ subject, and subject is
  unknown for 88 of 95 files. Session grouping defends against era leakage; it does **not** defend
  against subject leakage. Do not claim cross-subject generalization from it.
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

### 10.2 Recordings begin at rest **[measured, 26/28 files]**

An **external** label — it comes from how sessions are run, not from any algorithm. It gives every
file a standing reference measured on the same person, sensor and mounting minutes earlier. **This
is the mechanism that makes per-file calibration possible** (§7), and it is the
calibration-as-data-harvest insight arriving from the physics side.

**[open]** 26 files also open with a segment of exactly 10 rows (§3.2). Possibly the same 26 —
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
| 2026-07-16 | 0 | v1 seeded: channel trust, rate confound, label semantics, eval rules, NumPy gotcha |
| 2026-07-16 | 1 | v2 from the real corpus: 5 variants / 45-col contract / position-is-a-lie; two rate eras; quantization tiers; anti-aliasing proof; segments + startup burst; -1 vs 255; rev2 as lossy family; provenance tags |
| 2026-07-16 | 1 | v3 merging the physics/rule-discovery track. **RETRACTED §4.1** (angvel is reliable, r=0.999 — the noise claim was never verified and the "confirming" measurement was taken over a 93.7%-walking file). **NEW:** §4.1b gyro units inconsistent within a file (B=rad/s, L/R=deg/s; Y↔Z swap); §5.5 session ≠ subject, subject unknown for 88/95 (needs_human); §10 the swap rule + validated descriptors + richer states; §11 methodology warnings, broken tests, retractions. **UPGRADED:** §4.5 loco severed with evidence; §6.2 raw→rev2 mapping largely resolved; §7 select-by-time, per-file calibration, abstain-don't-force; §9 window length now an open tradeoff, gait band 0.13 Hz not 0.5–3.0 |