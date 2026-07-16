# DOMAIN NOTES — H-CARE IMU locomotion pipeline

**This file is injected into every agent's system prompt. It is the pipeline's institutional memory.**

Rules for this file:
- Every entry is a *finding with a reason*, not an instruction without justification.
- Agents may propose additions (via their rationale logs). Only a human commits them.
- If you are an agent reading this: these facts are established. Do not re-derive them, do not
  contradict them silently. If your evidence contradicts an entry, say so explicitly and flag
  `needs_human` rather than acting on it.

---

## 1. Channel trust

### 1.1 Angular velocity channels are untrustworthy as provided
`L_angvel_LPF` / `R_angvel_LPF` ring at ±40–80 deg/s during *static postures* where true angular
velocity is ~0. The noise floor is high enough to swamp real signal at rest.

**Policy:** do not use provided angular-velocity channels as features. Derive velocity numerically
from the filtered angle channels (`L_ang_LPF` / `R_ang_LPF`) via gradient.

### 1.2 Angle channels carry the real signal
`L_ang_LPF` / `R_ang_LPF` are trusted. Bilateral depth metrics and walking phase correlation are
built from these.

### 1.3 Yaw channels are drift-contaminated
In treadmill data, yaw correlates with session time at r ≈ −0.95 — i.e. it is measuring elapsed
time, not orientation. Gyroscopic drift.

**Policy:** yaw is excluded as a feature. Any new yaw-derived feature must first pass a
session-time correlation test.

### 1.4 `Time` is metadata, not a feature
Used for dt / cadence computation only. Never fed to a model as a feature.

---

## 2. Sampling rate

### 2.1 Sub-Hz rate differences dominate geometry
Rate experiment (id=69, n=42,421, subject held fixed while acquisition rate varied across
99.4 / 99.7 / 100.0 / 500.0 Hz): a k=2 partition clustered by **acquisition rate**, not by body
signal (ARI ≈ +0.315, purity ≈ 0.836). The geometry was reading the sampling rate.

**Policy:**
- Measure per-file rate to sub-Hz precision. "About 100 Hz" is not a passing check.
- Resample to the canonical rate or reject the file. Never mix rates silently.
- Stamp measured rate as metadata on every clean file so downstream stages cannot mix.

### 2.2 `gyro_energy` is rate-dependent
Flagged as unsuitable as a body-defined physics anchor because its value tracks acquisition rate.
Carried only until formally normalized or removed. Every anchor feature needs a rate-invariance
verdict before use.

---

## 3. Labels

### 3.1 STANDING labels include acceleration/deceleration ramps
Ground truth `STANDING` spans the treadmill ramp-up and ramp-down, not just the static plateau.
A prior analysis measured only the middle plateau and treated that as ground truth — this
mismatch **inverted a conclusion** that had already been drawn.

**Policy:** label semantics are a first-class failure mode. When predictions and ground truth
disagree systematically *around transitions*, the hypothesis space must include "the label
definition is wrong," not only "the model is wrong."

### 3.2 UNKNOWN encoding
Integer `255` = UNKNOWN is the only valid unknown encoding. NaN labels are an error, not a value —
they raise `ValueError` at load time.

### 3.3 Current corpus class coverage
The h-medi corpus contains essentially only STANDING and WALKING. Rare-class separability
(stairs, varied terrain) is **untestable** on current data. Any claim about rare-class performance
on this corpus is overclaiming.

---

## 4. Evaluation

- Headline metric: **macro-F1**.
- Error taxonomy is a strictly precedence-ordered MECE partition:
  `correct → omission → flicker → late → early → steady_confusion → remainder`.
- The measure layer is **blind**: objective numbers only, no opinion. All judgment lives in diagnose.
- Ground truth already locates transitions exactly. Do not implement cross-correlation lag search.
- UNKNOWN is excluded consistently across per-trial and corpus-level metrics.

---

## 5. Environment gotchas

- **NumPy 2.0:** the `.ptp()` ndarray method was removed. Use `np.ptp(array, axis=...)`.

---

## 6. Standing decisions

- Window size held at current value (100 Hz / 10 ms) unless performance data warrants revisiting.
- Random Forest is the starting model. The windowing/feature-extraction layer is the durable,
  model-agnostic boundary.
- HMM is a post-processing smoothing layer, not a standalone model. Its transition penalty trades
  off against transition lag — the same problem the existing rule-based algorithm has. Tune it
  deliberately; do not adopt naively.

---

## Changelog

| Date | Phase | Added |
|---|---|---|
| 2026-07-16 | 0 | v1 seeded: channel trust, rate confound, label semantics, eval rules, NumPy gotcha |
