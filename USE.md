# USE — promoting an S3 physics anchor into the S2 feature set

How to take a physics anchor that S3 computes ([`stages/s3_physics/anchors.py`](stages/s3_physics/anchors.py))
and let the S2 classifier train on it directly. Read `PLAN.md` (S2/S3 contracts) and
`DOMAIN_NOTES.md` §10 first; this doc is the operational procedure, not the rationale.

---

## What this is, and the decision before you start

There are two ways the physics view can reach the label:

| | where it lives | what it is |
|---|---|---|
| **late fusion** (today) | S4, [`fuse.py`](stages/s4_fusion/fuse.py) | a zero-parameter swap policy adjudicates **after** S2, with the calibrated abstain-on-disagreement tier |
| **early fusion** (this doc) | S2, [`window_features()`](stages/s2_ml/features.py) | the RandomForest gets the anchor as one more column and weighs it itself |

Early fusion can lift macro-F1, but it **spends** two things late fusion keeps: the interpretable
zero-parameter guarantee of the swap rule, and the HIGH/MED/LOW calibration a controller acts on.
Promote an anchor into features only when the signal is worth more inside the model than as an
external adjudicator. They are not mutually exclusive — an anchor can be a feature *and* still drive
the S4 policy — but do not assume "add it everywhere" is free.

**Key constraint:** the S2 `ExperimentSpec` ([`experiment.py`](stages/s2_ml/experiment.py)) can only
`drop_features`, set `window_s`/`stride_s`, or override `model_params`. It has **no add-feature
move**. So promoting an anchor is a **human code change** to `window_features()` — the agent loop
cannot do it, and it does not go through the critic. It is the same "human-directed baseline change"
path recorded in the current `champion.json` rationale.

---

## Two gates before an anchor is even a candidate

### 1. Rate-invariance (the S3 gate — non-negotiable)

An anchor that fails the rate audit is a claim about the sampling grid, not the body. Promoting it
injects the exact confound the audit exists to catch. Check [`runs/s3_physics/rate_audit.json`](runs/s3_physics/rate_audit.json):

| anchor | verdict | eligible? |
|---|---|---|
| `periodicity` | invariant | ✓ |
| `grav_stab` | invariant | ✓ |
| `gait_hz` | invariant | ✓ |
| `antiphase` | **rate_dependent** (Δ0.17) | ✗ never promote |
| `gyro_energy` | **rate_dependent** (Δ0.50) | ✗ never promote (failed at id=69) |

### 2. Non-redundancy

S2 already carries proxies for some anchors — `ang_LR_corr` ≈ `antiphase`, `*_angvel_dom_hz` ≈
`gait_hz`. The champion's whole story is *dropping* collinear features (§4.6; `Hip_Deg` cut at 0.991
corr). Before adding, correlate the candidate against the existing feature columns on the training
windows; if |r| is high against something already there, promoting it adds noise, not signal.

**Best candidates right now:** `grav_stab` (posture steadiness — and STAND is the weak class at 0.85
recall, so this targets the actual gap), then `periodicity` (autocorrelation rhythm strength,
distinct from S2's spectral band-fraction).

### A structural limit

`window_features()` sees **one window slice**. Anchors that need whole-segment or per-file context
— the stride-adaptive swap verdict (`swap_verdict_adaptive`, needs ~2 strides of context) and the
rest-anchor recentering (`interleg_center`, a per-file zero) — **cannot** become plain window
features without threading trial-level context through the feature builder. The window-local anchors
(`periodicity`, `grav_stab`, `gait_hz`, and `swap_count` at `interleg_center=0`) can. Promote the
window-local ones; leave the context-dependent swap logic in S4 where it already works.

---

## Procedure

1. **Confirm the verdict.** `rate_audit.json` says `invariant` for the anchor. If not, stop.

2. **Confirm it's new.** Correlate the candidate against `feature_columns(df)` on the train split;
   reject if collinear with an existing feature.

3. **Add the computation to `window_features()`** in [`stages/s2_ml/features.py`](stages/s2_ml/features.py).
   Compute it from the window slice only. Prefer importing the existing implementation from
   `anchors.py` (e.g. `_periodicity`, `grav_stab`'s formula) over re-deriving it, so the physics is
   single-sourced and cannot drift between S2 and S3. It flows automatically from there:
   `window_features` → `build_windows` → `feature_columns` → `train`.

4. **Rebuild and retrain, compare to the champion.**
   ```powershell
   python -m stages.s2_ml.features                          # sanity: new feature count printed
   python -m stages.s2_ml.train --out runs\s2_ml --taxonomy # LORO retrain + score
   ```
   Read `runs\s2_ml\locoeval.md`: the new **macro-F1** must beat the champion's **0.8977** by the
   promotion margin (**+0.005**) to justify promotion. Also check **stand recall** and the **weak
   revs (rev2, rev5)** did not regress — a headline gain bought by sacrificing them is not a gain.

5. **Promote through the metric gate.** Because no `ExperimentSpec` can add a feature, this is a
   human-directed promotion: record the change as a ledger entry with its measured macro-F1 and the
   new `git_sha`, and update `champion.json` only if it clears the margin. The champion is its
   `champion.json` + `git_sha` (the replay unit), not a model blob — the feature edit is part of the
   deterministic core, so the new `git_sha` captures it.

6. **Regenerate the downstream stages** so S3/S4 reflect the new champion:
   ```powershell
   python -m stages.s3_physics.run    # anchors + rate audit (unchanged, but re-verify)
   python -m stages.s4_fusion.run     # fusion now runs on the new champion's OOF
   ```

7. **Keep the lockbox sealed.** `rev8` opens **once**, at the very end. Do not measure a feature
   promotion against it — that spends the one unbiased final read (the same discipline that "spent"
   `rev13`).

---

## How to read the result

- **Win:** LORO macro-F1 in `locoeval.md` ≥ 0.8977 + 0.005, with stand recall and rev2/rev5 flat or
  up, and the new feature not collinear with an existing one.
- **No-op / reject:** gain under the margin, or a gain that comes by trading away stand recall or a
  weak rev. Log the rejection anyway — a recorded "tried, didn't clear" stops the idea being
  re-proposed (rejections are the valuable half of the ledger).
- **Red flag:** a large macro-F1 jump from an anchor whose rate verdict you didn't check, or one
  that's collinear with `ang_LR_corr` / `*_angvel_dom_hz`. That's usually a grid artifact or a
  double-counted feature, not real signal.

---

## Reference commands

```powershell
# retrain + score after a feature edit
python -m stages.s2_ml.train --out runs\s2_ml --taxonomy

# regenerate physics + fusion on the new champion
python -m stages.s3_physics.run
python -m stages.s4_fusion.run
```

Artifacts to read, in order: `runs\s2_ml\locoeval.md` (headline + per-rev + taxonomy) →
`runs\s2_ml\champion.json` (is it the new best?) → `runs\s4_fusion\fusion_report.md` (did the
external fusion still help, or did the feature absorb its value?).

---

# USE — exporting labeled trials with fused predictions

Write one CSV per labeled trial into `results/`, named identically to the input, carrying the
original rev2 columns plus the fused call mapped densely back onto every row.

```powershell
python -m stages.s4_fusion.export            # -> results\annotated_loco_rev*_trial_*.csv
python -m stages.s4_fusion.export --out out\somewhere   # optional: different target dir
```

Each `results\<same-name>.csv` = the 6 original columns (`Time`, `L/R_ang_LPF`,
`L/R_angvel_LPF`, `Label`) **plus two appended columns**:

| column | values | meaning |
|---|---|---|
| `fused_label` | `0` / `10` | the fused STAND/WALK call ([`fuse.py`](stages/s4_fusion/fuse.py)) |
| `confidence` | `high` / `medium` / `low` | S2×S3 agreement tier; `low` = the abstain (disagreement) cell |

**What "dense" means and where it's blank.** Each original row inherits the fused call of the 2 s
window covering its `Time`; both columns are **left blank** where no scored window exists —
transitions (mixed-label windows), dropped/short segments, and every row of the **lockbox + excluded
revs (`rev8`, `rev13`, `rev14`)**, which never enter fusion. Blank is deliberate, not a gap: the
export never guesses a label it didn't compute. A whole file being blank means it's lockbox/excluded.

**Inputs / regeneration.** Reads `runs\s4_fusion\fused_windows.csv`; if that's absent it runs the S4
deterministic core first ([`run.py`](stages/s4_fusion/run.py)). It does **not** retrain — export the
results *after* a champion change has flowed through `python -m stages.s4_fusion.run`, or the labels
will reflect the previous champion. Standalone from the orchestrator; run it whenever you want the
per-trial view refreshed.
