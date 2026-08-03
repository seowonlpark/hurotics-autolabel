# Caveats

Everything known to be weak, unverified, or deliberately left out. `DOMAIN_NOTES.md`
records what the corpus taught us; this file records what this pipeline is *shaky* about.

The goal it is judged against: **label stand/walk on any recording at >=95% accuracy on
the windows it claims, abstaining rather than guessing on the rest.** Anything that
threatens that number belongs here.

Current measured position (leave-one-rev-out, 4,812 label-pure windows):
**coverage 95.1% at confident accuracy 0.9760.** Every operating point in
`runs/s4_fusion/fusion.md` clears the target.

---

## 1. Load-bearing constants that rest on thin evidence

### 1.1 The asymmetric physics veto was REMOVED **[measured 2026-08-03]**

`fuse.py` used to let physics override the model toward WALK but not toward STAND. When
adopted, physics was right **0.788** of the time in that direction. After the S2 feature
set absorbed `ileg_swaps` and the rest-anchored interleg block, the same measurement read
**0.566 on 76 windows**.

Removed, and the deciding measurement is worth keeping: turning it off changed the
confident numbers **not at all** — coverage 95.1%, accuracy 0.9760, 110 errors, before
and after. It moved only the LOW tier (0.6128 -> 0.5702) and the call-everything row
(0.9582 -> 0.9562). **The veto only ever relabelled windows the pipeline abstains from**,
so it bought nothing on the deliverable while adding a rule that needed defending.

The lesson is the reusable part: a constant justified against one measurement had
silently expired when the feature set changed underneath it, and the whole-corpus accuracy
figure (+0.2 points) made it look live. Re-check the direction whenever features change.

**S2 decides what the window is; S3 decides whether to believe it.** That split is now
enforced in code — `fuse()` never alters the label.

### 1.2 `DEFAULT_PROBA_FLOOR = 0.70` is a choice on a curve, not an optimum

Maximising coverage subject to the >=95% target always drives the floor to its minimum,
producing a policy that ignores the model's probability entirely. The floor is a
judgement about how much coverage a point of accuracy is worth. The whole curve is
re-measured every run in `runs/s4_fusion/fusion.md`; the shipped value is marked on it.

### 1.3 `SWAP_DELTA_DEG = 1.0` and the `0 / 1 / >=2` bands are NOT fitted — keep it that way

The swap rule's standing claim is that it has zero fitted parameters (§10): `1°` is a
sensor noise floor measured at 0.76-0.88° on three files independently, and `1 swap` is
the only integer between measured standing (0) and measured walking (2). Tuning either to
improve a score would convert the one label-free opinion in the pipeline into another
fitted model, and the fusion would then be two models agreeing with themselves.

---

## 2. Known-imperfect behaviour

### 2.1 Windows containing human `-1` still train **[accepted, Lu, 2026-08-03]**

`label_windows` drops `-1` rows and votes over the rest, so a window can be "label-pure"
on a minority of its rows: 92 of 4,812 trainable windows contain `-1`, 20 are more than
half `-1`, the worst is 93.5%. Accepted as ~1.9% label noise. `unknown_frac` rides on
every window, so if per-rev accuracy ever splits along it, this is the first place to look.

### 2.2 The session-level rest fallback was dropped

S3 previously had a per-rev standing reference for trials that never rest on their own.
It now shares `features.rest_reference` with S2 instead. That is a deliberate trade: one
definition of "where rest is" across both stages beats a better fallback in one of them,
because a swap count taken about a different origin than the `ileg_*` features the model
reads is a silent disagreement rather than a visible one. Recordings that never rest fall
back to a whole-recording median and carry `rest_offset_trusted = False`, which S4 turns
into a stated reason on the window.

### 2.3 `gyro_energy` fails the rate audit, on purpose

It sums over samples, so halving the rate halves it (median relative Δ 0.4994). It is the
audit's **negative control** — an audit that has never rejected anything is not evidence
that the others passed (§11.1). It is not used as a feature or a fusion input.

### 2.4 The abstention target is a pair, and one half alone is meaningless

Confident accuracy without coverage is gameable to 1.000 by abstaining on all but the
easiest window. Every report here prints both. Treat any future summary that quotes one
without the other as broken.

---

## 3. Things that could be wrong and have not been checked

### 3.1 There is no serve path for a raw device CSV

`stages/s2_ml/transform.py` reproduces the labeled rev2 features from a raw recording and
is verified to ~1e-13 on 19 paired files — but **nothing calls it in anger**. Its only
caller, `verify_transform.py`, passes `TRUST_UNCHECKED`, so `load_trust` has zero
callers and `check_axis_trust` has never run against a real trust record. The pipeline
currently labels *labeled* trials. Pointing it at `data/raw` is unfinished work, and the
axis guard is unexercised until then.

### 3.2 The lockbox has never been opened

`rev8` and `rev13` are sealed and excluded from every number in this repo. The honest
generalization estimate does not exist yet. Every figure quoted here is leave-one-rev-out
over the six training revs, which share a small subject pool — §7's caveat that
cross-subject claims are bounded by how few distinct subjects exist still applies.

### 3.3 Six revs is a small basis for a threshold

The probability floor, the veto direction and the tier boundaries are all measured over
six revs. Per-rev accuracy ranges roughly 0.91-0.99, so a single unusual subject moves
these numbers more than any tuning does.

### 3.4 `profile_incumbent` pairs raw and annotated files with 5 s of slack

`find_pairs` accepts a 5000 ms end-time mismatch while comparing rows by index — §7's
"never select by index" hazard with a wide tolerance. Tightening it would drop pairs from
an already small set of 19, so it is flagged rather than changed.

---

## 4. Deliberately not built

This repo labels **stand and walk**, and nothing else. Excluded on purpose, not by
oversight — the experimental work lives in a sibling repo:

- **new-class discovery** and the discriminator registry (the seam for adding classes)
- **the stairs-ascent rule** — and note it was measured to be ascent-only: `lift_rest`
  separates stairs-up from walking, but stairs-down sits *below* walking and overlaps
  decline, so no threshold over those descriptors can name it
- **the 11-class taxonomy** — a sibling repo re-codes the labels (`0` becomes IDLE,
  STAND becomes 10, WALK becomes 20). **Data and code are not interchangeable between
  the two.** The same `rev2/trial_1.csv` reads `[-1, 0, 10]` here and `[-1, 10, 20]`
  there; mixing them relabels every row silently, with no error raised.
- **hypothesis generation from plots** — S3 here is deterministic and has no agent

---

## 5. Corrections made to inherited work

Recorded because both sibling repos still carry the originals.

- **The rate-invariance audit was measuring its own filter.** Decimating a bare 200-sample
  window applies a 41-tap FIR through `filtfilt`, whose edge transient covers over half
  the window. That artefact alone moved `antiphase` by 0.1734 and condemned it
  `rate_dependent` — the literal gait signature, rejected on a filter edge. Decimating
  with 1.28 s of real context either side drops the move to **0.0005**. The control that
  says the fix did not just blunt the test: `gyro_energy` is unchanged at 0.4994 and
  still fails.
- **`transition_report` computed runs on a sequence it had just filtered.** 213 of 244
  counted "true transitions" spanned a deleted window, so the reported 185 on-time
  boundaries were mostly an artefact of the deletion. The honest count is 27 timeable
  transitions, 14 on-time.
- **`MIN_SEGMENT_SAMPLES` meant 1 s at 100 Hz and 0.2 s at 500 Hz.** Now a duration.
- **`profile_incumbent` scored the incumbent without gap segmentation** while the
  classifier was scored with it — the comparison was between two scoring pipelines.
- **A latent `NameError`**: one sibling's grow gate calls `_corr` without importing it.
