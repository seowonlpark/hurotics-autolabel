# Caveats

Everything known to be weak, unverified, or deliberately left out. `DOMAIN_NOTES.md`
records what the corpus taught us; this file records what this pipeline is *shaky* about.

The goal it is judged against: **label stand/walk on any recording at >=95% accuracy on
the windows it claims, abstaining rather than guessing on the rest.** Anything that
threatens that number belongs here.

Current measured position, **development subjects** (leave-one-rev-out, 7 revs,
1,244,292 scored rows): **coverage 84.66% at 0.9901, worst subject `rev5` 0.9525**. Every
operating point in `runs/regen/s2_ml/roweval_loro.md` clears the target, and
`OPERATING_POINTS.md` is where the curve and the per-subject split are read.

There used to be a second pair quoted here — S4 fusion's window-grid coverage/accuracy.
**That stage is deleted (2026-08-04)**: it measured a policy `label.py` never ran, so the
number was true of nothing anyone shipped. One deliverable, one proof.

**How to read the window-grid numbers below.** Several entries quote a *confident
accuracy* over 2 s windows — 0.9805, 0.9972, 95.9%, 65.1%. Every one of those was
measured on the **S4 fusion stage, deleted 2026-08-04**. They are kept because the
measurement is what the entry is for, and most of them are before-vs-after *deltas*
whose value does not depend on the level. **None of them describes anything the
pipeline now runs.** The live pair is row-level and lives in `OPERATING_POINTS.md`.

**The lockbox does not.** rev8, held out of everything, scores **0.9308 on the rows it
commits to** (coverage 76.4%). The target is met on development subjects and **missed on
the one genuinely unseen subject**. §3.2 is now the most important entry in this file; read
it before quoting any number above it.

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

*(That A/B is a matched pair measured on the S4 fusion run of 2026-08-03, over 4,812
windows, and it is kept as measured because its value is the before-vs-after **delta**, not
the level. The stage it was measured on no longer exists, so there are no live numbers to
refresh it against — read it as a closed result, not a standing one.)*

The lesson is the reusable part: a constant justified against one measurement had
silently expired when the feature set changed underneath it, and the whole-corpus accuracy
figure (+0.2 points) made it look live. Re-check the direction whenever features change.

**S2 decides what the window is; S3 decides whether to believe it.** That split is now
enforced in code — `fuse()` never alters the label.

### 1.1b The physics is no longer an INDEPENDENT second opinion **[measured 2026-08-03]**

§1.1 recorded the symptom and misread the cause. The veto's hit rate fell 0.788 -> 0.566
not because the direction expired, but because **S2 absorbed the physics**: `ileg_swaps`
and `ileg_minhalf` are now model features computed from the same interleg signal the swap
rule reads. Two opinions over one signal are not two opinions.

Measured on the development revs, how often physics contradicts S2:

| S2 windows | physics contradicts |
|---|---|
| correct | 1.7% |
| stand called walk (the dominant error) | 31.4% |
| **errors at `s2_proba >= 0.95`** | **12.0%** |

So the guard is informative in aggregate — an 18x enrichment over the correct-window rate —
but **it fails precisely where it is needed**: when the model is confidently wrong, physics
usually agrees with it. Fusion abstains on only 39.6% of the dominant error.

That was the thing to fix if S4 was to be worth its complexity: the second opinion had to
read something S2 does not. Both routes were tried — see §1.1c, neither rescues a confident
error, and the reason is worth knowing. **When neither worked, the stage was deleted rather
than kept as an unearned second opinion (2026-08-04).** The swap rule survives where it was
never redundant: pointed at the *annotations*, in `label_audit.py`.

### 1.1c Two routes to an independent guard, both measured, neither shipped **[measured 2026-08-03]**

**Route A: take the interleg features away from S2.** Drop `ileg_swaps`, `ileg_minhalf`,
`ileg_minquarter` so physics reads a signal the model does not.

| S2 | fused coverage | fused accuracy | physics catches errors at p>=0.95 |
|---|---|---|---|
| with `ileg_*` | 95.84% | 0.9805 | 12.5% |
| without | 95.51% | 0.9811 | 15.4% |

Costs accuracy, buys 3 points on the guard, leaves it useless. **The correlation is not
about shared features.** Both opinions read the same 1 Hz-filtered interleg motion, and
where that signal is ambiguous they are wrong together however S2 is fed. Not adopted.

**Route B: give S3 a channel S2 has never seen.** The raw logs carry `L/R/B_Acc_*`; the
transform low-passes at 1 Hz, so every footfall impulse is gone before S2 looks. Acceleration
MAGNITUDE was used, not an axis, so it needs no sagittal lookup and cannot repeat §5's error.

It is genuinely informative (`R_acc_std`, AUC 0.88 mean / 0.83 worst over 3 revs) and
genuinely independent (fires on 6% of correct windows, 28% of errors). And it still does not
work, on the only 18 paired recordings that exist:

| S2 confidence | errors | physics | Acc | either | false alarms on correct |
|---|---|---|---|---|---|
| >=0.50 | 61 | 23.0% | 27.9% | **41.0%** | 7.6% |
| >=0.85 | 18 | 0.0% | 11.1% | 11.1% | 5.8% |
| >=0.95 | 7 | 0.0% | 0.0% | **0.0%** | 4.2% |

The guards are complementary where the pipeline already abstains — the union nearly doubles
overall error detection — and **empty where it does not**. At `p>=0.95` neither flags a
single error while Acc still fires on 4% of correct windows, which is a false-alarm
generator, not a safety net.

**The conclusion is about the errors, not the guards.** A confident error here is not a
window where one view dissents; it is a window that looks like the other class to *every*
measurement available. No second opinion over the same recording can catch it. What would:
a third class for the states `stand` is absorbing (§3.0), and more subjects.

Small sample, stated plainly: 18 errors at 0.85, 7 at 0.95, 3 revs, because only 18 of 41
labeled trials have surviving raw. Directional, not conclusive. Route B is also unusable on
an `lpf_view` CSV, which is most of the corpus.

### 1.1d A reason code was dead by construction and is removed **[measured 2026-08-03]**

`fuse.py` used to emit `verdict_from_grown_span` whenever a WALKING verdict needed the
§10.7 adaptive grow. The flag was never wrong, it was **uninformative**, in a way only
counting reveals:

| | fires | share of all windows | accuracy of those windows |
|---|---|---|---|
| `verdict_from_grown_span` | 4,800 | **75.7%** | 0.9815 |
| no reason at all (clean HIGH) | 786 | 12.4% | 0.9682 |

The grow is not the exception, it is *how walking is normally detected*: only 326 of about
5,000 WALKING verdicts come in at the base 2 s span. So the flag attached to three quarters
of the output, and it pointed the wrong way — the windows it marked were **more** accurate
than the ones it left clean. An audit trail that says "be careful" about most of the answer,
and preferentially about the correct part of it, is worse than silence.

This is the second instance of the failure: `label.py` lost a reason code (`moving_stand`,
since removed) the same way — a reason that cannot discriminate, because of how the
mechanism behind it behaves rather than because of a coding error. Both were caught the
same way — by measuring the accuracy of the guess the reason was suppressing, which is the
check every reason code in this pipeline now has to pass (`OPERATING_POINTS.md` runs it for
`label.py`'s surviving reasons, the table below for `fuse.py`'s). After the removal, every surviving reason sits well below the clean set:

| reason | pure windows | accuracy of the suppressed guess |
|---|---|---|
| (no reason: clean HIGH) | 3,834 | 0.9977 |
| `physics_contradicts` | 196 | 0.5051 |
| `model_split` | 309 | 0.6602 |
| `physics_abstains` | 318 | 0.8679 |
| `rest_zero_untrusted` | 78 | 0.8718 |
| `amplitude_ambiguous` | 1,930 | 0.9093 |

**What was NOT shipped, and why it is recorded here instead.** Span does carry a little
signal at the *ceiling*: windows whose grow ran all the way to 6.0 s score 0.9355 (n=667)
against 0.9883 for the HIGH tier at the time of measurement. That is a real gap, but it is a
*different* flag on much thinner evidence, and the relationship is non-monotonic — accuracy
by span runs 0.8657 / 0.9756 / 0.9908 / 1.0000 / 0.9889 / 0.9355 across the range, and span
correlates with error at only −0.0155. Adding it would be a new behaviour justified by one
bucket of a curve that does not otherwise trend. Left as a candidate, not a change.

### 1.2 The threshold is a choice on a curve, not an optimum

Maximising coverage subject to the >=95% target always drives the threshold to its
minimum, producing a policy that ignores the model's probability entirely. The threshold
is a judgement about how much coverage a point of accuracy is worth. The whole curve is
re-measured every run in `runs/regen/s2_ml/roweval_loro.md`, and `OPERATING_POINTS.md` marks
the shipped value on it with the reason: **0.85 is the lowest threshold at which every
held-out development subject independently clears 95%.**

*(This entry used to be about `fuse.DEFAULT_PROBA_FLOOR = 0.70`, S4's window-grid floor.
That stage is deleted; the argument was always about the shape of a selective curve and
transfers intact to the knob that survived.)*

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
on a minority of its rows: 94 of 5,984 trainable windows contain `-1`, 22 are more than
half `-1`, the worst is 93.5%. Accepted as ~1.6% label noise. `unknown_frac` rides on
every window, so if per-rev accuracy ever splits along it, this is the first place to look.

### 2.2 The session-level rest fallback was dropped

S3 previously had a per-rev standing reference for trials that never rest on their own.
It now shares `features.rest_reference` with S2 instead. That is a deliberate trade: one
definition of "where rest is" across both stages beats a better fallback in one of them,
because a swap count taken about a different origin than the `ileg_*` features the model
reads is a silent disagreement rather than a visible one. Recordings that never rest fall
back to a whole-recording median and carry `rest_offset_trusted = False`. `label.py`
appends that fact to every row's `reason_detail` rather than letting it pass silently.

### 2.3 `gyro_energy` fails the rate audit, on purpose **[closed 2026-08-04: the audit is deleted]**

It sums over samples, so halving the rate halves it (median relative Δ 0.4994). It is the
audit's **negative control** — an audit that has never rejected anything is not evidence
that the others passed (§11.1). It is not used as a feature or a fusion input.

### 2.4 Physics contradiction is asymmetric — measured, and deliberately NOT acted on **[measured 2026-08-03]**

`fuse()` sends every physics-contradicts-model window to LOW. The two directions are not
equally strong evidence, so the obvious proposal is to tier them differently. Measured on
196 contradicted development windows:

| direction | n | model right |
|---|---|---|
| model=stand, physics=walk | 79 | 0.4051 |
| model=walk, physics=stand | 117 | 0.5726 |

The asymmetry is real and has a mechanism — `>=2` committed alternations is *positive*
evidence the legs swapped, while `0` swaps is the *absence* of evidence, which slow or
in-phase gait produces as readily as standing. **But both directions sit far below the 0.95
target, so both correctly abstain.** Committing to the stronger one buys coverage 0.9586 ->
0.9691 and costs accuracy 0.9805 -> 0.9778, adding **17 confident errors** — the wrong
trade for a pipeline whose stated preference is to abstain rather than guess.

Recorded so it is not re-proposed. §1.1 removed an asymmetric rule that rested on 76
windows; this is the same rule re-derived from the other side, and it fails the same test.
A mechanism that is real is not by itself a reason to act on it.

### 2.5 The abstention target is a pair, and one half alone is meaningless

Confident accuracy without coverage is gameable to 1.000 by abstaining on all but the
easiest window. Every report here prints both. Treat any future summary that quotes one
without the other as broken.

### 2.5b Coverage was traded away twice, then the stage that traded it was deleted **[measured 2026-08-04]**

S4 fusion's final policy claimed **65.1%** of windows at confident accuracy **0.9972**, against **95.9%** at **0.9805** under its previous one. Nothing about the model changed between those two numbers — forced-call accuracy is 0.9622 either way — so they compare policies, not models. That is still the useful lesson; what changed since is that **both policies are gone with the stage.**

The trade did not survive contact with the unit that ships. Ported to the row path and measured over 1,271,256 rows, abstaining on the band reached 62.04% at 0.9968 while simply raising the threshold to 0.95 reached **67.32% at 0.9983 — more coverage, more accuracy, a third of the errors, same worst subject.** `label.BAND_ABSTAINS` is therefore `False` and the argument is recorded in `OPERATING_POINTS.md`.

**`BAND_ABSTAINS = True`** abstains on the whole 2.62–20.34° interleg band, a third of all
windows. The curve it was chosen from (`fuse.py`):

| band policy | coverage | accuracy | errors | errors removed per coverage point |
|---|---|---|---|---|
| abstain below 0.85 | 0.8917 | 0.9882 | 63 | 3.3 |
| abstain below 0.95 | 0.7828 | 0.9919 | 38 | 2.0 |
| **abstain on all of it** ← shipped | 0.6666 | 0.9947 | 21 | **1.5** |

The shipped point is the **worst exchange rate on its own curve**, and that is the caveat:
it is not chosen for error reduction, which would argue for 0.80. It is chosen so the
pipeline never claims certainty inside the range where the annotations contradict each
other. If you disagree with that argument, the flag is one line and the alternatives are
already measured.

**`MEDIUM_ABSTAINS = True`** then removes the `medium` tier from coverage: 96 windows at
0.8958 carrying 10 of the remaining 21 errors, the only confident tier under the 0.95
target. Costs 1.6 points of coverage at 6.3 errors per point — four times the band's rate.

Two consequences worth knowing. **The output is effectively binary** — `high` or abstain;
that stage's three tiers collapsed to `high`-or-abstain, which is one of the things that
made it hard to justify beside a row path already emitting a graded confidence. And **§2.5's companion heuristic stops applying inside the band**: the abstained
tier now scores 0.8972 rather than 0.5403, which reads like 2,000 wasted calls and is not.
The band is *defined* as the region where labels are unreliable, so agreement with them
there measures the annotator. Do not "fix" that number by re-admitting the band.

**Open:** whether 65% coverage is acceptable is a product question this repo cannot answer.
It depends on what the consumer does with a declined window. If there is no sane fallback
downstream, these two flags move the problem rather than solve it.

### 2.6 Nine alternative confidence signals were measured and rejected **[measured 2026-08-03]**

§3.2's finding — the confidence ordering does not transfer to a new subject — invites a
better confidence signal. Nine were built and scored. **None is measurably better than
`max(p, 1-p)`, and the one that looked best reverses under a stronger test.** Recorded
because the rejections are what stops this being re-proposed.

Two things had to be got right before any of it means anything.

**Recalibration cannot be the answer, and this is provable rather than measurable.**
Platt/temperature/isotonic scaling are *monotone* maps on `p`. They preserve the order of
every row, so accuracy at a fixed *coverage* is invariant under them; only the threshold
labels move. §3.2's non-monotonicity is an ordering defect — the rows dropped between 0.85
and 0.95 on rev8 were disproportionately **correct** — so no recalibration touches it.
Calibration remains worth doing for one narrower reason (below), but not for accuracy.

**The comparison must therefore be at matched coverage, not matched threshold.** Comparing
signals at threshold 0.85 measures their calibration; comparing at coverage 0.85 measures
their ordering, which is the thing in question.

Method: nested leave-one-rev-out over the 7 development revs. Each outer fold trains the
champion on 6 revs and, alongside it, 6 inner models each trained on 5 of those 6 — so
every inner model is blind to the held-out subject *and* to one other. Their spread is a
subject-shift sensitivity estimate that reads no held-out label. Window level, 5,984
label-pure windows. rev8 was not touched.

| signal | cov 0.95 | cov 0.90 | cov 0.85 | AURC |
|---|---|---|---|---|
| **`max(p, 1-p)` (shipped)** | 0.9782 / 0.9371 | 0.9870 / 0.9635 | **0.9902** / 0.9697 | **0.00501** |
| ensemble mean `max(p̄, 1-p̄)` | 0.9782 / 0.9433 | 0.9874 / 0.9635 | 0.9906 / 0.9697 | 0.00519 |
| `conf - 1·std` | 0.9778 / 0.9433 | **0.9877 / 0.9706** | 0.9908 / 0.9697 | 0.00525 |
| `conf - 10·std` | 0.9736 / **0.9568** | 0.9825 / 0.9701 | 0.9880 / 0.9695 | 0.00611 |
| pessimistic min over models | 0.9775 / 0.9433 | 0.9868 / 0.9708 | 0.9906 / 0.9697 | 0.00530 |

(pooled accuracy / worst held-out subject. AURC = area under the risk-coverage curve,
lower is better ordering.)

The `conf - λ·std` rows read as a win: **+0.0197 on the worst subject at coverage 0.95**,
and +0.0071 at 0.90 with pooled accuracy *also* up. Exactly the trade a subject-transfer
fix should make — give up a little on the average subject, buy the worst one.

**It does not survive strengthening the perturbation, and that is the finding.** Leaving
one of six subjects out barely moves the model (`p_std` mean 0.0160, max 0.1812). Rebuilding
the ensemble on 3-subject subsets — all C(6,3)=20 — nearly tripled the spread (mean 0.0428,
max 0.2877), so a real subject-shift signal should get **louder**:

| signal | cov 0.95 | cov 0.90 |
|---|---|---|
| `max(p, 1-p)` (shipped) | 0.9782 / 0.9371 | 0.9870 / 0.9635 |
| 3-subject basis, `conf - 1·std` | 0.9770 / 0.9371 | 0.9861 / 0.9638 |
| 3-subject basis, `conf - 10·std` | 0.9706 / **0.9306** | 0.9764 / 0.9493 |

It inverts. At λ=10 the worst-subject effect goes from **+0.0197 to -0.0065** when the
signal it is built from gets three times stronger. The 5-subject gain was noise on seven
subjects.

Stated exactly, because the shipped signal does not win everything: it has the best AURC
of the six that was computed over, and the best pooled accuracy at coverage 0.95, 0.80 and
0.70. `conf - 1·std` edges it at coverage 0.90 and 0.85 — by **+0.0007 and +0.0006, which
is 4 and 3 windows** out of the ~5,000 committed. That is the size of the entire pooled
case for every alternative here, and it is below anything seven subjects can resolve.

The reusable part is the test, not the result: **an effect that does not scale with the
mechanism it is attributed to is not that effect.** §1.1 is the same lesson from the other
direction — there, a constant kept a justification it had outlived; here, a candidate
acquired one it never had. Both were caught by re-measuring rather than re-reasoning.

**What is still open on this layer.** Calibration buys *coverage predictability*, not
accuracy: at threshold 0.85 the per-subject coverage runs 0.808 (rev2) to 0.992 (rev3), so
a preset does not mean the same amount of review work on different people. Fitting it on
inner-fold out-of-subject probabilities (never in-fold) would fix that and nothing else. It
is not built, and it must not be sold as closing §3.2.

**What this rules out.** The rev8 gap is not reachable from the confidence layer. §3.0
already names the cause — 86% of residual error is standing called walking, and all of it
on rev8 — and rev5 (§3.6) shows the *variance* between subjects sits there too. The
remaining moves are features that separate quiet standing, weight shifts and turning from
gait, or the third class §4 puts out of scope. Not the threshold, and not the estimator.

---

## 3. Things that could be wrong and have not been checked

### 3.0 STAND is the weak class, and the headline metric hides it

The corpus is **86% walking** (row level over the labeled revs; 85.8% of label-pure
windows), so accuracy is close to a walking detector's score. The dominant residual
failure, on every subject, is **standing called walking**: **96 of the 112** confident
errors in cross-validation (85.7%; the other 16 are walk called standing), and **100%** of
them on the lockbox. Quote macro-F1 or per-class recall; a pooled accuracy figure for this
task is close to meaningless (§5.4).

The cause is partly structural. `stand` is not one behaviour — it absorbs quiet standing,
weight shifts, turning, sitting and transfers, because the taxonomy has nowhere else to put
them. `posture_shift` exists as an ambiguity reason precisely to surface that, and its
rows do measurably worse than the confident set. A third class would address it properly;
that is deliberately out of scope here (§4).

**And the error rate is partly a measurement of the LABELS, not of the model
[measured 2026-08-03].** Of the **112** confident errors on development subjects, **87
(77.7%)** are windows where the physics *also* contradicts the annotation; on the other
**25** the physics sides with the label against the model.

That 87/25 split is the finding. (The tempting extra claim — "and on 100% of those the
physics agrees with the model" — is **not** evidence and was struck: with two classes,
`model != label` and `physics != label` forces `physics == model`. It is arithmetic, not
corroboration. §11.1 collects exactly this kind of check, which became tautological once
the quantity was defined by what it was tested against.)

What remains after removing the tautology is still substantive: a rule with no trained
parameter, which never sees an annotation, independently declines to read those 87 windows
the way the annotation does — and it *could* have gone the other way, as it did on 25.

    labeled stand, both read walking : 75 windows, median antiphase **+0.624**
    labeled walk, both read standing : 12 windows, median antiphase -0.361

Antiphase near +0.62 means the legs are firmly in opposition, which is what gait *is*.
This is the same signature as the two known label failures — `rev13/4`, and the STANDING
that included the accel/decel ramps.

Read carefully, because it does not license discounting the number. §1.1b is the reason:
the physics is **no longer independent** of S2 — both read the interleg signal — so "the
model and the physics both disagree with the label" is a weaker statement than it sounds.
A shared blind spot produces it just as readily as a bad label. `stand` also legitimately
contains ramps and turns that alternate the legs, and none of these 87 windows has been
adjudicated by a human.

What it does say is that **a confident-accuracy figure — 0.9901 on the shipped row path — is a lower bound on model quality and an upper bound on label quality, and the two cannot be separated by measuring harder.**
Separating them needs a person on the queue — `label_audit.py` is what nominates for it
now, model-free and per trial — or a second opinion that reads a channel S2 does not
(§1.1b), which was measured and not found.

This also bears on §3.2: rev8's failure is 100% standing-called-walking with stand recall
0.5752, the same signature. Whether rev8 is a worse subject or a worse-labelled one is
**not currently distinguishable**, and cannot be settled by re-reading it.

### 3.1 The raw-device serve path exists now — and building it found a wrong axis **[built + measured 2026-08-03]**

This entry used to read "nothing calls `transform.py` in anger": its only caller,
`verify_transform.py`, passed `TRUST_UNCHECKED`, so `load_trust` had zero callers and
`check_axis_trust` had never once run against a real record. `label.py` now takes either
shape of file — an `lpf_view` file, or a raw device log bridged by
`transform.raw_csv_to_features` — dispatched on the family marker column resolved by name
(§1.3). Every guard runs on every raw file, against that file's own trust record.

**The first thing the path did when pointed at real files was fail.**
`SAGITTAL_DEG_AXIS_BY_VARIANT` mapped `fb5ea2c2` to `Deg_X`. That is the majority variant —
62 of 91 raw files — so 68% of the corpus would have been served the frontal plane instead
of the sagittal one, silently, with every downstream number still looking plausible. The
correction and its evidence are in §5. **This is the entry's real lesson: the gap was never
the missing feature, it was that an unexercised path cannot be wrong out loud.**

Verified two ways, both re-runnable — and **both re-run on 2026-08-04**, reproducing every
number below unchanged. They had been deleted from the working tree at that point, silently
and without an entry in any of the deletion tables this repo keeps; they are restored and
wired back into `run_pipeline.py --verify`. **An evidence script that can be deleted without
a doc noticing is a claim with no owner** — §1.4 in its purest form, and the fix is that the
spine can now run them rather than that a note remembers to.

- `python -m stages.s2_ml.verify_transform` — 18/18 pairs reproduce the labeled columns to
  **≤7.3e-13**. It now reads through the same loader the serve path uses, so what it
  blesses is the read production performs, and it no longer honours the label quarantine
  (`EXCLUDED_TRIALS`), which has nothing to do with a label-free column check.
- `python -m stages.s2_ml.verify_serve` — the same recording labelled BOTH ways, raw and
  `lpf_view`, compared row for row: **536,590 rows, 0 disagreeing verdicts, max |Δconfidence| =
  0**, plus a corpus sweep (**78 of 91 files servable, 86%**; 13 abstain). No `Label`
  column is read, in code and not by intention, so the rev8 lockbox stays sealed (§7).
  *(The "12 unmapped variant, 1 with no trust record" split once quoted here does not
  reproduce: `labeled_raw/abstentions.jsonl` is 13 rows, every one an
  `UnknownVariantError`. `DOMAIN_NOTES` §6.2 records why that split was never a
  partition of the files in the first place.)*

What is still thin about it:

- **`fb5ea2c2`'s axis rests on one subject-day.** All 7 of its pairs are rev13. The variant
  is documented as covering rev14 too, and rev14 has no annotated counterpart to check.
- **86%, not 100%.** `e5f2660f` and `86069795` have no paired recording, so they abstain —
  correct behaviour (§6.2: signal-only axis detection scores *below* chance), but it means
  the newest two hardware batches cannot be served at all until one paired file exists.
  **What unblocks it is one annotated export per variant, not more raw data** — and what a
  variant *is* has never been confirmed upstream. Both are written as asks in
  `DOMAIN_NOTES` §6.3.
- **Agreement is not accuracy.** The pair check proves the raw route has no train/serve
  skew against the `lpf_view` route. It says nothing about whether either route is *right*:
  serving a raw file inherits §3.0 and §3.2 whole, including the lockbox subject the
  target is missed on.
- **The path reproduces an upstream bug on purpose.** `serve_dt` filters with the *final*
  inter-sample interval because `timestamp.m` does, and the training features carry that
  quirk. Using the honest median instead moves features by up to **20.1 deg** on the
  verifiable pairs. If HUROTICS ever fixes `timestamp.m`, the labeled features change and
  this choice must be re-measured, not preserved out of habit.

### 3.2 The lockbox WAS opened, and the target is missed on it **[measured 2026-08-03]**

`rev8`, held out of feature selection, model selection and threshold selection, scores at
the shipped `balanced` threshold (0.85):

| | coverage | accuracy on committed rows |
|---|---|---|
| development subjects (LORO, 7 revs) | 84.7% | 0.9901 |
| **rev8 (lockbox)** | **76.4%** | **0.9308** |

**The two rows no longer come from the same feature set [2026-08-04].** `roweval` was
fitting 42 features where the champion declares 38; the development row has been
re-measured at 38 (it moved 0.0003), the lockbox row has not and will not be. Re-running
`--lockbox` to make the provenance tidy would be a third read of a spent subject, which is
the one thing §7 exists to stop — and "the numbers in the table match" is precisely the
kind of reason that makes re-reading feel harmless. The gap below is far larger than the
correction, so nothing in this section turns on it.

**The gap is the finding, not the noise.** Three things make it worse than one number:

- **It is entirely one class.** On rows rev8 was confident about, walk recall is **1.0000**
  (26,581 rows) and stand recall is **0.5752** (5,172 rows). All 2,197 confident errors are
  standing called walking. Pooled accuracy hides this because rev8 is 72% walking; a
  "predict walk always" model scores 0.725 on it.
- **The confidence ordering does not transfer.** Accuracy is *non-monotonic* in the
  threshold on rev8: 0.8997 at 0.70, 0.9308 at 0.85, and back **down to 0.9184 at 0.95**.
  Abstention is supposed to buy accuracy monotonically. On a genuinely new subject here it
  does not, which means the threshold chosen on development subjects is not transferable
  and the `worst subject` column in `OPERATING_POINTS.md` is an optimistic floor.
- **It is concentrated.** Per trial: t2 1.0000, t3 1.0000, t1 0.9580, **t4 0.8779**. And t4
  is independently the highest physics-label disagreement of any non-flagged trial in the
  corpus (0.1429, §3.5). Whether that is model failure or label noise is **unresolved**,
  and cannot be resolved without re-reading a spent lockbox.

Read twice, both times honestly: once while rev13 was still sealed (0.9287), once after
freezing (0.9308). No decision in this repo was made using either result — the changes in
between were driven by rev13 and the development revs. It must not be read a third time.

### 3.3 Seven revs is a small basis for a threshold

The probability floor, the tier boundaries and the abstention threshold are all measured
over seven revs. Per-rev accuracy ranges roughly 0.91-0.99, so a single unusual subject
moves these numbers more than any tuning does — and §3.2 is exactly that happening.

### 3.5 One trial was quarantined for a label error, and the detector is new

`rev13/4` is excluded (`dataset.EXCLUDED_TRIALS`): 12,691 rows all annotated `stand`,
containing two runs under that one label — 7.0 s at 2.1 deg/s, then **119.9 s at 45.3
deg/s** with 71 deg of interleg swing. rev13's own labelled walking runs measure 35-50
deg/s. The second run is walking.

The evidence is internal to the file, so it stands with no model at all. That mattered:
`stages/s3_physics/label_audit.py` scores every trial's labels against the swap rule and
is **deliberately model-free**, because flagging the files the classifier dislikes would
delete precisely the hard cases, improve every metric and teach nothing. It does import
`s2_ml.dataset` and `s2_ml.features` — for trial loading and the window grid, so that it
scores the same windows S2 does — and that is fine. **The property to protect is that it
never imports a model, a probability or a prediction:** no `champion`, no `oof`, nothing
under `s2_ml.train`. Check for those, not for the string `s2_ml`.

Two things remain open. The flag line (`disagree > 0.50` over >=20 windows) has separated
exactly **one** case from 42 — rev13/4 at 0.95, next worst 0.30 — so it is validated
against a single positive example and its false-positive rate is unmeasured. And the
next-worst trials (rev5/3 at 0.30, rev8/4 at 0.14) have **not** been adjudicated; they may
be partial label errors that nothing currently excludes.

### 3.6 Nothing currently predicts WHICH new subject will degrade **[measured 2026-08-03]**

rev8 fell to 0.9308 and **no signal available beforehand would have warned us**. Five
candidates were built and tested against the seven development subjects (Spearman ρ
against selective accuracy at 0.85; all five are label-free, so all are available at serve
time):

- **Feature novelty** (share of features outside the training 1-99 percentile range):
  correlation with selective accuracy **+0.12**, with raw accuracy **-0.07**. No signal.
  rev2 has the highest novelty and the second-best accuracy; rev5 the lowest novelty and
  the worst. Not shipped — a safety signal with no signal is worse than none, because it
  gets trusted.
- **Model-physics disagreement rate**: ρ = **-0.57**. Real, and the best candidate found,
  but it rests on seven points and rev5 contradicts it outright. Not shipped as a gate;
  worth reporting per file as a diagnostic once there are enough subjects to calibrate it.
  **§1.1b caps how good this can get**: the swap rule reads the same interleg signal S2
  absorbed as `ileg_swaps` / `ileg_minhalf`, and physics contradicts S2 on only 12.0% of
  its confident errors. A flag built on it inherits that blind spot — rev5 has the second-
  *lowest* disagreement rate (0.026) of the seven and the worst accuracy.
- **Abstention rate** — the obvious candidate, and it fails: ρ = **-0.50**, and it fails
  for a *structural* reason rather than a numerical one. Abstention is derived from
  confidence, and every residual error in this pipeline is a **confident** one (§3.0). A
  flag routed through confidence cannot see the failure mode by construction, so this is
  not a sample-size problem that more subjects would fix. rev5 has the **lowest** abstention
  rate of all seven subjects (0.083) and the worst accuracy; rev2 has the second-highest
  (0.193) and the second-best. Mean confidence (ρ = +0.50) fails identically, for the
  identical reason.
- **Subject-holdout ensemble disagreement** (spread across models each blind to one
  training subject, §2.5): ρ = **-0.57** for the mean spread, **-0.61** for the share of
  windows with any disagreement. Ties the physics rate, and §2.5 shows the underlying
  signal does not survive strengthening. Not shipped.
- **Predicted walk fraction: ρ = +0.75, and the sign is the point.** Subjects predicted
  *mostly walking* score **better**. This is not a difficulty measure that came out
  backwards — it is class balance wearing a difficulty measure's clothes. The only failure
  mode is overwhelmingly standing called walking, so a recording with little standing in
  it has few chances to fail, and any statistic of the predictions inherits that. **It is the
  strongest correlation of the five and it is worthless.** Recorded because it is
  cheap, available, points the wrong way, and would be trusted.

The last one generalises: on a corpus this imbalanced, *any* flag built from prediction
statistics is measuring the class mix and not the difficulty. Check a candidate's sign
against `pred_walk_frac` before believing it.

Practical consequence, and it is the honest answer to "will it break on a new person":
**it might, and you will not know in advance.** What is available instead:

1. Watch **stand recall**, not accuracy. Every failure on every subject is standing called
   walking, so stand recall is the canary and pooled accuracy is the anaesthetic (§3.0).
2. Do not assume a higher threshold rescues a bad subject. On rev8 accuracy was
   *non-monotonic* in the threshold (§3.2) — 0.85 was better than 0.95.
3. The one thing these signals *can* do is separate **confidently easy** from the rest:
   rev3 and rev13 sit near zero on every candidate above and are the two best subjects.
   None of them rank *within* the rest, and the whole risk lives there. A flag that only
   says "this one is fine" is worth having, as long as silence is never read as a warning.
4. More distinct subjects is the only real fix. Seven is not enough to calibrate any of it.

### 3.4 `find_pairs` matches raw to annotated with 5 s of slack — measured harmless **[measured 2026-08-03]**

`find_pairs` accepts a 5000 ms end-time mismatch while comparing rows by index — §7's
"never select by index" hazard with a wide tolerance. It was flagged rather than changed
because tightening it would have shrunk an already small set. There is now **one** copy,
in `verify_transform.py`; the second (`profile_incumbent.py`, matching the subset of raw
files carrying `loco`) went with the incumbent-profiling code and is no longer in this
repo, though the sibling repos still carry it.

Now measured rather than assumed: across all 43 annotated trials, **every one matches
exactly zero or one raw candidate**, and each of the 18 that match has a raw `Time` vector
identical to the annotated one — `max|Δt| = 0`, not merely within tolerance. So the slack
buys nothing and risks nothing on this corpus, and index comparison is sound here because
the two files are the same recording exported twice. It is still the wrong *rule* to carry
into a corpus where it hasn't been measured, so it stays listed. (The correct count is
**18** pairs; the "19" quoted here and in DOMAIN_NOTES §6.2 was never reproducible.)

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

- **The sagittal axis for the majority variant was wrong [measured 2026-08-03].**
  `SAGITTAL_DEG_AXIS_BY_VARIANT` mapped `fb5ea2c2` — 62 of 91 raw files, rev13/rev14 — to
  `Deg_X`, and DOMAIN_NOTES §6.2's table said the same, tagged `[measured]`. Every pair the
  corpus holds contradicts it. Reproducing the labeled columns from raw, per axis, max
  abs error:

  | variant | pairs | `Deg_Y` → `Gyro_Z` | `Deg_X` → `Gyro_X` | `Deg_Z` → `Gyro_Y` |
  |---|---|---|---|---|
  | `fb5ea2c2` | 7 (rev13) | **≤2.4e-13** | 208–368 | 193–460 |
  | `0fda484e` | 10 (rev7, rev8) | **≤7.3e-13** | 177–232 | 99–350 |
  | `4bfd6ab2` | 1 (rev4) | **6.8e-13** | 381 | 247 |

  Bit-exact reproduction of all four columns on both legs over up to 132,005 rows is not a
  coincidence, and the losing axes are not near-misses. Corrected to `Deg_Y` on all three;
  `verify_transform` now passes 18/18 where it had been failing 6.

  Two things let it stand. It was **unreachable**: `verify_transform` was the only caller,
  and re-running it is what exposes the failure, so the bug was one command away the whole
  time and nothing ran that command. And it was **self-justifying in prose** — the source
  comment and §6.2 both argued at length that `fb5ea2c2` is "not an anomalous permutation,
  simply a revision whose sagittal plane is `Deg_X`", which reads as evidence and is an
  explanation of a number nobody re-derived. §6.2 also claimed 19 pairs; there are 18.

- **The rate-invariance audit was measuring its own filter.** Decimating a bare 200-sample
  window applies a 41-tap FIR through `filtfilt`, whose edge transient covers over half
  the window. That artefact alone moved `antiphase` by 0.1734 and condemned it
  `rate_dependent` — the literal gait signature, rejected on a filter edge. Decimating
  with 1.28 s of real context either side drops the move to **0.0005**. The control that
  says the fix did not just blunt the test: `gyro_energy` is unchanged at 0.4994 and
  still fails.
- **`transition_report` computed runs on a sequence it had just filtered.** 213 of 244
  counted "true transitions" spanned a deleted window, so the reported 185 on-time
  boundaries were mostly an artefact of the deletion. The replacement count quoted here is
  27 timeable transitions, 14 on-time. **It does not reconcile with the independent
  measurement added 2026-08-04** — `stages/s2_ml/transitions.py` scores the raw `truth`
  column with the unknowns left in rather than filtering first, over the leave-one-rev-out
  frame, and reports **308 candidates, 267 timeable, 170 of 266 within ±1 s**
  (`runs/regen/s2_ml/transitions_loro.md`). Neither figure is derivable from the other and the
  older pair predates the deleted S4 window grid, so **[open]** — settle it before quoting
  either.
- **`MIN_SEGMENT_SAMPLES` meant 1 s at 100 Hz and 0.2 s at 500 Hz.** Now a duration.
- **`profile_incumbent` scored the incumbent without gap segmentation** while the
  classifier was scored with it — the comparison was between two scoring pipelines.
  (That module is no longer in this repo; the correction is recorded because the sibling
  repos still carry the original.)
- **A latent `NameError`**: one sibling's grow gate calls `_corr` without importing it.
- **Two rest-calibration implementations existed, and they had drifted apart [removed
  2026-08-03].** `rest.rest_anchor` and `features.rest_reference` both searched for the
  recording's rest span and both returned an interleg zero, on *different* preference
  orders — `rest_anchor` carried a session-level fallback to a sibling trial that
  `rest_reference` does not have (§2.2 records why that fallback was dropped). Nothing
  called `rest_anchor`; the pipeline has always run `rest_reference`. The danger was not
  the dead code, it was that the two would be read as interchangeable and the wrong one
  cited as what the pipeline does. `rest_reference` is now the only implementation, and it
  has to be: it returns the per-side postures *and* the interleg offset from the same
  chosen span, so a second searcher could centre the angle features on one rest and the
  interleg features on another. `stillest_rest_offset` went with it (sole caller).

### 5.1 Dead code removed in the same pass **[2026-08-03]**

None of these changed a label, a confidence, or a number in any report — verified by
re-running S3 and S4 and diffing the reason table. Listed so the sibling repos can decide
independently.

| removed | why |
|---|---|
| `oof._folds_match_logo` | Never called, and placed *below* `if __name__ == "__main__"` so it could not be. Its docstring said it was "kept as an assertion rather than a comment" — it was neither. It was also tautological: LOGO yields one group per fold by definition. The guarantee it claimed to check is now stated where the loop is. |
| `rate_audit.decimate_window` | Its own docstring said `audit_anchors` does not call it and that on a bare window the result is mostly filter transient — i.e. it documented itself as unused and wrong. "Callers outside the audit may want it" is speculative generality; there are no such callers. |
| `s1_clean/peek.py` | 45-line parquet viewer, zero references in code, README, or notes. |
| `fuse.R_GROWN_SPAN` | Dead by construction — see §1.1d. |

Two stale *claims* were corrected in the same pass, which matter more than the dead code
because they were being read as current: `fuse.py` and `agents/s4_fusion.py` described S2 as
"a Random Forest over 23 windowed statistics" (it is ExtraTrees over 42), and `anchors.py`
opened by asserting the swap rule is "independent of S2" and "reads a different quantity" —
which §1.1b measured to be false. That claim is the load-bearing one for the whole fusion
argument, so it is now retracted in place rather than quietly softened.
