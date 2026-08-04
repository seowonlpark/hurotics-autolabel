# Operating points

The classifier never has to answer. Every row gets a **best guess**, a **confidence**, and —
below the threshold — an **`ambiguous` flag with a reason and an alternative**. The threshold
is the one knob that trades how much of a CSV gets labelled against how often those labels
are right.

This file is the evidence for choosing it.

---

## Read this before the tables

Two populations, and they disagree:

| | coverage | accuracy on committed rows |
|---|---|---|
| development subjects (leave-one-rev-out, 7 revs) | 84.66% | **0.9901** |
| **rev8 — sealed lockbox, one unseen subject** | 76.4% | **0.9308** |

**The 95% target is met on development subjects and missed on the lockbox.** Everything
below is measured on the development subjects, because the lockbox is spent and must not
be re-read. Treat the development numbers as an upper bound, not an estimate.

> **The two rows were measured with different feature sets [2026-08-04].** Until this date
> `roweval` fitted every column `build_windows` emits (42) rather than the 38 the champion
> declares, so the development row described a model that was never shipped. It has been
> re-measured at 38 — the change moved it by 0.0003, well inside the ablation's noise band.
> The lockbox row was NOT re-measured, because measuring it again is spending it again
> (§7). A provenance mismatch is the cheaper of the two costs, and it is stated rather than
> tidied away.

The gap is not sampling noise. On rev8, **walk recall is 1.0000 and stand recall is
0.5752** on committed rows; every confident error is standing called walking. And accuracy
there is *non-monotonic* in the threshold (0.8997 → 0.9308 → 0.9184 at 0.70/0.85/0.95),
so on that subject raising the threshold past 0.85 buys nothing at all. See `caveats.md`
§3.2 and §3.0.

---

## The three presets

Development subjects, row-level, 2 s window at 0.25 s inference stride.

| preset | threshold | coverage | accuracy on committed | worst subject |
|---|---|---|---|---|
| `high_coverage` | 0.70 | 92.75% | 0.9784 | 0.9099 |
| **`balanced` (default)** | **0.85** | **84.66%** | **0.9901** | **0.9525** |
| `high_precision` | 0.95 | 67.83% | 0.9980 | 0.9884 |

*Coverage* = share of rows the model commits to. *Accuracy on committed* = of those, how
many are right. *Worst subject* = the same accuracy on the single worst held-out
development subject — and rev8 came in below even that, which is the point of §3.2.

## Why 0.85 is the default

Pooled accuracy clears 95% from threshold 0.50 onward, so it is the wrong number to set a
threshold by: it averages a new subject together with six the model has effectively seen.
**0.85 is the lowest threshold at which every held-out development subject independently
clears 95%.**

| threshold | pooled | worst development subject |
|---|---|---|
| 0.80 | 0.9866 | 0.9383 |
| **0.85** | **0.9901** | **0.9525** |

## Per subject, at the default threshold

Each subject was scored by a model that never saw it — including its reference statistics,
so nothing about the held-out person entered the thresholds its rows were judged against.

| rev | rows scored | committed | coverage | errors kept | accuracy | stand recall | walk recall |
|---|---|---|---|---|---|---|---|
| `rev13` | 235,222 | 232,398 | 98.8% | 363 | 0.9984 | 0.9369 (n=5,754) | 1.0000 (n=226,644) |
| `rev2` | 362,401 | 277,106 | 76.5% | 1,331 | 0.9952 | 0.9689 (n=18,618) | 0.9971 (n=258,488) |
| `rev3` | 107,588 | 105,175 | 97.8% | 564 | 0.9946 | 0.9301 (n=8,064) | 1.0000 (n=97,111) |
| `rev4` | 20,726 | 16,750 | 80.8% | 75 | 0.9955 | 0.9797 (n=3,700) | 1.0000 (n=13,050) |
| **`rev5`** | 34,482 | 29,486 | 85.5% | 1,400 | **0.9525** | **0.6647 (n=4,175)** | 1.0000 (n=25,311) |
| `rev6` | 264,693 | 209,712 | 79.2% | 3,392 | 0.9838 | 0.9211 (n=42,967) | 1.0000 (n=166,745) |
| `rev7` | 219,180 | 182,778 | 83.4% | 3,266 | 0.9821 | 0.8719 (n=23,353) | 0.9983 (n=159,425) |

**Read coverage and accuracy together, never the accuracy alone.** A subject with high
accuracy over a small committed set is not one the pipeline handles well, it is one it
mostly declined to answer for. The pooled 84.66% is an average over subjects it claims 98.8%
of and subjects it claims 76.5% of.

**Then read the two recalls, because the accuracy column hides them.** The classes are
wildly unbalanced on the committed set — 90% of committed errors are stand called walk
(`runs/s2_ml/roweval_loro.md`) — so a subject can hold a high accuracy while missing most of
its standing. `rev5` holds 0.9525 accuracy while getting **a third of its standing wrong**:
0.6647 stand recall against 1.0000 walk recall, over 4,175 committed stand rows.

That is the same shape as the lockbox — 0.5752 against 1.0000 — **on a development
subject.** §3.2 currently reads the rev8 gap as evidence about unseen subjects; at least
part of it is evidence about subjects with small standing sets, and that part was visible in
development all along under a pooled number that averaged it out. The recall columns were
added on 2026-08-04 for exactly this reason: the direction-of-error section had been telling
the reader to watch stand recall since it was written, and no table reported it per subject.

`rev5` is the worst subject the 0.85 default is chosen against, and it is the *smallest*
development subject after `rev4` — 34,482 rows. A worst-subject floor set by one small
subject is a thin basis, which is §11.4's point, and it is why the lockbox mattered. It is
also the subject the transition audit times worst — median |offset| 6.41 s against a corpus
median of 0.73 s (`runs/s2_ml/transitions_loro.md`). Whether the bad timing and the missing
standing are one finding or two is open.

## Full curve (development subjects)

| threshold | coverage | accuracy | worst subject | wrong rows kept |
|---|---|---|---|---|
| 0.50 (no abstention) | 100.00% | 0.9555 | 0.8799 | 55,357 |
| 0.60 | 96.62% | 0.9682 | 0.8933 | 38,215 |
| 0.70 | 92.75% | 0.9784 | 0.9099 | 24,916 |
| 0.75 | 90.57% | 0.9824 | 0.9231 | 19,865 |
| 0.80 | 87.94% | 0.9866 | 0.9383 | 14,676 |
| **0.85** | **84.66%** | **0.9901** | **0.9525** | **10,391** |
| 0.90 | 79.53% | 0.9942 | 0.9663 | 5,727 |
| 0.95 | 67.83% | 0.9980 | 0.9884 | 1,650 |
| 0.98 | 54.85% | 0.9994 | 0.9937 | 390 |

Out of 1,244,292 scored rows over rev2-rev7 and rev13 (rows whose ground truth is a trained
class and which at least one window covers).

## How to read the tradeoff

Going **up** buys accuracy at a steepening price: 0.85 → 0.95 removes 8,741 of the 10,391
remaining errors but costs 17 points of coverage — 209,406 rows sent to review to fix 8,741
labels. Worth it only when a wrong label costs far more than an unreviewed one.

Going **down** is cheap at first and then not: 0.85 → 0.80 gains 3.3 points of coverage
(40,789 rows) for 4,285 new errors, while 0.70 → 0.50 gains 7.2 points for 30,441, because
by then the model is committing to rows it has no signal on.

No threshold reaches 100%. At 0.98 there are still 390 wrong rows in 682,437 — and on an
unseen subject the ceiling is far lower (§3.2).

## Setting it

```powershell
python -m stages.s2_ml.label input.csv --preset balanced          # default
python -m stages.s2_ml.label input.csv --preset high_precision
python -m stages.s2_ml.label input.csv --threshold 0.88           # any value in [0.5, 1.0)
```

The threshold applies at label time only. It does not change the model, so the same trained
champion serves every preset and sweeping it costs nothing. `label_all` takes the same two
flags and labels the whole corpus at one point.

**You can also move between points on a file already labelled, without re-running
anything.** The emitted frame carries `guess` and `confidence` on every row that was scored
— `guess` is the model's call whether or not it committed — so any threshold `t` is
recoverable from an existing output as `Label = guess where confidence >= t, else -1`, in
either direction. What it does **not** recover is `reason`: that is written only for rows
ambiguous at label time, so a row that becomes ambiguous under a higher `t` comes back with
an empty one. Re-derive post hoc when you want the call; re-run `label.py` when you want the
doubt explained.

**That recipe is only valid while abstention is a pure function of confidence, which is
true today because both other knobs ship OFF.** Turn on `BAND_ABSTAINS` or `PHYSICS_CEILING`
and a row can abstain at any confidence, so `confidence >= t` no longer reproduces what the
serve path did — silently, and in the direction of claiming more coverage than was
delivered. That is the same failure `roweval._committed` exists to prevent on the
measurement side (§6.7). If you flip either flag, re-run rather than re-derive.

**And it is exact everywhere except on the threshold itself** [measured 2026-08-04].
`confidence` is emitted rounded to 4 decimals while `ambiguous` was decided on the
unrounded float, and that float is a mean accumulated by `cumsum` — so a probability whose
exact value is 0.8500 comes back an ulp low, abstains, and still prints `0.8500`. A row
printing exactly `t` therefore cannot be re-decided from the column, and the recipe
**over-commits** it. Measured over the 78-file corpus: 450 such rows at the shipped
threshold and 4,352 at 0.98, under 0.3% of committed rows at every point, every one of them
carrying `reason=model_split` — the rows where the ensemble is most evenly divided.

Not a defect to route around so much as the resolution of a printed number. `label_all`
carries the count beside every swept coverage figure and refuses to publish the sweep if
the disagreement ever exceeds it (`runs/breakdown.md` §5); do the same if you re-derive by
hand, or re-run `label.py` when the rows on the boundary are the ones you care about.

## Ambiguity reasons

Every reason is validated by the accuracy the guess *would* have had if it had not been
flagged. A reason earns its place by sitting well below the confident set.

| reason | rows | accuracy of the suppressed guess |
|---|---|---|
| (confident, kept) | 1,053,405 | 0.9901 |
| `near_transition` | 69,600 | 0.6090 |
| `weight_shift_or_step` | 45,757 | 0.8767 |
| `model_split` | 42,576 | 0.8312 |
| `out_of_distribution` | 14,914 | 0.8427 |
| `posture_shift` | 11,698 | 0.9367 |
| `low_excursion_gait` | 6,342 | 0.7108 |

Independent check: the model abstains on **55.8%** of rows a human marked `-1` versus
15.3% elsewhere. `-1` never enters training, so the agreement is not circular (§5.2).

`near_transition` is the largest bucket and the weakest guess, and as of 2026-08-04 it is
measured rather than assumed. `runs/s2_ml/transitions_loro.md` times all 267 timeable
annotated boundaries against the model's own predicted state changes: the flag catches
**82%** of them, but **43% of its 384 contiguous regions contain no annotated boundary at
all** within a window. Read the 0.6090 against that — part of this bucket is the model
changing its mind where the annotation sees nothing to change about.

## The amplitude band, and why it is not a second knob

The **annotation-ambiguity band** is the interleg range where the two human labels both
occur — measured in `train.reference_stats` from the annotation and one label-free
descriptor, with no model output in either edge. Abstaining on all of it was the policy of
the deleted S4 fusion stage, which gave up ~30 points of coverage to it. That policy was
ported to this path, measured, and **left switched off** (`label.BAND_ABSTAINS = False`).

The reason is that it does not buy a better tradeoff; it only moves along the same one.
Measured leave-one-rev-out over 1,271,256 rows — *not* the 1,244,292 of the corrected curve
above, for the reason in the note directly below:

> **Measured before the 2026-08-04 feature-set correction**, so the three rows below come
> from a 42-feature fit. The no-band row is 67.32% / 0.9983 there against 67.83% / 0.9980
> in the corrected curve above — the arms move together and by far less than the margin
> being argued (5 points of coverage), so the conclusion stands. Re-running the two band
> arms would tighten the provenance and change nothing else; it is not blocked by anything
> except being worth doing.

| policy | coverage | selective acc | worst subject | errors kept |
|---|---|---|---|---|
| **threshold 0.95, no band** | **67.32%** | **0.9983** | **0.9904** | **1,413** |
| threshold 0.70, band abstains | 63.65% | 0.9956 | 0.9909 | 3,477 |
| threshold 0.85, band abstains | 62.04% | 0.9968 | 0.9908 | 2,433 |

Raising the threshold reaches **higher coverage, higher accuracy and the same worst
subject**. The band's best case was cross-subject consistency — it is trials, not rows,
that disagree about this range — and it does not win there either.

**The honest caveat, which is also why this cannot be settled by measuring harder:** inside
the band the *labels* are the untrustworthy thing, so errors scored there are partly
annotation noise, and the table above scores the band against exactly the labels it exists
to distrust. That makes it a judgement (§1.5), not a measured loss. An unmeasurable
judgement costing 22 points of coverage does not get to be the default — but the flag is
there, documented, for anyone who holds different evidence about the annotations.

**Why the deleted stage read it as a win, and this one does not.** On the 2 s window grid
the probability floor is a weak selector — sweeping it 0.50→0.90 moved coverage only
65.8%→63.5% — so almost anything looked like an improvement over it, and the band was never
compared against a floor swept far enough to be a rival. Extending that sweep before the
stage was removed put the band on roughly the *same* frontier as the floor, not below it:
11 confident errors at 65.1% coverage against ~13 interpolated for the floor alone, which
is noise at that count.

The row path does not share that weakness. A row's probability is the mean over ~8
overlapping windows, which sharpens the confidence *ordering* the threshold selects on —
the same reason `roweval` exists at all rather than quoting window numbers. That is why the
threshold wins here and merely ties there, and why deleting the stage cost no measurement:
**it was the weaker instrument reading the same quantity.**

## The physics floor and ceiling, and why they are the *other* knob that is off

`label.PHYSICS_CEILING` abstains where the swap rule is decisive and contradicts the model;
`label.PHYSICS_FLOOR` accepts 0.70 instead of the threshold where it is decisive and agrees.
**Both ship OFF, so every number above is a property of that.** The floor is *not* dominated
by the threshold, which makes stating the measurement here obligatory rather than optional.
The ceiling was not either, until the feature-set correction — see below.

Each arm matched against threshold-only **at the same coverage**, leave-one-rev-out over
1,244,292 rows (`roweval.physics_gate_table`). Negative = fewer surviving errors = better:

| threshold | committed errors | physics objects to | ceiling | floor |
|---|---|---|---|---|
| 0.50 | 55,357 | 16,003 (28.9%) | −1,958 | +8,309 |
| 0.60 | 38,215 | 8,255 (21.6%) | −1,986 | +3,218 |
| 0.70 | 24,916 | 3,470 (13.9%) | −1,092 | +0 |
| 0.75 | 19,865 | 2,033 (10.2%) | −465 | −466 |
| 0.80 | 14,676 | 1,003 (6.8%) | −422 | −1,480 |
| **0.85** | **10,391** | **580 (5.6%)** | **−11** | **−1,290** |
| 0.90 | 5,727 | 202 (3.5%) | −26 | −628 |
| 0.95 | 1,650 | 0 (0.0%) | +0 | +1,014 |
| 0.98 | 390 | 0 (0.0%) | +0 | +1,550 |

Physics is decisive on 1,167,595 of the 1,244,292 scored rows (1,134,679 agree, 32,916
contradict).

Read the third column first — it is the ceiling's entire addressable set. **At p ≥ 0.95 it
is exactly zero**, confirming the §1.1b/§1.1c retraction at the high-precision points: there
is nothing there for a ceiling to catch, and `+0` is the gate doing nothing rather than
breaking even.

**The ceiling's case at the shipped point did not survive the feature-set correction; the
floor's did** [measured 2026-08-04]. At 38 features the ceiling is worth **−11 errors** at
0.85 — noise — where the 42-feature fit put it at −147. Its addressable set barely moved
(647 → 580); what changed is that it now converts almost none of it. It is clearly worth
something only at 0.70 and below, where nobody operates. The floor still pays substantially
at the shipped point (−1,290) and still reverses sign at 0.95.

This is the second time a ceiling argument has been retracted by a measurement rather than
by an argument, and the reason is the same both times: S2 and the swap rule read the same
1 Hz-filtered interleg angle, so they are wrong together (`caveats.md` §1.1c) and how much
the gate appears to buy is mostly a fact about the feature set, not about the physics.
**Anything written against the −147 figure now has no measurement behind it.**

**They are off anyway, and the reason is not the measurement.** Every number in this
document — including the per-subject floor that picked 0.85 — is computed with both flags
False. Switching one on silently would make this file describe a policy nobody runs, which
is the exact failure that left the amplitude band ON in one stage and OFF in another for
weeks (§ above). To turn one on: flip it, re-run `s2_roweval`, and re-derive this document
from the new curve. The measurement has made that a decision someone can check rather than
argue.

## Reproducing

```powershell
python -m stages.s2_ml.train                        # champion + window-level CV
python -m stages.s2_ml.roweval                      # the tables above, incl. the gate sweep
                                                    #   + runs/s2_ml/transitions_loro.md
python -m stages.s3_physics.label_audit             # which trials contradict their labels
python -m stages.s3_physics.rate_audit              # body or clock? gyro_energy must FAIL
python -m stages.s3_physics.plausibility --calibrate --control   # file-level bounds
```

`roweval --lockbox` exists but **rev8 is spent** (read twice, 2026-08-03, both logged in
`caveats.md` §3.2). Do not run it again without a new sealed subject.
