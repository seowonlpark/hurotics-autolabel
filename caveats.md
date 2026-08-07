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

**The lockbox does not.** rev8, held out of everything, scores **0.9269 on the rows it
commits to** (coverage 75.6%). The target is met on development subjects and **missed on
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

### 1.6 The corpus was described by filename, clock and hand-edited constants **[replaced 2026-08-07]**

Four facts about this corpus lived in code rather than in data, and each was reached by a
different mechanism nobody could check:

| fact | how it was reached | what could go wrong silently |
|---|---|---|
| which subject a trial belongs to | `re.search(r"(rev\d+)", path)` | no match fell back to the literal `"rev?"` — a renamed corpus puts **every** subject in one group, `LeaveOneGroupOut` becomes meaningless, nothing raises |
| which raw file backs an annotation | `abs(t[0]-r0) < 50 and abs(t[-1]-r1) < 5000 and nr == len(t)` over all 91 raw files | two recordings starting within 50 ms with equal row counts pair wrong |
| which subject is sealed | `DEFAULT_LOCKBOX_REVS = ("rev8",)` | — |
| which trial is quarantined | `EXCLUDED_TRIALS = {("rev13", 4)}` | the reason lived in a comment beside it, not in the record |

All four are `data/corpus.json` now, read by `stages/s2_ml/corpus.py`, which validates on
load: unknown split, missing file, duplicate `(subject, session)` and an empty training set
are each fatal rather than absorbed. A quarantine carries its **reason** as a string, so
`breakdown` and `inspect_window` print why instead of printing a bare pair.

**Proven equivalent, not assumed.** The manifest was generated by transcribing what the old
code resolved, then checked three ways: the 18 declared pairings are identical to the ones
the scan produced (cross-checked against the shipped `raweval.json`'s 13, which is the same
set after the lockbox and quarantine drops); `locoeval.json` and `model_meta.json` re-run
byte-identical; `verify_transform` still passes 18 pairs at `max_err = 7.25e-13`.

**The limit is not a gap to close.** `subject` is the CV group, and the whole leave-one-rev-out
claim rests on two subjects being two people. **Nothing in the data can establish that** — not
this manifest, not a stricter validator, not a smarter loader. If one person were recorded twice
under two IDs, every fold would train on them and test on them, and the corpus would look exactly
as it does now. It is a fact about who was in the room, held by whoever ran the sessions.

So it is a **declaration**, and the point of §1.6 is that it is now written down as one. Do not
file this as a missing check and do not try to add one; the honest handling is a person
confirming the roster. The one adjacent thing that *is* checkable — two different subjects
claiming the same `raw` file — is currently **not** checked and would be worth adding.

Otherwise the manifest is hand-edited, exactly as the constants were. What changed is that it is
one file, validated on load, with reasons attached, instead of four mechanisms in three modules.

### 1.7 `raweval` measured what `verify_serve` already proved, and is deleted **[removed 2026-08-07]**

`stages/s2_ml/raweval.py` scored the raw device route against human labels and reported it beside
`roweval`'s `lpf_view` number. It read as a second, independent confirmation. It was not one.

`verify_serve` labels the same recording **both ways** and compares every column a caller
receives for exact equality, with confidence held to 1e-9 and coverage mismatches counted
separately. Identical verdicts against identical truth give identical accuracy — necessarily, not
approximately. An aggregate accuracy figure can absorb a handful of flipped rows; an
exact-equality check over 536,590 rows cannot. The weaker instrument was the one being quoted.

The same applies to the axis-map story `raweval` was justified by. A mis-mapped sagittal channel
breaks `verify_transform` at 1e-9, immediately and loudly. Catching it downstream as "one
variant's accuracy sits apart from the others" is strictly worse: later, noisier, and it needs a
human to notice a table.

**Measured before deleting, not argued.** Last run, on the three subjects it could pair:

| route | coverage | selective accuracy |
|---|---|---|
| raw device (`raweval`) | 0.9091 | 0.9914 |
| `lpf_view` (`roweval`, same three subjects) | 0.9088 | 0.9914 |

**What was done first, because deleting it removed a backstop.** `verify_serve` compared five
verdict columns and `label_csv` delivers nine. `reason_detail` was among the unchecked, and it
carries a suffix derived from `rest_trusted`, which is computed **per route** — so the one column
that could genuinely diverge between routes was the one nobody compared. `VERDICT_COLS` now reads
`label.LABEL_COLUMNS`, so a new column joins the comparison by existing and a renamed one raises.
Re-run over the widened set: **18 pairs, 536,590 rows, 0 disagreeing verdicts, max |Δconfidence| = 0.**

`raweval`'s per-variant table moved to `verify_serve` as **coverage, not accuracy**: every
servable variant must have a pair in the comparison, so one cannot pass by being absent. It is
wider there — 78 servable files rather than the 18 paired ones. All three variants are covered
today (10, 7 and 1 pairs); a variant with none is reported, not failed, because a freshly
collected variant legitimately has no annotation yet.

**What is genuinely lost.** `raweval` was the only place human labels were joined onto the raw
file's **own timestamps** rather than the export's grid. That matters only if the two grids can
differ, and `verify_transform` compares the derived channels element-wise — which cannot pass
unless they line up. So the loss is real and empty at the same time: no check disappeared, one
redundant statement of an existing check did.

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

### 2.3 `gyro_energy` fails the rate audit, on purpose **[live — the gate runs every pipeline; tag corrected 2026-08-07]**

It sums over samples, so halving the rate halves it (median relative Δ 0.4994). It is the
audit's **negative control** — an audit that has never rejected anything is not evidence
that the others passed (§11.1). It is not used as a feature or a fusion input.

`rate_audit.py` gates on it in both directions: a newly `rate_dependent` anchor exits non-zero,
and so does a *silent* control (`EXPECTED_RATE_DEPENDENT`), because a sweep that rejects nothing
leaves every `invariant` verdict in that run unverified. The spine runs it as `s3_rate_audit`.

**This entry was tagged `[closed 2026-08-04: the audit is deleted]`, and it was not.** What went
on 2026-08-04 was S4 and `s3_physics/run.py`, taking `anchors.csv` and the per-window tally with
them (`DOMAIN_NOTES` §10.5). The audit was **re-homed** to `stages/s3_physics/rate_audit.py`, not
removed — the same file that still reproduces that run's table exactly (`antiphase` 0.0005,
`gyro_energy` 0.4994). A note closed by proximity to a deletion rather than by checking whether
its subject survived, which left the docs claiming a live gate was gone.

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

**So `confidence` is not a probability, and no consumer may treat it as one.** It is
`max(p, 1-p)` over a `class_weight='balanced'` ensemble, averaged across the ~8 windows
covering the row, then folded at 0.5 — three steps away from a frequency, none of them
undone. The shipped 0.85 does not mean "85% likely"; it is the lowest threshold at which
every held-out development subject independently clears 95% (`OPERATING_POINTS.md`), an
empirical cut point on an ordering. It is fit for exactly one operation — comparison
against a threshold. Multiplying it, averaging it, feeding it to a downstream expectation
or reporting it to a user as a percentage is wrong, silently. And per the rounding note in
`OPERATING_POINTS.md`, the emitted column cannot even re-derive its own `ambiguous` flag,
so `Label` is the answer to "may I use this row", never a comparison you do yourself.

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
0.5622, the same signature — and the `walk → stand` confusion cell being exactly 0 is now in
the artifact rather than asserted from a printout. Whether rev8 is a worse subject or a
worse-labelled one is **not currently distinguishable**, and re-reading it does not settle
it: the 08-07 read confirmed the shape and moved nothing on this question.

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
  **≤7.3e-13**. It reads through the same loader the serve path uses, so what it blesses is
  the read production performs, and it does not honour the label quarantine, which has
  nothing to do with a label-free column check. The 18 pairs are declared in
  `data/corpus.json` rather than searched for (§1.6), so only those files are opened.
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

### 3.2 The lockbox WAS opened, and the target is missed on it **[re-measured 2026-08-07]**

`rev8`, held out of feature selection, model selection and threshold selection, scores at
the shipped `balanced` threshold (0.85):

| | coverage | accuracy on committed rows |
|---|---|---|
| development subjects (LORO, 7 revs) | 84.66% | 0.9901 |
| **rev8 (lockbox)** | **75.55%** | **0.9269** |

**Read three times: twice on 2026-08-03, once more on 2026-08-07.** Both rows now come from
the champion's 38 features, so the provenance mismatch this section used to carry is gone —
and the version of this section that carried it said in as many words that re-reading to
tidy provenance was the one thing not to do. **It was done anyway, deliberately, and the
reason was not provenance.** The 08-03 read predates `per_rev`, `confusion` and
`physics_gate`, which landed 08-04; those are not cosmetic fields, and the third read is the
only way rev8 ever gets a physics-gate table. What it bought is in §3.2b, and it is
load-bearing. The 08-03 artifact is preserved beside the new one as
`runs/keep/s2_ml/roweval_lockbox.2026-08-03.json` — the third read does not erase the second.

**The number moved down: 0.9308 → 0.9269, coverage 76.4% → 75.6%.** That is the 42→38
feature correction landing on this row, the same correction that moved the development row
by 0.0003. It moves rev8 by 0.0039, ten times as far, which is itself a small piece of
evidence that this subject sits somewhere the model is less stable. The gap below is far
larger than either correction, so nothing in this section turns on it.

**Do not read it a fourth time, and do not tune on the third read.** The gate table in §3.2b
is the dangerous artifact in this repo now: it is a policy comparison on a sealed subject,
which is exactly the input a lockbox must never provide. It is recorded as confirmation of a
decision already made on development subjects, not as grounds for a new one.

**The gap is the finding, not the noise.** Three things make it worse than one number:

- **It is entirely one class, and the artifact now proves it rather than asserting it.** On
  rows rev8 was confident about, walk recall is **1.0000** (26,160 rows) and stand recall is
  **0.5622** (5,247 rows). The confusion cell `walk → stand` is **0**: all 2,297 confident
  errors are standing called walking, not most of them. Pooled accuracy hides this because
  rev8 is 72% walking; a "predict walk always" model scores 0.725 on it.
- **The confidence ordering does not transfer.** Accuracy is *non-monotonic* in the
  threshold on rev8: 0.9018 at 0.70, 0.9269 at 0.85, 0.9303 at 0.90, and back **down to
  0.9064 at 0.95**.
  Abstention is supposed to buy accuracy monotonically. On a genuinely new subject here it
  does not, which means the threshold chosen on development subjects is not transferable
  and the `worst subject` column in `OPERATING_POINTS.md` is an optimistic floor.
- **It is concentrated.** Per trial, from the 08-03 read and **not** re-measured: t2 1.0000,
  t3 1.0000, t1 0.9580, **t4 0.8779**. The artifact still carries no per-trial breakdown —
  `per_rev` cuts by subject and rev8 is one subject — so these four numbers survive only as
  prose here. And t4 is independently the highest physics-label disagreement of any
  non-flagged trial in the corpus (0.1429, §3.5). Whether that is model failure or label
  noise is **unresolved**.

Read three times, every time honestly: twice on 2026-08-03 — once while rev13 was still
sealed (0.9287), once after freezing (0.9308) — and once on 2026-08-07 (0.9269), for the
schema, with the reason typed into the artifact's own `read_note` field. **No decision in
this repo was made using any of the three.** The 08-03 changes were driven by rev13 and the
development revs; nothing has been changed on the strength of the 08-07 read and nothing
may be. It must not be read a fourth time.

### 3.2b What the third read bought, and why it is not license for a fourth **[2026-08-07]**

The 08-03 artifact carried six keys. `per_rev`, `confusion` and `physics_gate` were added
2026-08-04 and could not be backfilled: the scored frame is never persisted, only its
summaries, so recomputing any of them means re-scoring rev8. Two of the three were partly
recoverable from prose already in this section; **`physics_gate` was not recoverable at all**,
and it is the one that mattered.

`anchors.py` retracted the physics ceiling on the claim that physics contradicts ~12% of
S2's high-confidence errors and **none at p >= 0.95**. That was measured on development
subjects. On rev8 it holds, out of sample:

| threshold | committed errors | physics objects to |
|---|---|---|
| 0.70 | 3,668 | 100 (2.7%) |
| 0.85 | 2,297 | **25 (1.1%)** |
| 0.90 | 1,597 | **0** |
| 0.95 | 901 | **0** |

The ceiling's entire addressable set on a genuinely unseen subject is 1.1% at the shipped
point and **exactly zero above it** — against 5.6% on the development subjects at 0.85. The
retraction was right, and it is now right for a reason that did not come from the subjects it
was argued on. That is the whole return on the third read, and it is confirmation of a
decision already taken, which is the only use a spent lockbox has left.

**One number in that table argues for a change, and it must be ignored.** The `floor` arm
comes out at **-275 errors** against threshold-only at matched coverage on rev8, and -1,290
on development. `PHYSICS_FLOOR` ships OFF. If that is worth revisiting, revisit it on the
development curve in `runs/regen/s2_ml/roweval_loro.json`, where the same sign is already
visible and the evidence is free. Turning it on because rev8 agreed would be using a sealed
subject to pick a policy, which is the exact failure the seal exists to prevent — and it
would retroactively make all three reads dishonest.

**One more transfer failure, visible only now.** The model abstains on 34.1% of the rows a
human marked `-1` and 24.4% elsewhere: a 1.4× separation, against 3.6× on development
(55.8% vs 15.3%). §5.2 reads that abstention/`-1` agreement as independent evidence the
abstentions are meaningful. On a new subject it mostly washes out.

### 3.3 Seven revs is a small basis for a threshold

The probability floor, the tier boundaries and the abstention threshold are all measured
over seven revs. Per-rev accuracy ranges roughly 0.91-0.99, so a single unusual subject
moves these numbers more than any tuning does — and §3.2 is exactly that happening.

### 3.5 One trial was quarantined for a label error, and the detector is new

`rev13/4` is excluded (`data/corpus.json`, which carries the reason with it — §1.6):
12,691 rows all annotated `stand`,
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

rev8 fell to 0.9269 (0.9308 when this section was written; §3.2 has why it moved, and the
finding below does not turn on it) and **no signal available beforehand would have warned
us**. Five
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

### 3.4 `find_pairs` guessed the raw↔annotated pairing — now it is declared **[resolved 2026-08-07]**

`find_pairs` used to accept a 5000 ms end-time mismatch while comparing rows by index —
§7's "never select by index" hazard with a wide tolerance. It was measured harmless on
this corpus: across all 43 annotated trials **every one matched exactly zero or one raw
candidate**, and each of the 18 that matched had a raw `Time` vector identical to the
annotated one — `max|Δt| = 0`, not merely within tolerance. (The correct count is **18**
pairs; the "19" quoted here and in DOMAIN_NOTES §6.2 was never reproducible.)

That measurement is what made it safe to *freeze*. The pairing now lives in
`data/corpus.json` and the search is gone — see §1.6. The 18 pairs the manifest declares
are byte-identical to the ones the scan produced, checked against the shipped
`runs/regen/s2_ml/raweval.json` before the old code was removed. What is retired with it:
the wide tolerance, the index comparison, and the 2.6 GB scan of all 91 raw files that
had to run before any of the three consumers could start.

### 3.7 Training and serving cut windows 8x apart, and the reason recorded for it was wrong **[corrected 2026-08-07, A/B still unrun]**

Training strides **2.0 s** over a 2.0 s window (non-overlapping, 5,984 windows); serving strides
**0.25 s**, so a row is averaged over up to 8 covering windows. Nothing matches the two.

**What is already measured, and it is more than it looks.** `roweval.py` refits per held-out rev
and scores that rev through `score_frame` at `inference_stride_s = 0.25` — the serve path, not a
training-grid proxy. So the shipped pair (84.66% at 0.9901) is measured *under* the mismatch and
already carries whatever it costs. There is no hidden optimism here.

**What is NOT measured: whether matching them would be better.** No model has ever been trained at
an overlapping stride at the 2 s window. That experiment is unrun, and the pipeline's preference
for non-overlap rests on argument alone.

**The argument it rested on was false.** `features.py` and `breakdown.py` both said overlapping
windows "leak between CV folds" / "flatter every metric". CV is `LeaveOneGroupOut(rev)` — folds are
**subject-disjoint**, so every near-duplicate of a window sits in the same fold as the original and
cannot reach a test set. Corrected in place at both sites; the honest cost of overlap is an inflated
effective sample count (split statistics and class weighting move), not a contaminated score. This
is §1.1's pattern again: a constant carrying a justification nobody re-derived.

**Nearest thing to the A/B, from the ledger.** `wider_window_4s` vs `window_4s_stride_2s` — same 23
features, same params, 4 s window, stride the only difference:

| spec | stride | train windows | macro-F1 | accuracy |
|---|---|---|---|---|
| `wider_window_4s` | 4 s (non-overlapping) | 2,477 | 0.8150 | 0.8882 |
| `window_4s_stride_2s` | 2 s (2x overlap) | 4,935 | **0.8188** | 0.8926 |

Doubling the training set by overlap moved LORO macro-F1 by **+0.0038**. If overlap flattered the
metric, that is where it would have shown. Directional only: it is 2x at a 4 s window, not 8x at
2 s, and both runs sit far below the champion's 0.9624.

**The larger train/serve asymmetry is purity, not stride.** Training takes only `PURITY_MIN = 1.0`
windows, so the model never sees a window straddling a transition; serving sees them constantly.
That is absorbed by the covering-window mean (a boundary row averages both sides toward 0.5 and
abstains) and it is inside the row-level coverage figure — but it is the axis to suspect first, not
the stride ratio.

To settle it: `ExperimentSpec(stride_s=0.25)` resolves independently of `window_s`
(`experiment.py`), and `run_experiment` already serves through the dense path. Compare coverage and
accuracy at threshold 0.85, worst subject, and **stand recall** (§3.0) — not macro-F1. Cost is
about 48,000 training windows through ExtraTrees400 over 7 folds.

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

---

## 6. Stale things removed from this repo's own work

§5 is about corrections the sibling repos still need; this section is about the ones only this
repo carried. Same rule applies — a removal is recorded, not silent, because the next reader
cannot tell a thing that was deliberately dropped from a thing nobody noticed.

### 6.1 The parquet decision outlived the parquet **[2026-08-05]**

S1 stopped persisting the cleaned frame — it measures the canonical grid and drops it, keeping
only `channel_trust.json` per file. **Nothing in this repo has written or read a parquet since,
and two things went on describing one:**

| removed | why |
|---|---|
| `pyarrow>=15.0` (`requirements.txt`) | The parquet engine, and its only reason to be installed. No module imports it, nothing passes `engine="pyarrow"` or an arrow dtype backend, and the worktree holds no `.parquet`, `.pkl` or `.npy` at all. |
| §9's "clean output is **parquet**" (`DOMAIN_NOTES.md`) | Corrected in place, not deleted. The *reasoning* — storage is a file-format problem, not a column-count one — is what licensed keeping the honest measured superset, and it still stands. Only the conclusion is dead. §4.1b's "anyone reading the parquet" went with it. |

**`.gitignore` still excludes `runs/**/*.parquet`, `*.pkl` and `*.npy`, and that is left alone
on purpose.** An ignore rule for a file type nothing produces costs nothing and fails safe; the
two that are load-bearing, `*.joblib` and `*.csv`, sit in the same list.

### 6.2 `runslayout.LOCKBOX_STEM` removed — it was the drift it existed to prevent **[2026-08-05]**

Never imported, while its two neighbours `LEDGER_FILENAME` and `PROPOSALS_FILENAME` are, by
`experiment.py`. Their shared comment says they are "named here rather than in each writer so
`runs/README.md` and the writers cannot drift" — and the third one had already lost that
argument: the literal `"roweval_lockbox"` is restated once in `roweval.py` and at four sites in
`breakdown.py`, so the canonical copy was the only one no code could reach.

**Deleting it is the smaller change; importing it at those five sites is the better one, and is
deliberately not done here** — that is a behaviour-neutral refactor of two live stage modules,
not a cleanup, and it should be its own commit with its own reason.

Nothing measured moved in either of the above. `verify_features` and `freshness --self-test`
pass, `run_pipeline.py --dry-run` still plans 14 steps, and no gate artifact, champion or figure
in `runs/breakdown.md` was touched.

### 6.3 What `raweval`'s removal orphaned, and one throwaway **[2026-08-07]**

Cutting a stage leaves its loaders behind, and they read as live corpus API rather than as
residue. §1.7 removed `raweval.py`; these were everything only it called, plus one file that was
never anything else:

| removed | why |
|---|---|
| `dataset.DEFAULT_LOCKBOX_REVS` | Sole reader was `raweval._drop_lockbox`. It was already only a re-derivation of `split == "lockbox"` from the manifest (§1.6), so anything wanting the sealed subject should ask `corpus.py` rather than a second name for the same query. |
| `dataset.find_trials` | Sole reader was `raweval`. Returned bare paths, which is the shape §1.6 exists to stop handing around — a path carries no subject, so it cannot be grouped, split or scored. `trials()` returns `Entry` rows instead. |
| `dataset.load_trial` | Zero readers, before and after. It was the path-shaped door into `load_entry`, and its whole body was the argument against itself: it took a `split` from the caller purely to refuse it when the manifest disagreed. |
| `corpus.entry_for` | Sole reader was `load_trial`. "A reader handed a file rather than the corpus may want it" is the same speculative generality §5.1 rejected for `decimate_window`; there are no such readers, and `git show` still produces it if one appears. |
| `_cost_probe.py` | A wall-clock probe of the S1 and label paths, its own first line calling itself a throwaway. Zero references in code, docs or notes, and the `_cost_probe.json` it wrote was never in the worktree. |

**The one thing genuinely lost is the refusal.** `entry_for` was the only place an *undeclared*
annotated file was fatal — every other route into the corpus starts from the manifest and so can
never meet one. That is not a hole a caller can fall into today; it becomes one the moment
something is written that takes a file path from a human. Put the refusal back with the caller
that needs it, not before.

Nothing measured moved. All 32 tests pass, `verify_features` and `freshness --self-test` pass,
`--dry-run` still plans 14 steps, and every module still imports.

One stale *claim* was corrected in the same pass, which matters more than the dead code because
it was being read as current: `RUNBOOK.md` §2 described the annotated corpus as a "glob in
`s2_ml/dataset.py`" with `rev8` and `("rev13", 4)` named in code. All three are what §1.6
replaced. §2 now names `data/corpus.json` as an input in its own right, which it never did.
