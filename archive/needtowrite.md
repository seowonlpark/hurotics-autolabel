# needtowrite.md

`README.md` and `PIPELINE_BREAKDOWN.md` were deleted on **2026-08-04**. This file is what
replaces them: the spec for rewriting them, the house style they have to be written in, and
— at the bottom — every claim from the originals that **cannot be regenerated** from code or
from anything under `runs/`.

> **Read this before using §5 or §6. The pipeline changed after this file was written.**
> On **2026-08-04** the S4 fusion stage was deleted, along with `agents/s4_fusion.py`,
> `stages/s2_ml/oof.py` and `stages/s3_physics/run.py`. It published a (coverage, accuracy)
> pair on the 2 s window grid that `label.py` never used — two accuracy proofs for one
> product, and the authoritative-looking one described a path no customer CSV ever took.
>
> `stages/s3_physics/rate_audit.py` went with them **and came back the same day** — an
> earlier version of this warning listed it as deleted, and that was wrong. It was cut for
> being `run.py`'s neighbour, not for failing its own test; it now drives itself and is a
> step in `run_pipeline.py`. Its docstring holds the argument.
>
> **§1–§4 have been updated to the current pipeline. The ARCHIVE inside §5 and §6 has not**
> — and the line between them is typographic, not by section number, because an earlier
> version of this warning drew it at "§5 and §6" and was wrong about its own file:
>
> - **Verbatim archive**: anything inside a ` ```markdown ` fence (all of §6) or behind a
>   `>` quote marker (most of §5). Transcribed from the deleted files and left as found.
> - **Live analysis**: the unquoted prose around it — §5.4's eight gaps most of all, which
>   is current-author work carrying 2026-08-04 corrections, not a transcription. It is
>   maintained, and two of its items were stale until this note was written.
>
> Anything in the archive naming S4, `oof.py`, `anchors.csv`, `fused.csv` or `fusion.md`,
> and every figure of the form *95.9% / 0.9805* or *65.1% / 0.9972*, describes a stage that
> no longer exists. **Do not carry it into the new README.** The live pair is row-level:
> **coverage 84.66% at 0.9901, worst subject `rev5` 0.9525**, from
> `runs/s2_ml/roweval_loro.json`, argued in `OPERATING_POINTS.md`.
>
> Two deliberate exceptions to "left as found". The `rev2_view` → `lpf_view` rename
> (2026-08-04, DOMAIN_NOTES §6.1) was applied **inside** the archive too, and so was the
> **`H-CARE` → `h-medi` correction (2026-08-05)** — the product name in the deleted README's
> header was simply wrong, and the corpus this pipeline reads was always h-medi
> (`DOMAIN_NOTES` §5.6). The archive exists to be rewritten from, and transcribing a name
> the code no longer uses — or never described — would carry the defect forward into the
> very file this one specifies. The `system_prompt.txt` files under `archive/runs/` are
> **not** covered by either exception: those record what an agent was actually told, and a
> corrected transcript would misdescribe the run that produced the verdicts beside it.
>
> Two more §5 claims have since gone stale and must not be transcribed forward. **§5.9's
> closing line — "one 2026-05 file is quarantined for a broken time base" — is no longer
> true:** that file was re-exported, `runs/s1_clean/quarantine.jsonl` is empty, and nothing
> in the corpus carries a degenerate time base (`DOMAIN_NOTES` §2.6). **§5.5's abstention
> split — "13 abstain — 12 unmapped variant, 1 with no trust record" — does not reproduce:**
> all 13 are `UnknownVariantError`, and the two failure sets were never disjoint
> (`DOMAIN_NOTES` §6.2). The measurements either side of them still stand; only these two
> counts moved.

Recovery, before anything else:

- `README.md` was tracked **but its working copy was 151 lines ahead of the last commit.**
  `git show HEAD:README.md` returns the *older* 273-line version, not what was deleted.
  Everything in the uncommitted delta is preserved below — distilled in §5, verbatim in §6.
- `PIPELINE_BREAKDOWN.md` was **never committed**. It is gone. Everything from it worth
  keeping is in §5.4, and the rest was superseded by `python -m stages.breakdown` — that is
  why it went.

---

## §1 — The house style

This repo has a distinctive voice and it is not decoration; it is the thing that made the
sagittal-axis bug findable and the stale-headline bug catchable. Rewrite in it or the
documents stop doing their job.

### The rules that carry weight

**1. Every number is a pair, or it carries its provenance.**
Coverage without accuracy is gameable to 1.000. Accuracy without coverage is meaningless.
The same applies more generally: a figure with no denominator, no population, and no
artifact behind it is decoration. Write `coverage 84.66% at accuracy 0.9901, worst
subject 0.9525 (runs/s2_ml/roweval_loro.json)`, never `~84% accurate`.

**2. Name the artifact beside the number, and say it goes stale.**
Any live figure in prose must name the file it was copied from and state that re-running
the stage invalidates it. This is not politeness — see §4, defect 1.

**3. Record what was rejected, with the measurement.**
The rejections are more useful than the choice. `champion_spec.json`'s "low importance is
not droppability" is worth more than the champion's own rationale, because it stops the
next person re-running an experiment that already failed. Every "we considered X" must
carry the number that killed it.

**4. Lead with the failure the thing prevents.**
Not "this module validates inputs" but "a stage that exits 0 without writing anything stops
the run rather than letting the next stage read a file from a previous run and quietly
report last week's numbers." The mechanism is obvious from the code; the failure is not.

**5. The "and it was" pattern.**
When a safeguard exists because something actually went wrong, say so, with the date. *"An
unexercised bridge cannot be wrong out loud, and it was: the sagittal axis for the majority
variant was wrong until the serve path was built."* A justification with a corpse attached
is believed; an abstract one is skipped.

**6. Provenance tags on anything factual.**
`DOMAIN_NOTES.md` uses **[measured]** (reproducible by re-running a stage), **[reported]** (a
human said so, unverified), **[decided]** (a design choice), **[open]** (known unknown).
Use them anywhere a reader might otherwise assume a number was measured when somebody just
said it.

**7. State the trade; do not render a verdict.**
"The shipped floor is 0.70. It is a deliberate point on this curve." Not "0.70 is optimal."
Where a choice is a judgement, say which judgement and what it cost. Where a rule is
mechanical, say what it thresholds on.

**8. Tables for anything with more than two dimensions.** Prose for anything with a *why*.
Do not put a rationale in a table cell and do not put a five-row comparison in prose.

**9. Second person, imperative, present tense for commands.** "Run both after touching
`transform.py`." Not "one should run" or "it is recommended that".

**10. Never let a heading promise more than the section delivers.**
A section called "Status" that lists five completes and one "running" is a status. A
section called "Status" that argues about methodology is a caveat wearing a status's hat.

### The rules about tone

- **No hedging on measured facts.** If it was measured, state it flatly.
- **No apologising for limitations.** State them. "78 of the 91 files under `data/raw` are
  servable today; the other 13 abstain, correctly."
- **Bold is for the load-bearing clause only** — usually one per paragraph, often the clause
  a skimmer would otherwise miss. Not for emphasis-in-general.
- **Em-dashes for the aside that carries the argument.** This repo uses them heavily and it
  works; keep it.
- Reports are written to be read as Markdown files where the typography is correct and
  wanted. The *console* is cp949 and cannot encode any of it — that is `stages/console.py`'s
  problem, not the writing's. Do not degrade the prose to ASCII.

---

## §2 — What `README.md` must contain

> ### ⚠ Before you write a line: re-arm the staleness check
>
> **Add `"README.md"` back to `PROSE_CLAIMS["S2 row-level"]` in `stages/breakdown.py`, in
> the same commit as the rewrite.**
>
> It was removed on **2026-08-04**. While the file was listed and missing, `breakdown`
> raised a gap flag every run — the designed behaviour, and correct — but once the deletion
> was deliberate that flag was a standing entry in §A that nobody was going to act on, and
> those teach a reader to skim the section that exists to be read. Removing it was the
> documented remedy (`breakdown._flag_stale_prose` names it), **not a decision that the
> README never needed checking.**
>
> The consequence of forgetting: a rewritten README quotes the headline pair, the champion
> or the threshold moves, and nothing anywhere says so. That is **§4 defect 1 rebuilt from
> scratch** — the defect this whole file exists downstream of, in the one document most
> likely to be read by someone who will not check the artifact.
>
> One line, and it is the difference between a check deferred and a check deleted.

Order matters: a first-time reader goes top to bottom and a returning one uses the headings
as an index. Sections, in order:

| # | section | must contain | must NOT contain |
|---|---|---|---|
| 1 | **One-paragraph what-is-this** | the pipeline's job in one sentence; the deterministic-code/agent-judgment split; the SDK link | any number |
| 2 | **Every command in this repo** | the small set of entry points as a table: command, what it is for, when to reach for it | modules the runner invokes — those go under Running it |
| 3 | **Read these first, in this order** | `DOMAIN_NOTES.md`, `caveats.md`, `OPERATING_POINTS.md`, this file, each with one line on what it is | duplication of their content |
| 4 | **Setup** | venv (PowerShell + bash), `requirements.txt`, `.env`, the Console spend-cap warning, the venv-in-every-terminal warning | anything version-specific that will rot |
| 5 | **Data layout** | the `data/` tree; that session date comes from the FOLDER; the quarantine-ledger rule; what is gitignored; the contamination rule | corpus counts — those live in the generated reports |
| 6 | **Running it** | the whole thing, then one subsection per stage with its command and its artifacts | any measured result |
| 7 | **What S1 actually does, and why** | the four load-bearing decisions with their evidence (§5.2 below) | S2/S3 behaviour |
| 8 | **Architecture** | **three** stages (S1 clean, S2 model + deliverable, S3 label audit) + the one agent role; the filesystem-only-interface rule; the four non-negotiables | implementation detail |
| 9 | **Status** | one row per stage, complete/running/not started, one clause each | argument |
| 10 | **What this pipeline is judged on** | the goal sentence; the current headline **with its artifact named and marked stale-on-rerun**; the lockbox warning pointing at `caveats.md` §3.2 | a second copy of the operating curve |

### Hard requirements

- **The command count in the heading must match the table.** "Every command in this repo:
  Five." was correct once and drifted. Either count them at write time and re-count on every
  change, or drop the number from the heading.
- **The `Label` / `guess` column semantics must be preserved exactly** — §5.3 below. They are
  the deliverable's contract and are not derivable from any artifact.
- **The judged-on section must name `runs/s2_ml/roweval_loro.md` and say the numbers go
  stale.** `stages/breakdown.py::PROSE_CLAIMS` mechanically checks that a listed document
  quotes the live **row-level** pair; if you write a number there, `python -m
  stages.breakdown` flags it the moment the champion or the threshold moves — **but only
  once this file is back on that list. See the callout at the top of this section.**
- **`python -m stages.breakdown` belongs under Running it** as the last step of every run.

---

## §3 — What the pipeline breakdown must contain

**Do not hand-write this file again.** That is the whole lesson of the last two days: a
hand-assembled breakdown was accurate for about six hours and then S4 was re-run.

It is generated: `python -m stages.breakdown --out runs` → `runs/breakdown.md` +
`runs/breakdown.json`, wired as the final step of `run_pipeline.py`. The spec now lives in
that module's docstring, and the section layout is §A flags, §0 corpus, §1–§4 the stages,
§5 the sweep, §B gaps.

If a *hand-written* companion is ever wanted again, it may only contain things no artifact
knows — and everything in that category as of 2026-08-04 is in §5.4 below. Concretely it
must **not** re-state coverage, accuracy, confusion, per-subject spread, reason tables,
quarantine counts, or file lists. Those are generated, and a second copy is a second thing
to go stale.

The generator's own rules, worth keeping if it is ever rewritten:

1. **It measures nothing.** Every number is copied from the artifact that owns it. A report
   that recomputed its figures would be a fifth opinion, and the first time it disagreed
   with a stage nobody could say which was right.
2. **A missing artifact is a stated gap, never a skipped section.** A stage that did not run
   must not read as a stage with nothing to say.
3. **Flags are mechanical and are not verdicts.** Each names what to look at; none conclude.

---

## §4 — Known defects in the deleted files — do not reproduce

**1. A stale headline that told the reader it might be stale.** `README.md` and `caveats.md`
sat at "coverage 95.9% at 0.9805" for a run that measured 65.1% / 0.9972. Both files said
"read the report, not this table." Both were still wrong to a reader who did not. The
disclaimer is not a substitute for the check — `stages/breakdown.py::_flag_stale_prose` is.

**2. A self-contradicting paragraph left behind by a policy change.** The deleted README
argued at length (correctly) that the abstained tier scoring 0.8972 is *not* evidence of
wasted calls, because the band is defined as the region where labels are unreliable — and
then, four lines later, kept the old sentence: *"The abstained set is supposed to score
badly: that gap between 0.9805 and 0.5403 is what makes the confidence signal informative."*
Both cannot be true. When a policy changes, grep the file for the old numbers and delete
the sentences that depended on them.

**3. A count in a heading that nothing re-counted.** "Five." See §2.

**4. Corpus counts in prose.** File counts, servable counts and row counts appeared in
README while also being regenerated every run. Point at the report instead.

---

## §5 — Unrecoverable content, preserved verbatim

Everything below is a **[measured]** or **[decided]** claim that no stage regenerates. Fold
it back into the rewrite. Dates are as they appeared in the originals.

### §5.1 — Setup and environment facts

- Requires **Python 3.10+**.
- PowerShell: `python -m venv .venv` · `.venv\Scripts\Activate.ps1` — *if blocked:*
  `Set-ExecutionPolicy -Scope Process RemoteSigned` · `pip install -r requirements.txt` ·
  `Copy-Item .env.example .env`
- bash: `python3 -m venv .venv && source .venv/bin/activate` · `pip install -r requirements.txt` · `cp .env.example .env`
- API key from the [Console](https://platform.claude.com). The SDK reads it from the process
  environment; `run_pipeline.py` calls `load_dotenv()` so `.env` is enough. **Set a monthly
  spend cap in the Console — that rail cannot be enforced from inside this repo.**
- "You must activate the venv in every new terminal. Forgetting is the single most common
  source of `ModuleNotFoundError` here."
- Data layout: `data/raw/<YYYYMMDD[_n]>/*.csv` unlabeled device logs, **session date comes
  from the FOLDER**; `data/labeled/` golden data, never mixed into raw; `data/clean/<session>/*.parquet`
  S1 output, canonical 100 Hz, 30 measured columns + `segment`, gyro normalized to deg/s.
- `data/` in its entirety, `.env` and `tracker/` are gitignored. `runs/` and `labeled_raw/`
  are **split rather than ignored wholesale**: the regenerable artifacts stay out, the
  measurements are versioned alongside the claims they support. **Nothing from HUROTICS
  leaves the machine via git.**
- **A `Label` column appearing under `data/raw/` is a contamination event, not a schema variant.**

### §5.2 — "What S1 actually does, and why" — the four decisions, verbatim

> **Resolves columns by name, never by position.** The `NN_` prefix is a per-file position,
> not an identifier. `loco` sits at index 47 — but in one header variant, index 47 is `Step`.
> Both are outdated columns (legacy algorithm output / firmware counter) and pruned by clean.
> See `DOMAIN_NOTES` §1.3.
>
> **Segments at gaps.** Gaps land anywhere. A file is a bag of continuous runs, and the
> **segment**, not the file, is the unit of analysis. Nothing is ever resampled across a gap
> — that would invent data that was never measured.
>
> **Normalizes rate to 100 Hz.** The corpus has two eras: an earlier ~100 Hz era and a later
> 500 Hz era. Downsampling uses `scipy.signal.decimate(..., ftype='fir')`, never `[::5]` —
> naive decimation folds everything above 50 Hz into the gait band as a full-amplitude fake
> signal. See `DOMAIN_NOTES` §2.5 for the measured proof. The odd rates (99.3789 / 99.688 /
> 99.961 Hz) are *timestamp quantization* (10 + 2⁻ᵏ ms), not different devices — they are
> grid-corrected, not discarded.
>
> **Keeps measured channels only.** The device *measures* IMU channels and load cells; it
> *computes* Cadence, Stride Length, GCP, admittance, PID state. Computed columns are the
> firmware's opinion, not observation. Dropping them collapses the schema variants into 1 and
> removes firmware-version signal from the feature set. The exception mechanism
> (`KEEP_EXCEPTIONS` in `stages/s1_clean/config.py`) is **currently empty** — canonical ==
> measured, no caveat. The one former exception, `Hip_Deg_L/R`, was cut once measured: 0.991
> correlated with the `Deg_Y` already kept, its residual carrying nothing but the firmware's
> zeroing convention, and dead on part of the corpus. See `DOMAIN_NOTES` §9.
>
> Note that "measured" means *not app-layer-computed*. The `Deg` channels are the IMU's own
> on-sensor fusion output, not a transducer reading — kept, but see `DOMAIN_NOTES` §4.6
> before treating them as ground truth.

### §5.3 — The deliverable's output contract — `Label` vs `guess`

Each labelled CSV **opens with the annotated corpus's own six columns** — `Time`, the four
`*_LPF` features, `Label` — then appends the verdict columns:

| column | meaning |
|---|---|
| `Label` | the committed call: `0` stand, `10` walk, `-1` scored but under the threshold, `255` no window covered the row |
| `guess` | the same call *before* the threshold: `0` or `10` on every row that was scored at all, `255` where nothing was. Never `-1` |
| `confidence` | `max(p, 1-p)` from the ensemble; empty on a `255` row, which was never scored |
| `ambiguous` | `True` for every `-1` and `255` row — the plain boolean, if you don't want to decode `Label` |
| `reason` | which doubt: `near_transition`, `weight_shift_or_step`, `model_split`, `posture_shift`, `low_excursion_gait`, `out_of_distribution`, `uncovered`. Empty on a confident row. |

> `Label` answers "may I use this row", `guess` answers "what did it think" — and `-1`
> destroys the second to give the first. **`Label != guess` selects exactly the rows that
> were scored and abstained**, which is the review queue with the model's own opinion
> attached rather than stripped out.
>
> The two abstention codes stay apart deliberately: `-1` is the machine analogue of an
> annotator's own "I looked and cannot call it", `255` is `config.LABEL_UNKNOWN_MACHINE` —
> nothing was measured well enough to judge. A reader that lumps them cannot tell an unsure
> model from an absent signal. `--no-verdict` drops everything after `Label`, leaving a file
> indistinguishable in shape from an annotated one, for a reader that needs exactly that;
> `--full` keeps the entire frame. `label` and `label_all` take the same two flags and
> default to the same shape.

> **The call is S2's alone.** The per-row state comes from the ExtraTrees ensemble; the
> physics diagnostics are used only to *name* the doubt. S4's fusion tier — the S2-vs-S3
> disagreement flag that `stages/s4_fusion/fuse.py` argues is the point of that stage — does
> **not** run here: S4 joins corpus-level OOF and anchor tables and has no serve route. So
> `ambiguous` means "the ensemble was unsure", not "the two opinions disagreed".

### §5.4 — The eight gaps `PIPELINE_BREAKDOWN.md` closed with

This is the only genuinely lost content — analysis, not measurement, and no artifact
regenerates it. Ranked by how much closing each would change a decision.

1. **No per-file accuracy anywhere.** Every accuracy number in this repo lives on 43
   annotated trials from 8 subjects. The 78-file corpus sweep has coverage only. There is no
   way today to say "this recording was labelled well" for any file in `data/raw`.
2. **No cost or runtime figures.** `runs/*/costs.json` exists per agent run, but nothing
   aggregates spend, wall-clock per stage, or the cost of a full `run_pipeline.py`. You
   cannot currently answer "what does one corpus sweep cost."
3. **Neither agent's output is summarized.** No agent run appears in the current `runs/` —
   the reviews in `runs/2026-08-03_run*` predate the S4 deletion, and one of the two agents
   they came from (`s4_review`) no longer exists. `label_suspect` is the verdict worth
   having and nothing has produced it recently.
4. **Freshness is stamped in two places in code and one on disk** [measured 2026-08-04].
   `freshness.py` exposes `stamp_inputs` / `check_all`. `stages/s2_ml/label_all.py` calls it
   (this item used to say it was the *only* caller) and so does `stages/s2_ml/train.py`,
   which stamps `champion_spec.json` beside the model it fits. But **`runs/s2_ml/` carries
   no `_inputs.json`**: the stamping line was added after the last training run, so
   `labeled_raw/_inputs.json` is still the one stamp that exists and **zero `runs/*`
   directories carry one**. One `python -m stages.s2_ml.train` closes that half.
   `stages/breakdown.py` checks `labeled_raw/`
   unconditionally and `runs/*` only where stamped, which today means it checks nothing
   there. That is not a bug in `breakdown.py` — a stage that stamps nothing has nothing to
   go stale against — but it does mean the guarantee covers the labelled corpus and not the
   pipeline that produced it. The `runs/s4_fusion/_inputs.json` this item used to cite as
   the working example went with S4. The prose-level version of the same failure (§4 defect
   1) is checked by `stages/breakdown.py`; nothing checks `DOMAIN_NOTES.md`.
5. ~~**Nothing quantifies the transition class.**~~ **CLOSED 2026-08-04** by
   `stages/s2_ml/transitions.py` → `runs/s2_ml/transitions_loro.md`, which answers all three
   questions this item asked. **267 timeable transitions** from 308 annotated boundaries (41
   rejected as short-flank, counted not dropped); median annotator `-1` width **0.14 s**;
   median timing offset **−0.18 s**, with **170 of 266 within ±1 s** and 223 within ±2 s.
   The flag's recall is 218 of 267 (82%) and its precision from the other side is weak —
   167 of 384 flagged regions (43%) contain no annotated transition within ±2 s.
   **The "27 timeable transitions, of which 14 are on-time" figure this item quotes does
   NOT reconcile with the 267, and that remains open** — `transitions.py` says so in its own
   docstring rather than explaining it away. Neither figure is derivable from the other and
   the older pair predates the deleted S4 window grid. What is closed is the *hole*: there
   is now a measurement under a stated definition. **Settle the discrepancy before either
   pair is quoted anywhere.**
6. **No error breakdown by session date or device.** Everything is sliced by rev (subject).
   The 12-file low-coverage cluster (one variant, four consecutive sessions) says session is
   an independent axis, and it has never been cut that way.
7. **The 43-vs-91 gap is unexplained.** 43 annotated trials against 91 raw files, 18 pairs.
   Which raw files correspond to which annotated trials — and why 25 annotated trials have
   no surviving raw — is not written down anywhere.
8. **No confusion matrix at the row level.** The direction-of-error figure is window-level
   OOF; the row-level evaluation reports coverage and selective accuracy but never a
   stand/walk confusion, so the "86% of errors are stand→walk" figure is inferred across
   units.

### §5.5 — Verification results, with their dates

Re-run to refresh; recorded because the *commands are slow* and the numbers are cited
elsewhere.

- `verify_transform` — the raw→`lpf_view` bridge on every paired recording: **18/18 pairs, ≤7.3e-13**.
- `verify_serve` — the same recording labelled both ways, row for row: **536,590 rows, 0
  disagreeing verdicts, max |Δconfidence| = 0**; corpus sweep **78 of 91 files servable (86%)**,
  13 abstain — 12 unmapped variant, 1 with no trust record.
- Per-variant axis reproduction, max abs error (`caveats.md` §5 holds the full table):
  `fb5ea2c2` 7 pairs (rev13) `Deg_Y`→`Gyro_Z` **≤2.4e-13**; `0fda484e` 10 pairs (rev7, rev8)
  **≤7.3e-13**; `4bfd6ab2` 1 pair (rev4) **6.8e-13**.

### §5.6 — The S4 coverage-policy narrative **[decided 2026-08-03; SUPERSEDED 2026-08-04 — do not reproduce]**

> **This entire section describes deleted code.** The stage, both flags and the 65%
> figure are gone. The band policy was ported to the row path, measured, and
> **rejected**: abstaining on the band reached 62.04% at 0.9968 where simply raising the
> threshold to 0.95 reached 67.32% at 0.9983 — more coverage, more accuracy, a third of
> the errors, same worst subject. `label.BAND_ABSTAINS` is `False`; the argument is in
> `OPERATING_POINTS.md`. Kept only as the record of an argument that was once
> load-bearing.

Preserved because it explains a 31-point coverage drop that otherwise reads as a regression,
and because `fuse.py`'s comments hold the curves but not the argument.

> **Coverage is 65%, not 96%, and that is a decision rather than a regression.** Two flags in
> `stages/s4_fusion/fuse.py` set it. `BAND_ABSTAINS` abstains on the whole 2.62–20.34°
> interleg band — a third of all windows — because that is the range where *the annotations
> themselves* are not separable, and a call there claims certainty the ground truth cannot
> support. `MEDIUM_ABSTAINS` then drops the `medium` tier from coverage as well, leaving the
> output effectively binary: `high` or abstain. Both carry their measured exchange-rate curves
> in `fuse.py`; the band point is deliberately the *worst* error-per-coverage-point trade on
> its curve, taken for the labelling argument rather than the error count.
>
> Set both to `False` and the same champion reports **95.9% at 0.9805** — that is the same
> model under the older policy, not an older model.
>
> One consequence worth stating plainly: the abstained tier now scores 0.8972, not the 0.5403
> it scored when abstention meant "the model is unsure". That is **not** evidence of 2,000
> wasted calls. The band is defined as the region where labels are unreliable, so agreement
> with them there measures the annotator, not the pipeline — the "abstentions should score
> badly" heuristic simply does not apply inside it.

*(Do not re-import the sentence that followed this in the original — see §4, defect 2.)*

### §5.7 — Architecture non-negotiables, verbatim

> Four stages. The **filesystem is the only interface** between them — no agent-to-agent
> messaging, no message queues. `run_pipeline.py` is a dumb sequencer: it shells out to each
> stage in order and checks that the gate artifact appeared. It never imports a stage to
> "help" it — a sequencer that could reshape a stage's output would be a fifth stage nobody
> documented.

| stage | deterministic core | agent role |
|---|---|---|
| **S1 clean** | schema census, rate normalization, gap segmentation, channel trust | exception queue only |
| **S2 ml** | windowing + features, train + locoeval, row-level labelling with abstention, OOF artifact | — (deterministic) |
| **S3 physics** | swap-rule anchors, rate-invariance audit | — (deterministic, label-free) |
| **S4 fusion** | join S2 + S3 → call, confidence, reason | judge the abstention queue |

*(This table is part of the verbatim transcription above — it carries no `>` markers only
because a markdown table inside a blockquote renders badly. **The S4 row describes a stage
deleted on 2026-08-04**, so "four stages" is three, and the agent column is wrong in both
directions: S4's reviewer is gone, while S2 and S3 have since gained their own. The paid
steps are now `s1_exception`, `s2_experiment` and `s3_label_review` — measured from
`run_pipeline.py --dry-run --with-agents`, which is the only thing worth trusting here.
Current status is §5.9, not this table.)*

> 1. **Code does the work; agents judge the work.** Agents never touch data values and never
>    crunch numbers themselves.
> 2. **No silent mutation.** Failures are logged to the quarantine ledger, decisions get
>    written rationale, low confidence escalates to `needs_human`.
> 3. **"Best" is defined by locoeval, not by an agent's opinion.** The champion is settled
>    here and changes only against a re-run metric; the agent-driven challenger loop that used
>    to gate promotions lives in the sibling experimental repo, and its ledger
>    (`runs/s2_ml/experiments.jsonl`, `proposals.jsonl`) is kept as a record of what was tried.
> 4. **Every stage closes with a `DOMAIN_NOTES.md` update.** Discoveries become permanent, not
>    conversational.
>
> Agents are never authorized to delete raw data, edit `DOMAIN_NOTES.md` without human review,
> or retrain the champion.

### §5.8 — Agent behaviour and run bookkeeping

> Both reviews are steps in the runner like any other, so there is one way to run anything in
> this repo. They are the only steps that run in-process rather than as a subprocess — an
> agent call is a network wait, and isolating it buys nothing. Each writes into its own
> `runs/<date>_runN/`, never overwriting a previous review.
>
> The S4 review agent (`agents/s4_fusion.py`) reads `runs/s4_fusion/fused.csv` and judges the
> windows the pipeline could not confidently *and* correctly call — abstentions, and confident
> errors against the human label, quota'd so neither crowds the other out. It assigns a cause
> from a closed vocabulary (`transition` / `label_suspect` / `slow_gait` / `weight_shift` /
> `data_quality` / `ambiguous`) plus whether a person is needed, and `collapse()` derives the
> disposition deterministically.
>
> `label_suspect` is the verdict worth having: if ground truth is wrong, the pipeline's
> "error" is not one. Those are printed as mislabel candidates and always routed to a human —
> an agent may nominate a label change, never enact one.
>
> The S1 exception agent (`agents/s1_exception.py`) reads the clean stage's exception queue
> (`quarantine.jsonl` + `observations.jsonl`), judges each item — `known_expected` / `novel` /
> `needs_human`, grounded in `DOMAIN_NOTES` — and the deterministic wrapper writes
> `exceptions_review.jsonl`. It is read-only: the agent judges, code does the work.
>
> Every run gets `runs/YYYY-MM-DD_runN/` containing `run_meta.json` (commit SHA — much of
> `runs/` is regenerated in place or unversioned, so an artifact that cannot name the commit
> that produced it is unattributable), `costs.json` (per-agent spend
> from the SDK's ResultMessage) and `system_prompt.txt` (exactly what the agent was told). A
> `run_log.jsonl` appears alongside them via a PostToolUse hook — one line per tool call, so a
> run whose agent used no tools writes none.

### §5.9 — Status table as of 2026-08-04

Hand-maintained; nothing regenerates it.

| stage | state |
|---|---|
| **S1** clean | complete — schema/rate/gaps, gyro unit+axis trust, yaw-drift trust, degenerate-time-base rejection, quarantine ledger; gate passes (every raw file accounted) |
| **S1** exception agent | complete — triages the exception queue into known_expected / novel / needs_human with grounded rationale; verified on the real corpus and signed off (gate closed) |
| **S2** ml | complete — windowing/features, leave-one-rev-out CV, per-row labelling with confidence + ambiguity reasons |
| **S2** experiment agent | complete — the champion/challenger loop, **ported back in**. This row used to say it "was removed: experimentation lives in the sibling repo, this one holds one settled champion", which `stages/s2_ml/experiment.py` and the live `s2_experiment` step contradict. An experimenter proposes an `ExperimentSpec`, a critic reviews it, and **neither promotes anything** — `decide()` does, on leave-one-rev-out macro-F1 beating the incumbent by `PROMOTION_MARGIN`. Every outcome, rejections included, lands in the ledger. It **does** carry the sibling's error taxonomy — `run_experiment(taxonomy=True)` buckets dense per-row predictions and `decide()` uses the `steady_confusion` share as a tiebreaker; this row previously said it did not. That is not the second accuracy claim S4 was deleted for: it describes the *shape* of a model's errors to rank two candidates, and publishes no coverage/accuracy pair. Genuinely narrower than the sibling in one respect: no channel selection (`champion_spec.json` fixes `channels: []`) |
| **S3** physics | complete — swap-rule anchors on the shared window grid; rate-invariance verdict recorded for all four anchors (gate closed) |
| **S3** label audit | complete — two trial-level detectors, both model-free: outright physics/annotation contradiction, and divergence from the corpus convention inside the ambiguity band |
| **S3** label review agent | complete — the consumer for `label_audit.py`'s nominations, which stopped at "a nomination is not a verdict" and went unadjudicated. Assigns each flagged trial a cause from a closed vocabulary; never recomputes, never opens a raw signal |
| **S3** rate audit | complete — deleted with S4 on 2026-08-04 for being `run.py`'s neighbour, restored the same day with its own CLI; `gyro_energy` is the negative control and is expected to FAIL |
| **S3** plausibility | complete — file-level serve bounds, `--calibrate` sites them and `--control` injects three synthetic channel faults to show they can fire at all |
| **S4** fusion | **deleted 2026-08-04** — it published a (coverage, accuracy) pair on the 2 s window grid that `label.py` never used. Two accuracy proofs for one product, and the authoritative-looking one described a path no customer CSV ever took. `agents/s4_fusion.py`, `stages/s2_ml/oof.py` and `stages/s3_physics/run.py` went with it |
| **verifiers** | complete — `verify_features` unconditional and first; `verify_transform` / `verify_serve` are `run_pipeline.py --verify` steps. Both were deleted from the working tree on 2026-08-04 and **restored the same day**: four documents were still quoting their numbers as re-runnable (`caveats.md` §3.1) |
| **breakdown** | complete (2026-08-04) — `stages/breakdown.py`, final step of every run, reads all four stages, states its own gaps, raises mechanical flags |
| hardening + handoff | not started |

The closing line here used to read *"one 2026-05 file is quarantined for a broken time
base"*. **That is no longer true** — the file was re-exported, `runs/s1_clean/quarantine.jsonl`
is empty, and nothing in the corpus carries a degenerate time base (`DOMAIN_NOTES` §2.6).

### §5.10 — The goal sentence, and the lockbox warning

> **Label stand/walk on any recording at ≥95% accuracy over the windows it claims, and
> abstain rather than guess on the rest.** Abstention is a feature: an ambiguous window gets
> a call, a confidence tier, the reason it is uncertain, and the alternative it was weighing.

> **These are development-subject numbers.** The sealed lockbox subject (`rev8`) scores
> **0.9308 on the rows it commits to** — the target is missed there. Read `caveats.md` §3.2
> before quoting anything above, along with what is thin, what is unverified, and what was
> deliberately left out.

`rev8` is **spent** — read twice on 2026-08-03, both logged in `caveats.md` §3.2. It must not
be read a third time.

---

## §6 — Verbatim appendix: the README prose §5 did not distil

These blocks were in the working copy and **not** in the last commit, so they exist
nowhere else. §5 above holds the measured claims; this holds the prose around them.
Prune freely when rewriting — it is here so the choice is yours, not the filesystem's.

### §6.1 — Header, command index, and reading order (lines 1–46)

````markdown
# h-medi-agents

A staged pipeline for turning raw h-medi IMU device logs into a locomotion classification system —
cleaning, ML experimentation, physics-based analysis, and reporting — where **deterministic code
does the work and Claude agents handle judgment at defined points**, with every decision logged and
reproducible.

Built on the [Claude Agent SDK](https://docs.claude.com/en/docs/agent-sdk/overview) (Python).

---

## Every command in this repo

Five. Everything else is a module the runner invokes — `python run_pipeline.py --dry-run` is the
index of those, and they are documented under [Running it](#running-it) for when you are
iterating on one stage by hand.

| command | what it is for | when |
|---|---|---|
| `python run_pipeline.py` | **the pipeline.** `data/raw` → `runs/s4_fusion/fusion.md`, every stage gated on its output artifact. `--with-agents` adds the two paid reviews, `--from KEY` resumes | after new data lands, or any change to a stage |
| `python -m stages.s2_ml.label FILE` | **the deliverable.** One recording in, one row out per row in, carrying `Label` / `confidence` / `ambiguous` / `reason`. Takes a raw device log or an `lpf_view` file | labelling one file |
| `python -m stages.s2_ml.label_all` | the deliverable over the whole of `data/raw` in one sweep, same code path per file | labelling the corpus |
| `python -m stages.s1_clean.validate FILE` | **is this recording usable at all?** Label-free, model-free pre-flight: rate, gaps, segment length, channel presence. Answers before you spend anything | a new recording arrives and you want to know if it can be scored |
| `python -m stages.s3_physics.inspect_window REV TRIAL --t SECONDS` | **adjudicate one suspect window by eye.** Prints the raw interleg trace and the ±1° crossings the swap rule counted, around one timestamp | a review flagged a window as a suspected mislabel and a person has to settle it |

`stages.s2_ml.roweval` is the sixth, and it is a documentation tool rather than a pipeline
step: it re-measures the coverage/accuracy tradeoff at every abstention threshold and
regenerates the tables in `OPERATING_POINTS.md`. Run it when the champion changes; nothing
consumes its output but a reader.

---

## Read these first, in this order

| file | what it is |
|---|---|
| **`DOMAIN_NOTES.md`** | Everything the corpus taught us the hard way. Injected into every agent's prompt. **Read before touching any data.** |
| **`caveats.md`** | What this pipeline is shaky about: thin constants, accepted imperfections, unverified paths, deliberate omissions. Read before trusting a number. |
| **`OPERATING_POINTS.md`** | The abstention threshold: what each preset costs and buys, and why the default is 0.85. |
| this file | How to run it. |

`DOMAIN_NOTES.md` is not background reading — it is the reason this pipeline is shaped the way it
is. Every entry carries a provenance tag: **[measured]** (reproducible by rerunning S1),
**[reported]** (a human said so, unverified), **[decided]** (a design choice), **[open]** (known
unknown). If you only read one thing, read §1.3 — the column prefix is a lie, and positional
indexing silently corrupts most of the corpus without ever raising an error.
````

### §6.2 — Running it — the whole thing, S1 census, S1 clean, S2→S3→S4, breakdown (94–172)

````markdown
## Running it

### The whole thing

```powershell
python run_pipeline.py                  # deterministic spine: data\raw -> runs\s4_fusion\fusion.md
python run_pipeline.py --with-agents    # + the two agent steps (spends API credit)
python run_pipeline.py --dry-run        # what would run, in order, without running it
```

Each step is its own subprocess and must produce its **gate artifact** before the next
starts — a stage that exits 0 without writing anything stops the run rather than letting
the next stage read a file from a previous run and quietly report last week's numbers. A
failure prints `--from <key>` to resume at that step.

The individual commands below are what the runner invokes; run them by hand when
iterating on one stage. If you do, note that `runs/` is overwritten in place, so
re-running one stage alone leaves the downstream reports describing inputs that no longer
exist. `run_pipeline.py` checks for exactly that at the end (`freshness.py`: S4 stamps the
sha256 of every artifact it consumed into `runs/s4_fusion/_inputs.json`) and says so — it
reports, it does not delete.

### S1 — census (measure the corpus, judge nothing)

```powershell
python -m stages.s1_clean.run --out runs\s1_census
```

Produces `census.md` (human-readable), `variants.json` (every distinct header + the stable prefix
per family), `manifest.jsonl` (one row per file: session, variant, measured rate, jitter, gaps,
label codes).

### S1 — clean (resample onto the canonical grid)

```powershell
python -m stages.s1_clean.clean --out runs\s1_clean
```

Produces `data/clean/**.parquet` (gyro normalized to deg/s) with a per-file
`channel_trust.json` sidecar, plus per-run `segments.jsonl`, `observations.jsonl`,
`quarantine.jsonl`, and `clean_report.md`.

### S2 → S3 → S4 (the labeling chain)

```powershell
python -m stages.s2_ml.train   --out runs\s2_ml       # train + leave-one-rev-out locoeval
python -m stages.s2_ml.oof     --out runs\s2_ml       # out-of-fold predictions + probabilities
python -m stages.s3_physics.run --out runs\s3_physics # swap-rule anchors + rate audit
python -m stages.s4_fusion.run  --out runs\s4_fusion  # call + confidence + reason
```

Run in that order — S4 inner-joins S2's `oof_champion.csv` to S3's `anchors.csv` on
`(rev, trial, segment, t_start_ms)` and refuses to proceed if the two stages windowed
differently. `runs/s4_fusion/fusion.md` is the number the pipeline is judged on.

### The breakdown (every stage on one page)

```powershell
python -m stages.breakdown --out runs                 # runs/breakdown.md + breakdown.json
```

The last step of every run, and the only one that reads all four stages. It **measures
nothing** — every number is copied from the artifact that owns it and the artifact is
named beside it, so it cannot disagree with a stage. Read `breakdown.md` to see the whole
pipeline at once; read the stage's own report to change a number.

Two things it does that no stage report can. It states its own gaps: an artifact that was
not there is printed in §B with the command that produces it, because **a stage that did
not run must not read as a stage with nothing to say**. And it raises §A flags from
mechanical rules — the lockbox below target, a subject spread this wide, corpus files
clustering on one variant, and *whether the checked-in prose still quotes the live
headline*. That last one exists because `README.md` and `caveats.md` both sat at "coverage
95.9% at 0.9805" for a run that measured 65.1% / 0.9972; both told the reader to prefer
the report, and both were still wrong to a reader who did not.

Flags are not verdicts. Each names what to look at and none of them conclude anything —
`breakdown.json` carries the same list for a future gate to act on.
````

### §6.3 — Labelling a file, and labelling the whole corpus (174–211)

````markdown
### Labelling a file (the deliverable)

```powershell
python -m stages.s2_ml.label data\raw\20251024\00321_63_2025_10_24_15_32_0.csv   # raw device log
python -m stages.s2_ml.label some_annotated_trial.csv --preset high_precision    # lpf_view
```

Takes **either** shape of file, dispatched on the header's family marker resolved by name: a
raw device log, or an `lpf_view` file. A raw log is bridged to the four `lpf_view` channels
first (`transform.raw_csv_to_features`), and the run prints which hardware revision was
resolved, which two channels were actually read, and which `channel_trust.json` was
consulted. Output is the caller's own rows with `state` / `label` / `confidence` /
`ambiguous` / `reason` / `reason_detail` / `alternative` appended — rows in, rows out.

It **abstains rather than guesses**, with a stated reason and no output file: an unmapped
hardware revision, a file whose measured permutation contradicts the axis about to be read,
a gyro that is not natively deg/s, a final timestamp that cannot carry the filter, or a
missing trust record. 78 of the 91 files under `data/raw` are servable today; the other 13
abstain, correctly.

### Labelling the whole corpus

```powershell
python -m stages.s2_ml.label_all                      # every raw file -> labeled_raw\
python -m stages.s2_ml.label_all --summary-only       # sweep only, writes no per-file CSVs
```

`label.py` in a loop — every file goes through the same `label_csv` entry point, so a batch
run cannot drift from what an operator gets labelling one recording by hand. One file's
abstention never ends the sweep: each is caught, recorded with its reason in
`abstentions.jsonl`, and the run continues. Every raw file is accounted for exactly once,
asserted rather than hoped, the same partition gate S1 enforces.

Writes `labeled_raw\<session>\<name>_labelled.csv` (mirroring the `data/raw` session
directories) plus `label_summary.csv`, `abstentions.jsonl` and `label_report.md`. The
per-file CSVs are a few GB and gitignored; the summary and report are not. An *unexpected*
exception — as opposed to a stated refusal — exits non-zero, so a batch run cannot pass
while quietly skipping files.
````

### §6.4 — Verifying the raw path (243–256)

````markdown
### Verifying the raw path

```powershell
python -m stages.s2_ml.verify_transform   # the four columns, rebuilt from raw to ~1e-13
python -m stages.s2_ml.verify_serve       # the same recording labelled BOTH ways, row for row
```

`verify_transform` checks the bridge's math on every paired recording (18/18, ≤7.3e-13).
`verify_serve` checks what a caller actually gets: it labels the raw log and its `lpf_view` export
and asserts identical state, confidence and reason on every row (536,590 rows, 0
disagreements), then sweeps `data/raw` reporting what is servable and what abstains, by
reason. Run both after touching `transform.py` — an unexercised bridge cannot be wrong out
loud, and it was: the sagittal axis for the majority variant was wrong until the serve path
was built (`caveats.md` §5).
````

