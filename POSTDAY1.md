# HANDOFF — session state for resuming with a fresh Claude

**Written:** 2026-07-16, end of day 1 · **Revised:** same day, merging a parallel analysis session
**For:** the next Claude context working on `h-care-agents` with Lu (intern, HUROTICS)
**Read alongside:** `DOMAIN_NOTES.md` (findings), `PLAN.md` (architecture + gates), `README.md` (how to run)

If you are Claude reading this at the start of a new conversation: this is your compaction. The
three docs above carry the durable content. This file carries what *isn't* in them — the reasoning,
the open threads, the corrections, and how to work with this person.

**Two tracks converged into this file.** Track A = the pipeline build (S1 census/clean, agents).
Track B = a parallel physics/rule-discovery session on labeled rev2/rev8 trials. Track B corrected
Track A on several points. §2 records the corrections; **apply them before writing new code.**

---

## 1. What this project is

Lu is an intern at HUROTICS (wearable exoskeleton startup, device = H-CARE) with a **hard end
date**. The mission: build an IMU-based locomotion classifier to improve on an existing rule-based
algorithm that struggles with atypical gait and — critically — has **no confidence signal** for
unreliable windows, a safety issue in a wearable control context.

The PL directed a restructure: split the problem into **stages, each its own AI agent**, improve
each stage, and — explicitly — *if a stage hits a ceiling where AI can't proceed and human
intervention is needed (e.g. a golden set), that discovery is itself valuable*. That clause is
load-bearing: **`needs_human` escalation is a deliverable, not a failure.** Everything must be
handoff-ready.

---

## 2. CORRECTIONS — apply these first

### 2.1 `angvel` is RELIABLE. DOMAIN_NOTES §4.1 is wrong. **[measured, Track B]**

`angvel_LPF` **is** `d(angle)/dt` — r = 0.999, slope 0.98. It is not noise; it is the derivative.

The old "rings at ±40–80 deg/s during static postures" claim is `[reported]` from an earlier
session and was never verified. Track A "confirmed" it by measuring |angvel| p99 ≈ 105 deg/s on
`annotated_loco_rev2_trial_1` — **but that was computed over the whole file, 93.7% of which is
walking.** Track A measured walking and called it rest. Retract it.

`DOMAIN_NOTES` §4.1 is injected into every agent prompt. **Fix it before running any agent**, or
agents will discard a good channel and re-derive it from angle for no reason.

### 2.2 Gyro units are inconsistent WITHIN a file **[measured, Track B — reproduced on 95 then 42 files]**

- **`B_Gyro` is rad/s. `L/R_Gyro` is deg/s.** Same naming convention, same file.
- **Y↔Z swap on every segment:** `d(Deg_Y)/dt` tracks `Gyro_Z`, not `Gyro_Y`.
- `Deg_Y` needs **no sign normalization** — raw L vs R is already anti-phase in 84% of files.

`stages/s1_clean/config.py: KEEP_MEASURED` keeps all 27 IMU channels with **zero unit awareness**.
Any feature mixing trunk and thigh gyro silently mixes units by 57.3×. **This is a live bug in
shipped code.** Fix: unit normalization in the clean layer (it is a measurement property, not a
feature choice), plus an axis-convention note in DOMAIN_NOTES.

### 2.3 Session ≠ subject. Subject is mostly UNKNOWN. **[measured, Track B]**

`20260114` contains both `sub1` and `sub2`, both under `id=69`. **Filename field 2 is not a
subject** — it tracks the date block (device / firmware / protocol). Only **7 of 95** files mark
the wearer at all.

Track A's "group by session for CV" is therefore **insufficient**. Subject leakage is the confound
that inflates scores, and subject is unrecoverable for 88 of 95 files. This is a genuine
`needs_human` finding: it is exactly the "AI can't proceed without human input" case the PL asked
to surface. Raise it with Lu — someone may be able to reconstruct wearer identity from session
records.

### 2.4 `loco` stays severed — better reason than we had **[measured, Track B]**

Its "standing" class contains a decile as periodic at the gait frequency as median walking. A label
that fails inspection cannot validate anything. (Track A excluded it for being "outdated"; the real
reason is that it is *wrong*, demonstrably.)

### 2.5 rev2 is 494 Hz, not 500 — and this is a *bug class*, not a trivia item **[measured, both tracks]**

Track A's census independently measured `hz_from_median_dt = 500.0` and `hz_from_span = 493.97`,
and flagged the disagreement. Track B found what the disagreement costs: **jitter accumulates to
4.2 s of drift by t=320 s.** `int(t*fs)` points 4.2 seconds past the event.

**Rule: always select by time, never by index.** Track B flags that the **wide path needs an
index-vs-time audit** — the same bug may exist there. Track A's `resample.py` appears clean (it
interpolates against `t` throughout), but this has not been audited. Do it.

### 2.6 The raw→rev2 mapping is now largely resolved **[measured, Track B]**

`*_ang_LPF ≈ LPF(*_Deg_Y)` — supported by Track B's finding that `Deg_Y` is the sagittal channel
that is already anti-phase L vs R, plus Track A's observation that `R_ang_LPF` opens at 85.43 and
raw `R_Deg_Y` at 85.69. And `angvel_LPF = d(ang_LPF)/dt` at r=0.999. LPF parameters remain unknown.
Upgrade DOMAIN_NOTES §6.2 from "hypothesis, do not encode" to "strongly supported, parameters open".

---

## 3. Track B's substantive result: the swap rule

**Walking is the legs alternating.** Not how far they swing — *whether they swap*.

> Count how many times `L_ang − R_ang` commits past `+1°` and then past `−1°` within a 2 s window.
> **0 swaps → STANDING. ≥2 swaps → WALKING. Exactly 1 → AMBIGUOUS.**

| file | stand_rec | walk_rec | coverage | macro-F1 |
|---|---|---|---|---|
| rev2_t1 | 0.930 | 0.989 | 0.830 | 0.928 |
| rev2_t3 | 1.000 | 0.930 | 0.729 | 0.776 |
| **rev8_t3** (held out) | **0.966** | **1.000** | **0.981** | **0.988** |

**Zero fitted parameters.** `delta = 1°` is a sensor noise floor (5× measured standing noise came
out 0.76–0.88° on all three files independently). `1 swap` is not a chosen threshold: standing
measures **0** and walking measures **2** on every file across a **4× amplitude range** — 1 is the
only integer between them.

Known weakness: `rev2_t3` slow walking (0.22 Hz stride) yields ~0.9 swaps per 2 s window → abstains.
A **window-length** problem, not a rule problem. 4 s catches it, at the cost of wider abstention at
transitions.

**Why this matters for the pipeline:** it is a working, honest S3-physics rule that needs no
training data and no fitted constants. It is the natural first content of S3, and a real baseline
for S2 to beat.

### Validated descriptors

| descriptor | status |
|---|---|
| `ileg_minhalf` (min of `ptp(L−R)` over the window's two halves) | AUC 0.967 / 0.972; medians 0.2–0.5 standing vs 39–41 walking. **Needs per-file calibration.** |
| `interleg_offset` (median of `L−R`) | **Posture only** — AUC 0.50 for walk/stand. Separates feet-together from split-stance: baseline −3.3°, splits at **+17°** / **−22°** (opposite legs leading). |
| `angvel_LPF` | Reliable. = `d(angle)/dt`, r=0.999. |

### States richer than the human labels

Human labels are `0` / `10` / `-1`. Physics distinguishes more: `WALKING`, `WALKING_SLOW` (cadence
< 0.6× the file's own median), `STANDING_FEET_TOGETHER`, `STANDING_SPLIT_L` / `_R` (stopped
mid-stride, one leg leading), `STANDING_SHIFTING` (**unvalidated — rests on 3 and 8 windows**).

In rev2_t1: the last **61.9 s** labeled WALKING is walking at a third the cadence; the two STANDING
bouts are **different postures**. This is direct evidence for §5.2 of DOMAIN_NOTES — the labels are
coarser than the signal.

---

## 4. Architecture (decided, do not relitigate)

Four stages. **The filesystem is the only interface** — no agent-to-agent messaging, no queues.
`orchestrator.py` is a dumb sequencer.

| stage | deterministic core | agent role |
|---|---|---|
| **S1 clean** | census, rate normalization, gap segmentation, unit normalization, channel trust | exception queue only |
| **S2 ml** | train + locoeval, champion/challenger | propose → critic subagent reviews → metric-gated promotion |
| **S3 physics** | anchor features (swap rule), plots | read plots, **discover rules**, hypotheses with window-level provenance |
| **S4 report** | — (Read/Grep only) | cross-reference S1–S3, label audit, flag anomalies |

Non-negotiables:
1. Code does the work; agents judge the work.
2. No silent mutation. Quarantine + written rationale + `needs_human` on low confidence.
3. "Best" is defined by locoeval, not agent opinion.
4. Every phase closes with a `DOMAIN_NOTES.md` update.

### Track B's architectural rulings — fold these in

- **Agent proposes, test disposes.** *"Every rule asserted was wrong; every rule the tests checked
  survived or died honestly."* **Agent = rule discovery, not per-window labeling.** This is the
  sharpest available statement of the S3 agent's job and belongs in its system prompt verbatim.
- **Per-file calibration, never corpus-wide.** A fixed threshold scores `walk_rec = 0.000` on rev8
  — it calls every walking window standing. Per-file calibration scores 1.000/1.000 on the same
  file. This **validates the frozen-base-model + per-user-wrapper architecture** already decided:
  global constants are the disease, not a missing better number.
- **Do not fit constants to more data.** More data yields a better *global* constant, and global is
  the problem. rev8 wants 0.426, rev2 wants 0.605. The answer is not needing one.
- **Abstain rather than force.** Coverage 73–98%; abstained windows genuinely contain both states.
  This mirrors the human `-1` label (281–1418 ms at transitions). Note the tension: the abstained
  windows are the transitions — **where a controller most needs an answer**.
- **Recordings begin at rest** — confirmed 26/28 wide-corpus files. This is an *external* label,
  from how sessions are run rather than from any algorithm, and it gives every file a standing
  reference measured on the same person, sensor and mounting minutes earlier. **This is the
  mechanism that makes per-file calibration possible.** It is also the calibration-as-data-harvest
  insight, arriving from the physics side.

---

## 5. Where things stand

| phase | state |
|---|---|
| 0 — skeleton | ✅ done, gate passed |
| 1 — S1 deterministic core | 🟡 ~70% (was ~75%; §2.2 unit bug is new debt) |
| 2 — S1 exception agent | not started |
| 3 — S2 loop | not started |
| 4 — S3 physics | not started, but **Track B has already produced its first real content** |
| 5 — S4 report | not started |
| 6 — hardening + handoff | not started |

**Built and working:** `orchestrator.py` (run dirs, `run_meta.json` with commit SHA);
`agents/base.py` (`run_agent()` — DOMAIN_NOTES injected into every prompt, raises if missing;
PostToolUse audit hook → `run_log.jsonl`; cost → `costs.json`; `max_turns=25`); `agents/smoke.py`
(Phase 0 wiring test — **delete at Phase 2**); `stages/s1_clean/` (`config`, `census`, `manifest`,
`resample`, `clean`, `run`); `DOMAIN_NOTES.md` v2; `PLAN.md`; `README.md`.

**Verified on the real corpus:** 88 files → 5 header variants → **45-column stable contract** → 134
segments → 91 usable → **477.4 minutes** at canonical 100 Hz. Zero read failures. Canonical output
= 32 columns, parquet.

### Phase 1 remaining

1. **Fix DOMAIN_NOTES §4.1** (§2.1 above). Highest priority — it is in every agent's prompt.
2. **Unit normalization** in the clean layer (§2.2). rad/s vs deg/s, Y↔Z convention.
3. **Channel trust checks.** With §2.1 corrected, the gyro-at-rest test becomes a *verification*
   that `Gyro ≈ d(Deg)/dt` per channel, which doubles as an axis-convention check.
4. **Quarantine wiring.** Failures are recorded but nothing moves to `data/quarantine/`. This is
   the gap between "measured" and "gate passed".
5. **Index-vs-time audit** of `resample.py` (§2.5).
6. Rerun + commit + DOMAIN_NOTES update.

---

## 6. Open threads

| thread | state |
|---|---|
| **Raw-format labeled data** | Lu is "allegedly" getting it, at 100 Hz **[reported]**. When it arrives: run the census on it and verify family, rate, label encoding. Do **not** block Phase 1 on it. |
| **Only 3 labeled trials exist** | rev2_t1, rev2_t3, rev8_t3. rev8 is **133 windows / 35 s** — the perfect held-out scores rest on ~35 seconds. Do not overclaim from them. |
| **Subject identity** | Unknown for 88 of 95 files (§2.3). Genuine `needs_human`. |
| **Window length** | 4 s catches slow gait (0.22 Hz); 2 s gives transition precision. Real tradeoff, now explicit. DOMAIN_NOTES §9 still says "100 Hz / 10 ms held unless data warrants" — **data now warrants revisiting.** |
| **Gait band** | `(0.5, 3.0) Hz` is a healthy-adult default. This population runs to **0.13 Hz** (cadence 16–102 steps/min). Any band-limited feature inherits this bug. |
| **`STANDING_SHIFTING`** | Rests on 3 and 8 windows. rev2_t3's opening stand has a real weight shift at t+4.5–6.2 s — evidence, not proof. |
| **~1264 ms gap** | Systematic across 9 of 12 500 Hz files, within 14 ms of each other. Cause unknown; Lu suspects an error. Position otherwise unpredictable. |
| **~10-sample startup burst** | 26 files open with a segment of *exactly* 10 rows. Possibly related to Track B's "recordings begin at rest" (26/28) — **worth checking whether these are the same 26 files.** |
| **Wide-corpus cache** | Built at `WINDOW_S=1.0`, needs rebuild at 2.0. |
| **`L/R_Ref_Force`** | Excluded as controller setpoints (commanded, not measured). **[open]** — unconfirmed with firmware. |
| **`Hip_Deg_L/R`** | Computed but retained (bridge to the open-source dataset's `Hip_Flex_L/R`). Claude's call; Lu did not object. One line in `config.py` to reverse. |
| **id=69 reframing** | Track A's quantization-tier finding (10 + 2⁻ᵏ ms) suggests id=69's "rate clustering" read timestamp granularity, not acquisition rate. Numerically exact on 3 values but **unconfirmed against the id=69 source**. Track B's "field 2 = date block, not subject" is consistent with this. |
| **Repo location** | Local + private personal GitHub. **Asked Lu three times to check with the PL about a company org repo.** Still open. Raise once more, without nagging. |
| **Prompt caching** | Phase 0 smoke cost $0.04 — DOMAIN_NOTES (~10 KB) rides in every request, identical across calls. Ideal cache target. Revisit at Phase 3. |

---

## 7. Findings that shaped the design

In DOMAIN_NOTES with detail. Here because they explain *why* the code looks like it does:

1. **The column prefix is a lie.** `47_loco` in 62 files; index 47 is `Step` in 11 others.
   `df.iloc[:, 47]` blends a step counter into a locomotion state and never raises.
2. **Two rate eras confounded with schema AND date.** 100 Hz through 2026-01, 500 Hz from 2026-05.
   Resampling fixes rate, not era.
3. **The "odd" rates are timestamp quantization**, exactly 10 + 2⁻ᵏ ms. Same 100 Hz devices.
4. **Naive decimation aliases.** A 120 Hz component survives `[::5]` as a full-amplitude fake 20 Hz
   signal; FIR decimation removes it. Clearest argument in the repo — keep the demo.
5. **Measured vs computed** is the column line. Every column that churns position between variants
   is a computed one → dropping them collapses 5 variants into 1.
6. **`-1` ≠ `255`.** `-1` = human looked and couldn't call it; `255` = machine error. `-1` is
   training-poison and evaluation-gold — a hand-drawn map of where confidence *should* be low,
   which is the missing signal the project exists to add. Track B's abstention (73–98% coverage)
   reproduces this from physics independently.
7. **The manifest is the product.** Lu's framing: "one of the biggest problems is the lack of
   organization."

---

## 8. Track B's methodology warnings — read before analysis

Four error classes, each hit in practice:

1. **Density needs mass.** Mode-finding declared files "unimodal — one behaviour" when a 5 s stand
   was 0.4% of the file. Hit **twice**, the second time one message after invoking it as a lesson.
   *A stop is not a mode; it is a stretch of time.*
2. **Windows straddling edges.** A "32× cadence-invariant" claim measured whole phases, not
   per-window. A bout analysis deleted UNKNOWNs *then* computed runs, silently merging across gaps.
3. **Index ≠ time.** 494 Hz vs 500 → 4.2 s drift by t=320. Always select by time.
4. **The eyeball is not truth.** Claude called a 6.0 s standing bout "blurred, should be 2.5 s";
   the labels said **6,246 ms**. Same error as DOMAIN_NOTES §3.1 — measuring only the static
   plateau and treating it as ground truth.

Tests that were themselves broken: `cluster_stability` scores a **continuum at 0.974** vs real
clusters 1.000 (it measures whether k-means cuts repeatably — a gradient does, deterministically —
not whether there is anything to cut). An anchor-independence check became **tautological** once the
anchor was defined by the descriptor it was tested against. A 0.994 check was **near-trivial**
(predicting high motion from other motion channels). **GMM + BIC counts Gaussians, not modes** —
selects k=8 with evenly-spaced means on a 2-state signal.

Retracted: **"walking with no rhythm" does not exist.** Those windows are identical to normal
walking on every descriptor except a `periodicity` measure that fails at **cadence changes**, not
arrhythmia. They occur at ~69 s and ~196 s in *both* trials independently — a protocol event,
probably a turn. Claude invented a category to explain its own artifact and nearly asked for it to
be defined.

---

## 9. How to work with Lu

- **Wants full strategic reasoning before code.** Do not jump to implementation.
- **Prefers options with explicit pros/cons** over a single recommendation. When making a call
  unilaterally (e.g. `Hip_Deg`), say so and say how to reverse it.
- **Pushes back precisely and is usually right.** The "delete useless columns" push produced the
  measured-vs-computed rule — better than either starting position. The correction that the
  annotated CSV never belongs in `raw/` reshaped the data layout. The refusal to fit constants to
  more data was correct on principle. When Lu corrects a claim, take it and adjust.
- **Environment:** Windows, PowerShell 5.1 (**no `&&`**), venv at `.venv`, repo on Desktop under a
  Hangul path. Most "bugs" so far were multi-line `&&` blocks silently not running.
- **Korean when PL-facing**, written natively, 합쇼체. English for working sessions.
- **Lean, visible code:** named constants at file heads, no dead code, targeted edits over rewrites.
- **Aims for maximum output within the internship**, not a scoped target.

### Claude's calibration record (day 1)

- Claimed "columns 00–66 are byte-identical" — wrong, it's 00–44.
- **Cited whole-file |angvel| p99 as evidence of noise-at-rest.** 93.7% of that file is walking.
  Measured the wrong thing and confirmed a claim with it. **The worst error of the day** — it
  corroborated an unverified `[reported]` finding with a bogus `[measured]` tag.
- Ran `mkdir -p` with brace expansion under `sh`; created a literal `{agents,stages/...}` directory
  that rode along in every zip until Lu spotted it. The failing `touch` right after was the signal;
  Claude fixed the symptom without asking why.
- Predicted the smoke agent would cost "a fraction of a cent"; it cost $0.04.
- Claimed parquet would be "5–10× smaller"; measured 1.5×.
- Shipped `python-dotenv` but never called `load_dotenv()`.
- Guessed the ~1264 ms gap sat at a fixed position; Lu said gaps land anywhere, and was right.

**Pattern: estimates are the weak spot; measurements are sound — but only when measuring the right
thing.** Track B's independent verdict on the same tendency: *"Every rule asserted was wrong; every
rule the tests checked survived or died honestly."* Assert nothing. Measure, then state.

---

## 10. Immediate next actions

1. **Unit normalization** in the clean layer — `B_Gyro` rad/s vs `L/R_Gyro` deg/s, Y↔Z convention
   (§2.2).
2. **Raise subject identity with Lu** (§2.3) — a genuine `needs_human`, and exactly the kind of
   finding the PL asked to surface.
3. **Index-vs-time audit** of `resample.py` (§2.5).
4. **Quarantine wiring** so the Phase 1 gate can pass.
5. Then **Phase 2**: the S1 exception agent. Delete `agents/smoke.py`.
6. **S3 has a head start** — the swap rule is real, zero-parameter, and already validated on held-out
   data. It is the physics track's first content and a genuine baseline for S2 to beat.

Budget: $50 credit, ~$0.04 spent. Always be careful with cost.

