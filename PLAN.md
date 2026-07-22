# H-CARE Agent Pipeline - Structure & Progression Plan

**Owner:** Lu - **Status:** Phases 0-5 complete - S4 **fusion** built + calibrated (fused macro-F1 0.921; act-on-confidence 0.969 acc at 94% coverage, DOMAIN_NOTES Section 12), judgement agent run live; data-collection director + guarded lockbox eval + governed new-class discovery agent (`--phase 6`) wired - lockbox unopened, new-class live run pending (both Lu's call) - **Last updated:** 2026-07-22

Goal: a staged, agent-assisted pipeline for IMU locomotion data - cleaning, ML experimentation, physics-based analysis, and reporting - where deterministic code does the work, Claude agents handle judgment at defined points, and every decision is logged and reconstructible. Built on the Claude Agent SDK (Python).

Design principles (non-negotiable):

1. **Code does the work; agents judge the work.** Agents never touch data values directly and never crunch numbers themselves.
2. **Filesystem is the interface.** Stages communicate only through artifacts on disk. No agent-to-agent messaging.
3. **No silent mutation.** Failed files are recorded in the quarantine ledger (raw stays put); agent decisions are logged with rationale; low confidence escalates to `needs_human`.
4. **"Best" is defined by locoeval, not by an agent's opinion.** Champion promotion is metric-gated.
5. **Every phase ends with a DOMAIN_NOTES.md update.** Discoveries become permanent, not conversational.

---

## Repository structure

```
h-care-agents/
|-- PLAN.md                # this file
|-- DOMAIN_NOTES.md        # institutional knowledge, injected into EVERY agent prompt
|-- orchestrator.py        # runs stages in sequence; no intelligence lives here
|-- agents/                # one file per agent: system prompt + allowed tools + max_turns
|-- stages/
|   |-- s1_clean/          # deterministic clean: census, resample, channel trust, quarantine
|   |-- s2_ml/             # train + locoeval wrapper, champion/challenger loop
|   |-- s3_physics/        # anchor features (gk lineage), plots, hypothesis agent
|   `-- s4_report/         # read-only reporter + label audit
|-- data/
|   |-- raw/               # untouched inputs
|   `-- clean/             # S1 output, trusted channels only
`-- runs/YYYY-MM-DD_runN/  # per-run artifacts, append-only, never overwritten
    |-- run_log.jsonl      # every tool call (via PostToolUse hook)
    |-- costs.json         # per-stage USD from SDK result messages
    `-- s1_clean/ ... s4_report/
```

---

## Stage contracts

Every stage has the same anatomy: **deterministic core -> agent role -> outputs -> gate.**
The gate is what `orchestrator.py` checks before the next stage may run.

### S1 - Clean

| | |
|---|---|
| **Deterministic core** | Schema check; sub-Hz-precision sampling-rate measurement (resample to canonical rate or reject); gap detection; gyro unit normalization + per-file channel-trust detection (`Gyro == d(Deg)/dt`; detect sagittal axis + unit, normalize to deg/s, abstain-and-fall-back on static files); yaw-drift/session-time correlation test. *(The v1 "static-window angular-velocity noise test" and "drop provided velocity, gradient-derive from angle" policy are **retracted** - angvel is reliable, DOMAIN_NOTES Section 4.1.)* |
| **Agent role** | Exception queue only. New failure modes -> inspect, fix-vs-delete decision with written rationale, observation tags, `needs_human` flag when confidence is low |
| **Outputs** | Clean parquet; per-file `channel_trust.json`; `observations.jsonl`; `quarantine.jsonl` ledger |
| **Gate** | [ ] Every raw file accounted for as clean / quarantined / human-flagged - zero silent drops |

### S2 - ML

| | |
|---|---|
| **Deterministic core** | `dataset` (labeled rev\* trials onto the canonical 100 Hz grid, grouped/split by rev, lockbox sealed); `transform` (raw->rev2 bridge, exact, with a standing `verify_transform` guard); `features` (windowing; feature selection lives here); `train` (leave-one-rev-out); `locoeval` (blind metrics); `taxonomy` (7-bucket MECE port); `predict` (dense 100 ms stride, centre-assigned); `experiment` (spec -> run -> `decide()` -> ledger) |
| **Agent role** | Experimenter proposes ONE declarative `ExperimentSpec` per cycle (features to drop, `window_s`, `stride_s`, whitelisted hyperparameters - never code, never data access, never training) -> read-only critic reviews it **before** training, seeing the ledger as well as the proposal -> run -> `experiment.decide()` gates on the measured metric. **Neither agent can promote anything.** |
| **Outputs** | `champion.json`; `experiments.jsonl` (measured); `proposals.jsonl` (raised, incl. those killed pre-training); locoeval + taxonomy reports |
| **Gate** | [ ] Champion only ever changes via a logged, metric-justified promotion |

Promotion rule (`experiment.decide()`, the only path to champion): macro-F1 must clear
`PROMOTION_MARGIN = 0.005`; on a statistical tie, a `steady_confusion` drop of >= `0.02` promotes -
at equal accuracy, prefer the model that fails passively (DOMAIN_NOTES Section 7). Champion state is
`champion.json` + the ledger's `git_sha`, **not** a git tag as originally planned: a re-run of a
logged spec reproduces the model exactly, so the spec is the revert unit and a tag would add a
second, drift-prone source of truth.

### S3 - Physics

| | |
|---|---|
| **Deterministic core** | Anchor feature computation (periodicity, antiphase, grav_stab, gyro_energy(+)); plot generation |
| **Agent role** | Reads plots (multimodal); writes hypotheses with confidence + the specific windows/features they rest on; runs rate-invariance audit on every anchor feature |
| **Outputs** | `hypotheses.jsonl`; plots |
| **Gate** | [ ] No hypothesis without window-level provenance; [ ] No anchor feature without a rate-invariance verdict |

(+) gyro_energy already failed rate-invariance (rate experiment id=69); carried only until formally cleared or removed.

### S4 - Fusion

Restructured from a read-only *report* to a **fusion** stage (2026-07-22): the S2 learned label and
the S3 physics verdict are combined into one call **plus a confidence** - the signal the incumbent
lacks. The label-audit role is subsumed (the agent classifies disagreements, which surfaces label
problems). Fusion is deterministic; the agent judges the disagreement cases, it does not arbitrate.

| | |
|---|---|
| **Deterministic core** | `fuse` (per window -> `(label, confidence)` from S2 pred+proba and S3 verdict; policy read off the measured S2xS3 contingency, DOMAIN_NOTES Section 12); S2 `oof` (champion out-of-fold predictions as an artifact, so fusion joins a file); `run` (join S2 OOF intersect S3 anchors, score fused-vs-S2, calibrate tiers, rank disagreements) |
| **Agent role** | Judges the fuser: characterises each LOW-confidence (disagreement) case - `label_problem` / `s2_error_physics_caught` / `s3_error` / `genuine_ambiguity` - with window-level provenance, and flags label problems for human review. Read-only; never re-labels, never changes the fusion. |
| **Outputs** | `fused_windows.csv` (per-window call + confidence); `fusion_report.md`; `fusion.json` (metrics + tier calibration); `disagreements.json`; `fusion_review.jsonl` (agent) |
| **Gate** | [x] Fused output carries a calibrated confidence (HIGH 0.98 / MED 0.88 / LOW 0.79 accuracy, monotonic); [ ] every agent finding names a real window (enforced in `validate_finding`) |

---

## Progression

**Rule:** Phase N+1 does not start until Phase N's gate passes on the real corpus. Every phase closes with a DOMAIN_NOTES.md update.

### Phase 0 - Skeleton (1/2 day)
- [ ] Repo scaffolded per structure above; git initialized
- [ ] `orchestrator.py` runs one trivial agent end-to-end via Agent SDK
- [ ] PreToolUse logging hook writes `run_log.jsonl`
- [ ] Per-stage cost logging from SDK result messages -> `costs.json`
- [ ] DOMAIN_NOTES.md v1 seeded with known findings: angvel ringing at rest (+/-40-80 deg/s) -> gradient-derive instead *(RETRACTED in v3 - angvel is reliable, Section 4.1; kept here as the Phase-0 seed of record)*; yaw drift (r ~= -0.95 vs. session time) -> excluded; sampling-rate confound (99.4 vs 100.0 Hz dominated clustering geometry; gyro_energy rate-dependent); STANDING label includes ramps vs. plateau-only ground truth; NumPy 2.0 `.ptp()` removal
- [ ] Spend cap set on API account; `max_turns` default defined

### Phase 1 - S1 deterministic core (2-3 days)
*No agent yet. Pure Python.*
- [ ] Validator implements all S1 deterministic checks
- [ ] Full raw corpus processed; S1 gate passes
- [ ] Quarantine ledger manually reviewed - checks catch what they should
- [ ] Exception rate measured (sizes Phase 2's workload)
- [ ] DOMAIN_NOTES updated

### Phase 2 - S1 exception agent (1-2 days)
*First real agent. Proving ground for prompt / tool-restriction / escalation patterns.*
- [x] Agent processes the real exception queue (`agents/s1_exception.py`; `orchestrator.py --phase 2`)
- [x] Every decision has written rationale grounded in a DOMAIN_NOTES section; `needs_human` fires
      (the degenerate-time-base quarantines -> re-export) while pipeline-handled anomalies/drift ->
      `known_expected`. Read-only agent; deterministic wrapper writes `exceptions_review.jsonl`
- [x] Human review: Lu signed off on every logged decision (at sign-off: 7 needs_human = the Section 2.6
      broken-clock files -> re-export; 16 known_expected = documented gyro-axis anomalies + yaw-drift
      flags; 0 novel). **The 2026-05-19 subject-100 subset was since removed from the corpus (not
      re-exported), so a current clean run shows 1 needs_human - `20260515` alone (Section 2.6).**
- [x] DOMAIN_NOTES updated

### Phase 3 - S2 loop (2-3 days)
- [x] Train + locoeval wrapped as one command; initial champion established (`baseline_v1`, LORO
      macro-F1 **0.8730**, 23 features) and recorded in `champion.json` + ledger `git_sha`
      (git tag deliberately dropped - see the S2 contract above)
- [x] Experimenter + critic run 3-5 proposal cycles (4 agent cycles live, ~$0.21 each; 7 ledger
      entries including the manually specified ones)
- [x] >=1 promotion and >=1 rejection have occurred - **4 promotions (1 seed + 3 agent-driven),
      8 non-promotions** across 12 ledger entries. Current champion `drop_static_offset_family` at
      **0.8977** (18 features) - the static-offset angle family is per-subject zeroing bias, not gait,
      and leaks under LORO. `drop_offset_only` (0.8862) was the first champion proposed by an agent
      rather than seeded by hand
- [x] Both fully reconstructible from `experiments.jsonl` + git history alone - **verified 2026-07-21**
      by `stages/s2_ml/replay.py`: reconstruct each spec from its ledger entry and re-run it. Every
      entry replayed at that point matched **exactly** (worst metric delta <= 1e-9 across macro-F1, per-rev,
      and taxonomy fractions), so the spec + `git_sha` is a faithful revert unit - the champion is
      truly its `champion.json`, not a model blob. (Same-sha entries must be exact by determinism;
      older-sha ones also matched, meaning the deterministic core has not drifted since `d5a15b3`.
      Ledger entries added since have not been re-swept - rerun `replay.py` to re-confirm.)
- [x] DOMAIN_NOTES updated
- [x] **Gate blockers closed (2026-07-21):**
  - the critic **demonstrably bites** - `agents/s2_critic_probe.py` feeds it three proposals it must
    not approve (a verbatim ledger repeat, the same change renamed, and a false-premise claim about
    `GAIT_BAND_HZ`); it `reject`ed all three, citing the ledger entry by name on the repeats and
    Grepping `features.py` to disprove the false premise. The filter is proven, not asserted. (It also
    caught that the recorded `drop_bad_band_frac` rationale carries that same false band premise -
    a rationale defect the metric gate could never have flagged, cf. Section 11.4.)
  - the lockbox: **`rev8` stays sealed** as the final test. **`rev13` was spent** (2026-07-21) proving
    the always-Y axis decision - it scored the X-vs-Y comparison (DOMAIN_NOTES Section 6.3), so it can no
    longer serve as an unbiased final rev. `rev8` (a clean Y-plane rev) opens **once**, at the
    very end; not a blocker to closing this gate.

### Phase 4 - S3 (1-2 days)
- [x] gk/anchor work ported into stage format - swap rule (Section 10) + `ileg_minhalf`/`interleg_offset` (Section 10.1) + four anchors in `stages/s3_physics/` (a fifth, `gait_hz`, was dropped Section 10.8 as degenerate at the 2 s window); faithful to Section 10 on non-lockbox files
- [x] Rate-invariance audit completed for all four anchors; verdicts recorded (`rate_audit.json`) - periodicity/grav_stab **invariant**, antiphase + gyro_energy **rate_dependent** (gyro_energy re-derives id=69)
- [x] Hypotheses reference real windows with provenance - gate **enforced in code** (`agents/s3_physics.validate_hypothesis`) and **exercised live 2026-07-22** (`runs/2026-07-22_run2`): 4 hypotheses, 4 passed the provenance gate, 0 flagged; the agent self-respected both `rate_dependent` verdicts (used `antiphase` qualitatively only), $0.33
- [x] **Rest-anchor centering** - the live run's top hypothesis (`offset_biased_gait_defeats_swap_rule`) surfaced that a per-subject interleg DC offset (Section 4.6, the same bias `drop_static_offset_family` drops) defeats the raw swap rule; measured, then fixed by subtracting the per-file rest zero (Section 10.2) before counting swaps - corpus Pareto gain (walk 0.626->0.691, stand 0.903->0.927). See DOMAIN_NOTES Section 10.5.
- [x] DOMAIN_NOTES updated (Section 10.4, Section 10.5)
- [x] **Stride-adaptive window** (product-critical, not polish) - fixed 2 s abstains on slow gait (Section 9/Section 10.4); the assistive/rehab population walks slowly, so this is a real lab->clinic generalization requirement. **Built + measured 2026-07-22 (Section 10.6):** per-cell window sized to ~2 detected strides (autocorr period, first peak past the lag-0 shoulder), capped 6 s, standing kept short. Recovers walk-recall 0.691->0.855 for stand-recall 0.927->0.915 - most of a long window's gain, little of its cost. Added as `swap_verdict_adaptive`; disagreement ranking now built on it. Validated on the lab corpus; live-cadence sizing on streaming clinic data remains to confirm.
- [x] Lockbox sealed in the stage (train+val only) - S3 feeds an agent, so it must seal `rev8`/`rev13` too, not just the training path (tracker POSTDAY4 Section 3)

### Phase 5 - S4 fusion
- [x] **Deterministic fuser built** (`stages/s4_fusion/`, `stages/s2_ml/oof.py`) - policy grounded in the measured S2xS3 contingency, not assumed: fused macro-F1 **0.921** (S2 alone 0.898); acting on HIGH+MED confidence covers **94%** at accuracy **0.969**, stand-recall **0.897** (S2 0.865 at full coverage). Two intuitive rules measured and rejected (physics-overrides-standing, higher-confidence-wins) - DOMAIN_NOTES Section 12
- [x] Confidence signal is calibrated + monotonic (HIGH 0.98 / MED 0.88 / LOW 0.79); LOW/abstain = a machine `-1` mirroring the human `-1` (Section 5.2) - the signal the incumbent lacks
- [x] Agent judges the disagreement cases (`agents/s4_fusion.py`, provenance gate enforced in code) - **run live 2026-07-22** (`runs/2026-07-22_run3`, 6/6 findings passed the gate, $1.41): validated the fuser (rev2_t5 - physics catches a 70 s S2 stand-error), flagged a label problem, and surfaced a brief-stop weakness
- [x] Surfaces >=1 real label-audit flag (rev2_t6, motion-contaminated "standing", `abstain_correct`)
- [x] DOMAIN_NOTES updated (Section 12 + Section 12.1 - including the agent's *refuted* mechanism: brief-stop errors are a window-resolution floor, not adaptive-window bridging; verified before recording, Section 11.4)
- [x] **Row-level safety payoff** (Section 12.2): fusion cuts `steady_confusion` 0.823 -> 0.785, row-acc -> 0.942 - helps the hazardous bucket, does not solve it (still 78% of errors); the residual is label contamination
- [x] **Data-collection director built** (`stages/s4_fusion/curate.py`) - the confidence signal routes flagged windows to `relabel_candidate` (**two-vote**: BOTH S2 and physics must contradict the label - one model disagreeing is usually that model's own error, Section 12.2) / `new_class_candidate` / `collect_more`. The highest-leverage pipeline-side move: the model directs the data effort. Code never edits a label; a human confirms every candidate
- [x] **Lockbox evaluation wired** (`stages/s4_fusion/lockbox.py`) - scores the **fused** model (not S2 alone) on the sealed revs: S2 fit on every non-lockbox rev predicts them unseen, S3 verdicts computed on them, fuse, score. Guarded one-way door: refuses without `--confirm OPEN-LOCKBOX`. Not opened - waits for Lu to call the fused model final
- [ ] **Open (recorded, not fixed):** brief (< 2 s) stops sit below the window resolution floor (Section 10.6/Section 12.1); confidence quarantines these (all 64 LOW -> abstained)
- [ ] **Governed new-class discovery BUILT, not yet run live** (`stages/s4_fusion/newclass.py` + `agents/s4_newclass.py`, `orchestrator.py --phase 6`): the `new_class_candidate` spans (28 spans / 347 windows across 6 revs, mean antiphase **-0.30** = legs moving *together*, not gait) become a physics-profile evidence bundle; the agent PROPOSES classes; a deterministic gate validates provenance + cluster mass (>=4 spans across >=2 revs) and routes every proposal to `needs_human`. Read-only, orthogonal to the deployed algorithm - code never adds a class or edits a label (Section 11.2)

**The honest bottom line (Section 12.2):** the dominant remaining lever is **data** - better labels on the motion-contaminated "standing" and more subjects (n=5, rev69 dominant) - not more macro-F1 in code. The pipeline's job now is to *direct* that data effort, which the curation queue does.

### Phase 6 - Hardening + handoff (remaining time)
- [ ] `max_turns` caps verified on every agent
- [ ] Clean-room run: fresh clone -> raw data -> report, no manual intervention
- [ ] Handoff doc assembled (~= DOMAIN_NOTES + this PLAN + run walkthrough)

**Total estimate:** 8-12 working days for the full loop. A genuinely useful system exists from end of Phase 2 onward.

**Sacrifice order if the clock bites:** Phase 4 first (S3 is enrichment), then Phase 5. Never Phases 1-3 - they are the handoff-critical spine.

---

## Cost & safety rails

- API is pay-as-you-go; expected full four-stage run ~= $0.50-3 on Sonnet-class, development month plausibly $50-150 before routing optimizations
- Routing: S1 tagging + S4 reporting -> Haiku-class; S2 experimenter/critic -> Sonnet-class
- Prepaid credits with hard spend cap; per-stage cost visible in `costs.json` from day one
- Agents never authorized to: delete raw data, modify DOMAIN_NOTES without human review, or change the champion outside the S2 promotion path
