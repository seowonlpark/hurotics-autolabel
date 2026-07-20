# H-CARE Agent Pipeline — Structure & Progression Plan

**Owner:** Lu · **Status:** Phase 0 complete (gate passed) · Phase 1 in progress · **Last updated:** 2026-07-20

Goal: a staged, agent-assisted pipeline for IMU locomotion data — cleaning, ML experimentation, physics-based analysis, and reporting — where deterministic code does the work, Claude agents handle judgment at defined points, and every decision is logged and reconstructible. Built on the Claude Agent SDK (Python).

Design principles (non-negotiable):

1. **Code does the work; agents judge the work.** Agents never touch data values directly and never crunch numbers themselves.
2. **Filesystem is the interface.** Stages communicate only through artifacts on disk. No agent-to-agent messaging.
3. **No silent mutation.** Failed data goes to quarantine; agent decisions are logged with rationale; low confidence escalates to `needs_human`.
4. **"Best" is defined by locoeval, not by an agent's opinion.** Champion promotion is metric-gated.
5. **Every phase ends with a DOMAIN_NOTES.md update.** Discoveries become permanent, not conversational.

---

## Repository structure

```
h-care-agents/
├── PLAN.md                # this file
├── DOMAIN_NOTES.md        # institutional knowledge — injected into EVERY agent prompt
├── orchestrator.py        # runs stages in sequence; no intelligence lives here
├── agents/                # one file per agent: system prompt + allowed tools + max_turns
├── stages/
│   ├── s1_clean/          # deterministic validator + derivation policy + exception agent
│   ├── s2_ml/             # train + locoeval wrapper, champion/challenger loop
│   ├── s3_physics/        # anchor features (gk lineage), plots, hypothesis agent
│   └── s4_report/         # read-only reporter + label audit
├── data/
│   ├── raw/               # untouched inputs
│   ├── clean/             # S1 output, trusted channels only
│   └── quarantine/        # failed files/channels, never deleted
└── runs/YYYY-MM-DD_runN/  # per-run artifacts, append-only, never overwritten
    ├── run_log.jsonl      # every tool call (via PreToolUse hook)
    ├── costs.json         # per-stage USD from SDK result messages
    ├── s1_clean/ … s4_report/
```

---

## Stage contracts

Every stage has the same anatomy: **deterministic core → agent role → outputs → gate.**
The gate is what `orchestrator.py` checks before the next stage may run.

### S1 — Clean

| | |
|---|---|
| **Deterministic core** | Schema check · sub-Hz-precision sampling-rate measurement (resample to canonical rate or reject) · gap detection · gyro unit normalization + per-file channel-trust detection (`Gyro == d(Deg)/dt`; detect sagittal axis + unit, normalize to deg/s, abstain-and-fall-back on static files) · yaw-drift/session-time correlation test. *(The v1 "static-window angular-velocity noise test" and "drop provided velocity, gradient-derive from angle" policy are **retracted** — angvel is reliable, DOMAIN_NOTES §4.1.)* |
| **Agent role** | Exception queue only. New failure modes → inspect, fix-vs-delete decision with written rationale, observation tags, `needs_human` flag when confidence is low |
| **Outputs** | Clean CSVs · per-file `channel_trust.json` · `observations.jsonl` · quarantine folder |
| **Gate** | ☐ Every raw file accounted for as clean / quarantined / human-flagged — zero silent drops |

### S2 — ML

| | |
|---|---|
| **Deterministic core** | Existing RF training pipeline + locoeval, wrapped as one command: train → evaluate → compare vs. champion |
| **Agent role** | Experimenter proposes a change (feature, layer, hyperparameter) → read-only critic subagent reviews the proposal (grounded in locoeval reports + DOMAIN_NOTES) → run → promote/reject on macro-F1 + error taxonomy → log to `experiments.jsonl` → git commit. Champion = git tag; revert = checkout |
| **Outputs** | Model artifacts · locoeval reports · `experiments.jsonl` |
| **Gate** | ☐ Champion only ever changes via a logged, metric-justified promotion |

### S3 — Physics

| | |
|---|---|
| **Deterministic core** | Anchor feature computation (periodicity, antiphase, grav_stab, gait_hz, gyro_energy†) · plot generation |
| **Agent role** | Reads plots (multimodal) · writes hypotheses with confidence + the specific windows/features they rest on · runs rate-invariance audit on every anchor feature |
| **Outputs** | `hypotheses.jsonl` · plots |
| **Gate** | ☐ No hypothesis without window-level provenance · ☐ No anchor feature without a rate-invariance verdict |

† gyro_energy already failed rate-invariance (rate experiment id=69); carried only until formally cleared or removed.

### S4 — Report

| | |
|---|---|
| **Deterministic core** | None — agent-only stage, tools restricted to Read/Grep |
| **Agent role** | Cross-references S1 observations, S2 experiment log, S3 hypotheses · flags ML-vs-physics disagreements · **label audit:** elevated late/early/flicker errors → hypothesis space must include "label definition is wrong" → human-review flag (precedent: labeled STANDING included accel/decel ramps vs. plateau-only ground truth) |
| **Outputs** | `report.md` (human-readable) · `anomalies.jsonl` (append-only ledger) |
| **Gate** | ☐ Every anomaly entry names its source stage, file, and window range |

---

## Progression

**Rule:** Phase N+1 does not start until Phase N's gate passes on the real corpus. Every phase closes with a DOMAIN_NOTES.md update.

### Phase 0 — Skeleton (½ day)
- ☐ Repo scaffolded per structure above; git initialized
- ☐ `orchestrator.py` runs one trivial agent end-to-end via Agent SDK
- ☐ PreToolUse logging hook writes `run_log.jsonl`
- ☐ Per-stage cost logging from SDK result messages → `costs.json`
- ☐ DOMAIN_NOTES.md v1 seeded with known findings: angvel ringing at rest (±40–80 deg/s) → gradient-derive instead · yaw drift (r ≈ −0.95 vs. session time) → excluded · sampling-rate confound (99.4 vs 100.0 Hz dominated clustering geometry; gyro_energy rate-dependent) · STANDING label includes ramps vs. plateau-only ground truth · NumPy 2.0 `.ptp()` removal
- ☐ Spend cap set on API account; `max_turns` default defined

### Phase 1 — S1 deterministic core (2–3 days)
*No agent yet. Pure Python.*
- ☐ Validator implements all S1 deterministic checks
- ☐ Full raw corpus processed; S1 gate passes
- ☐ Quarantine folder manually reviewed — checks catch what they should
- ☐ Exception rate measured (sizes Phase 2's workload)
- ☐ DOMAIN_NOTES updated

### Phase 2 — S1 exception agent (1–2 days)
*First real agent. Proving ground for prompt / tool-restriction / escalation patterns.*
- ☐ Agent processes the real quarantine queue
- ☐ Every decision has written rationale; `needs_human` fires on low confidence
- ☐ Human review: agree with (or at least understand) every logged decision
- ☐ DOMAIN_NOTES updated

### Phase 3 — S2 loop (2–3 days)
- ☐ Train + locoeval wrapped as single command; initial champion established and git-tagged
- ☐ Experimenter + critic run 3–5 proposal cycles
- ☐ ≥1 promotion and ≥1 rejection have occurred
- ☐ Both fully reconstructible from `experiments.jsonl` + git history alone
- ☐ DOMAIN_NOTES updated

### Phase 4 — S3 (1–2 days)
- ☐ gk/anchor work ported into stage format
- ☐ Rate-invariance audit completed for all five anchors; verdicts recorded
- ☐ Hypotheses reference real windows with provenance
- ☐ DOMAIN_NOTES updated

### Phase 5 — S4 (1 day)
- ☐ Reporter runs on a full 1→3 run
- ☐ Report surfaces ≥1 real ML-vs-physics disagreement or label-audit flag (bland agreement = prompt failure, iterate)
- ☐ DOMAIN_NOTES updated

### Phase 6 — Hardening + handoff (remaining time)
- ☐ `max_turns` caps verified on every agent
- ☐ Clean-room run: fresh clone → raw data → report, no manual intervention
- ☐ Handoff doc assembled (≈ DOMAIN_NOTES + this PLAN + run walkthrough)

**Total estimate:** 8–12 working days for the full loop. A genuinely useful system exists from end of Phase 2 onward.

**Sacrifice order if the clock bites:** Phase 4 first (S3 is enrichment), then Phase 5. Never Phases 1–3 — they are the handoff-critical spine.

---

## Cost & safety rails

- API is pay-as-you-go; expected full four-stage run ≈ $0.50–3 on Sonnet-class, development month plausibly $50–150 before routing optimizations
- Routing: S1 tagging + S4 reporting → Haiku-class; S2 experimenter/critic → Sonnet-class
- Prepaid credits with hard spend cap; per-stage cost visible in `costs.json` from day one
- Agents never authorized to: delete raw data, modify DOMAIN_NOTES without human review, or change the champion outside the S2 promotion path
