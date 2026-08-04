> # HISTORICAL RECORD — nothing regenerates this file
>
> Written by `stages/s3_physics/run.py` on **2026-08-03**. That module was deleted with the
> S4 fusion stage on 2026-08-04, and with it the only thing that ever produced this page.
> **The numbers below are a snapshot of that run and will not change again**, whatever
> happens to the corpus, the window grid or the swap rule. Do not cite them as current
> output; do not "fix" them by editing this file.
>
> Kept rather than deleted because two things in it exist nowhere else. The swap-verdict
> tally (`AMBIGUOUS` 1,543 -> 380 under the stride-adaptive span) is the measured evidence
> `DOMAIN_NOTES.md` §10.6/§10.7 rests on, and it came from `anchors.csv` — a 1.4 MB
> per-window table that existed only to feed the fusion join and is gone. Regenerating that
> tally today would mean reviving a stage deleted for having no reader.
>
> **Where the live equivalents are:**
>
> | this page's section | now produced by |
> |---|---|
> | rate-invariance audit | `python -m stages.s3_physics.rate_audit` -> `rate_audit.md` |
> | swap-rule verdicts, per trial | `python -m stages.s3_physics.label_audit` -> `label_audit.md` |
> | swap-rule verdicts, per window | **nothing** — this snapshot is the only record |
>
> The rate-audit table below is reproduced exactly by the live `rate_audit.md` (`antiphase`
> 0.0005, `gyro_energy` 0.4994 rate-dependent), which is a useful check that the rule itself
> did not drift when its driver was deleted and its audit re-homed.

---

# S3 Physics

- windows: **6,339** across **7** revs
- rest zero untrusted on **90** windows (1.4%) — those recordings never rest, so their interleg zero is a whole-recording median rather than a measured standing posture (§10.2)

## Swap-rule verdicts

The rule has zero fitted parameters: `delta = 1°` is a sensor noise floor and `1 swap` is the only integer between measured standing (0) and measured walking (2). The adaptive column sizes the span to ~2 detected strides (§10.6/10.7), which is what rescues slow gait from abstaining.

| verdict | fixed 2 s span | stride-adaptive span |
|---|---|---|
| `STANDING` | 955 | 833 |
| `AMBIGUOUS` | 1,543 | 380 |
| `WALKING` | 3,841 | 5,126 |

Median adaptive span: **3.3 s** (base 2.0 s, max 6.0 s).

## Rate-invariance audit

Every window decimated 100 Hz -> 50 Hz through S1's anti-aliasing FIR, anchors recomputed, tolerance 0.1. Gated to the 3,841 windows the swap rule calls WALKING (of 6,339) — standing has no cadence to compare, and the gate keeps the audit label-free.

Each window is decimated with **128 samples of real context** either side, then trimmed back: on a bare window the filter's edge transient covers over half the samples and reads as a rate dependence that is not there (it condemned `antiphase` at Δ 0.1734 vs 0.0005 corrected). 10 windows sat too close to a segment boundary to pad on both sides and were skipped; 3,831 were audited.

| anchor | metric | median Δ | p90 Δ | verdict |
|---|---|---|---|---|
| `periodicity` | abs | 0.0273 | 0.0324 | **invariant** |
| `antiphase` | abs | 0.0005 | 0.0024 | **invariant** |
| `grav_stab` | abs | 0.0001 | 0.0003 | **invariant** |
| `gyro_energy` | rel | 0.4994 | 0.5017 | **rate_dependent** |

*An anchor that fails is not a bug — `gyro_energy` is defined the way that fails, on purpose, as the audit's negative control. An audit that has never rejected anything is not evidence that the rest passed (§11.1).*