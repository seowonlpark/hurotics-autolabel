# hurotics-autolabel - Pipeline Technical Reference

this doc is the *how*: what each part does and how to run it. the *why* behind each choice lives in
`BUILDLOG.md`, cited here by its stable entry IDs (for example `[S1-3]`). to run the pipeline rather
than read about it, see `README.md`.

what it does, in one line: turns raw wearable-sensor logs into a locomotion call (standing / walking)
for every window of a recording, plus a **confidence** on that call. deterministic code does all the
computation; language-model agents judge at four defined points and can change nothing on their own
(`[X-1]`).

---

## 1. Architecture and non-negotiables

four stages in sequence - **S1 Clean**, **S2 ML**, **S3 Physics**, **S4 Fusion** - run by a
deliberately unintelligent orchestrator.

each stage has the same shape: a deterministic core does the work, an agent judges it at one point,
the stage writes artifacts, and a gate in code decides whether the next stage may run.

four principles hold everywhere, and the rest of the design follows from them:

1) code does the work; agents judge it (`[X-1]`)
2) the filesystem is the only interface between stages (`[X-2]`)
3) no silent mutation; needs_human is a deliverable (`[X-3]`)
4) assert nothing; measure, then state (`[X-4]`)

---

## 2. Data model

raw layout: `data/raw/<YYYYMMDD[_n]>/*.csv`, one folder per session, date taken from the folder name.
raw is never modified in place and never committed to git.

grouping unit: for the labeled data used to train and evaluate, the unit is the **rev** - one subject
recorded on one day - which is what cross-validation holds out, so a score measures generalization to
an unseen subject rather than memorization (`[S2-1]`).

canonical grid: S1 resamples every
recording to a canonical **100 Hz** and keeps only measured
channels - 30 columns plus `Label` when present (`[S1-0]`, `[S1-2]`).

- rate is normalized with an anti-aliasing filter, never naive slicing (`[S1-3]`)
- the odd sub-100 Hz rates are clock quantization of the same devices, grid-corrected (`[S1-4]`)
- gaps split a recording into segments that are never resampled across (`[S1-5]`)

what the shipped model labels: the classifier that ships trains on two classes - **stand** and
**walk** - from a four-channel sagittal view (`L/R_ang_LPF`, `L/R_angvel_LPF`), and treats the
human-unknown code as an abstention rather than a third class. that scope is a property of the
labels in the development corpus, not a ceiling of the pipeline: the same machinery trains more
classes from richer data - more channels, more labeled states. section 13 is the how.

the lockbox (held-out set): a subset of whole revs is sealed before any training and kept out of the
whole pipeline - not only training, but any stage whose output reaches an agent or a person
(`[S3-5]`). it is opened exactly once, at the very end, behind an explicit gate, and is single-use:
once opened it is spent (`[S4-5]`). you designate your own held-out revs; the pipeline seals whatever
you designate and asserts in code that it never enters training.

---

## 3. S1 - Clean

what: turns raw logs into one canonical, trustworthy table per recording, fully deterministically. a
**census** measures the corpus and judges nothing; **clean** resamples to the canonical grid, keeps
measured channels only, normalizes each gyro channel to deg/s by per-file measurement, records
per-file trust, and quarantines anything it cannot clean (`[S1-0]`).

why (build log):

- columns resolve by name, since the positional prefix is not a stable id (`[S1-1]`)
- computed firmware channels are dropped, which also collapses the header variants into one schema (`[S1-2]`)
- decimation is anti-aliased (`[S1-3]`); segments never cross a gap (`[S1-5]`)
- gyro units + axes are resolved per file by measurement, and sagittality is left to S2 (`[S1-6]`)
- a broken time base is quarantined, not repaired (`[S1-7]`)

artifacts + entry points: `stages/s1_clean/` (`run.py` census, `clean.py` clean; constants in
`config.py`). outputs `data/clean/**.parquet` with a `channel_trust.json` sidecar per file, plus
per-run `census.md`, `manifest.jsonl`, `clean_report.md`, `segments.jsonl`, `observations.jsonl`,
`quarantine.jsonl`. the S1 exception agent (`[A-1]`) then triages the flagged items.

---

## 4. S2 - ML

what: learns a locomotion classifier (a random forest) from the labeled trials and gates every change
on a measured metric. the deterministic core is a chain - `dataset`, `transform`, `features`, `train`
(leave-one-rev-out CV), `locoeval`, `taxonomy`, `experiment` (`[S2-0]`).

why (build log):

- CV holds out a whole rev because subject leakage is the confound (`[S2-1]`)
- the raw-to-derived transform is reconstructed exactly and guarded, with a causal filter (`[S2-2]`)
- a challenger is a declarative spec, never code (`[S2-3]`)
- the champion is its spec + a commit, not a model blob (`[S2-4]`)
- promotion is metric-gated with a fail-passive tiebreaker (`[S2-5]`)
- the row-level taxonomy needs dense inference (`[S2-7]`)

the agent loop: an experimenter proposes one spec (`[A-2]`); a critic reviews it against the full
ledger before any training is spent (`[A-3]`); code, not either agent, decides promotion. neither
agent can change the champion.

artifacts + entry points: `stages/s2_ml/`. outputs in `runs/s2_ml/`: `champion.json`, the
append-only `experiments.jsonl` ledger, `proposals.jsonl`, `locoeval.md`/`.json`, `taxonomy.json`,
`model_meta.json`, and `oof_champion.csv` (the fusion stage's input). `champion.joblib` +
`model_meta.json` are the ready-to-run deployment classifier, refit by `train.py` *from* the champion
spec (not a rival record of it, `[S2-4]`); a `--s2-cycle` promotion triggers an `s2_refit` step so they
never lag the spec.

---

## 5. S3 - Physics

what: computes model-free physics features ("anchors") per window, audits each for rate-invariance,
renders one figure per trial, and ranks where the physics disagrees with the label. centerpiece is the
**swap rule** (`[S3-0]`, `[S3-1]`).

why (build log):

- the swap rule has no fitted parameters and needs no labels - an independent baseline, not a copy (`[S3-1]`)
- every anchor earns a rate-invariance verdict; one is defined the failing way on purpose (`[S3-2]`)
- the swap count is recentered on each recording's rest posture (`[S3-3]`)
- the window is sized to the detected stride, so slow gait does not silently abstain (`[S3-4]`)
- because its figures feed an agent, S3 seals the held-out set like training does (`[S3-5]`)

artifacts + entry points: `stages/s3_physics/` (`anchors.py`, `rate_audit.py`, `plots.py`, `run.py`).
outputs in `runs/s3_physics/`: `anchors.csv`, `rate_audit.json`, `disagreement.json`, `plots/`. the
hypothesis agent (`[A-5]`) then reads the figures under a provenance gate enforced in code.

---

## 6. S4 - Fusion

what: combines the S2 label + the S3 verdict into a single call plus a **confidence** - the signal the
existing device algorithm lacks. the deterministic core joins the model's out-of-fold predictions with
the S3 anchors, applies a zero-parameter fusion policy, calibrates the confidence tiers, and ranks the
disagreements (`[S4-0]`).

why (build log):

- fusion is late, not a feature inside the model, so the swap rule's guarantee and an actionable tier survive (`[S4-1]`)
- the policy is read off the measured agreement, not assumed; two intuitive rules were rejected (`[S4-2]`)
- the tiers are calibrated and LOW is a deliberate abstention (`[S4-3]`)
- the confidence signal doubles as a data-collection director with a two-vote relabel gate (`[S4-4]`)
- the one honest test on the sealed set decided the deployable is the confidence-gated system (`[S4-5]`)

beyond the two labels: a governed discovery step proposes classes the taxonomy misses, validates each
for provenance + mass, and routes every proposal to a human regardless of confidence (`[S4-6]`, `[A-7]`).

artifacts + entry points: `stages/s4_fusion/` (`fuse.py`, `run.py`, `curate.py`, `newclass.py`,
`export.py`, and the single-use `lockbox.py`). outputs in `runs/s4_fusion/`: `fused_windows.csv`,
`fusion.json`, `fusion_report.md`, `disagreements.json`. the judgement agent (`[A-6]`) then
characterizes the disagreement cases.

---

## 7. Agent framework

all agents share one small framework (`agents/base.py`, `run_agent`). an agent is a fixed spec - a
role prompt, an allowed-tool list, a model, and a turn cap (`[A-0]`). four properties matter:

1) institutional context is injected on every call - the shared domain notes are prepended to each
   agent's prompt, and the framework refuses to run without them. the exact prompt is saved per run,
   and passed to the model as a file (the notes outgrew the OS command-line length limit).
2) effectively read-only - the only tools any agent gets are Read + Grep, so none can change data,
   promote a model, or edit a label. a deterministic wrapper enforces the real gate in code (`[X-1]`).
3) bounded - a turn cap makes a stuck agent fail fast (for example 10 for the critic, 30 for the
   figure-reading agents).
4) logged + costed - every tool call is appended to a per-run log, and each run's cost is recorded.

most agents run on a sonnet-class model; the one cheap tagging job (`[A-1]`) runs on haiku-class. the
seven agents are catalogued in the build log, `[A-1]` through `[A-7]`.

---

## 8. Commands

every command runs from the repo root with the virtual environment active.

the whole pipeline, in order (`run_pipeline.py` chains the stages, checks each stage's output before
the next runs, and never touches the single-use held-out set):

```
python run_pipeline.py                # free deterministic spine -> the fusion report
python run_pipeline.py --with-agents  # also run the four paid agent steps
python run_pipeline.py --s2-cycle     # also run one champion/challenger cycle (paid)
python run_pipeline.py --s2-cycles N  # run N champion/challenger cycles back to back (paid)
python run_pipeline.py --list         # print the selected steps and exit
python run_pipeline.py --dry-run      # print each command without running it
```

individual stages (finer control, or to regenerate one report):

```
python -m stages.s1_clean.run   --out runs/s1_census      # census.md, manifest.jsonl
python -m stages.s1_clean.clean --out runs/s1_clean       # data/clean/, clean_report.md, ledgers
python -m stages.s2_ml.train    --out runs/s2_ml --taxonomy   # locoeval.md, taxonomy.json
python -m stages.s2_ml.verify_transform                   # standing raw->derived guard
python -m stages.s2_ml.oof      --out runs/s2_ml          # oof_champion.csv (fusion input)
python -m stages.s3_physics.run                           # anchors.csv, rate_audit.json, plots/
python -m stages.s4_fusion.run                            # fused_windows.csv, fusion_report.md
python -m stages.s4_fusion.export                         # results/*.csv (per-row fused call)
```

the agent steps (each spends API credit):

```
python orchestrator.py s1_exception    # triage the S1 exception queue
python orchestrator.py s2_cycle        # one experimenter + critic cycle
python orchestrator.py s3_physics      # physics core, then the hypothesis agent
python orchestrator.py s4_fusion       # fuse, then the judgement agent
python orchestrator.py s4_newclass     # governed new-class discovery
```

each stage writes into a fixed `runs/<stage>/`, and each agent run also writes a timestamped
`runs/<date>_runN/` holding its prompt, tool log, and cost. the single-use held-out evaluation
(`stages/s4_fusion/lockbox.py`) is deliberately not part of any of the above and is guarded so it
cannot run by accident.

---

## 9. Report - what you get at the end

the deliverable is the fusion report and the artifacts around it:

- `runs/s4_fusion/fusion_report.md` - the headline: the fused call scored against the model alone,
  and the coverage-vs-accuracy trade at each confidence tier. the human-readable summary.
- `runs/s4_fusion/fused_windows.csv` - one row per scored window: the fused call + its confidence
  tier (HIGH / MED / LOW).
- `results/<recording>.csv` (from the export command) - each labeled recording written back out with
  two columns appended: the fused label + its confidence, mapped densely onto every row. rows the
  pipeline did not score - transitions, dropped or short segments, and every row of the sealed set -
  are left blank on purpose; the export never guesses a value it did not compute.

how the confidence is meant to be used: the call carries a tier, not just a label. a downstream
controller acts on HIGH + MED and holds on LOW, where LOW is an explicit abstention, not a
low-probability guess (`[S4-3]`). acting only on the confident tiers covers most of the data at higher
accuracy than the model at full coverage, and the abstentions land where the two views disagree.

supporting outputs: `disagreements.json` + the agent's `fusion_review.jsonl` explain the low-confidence
cases; the curation queue routes flagged windows to relabel / new-class / collect-more (`[S4-4]`); and
`s4_newclass` proposals surface structure the two-label taxonomy misses, each marked for a human (`[S4-6]`).

---

## 10. Reproducibility

every run records its commit: each `runs/<date>_runN/` carries a `run_meta.json` with the git commit
that produced it, so a gitignored run can always be tied back to the code that made it. each agent run
also writes `run_log.jsonl` (every tool call) and `costs.json` (per-agent spend).

the champion is a spec, not a blob: its identity is its declarative spec + the commit it was produced
at, in `champion.json` (`[S2-4]`). because the model is deterministic, re-running that spec reproduces
it exactly. `stages/s2_ml/replay.py` does this on demand - `--all` re-runs every ledger entry, `--name
<spec>` a single one - and confirms each reproduces to numerical tolerance.

clean-room re-seeding: because `runs/` is not in git, a fresh checkout has no champion. the runner
re-seeds it deterministically from the champion spec baked into `run_pipeline.py`, so a clean clone
plus raw data reproduces the recorded champion with no manual step. same idea as replay, applied to
the whole pipeline: the spec is the revert unit.

---

## 11. Known limits and open questions

- small subject pool: the labeled evidence in development came from five subjects, one dominant. LORO
  is the honest bound, but it is a bound on a thin pool - the strongest lever is more subjects, not
  more model (`[S2-1]`, `[S2-9]`).
- standing under-recall in the steady middle: the model's characteristic failure on unseen subjects,
  a sustained (safety-relevant) error, and a class-balance / label-quality ceiling more than a feature
  gap (`[S2-9]`).
- brief stops below the window floor: a stop shorter than the window can't be resolved by a windowed
  method; handled by abstention, not by shrinking the window (`[S3-4]`).
- rate-dependent anchors: two anchors track the sampling grid and must not be read as body-claims
  without acknowledging it (`[S3-2]`).
- the sealed set is spent: it has been opened, the raw fused label did not beat the model
  out-of-sample, and any future generalization number needs a fresh sealed set (`[S4-5]`).

---

## 12. Support tools

before ingesting a batch of raw CSVs, validate their schema and encoding with the companion tool:

**https://github.com/seowonlpark/hurotics-imu-csv-validation**

it checks what S1 assumes about an input file up front, so a malformed export is caught before it
reaches the pipeline rather than surfacing later as a quarantine or an odd number.

---

## 13. Extending to more classes and channels

nothing here is wired to two labels by nature. the stand/walk model is what the development labels
supported, not a limit of the code: the ML stage is multiclass-native (the forest and macro-F1
already average over whatever classes exist), so the reach of the model is set by the labels and
channels you feed it.

to add a class (say a third locomotion state):

1) label it - put trials carrying the new `Label` code under `data/labeled/rev*/`, same filename
   pattern. a class the model never sees in training it can never predict.
2) declare it - in `stages/s2_ml/dataset.py` add the code and extend `TRAIN_CLASSES`; in
   `stages/s2_ml/locoeval.py` add its name to `CLASS_NAMES` (the report's confusion table keys off
   the class set). nothing else in S2 is class-specific - windowing, features, training, and
   promotion all read the class set.
3) re-run S2 - LORO macro-F1 now scores the new class alongside the old ones; a class with too few
   or too impure windows surfaces as weak recall, not a crash (`[X-4]`).

to add input channels (beyond the four sagittal features):

1) the raw is usually already there - S1 keeps 30 measured channels, so the four-feature view is an
   S2 choice, not an S1 truncation (`[S1-2]`).
2) extend `FEATURES` in `dataset.py` and the per-channel block in `stages/s2_ml/transform.py`
   (`window_features`); if the new channel is derived from raw, extend `raw_to_features` too and
   let `verify_transform.py` guard it against train/serve skew (`[S2-2]`).
3) let the metric decide - a challenger spec can drop or keep the new features and the ledger scores
   it like any other experiment (`[S2-3]`). add channels wide, keep what the score justifies.

where it stops being automatic: S3 and S4 are built around the binary stand/walk split. the swap
rule is a stand-vs-walk physical test (`[S3-1]`), and the fusion policy + confidence tiers are read
off two-class agreement (`[S4-2]`, `[S4-3]`). a new class rides on the ML label immediately, but it
gets no independent physics second-opinion and no calibrated confidence until an anchor and a fusion
rule are designed for it. so many-class *labeling* is an S2 change today; many-class *confidence* is
new physics + fusion work - the honest boundary, not an oversight.
