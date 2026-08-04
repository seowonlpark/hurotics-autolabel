"""Label a CSV: per-row state, confidence, and — where confidence is low — WHY.

    python -m stages.s2_ml.label input.csv --out labelled.csv
    python -m stages.s2_ml.label input.csv --preset high_precision
    python -m stages.s2_ml.label input.csv --threshold 0.88

This is the deliverable. Everything else in S2 exists to make this row honest.

**It takes either shape of file**, dispatched on FAMILY resolved from the header by name
(§1.3): the `lpf_view` (`L/R_ang_LPF`, `L/R_angvel_LPF`) or a **raw device log**,
whose four channels are rebuilt by `transform.raw_csv_to_features` first. The raw path is
the one a device actually produces, and until 2026-08-03 it did not exist — the pipeline
could only label files that had already been through HUROTICS' MATLAB (caveats §3.1). It
refuses rather than guesses: an unmapped hardware revision, a file whose measured
permutation contradicts the axis about to be read, a gyro that is not natively deg/s, or a
final timestamp that cannot carry the filter all abstain with a stated reason and no
output.

**Abstention is the point.** A confident wrong label is worse than an admitted unknown,
so every row carries `confidence` and, below the threshold, `ambiguous=True` with a
`reason` and an `alternative`. Coverage bought at each threshold is measured in
`OPERATING_POINTS.md`; the default commits to 84.46% of rows and is right on 0.9904 of
them, with every held-out subject independently above 95% (the worst, `rev5`, at 0.9517).

**Rows in, rows out.** The model works on a 100 Hz canonical grid, but the output is
indexed by the INPUT file's own rows: probabilities are computed on the grid and mapped
back by time. A caller gets a row per row of their own file, not a resampled one they
have to re-join.

**What lands on disk is `verdict_frame`**: the six columns the annotated corpus uses
(`Time`, the four `*_LPF` features, `Label`) followed by `guess`, `confidence`,
`ambiguous` and `reason`. That prefix is deliberate — a file this module writes opens like
one HUROTICS annotated, so the same reader takes both — and the rest is the part a
machine-labelled file needs and a hand-annotated one does not. `--full` returns the whole
working frame instead; `--no-verdict` drops everything after `Label`, giving a file
indistinguishable in shape from an annotated one.

**`Label` and `guess` answer different questions and both are emitted.** `Label` is the
committed call, so an abstention shows as -1 and the opinion behind it is gone. `guess` is
that opinion: STAND or WALK on every row that was scored at all, whether or not the
threshold was cleared. Where they disagree IS the review queue — a reviewer settling an
ambiguous row should see what the model thought, not have to re-derive it.

**A row is only scored where it was measured.** Rows in a segment shorter than one window,
or outside any segment, come back `ambiguous` with reason `uncovered` and no guess at all.
Extrapolating a class into time that was never observed is exactly the invention §3.1
forbids.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from stages.console import use_replacement_encoding
from stages.s1_clean.census import family_of, read_header, strip_prefix
from stages.s1_clean.config import CANONICAL_HZ, FAMILY_MARKERS, LABEL_UNKNOWN_MACHINE
from stages.s1_clean.resample import resample_file
from stages.s1_clean.validate import health
from stages.s2_ml.dataset import (
    FEATURES,
    HUMAN_UNKNOWN,
    LABEL_COL,
    STAND,
    TIME_COL,
    WALK,
)
from stages.s2_ml.features import WindowSpec, feature_names, rest_reference, segment_features
# S3 imports S2, never the reverse — except here, and it is a leaf import of the swap rule
# only (`s3_physics.serve` -> `anchors` -> `s2_ml.{dataset,features,rest}`), so it closes no
# cycle. The alternative was a second copy of the rule on the serve side, which is the
# skew this repo refuses everywhere else.
from stages.s3_physics.plausibility import check as plausibility_check
from stages.s3_physics.serve import STANDING, WALKING, row_verdict, segment_verdicts
from stages.s2_ml.transform import (
    AxisConflictError,
    DegenerateClockError,
    GyroUnitError,
    NotRawDeviceError,
    UnknownVariantError,
    raw_csv_to_features,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = REPO_ROOT / "runs" / "s2_ml"

RAW_DEVICE, LPF_VIEW = "raw_device", "lpf_view"

# Every way the raw path can decline a file. Caught as a group at the CLI boundary so a
# refusal prints its reason instead of a traceback — an operator holding an unservable
# recording needs to read WHY, and each of these messages says so and names the remedy.
SERVE_REFUSALS = (UnknownVariantError, AxisConflictError, GyroUnitError,
                  DegenerateClockError, NotRawDeviceError, FileNotFoundError)

CLASS_NAME = {STAND: "stand", WALK: "walk"}

# ---------------------------------------------------------------------------
# The physics floor and ceiling. Both DEFAULT OFF.
#
# `PHYSICS_CEILING` abstains where the swap rule is decisive and contradicts the model.
# `PHYSICS_FLOOR` accepts a LOWER confidence where the swap rule is decisive and agrees.
# One raises precision by giving up coverage, the other buys coverage by trusting an
# agreement; they are opposite bets on the same signal, which is why they are one
# effective-threshold vector in `explain` rather than two independent overrides.
#
# **The two opinions are not independent [measured, 2026-08-03].** S2 reads `ileg_swaps`,
# `ileg_minhalf` and `ileg_minquarter` off the same 1 Hz-filtered interleg angle the swap
# rule reads, so they are wrong TOGETHER exactly where that signal is ambiguous. Removing
# the interleg block from S2 to restore the separation was tried and does not help - the
# correlation is in the signal, not the feature list (`caveats.md` §1.1c).
#
# **That objection is threshold-conditional, and the sweep says so precisely** [measured,
# 2026-08-04, leave-one-rev-out over 1,244,292 rows, `roweval.physics_gate_table`]. Each arm
# is matched against the threshold-only policy AT THE SAME COVERAGE and compared on
# surviving errors, because a policy that changes coverage cannot be judged on accuracy at a
# fixed threshold. Negative is better:
#
#     threshold  committed errors  physics objects to  ceiling    floor
#       0.50           55,357       16,003 (28.9%)     -1,958    +8,309
#       0.60           38,215        8,255 (21.6%)     -1,986    +3,218
#       0.70           24,916        3,470 (13.9%)     -1,092        +0
#       0.75           19,865        2,033 (10.2%)       -465      -466
#       0.80           14,676        1,003  (6.8%)       -422    -1,480
#       0.85           10,391          580  (5.6%)        -11    -1,290
#       0.90            5,727          202  (3.5%)        -26      -628
#       0.95            1,650            0  (0.0%)         +0    +1,014
#       0.98              390            0  (0.0%)         +0    +1,550
#
# Read the third column first: it is the ceiling's entire addressable set, and **at p >= 0.95
# it is exactly zero.** The 2026-08-03 retraction is confirmed, not overturned - at the
# high-precision operating points a ceiling has literally nothing to catch, and the `+0` in
# its column is the gate doing nothing rather than the gate breaking even.
#
# **RETRACTED at the shipped point [2026-08-04]: this block used to read -147 for the ceiling
# at 0.85 and conclude that BOTH gates beat the threshold there.** That was the 42-feature
# fit. The table above is the corrected 38-feature one the champion actually ships, and it
# puts the ceiling at -11 - noise. Its addressable set barely moved (647 -> 580); what
# changed is that it now converts almost none of it, and it is worth something only at 0.70
# and below, where nobody operates. The FLOOR's case survived: -1,290 at 0.85, still
# reversing sign at 0.95. The wrong reading is recorded here rather than deleted because it
# is the second ceiling argument a measurement has retracted, and for the same reason both
# times - S2 and the swap rule read the same 1 Hz-filtered interleg angle, so how much the
# gate appears to buy is mostly a fact about the feature set, not about the physics.
#
# **They still ship OFF, and the reason is not the measurement.** Every number in
# `OPERATING_POINTS.md` - the whole argument for 0.85, including the per-subject floor that
# picked it - is a property of these flags being False. Switching one on silently would make
# that document describe a policy nobody runs, which is precisely the failure that deleted S4
# and left the amplitude band shipping ON in one stage and OFF in another for weeks. Turning
# either on is a deliberate act: flip it, re-run `s2_roweval`, and re-derive
# `OPERATING_POINTS.md` from the new curve. What has changed is that the decision now costs a
# measurement to check rather than an argument to win.
PHYSICS_CEILING = False
PHYSICS_FLOOR = False

# The reduced bar a row must clear when the swap rule independently agrees with the model.
# Only read when `PHYSICS_FLOOR` is True. Not tuned - it is the `balanced` preset's own
# threshold, so the floor's claim is precisely "an agreeing physics verdict is worth the
# difference between the strict operating point and the balanced one", which is a statement
# a reader can evaluate rather than a number pulled from a sweep.
PHYSICS_FLOOR_THRESHOLD = 0.70

# Every row whose interleg swing sits in the annotation-ambiguity band abstains, whatever
# the model's probability.
#
# This was the S4 fusion stage's policy, ported here and then measured. Until 2026-08-04
# the two disagreed about it: S4 gave up ~30 points of coverage on the argument that the
# annotations inside the band cannot support a confident call, while this module — the
# thing that actually labels a customer's CSV — kept calling them. **Two policies for one
# question, and the one nobody validated was the one that shipped.** That is most of why
# the stage was deleted. The band itself is an S2 quantity (measured in
# `train.reference_stats` from the annotation and `ileg_minhalf`, no model output and no
# physics), so it survives here, where it can be switched and scored against the unit that
# ships.
#
# The justification is not "these guesses are wrong" — it is that inside the band there is
# no fact of the matter to be right about. The same interleg amplitude is annotated `walk`
# for 38% of one trial's in-band windows and 100% of another's, and that per-trial fraction
# rank-correlates with the trial's confident-error rate at -0.66 (p=0.004, n=17). A
# confident call there inherits whichever convention the training trials happened to
# favour; it is not knowledge.
#
# **DEFAULT OFF, and the measurement is why** [measured, 2026-08-04, leave-one-rev-out over
# 1,271,256 rows]. Turning it on was tried and the band is DOMINATED by simply raising the
# threshold — at matched coverage the threshold-only policy keeps strictly fewer errors:
#
#     policy                          coverage  selective acc  worst subject  errors kept
#     threshold 0.95, no band           0.6732         0.9983         0.9904        1,413
#     threshold 0.70, band abstains     0.6365         0.9956         0.9909        3,477
#     threshold 0.85, band abstains     0.6204         0.9968         0.9908        2,433
#
# Two points of coverage MORE and a third of the errors, at the same worst subject. The
# band's best case was cross-subject consistency — it is the trials, not the rows, that
# disagree about this range — and it does not win there either: worst subject lands within
# 0.0005 of the threshold-only policy at every comparable point.
#
# The defence, stated because it is real and because it is also why this cannot be settled
# by measuring harder: **inside the band the labels are the untrustworthy thing**, so
# "errors" scored there are partly annotation noise rather than model error, and the
# comparison above scores the band against exactly the labels it exists to distrust. That
# makes the band a JUDGEMENT (§1.5), not a measured improvement — and an unmeasurable
# judgement that costs 22 points of coverage does not get to be the default. `reason_report`
# says the same thing from the other side: the guesses it suppresses would have been right
# 0.9329 of the time, the weakest of any reason in the file.
#
# It stays implemented and switchable rather than deleted, because the argument against it
# is a measurement and not a proof: re-measure it after any retrain that moves the band,
# and after any change to the annotations, since the whole case rests on how inconsistent
# those are. `label_audit.py` is what tracks that — 8 of 41 trials currently annotate the
# band unlike the corpus.
#
# Set True to abstain on the whole band, and re-measure `OPERATING_POINTS.md` if you do:
# every number in it is a property of this flag.
BAND_ABSTAINS = False

# Reasons a row is ambiguous, most specific first. Ordered, so a row near a boundary is
# reported as a boundary rather than as whatever else also happens to be true there.
REASONS = {
    "uncovered": (
        "no full window covers this row (segment shorter than the window, or a gap)",
        "unknown - not enough measured signal to judge"),
    "out_of_distribution": (
        "feature values fall outside the range seen anywhere in training",
        "unmodelled state, or a channel mapping error - check the input columns"),
    "near_transition": (
        "a predicted state change falls within one window of this row",
        "the other class - the boundary is real, its exact timing is not resolvable"),
    "physics_contradicts": (
        "the swap rule is decisive here and says the opposite of the model "
        "(`PHYSICS_CEILING`)",
        "either class - two opinions that read the same interleg signal disagree, so "
        "neither is independent evidence for the other"),
    "weight_shift_or_step": (
        "interleg signal crosses once in the window: the swap rule's own AMBIGUOUS verdict",
        "stand (a weight shift) or walk (a single step) - one leg passing the other is both"),
    "posture_shift": (
        "both thighs sit well above this recording's own standing posture: not upright",
        "a state outside {stand, walk}: sitting, sit-to-stand, or stand-to-sit"),
    "low_excursion_gait": (
        "called walking, but the legs swing less than walking normally does",
        "stand (postural sway) or very slow shuffling gait"),
    "amplitude_ambiguous": (
        "the legs swing in the range where human annotators call both stand and walk, so "
        "no amplitude rule separates it and neither does a model reading amplitude",
        "either class - the ground truth itself is not consistent in this range"),
    "model_split": (
        "the ensemble is genuinely divided with no single diagnostic cause",
        "the other class"),
}


@lru_cache(maxsize=4)
def load_champion(model_dir: Path):
    """(model, meta). Both are required: the meta carries the window spec the model was
    fitted at and the reference statistics the reasons are stated against, so loading one
    without the other is how train/serve skew starts.

    Cached because a corpus sweep labels dozens of files in one process and unpickling 400
    trees each time dominates its runtime. Keyed on the directory, so pointing at a
    different champion still loads a different model.
    """
    import joblib

    meta = json.loads((model_dir / "model_meta.json").read_text(encoding="utf-8"))
    model = joblib.load(model_dir / "champion.joblib")
    return model, meta


def read_input(path: Path) -> pd.DataFrame:
    """Read an `lpf_view` CSV and resolve the required columns BY NAME (§1.3 - never by
    position)."""
    # index_col=False: resolving BY NAME does not protect against a trailing comma. Pandas
    # promotes column 0 to the index and the names stay put, so `Time` would hold the next
    # column's data under the right label.
    df = pd.read_csv(path, encoding="utf-8-sig", index_col=False)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in (TIME_COL, *FEATURES) if c not in df.columns]
    if missing:
        raise SystemExit(
            f"{path.name}: reads as family 'lpf_view' but is missing {missing}.\n"
            f"That path needs {TIME_COL} plus all of {list(FEATURES)}."
        )
    return df


def read_source(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(the caller's own rows, the `lpf_view` frame to score, provenance).

    Dispatch is on FAMILY, resolved from the header by name (§1.3) through the same census
    S1 uses — never on "does this file happen to have the columns I want". The two families
    share only `Time`, so the marker column is the honest question and duck-typing the
    columns would misroute a half-written file instead of rejecting it.

    A raw device log goes through the bridge first. That call is where every provenance
    guard fires — unmapped revision, contradicted axis, non-deg/s gyro, unusable final
    timestamp — and each one raises rather than returning a frame, so a refusal can never
    be mistaken for a labelled file.
    """
    family = family_of([strip_prefix(c) for c in read_header(path)])

    if family == LPF_VIEW:
        df = read_input(path)
        return df, df[[TIME_COL, *FEATURES]], {
            "source": str(path), "family": family, "derived": False,
            "n_rows": int(len(df)),
        }

    if family == RAW_DEVICE:
        rows, view, provenance = raw_csv_to_features(path)
        return rows, view, {**provenance, "derived": True}

    raise SystemExit(
        f"{path.name}: unrecognized file family {family!r}.\n"
        f"This module labels a raw device log (marker column "
        f"{FAMILY_MARKERS[RAW_DEVICE]!r}) or an lpf_view file (marker "
        f"{FAMILY_MARKERS[LPF_VIEW]!r}). Neither marker is present, so nothing here can "
        f"say what the columns mean, and resolving them by position is the §1.3 hazard."
    )


def _accumulate(values: np.ndarray, starts: np.ndarray, n: int, n_rows: int
                ) -> tuple[np.ndarray, np.ndarray]:
    """(sum, count) per row over every window covering it, by difference array.

    O(windows + rows) rather than O(windows * window length), which is what makes a
    0.25 s stride over million-row recordings affordable.
    """
    acc = np.zeros(n_rows + 1)
    cnt = np.zeros(n_rows + 1)
    np.add.at(acc, starts, values)
    np.add.at(acc, starts + n, -values)
    np.add.at(cnt, starts, 1.0)
    np.add.at(cnt, starts + n, -1.0)
    return np.cumsum(acc)[:n_rows], np.cumsum(cnt)[:n_rows]


def score_frame(model, meta: dict, frame: pd.DataFrame) -> pd.DataFrame:
    """Per-row P(walk) and diagnostics on the canonical grid.

    A row's probability is the MEAN over every window containing it. That is what makes a
    boundary abstain on its own: rows near a state change are covered by windows that
    straddle it, so their probabilities pull apart and the mean lands mid-scale. No
    separate boundary rule, and no post-hoc smoothing constant to tune.
    """
    spec = WindowSpec(window_s=meta["window_s"],
                      stride_s=meta.get("inference_stride_s", 0.25),
                      fs_hz=meta.get("fs_hz", CANONICAL_HZ))
    feats = meta["features"]
    idx = [feature_names().index(f) for f in feats]
    zeros, ileg_zero, rest_trusted = rest_reference(frame, spec.fs_hz)
    walk_col = list(model.classes_).index(WALK)

    out = []
    for seg_id, seg in frame.groupby("segment", sort=True):
        seg = seg.reset_index(drop=True)
        n_rows = len(seg)
        base = pd.DataFrame({
            TIME_COL: seg[TIME_COL].to_numpy(float),
            "segment": int(seg_id),
            "p_walk": np.nan, "n_windows": 0,
            "ileg_minhalf": np.nan, "ileg_swaps": np.nan, "n_oob": np.nan,
            "posture": np.nan, "physics_swaps": np.nan, "physics_span_s": np.nan,
        })
        if n_rows >= spec.n:
            chan = {c: seg[c].to_numpy(float) for c in FEATURES}
            X, starts = segment_features(chan, spec, zeros, ileg_zero)
            if X.shape[0]:
                Xf = X[:, idx]
                pw = model.predict_proba(Xf)[:, walk_col]
                lo = np.array([meta["reference_stats"]["feat_p01"][f] for f in feats])
                hi = np.array([meta["reference_stats"]["feat_p99"][f] for f in feats])
                oob = ((Xf < lo) | (Xf > hi)).sum(1).astype(float)
                names = feature_names()
                mh = X[:, names.index("ileg_minhalf")]
                sw = X[:, names.index("ileg_swaps")]
                post = np.minimum(X[:, names.index("L_ang_med_rest")],
                                  X[:, names.index("R_ang_med_rest")])

                # S3's verdict on the same windows. Computed ALWAYS, not only when a gate
                # consumes it: `plausibility.py` has to be able to say "the physics saw no
                # alternation anywhere in this file", and a sanity check cannot run on a
                # signal the run declined to measure. Costs ~1 s on a 14-minute segment.
                #
                # `starts` is asserted equal rather than assumed. Both walk the same
                # arithmetic, but `_accumulate` below spreads these values over rows using
                # the FEATURE grid's offsets — if the two ever fell out of step, every
                # physics number would be silently attributed to the wrong rows, and
                # nothing downstream could detect it.
                sw_ad, span_ad, starts_p = segment_verdicts(chan, spec, ileg_zero)
                if not np.array_equal(starts_p, starts):
                    raise AssertionError(
                        f"physics grid != feature grid on segment {seg_id} "
                        f"({len(starts_p)} vs {len(starts)} windows). "
                        f"features.segment_features and s3_physics.serve.segment_verdicts "
                        f"must walk the same window starts.")

                s_p, cnt = _accumulate(pw, starts, spec.n, n_rows)
                covered = cnt > 0
                with np.errstate(invalid="ignore", divide="ignore"):
                    base.loc[covered, "p_walk"] = s_p[covered] / cnt[covered]
                    for col, vals in (("ileg_minhalf", mh), ("ileg_swaps", sw),
                                      ("n_oob", oob), ("posture", post),
                                      ("physics_swaps", sw_ad), ("physics_span_s", span_ad)):
                        s, _ = _accumulate(vals, starts, spec.n, n_rows)
                        base.loc[covered, col] = s[covered] / cnt[covered]
                base["n_windows"] = cnt.astype(int)
        out.append(base)

    scored = pd.concat(out, ignore_index=True)
    scored["rest_trusted"] = rest_trusted
    # The categorical form, derived once here so every consumer reads the same bands off
    # the same averaged count (`s3_physics.serve.row_verdict`).
    scored["physics"] = row_verdict(scored["physics_swaps"].to_numpy(float))
    return scored


def explain(scored: pd.DataFrame, meta: dict, threshold: float,
            spec: WindowSpec) -> pd.DataFrame:
    """Attach guess, confidence, ambiguity and reason. Reasons are ordered by specificity."""
    ref = meta["reference_stats"]
    p = scored["p_walk"].to_numpy(float)
    covered = np.isfinite(p)

    guess = np.where(p >= 0.5, WALK, STAND).astype(float)
    guess[~covered] = np.nan
    conf = np.where(covered, np.maximum(p, 1.0 - p), np.nan)

    # The physics floor and ceiling, expressed as a per-row EFFECTIVE THRESHOLD rather
    # than as two separate overrides. One vector, so a row can never be gated twice in
    # opposite directions and the whole policy reads as one sentence: how much confidence
    # this row is required to clear, given what the swap rule says about it.
    #
    # Both default OFF; see the constants for the measurement that keeps them off.
    phys = scored["physics"].to_numpy(object)
    phys_class = np.where(phys == WALKING, float(WALK),
                          np.where(phys == STANDING, float(STAND), np.nan))
    decisive = covered & np.isfinite(phys_class)
    agrees = decisive & (phys_class == guess)
    contradicts = decisive & (phys_class != guess)

    eff = np.full(len(scored), float(threshold))
    if PHYSICS_FLOOR:
        eff[agrees] = PHYSICS_FLOOR_THRESHOLD
    if PHYSICS_CEILING:
        eff[contradicts] = np.inf  # nothing clears it: a contradiction abstains outright
    ambiguous = ~covered | (conf < eff)
    # Carried as its own column for the same reason `in_band` is: the ceiling abstains at
    # EVERY threshold, so anything sweeping the threshold (`roweval.curve`) has to account
    # for it explicitly or publish a coverage this module does not deliver.
    gated = PHYSICS_CEILING & contradicts

    # The annotation-ambiguity band. `ileg_minhalf` here is the mean over the windows
    # covering the row, the same averaging the probability gets, so the test is applied to
    # the row's own evidence rather than to one window's.
    #
    # The probability is deliberately NOT consulted: an unusable row does not become usable
    # because the model feels strongly about it, and ExtraTrees returns exactly 1.0 whenever
    # every tree agrees — so a probability test would wave through the most confidently
    # wrong rows in the band.
    band = (float(ref["band_lo"]), float(ref["band_hi"])) if "band_lo" in ref else None
    in_band = np.zeros(len(scored), dtype=bool)
    if band is not None and BAND_ABSTAINS:
        mh_all = scored["ileg_minhalf"].to_numpy(float)
        with np.errstate(invalid="ignore"):
            in_band = covered & (mh_all >= band[0]) & (mh_all <= band[1])
        ambiguous = ambiguous | in_band

    # A predicted state change within one window of this row.
    #
    # `scored` is segments concatenated, so a class difference ACROSS a segment boundary
    # counts as a change here and marks the rows either side `near_transition`. That is
    # deliberate rather than overlooked: the two sides are separated by a gap, so neither
    # has a full window of measured context and softening both is the conservative call.
    # (A no-op guard `segment == segment` sat here claiming to prevent it and could not —
    # an array is always equal to itself. Removed; this comment is what it should have said.)
    g = pd.Series(guess).ffill().bfill().to_numpy()
    change = np.zeros(len(g), dtype=bool)
    if len(g) > 1:
        at = np.flatnonzero(g[1:] != g[:-1]) + 1
        half = max(1, spec.n // 2)
        for i in at:
            change[max(0, i - half):min(len(g), i + half)] = True

    swaps = scored["ileg_swaps"].to_numpy(float)
    mh = scored["ileg_minhalf"].to_numpy(float)
    oob = scored["n_oob"].to_numpy(float)

    reason = np.full(len(scored), "", dtype=object)

    def mark(mask, key):
        m = mask & ambiguous & (reason == "")
        reason[m] = key

    posture = scored["posture"].to_numpy(float)

    mark(~covered, "uncovered")
    # Posture first: "not upright at all" outranks any stand-vs-walk story told about it.
    mark(posture > ref["posture_shift_p99"], "posture_shift")
    mark(np.nan_to_num(oob, nan=0.0) >= 3, "out_of_distribution")
    mark(change, "near_transition")
    # After `near_transition` on purpose: at a state change the two opinions disagree by
    # construction — the model's probability is mid-scale because its windows straddle the
    # boundary, and the swap rule is reading a span that contains both states. Attributing
    # that to a physics contradiction would name the boundary as a conflict of evidence.
    # Before `weight_shift_or_step`, which is the swap rule's own ABSTENTION; this is its
    # decisive-and-opposed verdict, and the stronger claim goes first.
    mark(gated, "physics_contradicts")
    mark((swaps >= 0.5) & (swaps < 1.5), "weight_shift_or_step")
    mark((guess == WALK) & (mh < ref["walk_minhalf_p05"]), "low_excursion_gait")
    # After the diagnostic reasons, before the catch-all. `low_excursion_gait` overlaps the
    # bottom of the band heavily (`walk_minhalf_p05` sits just above `band_lo`), and it is
    # the more specific claim — it names what the row probably is. The band names why the
    # question has no answer, so it takes the rows nothing sharper explained.
    mark(in_band, "amplitude_ambiguous")
    mark(np.ones(len(scored), dtype=bool), "model_split")

    scored = scored.copy()
    scored["state"] = [CLASS_NAME.get(v) if np.isfinite(v) else None
                       for v in np.where(covered, guess, np.nan)]
    scored["label"] = np.where(covered, guess, np.nan)
    scored["confidence"] = np.round(conf, 4)
    scored["ambiguous"] = ambiguous
    # Carried as its own column, not left implicit in `reason`: the band abstains at EVERY
    # threshold, so anything sweeping the threshold (`roweval.curve`) has to subtract it
    # explicitly or it will publish a coverage this module does not deliver. `reason` cannot
    # serve that purpose — a row can be in the band and be attributed to a sharper reason.
    scored["in_band"] = in_band
    scored["physics_gated"] = gated
    scored["reason"] = reason
    scored["reason_detail"] = [REASONS[r][0] if r else "" for r in reason]
    scored["alternative"] = [REASONS[r][1] if r else "" for r in reason]
    # A recording that never rests has a fallback interleg zero, so every interleg feature
    # on it is weaker evidence. Said once, on every row, rather than silently.
    if not bool(scored["rest_trusted"].iloc[0]):
        scored["reason_detail"] = scored["reason_detail"].astype(str) + (
            " | recording never rests: interleg zero is a fallback median, not a measured "
            "standing posture")
    return scored


def label_csv(path: Path, model_dir: Path, threshold: float) -> tuple[pd.DataFrame, dict]:
    """Label one CSV; returns (the INPUT rows with the label columns appended, provenance).

    Provenance travels with the frame rather than being printed and forgotten: on the raw
    path it records which hardware revision was resolved, which two channels were actually
    read, and which trust record was consulted. A labelled file whose axis resolution
    cannot be reconstructed later is exactly the "recorded but never read" hole that let
    the wrong sagittal axis ship for the majority variant.
    """
    model, meta = load_champion(model_dir)
    spec = WindowSpec(window_s=meta["window_s"],
                      stride_s=meta.get("inference_stride_s", 0.25),
                      fs_hz=meta.get("fs_hz", CANONICAL_HZ))
    raw, view, provenance = read_source(path)

    # Same normalization the model was trained under: segment at gaps, then the canonical
    # grid. Anything else is train/serve skew.
    frame, _segments = resample_file(view, TIME_COL)
    if frame.empty:
        raise SystemExit(f"{path.name}: no usable segment survived resampling")

    # S1's label-free gate, in front of the model rather than beside it. A dead channel or
    # a file with nothing scorable would otherwise produce confident numbers built on
    # nothing, which is worse than refusing.
    h = health(frame, window_s=spec.window_s, fs_hz=spec.fs_hz)
    for w in h.warnings:
        print(f"[label] warning: {w}")
    if not h.usable:
        raise SystemExit(f"{path.name}: not usable\n  "
                         + "\n  ".join(h.errors))

    scored = explain(score_frame(model, meta, frame), meta, threshold, spec)

    # File-level sanity bounds, recorded in the provenance for the same reason the axis
    # resolution is: a doubt that is printed and forgotten is a doubt nobody can act on
    # later. Computed on `scored` — the canonical grid — rather than on the returned frame,
    # because the fractions have to be taken over the rows that were actually scored, not
    # over the caller's rows after the nearest-time mapping has duplicated some and dropped
    # others. It REPORTS: no file is refused for tripping a bound.
    provenance["plausibility"] = plausibility_check(scored)

    # Map the grid back onto the caller's own rows by nearest time.
    t_grid = scored[TIME_COL].to_numpy(float)
    t_in = raw[TIME_COL].to_numpy(float)
    order = np.argsort(t_grid)
    tg = t_grid[order]
    j = np.searchsorted(tg, t_in).clip(1, tg.size - 1)
    nearest = order[np.where(np.abs(t_in - tg[j - 1]) <= np.abs(tg[j] - t_in), j - 1, j)]
    # Beyond half a grid step from any scored sample, the row sits in a gap.
    gap = np.abs(t_in - t_grid[nearest]) > (1000.0 / spec.fs_hz)

    cols = ["state", "label", "confidence", "ambiguous", "reason", "reason_detail",
            "alternative", "n_windows"]
    picked = scored.iloc[nearest][cols].reset_index(drop=True)
    picked.loc[gap, ["state", "label", "confidence"]] = None
    picked.loc[gap, "ambiguous"] = True
    picked.loc[gap, "reason"] = "uncovered"
    picked.loc[gap, "reason_detail"] = REASONS["uncovered"][0]
    picked.loc[gap, "alternative"] = REASONS["uncovered"][1]

    # The integer-coded guess, so the full frame carries it under the same name and the
    # same vocabulary the emitted shapes use. Derived here rather than in `verdict_frame`
    # alone, so there is one definition of "what did the model think" and every output
    # option inherits it instead of recomputing it.
    picked.insert(picked.columns.get_loc("label") + 1, GUESS_COL,
                  np.where(pd.to_numeric(picked["label"], errors="coerce").isna(),
                           LABEL_UNKNOWN_MACHINE, picked["label"]).astype(int))

    # On the raw path the four derived channels are the model's actual input and appear
    # nowhere in the caller's file, so they ride along. Without them the output states a
    # verdict over columns nobody can see, and the sagittal-axis bug would have been
    # invisible in the deliverable as well as in the code.
    parts = [raw.reset_index(drop=True)]
    if provenance["derived"]:
        parts.append(view[list(FEATURES)].reset_index(drop=True))
    parts.append(picked)
    return pd.concat(parts, axis=1), provenance


# The golden corpus's own shape: every `data/labeled/rev*/csv/*.csv` is exactly these six
# columns, in this order. Emitting them means a file this pipeline labels can be read by
# anything that already reads an annotated one — including this repo's own loaders.
#
# Named REV2_COLUMNS until 2026-08-04. The shape is the `lpf_view` family's plus `Label`,
# and it is shared by all eight annotated revisions (rev2-rev13) — naming it after one of
# them read as a per-product quirk rather than the corpus-wide contract it is.
GOLDEN_COLUMNS = (TIME_COL, *FEATURES, LABEL_COL)

# The model's call on every row it managed to score, WHETHER OR NOT it committed to that
# call: STAND, WALK, or 255 where no window covered the row. `Label` overwrites the
# ambiguous ones with -1, which answers "may I use this row" and destroys "what did it
# think" — and those are different questions. A reviewer settling an abstention needs the
# guess in front of them; without it they are re-deriving an opinion the pipeline already
# formed.
#
# It rides with the verdict columns rather than with the six, and `--no-verdict` drops it
# too. The golden shape is borrowed, and a borrowed shape with one extra column is no
# longer the thing a reader asked for when they asked for it exactly.
GUESS_COL = "guess"

# Appended after those by default. `Label` alone can say a row was not called; only these
# say how sure the call was and what the doubt was about, and an abstention whose reason
# is unreadable is most of the deliverable thrown away (§OPERATING_POINTS). They go AFTER
# the six so the golden prefix is positionally intact for a reader that expects it.
VERDICT_COLUMNS = ("confidence", "ambiguous", "reason")


def verdict_frame(out: pd.DataFrame, *, strict: bool = False) -> pd.DataFrame:
    """The deliverable's own shape: the annotated corpus's columns, plus the verdict.

    Named for what it emits rather than for the format it borrows. The six golden columns
    are the *prefix*, not the point — what a reader needs from a machine-labelled file and
    never needs from a hand-annotated one is how sure the call was, and that is what the
    `VERDICT_COLUMNS` carry. `strict=True` drops them and leaves the borrowed shape alone.

    The label vocabulary is the corpus's own (`dataset.py`, `config.py`), reused rather
    than invented, so the codes mean the same thing in a file we wrote as in one HUROTICS
    annotated:

        STAND (0) / WALK (10)   a confident call
        HUMAN_UNKNOWN (-1)      scored, but under the threshold. The machine analogue of
                                the annotator's own "I looked and cannot call it" — the
                                pipeline formed an opinion and declined to commit to it.
        LABEL_UNKNOWN_MACHINE   no full window covers the row at all: a gap, or a segment
        (255)                   shorter than one window. Nothing was measured well enough
                                to judge, which is what 255 already means here (§config).

    The two abstention codes are kept apart because they are different facts and a reader
    that lumps them cannot tell "the model was unsure" from "there was no signal here".

    `guess` carries the SAME vocabulary minus the -1: the model's call on every row it
    scored, committed or not. `Label` and `guess` differ exactly on the ambiguous rows, and
    that difference is the review queue.

    By default `guess` and the three `VERDICT_COLUMNS` follow, so the same file is readable
    three ways: filter on `Label` for the corpus vocabulary, on `ambiguous` for the
    threshold, or on `guess` for what the model thought regardless. `strict=True` drops all
    four, leaving the six borrowed columns and nothing else — that shape is for a reader
    who wants a file indistinguishable from an annotated one, and a seventh column would
    defeat the only reason to ask for it.

    **`Label` is built from OUR `label` column, never from an input `Label` that rode
    along on an `lpf_view` source.** This function's whole job is to write a column under
    the name ground truth uses, so it is the one place where copying ground truth forward
    would be both easy and invisible. Constructing a new frame rather than renaming in
    place is what makes that structurally impossible (§7).
    """
    lab = pd.to_numeric(out["label"], errors="coerce").to_numpy(float)
    amb = out["ambiguous"].fillna(True).astype(bool).to_numpy()
    # `lab` is already the guess wherever the row was scored — `explain` nulls it only
    # where nothing covered the row, never for being unsure — so `Label` is `guess` with
    # one extra step, and taking that step is the whole difference between them.
    code = np.where(np.isnan(lab), LABEL_UNKNOWN_MACHINE, np.where(amb, HUMAN_UNKNOWN, lab))
    frame = pd.DataFrame({
        TIME_COL: out[TIME_COL].to_numpy(float),
        **{c: out[c].to_numpy(float) for c in FEATURES},
        LABEL_COL: code.astype(int),
    })
    if not strict:
        frame[GUESS_COL] = out[GUESS_COL].to_numpy(int)
        frame["confidence"] = pd.to_numeric(out["confidence"], errors="coerce").to_numpy(float)
        frame["ambiguous"] = amb
        # "" on a confident row, so the column reads as "the doubt, where there was any"
        # rather than carrying a filler token every reader has to learn to ignore.
        frame["reason"] = out["reason"].fillna("").astype(str).to_numpy()
    return frame


def describe_source(provenance: dict) -> str:
    """The one-block answer to "what did the model actually read?".

    Printed for every run, not only the interesting ones. The axis a raw file resolves to
    is the single assumption this path cannot verify from inside the file, so it is stated
    every time rather than left to be reconstructed from the variant id later.
    """
    if not provenance["derived"]:
        return (f"source: {provenance['family']} (already the model's input, no bridge)\n"
                f"rows in: {provenance['n_rows']:,}")
    return "\n".join([
        f"source: raw device log, variant {provenance['variant_id']}",
        f"sagittal axis: Deg_{provenance['sagittal_deg_axis']} "
        f"-> Gyro_{provenance['gyro_axis']}  (per-revision lookup, never guessed)",
        f"channels read: {', '.join(provenance['columns_read'])}",
        f"filter dt: {provenance['dt_ms']:.5f} ms "
        f"({provenance['dt_last_over_median']:.4f}x the median interval)",
        f"trust record: {provenance['trust_record']}",
        f"axis detection: L={provenance['axis_check']['L']} R={provenance['axis_check']['R']}"
        f"   gyro unit: L={provenance['gyro_unit']['L']} R={provenance['gyro_unit']['R']}",
        f"rows in: {provenance['n_rows']:,}",
    ])


def summarize(df: pd.DataFrame, threshold: float) -> str:
    n = len(df)
    amb = int(df["ambiguous"].sum())
    lines = [f"rows: {n:,}   threshold: {threshold:.2f}",
             f"labelled: {n - amb:,} ({(n - amb) / n:.1%})   "
             f"ambiguous: {amb:,} ({amb / n:.1%})"]
    conf = df.loc[~df["ambiguous"], "state"].value_counts()
    if len(conf):
        lines.append("confident states: " + ", ".join(f"{k} {v:,}" for k, v in conf.items()))
    if amb:
        lines.append("ambiguity reasons:")
        for k, v in df.loc[df["ambiguous"], "reason"].value_counts().items():
            lines.append(f"  {k:<22} {v:>8,}  ({v / amb:.1%} of ambiguous)")
    return "\n".join(lines)


def main() -> None:
    use_replacement_encoding()   # --help itself has an em-dash in it; see stages/console.py
    ap = argparse.ArgumentParser(description="Label a CSV with stand/walk + confidence.")
    ap.add_argument("csv", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--preset", default=None, help="high_coverage | balanced | high_precision")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--full", action="store_true",
                    help="write the complete labelled frame — every input column, the "
                         "derived features, and state/label/confidence/ambiguous/reason/"
                         "reason_detail/alternative/n_windows")
    ap.add_argument("--no-verdict", action="store_true",
                    help=f"drop {GUESS_COL} and {', '.join(VERDICT_COLUMNS)}, leaving only "
                         f"the {len(GOLDEN_COLUMNS)} columns the annotated corpus uses; "
                         f"abstention then survives in Label alone "
                         f"({HUMAN_UNKNOWN} unsure, {LABEL_UNKNOWN_MACHINE} uncovered)")
    args = ap.parse_args()

    meta = json.loads((args.model_dir / "model_meta.json").read_text(encoding="utf-8"))
    presets = meta["presets"]
    if args.threshold is not None:
        threshold = args.threshold
    else:
        name = args.preset or meta["default_preset"]
        if name not in presets:
            raise SystemExit(f"unknown preset {name!r}; known: {sorted(presets)}")
        threshold = presets[name]

    try:
        out, provenance = label_csv(args.csv, args.model_dir, threshold)
    except SERVE_REFUSALS as exc:
        # An abstention, not a crash. The file is unservable for a stated reason and the
        # right outcome is no output at all — a labelled file the pipeline cannot defend
        # is worse than none, because a controller would act on it.
        raise SystemExit(f"{args.csv.name}: ABSTAINED — {type(exc).__name__}\n  {exc}")

    print(describe_source(provenance))
    print()
    print(summarize(out, threshold))
    dest = args.out or args.csv.with_name(args.csv.stem + "_labelled.csv")
    emitted = out if args.full else verdict_frame(out, strict=args.no_verdict)
    emitted.to_csv(dest, index=False)
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
