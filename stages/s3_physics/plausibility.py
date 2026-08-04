"""File-level sanity bounds: is this file's OUTPUT defensible as a whole?

    python -m stages.s3_physics.plausibility --calibrate    # the corpus numbers below
    python -m stages.s3_physics.plausibility --control      # the synthetic-fault controls

`label_all.py` refuses a file for INPUT reasons — an unmapped hardware revision, a measured
permutation contradicting the axis about to be read, a gyro that is not natively deg/s, a
missing trust record. Once a file passes those gates **nothing checks the output.** A
recording whose channels are miswired passes every input gate, produces confident labels,
and is wrong end to end with no artifact saying so.

**File level, not window level, and that distinction is the entire justification.** A
per-window physics gate on the classifier is what the deleted S4 fusion stage was, and it
is measured not to work: S2 reads `ileg_swaps` / `ileg_minhalf` / `ileg_minquarter` off the
same 1 Hz-filtered interleg angle the swap rule reads, so physics contradicts only ~12% of
S2's high-confidence errors and **none** at p >= 0.95 (`anchors.py`, `caveats.md` §1.1c).
That argument is about which WINDOW is right. It says nothing about "this recording commits
to nothing" or "the physics and the model describe different recordings" — statements about
the file, which correspond to diagnosable faults rather than hard windows.

## What actually catches faults [measured, 2026-08-04]

Three synthetic channel faults injected into rev13 t1 (a 99%-walk recording that scores
clean), each run through the real champion and the real serve path:

    fault                       model walk   physics walk   committed   caught by
    (unmodified)                    0.9935         0.9919      0.9981   - plausible -
    R := L   duplicated leg         0.0000         0.0000      0.0064   commitment_collapse
    R := const  dead sensor         0.0000         0.9894      0.0066   commitment_collapse
                                                                        + physics_loud
    L <-> R  swapped legs           0.9935         0.9919      0.9981   NOTHING

**`commitment_collapse` is the workhorse**, and that was not the expected answer. Both
destructive faults are caught because the model stops committing — 0.0064 and 0.0066 of
rows, against a corpus minimum of **0.1487** over the 78 real raw files in
`labeled_raw/label_summary.csv`. The bound sits at 0.02: ~7x below anything real, ~3x above
both faults.

**The swapped-leg fault is invisible, and that is arithmetic rather than an oversight.**
Swapping the legs negates `L - R`, and `swap_count` thresholds at +/-delta symmetrically,
so a negated interleg signal yields the identical swap count. Neither the swap rule nor
this module can see it, and no bound here should be read as covering it. `caveats.md` §3.4
keeps that blindness listed; it is not fixed here.

## What the corpus says about the physics bounds [measured, 2026-08-04, 43 trials]

Per-trial physics walk fraction against the ANNOTATED walk fraction — model-free on both
sides, so this calibration owes nothing to the classifier:

    trials with >5% annotated walk (n=40)   min 0.4174   p05 0.5355   median 0.8315
    trials with <5% annotated walk (n=1)    rev13 t4, physics 0.9429
    corr(annotated, physics) = 0.6264

**`physics_silent` and `physics_loud` have never fired on real data**, and the honest
reading of rev13 t4 says why they are unlikely to. That trial looks like the perfect
`physics_loud` case — 0.00 annotated walk against 0.9429 physics walk — but at serve time
the bound compares physics against the MODEL, and the model calls it 0.9916 walk. Model and
physics AGREE; the annotation is the outlier. A serve-path check has no annotation to
compare against, so it cannot catch that trial, and `label_audit` is what does. The two
stages divide the work rather than overlapping: **`label_audit` distrusts the labels,
`plausibility` distrusts the file.**

So these two bounds are guardrails justified by margin and by one synthetic control, not by
a catch on real data. Stated plainly rather than left for a reader to assume otherwise
(§11.1): a bound that has never rejected anything is not evidence that everything passed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stages.console import use_replacement_encoding
from stages.s2_ml.dataset import FEATURES, LABEL_COL, STAND, WALK, load_dataset
from stages.s2_ml.features import WindowSpec, rest_reference
from stages.s3_physics.serve import STANDING, WALKING, row_verdict, segment_verdicts

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# The bounds. Each is a JUDGEMENT sited outside a measured range, not a threshold tuned
# against a score — there is no score to tune against, because the faults being caught are
# absent from the corpus by construction: a recording with miswired channels never reached
# the annotation stage. Where a number has a measurement behind it, the measurement is
# named; where it is a stated line on a fraction, it says so.
# ---------------------------------------------------------------------------

# The model commits to less than this share of the file. Corpus minimum is 0.1487 over 78
# real raw files; both synthetic destructive faults land at ~0.0065. This is the bound with
# actual demonstrated catching power.
COMMITMENT_MIN = 0.02

# Physics finds leg alternation in at most this share of the file. Every corpus trial
# containing annotated walking sits at or above 0.4174, so this is ~20x clear of the
# observed floor. Set it higher and it starts firing on genuinely sedentary recordings,
# which are normal data and not a fault.
PHYSICS_SILENT_MAX = 0.02

# ...while the model commits at least this much of the file to walking. Both halves are
# required: physics finding no alternation is unremarkable on its own (someone stood still),
# and only the contradiction with a committed model is diagnostic.
MODEL_WALK_MIN = 0.20

# The mirror. "Over half the file alternates" — the only non-arbitrary line available on a
# fraction. The dead-sensor control lands at 0.9894.
PHYSICS_LOUD_MIN = 0.50

# ...while the model commits almost none of it to walking. Not zero: a model calling 4% of
# a walking recording `walk` is as broken as one calling none of it.
MODEL_WALK_MAX = 0.05

# Below this many covered rows the fractions above are too noisy to act on. 30 s at 100 Hz.
MIN_ROWS = 3000

BOUNDS = {
    "commitment_collapse": (
        "the model commits to almost none of this file",
        "a channel fault that destroys the signal the model reads — one leg duplicated "
        "onto both, or a dead/stuck sensor. Check `provenance` for the two channels "
        "actually read and the resolved axis. Both synthetic destructive faults land here."),
    "physics_silent": (
        "the model reports walking across the file, but the swap rule finds leg "
        "alternation almost nowhere in it",
        "a channel fault that destroys alternation while leaving the per-side features "
        "intact enough for the model to commit. Check the axis map and both channels."),
    "physics_loud": (
        "the swap rule finds sustained leg alternation across the file, but the model "
        "commits almost none of it to walking",
        "the recording is outside the model's distribution, or a channel it reads is "
        "degenerate while the interleg angle still swings — the dead-sensor signature."),
    "rest_untrusted": (
        "the recording never rests, so the interleg zero is a whole-recording median "
        "rather than a measured standing posture (§10.2)",
        "not a fault on its own. It makes every physics verdict in this file SOFT, so "
        "read any bound above as weaker evidence, not as a clean contradiction."),
    "too_short": (
        f"fewer than {MIN_ROWS:,} covered rows, so the file-level fractions are noise",
        "no action — the bounds are simply not evaluated for this file."),
}


def file_stats(scored: pd.DataFrame) -> dict:
    """The file-level quantities the bounds read, from `label.explain`'s output frame.

    Everything is taken over COVERED rows. An uncovered row has neither a probability nor a
    physics verdict, so including it would move every fraction toward zero at a rate set by
    how gappy the recording is — a statement about coverage, not about agreement.
    """
    covered = scored["n_windows"].to_numpy(int) > 0
    n = int(covered.sum())
    if not n:
        return {"n_rows": int(len(scored)), "n_covered": 0, "coverage": 0.0}

    phys = scored["physics"].to_numpy(object)[covered]
    committed = ~scored["ambiguous"].to_numpy(bool)[covered]
    state = scored["state"].to_numpy(object)[covered]

    return {
        "n_rows": int(len(scored)),
        "n_covered": n,
        "coverage": round(n / len(scored), 4),
        "n_committed": int(committed.sum()),
        "committed_frac": round(float(committed.mean()), 4),
        # Over COMMITTED rows: the question is what the model actually asserts about this
        # file, and an abstention asserts nothing. Scoring abstentions as "not walk" would
        # make a high-threshold run look like a model contradicting the physics when it has
        # merely declined to answer.
        "model_walk_frac": round(float((state[committed] == "walk").mean()), 4)
        if committed.any() else float("nan"),
        "physics_walk_frac": round(float((phys == WALKING).mean()), 4),
        "physics_stand_frac": round(float((phys == STANDING).mean()), 4),
        "rest_trusted": bool(scored["rest_trusted"].iloc[0]),
    }


def check(scored: pd.DataFrame) -> list[dict]:
    """Every bound this file trips. Returns [] for a plausible file.

    **It reports; it never refuses.** `label_all` already has a refusal path for inputs it
    cannot defend, and a file tripping a bound here is not undefendable — it is suspicious,
    and the operator needs the labels plus the doubt, not an empty output. Same call
    `label.py` makes row by row: state the verdict, name the doubt.

    The physics bounds are NOT suppressed when commitment has collapsed, even though
    `model_walk_frac` then rests on very few rows. Both findings are real and they say
    different things, so both are reported and the base size travels in the detail — the
    dead-sensor control trips both, and a reader shown only one of them would misdiagnose
    it. Suppressing a true finding to keep the output tidy is how a check earns distrust.
    """
    s = file_stats(scored)
    out = []

    # `rest_untrusted` is evaluated for every file, INCLUDING one too short for the
    # fraction bounds. It is not a statistic and has no sample size to be too small: the
    # recording either contained a rest span or it did not. Suppressing it with the
    # fraction bounds would lose the fact on exactly the short files where a fallback zero
    # does the most damage — 300 rows carry no second chance to find rest.
    if not s.get("rest_trusted", True):
        out.append(_finding("rest_untrusted", "soft", s,
                            "interleg zero is a whole-recording median"))

    if s["n_covered"] < MIN_ROWS:
        return out + [_finding("too_short", "info", s,
                               f"{s['n_covered']:,} covered rows < {MIN_ROWS:,}")]

    cf, mw, pw = s["committed_frac"], s["model_walk_frac"], s["physics_walk_frac"]
    if cf < COMMITMENT_MIN:
        out.append(_finding(
            "commitment_collapse", "implausible", s,
            f"committed on {cf:.2%} of covered rows ({s['n_committed']:,} of "
            f"{s['n_covered']:,}), below the {COMMITMENT_MIN:.0%} bound; the lowest real "
            f"file in the corpus commits 14.87%"))
    if np.isfinite(mw) and mw >= MODEL_WALK_MIN and pw <= PHYSICS_SILENT_MAX:
        out.append(_finding(
            "physics_silent", "implausible", s,
            f"model calls {mw:.1%} of its {s['n_committed']:,} committed rows walk; "
            f"physics finds alternation in {pw:.1%} of the file "
            f"(bound {PHYSICS_SILENT_MAX:.0%})"))
    if np.isfinite(mw) and pw >= PHYSICS_LOUD_MIN and mw <= MODEL_WALK_MAX:
        out.append(_finding(
            "physics_loud", "implausible", s,
            f"physics finds alternation in {pw:.1%} of the file; model calls only "
            f"{mw:.1%} of its {s['n_committed']:,} committed rows walk "
            f"(bound {MODEL_WALK_MAX:.0%})"))
    return out


def _finding(code: str, severity: str, stats: dict, detail: str) -> dict:
    what, remedy = BOUNDS[code]
    return {"code": code, "severity": severity, "what": what, "detail": detail,
            "remedy": remedy, "stats": stats}


def summarize(findings_by_file: dict[str, list[dict]], n_files: int = 0) -> list[str]:
    """The `label_report.md` section. One block per bound, worst first, or a line saying clean.

    `n_files` is the number of files SWEPT, passed in rather than inferred from the dict:
    the dict holds only files with findings, so counting it would report "no file tripped a
    bound across 0 files" on a clean run — a sentence that reads like nothing was checked.

    **Grouped by bound, not by file.** A bound means the same thing on every file it fires
    on, so its explanation and its remedy are printed once and the affected recordings are
    listed underneath. Per-file rows repeating one sentence verbatim bury the only part that
    varies — which CSVs — and make a bound that fired twice look twice as complicated as one
    that fired once. What is genuinely per-file, the measured numbers in `detail`, stays on
    the file's own line.
    """
    flat = [(f, x) for f, xs in findings_by_file.items() for x in xs]
    hard = [(f, x) for f, x in flat if x["severity"] == "implausible"]
    soft = [(f, x) for f, x in flat if x["severity"] == "soft"]

    lines = ["## Plausibility — does each file's output hold together?", "",
             "File-level bounds from `s3_physics/plausibility.py`. These report; they never "
             "refuse. A flagged file still has its labels — it has them *and* a stated "
             "doubt.", ""]
    if not hard:
        lines += [f"**No file tripped a hard bound** across {n_files} labelled file(s). "
                  f"A bound that never fires is not evidence that everything passed "
                  f"(§11.1) — the synthetic controls in `plausibility.py` are what show it "
                  f"can fire at all: `python -m stages.s3_physics.plausibility --control`.",
                  ""]
    else:
        lines += [f"**{len({f for f, _ in hard})} file(s) tripped a hard bound.**", ""]
        # The measured numbers differ per file, so they ride on each file's line; the bound
        # and its remedy do not, so they are stated once above and below the list.
        for code, group in _by_code(hard).items():
            lines += [f"### {code} — {len(group)} file(s)", "", BOUNDS[code][0], ""]
            lines += [f"- `{f}` — {x['detail']}" for f, x in group]
            lines += ["", f"**Remedy:** {BOUNDS[code][1]}", ""]
    # Named, not merely counted. A soft bound is the one an operator can still act on — by
    # re-recording the file so it begins at rest — and a bare count sends them to
    # `plausibility.jsonl` to find out which files it meant. Grouped by code rather than
    # hardcoding `rest_untrusted`, so a second soft bound cannot be silently swallowed by a
    # sentence naming the first. Counted over DISTINCT files, since one file can trip
    # several codes and `len(soft)` counts findings. No `detail` on these lines: a soft
    # bound's detail restates its `what` rather than measuring anything.
    for code, group in _by_code(soft).items():
        files = list(dict.fromkeys(f for f, _ in group))
        lines += [f"### {code} — {len(files)} file(s)", "", BOUNDS[code][0], ""]
        lines += [f"- `{f}`" for f in files]
        lines += ["", f"**Remedy:** {BOUNDS[code][1]}", ""]
    return lines


def _by_code(flagged: list[tuple[str, dict]]) -> dict[str, list[tuple[str, dict]]]:
    """{bound code: the (file, finding) pairs that tripped it}, in first-seen order."""
    out: dict[str, list[tuple[str, dict]]] = {}
    for f, x in flagged:
        out.setdefault(x["code"], []).append((f, x))
    return out


# ---------------------------------------------------------------------------
# Calibration and controls: the numbers in this module's docstring, regenerated on demand.
# ---------------------------------------------------------------------------

def corpus_calibration(spec: WindowSpec | None = None) -> pd.DataFrame:
    """Per-trial physics walk fraction against the ANNOTATED walk fraction.

    Model-free on both sides on purpose. Calibrating against model output would make "does
    the physics agree with the model" a question whose reference answer came from the
    model — the circularity `label_audit` refuses for the same reason.

    Loads with `excluded=set()`: the most informative trial is the excluded one, and a
    calibration that cannot see its own best evidence measures nothing.
    """
    spec = spec or WindowSpec(stride_s=0.25)
    rows = []
    for t in load_dataset(excluded=set()):
        frame = t.frame.reset_index(drop=True)
        _z, ileg, trusted = rest_reference(frame, spec.fs_hz)
        means, covs, labs = [], [], []
        for _sid, seg in frame.groupby("segment", sort=True):
            seg = seg.reset_index(drop=True)
            if len(seg) < spec.n:
                continue
            chan = {c: seg[c].to_numpy(float) for c in FEATURES}
            sw, _sp, starts = segment_verdicts(chan, spec, ileg)
            if not len(starts):
                continue
            acc, cnt = np.zeros(len(seg) + 1), np.zeros(len(seg) + 1)
            np.add.at(acc, starts, sw)
            np.add.at(acc, starts + spec.n, -sw)
            np.add.at(cnt, starts, 1.0)
            np.add.at(cnt, starts + spec.n, -1.0)
            acc, cnt = np.cumsum(acc)[:len(seg)], np.cumsum(cnt)[:len(seg)]
            cov = cnt > 0
            mean = np.full(len(seg), np.nan)
            mean[cov] = acc[cov] / cnt[cov]
            means.append(mean)
            covs.append(cov)
            labs.append(pd.to_numeric(seg[LABEL_COL], errors="coerce").to_numpy(float))
        if not means:
            continue
        cov, lab = np.concatenate(covs), np.concatenate(labs)
        v = row_verdict(np.concatenate(means))
        ann = np.isin(lab, [STAND, WALK]) & cov
        rows.append({
            "rev": t.rev, "trial": t.trial, "split": t.split, "n_covered": int(cov.sum()),
            "annotated_walk_frac": float((lab[ann] == WALK).mean()) if ann.any() else np.nan,
            "physics_walk_frac": float((v[cov] == WALKING).mean()),
            "rest_trusted": bool(trusted),
        })
    return pd.DataFrame(rows).sort_values("annotated_walk_frac").reset_index(drop=True)


# The synthetic faults. Injected rather than found, because a corpus of recordings that
# were annotated by hand contains no miswired files by construction — someone would have
# noticed before annotating. `None` is the unmodified control: if it ever trips a bound,
# the bound is wrong, not the file.
CONTROLS = {
    "unmodified": None,
    "duplicated_leg": lambda f, L, R: f.__setitem__(R, f[L].to_numpy()),
    "dead_sensor": lambda f, L, R: f.__setitem__(R, float(f[R].median())),
    "swapped_legs": lambda f, L, R: _swap(f, L, R),
}


def _swap(f: pd.DataFrame, L: str, R: str) -> None:
    a, b = f[L].to_numpy().copy(), f[R].to_numpy().copy()
    f[L], f[R] = b, a


def run_controls(rev: str = "rev13", trial: int = 1) -> pd.DataFrame:
    """Inject each synthetic fault into one clean recording; report what each bound says.

    Imported lazily: this is the one path in S3 that loads the champion, and a module the
    serve path imports must not pull scikit-learn in at import time.
    """
    from stages.s2_ml.label import DEFAULT_MODEL_DIR, explain, load_champion, score_frame

    model, meta = load_champion(DEFAULT_MODEL_DIR)
    spec = WindowSpec(window_s=meta["window_s"],
                      stride_s=meta.get("inference_stride_s", 0.25),
                      fs_hz=meta.get("fs_hz", 100.0))
    src = {(t.rev, t.trial): t for t in load_dataset(excluded=set())}[(rev, trial)]
    L, R = FEATURES[0], FEATURES[1]

    rows = []
    for name, fault in CONTROLS.items():
        frame = src.frame.reset_index(drop=True).copy()
        if fault is not None:
            fault(frame, L, R)
        scored = explain(score_frame(model, meta, frame), meta,
                         meta.get("threshold", 0.95), spec)
        s = file_stats(scored)
        codes = [x["code"] for x in check(scored) if x["severity"] == "implausible"]
        rows.append({"fault": name, "model_walk_frac": s["model_walk_frac"],
                     "physics_walk_frac": s["physics_walk_frac"],
                     "committed_frac": s["committed_frac"],
                     "caught_by": ", ".join(codes) if codes else "-- NOTHING --"})
    return pd.DataFrame(rows)


def main() -> None:
    use_replacement_encoding()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--calibrate", action="store_true",
                    help="recompute the corpus numbers the bounds are sited against")
    ap.add_argument("--control", action="store_true",
                    help="inject the synthetic channel faults and report what is caught")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "s3_physics")
    args = ap.parse_args()
    if not (args.calibrate or args.control):
        ap.error("nothing to do: this is a library for `label_all`. Pass --calibrate "
                 "and/or --control to regenerate the reference numbers.")

    out_dir = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict = {"commitment_min": COMMITMENT_MIN,
                     "physics_silent_max": PHYSICS_SILENT_MAX,
                     "model_walk_min": MODEL_WALK_MIN,
                     "physics_loud_min": PHYSICS_LOUD_MIN,
                     "model_walk_max": MODEL_WALK_MAX, "min_rows": MIN_ROWS}

    if args.calibrate:
        df = corpus_calibration()
        has = df[df["annotated_walk_frac"] > 0.05]
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"\n[calib] >5% annotated walk: n={len(has)}  "
              f"min physics {has['physics_walk_frac'].min():.4f}  "
              f"p05 {has['physics_walk_frac'].quantile(0.05):.4f}  "
              f"median {has['physics_walk_frac'].median():.4f}")
        print(f"[calib] PHYSICS_SILENT_MAX={PHYSICS_SILENT_MAX} sits "
              f"{has['physics_walk_frac'].min() / PHYSICS_SILENT_MAX:.0f}x below it")
        corr = df[["annotated_walk_frac", "physics_walk_frac"]].corr().iloc[0, 1]
        print(f"[calib] corr(annotated, physics) = {corr:.4f}")
        payload["corpus_min_physics_walk_with_annotated_walk"] = float(
            has["physics_walk_frac"].min())
        payload["corpus_corr_annotated_physics"] = float(corr)
        payload["trials"] = df.to_dict("records")

    if args.control:
        ctl = run_controls()
        print()
        print(ctl.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        missed = ctl[(ctl["fault"] != "unmodified")
                     & (ctl["caught_by"] == "-- NOTHING --")]["fault"].tolist()
        print(f"\n[control] uncaught faults: {', '.join(missed) if missed else 'none'}")
        if "swapped_legs" in missed:
            print("[control]   `swapped_legs` is EXPECTED here: swapping the legs negates "
                  "L-R and swap_count thresholds symmetrically, so the rule cannot see it.")
        if ctl.loc[ctl["fault"] == "unmodified", "caught_by"].iloc[0] != "-- NOTHING --":
            print("[control] WARNING: the unmodified control tripped a bound. The bound is "
                  "wrong, not the file.")
        payload["controls"] = ctl.to_dict("records")

    (out_dir / "plausibility.json").write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"[plaus] -> {out_dir / 'plausibility.json'}")


if __name__ == "__main__":
    main()
