# file-level sanity bounds on the OUTPUT; python -m stages.s3_physics.plausibility

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from freshness import stamp_inputs
from stages.console import use_replacement_encoding
from stages.s2_ml.dataset import FEATURES, LABEL_COL, STAND, WALK, load_dataset
from stages.s2_ml.features import WindowSpec, rest_reference
from stages.s3_physics.serve import STANDING, WALKING, row_verdict, segment_verdicts

from runslayout import REGEN

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---- the bounds; each a JUDGEMENT sited outside a measured range, not a tuned threshold ----

# the model commits to less than this share; corpus min 0.1487, synthetic faults land at ~0.0065
COMMITMENT_MIN = 0.02

# physics finds alternation in at most this share; every trial with annotated walking is >= 0.4174
PHYSICS_SILENT_MAX = 0.02

# ...while the model commits at least this much to walking; only the contradiction is diagnostic
MODEL_WALK_MIN = 0.20

# the mirror; "over half the file alternates" is the only non-arbitrary line on a fraction
PHYSICS_LOUD_MIN = 0.50

# ...while the model commits almost none to walking; not zero- 4% of a walking file is as broken
MODEL_WALK_MAX = 0.05

# below this many covered rows the fractions above are too noisy to act on; 30 s at 100 Hz
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


# the file-level quantities the bounds read, from `label.explain`'s output frame
def file_stats(scored: pd.DataFrame) -> dict:
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
        # over COMMITTED rows: an abstention asserts nothing, and scoring it as "not walk" misleads
        "model_walk_frac": round(float((state[committed] == "walk").mean()), 4)
        if committed.any() else float("nan"),
        "physics_walk_frac": round(float((phys == WALKING).mean()), 4),
        "physics_stand_frac": round(float((phys == STANDING).mean()), 4),
        "rest_trusted": bool(scored["rest_trusted"].iloc[0]),
    }


# every bound this file trips
def check(scored: pd.DataFrame) -> list[dict]:
    s = file_stats(scored)
    out = []

    # evaluated for every file, even one too short for the fraction bounds: it has no sample size
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


# the `label_report.md` section
def summarize(findings_by_file: dict[str, list[dict]], n_files: int = 0) -> list[str]:
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
                  f"can fire at all: `python -m stages.s3_physics.plausibility`.",
                  ""]
    else:
        lines += [f"**{len({f for f, _ in hard})} file(s) tripped a hard bound.**", ""]
        # the measured numbers ride on each file's line; the bound and its remedy are stated once
        for code, group in _by_code(hard).items():
            lines += [f"### {code} — {len(group)} file(s)", "", BOUNDS[code][0], ""]
            lines += [f"- `{f}` — {x['detail']}" for f, x in group]
            lines += ["", f"**Remedy:** {BOUNDS[code][1]}", ""]
    # named not counted, grouped by code; over DISTINCT files, since one file can trip several
    for code, group in _by_code(soft).items():
        files = list(dict.fromkeys(f for f, _ in group))
        lines += [f"### {code} — {len(files)} file(s)", "", BOUNDS[code][0], ""]
        lines += [f"- `{f}`" for f in files]
        lines += ["", f"**Remedy:** {BOUNDS[code][1]}", ""]
    return lines


# {bound code: the (file, finding) pairs that tripped it}, in first-seen order
def _by_code(flagged: list[tuple[str, dict]]) -> dict[str, list[tuple[str, dict]]]:
    out: dict[str, list[tuple[str, dict]]] = {}
    for f, x in flagged:
        out.setdefault(x["code"], []).append((f, x))
    return out


# ---- calibration and controls: the numbers in this module's header, regenerated on demand ----

# per-trial physics walk fraction against the ANNOTATED walk fraction
def corpus_calibration(spec: WindowSpec | None = None) -> pd.DataFrame:
    spec = spec or WindowSpec(stride_s=0.25)
    rows = []
    for t in load_dataset(include_excluded=True):
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


# synthetic faults, injected because a hand-annotated corpus has none; `None` is the control
CONTROLS = {
    "unmodified": None,
    "duplicated_leg": lambda f, L, R: f.__setitem__(R, f[L].to_numpy()),
    "dead_sensor": lambda f, L, R: f.__setitem__(R, float(f[R].median())),
    "swapped_legs": lambda f, L, R: _swap(f, L, R),
}


def _swap(f: pd.DataFrame, L: str, R: str) -> None:
    a, b = f[L].to_numpy().copy(), f[R].to_numpy().copy()
    f[L], f[R] = b, a


# inject each synthetic fault into one clean recording; report what each bound says
def run_controls(rev: str = "rev13", trial: int = 1) -> pd.DataFrame:
    from stages.s2_ml.label import DEFAULT_MODEL_DIR, explain, load_champion, score_frame

    model, meta = load_champion(DEFAULT_MODEL_DIR)
    spec = WindowSpec(window_s=meta["window_s"],
                      stride_s=meta.get("inference_stride_s", 0.25),
                      fs_hz=meta.get("fs_hz", 100.0))
    src = {(t.rev, t.trial): t for t in load_dataset(include_excluded=True)}[(rev, trial)]
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
    # run as a module this REGENERATES the numbers- a bound never fired at a fault is no evidence
    ap = argparse.ArgumentParser(
        description="Re-site the file-level sanity bounds: calibrate against the corpus and "
                    "fire them at injected faults. (Serving uses this module as a library.)")
    ap.add_argument("--out", type=Path, default=REGEN / "s3_physics")
    args = ap.parse_args()

    out_dir = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict = {"commitment_min": COMMITMENT_MIN,
                     "physics_silent_max": PHYSICS_SILENT_MAX,
                     "model_walk_min": MODEL_WALK_MIN,
                     "physics_loud_min": PHYSICS_LOUD_MIN,
                     "model_walk_max": MODEL_WALK_MAX, "min_rows": MIN_ROWS}

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

    # the controls score fault injections through the FITTED champion, so a refit moves this file
    from stages.s2_ml.label import DEFAULT_MODEL_DIR  # here, not at module scope- label.py imports us
    # `stage=`, because two other audits share runs/regen/s3_physics and a bare stamp names neither
    stamp_inputs(out_dir, {"champion": DEFAULT_MODEL_DIR / "champion.joblib",
                           "model_meta": DEFAULT_MODEL_DIR / "model_meta.json"},
                 stage="plausibility")

    print(f"[plaus] -> {out_dir / 'plausibility.json'}")


if __name__ == "__main__":
    main()
