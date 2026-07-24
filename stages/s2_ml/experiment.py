# champion/challenger machinery: run an experiment, gate it, log it
# PLAN S2's gate enforced in code: the champion only ever changes via a logged,
# metric-justified promotion. an agent proposes a declarative ExperimentSpec (never code);
# this module runs it, scores it with locoeval, applies the rule, logs every outcome to
# experiments.jsonl. rejections are the valuable half -- they stop re-proposals. see README.

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from runmeta import git_sha
from stages.s2_ml.dataset import load_dataset
from stages.s2_ml.features import TRANSITION, WindowSpec, build_windows, feature_columns
from stages.s2_ml.locoeval import evaluate
from stages.s2_ml.predict import DEFAULT_INFERENCE_STRIDE_S, dense_predict_trial
from stages.s2_ml.taxonomy import aggregate, bucket_errors

LEDGER_FILENAME = "experiments.jsonl"
CHAMPION_FILENAME = "champion.json"

# the champion's spec, tracked in git -- unlike champion.json, which lives under gitignored
# runs/ and is absent from a fresh clone. this is the single source the clean-room seed
# reproduces from; record() rewrites it on every promotion, so it can never silently drift
# from the champion it re-seeds (the old hand-synced CHAMPION_SPEC dict in run_pipeline.py).
CHAMPION_SPEC_PATH = Path(__file__).resolve().parent / "champion_spec.json"

BASE_MODEL_PARAMS = dict(n_estimators=300, random_state=0, n_jobs=-1,
                         class_weight="balanced")

# a challenger must clear the champion by this much on macro-F1; a margin not ">", because
# LORO over a handful of revs is noisy and a +0.001 win would ratchet on noise
PROMOTION_MARGIN = 0.005


# a declarative, replayable description of one challenger
@dataclass
class ExperimentSpec:
    name: str # unique spec name
    rationale: str # why this should help, in one line
    drop_features: list[str] = field(default_factory=list) # features to remove
    window_s: float | None = None # None => champion/default window
    # stride is INDEPENDENT of window length, and that independence is load-bearing:
    # tying them means changing window_s also changes the training-set size, confounding
    # "longer window" with "less data". None keeps the champion's stride
    stride_s: float | None = None
    model_params: dict = field(default_factory=dict) # overrides on BASE_MODEL_PARAMS

    # base params with this spec's overrides applied
    def resolved_params(self) -> dict:
        return {**BASE_MODEL_PARAMS, **self.model_params}

    def to_dict(self) -> dict:
        return asdict(self)


# scored outcome of one experiment
@dataclass
class ExperimentResult:
    spec: ExperimentSpec # the spec that produced it
    macro_f1: float # headline metric
    accuracy: float
    balanced_accuracy: float
    per_rev_macro_f1: dict # per-held-out-rev macro-F1
    n_features: int # features after drops
    n_train_windows: int # training windows used
    taxonomy: dict | None = None # row-level error taxonomy, if run

    def to_dict(self) -> dict:
        d = {"spec": self.spec.to_dict(), "macro_f1": self.macro_f1,
             "accuracy": self.accuracy, "balanced_accuracy": self.balanced_accuracy,
             "per_rev_macro_f1": self.per_rev_macro_f1,
             "n_features": self.n_features, "n_train_windows": self.n_train_windows}
        if self.taxonomy:
            d["taxonomy"] = {"row_accuracy": self.taxonomy["row_accuracy"],
                             "dominant": self.taxonomy["dominant"],
                             "fractions": self.taxonomy["fractions"]}
        return d


# the hyperparameters a proposal may touch, with bounds -- a whitelist, so a stray
# agent-authored key can't reach the estimator. random_state/n_jobs deliberately absent:
# reproducibility and machine resources are the pipeline's call, not a proposal's
ALLOWED_MODEL_PARAMS = {
    "n_estimators": (10, 2000),
    "max_depth": (1, 100),
    "min_samples_leaf": (1, 100),
    "min_samples_split": (2, 100),
    "max_features": None, # categorical: "sqrt" | "log2" | float | int
    "criterion": None, # categorical: "gini" | "entropy" | "log_loss"
    "class_weight": None, # categorical: "balanced" | "balanced_subsample" | None
}
WINDOW_S_RANGE = (0.5, 10.0)


# reject a proposal that steps outside the vocabulary (raises ValueError); runs before
# any training so a bad proposal costs nothing
def validate_spec(spec: ExperimentSpec) -> None:
    if not spec.name or not spec.rationale:
        raise ValueError("a spec needs both a name and a rationale")

    for key, value in spec.model_params.items():
        if key not in ALLOWED_MODEL_PARAMS:
            raise ValueError(
                f"model_params key {key!r} is not permitted; "
                f"allowed: {sorted(ALLOWED_MODEL_PARAMS)}"
            )
        bounds = ALLOWED_MODEL_PARAMS[key]
        if bounds and value is not None:
            lo, hi = bounds
            if not isinstance(value, (int, float)) or not (lo <= value <= hi):
                raise ValueError(f"model_params[{key!r}]={value!r} outside [{lo}, {hi}]")

    if spec.window_s is not None:
        lo, hi = WINDOW_S_RANGE
        if not (lo <= spec.window_s <= hi):
            raise ValueError(f"window_s={spec.window_s} outside [{lo}, {hi}] s")


# feature set after drops; unknown names are an error, not a silent no-op (a typo'd drop
# would otherwise 'pass' while changing nothing)
def select_features(all_feats: list[str], drop: list[str]) -> list[str]:
    unknown = [d for d in drop if d not in all_feats]
    if unknown:
        raise ValueError(f"drop_features names no such feature: {unknown}")
    keep = [f for f in all_feats if f not in drop]
    if not keep:
        raise ValueError("drop_features would remove every feature")
    return keep


# train + score one spec under leave-one-rev-out; the lockbox is never touched
def run_experiment(spec: ExperimentSpec, trials=None, *, taxonomy: bool = False,
                   stride_s: float = DEFAULT_INFERENCE_STRIDE_S) -> ExperimentResult:
    validate_spec(spec)
    trials = trials if trials is not None else load_dataset()
    wspec, params, _ = champion_config(spec)
    windows = build_windows(trials, wspec)

    train_df = windows[(windows["split"] == "train") &
                       (windows["label"] != TRANSITION)].reset_index(drop=True)
    assert "lockbox" not in set(train_df["split"]), "lockbox leaked into training"

    feats = select_features(feature_columns(windows), spec.drop_features)
    X = train_df[feats].to_numpy(float)
    y = train_df["label"].to_numpy(int)
    groups = train_df["rev"].to_numpy()

    # one leave-one-rev-out pass: the fold that holds a rev out is the same model that scores
    # that rev's windows (OOF) and, when asked, its rows (taxonomy) -- so train it once and use
    # it for both, rather than refitting the identical LORO forests a second time. deterministic
    # (fixed random_state), and the oof array is filled by mask, so fold order does not matter.
    oof = np.empty_like(y)
    per_run: list = []
    for rev in sorted(pd.unique(groups)):
        te = groups == rev
        model = RandomForestClassifier(**params)
        model.fit(X[~te], y[~te])
        oof[te] = model.predict(X[te])
        if not taxonomy:
            continue
        for tr in trials:
            if tr.split != "train" or tr.rev != rev:
                continue
            for gt, pred, t in dense_predict_trial(model, tr.frame, feats, wspec, stride_s):
                per_run.append(bucket_errors(gt, pred, t))

    result = evaluate(y, oof, groups=groups)
    tax = aggregate(per_run) if taxonomy else None

    return ExperimentResult(spec, result.macro_f1, result.accuracy,
                            result.balanced_accuracy, result.per_rev_macro_f1,
                            len(feats), len(train_df), tax)


# the current champion record, or None
def load_champion(out_dir: Path) -> dict | None:
    path = out_dir / CHAMPION_FILENAME
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# the tracked champion spec as an ExperimentSpec, for the clean-room re-seed. tolerant of an
# older/newer schema (unknown keys dropped) via _spec_from_dict
def load_champion_spec(path: Path = CHAMPION_SPEC_PATH) -> ExperimentSpec:
    return _spec_from_dict(json.loads(path.read_text(encoding="utf-8")))


# rewrite the tracked champion spec; called on every promotion so the git-tracked seed always
# matches the promoted champion.json (which is not in git). commit it alongside the promotion.
def save_champion_spec(spec: dict, path: Path = CHAMPION_SPEC_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")


# a spec dict -> ExperimentSpec, tolerant of an older/newer schema (unknown keys dropped)
def _spec_from_dict(data: dict) -> ExperimentSpec:
    valid = {f.name for f in fields(ExperimentSpec)}
    return ExperimentSpec(**{k: v for k, v in data.items() if k in valid})


# the CURRENT champion's spec, read from the runtime champion.json (which a within-session
# promotion may have advanced past the git-tracked seed). None when no champion exists yet.
# this is the source the serve path (oof / lockbox / s3 grid / export) must reconstruct from,
# so the deployed model is the champion in full -- not just its dropped features.
def champion_spec_from_json(out_dir: Path) -> ExperimentSpec | None:
    champ = load_champion(out_dir)
    return _spec_from_dict(champ["spec"]) if champ else None


# resolve a spec to the concrete (WindowSpec, model_params, drop_features) it is fit with --
# the SAME resolution train.main uses, so every serve-path reconstruction (oof, lockbox, the
# S3 join grid, export's row mapping) matches train.py's persisted champion exactly. a default
# champion (window_s/stride_s null, model_params {}) resolves to the plain WindowSpec() default,
# so nothing changes until a non-default window/params champion is actually promoted.
def champion_config(spec: ExperimentSpec) -> tuple[WindowSpec, dict, list[str]]:
    default = WindowSpec()
    win = spec.window_s or default.window_s
    stride = spec.stride_s or spec.window_s or default.stride_s
    return (WindowSpec(window_s=win, stride_s=stride),
            spec.resolved_params(), list(spec.drop_features))


# resolve the CURRENT champion (runtime champion.json) to the (WindowSpec, params, drops) the
# serve path fits with -- the single entry point oof / lockbox / s3 grid / export share, so they
# can never disagree on how a champion is rebuilt. window_override forces the sampling grid (a
# rare, explicit --window-s) while keeping the champion's params/drops. require=True raises when
# no champion exists (the deployment fits, which cannot proceed without one); require=False falls
# back to the plain defaults (the model-free stages that can still run standalone).
def resolve_champion(out_dir: Path, *, window_override: WindowSpec | None = None,
                     require: bool = True) -> tuple[WindowSpec, dict, list[str]]:
    csp = champion_spec_from_json(out_dir)
    if csp is None:
        if require:
            raise FileNotFoundError(
                f"no {CHAMPION_FILENAME} in {out_dir}; establish a champion first "
                f"(python -m stages.s2_ml.train --out {out_dir.name}, or run_pipeline.py).")
        return window_override or WindowSpec(), dict(BASE_MODEL_PARAMS), []
    wspec, params, drops = champion_config(csp)
    return window_override or wspec, params, drops


# secondary criterion, applied ONLY on a macro-F1 tie: at equal accuracy prefer lower
# steady_confusion -- a sustained wrong call becomes a sustained wrong ACTION on a powered
# device, whereas swallowed/omission are fail-passive. wrong action beats no action
STEADY_CONFUSION_MARGIN = 0.02


# steady_confusion share of a result or champion record, or None if no taxonomy
def _steady(result_or_champion) -> float | None:
    tax = (result_or_champion.taxonomy if isinstance(result_or_champion, ExperimentResult)
           else result_or_champion.get("taxonomy"))
    if not tax:
        return None
    return tax["fractions"]["steady_confusion"]


# the promotion rule -- objective, margin-based, the ONLY path to champion. primary is
# macro-F1 past PROMOTION_MARGIN; on a tie, lower steady_confusion wins. returns
# (promote, reason); the reason is logged either way to stop re-proposals
def decide(challenger: ExperimentResult, champion: dict | None) -> tuple[bool, str]:
    if champion is None:
        return True, "no incumbent champion; establishing baseline"

    delta = challenger.macro_f1 - champion["macro_f1"]
    if delta >= PROMOTION_MARGIN:
        return True, (f"macro-F1 {challenger.macro_f1:.4f} beats champion "
                      f"{champion['macro_f1']:.4f} by {delta:+.4f} >= {PROMOTION_MARGIN}")

    # tie on the headline metric -> fall through to the error-type preference
    if abs(delta) < PROMOTION_MARGIN:
        new, old = _steady(challenger), _steady(champion)
        if new is not None and old is not None:
            drop = old - new
            if drop >= STEADY_CONFUSION_MARGIN:
                return True, (
                    f"macro-F1 {challenger.macro_f1:.4f} ties champion "
                    f"{champion['macro_f1']:.4f} ({delta:+.4f}), but steady_confusion "
                    f"falls {old:.3f} -> {new:.3f} ({drop:.3f} >= "
                    f"{STEADY_CONFUSION_MARGIN}); at equal accuracy, prefer the model "
                    f"that fails passively")

    return False, (f"macro-F1 {challenger.macro_f1:.4f} vs champion "
                   f"{champion['macro_f1']:.4f} ({delta:+.4f}); "
                   f"below the {PROMOTION_MARGIN} promotion margin")


# append to the ledger; update champion.json only on promotion
def record(out_dir: Path, result: ExperimentResult, promoted: bool, reason: str,
           critic: dict | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "host": platform.node(),
        **result.to_dict(),
        "promoted": promoted,
        "decision_reason": reason,
    }
    if critic:
        entry["critic"] = critic
    with (out_dir / LEDGER_FILENAME).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    if promoted:
        # carry taxonomy so decide() can apply the error-type tiebreaker next time;
        # without it the secondary criterion silently never fires
        keys = ("ts", "git_sha", "spec", "macro_f1", "accuracy", "balanced_accuracy",
                "per_rev_macro_f1", "n_features", "taxonomy")
        (out_dir / CHAMPION_FILENAME).write_text(
            json.dumps({k: entry[k] for k in keys if k in entry}, indent=2),
            encoding="utf-8")
        # keep the git-tracked seed in lockstep with the champion it reproduces, automatically
        save_champion_spec(entry["spec"])
    return entry


PROPOSALS_FILENAME = "proposals.jsonl"


# log every proposal and its fate, including ones the critic stopped; kept separate from
# experiments.jsonl (measured runs) so the next cycle can still see an idea was refused
def record_proposal(out_dir: Path, proposal: dict, critic: dict, ran: bool,
                    note: str = "") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "proposal": proposal,
        "critic": critic,
        "ran": ran,
        "note": note,
    }
    with (out_dir / PROPOSALS_FILENAME).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


# parse a JSONL file into a list of records; empty list when the file is absent, blank lines
# skipped. the one reader for both the proposals log and the experiment ledger
def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# every logged proposal, in order
def proposals(out_dir: Path) -> list[dict]:
    return _read_jsonl(out_dir / PROPOSALS_FILENAME)


# every measured experiment, in order
def ledger(out_dir: Path) -> list[dict]:
    return _read_jsonl(out_dir / LEDGER_FILENAME)
