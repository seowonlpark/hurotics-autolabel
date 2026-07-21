# champion/challenger machinery: run an experiment, gate it, log it
# PLAN S2's gate enforced in code: the champion only ever changes via a logged,
# metric-justified promotion. an agent proposes a declarative ExperimentSpec (never code);
# this module runs it, scores it with locoeval, applies the rule, logs every outcome to
# experiments.jsonl. rejections are the valuable half -- they stop re-proposals. see README.

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut

from stages.s2_ml.calibrate import MODES as CALIBRATION_MODES
from stages.s2_ml.calibrate import CalibrationConfig, calibrate_trials, summarize
from stages.s2_ml.dataset import load_dataset
from stages.s2_ml.features import TRANSITION, WindowSpec, build_windows, feature_columns
from stages.s2_ml.locoeval import evaluate
from stages.s2_ml.predict import DEFAULT_INFERENCE_STRIDE_S, dense_predict_trial
from stages.s2_ml.taxonomy import aggregate, bucket_errors

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_FILENAME = "experiments.jsonl"
CHAMPION_FILENAME = "champion.json"

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
    # per-file calibration mode (calibrate.MODES) or None. None reproduces the global-features
    # champion exactly; a mode recentres/rescales each file onto the corpus-global reference
    calibrate: str | None = None

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
    calibration: dict | None = None # per-file calibration summary, if spec.calibrate set

    def to_dict(self) -> dict:
        d = {"spec": self.spec.to_dict(), "macro_f1": self.macro_f1,
             "accuracy": self.accuracy, "balanced_accuracy": self.balanced_accuracy,
             "per_rev_macro_f1": self.per_rev_macro_f1,
             "n_features": self.n_features, "n_train_windows": self.n_train_windows}
        if self.calibration:
            d["calibration"] = self.calibration
        if self.taxonomy:
            d["taxonomy"] = {"row_accuracy": self.taxonomy["row_accuracy"],
                             "dominant": self.taxonomy["dominant"],
                             "fractions": self.taxonomy["fractions"]}
        return d


# current HEAD short sha, or 'unknown'
def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


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

    if spec.calibrate is not None and spec.calibrate not in CALIBRATION_MODES:
        raise ValueError(f"calibrate={spec.calibrate!r} not in {sorted(CALIBRATION_MODES)}")


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

    cal_summary = None
    if spec.calibrate:
        cfg = CalibrationConfig(mode=spec.calibrate)
        trials, gref, records = calibrate_trials(trials, cfg)
        cal_summary = summarize(spec.calibrate, gref, records)

    default = WindowSpec()
    wspec = WindowSpec(
        window_s=spec.window_s if spec.window_s else default.window_s,
        stride_s=spec.stride_s if spec.stride_s else (spec.window_s or default.stride_s),
    )
    windows = build_windows(trials, wspec)

    train_df = windows[(windows["split"] == "train") &
                       (windows["label"] != TRANSITION)].reset_index(drop=True)
    assert "lockbox" not in set(train_df["split"]), "lockbox leaked into training"

    feats = select_features(feature_columns(windows), spec.drop_features)
    X = train_df[feats].to_numpy(float)
    y = train_df["label"].to_numpy(int)
    groups = train_df["rev"].to_numpy()

    oof = np.empty_like(y)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        model = RandomForestClassifier(**spec.resolved_params())
        model.fit(X[tr], y[tr])
        oof[te] = model.predict(X[te])

    result = evaluate(y, oof, groups=groups)

    tax = None
    if taxonomy:
        per_run = []
        for rev in sorted(pd.unique(groups)):
            fit = train_df[train_df["rev"] != rev]
            model = RandomForestClassifier(**spec.resolved_params())
            model.fit(fit[feats].to_numpy(float), fit["label"].to_numpy(int))
            for tr in trials:
                if tr.split != "train" or tr.rev != rev:
                    continue
                for gt, pred, t in dense_predict_trial(model, tr.frame, feats, wspec, stride_s):
                    per_run.append(bucket_errors(gt, pred, t))
        tax = aggregate(per_run)

    return ExperimentResult(spec, result.macro_f1, result.accuracy,
                            result.balanced_accuracy, result.per_rev_macro_f1,
                            len(feats), len(train_df), tax, cal_summary)


# the current champion record, or None
def load_champion(out_dir: Path) -> dict | None:
    path = out_dir / CHAMPION_FILENAME
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


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


# every logged proposal, in order
def proposals(out_dir: Path) -> list[dict]:
    path = out_dir / PROPOSALS_FILENAME
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# every measured experiment, in order
def ledger(out_dir: Path) -> list[dict]:
    path = out_dir / LEDGER_FILENAME
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
