"""Champion/challenger machinery: run an experiment, gate it, log it.

PLAN S2's gate: *the champion only ever changes via a logged, metric-justified
promotion.* That is enforced here, in code — not in an agent's judgement, and not in a
human's memory of what was tried.

Division of labour (non-negotiable, PLAN principle 1):
  - An agent proposes an `ExperimentSpec` — a **declarative** change drawn from a fixed
    vocabulary (features to drop, window length, model hyperparameters). It never writes
    code, never touches data, never runs training.
  - This module runs it, scores it with locoeval, and applies the promotion rule.
  - Every outcome lands in `experiments.jsonl`, promoted or not. Rejections are the more
    valuable half of the record: they are what stops the same idea being re-proposed.

Why a declarative spec rather than agent-authored code: a spec is reviewable before it
runs, reproducible after, and cannot do anything the vocabulary does not allow. It also
makes "revert" trivial — re-running a logged spec reproduces the model exactly.
"""

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

# A challenger must clear the champion by this much on the headline metric. A margin,
# not ">", because leave-one-rev-out over a handful of revs is noisy: promoting on a
# +0.001 difference would ratchet the champion on noise and call it progress.
PROMOTION_MARGIN = 0.005


@dataclass
class ExperimentSpec:
    """A declarative, replayable description of one challenger."""

    name: str
    rationale: str                                  # why this should help, in one line
    drop_features: list[str] = field(default_factory=list)
    window_s: float | None = None                   # None => champion/default window
    model_params: dict = field(default_factory=dict)  # overrides on BASE_MODEL_PARAMS

    def resolved_params(self) -> dict:
        return {**BASE_MODEL_PARAMS, **self.model_params}

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExperimentResult:
    spec: ExperimentSpec
    macro_f1: float
    accuracy: float
    balanced_accuracy: float
    per_rev_macro_f1: dict
    n_features: int
    n_train_windows: int
    taxonomy: dict | None = None

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


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def select_features(all_feats: list[str], drop: list[str]) -> list[str]:
    """Feature set after drops. Unknown names are an error, not a silent no-op —
    a typo'd drop would otherwise 'pass' while changing nothing."""
    unknown = [d for d in drop if d not in all_feats]
    if unknown:
        raise ValueError(f"drop_features names no such feature: {unknown}")
    keep = [f for f in all_feats if f not in drop]
    if not keep:
        raise ValueError("drop_features would remove every feature")
    return keep


def run_experiment(spec: ExperimentSpec, trials=None, *, taxonomy: bool = False,
                   stride_s: float = DEFAULT_INFERENCE_STRIDE_S) -> ExperimentResult:
    """Train + score one spec under leave-one-rev-out. The lockbox is never touched."""
    trials = trials if trials is not None else load_dataset()
    wspec = WindowSpec(window_s=spec.window_s, stride_s=spec.window_s) if spec.window_s else WindowSpec()
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
                            len(feats), len(train_df), tax)


def load_champion(out_dir: Path) -> dict | None:
    path = out_dir / CHAMPION_FILENAME
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def decide(challenger: ExperimentResult, champion: dict | None) -> tuple[bool, str]:
    """The promotion rule. Objective, margin-based, and the ONLY path to champion.

    Returns (promote, reason). The reason is recorded either way — a rejection with its
    number is what stops the same proposal coming back.
    """
    if champion is None:
        return True, "no incumbent champion; establishing baseline"
    delta = challenger.macro_f1 - champion["macro_f1"]
    if delta >= PROMOTION_MARGIN:
        return True, (f"macro-F1 {challenger.macro_f1:.4f} beats champion "
                      f"{champion['macro_f1']:.4f} by {delta:+.4f} >= {PROMOTION_MARGIN}")
    return False, (f"macro-F1 {challenger.macro_f1:.4f} vs champion "
                   f"{champion['macro_f1']:.4f} ({delta:+.4f}); "
                   f"below the {PROMOTION_MARGIN} promotion margin")


def record(out_dir: Path, result: ExperimentResult, promoted: bool, reason: str,
           critic: dict | None = None) -> dict:
    """Append to the ledger; update champion.json only on promotion."""
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
        (out_dir / CHAMPION_FILENAME).write_text(
            json.dumps({k: entry[k] for k in
                        ("ts", "git_sha", "spec", "macro_f1", "accuracy",
                         "balanced_accuracy", "per_rev_macro_f1", "n_features")},
                       indent=2), encoding="utf-8")
    return entry


def ledger(out_dir: Path) -> list[dict]:
    path = out_dir / LEDGER_FILENAME
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
