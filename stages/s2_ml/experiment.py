# champion/challenger machinery: run, gate, log; the champion changes only by a justified promotion

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from runmeta import git_sha
from runslayout import LEDGER_FILENAME, PROPOSALS_FILENAME, REGEN, keep_dir_for
from stages.s2_ml.dataset import (
    FEATURES,
    # this corpus carries no optional channel, so the selectable set IS the required one
    SELECTABLE_FEATURES,
    load_dataset,
)
from stages.s2_ml.features import WindowSpec, build_windows, feature_columns
from stages.s2_ml.locoeval import evaluate
# estimator and window filter come from train.py; restating them here is the wrong that still scores
from stages.s2_ml.train import MODEL_PARAMS, build_model, trainable

# rewritten whole on every promotion, so it regenerates- the ledger beside it does NOT
CHAMPION_FILENAME = "champion.json"

# the champion's spec, tracked in git; record() rewrites it on every promotion so it cannot drift
CHAMPION_SPEC_PATH = Path(__file__).resolve().parent / "champion_spec.json"

# the champion's own hyperparameters; a literal here would make every delta measure two changes
BASE_MODEL_PARAMS = MODEL_PARAMS

# a margin, not ">": LORO over a handful of revs is noisy and a +0.001 win would ratchet on noise
PROMOTION_MARGIN = 0.005


# a declarative, replayable description of one challenger
@dataclass
class ExperimentSpec:
    name: str # unique spec name
    rationale: str # why this should help, in one line
    # features to remove; None => the CHAMPION's drops, [] => drop nothing (defaulting to [] is a trap)
    drop_features: list[str] | None = None
    window_s: float | None = None # None => champion/default window
    # stride is INDEPENDENT of window length, or "longer window" confounds with "less data"
    stride_s: float | None = None
    model_params: dict = field(default_factory=dict) # overrides on BASE_MODEL_PARAMS
    # INPUT channels, coarser than drop_features: adding one changes which trials are even loadable
    channels: list[str] = field(default_factory=list)

    # base params with this spec's overrides applied
    def resolved_params(self) -> dict:
        return {**BASE_MODEL_PARAMS, **self.model_params}

    # the concrete drop list: the champion's unless this spec states its own
    def resolved_drops(self) -> list[str]:
        if self.drop_features is not None:
            return list(self.drop_features)
        return list(json.loads(
            CHAMPION_SPEC_PATH.read_text(encoding="utf-8"))["drop_features"])

    def to_dict(self) -> dict:
        return asdict(self)


# WHAT a macro-F1 was measured over; two are comparable only when these match, since class count moves it
def corpus_fingerprint(train_df: pd.DataFrame) -> dict:
    return {
        "n_windows": int(len(train_df)),
        "classes": sorted(int(c) for c in pd.unique(train_df["label"])),
        "revs": sorted(str(r) for r in pd.unique(train_df["rev"])),
    }


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
    corpus: dict | None = None # what it was measured OVER; see corpus_fingerprint

    def to_dict(self) -> dict:
        d = {"spec": self.spec.to_dict(), "macro_f1": self.macro_f1,
             "accuracy": self.accuracy, "balanced_accuracy": self.balanced_accuracy,
             "per_rev_macro_f1": self.per_rev_macro_f1,
             "n_features": self.n_features, "n_train_windows": self.n_train_windows}
        if self.corpus:
            d["corpus"] = self.corpus
        return d


# the hyperparameters a proposal may touch; a whitelist, so a stray key cannot reach the estimator
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


# reject a proposal outside the vocabulary; runs before any training, so a bad one costs nothing
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

    if spec.channels:
        unknown = [c for c in spec.channels if c not in SELECTABLE_FEATURES]
        if unknown:
            raise ValueError(f"channels names no such input channel: {unknown}; "
                             f"available: {list(SELECTABLE_FEATURES)}")
        # a spec may ADD an optional channel, never drop a required one- that moves the corpus itself
        missing = [c for c in FEATURES if c not in spec.channels]
        if missing:
            raise ValueError(f"channels must include the required {missing}; "
                             f"use drop_features to prune what is computed from them")
        # ...nor add one: it would load, become no feature, and file under a spec naming it- SILENTLY
        if tuple(spec.channels) != FEATURES:
            raise ValueError(
                f"channels={list(spec.channels)} is not {list(FEATURES)}; fitting a different "
                f"input channel set is not implemented. Thread the channel set through "
                f"features.segment_features, features.feature_names and "
                f"features.windows_of_trial first- they read features.FEATURES directly.")


# feature set after drops; an unknown name is an error, not a typo that 'passes' while changing nothing
def select_features(all_feats: list[str], drop: list[str]) -> list[str]:
    unknown = [d for d in drop if d not in all_feats]
    if unknown:
        raise ValueError(f"drop_features names no such feature: {unknown}")
    keep = [f for f in all_feats if f not in drop]
    if not keep:
        raise ValueError("drop_features would remove every feature")
    return keep


# train + score one spec under leave-one-rev-out; the lockbox is never touched
def run_experiment(spec: ExperimentSpec, trials=None) -> ExperimentResult:
    validate_spec(spec)
    # resolve the drop list ONCE: the record must name which features a number was measured over
    spec = replace(spec, drop_features=spec.resolved_drops())
    # `validate_spec` just guaranteed FEATURES and nothing else, so there is one corpus to load
    trials = trials if trials is not None else load_dataset()
    wspec, params, _ = champion_config(spec)
    windows = build_windows(trials, wspec)

    train_df = trainable(windows, "train")
    assert "lockbox" not in set(train_df["split"]), "lockbox leaked into training"

    feats = select_features(feature_columns(windows), spec.drop_features)
    X = train_df[feats].to_numpy(float)
    y = train_df["label"].to_numpy(int)
    groups = train_df["rev"].to_numpy()

    # one leave-one-rev-out pass; oof is filled by mask, not by order
    oof = np.empty_like(y)
    for rev in sorted(pd.unique(groups)):
        te = groups == rev
        model = build_model(params)
        model.fit(X[~te], y[~te])
        oof[te] = model.predict(X[te])

    result = evaluate(y, oof, groups=groups)

    return ExperimentResult(spec, result.macro_f1, result.accuracy,
                            result.balanced_accuracy, result.per_rev_macro_f1,
                            len(feats), len(train_df), corpus_fingerprint(train_df))


# the current champion record, or None
def load_champion(out_dir: Path) -> dict | None:
    path = out_dir / CHAMPION_FILENAME
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# the tracked champion spec as an ExperimentSpec; tolerant of an older/newer schema
def load_champion_spec(path: Path = CHAMPION_SPEC_PATH) -> ExperimentSpec:
    return _spec_from_dict(json.loads(path.read_text(encoding="utf-8")))


# the promoted challenger in champion_spec.json's OWN schema, which train.py has to be able to read
def champion_spec_dict(result: ExperimentResult, supersedes: str | None = None) -> dict:
    wspec, params, drops = champion_config(result.spec)
    out = {
        "name": result.spec.name,
        "rationale": result.spec.rationale,
        "model": type(build_model()).__name__,
        "params": params,
        "window_s": wspec.window_s,
        "stride_s": wspec.stride_s,
        "fs_hz": wspec.fs_hz,
        "n_features": result.n_features,
        "drop_features": sorted(drops),
    }
    if supersedes:
        out["supersedes"] = supersedes
    return out


# rewrite the tracked spec on every promotion, so the git seed matches the ungitted champion.json
def save_champion_spec(spec: dict, path: Path = CHAMPION_SPEC_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")


# a spec dict -> ExperimentSpec, tolerant of an older/newer schema (unknown keys dropped)
def _spec_from_dict(data: dict) -> ExperimentSpec:
    valid = {f.name for f in fields(ExperimentSpec)}
    return ExperimentSpec(**{k: v for k, v in data.items() if k in valid})


# a spec -> the (WindowSpec, params, drops) train.main uses; window and stride resolve INDEPENDENTLY
def champion_config(spec: ExperimentSpec) -> tuple[WindowSpec, dict, list[str]]:
    default = WindowSpec()
    return (WindowSpec(window_s=spec.window_s or default.window_s,
                       stride_s=spec.stride_s or default.stride_s),
            spec.resolved_params(), spec.resolved_drops())


# same measurement ground? a MISSING fingerprint fails like a different one, or the guard is decorative
def comparable(new: dict | None, old: dict | None) -> tuple[bool, str]:
    if new is None or old is None:
        which = "challenger" if new is None else "incumbent"
        return False, (f"the {which} carries no corpus fingerprint, so the two cannot be shown to "
                       f"have been measured over the same data")
    diffs = []
    if new["classes"] != old["classes"]:
        diffs.append(f"class set {old['classes']} -> {new['classes']}")
    if new["revs"] != old["revs"]:
        gone = sorted(set(old["revs"]) - set(new["revs"]))
        added = sorted(set(new["revs"]) - set(old["revs"]))
        diffs.append(f"revs changed (-{len(gone)} +{len(added)}; added {added[:4]})")
    if new["n_windows"] != old["n_windows"]:
        diffs.append(f"window count {old['n_windows']:,} -> {new['n_windows']:,}")
    if diffs:
        return False, "; ".join(diffs)
    return True, ""


# the ONLY path to champion: macro-F1 past PROMOTION_MARGIN, and a tie keeps the incumbent
def decide(challenger: ExperimentResult, champion: dict | None) -> tuple[bool, str]:
    if champion is None:
        return True, "no incumbent champion; establishing baseline"

    # a delta across corpora is meaningless; fail-passive, so an unverifiable one keeps the incumbent
    ok, why = comparable(challenger.corpus, champion.get("corpus"))
    if not ok:
        return False, (f"REFUSING to compare: {why}. macro-F1 averages per-class F1, so it does not "
                       f"survive a change of class set, corpus or window count -- subtracting across "
                       f"one is how a two-class incumbent scored six-class challengers for the whole "
                       f"GaTech import. Re-baseline the champion by re-running its own spec on this "
                       f"corpus and recording it as the incumbent, then challenge that.")

    delta = challenger.macro_f1 - champion["macro_f1"]
    if delta >= PROMOTION_MARGIN:
        return True, (f"macro-F1 {challenger.macro_f1:.4f} beats champion "
                      f"{champion['macro_f1']:.4f} by {delta:+.4f} >= {PROMOTION_MARGIN}")

    return False, (f"macro-F1 {challenger.macro_f1:.4f} vs champion "
                   f"{champion['macro_f1']:.4f} ({delta:+.4f}); "
                   f"below the {PROMOTION_MARGIN} promotion margin")


# append to the ledger; update champion.json only on promotion
def record(out_dir: Path, result: ExperimentResult, promoted: bool, reason: str,
           critic: dict | None = None, update_spec: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    # read before writing: the tracked spec records the name champion.json is about to stop holding
    prior = load_champion(out_dir)
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
    keep = keep_dir_for(out_dir)
    keep.mkdir(parents=True, exist_ok=True)
    with (keep / LEDGER_FILENAME).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    if promoted:
        keys = ("ts", "git_sha", "spec", "macro_f1", "accuracy", "balanced_accuracy",
                "per_rev_macro_f1", "n_features", "corpus")
        (out_dir / CHAMPION_FILENAME).write_text(
            json.dumps({k: entry[k] for k in keys if k in entry}, indent=2),
            encoding="utf-8")
        # keep the seed in lockstep; update_spec=False for the seed run, which keeps `rejected`
        if update_spec:
            save_champion_spec(champion_spec_dict(
                result, (prior or {}).get("spec", {}).get("name")))
    return entry


# log every proposal and its fate, critic-stopped included, so the next cycle sees what was refused
def record_proposal(out_dir: Path, proposal: dict, critic: dict, ran: bool,
                    note: str = "") -> dict:
    keep = keep_dir_for(out_dir)
    keep.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "proposal": proposal,
        "critic": critic,
        "ran": ran,
        "note": note,
    }
    with (keep / PROPOSALS_FILENAME).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


# JSONL -> records, empty when absent; the one reader for both the proposals log and the ledger
def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# every logged proposal, in order
def proposals(out_dir: Path) -> list[dict]:
    return _read_jsonl(keep_dir_for(out_dir) / PROPOSALS_FILENAME)


# every measured experiment, in order
def ledger(out_dir: Path) -> list[dict]:
    return _read_jsonl(keep_dir_for(out_dir) / LEDGER_FILENAME)


# split the ledger by corpus fingerprint; no fingerprint sorts earlier- unshown ground is not ground
def ledger_by_basis(out_dir: Path,
                    champion: dict | None = None) -> tuple[list[dict], list[dict]]:
    champion = champion if champion is not None else load_champion(out_dir)
    ref = (champion or {}).get("corpus")
    current, prior = [], []
    for row in ledger(out_dir):
        ok, _why = comparable(row.get("corpus"), ref)
        (current if ok else prior).append(row)
    return current, prior


# measure the tracked champion as the incumbent: its number must come from THIS code, corpus and grid
def seed(out_dir: Path, trials=None) -> dict:
    spec = load_champion_spec()
    print(f"[s2-exp] seeding the incumbent from champion_spec.json: '{spec.name}'")
    result = run_experiment(spec, trials)
    entry = record(out_dir, result, True,
                   "seed: the tracked champion spec, re-measured as the incumbent",
                   update_spec=False)
    print(f"[s2-exp] incumbent macro-F1 {result.macro_f1:.4f} over "
          f"{result.n_train_windows:,} windows, {result.n_features} features")
    return entry


# this CLI READS- starting a cycle needs the agents, and `_s2_cycle` already seeds when it has to
def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="print the S2 champion/challenger ledger (a cycle runs from run_pipeline.py)")
    ap.add_argument("--out", default=str(REGEN / "s2_ml"))
    args = ap.parse_args()

    out_dir = (Path(__file__).resolve().parents[2] / args.out).resolve()

    champ = load_champion(out_dir)
    print(f"champion: {champ['spec']['name']} at macro-F1 {champ['macro_f1']:.4f}"
          if champ else "champion: none recorded")
    current, prior = ledger_by_basis(out_dir, champ)
    for label, rows in (("this basis", current), ("an earlier basis", prior)):
        for e in rows:
            mark = "PROMOTED" if e["promoted"] else "rejected"
            print(f"  [{label}] {e['ts'][:19]}  {e['spec']['name']:<34} "
                  f"{e['macro_f1']:.4f}  {mark}: {e['decision_reason'][:80]}")
    for p in proposals(out_dir):
        if not p["ran"]:
            print(f"  [not run] {p['ts'][:19]}  "
                  f"{p['proposal'].get('name', '(unparsed)'):<34} {p['note'][:80]}")


if __name__ == "__main__":
    main()
