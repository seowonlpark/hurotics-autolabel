# champion/challenger machinery: run an experiment, gate it, log it. the champion only ever changes
# via a logged, metric-justified promotion. an agent proposes a declarative ExperimentSpec; this runs
# it, scores with locoeval, applies the rule, and logs every outcome (rejections included) to the ledger.

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from runmeta import git_sha
from stages.s2_ml.dataset import (
    FEATURES,
    # was SELECTABLE_FEATURES: this corpus carries no optional channel, so the selectable set
    # IS the required one. See the note on it in dataset.py.
    SELECTABLE_FEATURES,
    load_dataset,
    partition_trials,
    rev_of,
)
from stages.s2_ml.features import WindowSpec, build_windows, feature_columns
from stages.s2_ml.locoeval import evaluate
from stages.s2_ml.predict import DEFAULT_INFERENCE_STRIDE_S, dense_predict_trial
from stages.s2_ml.taxonomy import aggregate, bucket_errors
# The estimator and the training-window filter come from train.py rather than being restated
# here. This file arrived from a sibling repo where the champion was a RandomForest over a
# `label != TRANSITION` filter; both are wrong here, and both are the kind of wrong that
# still produces a number. `trainable` selects POSITIVELY on the trained classes, which is
# what keeps an all-unknown window out of `to_numpy(int)`.
from stages.s2_ml.train import MODEL_PARAMS, build_model, trainable

LEDGER_FILENAME = "experiments.jsonl"
CHAMPION_FILENAME = "champion.json"

# the champion's spec, tracked in git (unlike champion.json under gitignored runs/). the single source
# the clean-room seed reproduces from; record() rewrites it on every promotion so it cannot drift.
CHAMPION_SPEC_PATH = Path(__file__).resolve().parent / "champion_spec.json"

# The champion's own hyperparameters, which train.py reads from champion_spec.json. A literal
# here (it was `n_estimators=300` on a RandomForest) would mean a challenger with no overrides
# was not the champion at all, and every delta would be measuring two changes.
BASE_MODEL_PARAMS = MODEL_PARAMS

# a challenger must clear the champion by this much on macro-F1; a margin not ">", because
# LORO over a handful of revs is noisy and a +0.001 win would ratchet on noise
PROMOTION_MARGIN = 0.005


# a declarative, replayable description of one challenger
@dataclass
class ExperimentSpec:
    name: str # unique spec name
    rationale: str # why this should help, in one line
    # features to remove. None => the CHAMPION's drops, exactly as window_s/stride_s below; an
    # empty list means drop nothing, which is a real and different proposal.
    #
    # This defaulted to [] when the file arrived, and that is a trap rather than a convention:
    # the champion here drops four features, so a hyperparameter proposal that left the field
    # alone re-added them and was scored as TWO changes at once, with the ledger recording it
    # as one. `min_leaf2_extratrees` (2026-08-04) was measured at 42 features against a
    # 38-feature champion for exactly this reason, and the +0.0016 it reported is not
    # attributable to its min_samples_leaf. The corpus fingerprint cannot catch it -- window
    # count and class set are identical, only the feature set moved.
    drop_features: list[str] | None = None
    window_s: float | None = None # None => champion/default window
    # stride is INDEPENDENT of window length: tying them would confound "longer window" with "less
    # data". None keeps the champion's stride
    stride_s: float | None = None
    model_params: dict = field(default_factory=dict) # overrides on BASE_MODEL_PARAMS
    # INPUT channels to read from each trial, before windowing. empty => the four-channel FEATURES
    # contract. This is coarser than drop_features, which prunes the WINDOW features computed from
    # these; adding a channel here changes which trials are even loadable, so it is recorded on the
    # spec and travels into the ledger with the metric it produced.
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

    # the concrete channel set this spec reads
    def resolved_channels(self) -> tuple[str, ...]:
        return tuple(self.channels) if self.channels else FEATURES

    def to_dict(self) -> dict:
        return asdict(self)


# WHAT a macro-F1 was measured over. Two macro-F1s are comparable only when these match: macro-F1
# averages per-class F1, so adding four harder classes lowers it even when nothing about the model got
# worse. `decide()` subtracted across exactly that difference for the whole GaTech import (CAVEATS 1.1),
# scoring six-class challengers against a two-class incumbent, and nothing noticed.
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
    taxonomy: dict | None = None # row-level error taxonomy, if run
    corpus: dict | None = None # what it was measured OVER; see corpus_fingerprint

    def to_dict(self) -> dict:
        d = {"spec": self.spec.to_dict(), "macro_f1": self.macro_f1,
             "accuracy": self.accuracy, "balanced_accuracy": self.balanced_accuracy,
             "per_rev_macro_f1": self.per_rev_macro_f1,
             "n_features": self.n_features, "n_train_windows": self.n_train_windows}
        if self.taxonomy:
            d["taxonomy"] = {"row_accuracy": self.taxonomy["row_accuracy"],
                             "dominant": self.taxonomy["dominant"],
                             "fractions": self.taxonomy["fractions"]}
        if self.corpus:
            d["corpus"] = self.corpus
        return d


# the hyperparameters a proposal may touch, with bounds; a whitelist, so a stray agent-authored key
# can't reach the estimator. random_state/n_jobs deliberately absent: the pipeline's call, not a proposal's.
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

    if spec.channels:
        unknown = [c for c in spec.channels if c not in SELECTABLE_FEATURES]
        if unknown:
            raise ValueError(f"channels names no such input channel: {unknown}; "
                             f"available: {list(SELECTABLE_FEATURES)}")
        # the sagittal four are the corpus contract every trial carries: a spec may ADD an optional
        # channel, never drop a required one. Pruning belongs in drop_features, which acts on the
        # window features and is scored -- dropping an input channel would instead silently change
        # which trials are loadable, moving the corpus under the metric.
        missing = [c for c in FEATURES if c not in spec.channels]
        if missing:
            raise ValueError(f"channels must include the required {missing}; "
                             f"use drop_features to prune what is computed from them")


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


# the trial set one spec reads. An optional channel is present in only part of the corpus, so asking
# for it necessarily shrinks the trial set -- and comparing a challenger on a SMALLER corpus against a
# champion measured on the full one is not a comparison at all. So this refuses to shrink silently:
# whenever the requested channels exclude trials, it names the revs it dropped.
#
# The honest sequence for an optional-channel challenger is therefore: re-baseline the champion spec
# on this same reduced set first, then gate the challenger against THAT number.
def load_for(spec: ExperimentSpec, labeled_dir=None) -> list:
    channels = spec.resolved_channels()
    kwargs = {} if labeled_dir is None else {"labeled_dir": labeled_dir}
    if channels == FEATURES:
        return load_dataset(**kwargs)

    have, lack = partition_trials(channels, **kwargs)
    if lack:
        # count by rev, and say whether a rev lost EVERY trial or only some: naming a rev that still
        # contributes 150 trials as "excluded" reads as though the subject were dropped entirely
        kept = {r: 0 for r in {rev_of(p) for p in (*have, *lack)}}
        for p in have:
            kept[rev_of(p)] += 1
        dropped: dict[str, int] = {}
        for p in lack:
            dropped[rev_of(p)] = dropped.get(rev_of(p), 0) + 1
        whole = sorted(r for r in dropped if kept[r] == 0)
        partial = {r: dropped[r] for r in sorted(dropped) if kept[r] > 0}
        print(f"[experiment] spec {spec.name!r} reads {list(channels)}; "
              f"{len(lack)} of {len(have) + len(lack)} trials lack a channel and are EXCLUDED.")
        print(f"[experiment]   revs dropped ENTIRELY: {whole or 'none'}")
        print(f"[experiment]   revs losing only some trials: {partial or 'none'}")
        print(f"[experiment]   macro-F1 is comparable only to an incumbent re-baselined on this set.")
    if not have:
        raise ValueError(f"no trial carries all of {list(channels)}")
    return load_dataset(features=channels, skip_missing=True, **kwargs)


# train + score one spec under leave-one-rev-out; the lockbox is never touched
def run_experiment(spec: ExperimentSpec, trials=None, *, taxonomy: bool = False,
                   stride_s: float = DEFAULT_INFERENCE_STRIDE_S) -> ExperimentResult:
    validate_spec(spec)
    # Resolve the drop list ONCE, here, and carry the concrete list into the result and the
    # ledger. Resolving later would leave the record saying `null` where the reader needs to
    # know which 38 features a number was measured over.
    spec = replace(spec, drop_features=spec.resolved_drops())
    trials = trials if trials is not None else load_for(spec)
    wspec, params, _ = champion_config(spec)
    windows = build_windows(trials, wspec)

    train_df = trainable(windows, "train")
    assert "lockbox" not in set(train_df["split"]), "lockbox leaked into training"

    feats = select_features(feature_columns(windows), spec.drop_features)
    X = train_df[feats].to_numpy(float)
    y = train_df["label"].to_numpy(int)
    groups = train_df["rev"].to_numpy()

    # one leave-one-rev-out pass: the fold holding a rev out is the same model that scores that rev's
    # windows (OOF) and rows (taxonomy), so train it once for both. deterministic, and the oof array is
    # filled by mask so fold order does not matter.
    oof = np.empty_like(y)
    per_run: list = []
    for rev in sorted(pd.unique(groups)):
        te = groups == rev
        model = build_model(params)
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
                            len(feats), len(train_df), tax, corpus_fingerprint(train_df))


# the current champion record, or None
def load_champion(out_dir: Path) -> dict | None:
    path = out_dir / CHAMPION_FILENAME
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# the tracked champion spec as an ExperimentSpec, for the clean-room re-seed. tolerant of an
# older/newer schema (unknown keys dropped) via _spec_from_dict
def load_champion_spec(path: Path = CHAMPION_SPEC_PATH) -> ExperimentSpec:
    return _spec_from_dict(json.loads(path.read_text(encoding="utf-8")))


# the promoted challenger in champion_spec.json's OWN schema, which is not the spec's. train.py
# reads this file to fit the champion and to check itself against it (assert_matches_spec), so a
# promotion that wrote the experiment vocabulary here would leave the next train.py run unable to
# read its own declaration -- a promotion that breaks the pipeline it just improved.
#
# `rejected` is deliberately not carried forward. It argues for the OLD champion's ablation
# against alternatives measured at the time, and a new champion has not earned those sentences.
# `supersedes` replaces it: the name this beat, with the ledger holding the margin it beat it by.
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


# rewrite the tracked champion spec; called on every promotion so the git-tracked seed always
# matches the promoted champion.json (which is not in git). commit it alongside the promotion.
def save_champion_spec(spec: dict, path: Path = CHAMPION_SPEC_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")


# a spec dict -> ExperimentSpec, tolerant of an older/newer schema (unknown keys dropped)
def _spec_from_dict(data: dict) -> ExperimentSpec:
    valid = {f.name for f in fields(ExperimentSpec)}
    return ExperimentSpec(**{k: v for k, v in data.items() if k in valid})


# the CURRENT champion's spec, from the runtime champion.json (which a within-session promotion may
# have advanced past the git-tracked seed). None when no champion exists. the source the serve path
# reconstructs the deployed model from.
def champion_spec_from_json(out_dir: Path) -> ExperimentSpec | None:
    champ = load_champion(out_dir)
    return _spec_from_dict(champ["spec"]) if champ else None


# resolve a spec to the concrete (WindowSpec, model_params, drop_features) it is fit with, the SAME
# resolution train.main uses, so every serve-path reconstruction matches train.py's champion. a default
# spec resolves to the plain WindowSpec() default.
#
# window and stride resolve INDEPENDENTLY, as ExperimentSpec.stride_s documents: falling back to
# `spec.window_s` for an unset stride made a window-only proposal change the stride too, so its
# macro-F1 moved for two reasons at once -- the "longer window" vs "less data" confound the field
# exists to prevent, and the opposite of what train.py's --window-s already does.
def champion_config(spec: ExperimentSpec) -> tuple[WindowSpec, dict, list[str]]:
    default = WindowSpec()
    return (WindowSpec(window_s=spec.window_s or default.window_s,
                       stride_s=spec.stride_s or default.stride_s),
            spec.resolved_params(), spec.resolved_drops())


# resolve the CURRENT champion to the (WindowSpec, params, drops) the serve path fits with; the single
# entry point oof / lockbox / s3 grid / export share. window_override forces the grid; require=True
# raises when no champion exists, require=False falls back to plain defaults.
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


# trials at the CURRENT champion's channels. Every path that REFITS the champion (oof, lockbox) must
# read the same inputs it was scored with; calling load_dataset() there instead would fit a model on
# the four-channel corpus, then hand it the champion's feature list, and either crash or -- worse --
# quietly produce a different model under the champion's name.
def load_for_champion(out_dir: Path, labeled_dir=None) -> list:
    csp = champion_spec_from_json(out_dir)
    if csp is None:
        return load_dataset(**({} if labeled_dir is None else {"labeled_dir": labeled_dir}))
    return load_for(csp, labeled_dir=labeled_dir)


# are two fingerprints the same measurement ground? (ok, why-not). A MISSING fingerprint is not
# treated as a pass: a record written before fingerprints existed cannot be shown comparable, and
# "cannot show" must fail the same way as "shown different" or the guard is decorative.
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


# secondary criterion, only on a macro-F1 tie: prefer lower steady_confusion, since a sustained wrong
# call becomes a sustained wrong ACTION on a powered device, whereas omissions fail passive.
STEADY_CONFUSION_MARGIN = 0.02


# steady_confusion share of a result or champion record, or None if no taxonomy
def _steady(result_or_champion) -> float | None:
    tax = (result_or_champion.taxonomy if isinstance(result_or_champion, ExperimentResult)
           else result_or_champion.get("taxonomy"))
    if not tax:
        return None
    return tax["fractions"]["steady_confusion"]


# the promotion rule, the ONLY path to champion: macro-F1 past PROMOTION_MARGIN, and on a tie lower
# steady_confusion wins. returns (promote, reason); the reason is logged either way.
def decide(challenger: ExperimentResult, champion: dict | None) -> tuple[bool, str]:
    if champion is None:
        return True, "no incumbent champion; establishing baseline"

    # a delta is meaningless across corpora, so refuse to compute one rather than subtract anyway.
    # fail-passive: an unverifiable comparison keeps the incumbent, never promotes on it.
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
           critic: dict | None = None, update_spec: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    # read before writing: the tracked spec records the name it superseded, and champion.json is
    # about to stop being the record of what that was
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
    with (out_dir / LEDGER_FILENAME).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    if promoted:
        # carry taxonomy so decide() can apply the error-type tiebreaker next time;
        # without it the secondary criterion silently never fires
        keys = ("ts", "git_sha", "spec", "macro_f1", "accuracy", "balanced_accuracy",
                "per_rev_macro_f1", "n_features", "taxonomy", "corpus")
        (out_dir / CHAMPION_FILENAME).write_text(
            json.dumps({k: entry[k] for k in keys if k in entry}, indent=2),
            encoding="utf-8")
        # keep the git-tracked seed in lockstep with the champion it reproduces, automatically.
        # update_spec=False is for the seed run, which re-measures the CURRENT champion: rewriting
        # the declaration it just reproduced would silently discard the `rejected` prose that is
        # the only record of what that champion was chosen against.
        if update_spec:
            save_champion_spec(champion_spec_dict(
                result, (prior or {}).get("spec", {}).get("name")))
    return entry


PROPOSALS_FILENAME = "proposals.jsonl"


# log every proposal and its fate (critic-stopped ones included), kept separate from experiments.jsonl
# so the next cycle can still see an idea was refused
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


# split the ledger into (this basis, an earlier one) by corpus fingerprint.
#
# this repo inherited sixteen entries measured on a RandomForest over 16-23 features. they are real
# measurements and worth keeping -- an idea that lost is still an idea that lost -- but they are not
# evidence about the current 38-feature ExtraTrees champion, and handing an agent one undifferentiated
# history invites exactly the error champion_spec.json already had to write a paragraph about: the
# zeroing-family drop that THIS basis measures as a regression is a promotion in those rows.
#
# the same fingerprint that stops decide() subtracting across bases is what separates them here, so the
# two cannot drift apart. a row with no fingerprint sorts to the earlier basis: an entry whose
# measurement ground cannot be established is not one to reason from as if it could.
def ledger_by_basis(out_dir: Path,
                    champion: dict | None = None) -> tuple[list[dict], list[dict]]:
    champion = champion if champion is not None else load_champion(out_dir)
    ref = (champion or {}).get("corpus")
    current, prior = [], []
    for row in ledger(out_dir):
        ok, _why = comparable(row.get("corpus"), ref)
        (current if ok else prior).append(row)
    return current, prior


# measure the tracked champion and record it as the incumbent.
#
# not a formality. the incumbent's number has to have been produced by THIS code on THIS corpus at
# THIS window grid, or the first challenger is refused as incomparable -- correctly, and uselessly.
# quoting model_meta.json's macro-F1 instead would seed the gate with a number carrying no corpus
# fingerprint, which comparable() treats as unshowable rather than as a pass.
def seed(out_dir: Path, trials=None, *, taxonomy: bool = True) -> dict:
    spec = load_champion_spec()
    print(f"[s2-exp] seeding the incumbent from champion_spec.json: '{spec.name}'")
    result = run_experiment(spec, trials, taxonomy=taxonomy)
    entry = record(out_dir, result, True,
                   "seed: the tracked champion spec, re-measured as the incumbent",
                   update_spec=False)
    print(f"[s2-exp] incumbent macro-F1 {result.macro_f1:.4f} over "
          f"{result.n_train_windows:,} windows, {result.n_features} features"
          + (f", dominant error '{result.taxonomy['dominant']}'" if result.taxonomy else ""))
    return entry


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="S2 champion/challenger ledger")
    ap.add_argument("--out", default="runs/s2_ml")
    ap.add_argument("--seed", action="store_true",
                    help="re-measure the tracked champion and record it as the incumbent")
    ap.add_argument("--no-taxonomy", action="store_true",
                    help="skip the row-level error taxonomy (faster; loses the tiebreaker)")
    ap.add_argument("--show", action="store_true", help="print the ledger and exit")
    args = ap.parse_args()

    out_dir = (Path(__file__).resolve().parents[2] / args.out).resolve()

    if args.show:
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
        return

    if args.seed:
        seed(out_dir, taxonomy=not args.no_taxonomy)
        return

    ap.error("nothing to do: pass --seed or --show. a cycle runs from run_pipeline.py, "
             "which needs the agents.")


if __name__ == "__main__":
    main()
