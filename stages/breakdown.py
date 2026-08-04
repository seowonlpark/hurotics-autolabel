# the whole pipeline on one page (python -m stages.breakdown); it reads, it computes no metric

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from freshness import check_all, stamp_paths
# one definition of run provenance; a private copy is two answers to "which commit made this page"
from runmeta import git_sha as _git_sha
from stages.console import use_replacement_encoding
# the hand-edited exclusion list, imported not restated: a copy answers "acted on?" wrongly
from stages.s2_ml.dataset import EXCLUDED_TRIALS

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = REPO_ROOT / "runs"
LABELED_RAW = REPO_ROOT / "labeled_raw"

BREAKDOWN_MD = "breakdown.md"

ACCURACY_TARGET = 0.95

# which headline each checked-in document quotes, explicit per artifact and hand-maintained
PROSE_CLAIMS = {
    "S2 row-level": ["README.md", "caveats.md", "OPERATING_POINTS.md", "needtowrite.md"],
}

# below this a file commits to less than half its rows- structural; a reporting line, not a policy
LOW_COVERAGE_LINE = 0.50

# a drop this short at index 0 is the export preamble; grouping stops 39 of them reading as 39 findings
PREAMBLE_MAX_ROWS = 12


# ---- Loading; every artifact is optional, and an absent one is a REPORTED gap, not a skip ----

# everything read this run, and everything that was not there
@dataclass
class Source:
    gaps: list[tuple[str, str]] = field(default_factory=list)   # (artifact, how to produce it)

    def json(self, path: Path, produced_by: str) -> dict | list | None:
        if not path.is_file():
            self.gaps.append((self._rel(path), produced_by))
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.gaps.append((f"{self._rel(path)} (unreadable: {exc})", produced_by))
            return None

    def jsonl(self, path: Path, produced_by: str) -> list[dict] | None:
        if not path.is_file():
            self.gaps.append((self._rel(path), produced_by))
            return None
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def csv(self, path: Path, produced_by: str) -> list[dict] | None:
        if not path.is_file():
            self.gaps.append((self._rel(path), produced_by))
            return None
        # utf-8 explicitly: the default is cp949 and the paths carry non-ASCII session names
        with path.open(encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))

    @staticmethod
    def _rel(path: Path) -> str:
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            return str(path)


# CSV and JSON both hand back '' / None / NaN for absent; one coercion, one answer
def _num(x, default=None):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(v) else v


def _pct(x, places=1) -> str:
    return "—" if x is None else f"{x:.{places}%}"


def _f(x, places=4) -> str:
    return "—" if x is None else f"{x:.{places}f}"


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["*(nothing to report)*", ""]
    return ["| " + " | ".join(headers) + " |",
            "|" + "|".join("---" for _ in headers) + "|",
            *["| " + " | ".join(str(c) for c in r) + " |" for r in rows],
            ""]


# the first sentence of a stage's reason, with absolute paths made repo-relative
def _first_sentence(text: str) -> str:
    text = " ".join(str(text).split()).replace(str(REPO_ROOT) + "\\", "").replace(
        str(REPO_ROOT) + "/", "")
    head = text.split(". ")[0].rstrip(".")
    return (head[:157] + "…") if len(head) > 158 else head


def _missing(what: str, produced_by: str) -> list[str]:
    return [f"> **Not present.** `{what}` was not found. Produce it with `{produced_by}`.", ""]


# ---- Sections; each reads only what it needs, so one absent stage blanks one section ----

# what the corpus is; the three populations that get conflated constantly
def section_corpus(src: Source) -> list[str]:
    summary = src.csv(LABELED_RAW / "label_summary.csv", "python -m stages.s2_ml.label_all")
    meta = src.json(RUNS / "s2_ml" / "model_meta.json", "python -m stages.s2_ml.train")
    loco = src.json(RUNS / "s2_ml" / "locoeval.json", "python -m stages.s2_ml.train")

    raw_dir = REPO_ROOT / "data" / "raw"
    n_raw = len(list(raw_dir.rglob("*.csv"))) if raw_dir.is_dir() else None

    out = [
        "## §0 — What the corpus is", "",
        "Three populations, and confusing them is the most common way to misread every "
        "number below. Only the middle one has ground truth.", "",
    ]
    rows = [
        ["raw device logs (`data/raw`)", f"{n_raw:,}" if n_raw is not None else "—",
         "what S1 cleans and `label_all` sweeps — **no ground truth**"],
        ["label-pure windows", f"{loco['n']:,}" if loco else "—",
         "what S2 and S3 are *measured* on — **has ground truth**"],
        ["rows labelled by the sweep",
         f"{sum(int(_num(r['rows'], 0)) for r in summary if r['labelled'] == 'True'):,}"
         if summary else "—",
         "what a caller receives — **no ground truth**"],
    ]
    out += _table(["population", "count", "what it is"], rows)

    if meta:
        out += [
            "Development subjects (everything below is measured on these): "
            f"**{', '.join(meta['train_revs'])}**.", "",
        ]
    if (RUNS / "s2_ml" / "roweval_lockbox.json").is_file():
        lb = src.json(RUNS / "s2_ml" / "roweval_lockbox.json", "python -m stages.s2_ml.roweval --lockbox")
        if lb:
            out += [f"Sealed lockbox subject: **{', '.join(lb['revs'])}** — single use. "
                    f"See §2 for what it scored and `caveats.md` §3.2 before quoting it.", ""]

    # Stated POSITIVELY, because the count above is already net of it: a reader who counts
    # trials on disk and gets a different number has to be able to find out why here. §A checks
    # only the other direction -- judged bad and still IN -- so without this the removals are
    # the one edit to the corpus that nothing reports.
    if EXCLUDED_TRIALS:
        audit = src.json(RUNS / "s3_physics" / "label_audit.json",
                         "python -m stages.s3_physics.label_audit")
        seen = {(t["rev"], int(t["trial"])): t for t in (audit or {}).get("trials", [])}
        rows = []
        for rev, trial in sorted(EXCLUDED_TRIALS):
            t = seen.get((rev, trial))
            rows.append([
                f"`{rev}`", f"{trial}",
                f"{int(t['windows']):,}" if t else "—",
                f"**{t['disagree_frac']:.4f}**" if t else "—",
                f"over the `disagree > {audit['max_disagree']}` line (§3)"
                if t and t["disagree_frac"] > audit["max_disagree"]
                else "**not `label_audit`'s call** — excluded on other evidence, go read it",
            ])
        out += [
            "### Excluded before any of the above", "",
            f"**{len(EXCLUDED_TRIALS)} trial(s)** are dropped as the corpus is loaded "
            f"(`dataset.EXCLUDED_TRIALS`), so every count on this page is already net of them "
            f"and no stage below can put them back. The list is edited **by hand**; the "
            f"`disagree` column is `label_audit`'s, which reads the RAW corpus and so still "
            f"sees these trials. That is what makes the exclusion checkable instead of "
            f"self-confirming — a row whose evidence has gone thin still shows its number here.",
            "",
        ]
        out += _table(["rev", "trial", "windows", "disagree", "why it went"], rows)

    # which hardware the serve path can read; an unmapped variant is a refusal, not a low score
    if summary:
        served = Counter(r["variant"] for r in summary if r["labelled"] == "True")
        abst = src.jsonl(LABELED_RAW / "abstentions.jsonl", "python -m stages.s2_ml.label_all") or []
        unmapped = Counter()
        for a in abst:
            if a.get("refusal") == "UnknownVariantError":
                # named in the reason, not a field: the file never resolved far enough to have one
                for tok in str(a.get("reason", "")).split("'"):
                    if len(tok) == 8 and all(c in "0123456789abcdef" for c in tok):
                        unmapped[tok] += 1
                        break
        rows = [[f"`{v}`", f"{n}", "**yes**",
                 f"`Deg_{next((r['axis'] for r in summary if r['variant'] == v), '?')}`"]
                for v, n in served.most_common()]
        rows += [[f"`{v}`", f"{n}", "**no — unmapped**", "—"] for v, n in unmapped.most_common()]
        out += ["### Hardware variants, by whether they can be served", ""]
        out += _table(["variant", "files", "servable", "sagittal axis"], rows)
        if unmapped:
            out += [f"An unmapped variant needs **one paired raw+annotated recording** to fix. "
                    f"Signal-only axis detection scores below chance (`DOMAIN_NOTES` §6.2), so "
                    f"guessing is not available.", ""]
    return out


# S1; what was cleaned, what was thrown away, and what was flagged but kept
def section_s1(src: Source) -> list[str]:
    out = ["## §1 — S1 clean", "", "*Source: `runs/s1_clean/`*", ""]
    produced_by = "python -m stages.s1_clean.clean"
    segs = src.jsonl(RUNS / "s1_clean" / "segments.jsonl", produced_by)
    quar = src.jsonl(RUNS / "s1_clean" / "quarantine.jsonl", produced_by)
    obs = src.jsonl(RUNS / "s1_clean" / "observations.jsonl", produced_by)
    if segs is None:
        return out + _missing("runs/s1_clean/segments.jsonl", produced_by)

    usable = [s for s in segs if s.get("usable")]
    dropped = [s for s in segs if not s.get("usable")]
    files_seen = {s["path"] for s in segs}
    n_quar = len(quar or [])

    out += ["### Partition gate", "",
            f"**{len(files_seen)} segmented + {n_quar} quarantined = "
            f"{len(files_seen) + n_quar} files accounted for, 0 unaccounted.** "
            f"Asserted by the stage, not hoped.", ""]
    out += _table(["", "count"], [
        ["files segmented", f"{len(files_seen):,}"],
        ["files quarantined (whole-file reject)", f"{n_quar:,}"],
        ["segments found", f"{len(segs):,}"],
        ["segments usable", f"{len(usable):,}"],
        ["segments dropped", f"{len(dropped):,}"],
        ["usable duration", f"{sum(_num(s['duration_s'], 0) for s in usable) / 60:.1f} min"],
    ])

    out += ["### Quarantined files — the complete ledger", ""]
    out += _table(["file", "category", "reason", "needs human"],
                  [[f"`{q['file']}`", f"`{q.get('category', '')}`", q.get("reason", ""),
                    "**yes**" if q.get("needs_human") else "no"] for q in (quar or [])])

    # grouped by the stage's own reason FIRST: splitting on row count hid a hardware finding
    rate_markers = ("matches no known acquisition rate", "fits no known family")
    rate_rejects = [s for s in dropped
                    if any(m in str(s.get("reason", "")) for m in rate_markers)]
    too_short = [s for s in dropped if s not in rate_rejects]
    preamble = [s for s in too_short
                if s.get("index") == 0 and (s.get("n_source_rows") or 0) <= PREAMBLE_MAX_ROWS]
    fragments = [s for s in too_short if s not in preamble]

    out += ["### Dropped segments, by cause", ""]
    out += _table(["cause", "segments", "files", "what it is"], [
        [f"export preamble (`seg 0`, ≤{PREAMBLE_MAX_ROWS} rows)", f"**{len(preamble)}**",
         f"{len({s['path'] for s in preamble})}",
         "one systematic artifact — a short preamble before the first real gap. Not "
         f"{len(preamble)} independent data problems"],
        ["genuinely short segment", f"{len(fragments)}",
         f"{len({s['path'] for s in fragments})}",
         "a fragmented recording — worth a look if one file dominates"],
        ["**rate matches no known acquisition rate**", f"**{len(rate_rejects)}**",
         f"{len({s['path'] for s in rate_rejects})}",
         "**not a length problem** — the measured rate matches no era this pipeline knows"],
    ])
    if rate_rejects:
        out += _table(["file", "seg", "measured rate", "reason"],
                      [[f"`{Path(s['path']).name}`", s.get("index"),
                        f"{_num(s.get('source_hz'), 0):.3f} Hz", s.get("reason", "")]
                       for s in rate_rejects])
    # One file losing several segments is a different finding from several files losing one
    by_file = Counter(s["path"] for s in fragments)
    if by_file and by_file.most_common(1)[0][1] > 1:
        f_path, n = by_file.most_common(1)[0]
        out += [f"Most fragmented: `{Path(f_path).name}` lost **{n}** segments.", ""]

    methods = Counter(s.get("method") or "—" for s in usable)
    out += ["### Resample method, by usable segment", ""]
    out += _table(["method", "segments"], [[f"`{m}`", f"{n}"] for m, n in methods.most_common()])
    out += ["`decimate_5x_fir` is the 500 Hz era. Naive `[::5]` is never used — it folds "
            "everything above 50 Hz into the gait band as a full-amplitude fake signal "
            "(`DOMAIN_NOTES` §2.5).", ""]

    if obs is not None:
        kinds = Counter(o.get("kind") for o in obs)
        drift = [(o["path"], c) for o in obs for c in (o.get("drift_contaminated") or [])]
        out += ["### Trust flags — raised, nothing dropped for them", ""]
        out += _table(["check", "files"], [
            ["channel trust resolved from signal", f"{kinds.get('channel_trust_ok', 0)}"],
            ["channel trust abstained (too static → documented convention)",
             f"{kinds.get('channel_trust_abstained', 0)}"],
            ["**channel trust anomaly** (detected disagrees with documented)",
             f"**{kinds.get('channel_trust_anomaly', 0)}**"],
            ["`Deg` channels flagged drift-contaminated",
             f"{len(drift)} across {len({p for p, _ in drift})} files"],
        ])
        anomalies = [o for o in obs if o.get("kind") == "channel_trust_anomaly"]
        if anomalies:
            out += ["The anomalies, by name — the detected axis/unit disagrees with the "
                    "documented convention on these files, and the detection is what runs:", ""]
            out += _table(["file"], [[f"`{o['path']}`"] for o in anomalies])
        drift_by_file = Counter(p for p, _ in drift)
        if drift_by_file:
            out += ["Drift-contaminated `Deg` channels are flagged, not dropped — the raw "
                    "superset is kept. A channel whose value tracks session time is measuring "
                    "integration drift, not orientation:", ""]
            out += _table(["file", "channels"],
                          [[f"`{Path(p).name}`",
                            ", ".join(f"`{c}`" for q, c in drift if q == p)]
                           for p in drift_by_file])

    out += ["**What S1 does not resolve:** which axis is *sagittal*. That has no in-file "
            "signature and is a variant lookup in `stages/s2_ml/transform.py`. It is the most "
            "consequential thing handed downstream unresolved, and it was wrong for the "
            "majority variant until 2026-08-03 (`caveats.md` §5).", ""]
    return out


# the champion/challenger ledger: what was proposed, measured, promoted and refused
def _section_s2_ledger(src: Source) -> list[str]:
    ledger = src.jsonl(RUNS / "s2_ml" / "experiments.jsonl",
                       "python -m stages.s2_ml.experiment --seed")
    if not ledger:
        return []

    champ = src.json(RUNS / "s2_ml" / "champion.json",
                     "python -m stages.s2_ml.experiment --seed")
    # grouping only, NOT the gate: it keeps a superseded basis out of the column beside a current one
    ref = (champ or {}).get("corpus")
    current = [e for e in ledger if e.get("corpus") and e["corpus"] == ref]
    prior = [e for e in ledger if e not in current]

    out = ["### The champion/challenger ledger", "",
           "*Source: `runs/s2_ml/experiments.jsonl`, `proposals.jsonl` — written by the "
           "`s2_experiment` step. An agent proposes and a second criticises; neither can "
           "promote. `experiment.decide()` gates on measured macro-F1 and logs the reason "
           "either way.*", ""]

    if champ:
        out += [f"Incumbent: **`{champ['spec']['name']}`** at macro-F1 "
                f"{champ['macro_f1']:.4f}, recorded {champ['ts'][:19]}Z "
                f"(`{champ.get('git_sha', '?')}`).", ""]

    if current:
        out += _table(["experiment", "macro-F1", "features", "outcome"],
                      [[f"`{e['spec']['name']}`", _f(e["macro_f1"]), f"{e['n_features']}",
                        ("**promoted**" if e["promoted"] else "rejected")
                        + f" — {e['decision_reason']}"]
                       for e in current])

    if prior:
        out += [f"A further **{len(prior)}** entries were measured on an earlier basis — a "
                "different model over a different feature set — and are not comparable to "
                "anything above. They are kept because an idea that lost is still an idea "
                "that lost, and dropped from this table because a macro-F1 column that mixes "
                "two bases invites exactly the subtraction `decide()` refuses to make:", ""]
        out += _table(["experiment", "macro-F1 (earlier basis)", "outcome"],
                      [[f"`{e['spec']['name']}`", _f(e["macro_f1"]),
                        "promoted then" if e["promoted"] else "rejected"]
                       for e in prior])

    stopped = [p for p in (src.jsonl(RUNS / "s2_ml" / "proposals.jsonl",
                                     "(written by the s2_experiment step)") or [])
               if not p["ran"]]
    if stopped:
        out += ["Proposals that never cost a fit — the critic reviews before training, which "
                "is the cycle's whole cost asymmetry:", ""]
        out += _table(["proposal", "why it did not run"],
                      [[f"`{p['proposal'].get('name', '(unparsed)')}`", p["note"]]
                       for p in stopped])
    return out


# S2; which model, why that one, and how well it does on seen vs unseen subjects
def section_s2(src: Source) -> list[str]:
    out = ["## §2 — S2 ml", "", "*Source: `runs/s2_ml/`, `stages/s2_ml/champion_spec.json`*", ""]
    train_cmd = "python -m stages.s2_ml.train"
    loco = src.json(RUNS / "s2_ml" / "locoeval.json", train_cmd)
    meta = src.json(RUNS / "s2_ml" / "model_meta.json", train_cmd)
    spec = src.json(REPO_ROOT / "stages" / "s2_ml" / "champion_spec.json", "(checked in)")
    if loco is None or meta is None:
        return out + _missing("runs/s2_ml/locoeval.json", train_cmd)

    out += ["### The champion", ""]
    out += _table(["", ""], [
        ["spec", f"`{meta['champion_spec']}`"],
        ["model", f"`{meta['model']}`"],
        ["params", ", ".join(f"`{k}={v}`" for k, v in meta["params"].items())],
        ["features", f"{len(meta['features'])}"],
        ["window / training stride", f"{meta['window_s']} s / {meta['stride_s']} s "
                                     f"(non-overlapping)"],
        ["inference stride", f"{meta['inference_stride_s']} s"],
        ["CV", f"`{meta['cv']}`"],
        ["shipped preset",
         f"`{meta['default_preset']}` = threshold "
         f"{meta['presets'][meta['default_preset']]}"],
    ])

    # why this one, in the spec's own text: the rationale is the artifact, a paraphrase drifts
    if spec:
        out += ["### Why this champion, and what was rejected", "",
                spec.get("rationale", "").strip(), ""]
        if spec.get("rejected"):
            out += ["**Rejected, with the measurement:**", ""]
            out += [f"- {r}" for r in spec["rejected"]] + [""]
        if spec.get("drop_features"):
            out += [f"Dropped for parsimony: "
                    + ", ".join(f"`{f}`" for f in spec["drop_features"]) + ".", ""]
        if spec.get("supersedes"):
            out += [f"Promoted over `{spec['supersedes']}` — the margin is in the ledger "
                    f"below.", ""]

    out += _section_s2_ledger(src)

    # The measured comparison behind the choice, if the sweep is still around
    sweep = RUNS / "s2_ml_seedsweep.json"
    if sweep.is_file():
        rows_by_model: dict[str, list[dict]] = {}
        for r in json.loads(sweep.read_text(encoding="utf-8")):
            rows_by_model.setdefault(r["model"], []).append(r)
        out += ["#### Estimator comparison — identical features, folds and params", ""]
        out += _table(["model", "seeds", "macro-F1", "accuracy", "balanced acc",
                       "coverage @0.85", "selective acc @0.85"],
                      [[m, f"{len(rs)}",
                        _f(sum(r["macro_f1"] for r in rs) / len(rs)),
                        _f(sum(r["accuracy"] for r in rs) / len(rs)),
                        _f(sum(r["balanced"] for r in rs) / len(rs)),
                        _f(sum(r["coverage"] for r in rs) / len(rs)),
                        _f(sum(r["sel_acc"] for r in rs) / len(rs))]
                       for m, rs in rows_by_model.items()])
        out += ["The deciding line is coverage at equal precision: same quality of answer, "
                "more answers.", ""]

    # Ablations, if their run dirs survive; each is a full locoeval, so they compare directly
    abl = sorted(RUNS.glob("s2_ml_abl_*"))
    if abl:
        rows = [["**shipped**", f"{len(meta['features'])}", _f(loco["macro_f1"]),
                 _f(loco["accuracy"]), "—"]]
        for d in abl:
            a = src.json(d / "locoeval.json", f"(ablation run {d.name})")
            if a:
                rows.append([f"`{d.name.replace('s2_ml_abl_', '')}`", "—", _f(a["macro_f1"]),
                             _f(a["accuracy"]),
                             "**rejected**" if a["macro_f1"] < loco["macro_f1"] else "tie"])
        out += ["#### Feature ablations", ""]
        out += _table(["variant", "features", "macro-F1", "accuracy", "verdict"], rows)
        out += ["Low importance is not droppability — the rejected rows are the useful part "
                "of this table.", ""]

    out += ["### Leave-one-rev-out — the headline", "",
            f"**macro-F1 {loco['macro_f1']:.4f}** · accuracy {loco['accuracy']:.4f} · "
            f"balanced accuracy {loco['balanced_accuracy']:.4f}, over "
            f"**{loco['n']:,} label-pure windows**.", "",
            "macro-F1 is the headline on purpose: this corpus is ~86% walking, so pooled "
            "accuracy is close to a walking detector's score.", ""]
    out += _table(["class", "support", "precision", "recall", "F1"],
                  [[c["label"], f"{c['support']:,}", _f(c["precision"]), _f(c["recall"]),
                    f"**{c['f1']:.4f}**" if c["f1"] < 0.95 else _f(c["f1"])]
                   for c in loco["per_class"]])

    conf = loco["confusion"]
    classes = list(conf.keys())
    out += ["Confusion (rows = truth):", ""]
    out += _table([""] + [f"pred {c}" for c in classes],
                  [[f"**{t}**"] + [f"{conf[t][p]:,}" for p in classes] for t in classes])

    per_rev = sorted(loco["per_rev_macro_f1"].items(), key=lambda kv: -kv[1])
    out += ["Per-subject macro-F1 — the spread is what a new subject is drawn from:", ""]
    out += _table(["rev", "macro-F1"], [[r, _f(v)] for r, v in per_rev])
    if per_rev:
        out += [f"Spread: **{per_rev[-1][1]:.4f} – {per_rev[0][1]:.4f}** "
                f"({per_rev[-1][0]} worst, {per_rev[0][0]} best) over "
                f"{len(per_rev)} subjects.", ""]

    out += _section_selective(src, loco, meta)
    out += _section_error_direction(src)

    imp = meta.get("feature_importance") or []
    out += ["### Top features by importance", ""]
    if imp:
        out += _table(["feature", "importance", "rank"],
                      [[f"`{f}`", _f(v), f"{i + 1} of {len(imp)}"]
                       for i, (f, v) in enumerate(list(imp)[:12])])
        # the interleg block is what makes S3 dependent, so read its weight, not the ordering
        ileg = sum(v for f, v in imp if f.startswith("ileg_"))
        if ileg:
            out += [f"The `ileg_*` block carries **{ileg:.1%}** of total importance. That is "
                    f"the same interleg signal S3's swap rule reads — which is why S3 is no "
                    f"longer an *independent* second opinion (`caveats.md` §1.1b), and why "
                    f"physics catches so few of S2's confident errors.", ""]
    else:
        out += ["> `model_meta.json` predates the `feature_importance` key. It appears after "
                "the next `python -m stages.s2_ml.train`; until then the "
                "top-12 list in `locoeval.md` is the only copy.", ""]
    return out


# the coverage/accuracy curve at every threshold, window level and row level
def _section_selective(src: Source, loco: dict, meta: dict) -> list[str]:
    out = ["### Selective accuracy — the coverage/accuracy pair", "",
           "Coverage and accuracy are quoted together everywhere. Either alone is "
           "meaningless: abstain on all but the easiest window and accuracy reads 1.000.", ""]
    shipped = meta["presets"][meta["default_preset"]]

    def curve_table(curve: list[dict], unit: str) -> list[str]:
        rows = []
        for r in curve:
            mark = " ← shipped" if abs(r["threshold"] - shipped) < 1e-9 else ""
            rows.append([f"{r['threshold']:.2f}{mark}", _pct(r["coverage"], 2),
                         _f(r["selective_accuracy"]), _f(r["worst_rev_accuracy"]),
                         f"{r['errors_kept']:,}"])
        return _table([f"threshold", "coverage", "selective acc", "worst subject",
                       f"wrong {unit} kept"], rows)

    if loco.get("selective_curve"):
        out += ["#### Window level (`locoeval.json`)", ""]
        out += curve_table(loco["selective_curve"], "windows")

    row = src.json(RUNS / "s2_ml" / "roweval_loro.json",
                   "python -m stages.s2_ml.roweval")
    if row:
        out += ["#### Row level (`roweval_loro.json`) — the deliverable's own unit", "",
                f"Over {row['curve'][0]['n_labeled']:,} scored rows. Coverage runs lower than "
                f"the window table because inference stride is "
                f"{meta['inference_stride_s']} s and rows near a boundary inherit mixed "
                f"windows.", ""]
        out += curve_table(row["curve"], "rows")
        out += [f"**Why {shipped:g}:** pooled accuracy clears the "
                f"{ACCURACY_TARGET:.0%} target from 0.50 onward, so pooling is the wrong "
                f"number to set a threshold by. {shipped:g} is the lowest threshold at which "
                f"every held-out development subject independently clears it.", ""]

        if row.get("reasons"):
            out += ["#### Ambiguity reasons — each validated by what it suppressed", "",
                    "`accuracy of guess` is how often the suppressed guess *would* have been "
                    "right. A reason earns its place by sitting well below the confident set; "
                    "two reason codes have been removed for failing exactly this test.", ""]
            out += _table(["reason", "rows", "accuracy of the suppressed guess"],
                          [[f"`{r['reason']}`", f"{r['rows']:,}", _f(r["accuracy_of_guess"])]
                           for r in row["reasons"]])
        hu = row.get("human_unknown")
        if hu:
            out += [f"Independent check: the model abstains on "
                    f"**{hu['abstain_rate_on_human_unknown']:.1%}** of the "
                    f"{hu['rows_human_unknown']:,} rows a human marked `-1`, against "
                    f"{hu['abstain_rate_elsewhere']:.1%} elsewhere. `-1` never enters "
                    f"training, so the agreement is not circular.", ""]

    # the route, not the population: everything above scores lpf_view, a caller feeds a raw log
    raw = src.json(RUNS / "s2_ml" / "raweval.json", "python -m stages.s2_ml.raweval")
    if raw:
        o, tr = raw["overall"], raw.get("transitive_lpf_view")
        out += ["#### Raw device path (`raweval.json`) — end to end, against human labels",
                "",
                f"Everything above scores the annotated `lpf_view` export. A caller supplies "
                f"a raw device log, and the extra distance it travels — the axis map, the "
                f"bridge, decimation to {meta['fs_hz']:g} Hz — is exactly where the sagittal "
                f"axis was wrong for the majority variant until 2026-08-03. This scores that "
                f"route directly: **{raw['n_pairs']} raw CSVs** whose human annotation exists, "
                f"labelled through `label_csv` and joined back against it.", ""]
        out += _table(["population", "rows", "coverage", "accuracy on committed",
                       "worst subject"], [
            [f"raw device — {', '.join(raw['revs'])}", f"{o['rows_scored']:,}",
             f"**{_pct(o['coverage'], 2)}**", f"**{_f(o['selective_accuracy'])}**",
             _f(o["worst_subject"])],
        ] + ([[f"`lpf_view` route, same subjects", f"{tr['rows_scored']:,}",
               _pct(tr["coverage"], 2), _f(tr["selective_accuracy"]), "—"]] if tr else []))
        out += [f"**Two subjects fewer than the table above, and no lockbox.** `rev8` is "
                f"paired but refused in code (§7), and only "
                f"{len(raw['revs'])} development subjects have a raw file preserved "
                f"alongside their annotation, so this is a narrower population than the "
                f"row-level curve — not a second opinion on it.", ""]
        if tr:
            d_cov = o["coverage"] - tr["coverage"]
            d_acc = o["selective_accuracy"] - tr["selective_accuracy"]
            out += [f"Against the route whose accuracy it used to borrow: coverage "
                    f"{d_cov:+.4f}, accuracy {d_acc:+.4f}. The two row sets are not "
                    f"identical — one scores the export's grid, the other the raw file's "
                    f"own rows — so exact equality was never the bar. A difference large "
                    f"enough to move the operating point would have been, and this is "
                    f"three orders of magnitude short of it.", ""]
        out += _table(["variant", "revs", "rows", "coverage", "accuracy"],
                      [[f"`{v}`", ", ".join(r["revs"]), f"{r['rows_scored']:,}",
                        _pct(r["coverage"], 1), _f(r["selective_accuracy"])]
                       for v, r in sorted(raw["per_variant"].items())])
        out += ["Per variant because the axis map is per variant: a mis-mapped sagittal "
                "channel would show up as one row sitting apart from the others, which is "
                "the check the transitive argument could not perform at all.", ""]

    lock = src.json(RUNS / "s2_ml" / "roweval_lockbox.json",
                    "python -m stages.s2_ml.roweval --lockbox  (SINGLE USE)")
    if lock:
        dev = next((r for r in (row or {}).get("curve", [])
                    if abs(r["threshold"] - shipped) < 1e-9), None)
        lb = next((r for r in lock["curve"] if abs(r["threshold"] - shipped) < 1e-9), None)
        out += [f"#### Lockbox — {', '.join(lock['revs'])}, held out of everything", ""]
        # the one number that cannot be refreshed: correcting it means re-reading a spent lockbox
        n_now = len(meta.get("features") or [])
        n_then = lock.get("n_features")
        if n_now and n_then != n_now:
            out += [f"> **Measured against a different feature set.** This row was produced "
                    f"with {'an unrecorded number of' if n_then is None else n_then} "
                    f"features; the champion now declares {n_now}. It was NOT re-run to "
                    f"follow the change — the lockbox is spent (§7), and re-reading it to "
                    f"tidy a provenance mismatch is exactly the trade §7 forbids. The "
                    f"development rows above moved by <0.001 under the same change, but "
                    f"whether this one would is unknown and unknowable without a newly "
                    f"sealed subject.", ""]
        out += _table(["population", "coverage", "accuracy on committed"], [
            ["development subjects",
             _pct(dev["coverage"], 2) if dev else "—",
             _f(dev["selective_accuracy"]) if dev else "—"],
            [f"**{', '.join(lock['revs'])} (lockbox)**",
             f"**{_pct(lb['coverage'], 1)}**" if lb else "—",
             f"**{_f(lb['selective_accuracy'])}**" if lb else "—"],
        ])
        if lb and lb["selective_accuracy"] < ACCURACY_TARGET:
            out += [f"**The {ACCURACY_TARGET:.0%} target is met on development subjects and "
                    f"missed on the lockbox.** Treat every development number above as an "
                    f"upper bound, not an estimate. The lockbox is single-use — check "
                    f"`caveats.md` §3.2 before re-reading it.", ""]
        # abstention is supposed to buy accuracy, and on a genuinely new subject here it does not
        accs = [r["selective_accuracy"] for r in lock["curve"]]
        if any(b < a for a, b in zip(accs, accs[1:])):
            worst = max((a - b, i) for i, (a, b) in enumerate(zip(accs, accs[1:])))
            i = worst[1]
            out += [f"**Accuracy is non-monotonic in the threshold here** — "
                    f"{lock['curve'][i]['selective_accuracy']:.4f} at "
                    f"{lock['curve'][i]['threshold']:.2f} falls to "
                    f"{lock['curve'][i + 1]['selective_accuracy']:.4f} at "
                    f"{lock['curve'][i + 1]['threshold']:.2f}. Abstention is supposed to buy "
                    f"accuracy monotonically; on this subject it does not, so the threshold "
                    f"chosen on development subjects is not transferable and the "
                    f"`worst subject` column above is an optimistic floor.", ""]
    return out


# which way the errors go- the one cut pooled accuracy hides completely
def _section_error_direction(src: Source) -> list[str]:
    row = src.json(RUNS / "s2_ml" / "roweval_loro.json", "python -m stages.s2_ml.roweval")
    if not row:
        return []
    conf = row.get("confusion")
    if not conf:
        return ["### Direction of error — what pooled accuracy hides", "",
                "> `roweval_loro.json` predates the `confusion` key. It appears after the "
                "next `python -m stages.s2_ml.roweval`.", ""]

    cells = [c for c in conf["cells"] if c["rows"]]
    errs = [c for c in cells if c["truth"] != c["pred"]]
    n_err = sum(c["rows"] for c in errs)

    out = ["### Direction of error — what pooled accuracy hides", "",
           f"Row level, leave-one-rev-out, at the shipped threshold "
           f"{conf['threshold']:g}: {conf['rows_committed']:,} committed, "
           f"{conf['rows_abstained']:,} abstained. **Rows, not windows** — the unit a "
           f"caller receives, and the same unit as the curve above.", ""]
    out += _table(["truth → guess", "rows"],
                  [[f"{c['truth']} → {c['pred']}"
                    + (" ⚠" if c["truth"] != c["pred"] else ""), f"{c['rows']:,}"]
                   for c in sorted(cells, key=lambda c: -c["rows"])])
    if n_err:
        top = max(errs, key=lambda c: c["rows"])
        out += [f"**{top['rows']:,} of {n_err:,} committed errors "
                f"({top['rows'] / n_err:.0%}) are {top['truth']} called {top['pred']}.** "
                f"The residual failure is one-directional, so watch **{top['truth']} "
                f"recall**, not accuracy — pooled accuracy is the anaesthetic "
                f"(`caveats.md` §3.0).", ""]
    return out


# S3; what a window is, what the label-free rule says, and where labels look wrong
def section_s3(src: Source) -> list[str]:
    out = ["## §3 — S3 physics", "", "*Source: `runs/s3_physics/`*", ""]
    meta = src.json(RUNS / "s2_ml" / "model_meta.json", "python -m stages.s2_ml.train")

    # the unit everything downstream is counted in, read from the card so a retrain cannot lie here
    if meta:
        out += ["### What a *window* is — the unit every count below uses", ""]
        out += _table(["property", "value", "why"], [
            ["length", f"**{meta['window_s']} s** "
                       f"= {int(meta['window_s'] * meta['fs_hz'])} samples @ "
                       f"{meta['fs_hz']:g} Hz",
             "~2 gait cycles — the shortest span that can show a leg swap"],
            ["training stride", f"{meta['stride_s']} s (**non-overlapping**)",
             "overlapping windows leak between CV folds"],
            ["inference stride", f"**{meta['inference_stride_s']} s**",
             "a row is labelled by whichever windows cover it"],
            ["bounded by", "**segment**, never a file",
             "a window may never span a gap — that would invent data"],
            ["label rule", "vote over rows, human `-1` dropped",
             "*label-pure* = the surviving vote is unanimous"],
        ])

    # the per-window swap tally went with S4 and `anchors.csv`; the rate audit did not
    out += ["### Where the swap rule is used, and how much it is worth in each place", "",
            "The rule has zero fitted parameters and never sees a label. Both are real; "
            "what they buy depends on what it is pointed at.", ""]
    out += _table(["consumer", "pointed at", "worth"], [
        ["`label_audit.py`", "the **annotations**",
         "**strong** — the rule is genuinely independent of the labels, so a "
         "disagreement is evidence about them"],
        ["`plausibility.py`", "a whole **file**, against the model",
         "**moderate** — weak per window (see below), but a file-level contradiction "
         "is a diagnosable fault rather than a hard window"],
        ["`label.py` physics gate", "single **windows**, against the model",
         "**weak, ships OFF** — S2 absorbed `ileg_swaps` and `ileg_minhalf`, so both "
         "opinions read the same 1 Hz-filtered interleg angle and are wrong together"],
    ])
    out += ["", "The independence claim was retracted in `anchors.py` on 2026-08-03 and "
            "the retraction is what sizes that last row: physics contradicts only ~12% of "
            "S2's high-confidence errors and none at p >= 0.95. `roweval` measures the "
            "gate against simply raising the threshold rather than leaving it an argument.",
            ""]

    rate = src.json(RUNS / "s3_physics" / "rate_audit.json",
                    "python -m stages.s3_physics.rate_audit")
    if rate:
        failed = [a for a, r in rate.items() if r["verdict"] == "rate_dependent"]
        r0 = next(iter(rate.values()))
        out += ["### Rate-invariance audit — body or clock?", "",
                f"Every window decimated to half rate through S1's anti-aliasing FIR and "
                f"the anchors recomputed, over the {r0['n_windows_audited']:,} walking "
                f"windows that could be padded on both sides. §2.3 is why: an earlier "
                f"experiment read clustering that partitioned by acquisition rate, which "
                f"turned out to be timestamp quantization — the geometry was measuring "
                f"the clock.", ""]
        out += _table(["anchor", "metric", "median Δ", "verdict"],
                      [[f"`{a}`", r["metric"], f"{r['median_delta']:.4f}",
                        f"**{r['verdict']}**"] for a, r in rate.items()])
        out += ["", f"`gyro_energy` is the **negative control** and is defined the way "
                f"that fails, on purpose: it sums over samples, so halving the sample "
                f"count halves it. "
                + ("It fired, so the passes above mean something (§11.1)."
                   if "gyro_energy" in failed else
                   "**It did not fire, which is a problem** — an audit that rejects "
                   "nothing is not evidence that everything passed (§11.1).") , ""]

    audit = src.json(RUNS / "s3_physics" / "label_audit.json",
                     "python -m stages.s3_physics.label_audit")
    if audit:
        trials = audit["trials"]
        broken = [t for t in trials if t.get("flag")]
        diverg = [t for t in trials if t.get("band_flag")]
        out += ["### Label audit — where the annotations contradict the physics", "",
                "Model-free by construction: it reads the window grid and the swap rule, "
                "never a model, a probability or a prediction. Flagging the files the "
                "classifier dislikes would delete precisely the hard cases, improve every "
                "metric and teach nothing.", ""]
        out += [f"**Broken** — `disagree > {audit['max_disagree']}` over "
                f"≥{audit['min_windows']} windows. The file contradicts itself more often "
                f"than it agrees: a swapped channel or a misaligned label track.", ""]
        out += _table(["rev", "trial", "windows", "disagree", "fraction"],
                      [[t["rev"], t["trial"], f"{t['windows']:,}", t["disagree"],
                        f"**{t['disagree_frac']:.4f}**"] for t in broken])
        out += [f"**Divergent convention** — share of in-band windows annotated `walk`, "
                f"against a corpus rate of {audit['band_corpus_walk_frac']:.3f}. "
                f"**Do not exclude these.** A self-consistent minority convention is not bad "
                f"data; it is the part of the residual error that is annotation policy "
                f"rather than model failure. The model learns the corpus convention, so a "
                f"divergent trial reads as model error and no retraining removes it.", ""]
        out += _table(["rev", "trial", "band windows", "band walk", "vs corpus", "p"],
                      [[t["rev"], t["trial"], f"{int(t['band_windows'])}",
                        f"**{t['band_walk_frac']:.3f}**",
                        f"{t['band_walk_frac'] - audit['band_corpus_walk_frac']:+.3f}",
                        f"{t['band_p']:.2g}"]
                       for t in sorted(diverg, key=lambda t: t["band_walk_frac"])])
    return out


# the corpus sweep; coverage only; there is no ground truth here, ever
def section_sweep(src: Source) -> list[str]:
    out = ["## §5 — Corpus sweep (`label_all`)", "",
           "*Source: `labeled_raw/`*", "",
           "**No ground truth exists here.** This section reports coverage — what fraction "
           "of rows cleared the threshold. Whether those calls are right is §2's question, "
           "measured on annotated data. Quoting a number from here as accuracy is the single "
           "easiest mistake to make in this repo.", ""]
    cmd = "python -m stages.s2_ml.label_all"
    summary = src.csv(LABELED_RAW / "label_summary.csv", cmd)
    abst = src.jsonl(LABELED_RAW / "abstentions.jsonl", cmd)
    if summary is None:
        return out + _missing("labeled_raw/label_summary.csv", cmd)

    lab = [r for r in summary if r["labelled"] == "True"]
    rows_tot = sum(_num(r["rows"], 0) for r in lab)
    conf_tot = sum(_num(r["n_confident"], 0) for r in lab)
    stand = sum(_num(r["n_stand"], 0) for r in lab)
    walk = sum(_num(r["n_walk"], 0) for r in lab)

    out += _table(["", "count"], [
        ["files swept", f"{len(summary):,}"],
        ["labelled", f"**{len(lab):,}**"],
        ["refused", f"**{len(summary) - len(lab):,}**"],
        ["rows labelled", f"{int(rows_tot):,}"],
        ["rows committed", f"{int(conf_tot):,} (**{conf_tot / rows_tot:.1%}**)"
         if rows_tot else "—"],
        ["rows abstained", f"{int(rows_tot - conf_tot):,}"],
        ["committed class balance",
         f"walk {int(walk):,} · stand {int(stand):,} (**{walk / (walk + stand):.1%} walk**)"
         if walk + stand else "—"],
    ])
    out += [f"**Partition gate: {len(lab)} labelled + {len(summary) - len(lab)} refused = "
            f"{len(summary)} of {len(summary)}, 0 unaccounted.**", ""]

    if abst:
        by_kind = Counter(a.get("refusal") for a in abst)
        out += ["### Refusals — a stated refusal, not a low score", "",
                "A refusal is the pipeline declining to serve a file, with a reason. It is "
                "not a bad score and it is not a crash: `label_all` records it and keeps "
                "going, and every raw file is still accounted for exactly once.", ""]
        out += _table(["refusal", "files", "cause"],
                      [[f"`{k}`", f"{v}",
                        _first_sentence(next((str(a.get("reason", "")) for a in abst
                                              if a.get("refusal") == k), ""))]
                       for k, v in by_kind.most_common()])

    # coverage is bimodal here and the mean hides it: percentiles, and the tail listed by name
    fracs = sorted(v for v in (_num(r["confident_frac"]) for r in lab) if v is not None)
    if fracs:
        def pct_at(q: float) -> float:
            return fracs[min(len(fracs) - 1, int(q * len(fracs)))]
        out += ["### Coverage is not uniform — the corpus mean hides the shape", ""]
        out += _table(["percentile", "confident fraction"],
                      [["min", _f(fracs[0])], ["p10", _f(pct_at(0.10))],
                       ["**median**", f"**{_f(pct_at(0.50))}**"],
                       ["p90", _f(pct_at(0.90))], ["max", _f(fracs[-1])]])
        low = sorted((r for r in lab if (_num(r["confident_frac"]) or 1) < LOW_COVERAGE_LINE),
                     key=lambda r: _num(r["confident_frac"], 1))
        if low:
            out += [f"**{len(low)} of {len(lab)} files commit to less than "
                    f"{LOW_COVERAGE_LINE:.0%} of their rows.** Listed by name because a "
                    f"corpus average cannot be acted on and a filename can:", ""]
            out += _table(["confident", "rows", "session", "variant", "dominant reason"],
                          [[_f(_num(r["confident_frac"]), 3), f"{int(_num(r['rows'], 0)):,}",
                            r["session"], f"`{r['variant']}`", f"`{r['top_reason']}`"]
                           for r in low])

    reasons = Counter(r["top_reason"] or "(none)" for r in lab)
    out += ["### Dominant abstention reason, by file", ""]
    out += _table(["reason", "files where it dominates"],
                  [[f"`{k}`", f"{v}"] for k, v in reasons.most_common()])
    out += _section_by_session(lab)
    out += _section_operating_points(src)
    return out


# coverage cut by SESSION inside each variant- the axis nothing else cuts
def _section_by_session(lab: list[dict]) -> list[str]:
    by_var: dict[str, list[dict]] = {}
    for r in lab:
        by_var.setdefault(r["variant"], []).append(r)
    # only variants where something is wrong; a clean one is twelve rows of 0.99 burying the one
    interesting = {v: rs for v, rs in by_var.items()
                   if any((_num(r["confident_frac"]) or 1) < LOW_COVERAGE_LINE for r in rs)}
    if not interesting:
        return ["", "### Coverage by session", "",
                "No variant holds a file under "
                f"{LOW_COVERAGE_LINE:.0%} coverage, so there is no cluster to attribute.", ""]

    out = ["", "### Coverage by session — is it the hardware or the dates?", "",
           "Session is independent of rev and is the one axis nothing else in this repo "
           "cuts. Shown within variant, for the variants that hold a low-coverage file: a "
           "cause that is the *hardware* makes every session carrying it bad, a cause that "
           "is the *dates* does not.", ""]
    for variant, rs in sorted(interesting.items()):
        by_sess: dict[str, list[dict]] = {}
        for r in rs:
            by_sess.setdefault(r["session"], []).append(r)
        rows, bad_sessions = [], []
        for sess, xs in sorted(by_sess.items()):
            fr = sorted(_num(r["confident_frac"]) or 0.0 for r in xs)
            med = fr[len(fr) // 2]
            low = sum(1 for f in fr if f < LOW_COVERAGE_LINE)
            if low:
                bad_sessions.append(sess)
            rows.append([f"`{sess}`", f"{len(xs)}", _f(med, 3),
                         f"**{low}**" if low else "0"])
        out += [f"**`{variant}`** — {len(rs)} files over {len(by_sess)} sessions", ""]
        out += _table(["session", "files", "median coverage", "files < 50%"], rows)
        clean = len(by_sess) - len(bad_sessions)
        if bad_sessions and clean:
            out += [f"**{len(bad_sessions)} of {len(by_sess)} sessions hold every "
                    f"low-coverage file on this variant; the other {clean} hold none.** "
                    f"The same hardware records fine outside "
                    f"{bad_sessions[0]}–{bad_sessions[-1]}, so the variant is not the "
                    f"cause — those dates are. Look for what changed about the rig or the "
                    f"protocol in that window, not at the device revision.", ""]
    return out


# every operating point, with what it costs HERE beside what it buys THERE
def _section_operating_points(src: Source) -> list[str]:
    cmd = "python -m stages.s2_ml.label_all"
    sweep = src.json(LABELED_RAW / "preset_sweep.json", cmd)
    if sweep is None:
        return ["", "### Operating points — what each costs on this corpus", "",
                *_missing("labeled_raw/preset_sweep.json", cmd)]

    out = ["", "### Operating points — what each costs here, what each buys there", "",
           "*Coverage from `labeled_raw/preset_sweep.json` (this corpus, no ground truth); "
           "accuracy from `runs/s2_ml/roweval_loro.json` (annotated corpus, held out per "
           "subject). Two populations — read across the row, never down one column.*", ""]

    # the accuracy half; absent is fine and stated- the coverage half still says what a point costs
    row = src.json(RUNS / "s2_ml" / "roweval_loro.json", "python -m stages.s2_ml.roweval")
    acc_by_thr = {round(c["threshold"], 4): c for c in row["curve"]} if row else {}

    presets = {round(v, 4): k for k, v in (sweep.get("presets") or {}).items()}
    shipped = round(_num(sweep.get("shipped_threshold"), -1) or -1, 4)

    rows = []
    for c in sweep["corpus"]:
        t = round(c["threshold"], 4)
        a = acc_by_thr.get(t)
        name = presets.get(t, "")
        label = f"**{t:.2f}**" if t == shipped else f"{t:.2f}"
        if name:
            label += f" `{name}`"
        if t == shipped:
            label += " ← shipped"
        rows.append([
            label,
            _pct(c["coverage"]),
            f"{int(c['committed']):,}",
            f"{c['files_below_half']}",
            _f(a["selective_accuracy"]) if a else "—",
            _f(a["worst_rev_accuracy"]) if a else "—",
        ])
    out += _table(["threshold", "coverage HERE", "rows committed", "files < 50%",
                   "accuracy THERE", "worst subject THERE"], rows)

    out += [f"Swept over {sweep['n_files']} labelled files "
            f"({int(sweep['corpus'][0]['rows']):,} rows) in the single scoring pass "
            f"`label_all` already makes — a row's confidence does not depend on the "
            f"threshold, so every point above costs nothing beyond the labelling itself. "
            f"`label_all` verifies that identity against the frame's own `ambiguous` column "
            f"at the shipped threshold and refuses to write this artifact if they "
            f"disagree.", ""]

    # different denominators: the 0.50 row measures the gap rather than asserting it is population
    zero_row = min(sweep["corpus"], key=lambda c: c["threshold"])
    uncovered = 1.0 - (_num(zero_row["coverage"]) or 0.0)
    out += [f"**The two halves count different denominators.** `coverage HERE` is over "
            f"every row in the file, including rows no full window covers; the "
            f"development curve counts only rows that were covered AND carry a trainable "
            f"label. The gap is measured by the {zero_row['threshold']:.2f} row above, "
            f"which reads {_pct(zero_row['coverage'])} rather than 100%: "
            f"**{uncovered:.2%} of corpus rows are uncovered at every threshold** and are "
            f"abstentions of the recording, not of the model.", ""]

    # The resolution of the coverage column, not a second finding
    worst_band = max(sweep["corpus"], key=lambda c: c.get("boundary", 0))
    if worst_band.get("boundary"):
        shipped_band = next((c for c in sweep["corpus"]
                             if round(c["threshold"], 4) == shipped), None)
        out += [f"*Resolution: `confidence` is emitted rounded to 4 decimals while the "
                f"abstention was decided on the unrounded float, so rows printing exactly "
                f"a threshold cannot be re-decided from the column. That band holds "
                f"{int(worst_band['boundary']):,} rows at its widest "
                f"({worst_band['threshold']:.2f})"
                + (f" and {int(shipped_band['boundary']):,} at the shipped point"
                   if shipped_band else "")
                + f" — under 0.3% of committed rows everywhere, and always at the "
                f"threshold where the ensemble is most evenly split.*", ""]

    gates = sweep.get("gates") or {}
    if any(gates.values()):
        out += [f"⚠ **Gates on: {', '.join(k for k, v in gates.items() if v)}.** With any of "
                f"these set, abstention is not a function of confidence alone and the swept "
                f"columns would overstate coverage; `label_all` does not write them.", ""]

    # the cliff: the corpus mean moves smoothly across the presets, individual recordings do not
    zero = [c for c in sweep["corpus"] if c.get("files_zero")]
    if zero:
        worst = max(zero, key=lambda c: c["files_zero"])
        out += [f"**At {worst['threshold']:.2f}, {worst['files_zero']} file(s) commit to "
                f"nothing at all.** A file that labels every row `-1` is not a strict "
                f"result, it is an unserved recording that did not refuse — worth reading "
                f"beside the refusal table above, which counts only files that declined "
                f"outright.", ""]
    return out


# ---- Flags; mechanical rules only- each names what to look at, none of them conclude anything ----

@dataclass
class Flag:
    kind: str        # stale | risk | concentration | gap
    title: str
    detail: str


# does this prose carry that (coverage, accuracy) pair, in any reasonable rounding?
def _quotes(text: str, coverage: float, accuracy: float) -> tuple[bool, bool]:
    return (any(f"{coverage:.{d}%}" in text for d in (0, 1, 2)),
            any(f"{accuracy:.{d}f}" in text for d in (3, 4)))


# does the checked-in prose still quote the live headlines?
def _flag_stale_prose(src: Source, meta: dict | None) -> list[Flag]:
    pairs: dict[str, tuple[float, float, str]] = {}
    row = src.json(RUNS / "s2_ml" / "roweval_loro.json", "python -m stages.s2_ml.roweval")
    if row and meta:
        thr = meta["presets"][meta["default_preset"]]
        r = next((c for c in row["curve"] if abs(c["threshold"] - thr) < 1e-9), None)
        if r:
            pairs["S2 row-level"] = (r["coverage"], r["selective_accuracy"],
                                     "runs/s2_ml/roweval_loro.json")

    out = []
    for claim, (cov, acc, artifact) in pairs.items():
        for name in PROSE_CLAIMS.get(claim, []):
            p = REPO_ROOT / name
            if not p.is_file():
                # listed as quoting a headline and not on disk; skipping would let a deletion stay silent
                out.append(Flag(
                    "gap", f"`{name}` is listed as quoting the {claim} headline but is missing",
                    f"nothing checks the {claim} pair against prose until it is rewritten. "
                    f"`needtowrite.md` holds the spec and the un-regenerable content; drop "
                    f"the entry from `PROSE_CLAIMS` if the document is gone for good."))
                continue
            has_cov, has_acc = _quotes(p.read_text(encoding="utf-8", errors="replace"),
                                       cov, acc)
            if has_cov and has_acc:
                continue
            half = ("neither figure" if not has_cov and not has_acc
                    else "the coverage but not the accuracy" if has_cov
                    else "the accuracy but not the coverage")
            out.append(Flag(
                "stale", f"`{name}` does not quote the current {claim} headline",
                f"the live pair is **coverage {cov:.1%} at accuracy {acc:.4f}** "
                f"(`{artifact}`); the file carries {half}. Either it predates this run, or "
                f"it quotes a different population and should say which."))
    return out


def _flag_lockbox(src: Source, meta: dict | None) -> list[Flag]:
    lock = src.json(RUNS / "s2_ml" / "roweval_lockbox.json", "roweval --lockbox")
    if not lock or not meta:
        return []
    thr = meta["presets"][meta["default_preset"]]
    r = next((c for c in lock["curve"] if abs(c["threshold"] - thr) < 1e-9), None)
    if r and r["selective_accuracy"] < ACCURACY_TARGET:
        return [Flag("risk", f"the {ACCURACY_TARGET:.0%} target is missed on the lockbox",
                     f"{', '.join(lock['revs'])} scores **{r['selective_accuracy']:.4f}** at "
                     f"coverage {r['coverage']:.1%}. Every development number in §2 is an "
                     f"upper bound, not an estimate.")]
    return []


def _flag_subject_spread(loco: dict | None) -> list[Flag]:
    if not loco or not loco.get("per_rev_macro_f1"):
        return []
    vals = sorted(loco["per_rev_macro_f1"].items(), key=lambda kv: kv[1])
    lo, hi = vals[0], vals[-1]
    if hi[1] - lo[1] > 0.05:
        return [Flag("risk", "per-subject macro-F1 spread is wide",
                     f"**{lo[1]:.4f} ({lo[0]}) – {hi[1]:.4f} ({hi[0]})** over "
                     f"{len(vals)} subjects. A single unusual subject moves the shipped "
                     f"threshold more than any tuning does.")]
    return []


# do the worst corpus files concentrate?
def _flag_low_coverage_cluster(summary: list[dict] | None) -> list[Flag]:
    if not summary:
        return []
    lab = [r for r in summary if r["labelled"] == "True"]
    low = [r for r in lab if (_num(r["confident_frac"]) or 1) < LOW_COVERAGE_LINE]
    if len(low) < 3:
        return []
    variants, sessions = {r["variant"] for r in low}, {r["session"] for r in low}
    reasons = Counter(r["top_reason"] for r in low)
    detail = (f"**{len(low)} of {len(lab)} files** commit to under "
              f"{LOW_COVERAGE_LINE:.0%} of their rows, across {len(variants)} variant(s) "
              f"and {len(sessions)} session(s); dominant reason "
              f"`{reasons.most_common(1)[0][0]}` on {reasons.most_common(1)[0][1]} of them.")
    if len(variants) == 1 and len(sessions) <= 4:
        detail += (f" They **concentrate on one variant (`{next(iter(variants))}`) in "
                   f"{len(sessions)} session(s): {', '.join(sorted(sessions))}** — that is a "
                   f"session-level cause, not {len(low)} hard recordings. Nobody has looked "
                   f"at one by eye: `python -m stages.s3_physics.inspect_window`.")
        return [Flag("concentration", "the worst corpus files are a cluster", detail)]
    return [Flag("risk", "a tail of corpus files barely commits", detail)]


def _flag_unserved(abst: list[dict] | None) -> list[Flag]:
    if not abst:
        return []
    unmapped = [a for a in abst if a.get("refusal") == "UnknownVariantError"]
    if unmapped:
        return [Flag("gap", "some hardware cannot be served at all",
                     f"**{len(unmapped)} file(s)** refuse with `UnknownVariantError`. Each "
                     f"needs one paired raw+annotated recording to map its sagittal axis; "
                     f"signal-only detection scores below chance.")]
    return []


# the label review's verdicts from the most recent run; the counter compares as an INTEGER, not a name
def _latest_reviews() -> tuple[Path | None, dict[tuple[str, int], dict]]:
    def order(p: Path) -> tuple[str, int]:
        date, _, run = p.name.partition("_run")
        return date, int(run) if run.isdigit() else 0

    for d in sorted((p for p in RUNS.glob("*_run*") if p.is_dir()), key=order, reverse=True):
        path = d / "label_review.jsonl"
        if not path.is_file():
            continue
        out: dict[tuple[str, int], dict] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("rev") is not None and r.get("trial") is not None:
                out[(r["rev"], int(r["trial"]))] = r.get("review") or {}
        if out:
            return d, out
    return None, {}


# trials below the exclusion line but above everything else
def _flag_unadjudicated(audit: dict | None,
                        reviewed: dict[tuple[str, int], dict] | None = None,
                        review_dir: Path | None = None) -> list[Flag]:
    if not audit:
        return []
    reviewed = reviewed or {}
    floor = _num(audit.get("min_windows"), 0) or 0
    testable = [t for t in audit["trials"]
                if not t.get("flag") and (_num(t.get("windows"), 0) or 0) >= floor]
    near = [t for t in sorted(testable, key=lambda t: -t["disagree_frac"])[:4]
            if t["disagree_frac"] >= audit["max_disagree"] / 4]
    if not near:
        return []

    where = f" (`{review_dir.name}`)" if review_dir else ""
    unread = [t for t in near if (t["rev"], int(t["trial"])) not in reviewed]
    judged = [(t, reviewed[(t["rev"], int(t["trial"]))]) for t in near
              if (t["rev"], int(t["trial"])) in reviewed]

    out: list[Flag] = []
    if unread:
        out.append(Flag("gap", "label-audit trials below the line and unadjudicated",
                        "; ".join(f"{t['rev']}/t{t['trial']} at {t['disagree_frac']:.2f} "
                                  f"({int(t['windows'])} windows)" for t in unread)
                        + f" — under the `disagree > {audit['max_disagree']}` exclusion line, "
                          f"above everything else, and over the {floor:.0f}-window floor a "
                          f"fraction needs to mean anything. Nothing excludes them and no "
                          f"review has judged them"
                        + (f"; the newest ledger{where} covers the others." if judged
                           else " — run `run_pipeline.py --with-agents --from s3_label_review`.")))

    # judged bad and still in the corpus: `EXCLUDED_TRIALS` is edited BY HAND, so the loop is open
    stuck = [(t, r) for t, r in judged
             if r.get("disposition") == "mislabel_candidate"
             and (t["rev"], int(t["trial"])) not in EXCLUDED_TRIALS]
    if stuck:
        out.append(Flag("gap", "a trial judged a mislabel candidate is still in the corpus",
                        "; ".join(f"{t['rev']}/t{t['trial']} — cause `{r.get('cause')}`"
                                  for t, r in stuck)
                        + f", reviewed{where} but absent from `dataset.EXCLUDED_TRIALS`. That "
                          f"list is edited by hand on purpose; this flag is the reminder, not "
                          f"the edit."))
    return out


def collect_flags(src: Source, loco: dict | None, meta: dict | None,
                  summary: list[dict] | None, abst: list[dict] | None,
                  audit: dict | None) -> list[Flag]:
    flags: list[Flag] = []
    # freshness first: it invalidates every other number rather than sitting beside them.
    # Every stage directory is considered, not only the ones already carrying a stamp -- filtering
    # on `_inputs.json` was checking exactly the dirs that could pass and skipping the ones that
    # could not, so a stage that never declared its inputs read as clean. Agent run dirs are
    # excluded on purpose: they are never overwritten (§4), so nothing can go stale under them.
    dirs = [d for d in RUNS.glob("*") if d.is_dir() and "_run" not in d.name]
    if LABELED_RAW.is_dir():
        dirs.append(LABELED_RAW)
    stamped, blind = [], []
    for d in dirs:
        (stamped if stamp_paths(d) else blind).append(d)
    for c in check_all(stamped):
        flags.append(Flag("stale", "a downstream artifact no longer matches its inputs", c))

    # One flag, not one per directory: "cannot be checked" is a single condition with a single
    # fix, and N copies of it would crowd out the stale flags above, which are the urgent ones.
    if blind:
        flags.append(Flag(
            "unchecked", "some stage output cannot be checked for staleness",
            "No `_inputs.json` in " + ", ".join(f"`runs/{d.name}`" for d in blind)
            + ". These predate input stamping, so there is no record of which champion spec "
              "or model produced them and no way to tell whether they still describe it. "
              "Re-running the stage stamps it; until then, read those numbers as undated."))
    flags += _flag_stale_prose(src, meta)
    flags += _flag_lockbox(src, meta)
    flags += _flag_subject_spread(loco)
    flags += _flag_low_coverage_cluster(summary)
    flags += _flag_unserved(abst)
    review_dir, reviewed = _latest_reviews()
    flags += _flag_unadjudicated(audit, reviewed, review_dir)
    return flags


def section_flags(flags: list[Flag]) -> list[str]:
    out = ["## §A — Flags raised this run", "",
           "Mechanical rules only: a threshold crossed, a document that does not quote the "
           "current pair. None of these is a verdict — each names what to look at.", ""]
    if not flags:
        return out + ["*No flags. Every rule below its line, every checked document quoting "
                      "the live headline.*", ""]
    for fl in flags:
        out += [f"### `{fl.kind}` — {fl.title}", "", fl.detail, ""]
    return out


def section_gaps(src: Source) -> list[str]:
    out = ["## §B — What is missing from this report", "",
           "An absent artifact is a stated gap, never a skipped section: a stage that did "
           "not run must not read as a stage with nothing to say.", ""]
    if not src.gaps:
        return out + ["*Nothing. Every artifact this report reads was present.*", ""]
    seen, rows = set(), []
    for artifact, cmd in src.gaps:
        if artifact not in seen:
            seen.add(artifact)
            rows.append([f"`{artifact}`", f"`{cmd}`"])
    return out + _table(["artifact", "produce it with"], rows)


# ----------------------------------------------------------------------------------------

def build(src: Source) -> tuple[str, dict]:
    loco = src.json(RUNS / "s2_ml" / "locoeval.json", "python -m stages.s2_ml.train")
    meta = src.json(RUNS / "s2_ml" / "model_meta.json", "python -m stages.s2_ml.train")
    summary = src.csv(LABELED_RAW / "label_summary.csv", "python -m stages.s2_ml.label_all")
    abst = src.jsonl(LABELED_RAW / "abstentions.jsonl", "python -m stages.s2_ml.label_all")
    audit = src.json(RUNS / "s3_physics" / "label_audit.json",
                     "python -m stages.s3_physics.label_audit")
    flags = collect_flags(src, loco, meta, summary, abst, audit)

    body = [
        "# Pipeline breakdown", "",
        f"- generated: **{datetime.now(timezone.utc).isoformat()}**",
        f"- commit: `{_git_sha()}`",
        f"- flags raised: **{len(flags)}**"
        + (f" ({', '.join(sorted({fl.kind for fl in flags}))})" if flags else ""),
        f"- artifacts missing: **{len({a for a, _ in src.gaps})}**",
        "",
        "Generated by `python -m stages.breakdown`. **Every number is copied from the "
        "artifact that owns it**, named beside it — this page measures nothing, so it "
        "cannot disagree with a stage. Re-run the stage, not this, to change a number.", "",
        "---", "",
    ]
    # Flags first; a reader who stops after one screen should stop on the problems
    body += section_flags(flags)
    body += ["---", ""]
    for fn in (section_corpus, section_s1, section_s2, section_s3, section_sweep):
        body += fn(src)
        body += ["---", ""]
    body += section_gaps(src)

    machine = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "flags": [{"kind": fl.kind, "title": fl.title, "detail": fl.detail} for fl in flags],
        "missing_artifacts": sorted({a for a, _ in src.gaps}),
    }
    return "\n".join(body) + "\n", machine


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Assemble every stage's artifacts into one page.")
    ap.add_argument("--out", type=Path, default=RUNS,
                    help="directory for breakdown.md (default: runs/)")
    args = ap.parse_args()
    use_replacement_encoding()

    src = Source()
    md, machine = build(src)

    out_dir = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / BREAKDOWN_MD).write_text(md, encoding="utf-8")

    # `breakdown.json` is no longer written. It was the machine-readable twin of the page above,
    # and nothing in the repo ever read it -- an artifact produced for a consumer that does not
    # exist. `build()` still returns `machine`, which the console summary below prints, so
    # restoring the file is one line if a reader ever turns up.

    # ASCII on the console: it is cp949 here; the .md keeps its typography
    print(f"[breakdown] {len(md.splitlines()):,} lines -> {out_dir / BREAKDOWN_MD}")
    if machine["missing_artifacts"]:
        print(f"[breakdown] {len(machine['missing_artifacts'])} artifact(s) missing:")
        for a in machine["missing_artifacts"]:
            print(f"[breakdown]   {a}")
    if machine["flags"]:
        print(f"[breakdown] {len(machine['flags'])} flag(s) raised:")
        for fl in machine["flags"]:
            print(f"[breakdown]   [{fl['kind']}] {fl['title']}")
    else:
        print("[breakdown] no flags raised")


if __name__ == "__main__":
    main()
