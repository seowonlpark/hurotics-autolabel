# S2 replay: reconstruct a logged experiment from the ledger and re-run it
# the PLAN S2 gate ("reconstructible from experiments.jsonl + git history alone") made
# executable. deterministic code + fixed data => a same-HEAD entry must replay exactly;
# an older-sha entry may differ (that's why the ledger stores the sha). a mismatch on a
# same-sha entry means the ledger isn't a faithful revert unit. see README / PLAN.

from __future__ import annotations

import argparse
from pathlib import Path

from runmeta import git_sha
from stages.s2_ml.dataset import load_dataset
from stages.s2_ml.experiment import (
    ExperimentSpec,
    ledger,
    run_experiment,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
S2_RUN_DIR = REPO_ROOT / "runs" / "s2_ml"

# float-roundoff tolerance only, not "close enough"; above it on a same-sha entry is
# genuine non-reproducibility, not noise
EXACT_TOL = 1e-9


# rebuild the spec from a ledger entry -- all the entry stores about how the model was made
def reconstruct_spec(entry: dict) -> ExperimentSpec:
    s = entry["spec"]
    return ExperimentSpec(
        name=s["name"],
        rationale=s["rationale"],
        drop_features=list(s.get("drop_features") or []),
        window_s=s.get("window_s"),
        stride_s=s.get("stride_s"),
        model_params=dict(s.get("model_params") or {}),
    )


# (field, recorded, replayed, abs_delta) for each scalar metric
def _compare_scalars(recorded: dict, got: dict) -> list[tuple[str, float, float, float]]:
    rows = []
    for field in ("macro_f1", "accuracy", "balanced_accuracy"):
        if field in recorded:
            r, g = float(recorded[field]), float(got[field])
            rows.append((field, r, g, abs(r - g)))
    return rows


# same comparison for a dict of metrics, keys prefixed; missing key => inf delta
def _compare_dict(recorded: dict, got: dict, prefix: str
                  ) -> list[tuple[str, float, float, float]]:
    rows = []
    for k in sorted(recorded):
        if k not in got:
            rows.append((f"{prefix}.{k}", float(recorded[k]), float("nan"), float("inf")))
            continue
        r, g = float(recorded[k]), float(got[k])
        rows.append((f"{prefix}.{k}", r, g, abs(r - g)))
    return rows


# replay one entry, return a verdict dict; never raises on a mismatch (that's the finding)
def replay_entry(entry: dict, trials, tol: float = EXACT_TOL) -> dict:
    spec = reconstruct_spec(entry)
    want_tax = "taxonomy" in entry
    result = run_experiment(spec, trials, taxonomy=want_tax)
    got = result.to_dict()

    diffs = _compare_scalars(entry, got)
    if "n_features" in entry:
        r, g = int(entry["n_features"]), int(got["n_features"])
        diffs.append(("n_features", r, g, abs(r - g)))
    if "per_rev_macro_f1" in entry:
        diffs += _compare_dict(entry["per_rev_macro_f1"],
                               got.get("per_rev_macro_f1", {}), "per_rev")
    if want_tax and "fractions" in entry["taxonomy"]:
        diffs += _compare_dict(entry["taxonomy"]["fractions"],
                               got.get("taxonomy", {}).get("fractions", {}), "tax")
        if "row_accuracy" in entry["taxonomy"]:
            r = float(entry["taxonomy"]["row_accuracy"])
            g = float(got["taxonomy"]["row_accuracy"])
            diffs.append(("tax.row_accuracy", r, g, abs(r - g)))

    worst = max((d[3] for d in diffs), default=0.0)
    return {
        "name": entry["spec"]["name"],
        "recorded_sha": entry.get("git_sha", "unknown"),
        "exact": worst <= tol,
        "worst_delta": worst,
        "diffs": diffs,
    }


# replay the selected entries; fail if a same-sha entry doesn't reproduce
def main() -> None:
    parser = argparse.ArgumentParser(description="Replay logged S2 experiments.")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="replay every ledger entry")
    g.add_argument("--name", help="replay a single entry by spec name")
    parser.add_argument("--tol", type=float, default=EXACT_TOL,
                        help=f"max abs metric delta to call exact (default {EXACT_TOL})")
    parser.add_argument("--run-dir", type=Path, default=S2_RUN_DIR)
    args = parser.parse_args()

    rows = ledger(args.run_dir)
    if not rows:
        raise SystemExit(f"no ledger at {args.run_dir / 'experiments.jsonl'}")
    if args.name:
        rows = [e for e in rows if e["spec"]["name"] == args.name]
        if not rows:
            raise SystemExit(f"no ledger entry named {args.name!r}")

    head = git_sha()
    print(f"HEAD is {head}. Deterministic code + fixed data => same-sha entries must "
          f"replay exactly (tol {args.tol:g}).\n")

    trials = load_dataset() # load once; every entry re-windows it internally
    all_ok = True
    for entry in rows:
        v = replay_entry(entry, trials, args.tol)
        same_sha = v["recorded_sha"] == head
        # a same-sha entry MUST be exact; an older-sha entry is only informative
        ok = v["exact"] if same_sha else True
        all_ok = all_ok and ok
        tag = "EXACT" if v["exact"] else f"DIFF (worst {v['worst_delta']:.2e})"
        sha_note = "" if same_sha else f"  [recorded at {v['recorded_sha']} != HEAD; " \
                                       f"checkout that sha for a strict replay]"
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {v['name']:32} {tag}{sha_note}")
        # show the offending fields whenever it is not exact, sha aside
        if not v["exact"]:
            for field, r, g, d in v["diffs"]:
                if d > args.tol:
                    print(f"         {field:24} recorded={r:.6f} replayed={g:.6f} "
                          f"delta={d:.2e}")

    print()
    if all_ok:
        print("RECONSTRUCTIBLE: every same-HEAD entry replayed exactly from its spec. "
              "Older-sha entries need `git checkout <sha>` for a strict replay (expected).")
    else:
        raise SystemExit("NON-REPRODUCIBLE: a same-sha entry did not replay - the ledger "
                         "is not a faithful revert unit. See the DIFF rows above.")


if __name__ == "__main__":
    main()
