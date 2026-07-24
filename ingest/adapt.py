# ingest/adapt.py -- turn an arbitrary sheet into a file this pipeline can read.
#
# Two targets, both defined by dataset_profile / transform (never re-hardcoded here):
#
#   labeled  a labeled trial for TRAINING: data/labeled/<rev>/annotated_loco_<rev>_trial_<n>.csv
#            columns Time + the four rev2 features + Label (canonical STAND/WALK/unknown codes).
#
#   raw      an unlabeled recording for SCORING: data/raw/<YYYYMMDD>/<name>.csv, holding the
#            five columns the verified raw->feature bridge reads (transform.raw_to_features):
#            Time + the left/right SAGITTAL angle (_Deg_Y) and its rate (_Gyro_Z). This is the
#            minimal honest contract -- the full S1 clean stage wants the complete 30-channel
#            device schema, but the bridge (and the model) only ever read these sagittal channels.
#
# The mapping names which of your columns feed the target. Anything you do not map is dropped --
# a raw sheet that happens to carry a label column simply keeps its five channels and leaves the
# label behind (a raw/scoring file has no label; the pipeline drops labels from data/raw anyway).
#
# Deterministic: it renames columns and (for labeled) remaps label words to codes, then validates
# the result against the target's own loader -- dataset._read_raw for labeled, raw_to_features for
# raw -- so a file this tool emits can never fail the contract the pipeline enforces. It never
# guesses a value it was not told; --scaffold only proposes a mapping for you to confirm, and the
# agent auto-mapper (ingest/automap.py) likewise only proposes -- this core is the gate both pass.
#
# Typical use:
#   python -m ingest.adapt my_sheet.csv --target raw --scaffold > map.json   # draft from headers
#   # edit map.json: point each canonical column at one of yours
#   python -m ingest.adapt my_sheet.csv --target raw --map map.json          # write the file

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dataset_profile import (
    FEATURES,
    HUMAN_UNKNOWN,
    LABEL_COL,
    MACHINE_UNKNOWN,
    REV_PATTERN,
    TIME_COL,
    TRAIN_CLASSES,
    TRIAL_PATTERN,
)
from stages.s2_ml.transform import SAGITTAL_DEG_AXIS, SAGITTAL_GYRO_AXIS

# the only integer codes allowed in Label after mapping
ALLOWED_CODES = set(TRAIN_CLASSES) | {HUMAN_UNKNOWN, MACHINE_UNKNOWN}
# an 8-digit session date, in a filename or a parent folder name (data/raw/<YYYYMMDD>/...)
_SESSION_RE = re.compile(r"(\d{8})")
# how to encode label words, shown in scaffolds and error messages
STAND_HINT = "0=stand, 10=walk, -1=human-unknown, 255=machine-unknown"


class AdaptError(Exception):
    """A mapping or validation problem stated in terms the caller can act on."""


# --- targets --------------------------------------------------------------------------------
# one target = the canonical columns a written file must carry, whether it has a Label, a short
# human note about what each column means (shown in scaffolds and to the agent mapper), and how
# the output is named/validated.

@dataclass(frozen=True)
class Target:
    name: str
    columns: tuple[str, ...]
    has_label: bool
    column_notes: dict[str, str]
    out_root: Path
    session_named: bool = False   # True: data/raw/<session>/<name>.csv naming (else rev/trial)


LABELED = Target(
    name="labeled",
    columns=(TIME_COL, *FEATURES, LABEL_COL),
    has_label=True,
    column_notes={
        TIME_COL: "device uptime in milliseconds",
        "L_ang_LPF": "left sagittal joint angle (low-pass filtered), degrees",
        "R_ang_LPF": "right sagittal joint angle (low-pass filtered), degrees",
        "L_angvel_LPF": "left sagittal angular velocity (low-pass filtered), deg/s",
        "R_angvel_LPF": "right sagittal angular velocity (low-pass filtered), deg/s",
        LABEL_COL: "human standing/walking annotation",
    },
    out_root=Path("data") / "labeled",
)

# the raw bridge reads the sagittal Deg axis as the ANGLE and the permuted Gyro axis as its RATE
_L_DEG, _R_DEG = f"L_Deg_{SAGITTAL_DEG_AXIS}", f"R_Deg_{SAGITTAL_DEG_AXIS}"
_L_GYRO, _R_GYRO = f"L_Gyro_{SAGITTAL_GYRO_AXIS}", f"R_Gyro_{SAGITTAL_GYRO_AXIS}"

RAW = Target(
    name="raw",
    columns=(TIME_COL, _L_DEG, _R_DEG, _L_GYRO, _R_GYRO),
    has_label=False,
    column_notes={
        TIME_COL: "device uptime in milliseconds",
        _L_DEG: "left sagittal joint ANGLE (raw, unfiltered), degrees",
        _R_DEG: "right sagittal joint ANGLE (raw, unfiltered), degrees",
        _L_GYRO: "left sagittal ANGULAR VELOCITY (raw gyro), deg/s",
        _R_GYRO: "right sagittal ANGULAR VELOCITY (raw gyro), deg/s",
    },
    out_root=Path("data") / "raw",
    session_named=True,
)

TARGETS = {t.name: t for t in (LABELED, RAW)}


# --- reading the input ----------------------------------------------------------------------

# read a sheet as a DataFrame. header whitespace is stripped so " Time" and "Time" are one column
def load_sheet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise AdaptError(f"no such file: {path}")
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


# --- mapping --------------------------------------------------------------------------------

# collapse a header to letters+digits, lowercased, for a loose compare in the scaffold guesser
def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


# best-effort guess of which source header feeds a canonical column: exact (case-insensitive),
# then normalized-equal, then normalized-substring. returns "" when nothing is close
def _guess_source(canonical: str, headers: list[str]) -> str:
    for h in headers:
        if h.lower() == canonical.lower():
            return h
    cn = _norm(canonical)
    for h in headers:
        if _norm(h) == cn:
            return h
    for h in headers:
        if cn and (cn in _norm(h) or _norm(h) in cn):
            return h
    return ""


# a mapping skeleton drafted from the sheet's headers for one target: each canonical column
# pointed at a guessed source column (or "" to fill in), plus -- for a labeled target -- the
# label column's distinct values to encode. every header is echoed under _available_columns
def scaffold(df: pd.DataFrame, target: Target = LABELED) -> dict:
    headers = list(df.columns)
    columns = {c: _guess_source(c, headers) for c in target.columns}
    out: dict = {
        "_available_columns": headers,
        "_columns_mean": dict(target.column_notes),
        "_note": "point each 'columns' entry at one of _available_columns; delete the _ keys.",
        "columns": columns,
    }
    if target.has_label:
        label_src = columns.get(LABEL_COL) or ""
        values = df[label_src].dropna().unique() if label_src in df.columns else []
        out["label_values"] = {str(v): 0 for v in sorted(values, key=str)}
        out["_note"] += f" encode each 'label_values' entry as {STAND_HINT}."
    return out


# rename the mapped source columns to their canonical names and keep only the target's columns
# (anything unmapped is dropped). a canonical name absent from the mapping is assumed to already
# be present under that name
def _apply_columns(df: pd.DataFrame, columns: dict[str, str], target: Target) -> pd.DataFrame:
    rename = {}
    for canonical in target.columns:
        src = columns.get(canonical, canonical)
        if src not in df.columns:
            raise AdaptError(
                f"column '{canonical}' ({target.column_notes.get(canonical, '')}) maps to "
                f"'{src}', which is not in the sheet. available columns: "
                f"{', '.join(map(str, df.columns))}"
            )
        rename[src] = canonical
    return df.rename(columns=rename)[list(target.columns)].copy()


# turn the Label column into canonical integer codes. with label_values every distinct value
# must be listed (unmapped values fail loudly); without it the values must already be codes
def _apply_labels(df: pd.DataFrame, label_values: dict | None) -> pd.DataFrame:
    df = df.copy()
    if label_values:
        keyed = {str(k): v for k, v in label_values.items()}
        missing = sorted({str(v) for v in df[LABEL_COL].dropna().unique()} - set(keyed))
        if missing:
            raise AdaptError(
                f"label value(s) {missing} have no entry in label_values. add them ({STAND_HINT})."
            )
        df[LABEL_COL] = df[LABEL_COL].map(lambda v: keyed.get(str(v)))
    bad = sorted(set(df[LABEL_COL].dropna().unique()) - ALLOWED_CODES)
    if bad:
        raise AdaptError(
            f"Label holds code(s) {bad} outside the allowed set {sorted(ALLOWED_CODES)}. "
            f"map them through 'label_values' first."
        )
    df[LABEL_COL] = df[LABEL_COL].round().astype(int)
    return df


# full mapping for one target: rename columns (dropping anything unmapped), encode labels
# (labeled only), coerce numerics, and prove the result loads through the target's contract check
def apply_mapping(df: pd.DataFrame, mapping: dict, target: Target = LABELED) -> pd.DataFrame:
    out = _apply_columns(df, mapping.get("columns", {}), target)
    if target.has_label:
        out = _apply_labels(out, mapping.get("label_values"))
    numeric_cols = [c for c in target.columns if not (target.has_label and c == LABEL_COL)]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(out[col]):
            coerced = pd.to_numeric(out[col], errors="coerce")
            if coerced.isna().all():
                raise AdaptError(f"column '{col}' is not numeric and could not be parsed as numbers.")
            out[col] = coerced
    if out.empty:
        raise AdaptError("the mapped sheet has no rows.")
    _contract_check(out, target)
    return out


# the target's own loader is the final gate: labeled goes through dataset._read_raw, raw through
# the verified raw_to_features bridge. if either rejects the frame, the mapping was wrong
def _contract_check(df: pd.DataFrame, target: Target) -> None:
    if target is RAW:
        from stages.s2_ml.transform import TRUST_UNCHECKED, raw_to_features
        feats = raw_to_features(df, trust=TRUST_UNCHECKED)
        if feats[list(FEATURES)].isna().all().any():
            raise AdaptError("the raw bridge produced empty features -- check the angle/rate columns.")
    else:
        if list(df.columns) != list(target.columns):
            raise AdaptError(f"mapped columns {list(df.columns)} != contract {list(target.columns)}.")


# --- naming and writing ---------------------------------------------------------------------

def infer_rev(name: str) -> str | None:
    m = re.search(REV_PATTERN, name)
    return m.group(1) if m else None


def infer_trial(name: str) -> int | None:
    m = re.search(TRIAL_PATTERN, name)
    return int(m.group(1)) if m else None


def infer_session(*names: str) -> str | None:
    for n in names:
        m = _SESSION_RE.search(n)
        if m:
            return m.group(1)
    return None


# the canonical destination for a labeled trial
def trial_path(out_dir: Path, rev: str, trial: int) -> Path:
    return out_dir / rev / f"annotated_loco_{rev}_trial_{trial}.csv"


# the canonical destination for a raw recording: data/raw/<session>/<name>.csv
def raw_path(out_dir: Path, session: str, name: str) -> Path:
    return out_dir / session / f"{name}.csv"


# write the mapped frame to `dest`, then re-read it through the target's contract check so a
# file this tool emits can never fail the loader the pipeline uses
def write_output(df: pd.DataFrame, dest: Path, target: Target) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    if target.session_named:          # a raw scoring file: must bridge to features cleanly
        _contract_check(pd.read_csv(dest), target)
    else:                             # a labeled trial: must pass the pipeline's own loader
        from stages.s2_ml.dataset import _read_raw
        _read_raw(dest)
    return dest


# kept for callers/tests that write a labeled trial directly
def write_trial(df: pd.DataFrame, rev: str, trial: int, out_dir: Path) -> Path:
    return write_output(df, trial_path(out_dir, rev, trial), LABELED)


# --- CLI ------------------------------------------------------------------------------------

# resolve the destination path for a target from flags / mapping / the input filename
def resolve_dest(target: Target, mapping: dict, args, input_path: Path) -> Path:
    out_dir = args.out or target.out_root
    if target.session_named:
        session = args.session or mapping.get("session") or infer_session(input_path.name, input_path.parent.name)
        name = args.name or mapping.get("name") or input_path.stem
        if not session:
            raise AdaptError(
                "could not determine the session date. pass --session YYYYMMDD, put it in the "
                "mapping, or name the input/folder with an 8-digit date (e.g. 20260114)."
            )
        return raw_path(out_dir, session, name)
    rev = args.rev or mapping.get("rev") or infer_rev(input_path.name)
    trial = args.trial if args.trial is not None else mapping.get("trial")
    if trial is None:
        trial = infer_trial(input_path.name)
    if not rev or trial is None:
        raise AdaptError(
            "could not determine rev and trial. pass --rev revN --trial N, put them in the "
            "mapping, or name the input so they can be read (e.g. ..._rev3_trial_2.csv)."
        )
    return trial_path(out_dir, rev, int(trial))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Map an arbitrary sheet into a labeled trial or a raw recording this pipeline can read.")
    ap.add_argument("sheet", type=Path, help="the input .csv to adapt")
    ap.add_argument("--target", choices=sorted(TARGETS), required=True,
                    help="labeled trial (training) or raw recording (scoring) -- you must choose; "
                         "the tool never guesses which one a sheet is")
    ap.add_argument("--map", type=Path, help="mapping JSON (see module docstring)")
    ap.add_argument("--scaffold", action="store_true",
                    help="print a mapping template drafted from the sheet's headers, then exit")
    ap.add_argument("--rev", help="[labeled] subject id, e.g. rev1")
    ap.add_argument("--trial", type=int, help="[labeled] trial number")
    ap.add_argument("--session", help="[raw] session date YYYYMMDD")
    ap.add_argument("--name", help="[raw] output filename stem (default: the input's)")
    ap.add_argument("--out", type=Path, help="output root (default data/labeled or data/raw)")
    ap.add_argument("--dry-run", action="store_true", help="validate and report without writing")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    target = TARGETS[args.target]
    try:
        df = load_sheet(args.sheet)

        if args.scaffold:
            print(json.dumps(scaffold(df, target), indent=2, ensure_ascii=True))
            return

        mapping = json.loads(args.map.read_text(encoding="utf-8")) if args.map else {}
        mapped = apply_mapping(df, mapping, target)
        dest = resolve_dest(target, mapping, args, args.sheet)

        if args.dry_run:
            print(f"[adapt] OK ({target.name}): {len(mapped)} rows -> would write {dest}")
            if target.has_label:
                counts = mapped[LABEL_COL].value_counts().sort_index()
                print("[adapt] label codes: " +
                      ", ".join(f"{int(k)}={int(v)}" for k, v in counts.items()))
            return

        write_output(mapped, dest, target)
        print(f"[adapt] wrote {len(mapped)} rows -> {dest}")
    except AdaptError as exc:
        sys.exit(f"[adapt] {exc}")


if __name__ == "__main__":
    main()
