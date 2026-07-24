# ingest/automap.py -- let an agent PROPOSE the column mapping, then let code validate it.
#
# ingest/adapt.py needs a mapping (which of your columns is Time, which is the left angle, how
# your label words encode to STAND/WALK). Writing it by hand is the honest default; this module
# is the assisted path: it shows an agent the target contract, your sheet's headers, a sample of
# rows, and the distinct values of each low-cardinality column, and asks for the mapping JSON.
#
# The agent only PROPOSES. Its reply is run through ingest.adapt.apply_mapping -- the exact same
# deterministic gate a hand-written mapping passes -- so a wrong guess fails loudly (unmapped
# label value, mis-pointed column) rather than writing a mislabeled file. Same rule as every
# other agent here: the model judges, code decides. It never touches your data.
#
#   python -m ingest.automap my_sheet.csv --target raw            # print the proposed mapping
#   python -m ingest.automap my_sheet.csv --target raw --write    # propose, validate, and write

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from agents.base import MODEL_SMART, AgentSpec, extract_json_object, run_agent
from ingest.adapt import (
    LABELED, STAND_HINT, TARGETS, AdaptError, Target, apply_mapping, load_sheet,
    resolve_dest, write_output,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"
# distinct values are listed for a column only when it has at most this many -- enough to expose
# a label column's classes, not so many that a continuous sensor channel floods the prompt
MAX_DISTINCT_LISTED = 30

SYSTEM_PROMPT = (
    "You map the columns of an arbitrary sensor spreadsheet onto a FIXED target schema so a "
    "data pipeline can read it. You are given the target's canonical columns and what each one "
    "means, the source sheet's headers, a sample of its rows, and the distinct values of its "
    "low-cardinality columns.\n\n"
    "Return ONE JSON object and nothing else:\n"
    '  {"columns": {"<canonical>": "<source header>", ...}, '
    '"label_values": {"<source value>": <code>, ...}}\n\n'
    "Rules:\n"
    "- Every source header you name MUST be one of the provided headers. Never invent a column.\n"
    "- Map every canonical column. Use the meanings to disambiguate (e.g. an ANGLE column vs an "
    "angular-VELOCITY column; left vs right).\n"
    "- Include 'label_values' ONLY if the target has a Label column. Then map EVERY distinct "
    f"value of the chosen label column to a code ({STAND_HINT}).\n"
    "- If you are unsure, still make your single best choice; do not leave a canonical column out."
)


# a compact, one-off run dir for this mapping call's prompt/log/cost audit trail
def new_run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return RUNS_DIR / "ingest_map" / stamp


# the distinct values of each column with few enough of them to be worth showing (a label column
# and other categoricals); continuous sensor channels are skipped
def _low_cardinality_values(df: pd.DataFrame) -> dict[str, list]:
    out = {}
    for col in df.columns:
        vals = df[col].dropna().unique()
        if 0 < len(vals) <= MAX_DISTINCT_LISTED:
            out[col] = sorted((str(v) for v in vals))
    return out


# the user prompt: the concrete evidence the agent maps over
def build_prompt(df: pd.DataFrame, target: Target) -> str:
    schema = "\n".join(f"  - {c}: {target.column_notes.get(c, '')}" for c in target.columns)
    sample = df.head(8).to_csv(index=False).strip()
    distinct = _low_cardinality_values(df)
    distinct_block = ("\n".join(f"  - {c}: {vals}" for c, vals in distinct.items())
                      or "  (none -- every column is continuous)")
    label_line = ("\nThis target HAS a Label column: include 'label_values'."
                  if target.has_label else
                  "\nThis target has NO Label column: omit 'label_values'.")
    return (
        f"TARGET '{target.name}' -- map onto these canonical columns:\n{schema}\n"
        f"{label_line}\n\n"
        f"SOURCE headers: {list(df.columns)}\n\n"
        f"SOURCE sample rows (CSV):\n{sample}\n\n"
        f"SOURCE distinct values (low-cardinality columns only):\n{distinct_block}\n\n"
        "Return the mapping JSON now."
    )


def _agent(model: str) -> AgentSpec:
    # no tools: the agent reasons over the prompt, it does not read or write files. domain notes
    # are irrelevant to column matching, so only the always-on orientation section is injected.
    return AgentSpec(name="ingest_mapper", system_prompt=SYSTEM_PROMPT,
                     allowed_tools=[], model=model, domain_sections=())


# propose a mapping for `df` against `target`; returns (mapping_or_None, raw_reply). does not
# validate -- automap() does that so the failure carries the same message a hand-map would
async def propose(df: pd.DataFrame, target: Target, run_dir: Path,
                  model: str = MODEL_SMART) -> tuple[dict | None, str]:
    res = await run_agent(_agent(model), build_prompt(df, target), run_dir)
    return extract_json_object(res.final_text), res.final_text


# the full assisted flow: propose, then validate through the deterministic gate. raises
# AdaptError (the same one a bad hand-mapping raises) if the proposal does not hold up, after
# saving the raw reply next to the run's audit trail so the failure is inspectable
def automap(df: pd.DataFrame, target: Target, run_dir: Path,
            model: str = MODEL_SMART) -> dict:
    mapping, raw = asyncio.run(propose(df, target, run_dir, model))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "proposed_mapping_raw.txt").write_text(raw or "", encoding="utf-8")
    if mapping is None:
        raise AdaptError("the agent's reply did not contain a JSON mapping (see "
                         f"{run_dir / 'proposed_mapping_raw.txt'}).")
    apply_mapping(df, mapping, target)  # the gate: raises if the proposal is wrong
    (run_dir / "proposed_mapping.json").write_text(
        json.dumps(mapping, indent=2, ensure_ascii=True), encoding="utf-8")
    return mapping


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Have an agent propose the ingest mapping for a sheet, then validate it.")
    ap.add_argument("sheet", type=Path, help="the input .csv to map")
    ap.add_argument("--target", choices=sorted(TARGETS), required=True,
                    help="labeled trial (training) or raw recording (scoring) -- you must choose; "
                         "the tool never guesses which one a sheet is")
    ap.add_argument("--model", default=MODEL_SMART, help="model alias (default: the smart model)")
    ap.add_argument("--write", action="store_true",
                    help="after validating, write the file (else just print the mapping)")
    ap.add_argument("--rev", help="[labeled] subject id, e.g. rev1")
    ap.add_argument("--trial", type=int, help="[labeled] trial number")
    ap.add_argument("--session", help="[raw] session date YYYYMMDD")
    ap.add_argument("--name", help="[raw] output filename stem (default: the input's)")
    ap.add_argument("--out", type=Path, help="output root (default data/labeled or data/raw)")
    return ap


def main() -> None:
    for stream in (sys.stdout, sys.stderr):  # agent text may carry cp949-unencodable chars
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    load_dotenv(REPO_ROOT / ".env")  # the SDK reads ANTHROPIC_API_KEY from the environment

    args = build_parser().parse_args()
    target = TARGETS[args.target]
    run_dir = new_run_dir()
    try:
        df = load_sheet(args.sheet)
        print(f"[automap] proposing a '{target.name}' mapping for {args.sheet.name} "
              f"({len(df.columns)} columns) -> {run_dir}")
        mapping = automap(df, target, run_dir, args.model)
        print(json.dumps(mapping, indent=2, ensure_ascii=True))

        if args.write:
            mapped = apply_mapping(df, mapping, target)
            dest = resolve_dest(target, mapping, args, args.sheet)
            write_output(mapped, dest, target)
            print(f"[automap] wrote {len(mapped)} rows -> {dest}")
        else:
            print("[automap] validated. re-run with --write to apply it, or feed this JSON to "
                  "`python -m ingest.adapt ... --map`.")
    except AdaptError as exc:
        sys.exit(f"[automap] {exc}")


if __name__ == "__main__":
    main()
