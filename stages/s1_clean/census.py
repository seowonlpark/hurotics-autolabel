# align on what each col header is

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from stages.s1_clean.config import (
    COLUMN_PREFIX_PATTERN,
    FAMILY_MARKERS,
    FAMILY_UNKNOWN,
    LABEL_COLUMNS,
    ROLE_BY_NAME,
)

_PREFIX = re.compile(COLUMN_PREFIX_PATTERN)

# white space normalized
def strip_prefix(raw_column: str) -> str:
    return _PREFIX.sub("", raw_column.strip()).strip()

# just read the first row for header names
def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        row = next(csv.reader(fh))
    while row and row[-1].strip() == "":
        row.pop()
    return row

# Which of the two products this header is. Load-bearing at serve time, not a report
# field: stages/s2_ml/label.py dispatches on it and transform.py refuses on it. The two
# families share only `Time`, so a wrong answer here resolves columns that do not exist.
#
# Unrelated to `nominal_rate` in resample.py, which snaps a MEASURED RATE to its nominal
# value. Same English word, different question. Keep them apart when reading.
def family_of(names: list[str]) -> str:
    for family, marker in FAMILY_MARKERS.items():
        if marker in names:
            return family
    return FAMILY_UNKNOWN


@dataclass
class Variant:
    variant_id: str
    family: str
    n_cols: int
    names: list[str]
    files: list[str] = field(default_factory=list)


# stable ids
def fingerprint(raw_columns: list[str]) -> str:
    joined = "|".join(strip_prefix(c) for c in raw_columns)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:8]


@dataclass
class Resolution:
    """What one header is.

    Only `index_by_name` (and `label_columns`, one name) is read by code: clean.py
    resolves columns through it, manifest.py picks the label census from it. The rest
    is measured for the manifest ledger and read by humans.

    `roles_present` / `unknown_names` are kept deliberately, despite no code consumer:
    they are the hole detector for ROLE_BY_NAME. A column a new firmware revision adds
    appears in `unknown_names` and nowhere else in the pipeline. `loco` (the outdated
    rule-based algorithm output) lands there too — it has no ROLE_BY_NAME entry, so it
    needs no separate field to stay visible.
    """

    variant_id: str
    family: str
    n_cols: int
    index_by_name: dict[str, int]
    roles_present: dict[str, list[str]]
    unknown_names: list[str]
    label_columns: list[str]


# map header to role by name
def resolve(raw_columns: list[str]) -> Resolution:
    names = [strip_prefix(c) for c in raw_columns]
    index_by_name = {n: i for i, n in enumerate(names)}

    roles: dict[str, list[str]] = {}
    unknown: list[str] = []
    for n in names:
        role = ROLE_BY_NAME.get(n)
        if role is None:
            unknown.append(n)
        else:
            roles.setdefault(role, []).append(n)

    return Resolution(
        variant_id=fingerprint(raw_columns),
        family=family_of(names),
        n_cols=len(names),
        index_by_name=index_by_name,
        roles_present=roles,
        unknown_names=unknown,
        label_columns=[n for n in LABEL_COLUMNS if n in index_by_name],
    )

# pass for all headers
def build_registry(paths: list[Path], repo_root: Path) -> dict[str, Variant]:
    registry: dict[str, Variant] = {}
    for p in paths:
        raw = read_header(p)
        vid = fingerprint(raw)
        if vid not in registry:
            names = [strip_prefix(c) for c in raw]
            registry[vid] = Variant(
                variant_id=vid,
                family=family_of(names),
                n_cols=len(raw),
                names=names,
            )
        registry[vid].files.append(str(p.relative_to(repo_root)))
    return registry

# `stable_prefix` (longest column prefix common to every variant in a family) was removed
# 2026-08-04 along with variants.json, its only output. It described a contract nothing
# enforced: clean.py checks columns against config.KEEP_MEASURED and reports what is
# missing, which is the actual contract. Recomputable from `build_registry` if ever wanted.
