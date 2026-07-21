# schema resolution: resolve columns by name, never by position
# the same index carries different columns across header shapes (e.g. index 47 is `loco`
# in most files, `Step` in others), so name is the only safe key -- this is the one place
# that decides what a column *is*

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from stages.s1_clean.config import (
    COLUMN_PREFIX_PATTERN,
    FAMILY_MARKERS,
    FAMILY_UNKNOWN,
    LABEL_COLUMNS,
    LEGACY_ALGO_COLUMNS,
    ROLE_BY_NAME,
)

_PREFIX = re.compile(COLUMN_PREFIX_PATTERN)


# '47_loco' -> 'loco', '28_L LC' -> 'L LC'; whitespace normalized
def strip_prefix(raw_column: str) -> str:
    return _PREFIX.sub("", raw_column.strip()).strip()


# first row only; trailing empty fields tolerated and dropped
def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        row = next(csv.reader(fh))
    while row and row[-1].strip() == "":
        row.pop()
    return row


# which product this file belongs to; contracts are per family, never global
def family_of(names: list[str]) -> str:
    for family, marker in FAMILY_MARKERS.items():
        if marker in names:
            return family
    return FAMILY_UNKNOWN


# what a single file's header actually contains, by name
@dataclass
class Resolution:
    family: str # product family
    n_cols: int # column count
    index_by_name: dict[str, int] # name -> position
    roles_present: dict[str, list[str]] # role -> [names]
    unknown_names: list[str] # names with no registered role
    label_columns: list[str] # label columns present
    legacy_algo_columns: list[str] # legacy loco/step columns present


# map a header to roles by name; absence is a fact, not an error
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
        family=family_of(names),
        n_cols=len(names),
        index_by_name=index_by_name,
        roles_present=roles,
        unknown_names=unknown,
        label_columns=[n for n in LABEL_COLUMNS if n in index_by_name],
        legacy_algo_columns=[n for n in LEGACY_ALGO_COLUMNS if n in index_by_name],
    )
