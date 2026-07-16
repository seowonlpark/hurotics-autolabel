"""Schema census: fingerprint every distinct header, resolve columns by name.

The corpus has multiple header variants that differ in width AND in meaning at the
same index. This module is the only place that decides what a column *is*.
"""

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
    LEGACY_ALGO_COLUMNS,
    ROLE_BY_NAME,
)

_PREFIX = re.compile(COLUMN_PREFIX_PATTERN)


def strip_prefix(raw_column: str) -> str:
    """'47_loco' -> 'loco'. '28_L LC' -> 'L LC'. Whitespace normalized."""
    return _PREFIX.sub("", raw_column.strip()).strip()


def read_header(path: Path) -> list[str]:
    """First row only. Trailing empty fields are tolerated and dropped."""
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        row = next(csv.reader(fh))
    while row and row[-1].strip() == "":
        row.pop()
    return row


def family_of(names: list[str]) -> str:
    """Which product this file belongs to. Contracts are per family, never global."""
    for family, marker in FAMILY_MARKERS.items():
        if marker in names:
            return family
    return FAMILY_UNKNOWN


@dataclass
class Variant:
    """One distinct header shape seen in the corpus."""

    variant_id: str
    family: str
    n_cols: int
    raw_columns: list[str]
    names: list[str]
    files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "variant_id": self.variant_id,
            "family": self.family,
            "n_cols": self.n_cols,
            "n_files": len(self.files),
            "names": self.names,
            "files": sorted(self.files),
        }


def fingerprint(raw_columns: list[str]) -> str:
    """Stable id for a header. Keyed on names, not positions or widths."""
    joined = "|".join(strip_prefix(c) for c in raw_columns)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:8]


@dataclass
class Resolution:
    """What a single file's header actually contains, by name."""

    variant_id: str
    family: str
    n_cols: int
    index_by_name: dict[str, int]
    roles_present: dict[str, list[str]]   # role -> [names]
    unknown_names: list[str]             # names with no registered role
    label_columns: list[str]
    legacy_algo_columns: list[str]

    def has(self, name: str) -> bool:
        return name in self.index_by_name


def resolve(raw_columns: list[str]) -> Resolution:
    """Map a header to roles by name. Absence is a fact, not an error."""
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
        legacy_algo_columns=[n for n in LEGACY_ALGO_COLUMNS if n in index_by_name],
    )


def build_registry(paths: list[Path], repo_root: Path) -> dict[str, Variant]:
    """One pass over headers only. Cheap enough to run on every file."""
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
                raw_columns=raw,
                names=names,
            )
        registry[vid].files.append(str(p.relative_to(repo_root)))
    return registry


def stable_prefix(registry: dict[str, Variant], family: str | None = None) -> list[str]:
    """Longest run of leading names identical across every variant IN A FAMILY.

    This is the real contract. Measured, not assumed. Computed across families it
    is meaningless: raw device logs and the rev2 view share only Time.
    """
    variants = [v for v in registry.values() if family is None or v.family == family]
    if not variants:
        return []
    name_lists = [v.names for v in variants]
    shortest = min(len(n) for n in name_lists)
    out: list[str] = []
    for i in range(shortest):
        col = {n[i] for n in name_lists}
        if len(col) != 1:
            break
        out.append(col.pop())
    return out
