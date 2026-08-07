# the corpus DECLARED: which recordings exist, whose they are, which are withheld and why (§1.6)

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = REPO_ROOT / "data" / "corpus.json"

# bumped when a field changes meaning; a manifest from another schema is refused, never guessed at
SCHEMA = 1

# two, not three: held-out eval is grouped CV over the training subjects, so nothing else is carved
SPLITS = ("train", "lockbox")


# one annotated recording and everything the pipeline needs that is not IN the file
@dataclass(frozen=True)
class Entry:
    subject: str          # the CV group; nothing parses this out of a path any more
    session: int          # distinguishes one subject's recordings, no ordering implied
    annotated: Path       # the lpf_view export carrying human labels
    raw: Path | None      # the device log it came from, where that still exists
    variant: str | None   # `transform.fingerprint` of the raw header, re-derived and checked on load
    split: str
    exclude: str | None   # the REASON, not a flag- a quarantine nobody can audit is just a deletion

    @property
    def key(self) -> tuple[str, int]:
        return (self.subject, self.session)


def _fail(path: Path, problem: str) -> None:
    raise SystemExit(
        f"[corpus] {path.name} is not a usable manifest:\n  {problem}\n"
        f"[corpus] This file is the only authority on which recordings exist and how they group. "
        f"Fix it rather than working around it -- every split, fold and pairing below reads it."
    )


# every entry, validated; an unreadable manifest is fatal, because guessing is what it replaced
def load_manifest(path: Path = CORPUS_PATH) -> list[Entry]:
    if not path.exists():
        _fail(path, f"no such file at {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _fail(path, f"not valid JSON: {exc}")
    if raw.get("schema") != SCHEMA:
        _fail(path, f"schema {raw.get('schema')!r}, this code reads schema {SCHEMA}")

    entries, seen = [], {}
    for i, row in enumerate(raw.get("trials", [])):
        where = f"trials[{i}]"
        for field in ("subject", "session", "annotated", "split"):
            if row.get(field) in (None, ""):
                _fail(path, f"{where}: missing required field {field!r}")
        if row["split"] not in SPLITS:
            _fail(path, f"{where}: split {row['split']!r} is not one of {list(SPLITS)}")
        ann = REPO_ROOT / row["annotated"]
        if not ann.exists():
            _fail(path, f"{where}: annotated file does not exist: {row['annotated']}")
        # raw is unchecked here- `data/raw` is gitignored and train never opens it; `pairs` checks
        rawp = (REPO_ROOT / row["raw"]) if row.get("raw") else None

        e = Entry(subject=str(row["subject"]), session=int(row["session"]), annotated=ann,
                  raw=rawp, variant=row.get("variant"), split=row["split"],
                  exclude=row.get("exclude") or None)
        # (subject, session) is the identity of a recording here, so a repeat is a corpus error
        if e.key in seen:
            _fail(path, f"{where}: duplicate (subject, session) {e.key}, already at {seen[e.key]}")
        seen[e.key] = where
        entries.append(e)

    if not entries:
        _fail(path, "no trials declared")
    if not [e for e in entries if e.split == "train" and not e.exclude]:
        _fail(path, "every trial is excluded or sealed; nothing would train")
    # sorted by key so fold order, and therefore every artifact below, is reproducible
    return sorted(entries, key=lambda e: e.key)


# the usable corpus; `include_excluded` loads the quarantine too, as the auditors deliberately do
def trials(include_excluded: bool = False, path: Path = CORPUS_PATH) -> list[Entry]:
    return [e for e in load_manifest(path) if include_excluded or not e.exclude]


# entries with a raw device log recorded- NOT a search, the manifest declares the pairing
def pairs(include_excluded: bool = True, path: Path = CORPUS_PATH) -> list[Entry]:
    declared = [e for e in trials(include_excluded, path) if e.raw is not None]
    # a missing file is refused, not skipped- skipping shrinks every raw-route denominator silently
    if (gone := [e for e in declared if not e.raw.exists()]):
        raise SystemExit(
            f"[corpus] {len(gone)} declared raw file(s) are missing, first "
            f"{gone[0].key}: {gone[0].raw}\n[corpus] `data/raw` is gitignored, so a fresh "
            f"clone has none of it. Fetch it, or drop the `raw` field for those rows."
        )
    return declared


# (frame, variant) for one entry, with the RECORDED variant re-derived and enforced
def load_raw(entry: Entry):
    from stages.s2_ml.transform import load_raw_frame

    if entry.raw is None:
        raise SystemExit(f"[corpus] {entry.key} declares no raw file")
    df, vid, _family = load_raw_frame(entry.raw)
    # the axis map is chosen from this fingerprint, so a swapped file would be scored as a guess
    if entry.variant and vid != entry.variant:
        raise SystemExit(
            f"[corpus] {entry.raw.name}: header fingerprint {vid!r}, manifest records "
            f"{entry.variant!r}. Re-check the pairing, then update the manifest on purpose."
        )
    return df, vid
