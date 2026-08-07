# artifact freshness: stages stamp a content hash (not mtime) of every upstream artifact read

from __future__ import annotations

import hashlib
import json
from pathlib import Path

INPUTS_FILENAME = "_inputs.json"
_CHUNK = 1 << 20
REPO_ROOT = Path(__file__).resolve().parent


# one stamp per STAGE, not per dir- else the last stage to run vouches for the others' stale reports
def stamp_name(stage: str | None) -> str:
    # stage=None keeps the bare `_inputs.json`, so stamps written before the split still read
    return INPUTS_FILENAME if stage is None else f"_inputs.{stage}.json"


def stamp_paths(out_dir: Path) -> list[Path]:
    return sorted(Path(out_dir).glob("_inputs*.json"))


# the inverse of stamp_name: which stage wrote this stamp, or None for a bare pre-stage one
def stage_of(stamp: Path) -> str | None:
    _, _, stage = stamp.stem.partition(".")
    return stage or None


# repo-relative inside the repo, absolute outside: absolute everywhere made stamps machine-specific
def _store(path: Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


# stored name back to a path; an absolute value is honoured as written, so older stamps still read
def _locate(stored: str) -> Path:
    p = Path(stored)
    return p if p.is_absolute() else REPO_ROOT / p


# sha256 + size, or None when absent- None is legitimate: an optional upstream may not exist yet
def artifact_id(path: Path) -> dict | None:
    path = Path(path)
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return {"sha256": h.hexdigest(), "bytes": path.stat().st_size}


# record what this stage consumed; an EMPTY `inputs` DECLARES no stale-able upstream, unlike no stamp
def stamp_inputs(out_dir: Path, inputs: dict[str, Path], stage: str | None = None) -> dict:
    record = {label: {"path": _store(p), **(artifact_id(p) or {"missing": True})}
              for label, p in inputs.items()}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / stamp_name(stage)).write_text(
        json.dumps(record, indent=2), encoding="utf-8")
    return record


# "checked, nothing upstream" rather than "nobody can tell"- what keeps a stage off §A's blind list
def declare_no_inputs(out_dir: Path, stage: str | None = None) -> dict:
    return stamp_inputs(out_dir, {}, stage=stage)


# complaints about one stamp file; `where` names it the way a reader would look it up
def _check_stamp(stamp: Path, where: str) -> list[str]:
    try:
        record = json.loads(stamp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{where}: unreadable {stamp.name} ({exc})"]

    out = []
    for label, want in record.items():
        now = artifact_id(_locate(want["path"]))
        if want.get("missing"):
            if now is not None:
                out.append(f"{where}: {label} did not exist when this ran, but does now")
            continue
        if now is None:
            out.append(f"{where}: {label} has since been deleted ({want['path']})")
        elif now["sha256"] != want["sha256"]:
            out.append(f"{where}: {label} changed since this ran -- "
                       f"{where} describes an older {Path(want['path']).name}")
    return out


# how a dir is named in a complaint- `.name` collides: regen/s2_ml and keep/s2_ml are both "s2_ml"
def _where(out_dir: Path) -> str:
    try:
        return Path(out_dir).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(out_dir)


# complaints about `out_dir`; an ABSENT stamp reports the same as stale- the caller decides
def check_inputs(out_dir: Path) -> list[str]:
    out_dir = Path(out_dir)
    where = _where(out_dir)
    stamps = stamp_paths(out_dir)
    if not stamps:
        return [f"{where}: no {INPUTS_FILENAME}; cannot tell which inputs produced it"]
    # each stage is checked on its own record, so one fresh run cannot vouch for another's report
    out = []
    for s in stamps:
        stage = stage_of(s)
        out += _check_stamp(s, where if stage is None else f"{where} [{stage}]")
    return out


# check several stage directories at once; returns every complaint, in order
def check_all(out_dirs: list[Path]) -> list[str]:
    return [c for d in out_dirs for c in check_inputs(Path(d))]


# ---- self-test: a check that silently stopped firing looks exactly like a clean pipeline ----

def _self_test() -> list[str]:
    import tempfile

    failures: list[str] = []

    def expect(label: str, got: list[str], want_hit: bool) -> None:
        if bool(got) != want_hit:
            failures.append(f"{label}: expected {'a complaint' if want_hit else 'silence'}, "
                            f"got {got or 'silence'}")

    tmp = Path(tempfile.mkdtemp())
    src, out = tmp / "spec.json", tmp / "stage"
    src.write_text('{"v": 1}', encoding="utf-8")

    stamp_inputs(out, {"spec": src})
    expect("a stamp taken just now", check_inputs(out), False)

    src.write_text('{"v": 2}', encoding="utf-8")
    expect("an input whose CONTENT changed", check_inputs(out), True)

    stamp_inputs(out, {"spec": src})
    expect("the same stamp retaken", check_inputs(out), False)

    src.unlink()
    expect("an input since deleted", check_inputs(out), True)

    expect("a directory with no stamp at all", check_inputs(tmp / "never-stamped"), True)

    # an empty stamp reads CLEAN while no stamp complains: declaring nothing is not never declaring
    empty = tmp / "declares-nothing"
    declare_no_inputs(empty)
    expect("a stage that declared no upstream", check_inputs(empty), False)

    # two stages sharing ONE directory- the stale one must still be caught after the fresh one runs
    shared, dep = tmp / "two-stages", tmp / "spec2.json"
    dep.write_text('{"v": 1}', encoding="utf-8")
    stamp_inputs(shared, {"spec": dep}, stage="early")
    dep.write_text('{"v": 2}', encoding="utf-8")
    stamp_inputs(shared, {"spec": dep}, stage="late")
    got = check_inputs(shared)
    expect("a shared directory with one stage stale", got, True)
    if len(got) != 1 or "[early]" not in got[0]:
        failures.append(f"the stale stage must be named, and only it; got {got}")

    (out / INPUTS_FILENAME).write_text("{not json", encoding="utf-8")
    expect("an unreadable stamp", check_inputs(out), True)

    # an input that did not exist is recorded as such, and its later APPEARANCE is a change too
    absent = tmp / "not-yet.json"
    stamp_inputs(out, {"optional": absent})
    expect("an input absent both times", check_inputs(out), False)
    absent.write_text("{}", encoding="utf-8")
    expect("an input that has since APPEARED", check_inputs(out), True)

    # repo-relative in the stamp so a move survives; absolute only for inputs outside the repo
    inside = REPO_ROOT / "freshness.py"
    if _store(inside) != "freshness.py":
        failures.append(f"a repo file should be stored relative, got {_store(inside)!r}")
    if not Path(_store(tmp / "outside.json")).is_absolute():
        failures.append("a file outside the repo must keep its absolute path")
    if _locate("freshness.py") != REPO_ROOT / "freshness.py":
        failures.append("a relative stored path must resolve against the repo root")
    if _locate(str(tmp)) != tmp:
        failures.append("an absolute stored path must be honoured as written")

    # a stamp written before paths were made relative still has to read, or the fix invalidates all
    legacy = tmp / "legacy"
    legacy.mkdir()
    (legacy / INPUTS_FILENAME).write_text(json.dumps(
        {"spec": {"path": str((REPO_ROOT / "freshness.py").resolve()),
                  **artifact_id(REPO_ROOT / "freshness.py")}}), encoding="utf-8")
    expect("a legacy stamp holding an absolute path", check_inputs(legacy), False)

    return failures


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Is every run artifact newer than the inputs it describes?")
    ap.add_argument("--self-test", action="store_true",
                    help="exercise every complaint against a case built to trip it")
    args = ap.parse_args()

    if args.self_test:
        failures = _self_test()
        for f in failures:
            print(f"[freshness] FAIL {f}")
        if failures:
            raise SystemExit(f"[freshness] {len(failures)} check(s) no longer fire")
        print("[freshness] self-test OK: every complaint fires on its own case")
        return

    # every stage dir, not only the stamped ones- selecting on "has a stamp" checks only what can pass
    from runslayout import checkable_dirs  # whose dirs those are is runslayout's call, not this file's

    dirs = checkable_dirs()
    complaints = check_all(dirs)
    for c in complaints:
        print(f"[freshness] {c}")
    print(f"[freshness] {len(dirs)} directory(s) checked, "
          f"{len(complaints)} complaint(s)")


if __name__ == "__main__":
    main()
