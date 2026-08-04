# artifact freshness: a stage records WHICH VERSION of each upstream artifact it consumed
# runs/ is gitignored and stages overwrite in place, so re-running one alone leaves every
# downstream artifact describing a model that no longer exists, all present and all wrong
# content hash, not mtime: a checkout or a touch moves mtime without changing what was read

from __future__ import annotations

import hashlib
import json
from pathlib import Path

INPUTS_FILENAME = "_inputs.json"
_CHUNK = 1 << 20

REPO_ROOT = Path(__file__).resolve().parent


# repo-relative with forward slashes inside the repo, absolute outside
# absolute everywhere made the stamp machine-specific: moving the repo turned every
# input into "has since been deleted", a false stale flag on an unchanged corpus
def _store(path: Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


# the reverse: a stored name back to a path on this machine; an absolute value is honoured as
# written, which is what keeps stamps made before this change readable rather than making the
# fix itself the thing that invalidates them
def _locate(stored: str) -> Path:
    p = Path(stored)
    return p if p.is_absolute() else REPO_ROOT / p


# sha256 + size of one file, or None when it does not exist; None is a legitimate answer, not an
# error: a stage may legitimately run before an optional upstream artifact is ever produced
def artifact_id(path: Path) -> dict | None:
    path = Path(path)
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return {"sha256": h.hexdigest(), "bytes": path.stat().st_size}


# record what this stage consumed, next to what it produced; `inputs` maps a human label to a path
def stamp_inputs(out_dir: Path, inputs: dict[str, Path]) -> dict:
    record = {label: {"path": _store(p), **(artifact_id(p) or {"missing": True})}
              for label, p in inputs.items()}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / INPUTS_FILENAME).write_text(
        json.dumps(record, indent=2), encoding="utf-8")
    return record


# complaints about `out_dir`, empty when it is current; an ABSENT stamp is reported, not passed:
# an unstamped directory is exactly the state that hides this bug, so "cannot tell" and "stale"
# are reported the same way- the caller decides how loudly to fail
def check_inputs(out_dir: Path) -> list[str]:
    out_dir = Path(out_dir)
    stamp = out_dir / INPUTS_FILENAME
    if not stamp.is_file():
        return [f"{out_dir.name}: no {INPUTS_FILENAME}; cannot tell which inputs produced it"]
    try:
        record = json.loads(stamp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{out_dir.name}: unreadable {INPUTS_FILENAME} ({exc})"]

    out = []
    for label, want in record.items():
        now = artifact_id(_locate(want["path"]))
        if want.get("missing"):
            if now is not None:
                out.append(f"{out_dir.name}: {label} did not exist when this ran, but does now")
            continue
        if now is None:
            out.append(f"{out_dir.name}: {label} has since been deleted ({want['path']})")
        elif now["sha256"] != want["sha256"]:
            out.append(f"{out_dir.name}: {label} changed since this ran -- "
                       f"{out_dir.name} describes an older {Path(want['path']).name}")
    return out


# check several stage directories at once; returns every complaint, in order
def check_all(out_dirs: list[Path]) -> list[str]:
    return [c for d in out_dirs for c in check_inputs(Path(d))]


# ------------------------------------------------------------------------------------------
# self-test: a check that silently stopped firing looks exactly like a clean pipeline
# the stale case is the one that matters; absent or unreadable announces itself
# ------------------------------------------------------------------------------------------

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

    (out / INPUTS_FILENAME).write_text("{not json", encoding="utf-8")
    expect("an unreadable stamp", check_inputs(out), True)

    # An input that did not exist is recorded as such, and its later APPEARANCE is a change
    # too: a stage that ran without an optional upstream is not the same stage as one that
    # ran with it, and reporting only the reverse direction would miss half of that
    absent = tmp / "not-yet.json"
    stamp_inputs(out, {"optional": absent})
    expect("an input absent both times", check_inputs(out), False)
    absent.write_text("{}", encoding="utf-8")
    expect("an input that has since APPEARED", check_inputs(out), True)

    # Paths: repo-relative in the stamp, so it survives a move; absolute only when the input
    # lives outside the repo, where nothing else could identify it
    inside = REPO_ROOT / "freshness.py"
    if _store(inside) != "freshness.py":
        failures.append(f"a repo file should be stored relative, got {_store(inside)!r}")
    if not Path(_store(tmp / "outside.json")).is_absolute():
        failures.append("a file outside the repo must keep its absolute path")
    if _locate("freshness.py") != REPO_ROOT / "freshness.py":
        failures.append("a relative stored path must resolve against the repo root")
    if _locate(str(tmp)) != tmp:
        failures.append("an absolute stored path must be honoured as written")

    # A stamp written before paths were made relative still has to read, or this fix would
    # itself be the thing that invalidated every artifact it was meant to protect
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
    ap.add_argument("--check", nargs="*", metavar="DIR",
                    help="report staleness for these stage directories (default: every "
                         "runs/* carrying a stamp, plus labeled_raw/)")
    args = ap.parse_args()

    if args.self_test:
        failures = _self_test()
        for f in failures:
            print(f"[freshness] FAIL {f}")
        if failures:
            raise SystemExit(f"[freshness] {len(failures)} check(s) no longer fire")
        print("[freshness] self-test OK: every complaint fires on its own case")
        return

    dirs = [Path(d) for d in args.check] if args.check else [
        *(d for d in (REPO_ROOT / "runs").glob("*") if (d / INPUTS_FILENAME).is_file()),
        *( [REPO_ROOT / "labeled_raw"] if (REPO_ROOT / "labeled_raw").is_dir() else []),
    ]
    complaints = check_all(dirs)
    for c in complaints:
        print(f"[freshness] {c}")
    print(f"[freshness] {len(dirs)} directory(s) checked, "
          f"{len(complaints)} complaint(s)")


if __name__ == "__main__":
    main()
