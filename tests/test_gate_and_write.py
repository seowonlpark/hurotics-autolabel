# gate_and_write is the shared "code judges the agent" sink: it must persist the raw reply for
# audit even when nothing parses, write one JSONL line per validated item, and count passers by
# a caller-supplied predicate. these tests hold that contract independent of any one stage.

import json

from agents.base import append_ledger, gate_and_write, read_ledger

# a validator that stamps the standard validation.ok block from an item's own "good" flag
def _validate_ok(item):
    return {**item, "validation": {"ok": bool(item.get("good"))}}


def test_empty_items_writes_empty_file_and_raw(tmp_path):
    path, n_ok, n_flagged = gate_and_write(
        tmp_path, [], _validate_ok, "raw reply text",
        out_name="out.jsonl")
    assert (path, n_ok, n_flagged) == (tmp_path / "out.jsonl", 0, 0)
    assert path.read_text(encoding="utf-8") == ""
    # the raw reply is kept even with nothing to record, next to out_name as <stem>_raw.txt
    assert (tmp_path / "out_raw.txt").read_text(encoding="utf-8") == "raw reply text"


def test_none_items_still_writes_raw(tmp_path):
    path, n_ok, n_flagged = gate_and_write(
        tmp_path, None, _validate_ok, "unparseable",
        out_name="out.jsonl")
    assert (n_ok, n_flagged) == (0, 0)
    assert (tmp_path / "out_raw.txt").read_text(encoding="utf-8") == "unparseable"


def test_counts_passers_and_writes_one_line_each(tmp_path):
    items = [{"good": True}, {"good": False}, {"good": True}]
    path, n_ok, n_flagged = gate_and_write(
        tmp_path, items, _validate_ok, "reply",
        out_name="out.jsonl")
    assert (n_ok, n_flagged) == (2, 1)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    # every line carries the stamped validation block
    assert all("validation" in json.loads(line) for line in lines)


def test_is_ok_override(tmp_path):
    # new-class discovery passes on validation.support == "supported", not validation.ok
    def validate(item):
        return {**item, "validation": {"support": item["support"]}}

    items = [{"support": "supported"}, {"support": "insufficient_evidence"}]
    _, n_ok, n_flagged = gate_and_write(
        tmp_path, items, validate, "reply",
        out_name="out.jsonl",
        is_ok=lambda v: v["validation"]["support"] == "supported")
    assert (n_ok, n_flagged) == (1, 1)


def test_creates_missing_out_dir(tmp_path):
    nested = tmp_path / "runs" / "s4"
    path, _, _ = gate_and_write(
        nested, [], _validate_ok, "x", out_name="o.jsonl")
    assert path.exists()


# cross-run analyst ledger. gate_and_write appends ONLY the passers (by is_ok), each stamped with its
# run; read_ledger reads it back. flagged items and empty runs must not pollute the ledger.

def test_ledger_appends_only_passers_stamped_with_run(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    items = [{"good": True, "id": "a"}, {"good": False, "id": "b"}, {"good": True, "id": "c"}]
    gate_and_write(tmp_path, items, _validate_ok, "reply",
                   out_name="out.jsonl", ledger_path=ledger, run_id="2026-07-24_run1")
    recorded = read_ledger(ledger)
    assert [r["id"] for r in recorded] == ["a", "c"]          # only the passers
    assert all(r["run"] == "2026-07-24_run1" for r in recorded)  # traceable to the run
    assert all("ts" in r and "git_sha" in r for r in recorded)


def test_ledger_accumulates_across_runs(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    gate_and_write(tmp_path, [{"good": True, "id": "a"}], _validate_ok, "r1",
                   out_name="out.jsonl", ledger_path=ledger, run_id="run1")
    gate_and_write(tmp_path, [{"good": True, "id": "b"}], _validate_ok, "r2",
                   out_name="out.jsonl", ledger_path=ledger, run_id="run2")
    assert [r["id"] for r in read_ledger(ledger)] == ["a", "b"]  # append-only, not overwritten


def test_ledger_untouched_when_no_passers(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    gate_and_write(tmp_path, [{"good": False}], _validate_ok, "r",
                   out_name="out.jsonl", ledger_path=ledger, run_id="run1")
    assert not ledger.exists()  # a run that recorded nothing leaves no ledger trace


def test_no_ledger_path_writes_no_ledger(tmp_path):
    # the default (no ledger_path) preserves the original behaviour: per-run file only
    gate_and_write(tmp_path, [{"good": True}], _validate_ok, "r", out_name="out.jsonl")
    assert list(tmp_path.glob("*ledger*")) == []


def test_read_ledger_absent_is_empty(tmp_path):
    assert read_ledger(tmp_path / "nope.jsonl") == []


def test_append_ledger_empty_is_noop(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    append_ledger(ledger, [], "run1")
    assert not ledger.exists()
