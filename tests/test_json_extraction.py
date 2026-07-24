# the agent-reply JSON extractor is the first gate every agent output crosses: an answer wrapped in
# prose or a code fence (or after a worked example) must still parse to the intended value, and garbage
# must fail safe to None. these cases pin that contract.

from agents.base import (
    _balanced_spans,
    extract_json_array,
    extract_json_object,
)


def test_plain_object():
    assert extract_json_object('{"a": 1, "b": 2}') == {"a": 1, "b": 2}


def test_object_in_prose():
    assert extract_json_object('Here is my answer: {"verdict": "approve"} done.') == {
        "verdict": "approve"
    }


def test_fenced_block_is_preferred():
    text = 'loose {"a": 1}\n```json\n{"a": 2}\n```\n'
    # the fenced block wins over the loose prose object
    assert extract_json_object(text) == {"a": 2}


def test_last_valid_span_wins():
    # a worked example first, then the real answer -- keep the last one that parses
    text = 'example {"x": 1} then the answer {"y": 2}'
    assert extract_json_object(text) == {"y": 2}


def test_prose_brace_is_skipped():
    # "{this}" is balanced but not JSON; the real object still wins
    assert extract_json_object('note {this thing} then {"a": 3}') == {"a": 3}


def test_braces_inside_strings_are_ignored():
    assert extract_json_object('{"a": "}{"}') == {"a": "}{"}


def test_escaped_quote_inside_string():
    assert extract_json_object(r'{"a": "he said \"hi\""}') == {"a": 'he said "hi"'}


def test_garbage_returns_none():
    assert extract_json_object("no json here at all") is None


def test_wrong_type_returns_none():
    # asking for an object but given only an array yields None, not the array
    assert extract_json_object("[1, 2, 3]") is None


def test_empty_text_returns_none():
    assert extract_json_object("") is None
    assert extract_json_array("") is None


def test_plain_array():
    assert extract_json_array("[1, 2, 3]") == [1, 2, 3]


def test_array_in_prose():
    assert extract_json_array('answer: [{"ref": "a"}] ok') == [{"ref": "a"}]


def test_balanced_spans_top_level_only():
    spans = list(_balanced_spans('{"a": {"b": 1}} tail {"c": 2}', "{", "}"))
    # nested braces do not start a new top-level span
    assert spans == ['{"a": {"b": 1}}', '{"c": 2}']
