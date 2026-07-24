# _select_sections curates DOMAIN_NOTES down to the sections one agent needs. under-scoping
# must stay recoverable (Section 0 always kept, a pointer to the full file added), and a
# subsection must ride with its parent rather than being dropped for lacking its own top header.

from agents.base import _select_sections

DOC = """Preamble line about the corpus.

## 0. Orientation
always keep me.

## 4. Channel trust
axis stuff.

## 5. Labels
label codes.

### 5.2 minus-one
subsection riding with 5.

## 9. Features
feature stuff.
"""


def test_keeps_requested_plus_section_zero():
    out = _select_sections(DOC, ("5",))
    assert "## 0. Orientation" in out
    assert "## 5. Labels" in out
    assert "## 4. Channel trust" not in out
    assert "## 9. Features" not in out


def test_subsection_rides_with_parent():
    out = _select_sections(DOC, ("5",))
    # 5.2 has no top-level "## " header, so it stays inside section 5's span
    assert "5.2 minus-one" in out


def test_preamble_and_curation_note_present():
    out = _select_sections(DOC, ("4",))
    assert "Preamble line about the corpus." in out
    # a note tells the agent which sections were shown and where the full file lives
    assert "sections 0, 4" in out
    assert "full DOMAIN_NOTES.md" in out


def test_no_headers_returns_full_text():
    plain = "just some text, no markdown headers"
    assert _select_sections(plain, ("5",)) == plain
