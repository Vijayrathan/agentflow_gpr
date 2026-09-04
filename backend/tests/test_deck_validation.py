"""Key-free tests for the uploaded-deck syntax validation.

The validator mirrors gprMax's input preprocessing WITHOUT executing anything,
then applies gprMax's own command-syntax rules (check_cmd_names): valid names,
space after the colon, single-instance commands, essential commands present.
"""

import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.deck_validation import (
    MAX_DECK_BYTES,
    extract_deck_members,
    validate_deck_bytes,
    validate_deck_text,
)

VALID_DECK = """\
#title: demo_1
#domain: 1.2 0.9 0.002
#dx_dy_dz: 0.002 0.002 0.002
#time_window: 1.2e-8

#waveform: ricker 1 9e8 src_wave
#hertzian_dipole: z 0.6 0.8 0 src_wave
#rx: 0.65 0.8 0
#cylinder: 0.5 0.4 0 0.5 0.4 0.002 0.05 pec n
"""


def test_valid_deck_passes():
    assert validate_deck_text(VALID_DECK) == []


def test_comments_free_text_and_blanks_are_ignored():
    deck = "## a comment\nsome free text\n\n" + VALID_DECK
    assert validate_deck_text(deck) == []


def test_invalid_command_name_rejected():
    errors = validate_deck_text(VALID_DECK + "#not_a_command: 1 2 3\n")
    assert len(errors) == 1
    assert "#not_a_command" in errors[0]


def test_missing_essential_commands_rejected():
    errors = validate_deck_text("#title: t\n#waveform: ricker 1 9e8 w\n")
    assert len(errors) == 1
    assert "essential" in errors[0].lower()


def test_duplicate_single_instance_command_rejected():
    errors = validate_deck_text(VALID_DECK + "#domain: 1 1 0.002\n")
    assert len(errors) == 1
    assert "#domain" in errors[0]


def test_missing_space_after_colon_rejected():
    deck = VALID_DECK.replace("#domain: 1.2", "#domain:1.2")
    errors = validate_deck_text(deck)
    assert len(errors) == 1
    assert "space" in errors[0].lower()


def test_missing_colon_rejected():
    errors = validate_deck_text(VALID_DECK + "#messages\n")
    assert any("missing ':'" in e for e in errors)


def test_python_block_rejected_not_executed(tmp_path):
    marker = tmp_path / "pwned"
    deck = (
        VALID_DECK
        + "#python:\n"
        + f"open({str(marker)!r}, 'w').write('x')\n"
        + "#end_python:\n"
    )
    errors = validate_deck_text(deck)
    assert any("#python" in e for e in errors)
    assert not marker.exists()


def test_include_file_rejected():
    errors = validate_deck_text(VALID_DECK + "#include_file: other.in\n")
    assert any("#include_file" in e for e in errors)


def test_empty_deck_rejected():
    assert validate_deck_text("## only a comment\n") == [
        "file contains no gprMax commands"
    ]


def test_non_utf8_bytes_rejected():
    assert validate_deck_bytes(b"\xff\xfe\x00bad") == [
        "file is not valid UTF-8 text"
    ]


# ---------------------------------------------------------------------------
# zip extraction
# ---------------------------------------------------------------------------

def _zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, blob in files.items():
            zf.writestr(name, blob)
    return buf.getvalue()


def test_extract_keeps_only_in_files_flattened():
    data = _zip({
        "a.in": b"#title: a\n",
        "nested/dir/b.in": b"#title: b\n",
        "readme.txt": b"ignore me",
        "__MACOSX/a.in": b"resource fork junk",
        ".hidden.in": b"#title: hidden\n",
    })
    decks, errors = extract_deck_members(data)
    assert [name for name, _ in decks] == ["a.in", "b.in"]
    assert errors == []


def test_extract_flags_duplicate_basenames():
    data = _zip({"x/a.in": b"#title: 1\n", "y/a.in": b"#title: 2\n"})
    decks, errors = extract_deck_members(data)
    assert [name for name, _ in decks] == ["a.in"]
    assert len(errors) == 1 and errors[0]["filename"] == "a.in"


def test_extract_flags_oversized_member():
    data = _zip({"big.in": b"#" * (MAX_DECK_BYTES + 1), "ok.in": b"#title: t\n"})
    decks, errors = extract_deck_members(data)
    assert [name for name, _ in decks] == ["ok.in"]
    assert len(errors) == 1 and "limit" in errors[0]["error"]


def test_extract_rejects_broken_archive():
    try:
        extract_deck_members(b"this is not a zip")
    except ValueError as exc:
        assert "zip" in str(exc)
    else:
        raise AssertionError("expected ValueError for a broken archive")
