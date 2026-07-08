"""
Syntax validation for user-uploaded gprMax .in decks.

The authority for what counts as a valid command line is gprMax's OWN input
checker (`gprMax.input_cmds_file.check_cmd_names`): valid command names, the
space-after-colon rule, single-instance commands appearing once, and the
essential commands (#domain, #dx_dy_dz, #time_window) being present.

Preprocessing deliberately does NOT reuse gprMax's
`process_python_include_code`: that function `exec()`s any `#python:` block,
which must never happen for uploaded content. This module mirrors its line
handling (drop `##` comments, blank lines and free text; keep `#` command
lines) and REJECTS `#python:` / `#end_python:` / `#include_file:` decks with a
clear error instead of executing or resolving them.

gprMax imports are lazy (same pattern as simulate.py) so importing this module
stays cheap and key-free tests can import it without the solver stack.
"""
from __future__ import annotations

import contextlib
import io
import zipfile
from pathlib import PurePosixPath

# Anti-zip-bomb caps: member sizes are checked on the DECOMPRESSED size before
# reading, and only .in members are ever read at all.
MAX_DECK_FILES = 2000
MAX_DECK_BYTES = 5 * 1024 * 1024  # a .in deck is plain text; 5 MB is generous

_UNSUPPORTED_CMDS = {
    "#python": "embedded Python blocks (#python:) are not supported in uploaded decks",
    "#end_python": "embedded Python blocks (#end_python:) are not supported in uploaded decks",
    "#include_file": "#include_file is not supported in uploaded decks (external file references cannot be resolved)",
}


def preprocess_deck(text: str) -> tuple[list[str], list[str]]:
    """Mirror gprMax's input preprocessing WITHOUT executing anything.

    Returns (processed command lines ready for check_cmd_names, structural
    errors). Lines gprMax would silently ignore (## comments, blanks, free
    text) are ignored here too, so validation matches what gprMax would run.
    """
    errors: list[str] = []
    processed: list[str] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip()
        if not line or line.startswith("##") or not line.startswith("#"):
            continue
        if ":" not in line:
            errors.append(
                f"line {lineno}: missing ':' between command name and parameters: '{line}'"
            )
            continue
        cmdname = line.split(":", 1)[0]
        if cmdname in _UNSUPPORTED_CMDS:
            errors.append(f"line {lineno}: {_UNSUPPORTED_CMDS[cmdname]}")
            continue
        # check_cmd_names expects the trailing newline the file reader keeps.
        processed.append(line + "\n")
    return processed, errors


def validate_deck_text(text: str) -> list[str]:
    """Error messages for one deck; an empty list means the deck passes
    gprMax's command-syntax rules (including the essential-commands check)."""
    processed, errors = preprocess_deck(text)
    if errors:
        return errors
    if not processed:
        return ["file contains no gprMax commands"]

    # Lazy import: pulls in the gprMax package (same pattern as simulate.py).
    from gprMax.exceptions import CmdInputError
    from gprMax.input_cmds_file import check_cmd_names

    try:
        # CmdInputError prints a bare color escape on construction — keep it
        # out of the server log / captured chat output.
        with contextlib.redirect_stdout(io.StringIO()):
            check_cmd_names(processed, checkessential=True)
    except CmdInputError as exc:
        return [str(exc.message)]
    except Exception as exc:  # a line malformed enough to crash the checker
        return [f"unparseable command line: {exc}"]
    return []


def validate_deck_bytes(data: bytes) -> list[str]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return ["file is not valid UTF-8 text"]
    return validate_deck_text(text)


def extract_deck_members(data: bytes) -> tuple[list[tuple[str, bytes]], list[dict]]:
    """The .in members of a zip as ordered (basename, bytes) pairs, plus
    per-member errors for .in files that could not be accepted (oversized,
    duplicate basename). Non-.in members, directories and macOS metadata are
    ignored. Raises ValueError for a broken archive or one over the file cap.
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ValueError("not a valid zip archive") from exc

    decks: list[tuple[str, bytes]] = []
    errors: list[dict] = []
    seen: set[str] = set()
    with zf:
        for info in zf.infolist():
            if info.is_dir() or "__MACOSX" in info.filename:
                continue
            name = PurePosixPath(info.filename).name
            if not name or name.startswith(".") or not name.lower().endswith(".in"):
                continue
            if info.file_size > MAX_DECK_BYTES:
                errors.append({
                    "filename": name,
                    "error": f"file exceeds the {MAX_DECK_BYTES // (1024 * 1024)} MB per-deck limit",
                })
                continue
            if name in seen:
                errors.append({
                    "filename": name,
                    "error": "duplicate filename in archive (nested copies collapse to one name)",
                })
                continue
            if len(decks) >= MAX_DECK_FILES:
                raise ValueError(
                    f"archive contains more than {MAX_DECK_FILES} .in files"
                )
            seen.add(name)
            decks.append((name, zf.read(info)))
    return decks, errors
