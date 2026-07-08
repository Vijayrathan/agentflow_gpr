"""Key-free tests for the dataset-upload endpoint path.

Covers _import_deck_zip (per-file syntax gate, manifest shape, session
override) and that the existing dataset endpoints serve the uploaded files
exactly like a generated dataset.
"""

import io
import json
import sys
import zipfile
from contextlib import contextmanager
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend import api
from fastapi import HTTPException

VALID_DECK = """\
#title: demo_1
#domain: 1.2 0.9 0.002
#dx_dy_dz: 0.002 0.002 0.002
#time_window: 1.2e-8
#waveform: ricker 1 9e8 src_wave
#hertzian_dipole: z 0.6 0.8 0 src_wave
#rx: 0.65 0.8 0
"""

INVALID_DECK = "#title: broken\n#not_a_command: 1\n"


def _zip(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, text in files.items():
            zf.writestr(name, text)
    return buf.getvalue()


@contextmanager
def _session(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "DATASET_ROOT", str(tmp_path))
    sid = "upload-session"
    chat = api._new_chat_session(sid)
    api.sessions[sid] = chat
    try:
        yield sid, chat
    finally:
        api.sessions.pop(sid, None)


def test_import_writes_valid_rejects_invalid(tmp_path, monkeypatch):
    with _session(tmp_path, monkeypatch) as (sid, chat):
        data = _zip({"good.in": VALID_DECK, "bad.in": INVALID_DECK})
        result = api._import_deck_zip(chat, data, "My Decks.zip")

        assert result["status"] == "partial"
        assert result["n_written"] == 1
        assert [f["filename"] for f in result["files"]] == ["good.in"]
        assert result["rejected"][0]["filename"] == "bad.in"
        assert "#not_a_command" in result["rejected"][0]["error"]

        out_dir = Path(result["output_dir"])
        assert out_dir == tmp_path / "My_Decks"  # sanitized zip stem
        assert (out_dir / "in_files" / "good.in").read_text() == VALID_DECK
        assert not (out_dir / "in_files" / "bad.in").exists()

        manifest = json.loads((out_dir / "emitted_files.json").read_text())
        assert manifest["source"] == "upload"
        assert manifest["n_written"] == 1

        # Session now serves the upload through the existing endpoints.
        assert chat.uploaded_output_dir == str(out_dir)
        assert chat.dataset_result is result
        listing = api.list_dataset_files(sid)
        assert [f["filename"] for f in listing["files"]] == ["good.in"]
        content = api.get_dataset_file(sid, "good.in")
        assert content.body.decode() == VALID_DECK


def test_import_all_invalid_raises_422(tmp_path, monkeypatch):
    with _session(tmp_path, monkeypatch) as (_sid, chat):
        with pytest.raises(HTTPException) as exc:
            api._import_deck_zip(chat, _zip({"bad.in": INVALID_DECK}), "b.zip")
        assert exc.value.status_code == 422
        assert "bad.in" in exc.value.detail
        assert chat.uploaded_output_dir is None


def test_import_no_in_files_raises_422(tmp_path, monkeypatch):
    with _session(tmp_path, monkeypatch) as (_sid, chat):
        with pytest.raises(HTTPException) as exc:
            api._import_deck_zip(chat, _zip({"readme.txt": "hi"}), "c.zip")
        assert exc.value.status_code == 422
        assert ".in" in exc.value.detail


def test_import_broken_zip_raises_422(tmp_path, monkeypatch):
    with _session(tmp_path, monkeypatch) as (_sid, chat):
        with pytest.raises(HTTPException) as exc:
            api._import_deck_zip(chat, b"not a zip", "d.zip")
        assert exc.value.status_code == 422


def test_uploaded_dir_takes_precedence_over_pipeline_config(tmp_path, monkeypatch):
    with _session(tmp_path, monkeypatch) as (sid, chat):
        pipeline_dir = tmp_path / "pipeline_ds"
        pipeline_dir.mkdir()
        chat.state["dataset_config"] = {
            "num_samples": 1,
            "model_basename": "demo",
            "output_dir": str(pipeline_dir),
        }
        api._import_deck_zip(chat, _zip({"good.in": VALID_DECK}), "up.zip")
        assert api._session_output_dir(sid) == tmp_path / "up"

        # The pipeline emitting its own dataset clears the override.
        chat.uploaded_output_dir = None
        assert api._session_output_dir(sid) == pipeline_dir


def test_record_simulation_outputs_skips_uploaded_manifests(tmp_path, monkeypatch):
    with _session(tmp_path, monkeypatch) as (_sid, chat):
        manifest = {
            "source": "upload",
            "files": [{"sample_id": 1, "filename": "good.in"}],
        }
        result = {"outputs": [{"filename": "good.in", "out_file": "x.out"}]}
        assert api._record_simulation_outputs(chat, manifest, result) == 0


def test_upload_summary_lists_rejections():
    summary = api._upload_summary({
        "n_written": 2,
        "rejected": [{"filename": "bad.in", "error": "invalid command"}],
    })
    assert "2 gprMax input file(s)" in summary
    assert "bad.in" in summary and "invalid command" in summary
