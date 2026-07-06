"""Key-free tests for the forward-model outcome (A-scan) endpoints.

Covers read_ascan's HDF5 parsing plus the /datasets/{sid}/outputs endpoints:
the availability list is filesystem-derived, and the per-file payload serves
only the session's own out_files dir.
"""

import sys
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend import api
from backend.signal_extraction import read_ascan
from fastapi import HTTPException


def _write_out_file(path: Path, components: dict[str, list[float]],
                    dt: float = 2e-12, iterations: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(path), "w") as f:
        f.attrs["dt"] = dt
        f.attrs["Iterations"] = iterations
        f.attrs["nrx"] = 1
        rx = f.create_group("/rxs/rx1")
        for comp, values in components.items():
            rx.create_dataset(comp, data=np.asarray(values, dtype=np.float64))


@contextmanager
def _session(tmp_path: Path, filenames: list[str]):
    sid = "outputs-session"
    chat = api._new_chat_session(sid)
    api.sessions[sid] = chat
    try:
        in_dir = tmp_path / "in_files"
        in_dir.mkdir(parents=True, exist_ok=True)
        for name in filenames:
            (in_dir / name).write_text("#title: t\n")
        files = [
            {"sample_id": i + 1, "filename": name}
            for i, name in enumerate(filenames)
        ]
        (tmp_path / "emitted_files.json").write_text(
            __import__("json").dumps(
                {"n_written": len(files), "in_dir": str(in_dir), "files": files}
            )
        )
        chat.state["dataset_config"] = {
            "num_samples": len(files),
            "model_basename": "demo",
            "output_dir": str(tmp_path),
        }
        yield sid, tmp_path / "out_files"
    finally:
        api.sessions.pop(sid, None)


def test_read_ascan_returns_time_axis_and_components(tmp_path):
    out = tmp_path / "demo_0001.out"
    _write_out_file(out, {"Ez": [0.0, 1.0, -1.0], "Hy": [0.1, 0.2, 0.3]},
                    dt=1.5e-12, iterations=3)
    data = read_ascan(out)
    assert data["dt"] == pytest.approx(1.5e-12)
    assert data["iterations"] == 3
    assert set(data["components"]) == {"Ez", "Hy"}
    assert data["components"]["Ez"] == [0.0, 1.0, -1.0]


def test_read_ascan_missing_receiver_raises(tmp_path):
    out = tmp_path / "demo_0001.out"
    with h5py.File(str(out), "w") as f:
        f.attrs["dt"] = 1e-12
        f.attrs["Iterations"] = 1
    with pytest.raises(ValueError):
        read_ascan(out)


def test_read_ascan_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_ascan(tmp_path / "nope.out")


def test_outputs_list_reflects_disk(tmp_path):
    with _session(tmp_path, ["demo_0001.in", "demo_0002.in"]) as (sid, out_dir):
        assert api.list_dataset_outputs(sid) == {"files": []}
        _write_out_file(out_dir / "demo_0001.out", {"Ez": [0.0]})
        assert api.list_dataset_outputs(sid) == {"files": ["demo_0001.in"]}
        _write_out_file(out_dir / "demo_0002.out", {"Ez": [0.0]})
        assert api.list_dataset_outputs(sid) == {
            "files": ["demo_0001.in", "demo_0002.in"]
        }


def test_output_endpoint_serves_ascan_payload(tmp_path):
    with _session(tmp_path, ["demo_0001.in"]) as (sid, out_dir):
        _write_out_file(out_dir / "demo_0001.out", {"Ez": [0.0, 2.0]},
                        dt=2e-12, iterations=2)
        payload = api.get_dataset_output(sid, "demo_0001.in")
        assert payload["filename"] == "demo_0001.out"
        assert payload["iterations"] == 2
        assert payload["components"]["Ez"] == [0.0, 2.0]


def test_output_endpoint_404s_when_missing(tmp_path):
    with _session(tmp_path, ["demo_0001.in"]) as (sid, _out_dir):
        with pytest.raises(HTTPException) as exc:
            api.get_dataset_output(sid, "demo_0001.in")
        assert exc.value.status_code == 404


def test_output_endpoint_ignores_path_components(tmp_path):
    with _session(tmp_path, ["demo_0001.in"]) as (sid, out_dir):
        _write_out_file(out_dir / "demo_0001.out", {"Ez": [0.0]})
        # traversal-ish names collapse to their basename inside the session dir
        payload = api.get_dataset_output(sid, "../in_files/demo_0001.in")
        assert payload["filename"] == "demo_0001.out"
        with pytest.raises(HTTPException) as exc:
            api.get_dataset_output(sid, "../../etc/passwd")
        assert exc.value.status_code == 404


def test_output_endpoint_unreadable_file_is_422(tmp_path):
    with _session(tmp_path, ["demo_0001.in"]) as (sid, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "demo_0001.out").write_bytes(b"not an hdf5 file")
        with pytest.raises(HTTPException) as exc:
            api.get_dataset_output(sid, "demo_0001.in")
        assert exc.value.status_code == 422
