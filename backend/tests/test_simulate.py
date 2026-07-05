"""Key-free tests for the batch gprMax runner and its API wiring.

The solver itself is stubbed (run_simulation is monkeypatched): these tests
cover the batch loop mechanics — output-dir injection, progress events,
skip/failure accounting — and the api.py plumbing that records .out paths
and guards the /simulate endpoint.
"""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend import api, simulate
from fastapi import HTTPException


def _write_in_files(in_dir: Path, names: list[str]) -> None:
    in_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        (in_dir / name).write_text("#title: t\n#domain: 1 1 0.002\n")


def _fake_runner(out_root: Path, fail: set[str] = frozenset()):
    def run(tmp_in: Path, n, gpu_arg, verbose):
        if tmp_in.name in fail:
            raise RuntimeError("solver exploded")
        (out_root / f"{tmp_in.stem}.out").write_bytes(b"\x89HDF")
        # gprMax leaves geometry-view artifacts next to the input file
        tmp_in.with_name(f"{tmp_in.stem}_geo.vti").write_bytes(b"vti")
    return run


def test_inject_output_dir_replaces_existing_line(tmp_path):
    content = "#title: t\n#output_dir: /old/place\n#domain: 1 1 0.002\n"
    result = simulate.inject_output_dir(content, tmp_path)
    assert f"#output_dir: {tmp_path.resolve()}" in result
    assert "/old/place" not in result
    assert result.count("#output_dir") == 1


def test_inject_output_dir_prepends_when_missing(tmp_path):
    result = simulate.inject_output_dir("#title: t\n", tmp_path)
    assert result.splitlines()[0] == f"#output_dir: {tmp_path.resolve()}"


def test_run_batch_success_and_progress_events(tmp_path, monkeypatch):
    in_dir = tmp_path / "in_files"
    _write_in_files(in_dir, ["demo_1.in", "demo_2.in"])
    out_dir = tmp_path / "out_files"
    out_dir.mkdir()
    monkeypatch.setattr(simulate, "run_simulation", _fake_runner(out_dir))

    events = []
    result = simulate.run_batch_simulation(
        in_dir, output_dir=out_dir, progress=events.append
    )

    assert result["succeeded"] == 2
    assert result["failed"] == 0
    assert result["total"] == 2
    assert [o["filename"] for o in result["outputs"]] == ["demo_1.in", "demo_2.in"]
    assert all(Path(o["out_file"]).exists() for o in result["outputs"])
    # start/done pair per file, indices 1-based
    assert [(e["event"], e["index"]) for e in events] == [
        ("start", 1), ("done", 1), ("start", 2), ("done", 2),
    ]
    assert all(e["status"] == "ok" for e in events if e["event"] == "done")
    # tmp dir is cleaned up; sibling artifacts moved to the output dir
    assert not (out_dir / "_tmp").exists()
    assert (out_dir / "demo_1_geo.vti").exists()


def test_run_batch_failure_is_isolated(tmp_path, monkeypatch):
    in_dir = tmp_path / "in_files"
    _write_in_files(in_dir, ["a_1.in", "b_2.in"])
    out_dir = tmp_path / "out_files"
    out_dir.mkdir()
    monkeypatch.setattr(
        simulate, "run_simulation", _fake_runner(out_dir, fail={"a_1.in"})
    )

    events = []
    result = simulate.run_batch_simulation(
        in_dir, output_dir=out_dir, progress=events.append
    )

    assert result["succeeded"] == 1
    assert result["failed"] == 1
    assert result["errors"][0]["filename"] == "a_1.in"
    assert [o["filename"] for o in result["outputs"]] == ["b_2.in"]
    statuses = [e["status"] for e in events if e["event"] == "done"]
    assert statuses == ["failed", "ok"]


def test_run_batch_skip_existing_counts_output(tmp_path, monkeypatch):
    in_dir = tmp_path / "in_files"
    _write_in_files(in_dir, ["demo_1.in"])
    out_dir = tmp_path / "out_files"
    out_dir.mkdir()
    (out_dir / "demo_1.out").write_bytes(b"\x89HDF")

    def boom(*a, **k):
        raise AssertionError("solver must not run for skipped files")

    monkeypatch.setattr(simulate, "run_simulation", boom)

    events = []
    result = simulate.run_batch_simulation(
        in_dir, output_dir=out_dir, skip_existing=True, progress=events.append
    )

    assert result["skipped"] == 1
    # existing .out still reported so DB paths can be (re)recorded
    assert result["outputs"][0]["out_file"].endswith("demo_1.out")
    assert events == [{
        "event": "done", "index": 1, "total": 1, "filename": "demo_1.in",
        "status": "skipped", "out_file": result["outputs"][0]["out_file"],
    }]


def test_run_batch_progress_callback_errors_are_swallowed(tmp_path, monkeypatch):
    in_dir = tmp_path / "in_files"
    _write_in_files(in_dir, ["demo_1.in"])
    out_dir = tmp_path / "out_files"
    out_dir.mkdir()
    monkeypatch.setattr(simulate, "run_simulation", _fake_runner(out_dir))

    def bad_progress(event):
        raise RuntimeError("reporting broke")

    result = simulate.run_batch_simulation(
        in_dir, output_dir=out_dir, progress=bad_progress
    )
    assert result["succeeded"] == 1


def test_run_batch_filenames_restricts_to_manifest(tmp_path, monkeypatch):
    # the dataset in_files dir is shared across sessions — stale decks from
    # an earlier run must not be simulated (regression: 100-sample dataset
    # kicked off a 220-file batch)
    in_dir = tmp_path / "in_files"
    _write_in_files(in_dir, ["soil_sample_1.in", "big_test_1.in", "big_test_2.in"])
    out_dir = tmp_path / "out_files"
    out_dir.mkdir()
    monkeypatch.setattr(simulate, "run_simulation", _fake_runner(out_dir))

    events = []
    result = simulate.run_batch_simulation(
        in_dir, output_dir=out_dir,
        filenames=["soil_sample_1.in"], progress=events.append,
    )

    assert result["total"] == 1
    assert [o["filename"] for o in result["outputs"]] == ["soil_sample_1.in"]
    assert all(e["total"] == 1 for e in events)
    assert not (out_dir / "big_test_1.out").exists()


def test_run_batch_missing_inputs_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        simulate.run_batch_simulation(tmp_path / "nope")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        simulate.run_batch_simulation(empty)


def test_record_simulation_outputs_maps_samples(monkeypatch):
    chat = api._new_chat_session("session-x")
    manifest = {"files": [
        {"sample_id": 1, "filename": "demo_1.in"},
        {"sample_id": 2, "filename": "demo_2.in"},
    ]}
    result = {"outputs": [
        {"filename": "demo_1.in", "out_file": "/abs/demo_1.out"},
        {"filename": "orphan.in", "out_file": "/abs/orphan.out"},
    ]}

    captured = {}

    def fake_set(session_uuid, outputs_by_sample):
        captured["session"] = session_uuid
        captured["outputs"] = outputs_by_sample
        return len(outputs_by_sample)

    monkeypatch.setattr(api, "set_simulation_outputs", fake_set)
    updated = api._record_simulation_outputs(chat, manifest, result)

    assert updated == 1
    assert captured["outputs"] == {1: "/abs/demo_1.out"}
    assert captured["session"] == api._coerce_uuid("session-x")


def test_record_simulation_outputs_swallows_db_errors(monkeypatch):
    chat = api._new_chat_session("session-x")
    manifest = {"files": [{"sample_id": 1, "filename": "demo_1.in"}]}
    result = {"outputs": [{"filename": "demo_1.in", "out_file": "/abs/demo_1.out"}]}

    def broken(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(api, "set_simulation_outputs", broken)
    assert api._record_simulation_outputs(chat, manifest, result) == 0


def test_simulate_endpoint_rejects_unknown_session():
    with pytest.raises(HTTPException) as exc:
        asyncio.run(api.start_forward_model("no-such-session"))
    assert exc.value.status_code == 404


def test_simulate_endpoint_guards_running_states(tmp_path, monkeypatch):
    chat = api._new_chat_session("guard-session")
    api.sessions["guard-session"] = chat
    try:
        in_dir = tmp_path / "in_files"
        _write_in_files(in_dir, ["demo_1.in"])
        (tmp_path / "emitted_files.json").write_text(
            '{"n_written": 1, "in_dir": "%s", '
            '"files": [{"sample_id": 1, "filename": "demo_1.in"}]}' % in_dir
        )
        chat.state["dataset_config"] = {
            "num_samples": 1,
            "model_basename": "demo",
            "output_dir": str(tmp_path),
        }

        chat.simulating = True
        with pytest.raises(HTTPException) as exc:
            asyncio.run(api.start_forward_model("guard-session"))
        assert exc.value.status_code == 409

        chat.simulating = False
        chat.busy = True
        with pytest.raises(HTTPException) as exc:
            asyncio.run(api.start_forward_model("guard-session"))
        assert exc.value.status_code == 409
    finally:
        api.sessions.pop("guard-session", None)
