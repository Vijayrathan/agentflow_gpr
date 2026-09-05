"""Key-free tests for the batch gprMax runner and its API wiring.

The solver itself is stubbed (run_simulation is monkeypatched): these tests
cover the batch loop mechanics — output-dir injection, progress events,
skip/failure accounting — and the api.py plumbing that records .out paths
and guards the /simulate endpoint.
"""

import asyncio
import sys
import time
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
        "event": "done", "index": 1, "total": 1, "completed": 1,
        "filename": "demo_1.in", "status": "skipped",
        "out_file": result["outputs"][0]["out_file"],
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


def _clear_backend_env(monkeypatch):
    for name in ("GPR_GPU", "GPR_GPU_IDS", "GPR_SIM_WORKERS"):
        monkeypatch.delenv(name, raising=False)


def test_execution_plan_defaults_to_cpu_serial(monkeypatch):
    _clear_backend_env(monkeypatch)
    plan = simulate.resolve_execution()
    assert (plan.gpu, plan.workers, plan.gpu_arg_for(0)) == (False, 1, None)
    # a serial CPU run must not throttle gprMax's own OpenMP threads
    assert plan.omp_threads is None


def test_execution_plan_from_env(monkeypatch):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("GPR_GPU", "1")
    plan = simulate.resolve_execution()
    assert plan.gpu is True
    # GPU default: overlap models so the device is not idle during model build
    assert plan.workers == simulate._DEFAULT_WORKERS_PER_GPU
    assert plan.omp_threads >= 1

    monkeypatch.setenv("GPR_SIM_WORKERS", "5")
    assert simulate.resolve_execution().workers == 5
    # explicit arguments beat the environment
    assert simulate.resolve_execution(gpu=False, workers=1).gpu is False


def test_execution_plan_ignores_malformed_env(monkeypatch):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("GPR_GPU_IDS", "zero,one")
    monkeypatch.setenv("GPR_SIM_WORKERS", "lots")
    plan = simulate.resolve_execution()
    # a typo in a deployment env var must not make the batch unstartable
    assert (plan.gpu, plan.gpu_ids, plan.workers) == (False, [], 1)


def test_execution_plan_spreads_models_over_devices(monkeypatch):
    _clear_backend_env(monkeypatch)
    plan = simulate.resolve_execution(gpu_ids=[0, 1])
    assert plan.gpu is True
    assert plan.workers == 2 * simulate._DEFAULT_WORKERS_PER_GPU
    assert [plan.gpu_arg_for(i) for i in range(3)] == [[[0]], [[1]], [[0]]]
    # no explicit IDs => gprMax picks device 0
    assert simulate.resolve_execution(gpu=True).gpu_arg_for(0) == [[]]


@pytest.mark.parametrize("gpu_source", ["argument", "environment", "device_ids", "env_device_ids"])
@pytest.mark.parametrize("workers", [1, 3])
def test_gpu_transmission_line_preflight_rejects_entire_batch(tmp_path, monkeypatch, gpu_source, workers):
    _clear_backend_env(monkeypatch)
    kwargs = {}
    if gpu_source == "argument":
        kwargs["gpu"] = True
    elif gpu_source == "environment":
        monkeypatch.setenv("GPR_GPU", "1")
    elif gpu_source == "device_ids":
        kwargs["gpu_ids"] = [0]
    else:
        monkeypatch.setenv("GPR_GPU_IDS", "0")

    in_dir = tmp_path / "in_files"
    _write_in_files(in_dir, ["a_dipole.in", "z_transmission.in"])
    (in_dir / "z_transmission.in").write_text("#transmission_line: z 0 0 0 75 pulse\n")
    out_dir = tmp_path / "out_files"
    events = []

    def unexpected(*args, **kwargs):
        pytest.fail("preflight must reject before creating workers or executing a model")

    monkeypatch.setattr(simulate, "_make_pool", unexpected)
    monkeypatch.setattr(simulate, "_execute_one", unexpected)
    with pytest.raises(ValueError, match="z_transmission.in: transmission_line sources require CPU solving"):
        simulate.run_batch_simulation(
            in_dir, output_dir=out_dir, workers=workers, progress=events.append, **kwargs
        )

    assert events == []
    assert not out_dir.exists()


def test_cpu_transmission_line_runs_with_source_preserved(tmp_path, monkeypatch):
    _clear_backend_env(monkeypatch)
    # Explicit CPU settings must also work on a GPU-configured deployment.
    monkeypatch.setenv("GPR_GPU", "1")
    monkeypatch.setenv("GPR_GPU_IDS", "0")
    in_dir = tmp_path / "in_files"
    _write_in_files(in_dir, ["transmission.in"])
    source = "#transmission_line: z 0 0 0 75 pulse\n"
    (in_dir / "transmission.in").write_text(source)
    out_dir = tmp_path / "out_files"
    runner = _fake_runner(out_dir)
    seen = []

    def spy(tmp_in, n, gpu_arg, verbose):
        seen.append((gpu_arg, tmp_in.read_text()))
        return runner(tmp_in, n, gpu_arg, verbose)

    monkeypatch.setattr(simulate, "run_simulation", spy)
    result = simulate.run_batch_simulation(
        in_dir, output_dir=out_dir, gpu=False, gpu_ids=[], workers=1
    )
    assert (result["mode"], result["succeeded"], result["failed"]) == ("cpu", 1, 0)
    assert len(seen) == 1
    assert seen[0][0] is None
    assert source in seen[0][1]


@pytest.mark.parametrize("exclude_transmission", ["manifest", "skip_existing"])
def test_gpu_preflight_ignores_transmission_lines_not_pending(tmp_path, monkeypatch, exclude_transmission):
    import concurrent.futures as futures

    _clear_backend_env(monkeypatch)
    in_dir = tmp_path / "in_files"
    _write_in_files(in_dir, ["dipole.in", "transmission.in"])
    (in_dir / "transmission.in").write_text("#transmission_line: z 0 0 0 75 pulse\n")
    out_dir = tmp_path / "out_files"
    out_dir.mkdir()
    kwargs = {}
    if exclude_transmission == "manifest":
        kwargs["filenames"] = ["dipole.in"]
    else:
        (out_dir / "transmission.out").write_bytes(b"existing output")
        kwargs["skip_existing"] = True

    seen = []
    runner = _fake_runner(out_dir)

    def spy(tmp_in, n, gpu_arg, verbose):
        seen.append(tmp_in.name)
        return runner(tmp_in, n, gpu_arg, verbose)

    monkeypatch.setattr(simulate, "run_simulation", spy)
    monkeypatch.setattr(
        simulate, "_make_pool", lambda plan: futures.ThreadPoolExecutor(plan.workers)
    )
    result = simulate.run_batch_simulation(in_dir, output_dir=out_dir, gpu=True, **kwargs)
    assert seen == ["dipole.in"]
    assert (result["succeeded"], result["failed"]) == (1, 0)
    assert result["skipped"] == (1 if exclude_transmission == "skip_existing" else 0)


def test_cli_reports_gpu_transmission_line_preflight_error(tmp_path, monkeypatch, capsys):
    _clear_backend_env(monkeypatch)
    _write_in_files(tmp_path, ["transmission.in"])
    (tmp_path / "transmission.in").write_text("#transmission_line: z 0 0 0 75 pulse\n")
    monkeypatch.setattr(sys, "argv", [
        "simulate.py", "--input-dir", str(tmp_path), "--gpu",
    ])
    with pytest.raises(SystemExit) as exc:
        simulate.main()
    assert exc.value.code == 1
    output = capsys.readouterr()
    assert "[ERROR] transmission.in: transmission_line sources require CPU solving" in output.out
    assert "Disable GPU execution" in output.out
    assert "Traceback" not in output.err


def test_gpu_preflight_preserves_per_file_read_failures(tmp_path, monkeypatch):
    import concurrent.futures as futures

    _clear_backend_env(monkeypatch)
    in_dir = tmp_path / "in_files"
    _write_in_files(in_dir, ["invalid.in", "valid.in"])
    (in_dir / "invalid.in").write_bytes(b"\xff")
    out_dir = tmp_path / "out_files"
    monkeypatch.setattr(simulate, "run_simulation", _fake_runner(out_dir))
    monkeypatch.setattr(
        simulate, "_make_pool", lambda plan: futures.ThreadPoolExecutor(plan.workers)
    )
    result = simulate.run_batch_simulation(in_dir, output_dir=out_dir, gpu=True)
    assert (result["succeeded"], result["failed"]) == (1, 1)
    assert result["errors"][0]["filename"] == "invalid.in"
    assert "UnicodeDecodeError" in result["errors"][0]["error"]
    assert result["outputs"][0]["filename"] == "valid.in"


def test_run_batch_passes_gpu_arg_to_solver(tmp_path, monkeypatch):
    import concurrent.futures as futures

    _clear_backend_env(monkeypatch)
    in_dir = tmp_path / "in_files"
    _write_in_files(in_dir, ["demo_1.in"])
    (in_dir / "demo_1.in").write_text(
        "## #transmission_line: z 0 0 0 75 pulse\n"
        "#title: transmission_line example\n"
        "#hertzian_dipole: z 0 0 0 pulse\n"
        "#voltage_source: z 0 0 0 50 pulse\n"
    )
    out_dir = tmp_path / "out_files"
    out_dir.mkdir()
    # GPU mode always uses the pool; swap in threads so the stub applies
    monkeypatch.setattr(
        simulate, "_make_pool", lambda plan: futures.ThreadPoolExecutor(plan.workers)
    )

    seen = []
    runner = _fake_runner(out_dir)

    def spy(tmp_in, n, gpu_arg, verbose):
        seen.append(gpu_arg)
        return runner(tmp_in, n, gpu_arg, verbose)

    monkeypatch.setattr(simulate, "run_simulation", spy)
    result = simulate.run_batch_simulation(
        in_dir, output_dir=out_dir, gpu=True, gpu_ids=[1], workers=1
    )

    assert seen == [[[1]]]
    assert (result["mode"], result["workers"], result["gpu_ids"]) == ("gpu", 1, [1])


def test_run_batch_parallel_runs_every_file(tmp_path, monkeypatch):
    # The parallel driver is exercised through a thread pool so the stubbed
    # solver still applies (real runs use spawned processes — see _make_pool).
    import concurrent.futures as futures

    _clear_backend_env(monkeypatch)
    in_dir = tmp_path / "in_files"
    names = [f"demo_{i}.in" for i in range(1, 6)]
    _write_in_files(in_dir, names)
    out_dir = tmp_path / "out_files"
    out_dir.mkdir()
    monkeypatch.setattr(simulate, "run_simulation", _fake_runner(out_dir, fail={"demo_3.in"}))

    inflight = {"now": 0, "peak": 0}
    real_execute = simulate._execute_one

    def counting_execute(task):
        inflight["now"] += 1
        inflight["peak"] = max(inflight["peak"], inflight["now"])
        try:
            time.sleep(0.05)  # hold the slot so genuine overlap is observable
            return real_execute(task)
        finally:
            inflight["now"] -= 1

    monkeypatch.setattr(simulate, "_execute_one", counting_execute)
    monkeypatch.setattr(
        simulate, "_make_pool", lambda plan: futures.ThreadPoolExecutor(plan.workers)
    )

    events = []
    result = simulate.run_batch_simulation(
        in_dir, output_dir=out_dir, workers=3, progress=events.append
    )

    assert result["succeeded"] == 4
    assert result["failed"] == 1
    assert result["errors"][0]["filename"] == "demo_3.in"
    assert sorted(o["filename"] for o in result["outputs"]) == sorted(
        n for n in names if n != "demo_3.in"
    )
    assert result["workers"] == 3
    # models really do overlap, and never more than `workers` at a time
    assert 1 < inflight["peak"] <= 3
    # progress stays monotonic even though files finish out of order
    done_counts = [e["completed"] for e in events if e["event"] == "done"]
    assert done_counts == [1, 2, 3, 4, 5]
    assert not (out_dir / "_tmp").exists()


def test_run_batch_parallel_survives_dead_worker(tmp_path, monkeypatch):
    # A worker process can die outright (CUDA fault, OOM): that file fails,
    # the batch continues.
    import concurrent.futures as futures

    _clear_backend_env(monkeypatch)
    in_dir = tmp_path / "in_files"
    _write_in_files(in_dir, ["a_1.in", "b_2.in"])
    out_dir = tmp_path / "out_files"
    out_dir.mkdir()
    monkeypatch.setattr(simulate, "run_simulation", _fake_runner(out_dir))

    real_execute = simulate._execute_one

    def flaky(task):
        if task["filename"] == "a_1.in":
            raise RuntimeError("worker process died")
        return real_execute(task)

    monkeypatch.setattr(simulate, "_execute_one", flaky)
    monkeypatch.setattr(
        simulate, "_make_pool", lambda plan: futures.ThreadPoolExecutor(plan.workers)
    )

    result = simulate.run_batch_simulation(in_dir, output_dir=out_dir, workers=2)

    assert (result["succeeded"], result["failed"]) == (1, 1)
    assert "worker process died" in result["errors"][0]["error"]


def test_failed_model_does_not_poison_the_next_one(tmp_path, monkeypatch):
    # gprMax leaves its built grid in a module-level global when a model
    # raises; the next api() call in the same process would then silently
    # reuse the DEAD model's geometry ("not re-processed, i.e. geometry
    # fixed") instead of building the new one.
    import types

    _clear_backend_env(monkeypatch)
    fake_mbr = types.ModuleType("gprMax.model_build_run")
    fake_mbr.G = object()
    monkeypatch.setitem(sys.modules, "gprMax.model_build_run", fake_mbr)

    in_dir = tmp_path / "in_files"
    _write_in_files(in_dir, ["a_1.in", "b_2.in"])
    out_dir = tmp_path / "out_files"
    out_dir.mkdir()

    saw_leftover_grid = []
    runner = _fake_runner(out_dir, fail={"a_1.in"})

    def spy(tmp_in, n, gpu_arg, verbose):
        saw_leftover_grid.append(hasattr(fake_mbr, "G"))
        return runner(tmp_in, n, gpu_arg, verbose)

    monkeypatch.setattr(simulate, "run_simulation", spy)
    result = simulate.run_batch_simulation(in_dir, output_dir=out_dir)

    # first file inherits the pre-existing global, second must start clean
    assert saw_leftover_grid == [True, False]
    assert (result["succeeded"], result["failed"]) == (1, 1)


def test_gpu_runs_never_execute_in_the_calling_process(tmp_path, monkeypatch):
    # api.py runs batches in a thread of the uvicorn process; a CUDA context
    # must never be created there, so GPU mode uses the pool even at workers=1.
    import concurrent.futures as futures

    _clear_backend_env(monkeypatch)
    in_dir = tmp_path / "in_files"
    _write_in_files(in_dir, ["demo_1.in"])
    out_dir = tmp_path / "out_files"
    out_dir.mkdir()
    monkeypatch.setattr(simulate, "run_simulation", _fake_runner(out_dir))

    pools = []

    def tracking_pool(plan):
        pools.append(plan)
        return futures.ThreadPoolExecutor(plan.workers)

    monkeypatch.setattr(simulate, "_make_pool", tracking_pool)
    simulate.run_batch_simulation(in_dir, output_dir=out_dir, gpu=True, workers=1)
    assert [p.workers for p in pools] == [1]

    # CPU at workers=1 stays in-process (no pool, no spawn overhead)
    pools.clear()
    simulate.run_batch_simulation(
        in_dir, output_dir=out_dir, gpu=False, workers=1, skip_existing=False
    )
    assert pools == []


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

    # result has no output_dir -> signal extraction fails and is swallowed
    assert updated == (1, 0)
    assert captured["outputs"] == {1: "/abs/demo_1.out"}
    assert captured["session"] == api._coerce_uuid("session-x")


def test_record_simulation_outputs_swallows_db_errors(monkeypatch):
    chat = api._new_chat_session("session-x")
    manifest = {"files": [{"sample_id": 1, "filename": "demo_1.in"}]}
    result = {"outputs": [{"filename": "demo_1.in", "out_file": "/abs/demo_1.out"}]}

    def broken(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(api, "set_simulation_outputs", broken)
    assert api._record_simulation_outputs(chat, manifest, result) == (0, 0)


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


# ---------------------------------------------------------------------------
# Forward-model reuse: signal extraction, recommendation gate, adoption
# ---------------------------------------------------------------------------

import json
import types
from datetime import datetime, timezone

from db.db import Simulation


async def _noop_persist(chat):
    return None


def _stub_signal_extraction(monkeypatch, extract):
    monkeypatch.setitem(
        sys.modules, "signal_extraction",
        types.SimpleNamespace(extract_and_prepare_batch=extract),
    )


def test_record_simulation_outputs_extracts_signals(monkeypatch):
    chat = api._new_chat_session("session-x")
    manifest = {"files": [{"sample_id": 1, "filename": "demo_1.in"}]}
    result = {
        "output_dir": "/abs/out_files",
        "outputs": [{"filename": "demo_1.in", "out_file": "/abs/demo_1.out"}],
    }
    monkeypatch.setattr(api, "set_simulation_outputs", lambda *a: 1)
    seen = {}

    def fake_extract(out_dir, session_uuid):
        seen["out_dir"] = str(out_dir)
        seen["session"] = session_uuid
        return {"updates": [{"id": "row-1", "signal_ez": [0.1], "signal_length": 1}]}

    _stub_signal_extraction(monkeypatch, fake_extract)
    monkeypatch.setattr(api, "bulk_update_signals", lambda updates: len(updates))

    assert api._record_simulation_outputs(chat, manifest, result) == (1, 1)
    assert seen["out_dir"] == "/abs/out_files"
    assert seen["session"] == api._coerce_uuid("session-x")


def test_record_simulation_outputs_signal_failure_swallowed(monkeypatch):
    chat = api._new_chat_session("session-x")
    manifest = {"files": [{"sample_id": 1, "filename": "demo_1.in"}]}
    result = {
        "output_dir": "/abs/out_files",
        "outputs": [{"filename": "demo_1.in", "out_file": "/abs/demo_1.out"}],
    }
    monkeypatch.setattr(api, "set_simulation_outputs", lambda *a: 1)

    def broken_extract(out_dir, session_uuid):
        raise RuntimeError("h5py exploded")

    _stub_signal_extraction(monkeypatch, broken_extract)
    assert api._record_simulation_outputs(chat, manifest, result) == (1, 0)


def _dataset_chat(tmp_path, session_id, manifest_extra=None):
    """A chat session with an on-disk emitted dataset (mirrors the guard test)."""
    chat = api._new_chat_session(session_id)
    api.sessions[session_id] = chat
    in_dir = tmp_path / "in_files"
    _write_in_files(in_dir, ["demo_1.in"])
    manifest = {
        "n_written": 1,
        "in_dir": str(in_dir),
        "output_dir": str(tmp_path),
        "files": [{"sample_id": 1, "filename": "demo_1.in"}],
    }
    manifest.update(manifest_extra or {})
    (tmp_path / "emitted_files.json").write_text(json.dumps(manifest))
    chat.state["dataset_config"] = {
        "num_samples": 1,
        "model_basename": "demo",
        "output_dir": str(tmp_path),
    }
    return chat


def test_simulate_gate_recommends_and_does_not_start(tmp_path, monkeypatch):
    chat = _dataset_chat(tmp_path, "gate-session")
    try:
        rec = {"source_session_id": "src-1", "similarity_pct": 97.0,
               "num_samples": 5, "requested_samples": 1,
               "source_user_id": "u1", "params_diff": []}
        monkeypatch.setattr(api, "_find_reuse_candidate", lambda c: rec)
        monkeypatch.setattr(api, "_persist_chat", _noop_persist)

        resp = asyncio.run(api.start_forward_model("gate-session"))

        assert resp["status"] == "reuse_recommended"
        assert resp["recommendation"] == rec
        assert chat.reuse_recommendation == rec
        assert chat.simulating is False  # run never started
        recorded = [e for e in chat.transcript if e["type"] == "reuse_recommendation"]
        assert len(recorded) == 1
        assert recorded[0]["recommendation"] == rec
    finally:
        api.sessions.pop("gate-session", None)


def test_simulate_force_bypasses_gate(tmp_path, monkeypatch):
    chat = _dataset_chat(tmp_path, "force-session")
    try:
        def must_not_run(c):
            raise AssertionError("gate must be skipped with force=true")

        monkeypatch.setattr(api, "_find_reuse_candidate", must_not_run)
        monkeypatch.setattr(api, "_persist_chat", _noop_persist)

        async def fake_run(*a, **k):
            chat.simulating = False

        monkeypatch.setattr(api, "_run_forward_model", fake_run)
        resp = asyncio.run(api.start_forward_model("force-session", force=True))
        assert resp["status"] == "started"
    finally:
        api.sessions.pop("force-session", None)


def test_simulate_gate_skipped_for_uploads(tmp_path, monkeypatch):
    chat = _dataset_chat(tmp_path, "upload-session", {"source": "upload"})
    try:
        def must_not_run(c):
            raise AssertionError("gate must be skipped for uploaded datasets")

        monkeypatch.setattr(api, "_find_reuse_candidate", must_not_run)
        monkeypatch.setattr(api, "_persist_chat", _noop_persist)

        async def fake_run(*a, **k):
            chat.simulating = False

        monkeypatch.setattr(api, "_run_forward_model", fake_run)
        resp = asyncio.run(api.start_forward_model("upload-session"))
        assert resp["status"] == "started"
    finally:
        api.sessions.pop("upload-session", None)


def test_simulate_gate_similarity_failure_falls_through(tmp_path, monkeypatch):
    """A broken similarity stack must never block the run (real
    _find_reuse_candidate + a raising search)."""
    chat = _dataset_chat(tmp_path, "fallthrough-session")
    try:
        def boom(*a, **k):
            raise RuntimeError("qdrant down")

        monkeypatch.setattr(api.sim_similarity, "find_similar_session", boom)
        monkeypatch.setattr(api, "_persist_chat", _noop_persist)

        async def fake_run(*a, **k):
            chat.simulating = False

        monkeypatch.setattr(api, "_run_forward_model", fake_run)
        resp = asyncio.run(api.start_forward_model("fallthrough-session"))
        assert resp["status"] == "started"
        assert chat.reuse_recommendation is None
    finally:
        api.sessions.pop("fallthrough-session", None)


def _make_source_dataset(src_dir, filenames, with_manifests=True, missing_out=()):
    src_in = src_dir / "in_files"
    src_out = src_dir / "out_files"
    _write_in_files(src_in, filenames)
    src_out.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        stem = Path(name).stem
        if stem + ".out" not in missing_out:
            (src_out / (stem + ".out")).write_bytes(b"\x89HDF")
    manifest = {
        "n_written": len(filenames),
        "output_dir": str(src_dir),
        "in_dir": str(src_in),
        "files": [
            {"sample_id": i + 1, "filename": n, "path": str(src_in / n)}
            for i, n in enumerate(filenames)
        ],
    }
    (src_dir / "emitted_files.json").write_text(json.dumps(manifest))
    if with_manifests:
        for m in ("sampled_layers.json", "derived_layers.json", "global_derive.json"):
            (src_dir / m).write_text("{}")
    return manifest


def _make_source_rows(src_dir, session_uuid, filenames):
    rows = []
    for i, name in enumerate(filenames):
        stem = Path(name).stem
        rows.append(Simulation(
            session_id=session_uuid,
            user_id="source-user",
            sample_index=i + 1,
            layers=[{"name": "topsoil", "thickness_m": 0.3}],
            num_layers=1,
            signal_ez=[0.1, 0.2, 0.3],
            signal_length=3,
            simulation_completed_at=datetime.now(timezone.utc),
            input_file_path=str(src_dir / "in_files" / name),
            output_file_path=str(src_dir / "out_files" / (stem + ".out")),
        ))
    return rows


def test_adopt_copies_files_and_rekeys_rows(tmp_path, monkeypatch):
    cur_dir = tmp_path / "current"
    src_dir = tmp_path / "source"
    cur_dir.mkdir()
    _make_source_dataset(src_dir, ["soil_1.in", "soil_2.in"])
    src_uuid = api._coerce_uuid("src-session")
    src_rows = _make_source_rows(src_dir, src_uuid, ["soil_1.in", "soil_2.in"])

    chat = api._new_chat_session("adopt-session")
    api.sessions["adopt-session"] = chat
    try:
        chat.state["dataset_config"] = {
            "num_samples": 2, "model_basename": "demo", "output_dir": str(cur_dir),
        }
        chat.reuse_recommendation = {
            "source_session_id": "src-session", "similarity_pct": 97.2,
            "num_samples": 2, "source_output_dir": str(src_dir),
            "source_user_id": "source-user",
        }
        inserted = {}
        monkeypatch.setattr(api, "get_extraction_session", lambda u: None)
        monkeypatch.setattr(api, "get_simulations_for_session", lambda u: src_rows)
        monkeypatch.setattr(api, "delete_simulations_for_session",
                            lambda u: inserted.setdefault("deleted", u))
        monkeypatch.setattr(api, "batch_insert_simulations",
                            lambda rows: inserted.setdefault("rows", rows) and len(rows))
        monkeypatch.setattr(api, "_persist_chat", _noop_persist)

        result = asyncio.run(api.adopt_dataset(
            "adopt-session", api.AdoptDatasetPayload(source_session_id="src-session")
        ))

        assert result["status"] == "adopted"
        assert result["adopted_from"] == "src-session"
        # files + manifests copied, manifest rewritten to current paths
        assert (cur_dir / "in_files" / "soil_1.in").is_file()
        assert (cur_dir / "out_files" / "soil_2.out").is_file()
        assert (cur_dir / "sampled_layers.json").is_file()
        new_manifest = json.loads((cur_dir / "emitted_files.json").read_text())
        assert new_manifest["output_dir"] == str(cur_dir)
        assert new_manifest["in_dir"] == str(cur_dir / "in_files")
        assert new_manifest["adopted_from"] == "src-session"
        assert all(f["path"].startswith(str(cur_dir)) for f in new_manifest["files"])
        # rows re-keyed to this session, signals carried, paths repointed
        rows = inserted["rows"]
        assert inserted["deleted"] == api._coerce_uuid("adopt-session")
        assert len(rows) == 2
        assert all(r["session_id"] == api._coerce_uuid("adopt-session") for r in rows)
        assert all(r["user_id"] == chat.user_id for r in rows)
        assert [r["sample_index"] for r in rows] == [1, 2]
        assert all(r["signal_ez"] == [0.1, 0.2, 0.3] for r in rows)
        assert all(r["id"] not in {s.id for s in src_rows} for r in rows)
        assert rows[0]["input_file_path"] == str(cur_dir / "in_files" / "soil_1.in")
        assert rows[0]["output_file_path"] == str(cur_dir / "out_files" / "soil_1.out")
        # chat state converged
        assert chat.reuse_recommendation is None
        assert chat.dataset_result["status"] == "adopted"
        assert chat.simulation_result["succeeded"] == 2
        assert chat.simulation_result["adopted_from"] == "src-session"
        assert all(chat.viz_flags.values())
        types_seen = [e["type"] for e in chat.transcript]
        assert "dataset_ready" in types_seen and "simulation_complete" in types_seen
        assert chat.simulating is False
    finally:
        api.sessions.pop("adopt-session", None)


def test_adopt_guards(tmp_path, monkeypatch):
    chat = _dataset_chat(tmp_path, "adopt-guard")
    try:
        payload = api.AdoptDatasetPayload(source_session_id="src-session")

        # no pending recommendation
        with pytest.raises(HTTPException) as exc:
            asyncio.run(api.adopt_dataset("adopt-guard", payload))
        assert exc.value.status_code == 409

        # mismatched recommendation
        chat.reuse_recommendation = {"source_session_id": "someone-else"}
        with pytest.raises(HTTPException) as exc:
            asyncio.run(api.adopt_dataset("adopt-guard", payload))
        assert exc.value.status_code == 409

        # busy states
        chat.reuse_recommendation = {"source_session_id": "src-session"}
        for flag in ("simulating", "busy", "regenerating"):
            setattr(chat, flag, True)
            with pytest.raises(HTTPException) as exc:
                asyncio.run(api.adopt_dataset("adopt-guard", payload))
            assert exc.value.status_code == 409
            setattr(chat, flag, False)
    finally:
        api.sessions.pop("adopt-guard", None)


def test_adopt_verifies_before_deleting(tmp_path, monkeypatch):
    """A missing source .out must 409 BEFORE the current dataset is touched."""
    cur_dir = tmp_path / "current"
    src_dir = tmp_path / "source"
    _write_in_files(cur_dir / "in_files", ["mine_1.in"])
    marker = cur_dir / "in_files" / "mine_1.in"
    _make_source_dataset(src_dir, ["soil_1.in"], missing_out={"soil_1.out"})

    chat = api._new_chat_session("adopt-verify")
    api.sessions["adopt-verify"] = chat
    try:
        chat.state["dataset_config"] = {
            "num_samples": 1, "model_basename": "demo", "output_dir": str(cur_dir),
        }
        chat.reuse_recommendation = {
            "source_session_id": "src-session",
            "source_output_dir": str(src_dir),
        }
        monkeypatch.setattr(api, "get_extraction_session", lambda u: None)
        monkeypatch.setattr(api, "_persist_chat", _noop_persist)

        with pytest.raises(HTTPException) as exc:
            asyncio.run(api.adopt_dataset(
                "adopt-verify", api.AdoptDatasetPayload(source_session_id="src-session")
            ))
        assert exc.value.status_code == 409
        assert marker.is_file()  # current dataset untouched
        assert chat.reuse_recommendation is not None  # still pending
    finally:
        api.sessions.pop("adopt-verify", None)


def _forward_model_chat(tmp_path, session_id):
    chat = api._new_chat_session(session_id)
    chat.state["dataset_config"] = {
        "num_samples": 1, "model_basename": "demo", "output_dir": str(tmp_path),
    }
    return chat


def test_forward_model_reports_gpu_transmission_line_preflight_once(tmp_path, monkeypatch):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("GPR_GPU", "1")
    chat = _forward_model_chat(tmp_path, "preflight-session")
    chat.simulating = True
    in_dir = tmp_path / "in_files"
    names = ["a.in", "b.in"]
    _write_in_files(in_dir, names)
    for name in names:
        (in_dir / name).write_text("#transmission_line: z 0 0 0 75 pulse\n")
    manifest = {"files": [{"sample_id": i, "filename": name} for i, name in enumerate(names, 1)]}

    monkeypatch.setattr(api, "run_batch_simulation", simulate.run_batch_simulation)
    monkeypatch.setattr(api, "_persist_chat", _noop_persist)
    asyncio.run(api._run_forward_model(chat, manifest, in_dir, tmp_path / "out_files", names))

    assert chat.simulating is False
    complete = [e for e in chat.transcript if e["type"] == "simulation_complete"]
    assert len(complete) == 1
    assert "Forward model failed to start" in complete[0]["content"]
    assert "transmission_line sources require CPU solving" in complete[0]["content"]
    assert len(complete[0]["result"]["errors"]) == 1
    assert not any(e["type"] == "simulation_progress" for e in chat.transcript)


def test_forward_model_indexes_successful_run(tmp_path, monkeypatch):
    chat = _forward_model_chat(tmp_path, "index-session")
    manifest = {"files": [{"sample_id": 1, "filename": "demo_1.in"}]}
    result = {"succeeded": 1, "failed": 0, "skipped": 0, "total": 1,
              "output_dir": str(tmp_path / "out_files"), "outputs": [], "errors": []}

    async def run_thread(fn, *a, **k):
        return fn(*a, **k)

    monkeypatch.setattr(api, "run_batch_simulation", lambda **k: result)
    monkeypatch.setattr(api, "_record_simulation_outputs", lambda *a: (1, 1))
    monkeypatch.setattr(api, "_persist_chat", _noop_persist)
    indexed = {}

    def fake_index(state, **meta):
        indexed.update(meta)
        return True

    monkeypatch.setattr(api.sim_similarity, "index_completed_session", fake_index)
    asyncio.run(api._run_forward_model(
        chat, manifest, tmp_path / "in_files", tmp_path / "out_files", ["demo_1.in"]
    ))

    assert indexed["session_id"] == str(api._coerce_uuid("index-session"))
    assert indexed["num_samples"] == 1
    assert indexed["output_dir"] == str(tmp_path)
    assert chat.simulation_result["signals_updated"] == 1


def test_forward_model_skips_index_on_failures(tmp_path, monkeypatch):
    chat = _forward_model_chat(tmp_path, "index-fail-session")
    manifest = {"files": []}
    result = {"succeeded": 0, "failed": 1, "skipped": 0, "total": 1,
              "output_dir": str(tmp_path / "out_files"), "outputs": [],
              "errors": [{"filename": "demo_1.in", "error": "boom"}]}

    monkeypatch.setattr(api, "run_batch_simulation", lambda **k: result)
    monkeypatch.setattr(api, "_record_simulation_outputs", lambda *a: (0, 0))
    monkeypatch.setattr(api, "_persist_chat", _noop_persist)

    def must_not_index(*a, **k):
        raise AssertionError("failed runs must not be indexed")

    monkeypatch.setattr(api.sim_similarity, "index_completed_session", must_not_index)
    asyncio.run(api._run_forward_model(
        chat, manifest, tmp_path / "in_files", tmp_path / "out_files", ["demo_1.in"]
    ))
    assert chat.simulation_result["failed"] == 1


def test_forward_model_index_errors_swallowed(tmp_path, monkeypatch):
    chat = _forward_model_chat(tmp_path, "index-boom-session")
    manifest = {"files": []}
    result = {"succeeded": 1, "failed": 0, "skipped": 0, "total": 1,
              "output_dir": str(tmp_path / "out_files"), "outputs": [], "errors": []}

    monkeypatch.setattr(api, "run_batch_simulation", lambda **k: result)
    monkeypatch.setattr(api, "_record_simulation_outputs", lambda *a: (1, 1))
    monkeypatch.setattr(api, "_persist_chat", _noop_persist)

    def boom(*a, **k):
        raise RuntimeError("qdrant down")

    monkeypatch.setattr(api.sim_similarity, "index_completed_session", boom)
    asyncio.run(api._run_forward_model(
        chat, manifest, tmp_path / "in_files", tmp_path / "out_files", ["demo_1.in"]
    ))
    assert chat.simulation_result["succeeded"] == 1  # run completed normally
