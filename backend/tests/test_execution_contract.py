"""Integrity, admission, release and reuse failure paths without field solves."""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.dataset_sampling.tests.test_native_contract import native_fixture
from backend.dataset_sampling.contract import digest, file_digest
from backend.signal_extraction import validate_output
from backend.simulate import run_batch_simulation, ExecutionPlan
from backend.resources import admit
from backend.qualification import assess, population_digest, qualification_status


@pytest.fixture
def fixture(tmp_path):
    manifest = native_fixture(tmp_path)
    entry, contract = manifest["files"][0], manifest["contract"]
    scene, grid = entry["resolved_scene"], contract["grid"]
    out = tmp_path / "out_files"
    out.mkdir()
    path = out / (Path(entry["filename"]).stem + ".out")
    with h5py.File(path, "w") as f:
        for key, value in {"dt": grid["dt_s"], "Iterations": grid["iterations"],
            "nx_ny_nz": [grid[k] for k in ("nx", "ny", "nz")], "dx_dy_dz": [grid["dx_m"]] * 3,
            "gprMax": contract["solver"]["version"], "Title": scene["title"],
            "nrx": 1, "nsrc": 1, "srcsteps": [0, 0, 0], "rxsteps": [0, 0, 0]}.items():
            f.attrs[key] = value
        rx = f.create_group("rxs/rx1")
        rx.attrs["Position"] = scene["receiver"]["position_m"]
        src = f.create_group("srcs/src1")
        src.attrs["Position"] = scene["source"]["position_m"]
        src.attrs["Type"] = "HertzianDipole"
        for field in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            rx.create_dataset(field, data=np.zeros(grid["iterations"], dtype=np.float32))
    geometry = path.with_suffix(".geometry.h5")
    geometry.write_bytes(b"fixture geometry")
    receipt = {"contract_digest": contract["digest"], "scene_digest": scene["digest"],
        "input_sha256": entry["input_sha256"], "output_sha256": file_digest(path),
        "preflight": {"status": "passed", "geometry_sha256": file_digest(geometry)}, "backend": "cpu"}
    receipt["snapshots"] = []
    for snapshot in scene["snapshots"]:
        relative = Path(path.stem + "_snaps") / (snapshot["filename"] + ".vti")
        (out / relative).parent.mkdir(exist_ok=True)
        (out / relative).write_bytes(b"snapshot fixture")
        receipt["snapshots"].append({"path": str(relative), "sha256": file_digest(out / relative)})
    path.with_suffix(".execution.json").write_text(json.dumps(receipt))
    return tmp_path, manifest, entry, path


def check(fixture):
    root, manifest, entry, path = fixture
    return validate_output(path, manifest["contract"], entry["resolved_scene"],
                           input_sha256=entry["input_sha256"], require_receipt=True)


def test_zero_components_are_valid_but_still_unqualified(fixture):
    root, manifest, entry, path = fixture
    assert check(fixture)["iterations"] == manifest["contract"]["grid"]["iterations"]
    assert not qualification_status(root, manifest)["reuse_eligible"]
    from backend.sim_similarity import eligible_manifest
    assert eligible_manifest(root) is None


@pytest.mark.parametrize("change", ["dt", "counts", "identity", "position", "short", "nan", "missing", "receipt", "geometry"])
def test_corrupt_or_stale_output_rejected(fixture, change):
    root, manifest, entry, path = fixture
    if change == "receipt":
        path.with_suffix(".execution.json").write_text("{}")
    elif change == "geometry":
        path.with_suffix(".geometry.h5").write_bytes(b"changed")
    else:
        with h5py.File(path, "r+") as f:
            if change == "dt": f.attrs["dt"] *= 2
            if change == "counts": f.attrs["nx_ny_nz"] = [100, 100, 1]
            if change == "identity": f.attrs["Title"] = "different_sample"
            if change == "position": f["rxs/rx1"].attrs["Position"] = [0, 0, 0]
            if change == "nan": f["rxs/rx1/Hy"][3] = float("nan")
            if change in ("missing", "short"):
                del f["rxs/rx1/Hx"]
                if change == "short": f["rxs/rx1"].create_dataset("Hx", data=[0.0])
    with pytest.raises((ValueError, KeyError)):
        check(fixture)


@pytest.mark.parametrize("change", ["input", "scene", "contract", "n", "name", "duplicate"])
def test_manifest_admission_rejects_before_any_worker(fixture, change, monkeypatch):
    from backend import simulate
    root, manifest, entry, path = fixture
    monkeypatch.setattr(simulate, "_make_pool", lambda *_: pytest.fail("must reject before worker allocation"))
    args = {}
    if change == "input": Path(entry["path"]).write_text("#domain: .4 .4 .004\n")
    if change == "scene": entry["resolved_scene"]["source"]["axis"] = "y"
    if change == "contract": manifest["contract"]["dimensionality"] = "2D"
    if change == "n": args["n"] = 2
    if change == "name": args["filenames"] = ["stale.in"]
    if change == "duplicate": manifest["files"].append(copy.deepcopy(entry))
    with pytest.raises(ValueError):
        run_batch_simulation(root / "in_files", gpu=False, gpu_ids=[], manifest=manifest, **args)


def test_cpu_admission_reduces_workers_without_modifying_scene(tmp_path, monkeypatch):
    import psutil
    monkeypatch.setenv("GPR_HOST_RESERVE_BYTES", "0")
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=2500))
    estimate = {"host_peak_bytes": 1000, "device_peak_bytes": 600, "coefficient_bytes": 20,
                "output_bytes": 10, "scratch_bytes": 10}
    before = copy.deepcopy(estimate)
    assert admit(ExecutionPlan(workers=4), estimate, tmp_path, 3).workers == 2
    assert estimate == before
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=999))
    with pytest.raises(ValueError, match="host RAM"):
        admit(ExecutionPlan(), estimate, tmp_path, 1)


def test_gpu_requires_per_device_capacity_and_caps_one_worker(tmp_path, monkeypatch):
    import subprocess
    import psutil
    monkeypatch.setenv("GPR_HOST_RESERVE_BYTES", "0")
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=10000))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=json.dumps({"0": {"free": 700, "constant": 100}})))
    estimate = {"host_peak_bytes": 1000, "device_peak_bytes": 600, "coefficient_bytes": 20,
                "output_bytes": 10, "scratch_bytes": 10}
    assert admit(ExecutionPlan(gpu=True, gpu_ids=[0], workers=4), estimate, tmp_path, 3).workers == 1
    estimate["coefficient_bytes"] = 101
    with pytest.raises(ValueError, match="coefficient"):
        admit(ExecutionPlan(gpu=True, gpu_ids=[0]), estimate, tmp_path, 1)


def test_qualification_is_reviewed_and_separate_from_immutable_identity(fixture):
    root, manifest, entry, path = fixture
    identity = {"contract_digest": manifest["contract"]["digest"], "population_digest": population_digest(manifest)}
    evidence = {"solver": manifest["contract"]["solver"], "covered_experiments": [identity], "error": .02}
    criteria = {"approved_experiments": [identity], "approved_by": "test reviewer", "intended_use": "test only", "maximum_errors": {"error": .01}, "allowed_backends": ["cpu"]}
    with pytest.raises(ValueError, match="criterion failed"):
        assess(manifest, evidence, criteria)
    criteria["maximum_errors"]["error"] = .03
    original = copy.deepcopy(manifest)
    from backend.qualification import record_qualification
    record_qualification(root, evidence, criteria)
    assert qualification_status(root, manifest)["reuse_eligible"]
    assert manifest == original
    from backend.sim_similarity import eligible_manifest, equivalent_contracts
    assert eligible_manifest(root) is not None
    changed = copy.deepcopy(manifest)
    changed["files"][0]["input_sha256"] = "changed"
    assert not equivalent_contracts(manifest, changed)
    assert not qualification_status(root, changed)["reuse_eligible"]


def test_mode_and_execution_labels_survive_projection_and_storage(fixture):
    from backend.viz_projection import build_scene
    from backend.schema import DatasetConfig, ExtractedWaveform, ExtractedAntenna
    from backend.api import _build_simulation_rows
    import uuid
    root, manifest, entry, path = fixture
    requested = manifest["contract"]["requested"]
    scene = build_scene(requested, {"emitted": True, "sampled": True, "grid": True}, root)
    assert scene["dimensionality"] == "3D"
    assert scene["samples"]["items"][0]["resolved_scene"]["targets"] == entry["resolved_scene"]["targets"]
    rows = _build_simulation_rows(session_uuid=uuid.uuid4(), user_id=None,
        cfg=DatasetConfig.model_validate(requested["dataset_config"]),
        wf=ExtractedWaveform.model_validate(requested["waveform"]),
        ant=ExtractedAntenna.model_validate(requested["antenna"]), adv=None,
        sampled_manifest=json.loads((root / "sampled_layers.json").read_text()),
        derived_manifest=json.loads((root / "derived_layers.json").read_text()),
        global_derive=manifest["contract"]["grid"], emitted_manifest=manifest)
    assert rows[0]["domain_z"] == .4
    assert rows[0]["contract_digest"] == manifest["contract"]["digest"]
    assert rows[0]["resolved_scene"] == entry["resolved_scene"]


def test_gpu_scheduler_reserves_device_until_its_worker_finishes(tmp_path, monkeypatch):
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor
    from backend import simulate, resources
    from backend.dataset_sampling.tests.test_3d_contract import plan
    from backend.dataset_sampling.emit import emit_dataset
    cfg, layers, wf, ant, grid, samples = plan(tmp_path)
    emit_dataset(str(tmp_path), cfg, wf, ant, layers=layers)
    manifest = json.loads((tmp_path / "emitted_files.json").read_text())
    monkeypatch.setattr(resources, "admit", lambda p, *a: p)
    monkeypatch.setattr(simulate, "_make_pool", lambda p: ThreadPoolExecutor(max_workers=p.workers))
    lock, active, assigned = threading.Lock(), set(), {}
    def execute(task):
        device = task["gpu_arg"][0][0]
        with lock:
            assert device not in active
            active.add(device)
            assigned[task["index"]] = device
        time.sleep(.04 if task["index"] == 1 else .002)
        with lock:
            active.remove(device)
        return {"status": "ok", "out_file": "unit-test-only.out"}
    monkeypatch.setattr(simulate, "_execute_one", execute)
    result = simulate.run_batch_simulation(tmp_path / "in_files", manifest=manifest, gpu=True, gpu_ids=[0, 1], workers=2)
    assert result["succeeded"] == 3
    assert assigned == {1: 0, 2: 1, 3: 1}


def test_transmission_line_gpu_rejected_before_resource_probe(tmp_path, monkeypatch):
    from backend import resources
    manifest = native_fixture(tmp_path, kind="transmission_line")
    monkeypatch.setattr(resources, "admit", lambda *a: pytest.fail("source must fail before resource probe"))
    with pytest.raises(ValueError, match="require CPU"):
        run_batch_simulation(tmp_path / "in_files", manifest=manifest, gpu=True, gpu_ids=[0])
