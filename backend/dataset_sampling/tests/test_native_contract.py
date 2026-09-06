"""Native adapter integration plus one fully derived, UNQUALIFIED 3D fixture.

GPR_RUN_NATIVE_TESTS=1 python -m pytest backend/dataset_sampling/tests/test_native_contract.py -q
The small fixtures isolate serialization/material/geometry/output parity.
The final dry-soil fixture executes the complete common plan without overrides;
neither fixture family establishes scientific convergence or population coverage.
"""
import json
import os
from pathlib import Path

import pytest

from backend.dataset_sampling.tests.test_3d_contract import inputs
from backend.dataset_sampling.layer_sampler import sample_layers, write_samples
from backend.dataset_sampling.peplinski_derive import derive_samples, write_derived
from backend.dataset_sampling.global_derive import derive_global, write_global
from backend.dataset_sampling.numerics import native_dt, time_axis
from backend.dataset_sampling.emit import emit_dataset
from backend.schema import SampledTarget, ExtractedAdvancedParams, ExtractedTargetRanges
from backend.preflight import native_build_checks
from backend.simulate import inject_output_dir, _reset_gprmax_state, _execute_one, run_batch_simulation

pytestmark = pytest.mark.skipif(os.environ.get("GPR_RUN_NATIVE_TESTS") != "1", reason="Opt-in native CPU integration")


def native_fixture(tmp_path, kind="hertzian_dipole", axis="z", mode="3D", rough=False):
    cfg, layers, wf, ant = inputs(tmp_path, mode, axis)
    cfg = cfg.model_copy(update={"num_samples": 1, "num_threads": 1, "fractal_nbins": 8})
    layers = layers.model_copy(update={"num_layers": 1, "layers": [layers.layers[0]]})
    ant = ant.model_copy(update={"antenna_kind": kind, "resistance": 75 if kind != "hertzian_dipole" else None})
    samples, _ = sample_layers(layers, 1, dataset_config=cfg)
    derived, agg = derive_samples(samples, cfg, wf)
    grid = derive_global(cfg, wf, ant, layers, agg.eps_r_max, agg.eps_r_min,
                         spectral_lambda_min_m=agg.spectral_lambda_min_m,
                         spectral_index_max=agg.spectral_index_max,
                         min_layer_thickness_m=agg.min_layer_thickness_m,
                         min_relaxation_time_s=agg.min_relaxation_time_s)
    dx = .004
    dt = native_dt(dx, mode)
    iterations, final = time_axis(dt, 8e-9)
    grid = grid.model_copy(update={"dx_m": dx, "nx": 100, "ny": 100, "nz": 100 if mode == "3D" else 1,
        "domain_x_m": .4, "domain_y_m": .4, "domain_z_m": .4 if mode == "3D" else dx,
        "ground_y_m": .2, "soil_depth_m": .2, "depth_z_m": .2,
        "tx_x_m": .2, "tx_y_m": .26, "tx_z_m": .2 if mode == "3D" else 0,
        "rx_x_m": .232, "rx_y_m": .26, "rx_z_m": .212 if mode == "3D" else 0,
        "dt_s": dt, "iterations": iterations, "time_window_s": final,
        "requested_time_window_s": 8e-9,
        "derivation": {**grid.derivation, "integration_fixture_only": True}})
    if mode == "3D":
        samples[0].targets = [SampledTarget(kind="cylinder", name="pipe", x_offset_m=-.048,
            z_offset_m=0, depth_m=.052, radius_m=.02, length_m=.04, cylinder_axis=axis),
            SampledTarget(kind="box", name="block", x_offset_m=.048, z_offset_m=0,
            depth_m=.052, width_m=.04, height_m=.04, crossline_size_m=.04)]
    write_samples(samples, str(tmp_path), [])
    write_derived(derived, agg, str(tmp_path))
    write_global(grid, str(tmp_path))
    advanced = ExtractedAdvancedParams(surface_roughness={"amplitude_m": .008} if rough else None,
        snapshots=[dict(filename="field_sample", time_s=1.73e-9, dx=.04, dy=.04, dz=.04 if mode == "3D" else dx)])
    emit_dataset(str(tmp_path), cfg, wf, ant, advanced, layers=layers)
    return json.loads((tmp_path / "emitted_files.json").read_text())


@pytest.mark.parametrize("kind", ["hertzian_dipole", "voltage_source", "transmission_line"])
@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_native_build_source_axes_finite_targets_and_material_parity(tmp_path, kind, axis):
    from gprMax.gprMax import api
    manifest = native_fixture(tmp_path, kind, axis)
    entry = manifest["files"][0]
    path = Path(entry["path"])
    path.write_text(inject_output_dir(path.read_text(), tmp_path))
    try:
        with native_build_checks(manifest["contract"], entry["resolved_scene"], tmp_path / "fixture.out") as receipt:
            api(str(path), geometry_only=True)
        assert receipt["status"] == "passed"
        assert receipt["mode"] == "3D"
        assert all(t["occupied_pec_voxels"] > 0 for t in receipt["targets"])
        assert len(receipt["material_table_digests"]) == 1
        import h5py
        import numpy as np
        with h5py.File(receipt["geometry_file"]) as f:
            # A volume must contain genuine crossline variation, not tiled 2D.
            assert not np.array_equal(f["solid"][:, :30, 25], f["solid"][:, :30, 65])
    finally:
        _reset_gprmax_state()


@pytest.mark.parametrize("mode,rough,kind,axis", [("3D", False, "hertzian_dipole", "z"),
    ("2D", False, "hertzian_dipole", "z"), ("2D", True, "hertzian_dipole", "z"),
    ("3D", False, "voltage_source", "y"), ("3D", False, "transmission_line", "x")])
def test_native_field_execution_and_receipt(tmp_path, mode, rough, kind, axis):
    manifest = native_fixture(tmp_path, kind=kind, axis=axis, mode=mode, rough=rough)
    entry = manifest["files"][0]
    out = tmp_path / "out_files"
    out.mkdir()
    if mode == "3D":
        batch = run_batch_simulation(tmp_path / "in_files", out, gpu=False, gpu_ids=[], manifest=manifest)
        assert batch["succeeded"] == 1, batch["errors"]
        result = batch["outputs"][0]
        skipped = run_batch_simulation(tmp_path / "in_files", out, gpu=False, gpu_ids=[], manifest=manifest, skip_existing=True)
        assert skipped["skipped"] == 1
    else:
        result = _execute_one({"in_file": entry["path"], "output_dir": str(out),
            "tmp_dir": str(out / "_tmp"), "n": 1, "gpu_arg": None, "verbose": False,
            "index": 1, "contract": manifest["contract"], "entry": entry})
        assert result["status"] == "ok", result.get("error")
    from backend.signal_extraction import validate_output
    metadata = validate_output(result["out_file"], manifest["contract"], entry["resolved_scene"],
                                input_sha256=entry["input_sha256"], require_receipt=True)
    assert metadata["cell_counts"][2] == (100 if mode == "3D" else 1)


def test_native_full_derived_3d_plan(tmp_path):
    """One declared dry-soil experiment, with no grid/domain/time overrides.

    This validates end-to-end planning/execution, not population convergence.
    It needs approximately 4.14 GiB plus the host reserve; admission still applies.
    """
    from backend.dataset_sampling.global_validation import validate_global
    from backend.dataset_sampling.target_placement import validate_and_place
    from backend.signal_extraction import validate_output
    cfg, layers, wf, ant = inputs(tmp_path, axis="z")
    cfg = cfg.model_copy(update={"num_samples": 1, "num_threads": 4, "fractal_nbins": 8})
    layer = layers.layers[0].model_copy(update={"thickness_m_min": .08, "thickness_m_max": .08,
                                               "theta_v_min": .001, "theta_v_max": .005})
    layers = layers.model_copy(update={"num_layers": 1, "layers": [layer]})
    ranges = ExtractedTargetRanges(boxes=[dict(name="block", x_offset_min_m=0, x_offset_max_m=0,
        z_offset_min_m=0, z_offset_max_m=0, depth_min_m=.05, depth_max_m=.05,
        width_min_m=.05, width_max_m=.05, height_min_m=.05, height_max_m=.05,
        crossline_size_min_m=.05, crossline_size_max_m=.05)])
    samples, warnings = sample_layers(layers, 1, dataset_config=cfg, target_ranges=ranges)
    derived, agg = derive_samples(samples, cfg, wf, ranges)
    keys = {"smallest_feature_global_m", "largest_extent_global_m", "deepest_target_bottom_global_m",
            "static_x_halfwidth_global_m", "z_halfwidth_global_m", "spectral_lambda_min_m",
            "spectral_index_max", "min_layer_thickness_m", "min_relaxation_time_s"}
    grid = derive_global(cfg, wf, ant, layers, agg.eps_r_max, agg.eps_r_min,
                         **{k: v for k, v in agg.model_dump().items() if k in keys})
    assert validate_global(grid, cfg, wf, ant, layers, target_ranges=ranges).ok
    frozen = grid.model_dump()
    placed = validate_and_place(samples, grid, cfg, ranges)
    assert len(placed.surviving) == 1
    write_samples(placed.surviving, str(tmp_path), warnings)
    write_derived(derived, agg, str(tmp_path))
    write_global(grid, str(tmp_path))
    assert emit_dataset(str(tmp_path), cfg, wf, ant, layers=layers, target_ranges=ranges).n_written == 1
    manifest = json.loads((tmp_path / "emitted_files.json").read_text())
    result = run_batch_simulation(tmp_path / "in_files", tmp_path / "out_files",
                                  gpu=False, gpu_ids=[], workers=1, manifest=manifest)
    assert result["succeeded"] == 1, result["errors"]
    entry = manifest["files"][0]
    metadata = validate_output(result["outputs"][0]["out_file"], manifest["contract"], entry["resolved_scene"],
                               input_sha256=entry["input_sha256"], require_receipt=True)
    assert metadata["cell_counts"] == [grid.nx, grid.ny, grid.nz]
    assert metadata["dt_s"] == grid.dt_s
    assert metadata["iterations"] == grid.iterations
    assert grid.model_dump() == frozen
