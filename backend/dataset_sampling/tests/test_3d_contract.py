"""New contract invariants; independent of accounts, databases or a GPU."""
import json
import math
import random
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from backend.schema import (DatasetConfig, ExtractedLayers, ExtractedWaveform,
                            ExtractedAntenna, ExtractedTargetRanges, SampledTarget,
                            ExtractedAdvancedParams)
from backend.dataset_sampling.contract import validate_capabilities
from backend.dataset_sampling.layer_sampler import sample_layers, write_samples
from backend.dataset_sampling.peplinski_derive import derive_samples, write_derived, native_material_table
from backend.dataset_sampling.global_derive import derive_global, write_global
from backend.dataset_sampling.global_validation import validate_global
from backend.dataset_sampling.numerics import native_dt, cell_index
from backend.dataset_sampling.scene import resolve_scene, resolve_target, validate_resolved_target
from backend.dataset_sampling.target_placement import validate_and_place
from backend.dataset_sampling.emit import emit_dataset


def inputs(tmp_path, mode="3D", axis="x"):
    cfg = DatasetConfig(num_samples=3, dimensionality=mode, contract_version=2, output_dir=str(tmp_path), seed=42)
    layers = ExtractedLayers(num_layers=2, layers=[dict(
        name=name, thickness_m_min=lo, thickness_m_max=hi,
        sand_pct_min=30, sand_pct_max=40, clay_pct_min=10, clay_pct_max=15,
        theta_v_min=.05, theta_v_max=.15, bulk_density_gcm3_min=1.4,
        bulk_density_gcm3_max=1.5, particle_density_gcm3_min=2.6, particle_density_gcm3_max=2.7)
        for name, lo, hi in [("top", .15, .2), ("bottom", .2, .3)]])
    wf = ExtractedWaveform(waveform_center_freq_hz=.78e9, waveform_name="pulse")
    ant = ExtractedAntenna(antenna_axis=axis, tx_rx_offset_m=.1,
                           tx_rx_crossline_offset_m=.06 if mode == "3D" else 0)
    return cfg, layers, wf, ant


def targets():
    return ExtractedTargetRanges(cylinders=[dict(name="pipe", x_offset_min_m=-.2, x_offset_max_m=-.1,
        z_offset_min_m=.20, z_offset_max_m=.25, depth_min_m=.25, depth_max_m=.25,
        radius_min_m=.025, radius_max_m=.04, length_min_m=.1, length_max_m=.15, cylinder_axis="x")],
        boxes=[dict(name="block", x_offset_min_m=.1, x_offset_max_m=.2,
        z_offset_min_m=-.2, z_offset_max_m=-.1, depth_min_m=.2, depth_max_m=.3,
        width_min_m=.06, width_max_m=.08, height_min_m=.05, height_max_m=.08,
        crossline_size_min_m=.07, crossline_size_max_m=.1)])


def plan(tmp_path, mode="3D", axis="x", tr=None):
    cfg, layers, wf, ant = inputs(tmp_path, mode, axis)
    samples, warnings = sample_layers(layers, cfg.num_samples, seed=cfg.seed, target_ranges=tr, dataset_config=cfg)
    derived, agg = derive_samples(samples, cfg, wf, tr)
    kw = {k: v for k, v in agg.model_dump().items() if k in {
        "smallest_feature_global_m", "largest_extent_global_m", "deepest_target_bottom_global_m",
        "static_x_halfwidth_global_m", "z_halfwidth_global_m", "spectral_lambda_min_m",
        "spectral_index_max", "min_layer_thickness_m", "min_relaxation_time_s"}}
    grid = derive_global(cfg, wf, ant, layers, agg.eps_r_max, agg.eps_r_min, **kw)
    assert validate_global(grid, cfg, wf, ant, layers, target_ranges=tr).ok
    survivors = validate_and_place(samples, grid, cfg, tr)
    assert len(survivors.surviving) == cfg.num_samples
    write_samples(survivors.surviving, str(tmp_path), warnings)
    write_derived(derived, agg, str(tmp_path))
    write_global(grid, str(tmp_path))
    return cfg, layers, wf, ant, grid, survivors.surviving


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_full_chain_emits_finite_3d_and_exact_shared_contract(tmp_path, axis):
    tr = targets()
    cfg, layers, wf, ant, grid, samples = plan(tmp_path, axis=axis, tr=tr)
    frozen = grid.model_dump()
    result = emit_dataset(str(tmp_path), cfg, wf, ant, layers=layers, target_ranges=tr)
    assert result.errors == []
    assert result.n_written == 3
    manifest = json.loads((tmp_path / "emitted_files.json").read_text())
    contract = manifest["contract"]
    assert min(grid.nx, grid.ny, grid.nz) > 1
    assert contract["pml_faces"] == [10] * 6
    assert grid.dt_s == native_dt(grid.dx_m, "3D")
    assert (grid.iterations - 1) * grid.dt_s == grid.time_window_s
    assert grid.soil_depth_m == grid.depth_z_m
    for item in result.files:
        text = Path(item["path"]).read_text()
        assert f"#hertzian_dipole: {axis} " in text
        assert f"#time_window: {grid.iterations}\n" in text
        scene = item["resolved_scene"]
        assert scene["source"]["position_m"][2] > 0
        assert item["contract_digest"] == contract["digest"]
        assert scene["targets"][0]["cylinder_axis"] == "x"
        assert scene["targets"][0]["end_m"][0] > scene["targets"][0]["start_m"][0]
        assert all(l["end_m"][2] == grid.domain_z_m for l in scene["layers"])
        assert scene["layers"][-1]["terminal_halfspace"]
        assert scene["layers"][-1]["y_bottom_m"] == 0
        assert scene["targets"][0]["requested"]["depth_m"] == .25
    assert grid.model_dump() == frozen
    assert not contract["qualification"]["reuse_eligible"]


def test_independent_target_fields_do_not_change_soil_draws(tmp_path):
    cfg, layers, wf, ant = inputs(tmp_path)
    a, _ = sample_layers(layers, 3, dataset_config=cfg, target_ranges=targets())
    b, _ = sample_layers(layers, 3, dataset_config=cfg)
    assert [[l.model_dump() for l in s.layers] for s in a] == [[l.model_dump() for l in s.layers] for s in b]
    assert [s.model_dump() for s in a] == [s.model_dump() for s in sample_layers(layers, 3, dataset_config=cfg, target_ranges=targets())[0]]


def test_mode_missing_geometry_roughness_and_arrays_rejected(tmp_path):
    cfg, layers, wf, ant = inputs(tmp_path)
    incomplete = ExtractedTargetRanges(cylinders=[dict(x_offset_min_m=0, x_offset_max_m=0,
        depth_min_m=.1, depth_max_m=.1, radius_min_m=.02, radius_max_m=.02)])
    with pytest.raises(ValueError, match="finite length"):
        validate_capabilities(cfg, target_ranges=incomplete)
    with pytest.raises(ValueError, match="roughness"):
        validate_capabilities(cfg, advanced=ExtractedAdvancedParams(surface_roughness={}))
    with pytest.raises(ValueError, match="receiver_height"):
        validate_capabilities(cfg, antenna=ant.model_copy(update={"rx_same_height": False}))
    with pytest.raises(ValueError, match="z polarization"):
        validate_capabilities(DatasetConfig(num_samples=1, contract_version=2), antenna=ant)


@pytest.mark.parametrize("nbins,lo,hi", [(1, .05, .15), (50, .29, .30), (2, .1, .29)])
def test_native_shifted_moisture_rejected(nbins, lo, hi):
    with pytest.raises(ValueError):
        native_material_table("soil", 35, 10, 1.5, 2.66, lo, hi, nbins)


def test_native_half_cell_and_full_3d_bounds(tmp_path):
    assert cell_index(2.5, 1) == 2
    assert cell_index(3.5, 1) == 3
    cfg, _, _, _, grid, _ = plan(tmp_path)
    for axis in "xyz":
        t = SampledTarget(kind="cylinder", x_offset_m=0, z_offset_m=0,
                          depth_m=.2, radius_m=.03, length_m=.12, cylinder_axis=axis)
        scene = resolve_target(t, grid, cfg)
        assert not validate_resolved_target(scene, grid, cfg)
        assert scene["end_cells"]["xyz".index(axis)] > scene["start_cells"]["xyz".index(axis)]
    target = t.model_copy(update={"z_offset_m": grid.domain_z_m / 2})
    assert any("on z" in e for e in validate_resolved_target(resolve_target(target, grid, cfg), grid, cfg))


def test_source_delay_expands_shared_timing(tmp_path):
    cfg, layers, wf, ant, grid, samples = plan(tmp_path)
    delayed = wf.model_copy(update={"source_start_time": 10e-9, "source_end_time": 20e-9})
    shifted = derive_global(cfg, delayed, ant, layers, grid.eps_r_max_global, grid.eps_r_min_global)
    assert shifted.time_window_s > grid.time_window_s
    with pytest.raises(ValueError, match="truncates"):
        derive_global(cfg, wf.model_copy(update={"source_end_time": 1e-9}), ant, layers, 12, 3)


def test_invalid_snapshot_and_thin_nominal_3d_fail(tmp_path):
    cfg, layers, wf, ant, grid, samples = plan(tmp_path)
    invalid = grid.model_copy(update={"nz": 1, "domain_z_m": grid.dx_m})
    assert not validate_global(invalid, cfg, wf, ant, layers).ok
    with pytest.raises(ValueError, match="Snapshot time"):
        resolve_scene(samples[0], grid, cfg, wf, ant, ExtractedAdvancedParams(snapshots=[dict(time_s=1, filename="late")]))


def test_new_2d_retains_tmz_with_native_timing(tmp_path):
    cfg, layers, wf, ant, grid, samples = plan(tmp_path, "2D", "z")
    result = emit_dataset(str(tmp_path), cfg, wf, ant, layers=layers)
    assert not result.errors
    assert grid.nz == 1 and grid.domain_z_m == grid.dx_m
    assert grid.dt_s == native_dt(grid.dx_m, "2D")
    assert [l["resolved_scene"]["source"]["axis"] for l in result.files] == ["z"] * 3


def test_partial_fixed_fields_survive_shrink_and_range_preserving_retry(tmp_path):
    cfg, layers, wf, ant, grid, samples = plan(tmp_path)
    tr = ExtractedTargetRanges(cylinders=[dict(name="partially-fixed", x_offset_min_m=.02, x_offset_max_m=.02,
        z_offset_min_m=-.02, z_offset_max_m=-.02, depth_min_m=.03, depth_max_m=.03,
        radius_min_m=.02, radius_max_m=.05, length_min_m=.08, length_max_m=.08, cylinder_axis="x")])
    original = SampledTarget(kind="cylinder", name="partially-fixed", x_offset_m=.02, z_offset_m=-.02,
                             depth_m=.03, radius_m=.05, length_m=.08, cylinder_axis="x")
    sample = samples[0].model_copy(update={"targets": [original], "provenance": {}})
    result = validate_and_place([sample], grid, cfg, tr)
    assert not result.dropped
    target = result.surviving[0].targets[0]
    for field in ("x_offset_m", "z_offset_m", "depth_m", "length_m", "cylinder_axis"):
        assert getattr(target, field) == getattr(original, field)
    assert .02 <= target.radius_m < original.radius_m
    assert 1 <= result.surviving[0].provenance["placement_attempts"][0] <= 20
    assert result.surviving[0].provenance["original_targets"] == [original.model_dump()]


def test_infeasible_partial_fixed_target_drops_whole_sample_without_backfill(tmp_path):
    cfg, layers, wf, ant, grid, samples = plan(tmp_path)
    tr = ExtractedTargetRanges(cylinders=[dict(x_offset_min_m=0, x_offset_max_m=0,
        z_offset_min_m=0, z_offset_max_m=0, depth_min_m=.001, depth_max_m=.001,
        radius_min_m=.02, radius_max_m=.05, length_min_m=.08, length_max_m=.08, cylinder_axis="z")])
    target = SampledTarget(kind="cylinder", x_offset_m=0, z_offset_m=0, depth_m=.001,
                            radius_m=.05, length_m=.08, cylinder_axis="z")
    result = validate_and_place([samples[0].model_copy(update={"targets": [target]})], grid, cfg, tr)
    assert result.surviving == []
    assert result.dropped[0]["sample_id"] == samples[0].sample_id
    assert result.dropped[0]["placement_attempts"][0] <= 20


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("face", ["min", "max"])
def test_all_six_resolved_faces_require_clearance(tmp_path, axis, face):
    cfg, _, _, _, grid, _ = plan(tmp_path)
    target = resolve_target(SampledTarget(kind="box", x_offset_m=0, z_offset_m=0, depth_m=.2,
        width_m=.08, height_m=.08, crossline_size_m=.08), grid, cfg)
    if face == "min":
        target["bbox_min_m"][axis] = 0
    else:
        target["bbox_max_m"][axis] = [grid.domain_x_m, grid.domain_y_m, grid.domain_z_m][axis]
    assert any("on " + "xyz"[axis] in message for message in validate_resolved_target(target, grid, cfg))


def test_static_overlap_is_a_gate_error_not_a_redraw(tmp_path):
    cfg, layers, wf, ant, grid, samples = plan(tmp_path)
    spec = dict(x_offset_min_m=0, x_offset_max_m=0, z_offset_min_m=0, z_offset_max_m=0,
        depth_min_m=.2, depth_max_m=.2, width_min_m=.08, width_max_m=.08,
        height_min_m=.08, height_max_m=.08, crossline_size_min_m=.08, crossline_size_max_m=.08)
    tr = ExtractedTargetRanges(boxes=[spec, spec])
    report = validate_global(grid, cfg, wf, ant, layers, target_ranges=tr)
    assert any("disjoint_bounds" in error for error in report.errors)


def test_quantization_exact_and_snapshot_iteration_policy(tmp_path):
    from backend.dataset_sampling.scene import resolve_outputs
    cfg, _, _, _, grid, _ = plan(tmp_path)
    target = SampledTarget(kind="box", x_offset_m=.00012345, z_offset_m=0, depth_m=.2,
                          width_m=.06, height_m=.06, crossline_size_m=.06)
    with pytest.raises(ValueError, match="exact policy"):
        resolve_target(target, grid, cfg.model_copy(update={"quantization_policy": "exact"}))
    adv = ExtractedAdvancedParams(snapshots=[dict(time_s=3.5 * grid.dt_s, filename="early")])
    resolved = resolve_outputs(adv, grid, "sample")[0]
    assert resolved["iteration"] == cell_index(3.5 * grid.dt_s, grid.dt_s) + 1
    assert resolved["effective_time_s"] == (resolved["iteration"] - 1) * grid.dt_s


def test_unsupported_nested_output_fields_and_nonfinite_inputs_rejected():
    with pytest.raises(ValidationError):
        ExtractedAdvancedParams(snapshots=[dict(time_s=1e-9, filename="field", fields=["Ez"])])
    with pytest.raises(ValidationError):
        DatasetConfig(num_samples=1, high_freq_factor=float("nan"))


def test_unsafe_basename_resolves_to_one_file_identity(tmp_path):
    cfg, layers, wf, ant, grid, samples = plan(tmp_path)
    cfg = cfg.model_copy(update={"model_basename": "../../survey name"})
    result = emit_dataset(str(tmp_path), cfg, wf, ant, layers=layers)
    for entry in result.files:
        assert Path(entry["path"]).parent == tmp_path / "in_files"
        assert Path(entry["filename"]).stem == entry["resolved_scene"]["title"]


def test_receiver_height_is_resolved_and_exact_policy_cannot_discard_rounding(tmp_path):
    cfg, layers, wf, ant, grid, samples = plan(tmp_path)
    ant = ant.model_copy(update={"rx_same_height": False, "receiver_height_m": .123456,
                                 "tx_rx_offset_m": 0, "tx_rx_crossline_offset_m": 0})
    changed = derive_global(cfg, wf, ant, layers, grid.eps_r_max_global, grid.eps_r_min_global)
    assert changed.rx_y_m != changed.tx_y_m
    assert changed.derivation["requested_rx_m"][1] - changed.ground_y_m == pytest.approx(.123456)
    with pytest.raises(ValueError, match="Acquisition coordinates"):
        derive_global(cfg.model_copy(update={"quantization_policy": "exact"}), wf, ant, layers,
                      grid.eps_r_max_global, grid.eps_r_min_global)
    # "Same height" inherits the requested source height, before either snaps.
    ant = ant.model_copy(update={"rx_same_height": True, "receiver_height_m": None,
                                 "source_height_m": grid.lambda_max_m + .123456})
    changed = derive_global(cfg, wf, ant, layers, grid.eps_r_max_global, grid.eps_r_min_global)
    assert changed.rx_y_m == changed.tx_y_m
    assert changed.derivation["requested_rx_m"][1] == changed.derivation["requested_tx_m"][1]
    assert changed.derivation["requested_rx_m"][1] - changed.ground_y_m == pytest.approx(ant.source_height_m)
