"""Equal moisture bounds must survive collection, sampling and native materials."""
import sys
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
for _path in (_ROOT, _ROOT / "backend", _ROOT / "gprMax"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from backend.schema import (  # noqa: E402
    DatasetConfig, ExtractedAntenna, ExtractedLayerParams, ExtractedLayers,
    ExtractedWaveform, SampledLayer,
)
from backend.validation_tools_new import validate_sampled_layer  # noqa: E402
from dataset_sampling.emit import build_in_text  # noqa: E402
from dataset_sampling.global_derive import derive_global  # noqa: E402
from dataset_sampling.layer_sampler import (  # noqa: E402
    read_samples, sample_layers, write_samples,
)
from dataset_sampling.peplinski_derive import derive_samples  # noqa: E402
from gprMax.grid import FDTDGrid  # noqa: E402
from gprMax.input_cmds_file import check_cmd_names  # noqa: E402
from gprMax.input_cmds_multiuse import process_multicmds  # noqa: E402


def layer_spec(lo, hi):
    return ExtractedLayerParams(
        name="soil", sand_pct_min=40, sand_pct_max=40,
        clay_pct_min=15, clay_pct_max=15, theta_v_min=lo, theta_v_max=hi,
        bulk_density_gcm3_min=1.5, bulk_density_gcm3_max=1.5,
        particle_density_gcm3_min=2.65, particle_density_gcm3_max=2.65,
    )


@pytest.mark.parametrize("lo,hi", [(0.10, 0.10), (0.20, 0.20), (0.30, 0.30), (0.10, 0.20)])
@pytest.mark.parametrize("nbins", [2, 50])
def test_bounds_survive_pipeline_and_native_material_build(tmp_path, lo, hi, nbins):
    layers = ExtractedLayers(num_layers=1, soil_depth_m=0.5, layers=[layer_spec(lo, hi)])
    samples, warnings = sample_layers(layers, num_samples=3, seed=42)
    assert not warnings
    write_samples(samples, str(tmp_path))
    samples = read_samples(str(tmp_path))
    # This policy preserves both bounds across samples, including nonzero bands.
    assert [(s.layers[0].theta_v_min, s.layers[0].theta_v_max) for s in samples] == [(lo, hi)] * 3

    cfg = DatasetConfig(num_samples=3, fractal_nbins=nbins)
    wf = ExtractedWaveform(waveform_center_freq_hz=750e6, waveform_name="pulse")
    ant = ExtractedAntenna(tx_rx_offset_m=0.08)
    derived, aggregate = derive_samples(samples, cfg, wf)
    grid = derive_global(cfg, wf, ant, layers, aggregate.eps_r_max, aggregate.eps_r_min)
    deck, _ = build_in_text(samples[0], grid, cfg, wf, ant, adv=None)
    soil_line = next(line for line in deck.splitlines() if line.startswith("#soil_peplinski:"))
    assert tuple(map(float, soil_line.split()[5:7])) == (lo, hi)
    fractal_line = next(line for line in deck.splitlines() if line.startswith("#fractal_box:"))
    assert int(fractal_line.split()[11]) == nbins

    # Parse the emitted native material declaration and build every Debye bin.
    # No field solve is needed to verify this input-policy change.
    commands = [line for line in deck.splitlines(keepends=True) if line.startswith("#")]
    _, multicmds, _ = check_cmd_names(commands)
    solver_grid = FDTDGrid()
    solver_grid.messages = False
    material_commands = {key: None for key in multicmds}
    material_commands["#soil_peplinski"] = multicmds["#soil_peplinski"]
    process_multicmds(material_commands, solver_grid)
    soil = solver_grid.mixingmodels[0]
    assert tuple(soil.mu) == (lo, hi)
    start = len(solver_grid.materials)
    soil.calculate_debye_properties(nbins, solver_grid, "soil_box")
    materials = solver_grid.materials[start:]
    assert len(materials) == nbins
    response = [m.calculate_er(750e6) for m in materials]
    assert all(r.real > 0 for r in response)
    labels = derived[0].layers[0]
    assert labels.eps_r_dry == response[0].real
    assert labels.eps_r_wet == response[-1].real
    if lo == hi:
        assert all(r == response[0] for r in response)
        assert all((m.er, m.se, m.deltaer, m.tau) ==
                   (materials[0].er, materials[0].se, materials[0].deltaer, materials[0].tau)
                   for m in materials)
        assert labels.sigma_dry == labels.sigma_wet
    else:
        assert response[0] != response[-1]


def test_reversed_bounds_still_rejected_at_each_gate():
    with pytest.raises(ValueError, match="theta_v"):
        layer_spec(0.20, 0.10)
    valid = layer_spec(0.10, 0.10)
    # Exercise the sampler's defensive check even when a caller bypasses schema validation.
    invalid = valid.model_copy(update={"theta_v_min": 0.20})
    layers = ExtractedLayers.model_construct(num_layers=1, soil_depth_m=0.5, layers=[invalid])
    with pytest.raises(ValueError, match="theta_v_min"):
        sample_layers(layers, num_samples=1)
    with pytest.raises(ValueError, match="theta_v_min"):
        SampledLayer(name="soil", sand_pct=40, silt_pct=45, clay_pct=15,
                     theta_v_min=0.20, theta_v_max=0.10,
                     bulk_density_gcm3=1.5, particle_density_gcm3=2.65)
    errors, _ = validate_sampled_layer(40, 45, 15, 0.20, 0.10, 1.5, 2.65)
    assert any("theta_v_min" in error for error in errors)


@pytest.mark.parametrize("theta,bulk,reason", [(0.31, 1.5, "Peplinski max"), (0.25, 2.2, "porosity")])
def test_uniform_moisture_still_obeys_physical_limits(theta, bulk, reason):
    errors, _ = validate_sampled_layer(40, 45, 15, theta, theta, bulk, 2.65)
    assert any(reason in error for error in errors)


def test_uniform_moisture_does_not_allow_one_material_bin():
    with pytest.raises(ValueError):
        DatasetConfig(num_samples=1, fractal_nbins=1)


def paired_layers(varied):
    top = layer_spec(0.10, 0.30).model_copy(update={
        "name": "top", "thickness_m_min": 0.12 if varied else 0.18,
        "thickness_m_max": 0.24 if varied else 0.18,
    })
    bottom = layer_spec(0.05, 0.05).model_copy(update={"name": "bottom"})
    return ExtractedLayers(num_layers=2, soil_depth_m=0.5, layers=[top, bottom])


def moisture_labels(samples):
    return [(s.sample_id, s.layers[0].theta_v_min) for s in samples]


def test_moisture_pairing_independent_of_geometry_draws_and_retries(monkeypatch):
    from dataset_sampling import layer_sampler as sampler

    a, _ = sample_layers(paired_layers(True), 100, seed=42,
                        moisture_sampling="uniform_per_sample", moisture_seed=71)
    original = sampler._sample_one_layer

    def consume_geometry_randomness(layer, rng, *args, **kwargs):
        # Reproduce extra RNG consumption caused by retries or changed geometry.
        for _ in range(17):
            rng.random()
        return original(layer, rng, *args, **kwargs)

    monkeypatch.setattr(sampler, "_sample_one_layer", consume_geometry_randomness)
    b, _ = sample_layers(paired_layers(False), 100, seed=999,
                        moisture_sampling="uniform_per_sample", moisture_seed=71)
    assert moisture_labels(a) == moisture_labels(b)
    assert len({s.layers[0].thickness_m for s in a}) > 1
    assert {s.layers[0].thickness_m for s in b} == {0.18}
    assert len(set(moisture_labels(a))) == 100
    assert len({s.layers[0].theta_v_min for s in a}) > 90
    for s in a + b:
        assert 0.10 <= s.layers[0].theta_v_min == s.layers[0].theta_v_max <= 0.30
        assert s.layers[1].theta_v_min == s.layers[1].theta_v_max == 0.05


def test_moisture_seed_repeats_changes_and_preserves_prefix():
    def draw(n, seed):
        return sample_layers(paired_layers(True), n, seed=42,
                             moisture_sampling="uniform_per_sample", moisture_seed=seed)[0]
    a = draw(10, 71)
    assert moisture_labels(a) == moisture_labels(draw(10, 71))
    assert moisture_labels(a) == moisture_labels(draw(20, 71)[:10])
    assert moisture_labels(a) != moisture_labels(draw(10, 72))


def test_sampling_node_persists_policy_and_emits_exact_labels(tmp_path):
    from backend.agentflow_single_agent import layer_sampling_node

    layers = paired_layers(True)
    cfg = DatasetConfig(num_samples=5, output_dir=str(tmp_path),
                        moisture_sampling="uniform_per_sample", moisture_seed=71)
    state = {"dataset_config": cfg.model_dump(), "layers": layers.model_dump(), "target_ranges": {}}
    result = layer_sampling_node(state)
    assert result["sampling_snapshot"]["dataset_config"]["moisture_seed"] == 71
    saved = json.loads((tmp_path / "sampled_layers.json").read_text())
    assert saved["sampling"]["moisture_seed"] == 71
    assert saved["sampling"]["moisture_sampling"] == "uniform_per_sample"
    assert saved["sampling"]["requested_layers"]["layers"][0]["theta_v_max"] == 0.30
    samples = read_samples(str(tmp_path))
    wf = ExtractedWaveform(waveform_center_freq_hz=750e6, waveform_name="pulse")
    ant = ExtractedAntenna(tx_rx_offset_m=0.08)
    _, aggregate = derive_samples(samples, cfg, wf)
    grid = derive_global(cfg, wf, ant, layers, aggregate.eps_r_max, aggregate.eps_r_min)
    for s in samples:
        deck, _ = build_in_text(s, grid, cfg, wf, ant, adv=None)
        lines = [line for line in deck.splitlines() if line.startswith("#soil_peplinski:")]
        for line, layer in zip(lines, s.layers, strict=True):
            lo, hi = map(float, line.split()[5:7])
            assert lo == hi == layer.theta_v_min == layer.theta_v_max


@pytest.mark.parametrize("lo,hi", [(0, 0.20), (-0.01, 0.20), (0.10, 0.31)])
def test_invalid_population_range_fails_without_resampling_moisture(lo, hi):
    layers = ExtractedLayers(num_layers=1, soil_depth_m=0.5, layers=[layer_spec(lo, hi)])
    with pytest.raises(ValueError):
        sample_layers(layers, 1, moisture_sampling="uniform_per_sample", moisture_seed=71)
