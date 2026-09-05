"""Source selection must survive emission and gprMax's actual input parser."""
import math
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "backend"), str(_REPO_ROOT / "gprMax")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from backend.schema import (  # noqa: E402
    DatasetConfig, ExtractedAntenna, ExtractedLayers, ExtractedWaveform,
    SampledLayer, SampledSample,
)
from dataset_sampling.emit import build_in_text  # noqa: E402
from dataset_sampling.global_derive import derive_global  # noqa: E402
from dataset_sampling.sample_validation import validate_waveform_antenna  # noqa: E402
from gprMax.grid import FDTDGrid  # noqa: E402
from gprMax.input_cmds_file import check_cmd_names  # noqa: E402
from gprMax.input_cmds_multiuse import process_multicmds  # noqa: E402


@pytest.fixture
def scenario():
    cfg = DatasetConfig(num_samples=1)
    wf = ExtractedWaveform(waveform_center_freq_hz=0.7e9, waveform_name="test pulse")
    ant = ExtractedAntenna(tx_rx_offset_m=0.1)
    layers = ExtractedLayers(num_layers=1, layers=[{
        "name": "soil", "thickness_m_min": 0.3, "thickness_m_max": 0.3,
        "sand_pct_min": 35, "sand_pct_max": 35,
        "clay_pct_min": 10, "clay_pct_max": 10,
        "theta_v_min": 0.05, "theta_v_max": 0.2,
        "bulk_density_gcm3_min": 1.5, "bulk_density_gcm3_max": 1.5,
        "particle_density_gcm3_min": 2.66, "particle_density_gcm3_max": 2.66,
    }])
    grid = derive_global(cfg, wf, ant, layers, 16.0, 4.0)
    sample = SampledSample(sample_id=1, layers=[SampledLayer(
        name="soil", thickness_m=0.3, sand_pct=35, clay_pct=10, silt_pct=55,
        theta_v_min=0.05, theta_v_max=0.2,
        bulk_density_gcm3=1.5, particle_density_gcm3=2.66,
    )])
    return sample, grid, cfg, wf, ant


@pytest.mark.parametrize("kind,source_list,resistance", [
    ("hertzian_dipole", "hertziandipoles", None),
    ("voltage_source", "voltagesources", 50.0),
    ("transmission_line", "transmissionlines", 75.0),
])
@pytest.mark.parametrize("timed", [False, True])
def test_emitted_deck_builds_selected_gprmax_source(scenario, kind, source_list, resistance, timed):
    sample, grid, cfg, wf, _ = scenario
    ant = ExtractedAntenna(antenna_kind=kind, tx_rx_offset_m=0.1, resistance=resistance)
    if timed:
        wf = wf.model_copy(update={"source_start_time": 1e-9, "source_end_time": 3e-9})
    deck, _ = build_in_text(sample, grid, cfg, wf, ant, adv=None)
    commands = [line for line in deck.splitlines(keepends=True) if line.startswith("#")]
    _, multicmds, _ = check_cmd_names(commands)

    # Use gprMax's source construction to verify parameter order and semantics,
    # without a field solve or stubbing the parser.
    solver_grid = FDTDGrid()
    solver_grid.messages = False
    solver_grid.mode = "2D TMz"
    solver_grid.dx = solver_grid.dy = solver_grid.dz = grid.dx_m
    solver_grid.nx = round(grid.domain_x_m / grid.dx_m)
    solver_grid.ny = round(grid.domain_y_m / grid.dx_m)
    solver_grid.nz = 1
    solver_grid.dt = grid.dt_s
    solver_grid.timewindow = grid.time_window_s
    solver_grid.iterations = math.ceil(grid.time_window_s / grid.dt_s) + 1
    solver_grid.pmlthickness["z0"] = solver_grid.pmlthickness["zmax"] = 0
    process_multicmds(multicmds, solver_grid)

    sources = getattr(solver_grid, source_list)
    assert len(sources) == 1
    assert sum(len(getattr(solver_grid, name)) for name in
               ("hertziandipoles", "voltagesources", "transmissionlines", "magneticdipoles")) == 1
    source = sources[0]
    assert source.polarisation == "z"
    assert source.waveformID == "test_pulse"
    assert source.xcoord == round(grid.tx_x_m / grid.dx_m)
    assert source.ycoord == round(grid.tx_y_m / grid.dx_m)
    assert source.zcoord == 0
    assert source.start == pytest.approx(1e-9 if timed else 0, abs=1e-15)
    assert source.stop == pytest.approx(3e-9 if timed else grid.time_window_s, abs=1e-15)
    if resistance is not None:
        assert source.resistance == resistance


@pytest.mark.parametrize("kind", ["unknown_source", "", None])
def test_unvalidated_source_cannot_fall_back_at_gate_or_emission(scenario, kind):
    sample, grid, cfg, wf, ant = scenario
    # model_copy bypasses schema validation, exercising the defensive checks
    # for programmatic callers and stale in-memory objects.
    invalid = ant.model_copy(update={"antenna_kind": kind})
    report = validate_waveform_antenna(cfg, wf, invalid)
    assert not report.ok
    assert any("unsupported antenna kind" in error for error in report.errors)
    with pytest.raises(ValueError, match="Unsupported antenna_kind"):
        build_in_text(sample, grid, cfg, wf, invalid, adv=None)


@pytest.mark.parametrize("kind", ["voltage_source", "transmission_line"])
def test_unvalidated_resistive_source_without_resistance_fails_emission(scenario, kind):
    sample, grid, cfg, wf, ant = scenario
    invalid = ant.model_copy(update={"antenna_kind": kind, "resistance": None})
    with pytest.raises(ValueError, match=f"{kind} requires a resistance"):
        build_in_text(sample, grid, cfg, wf, invalid, adv=None)


@pytest.mark.parametrize("kind", ["hertzian_dipole", "voltage_source", "transmission_line"])
@pytest.mark.parametrize("resistance", [0, -1, 376.73, float("inf"), float("nan")])
def test_unvalidated_resistance_rejected_at_gate(scenario, kind, resistance):
    _, _, cfg, wf, ant = scenario
    invalid = ant.model_copy(update={"antenna_kind": kind, "resistance": resistance})
    report = validate_waveform_antenna(cfg, wf, invalid)
    assert not report.ok
    assert any("resistance must satisfy 0 < R <" in error for error in report.errors)
