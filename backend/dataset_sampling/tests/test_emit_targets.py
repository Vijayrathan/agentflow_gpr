"""
Tests for buried-object emission (STAGE 8): the .in text must transcribe the
already-placed targets with the offset/depth conventions resolved against the
grid, and never contain legacy advanced-geometry objects.

Run: pytest backend/dataset_sampling/tests/test_emit_targets.py
"""
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BACKEND = _REPO_ROOT / "backend"
for _p in (str(_REPO_ROOT), str(_BACKEND), str(_REPO_ROOT / "gprMax")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from backend.schema import (  # noqa: E402
    DatasetConfig,
    ExtractedAntenna,
    ExtractedLayers,
    ExtractedLayerParams,
    ExtractedTargetRanges,
    ExtractedWaveform,
    SampledSample,
)
from dataset_sampling.layer_sampler import sample_layers  # noqa: E402
from dataset_sampling.peplinski_derive import derive_samples  # noqa: E402
from dataset_sampling.global_derive import derive_global  # noqa: E402
from dataset_sampling.emit import build_in_text  # noqa: E402


def _layers():
    L = ExtractedLayerParams(
        name="topsoil", thickness_m_min=0.2, thickness_m_max=0.5,
        sand_pct_min=30, sand_pct_max=45, clay_pct_min=6, clay_pct_max=18,
        theta_v_min=0.05, theta_v_max=0.20,
        bulk_density_gcm3_min=1.4, bulk_density_gcm3_max=1.6,
        particle_density_gcm3_min=2.6, particle_density_gcm3_max=2.7,
    )
    return ExtractedLayers(num_layers=1, layers=[L])


def _fixture():
    ext = _layers()
    cfg = DatasetConfig(num_samples=2)
    wf = ExtractedWaveform(waveform_center_freq_hz=0.7e9, waveform_name="w")
    ant = ExtractedAntenna(tx_rx_offset_m=0.1)
    samples, _w = sample_layers(ext, 2, seed=7)
    _d, agg = derive_samples(samples, cfg, wf)
    grid = derive_global(
        cfg, wf, ant, ext, agg.eps_r_max, agg.eps_r_min,
        smallest_feature_global_m=0.04,       # keep dx target-tightened like a real run
        largest_extent_global_m=0.3,
        deepest_target_bottom_global_m=0.4,
    )
    return samples[0], grid, cfg, wf, ant


def _with_targets(sample: SampledSample, targets: list) -> SampledSample:
    return SampledSample.model_validate({
        "sample_id": sample.sample_id,
        "layers": [l.model_dump() for l in sample.layers],
        "targets": targets,
    })


def test_cylinder_and_box_lines_resolved_against_grid():
    sample, grid, cfg, wf, ant = _fixture()
    s = _with_targets(sample, [
        {"kind": "cylinder", "name": "pipe", "material": "pec",
         "x_offset_m": -0.2, "depth_m": 0.25, "radius_m": 0.03},
        {"kind": "box", "name": "slab", "material": "pec",
         "x_offset_m": 0.1, "depth_m": 0.3, "width_m": 0.2, "height_m": 0.06},
    ])
    text, _labels = build_in_text(s, grid, cfg, wf, ant, adv=None)

    cyl_lines = [l for l in text.splitlines() if l.startswith("#cylinder:")]
    box_lines = [l for l in text.splitlines() if l.startswith("#box:")]
    assert len(cyl_lines) == 1 and len(box_lines) == 1
    assert "#sphere" not in text

    # cylinder: x resolved as domain_x/2 + offset; y as ground_y - depth;
    # thin z 0 -> dx; smoothing OFF so the PEC replaces the fractal soil.
    cx = grid.domain_x_m / 2.0 - 0.2
    cy = grid.ground_y_m - 0.25
    parts = cyl_lines[0].split()
    assert float(parts[1]) == pytest.approx(cx, abs=1e-9)
    assert float(parts[2]) == pytest.approx(cy, abs=1e-9)
    assert float(parts[3]) == 0.0
    assert float(parts[6]) == pytest.approx(grid.dx_m, abs=1e-12)
    assert float(parts[7]) == pytest.approx(0.03)
    assert parts[8] == "pec" and parts[9] == "n"

    # box: corners at center +- half extents, thin z, smoothing off
    bx = grid.domain_x_m / 2.0 + 0.1
    by = grid.ground_y_m - 0.3
    parts = box_lines[0].split()
    assert float(parts[1]) == pytest.approx(bx - 0.1, abs=1e-9)
    assert float(parts[2]) == pytest.approx(by - 0.03, abs=1e-9)
    assert float(parts[3]) == 0.0
    assert float(parts[4]) == pytest.approx(bx + 0.1, abs=1e-9)
    assert float(parts[5]) == pytest.approx(by + 0.03, abs=1e-9)
    assert float(parts[6]) == pytest.approx(grid.dx_m, abs=1e-12)
    assert parts[7] == "pec" and parts[8] == "n"


def test_targets_emitted_after_fractal_soil():
    sample, grid, cfg, wf, ant = _fixture()
    s = _with_targets(sample, [
        {"kind": "cylinder", "name": "pipe", "material": "pec",
         "x_offset_m": 0.0, "depth_m": 0.25, "radius_m": 0.03},
    ])
    text, _labels = build_in_text(s, grid, cfg, wf, ant, adv=None)
    # the PEC object must come AFTER the fractal soil boxes so it overrides them
    assert text.index("#fractal_box") < text.index("#cylinder:")


def test_no_targets_emits_no_object_lines():
    sample, grid, cfg, wf, ant = _fixture()
    text, _labels = build_in_text(sample, grid, cfg, wf, ant, adv=None)
    assert "#cylinder" not in text
    assert "#sphere" not in text
    # (#box appears only via fractal_box for soil; no plain PEC box line)
    assert not [l for l in text.splitlines() if l.startswith("#box:")]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
