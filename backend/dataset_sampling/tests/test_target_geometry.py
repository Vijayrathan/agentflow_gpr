"""
Tests for variable buried-target geometry on a fixed global grid.

Covers the invariants that make the feature safe:
  - a target landing in the PML is re-drawn into the valid region,
  - a target that can NEVER fit (domain too small for r_floor) is dropped
    immediately (no wasted redraws),
  - a target that fails only on size shrinks to recover,
  - target variation does NOT change the global grid (one grid for all samples),
  - box/sphere targets are stubbed cleanly.

Run: pytest backend/dataset_sampling/tests/test_target_geometry.py
"""
import sys
from pathlib import Path

import pytest

# Mirror the runtime path setup (agentflow): repo root for `backend.*`, the
# `backend/` dir so the dataset_sampling package imports as bare `dataset_sampling.*`
# (the convention its modules use internally), and the inner gprMax package.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_BACKEND = _REPO_ROOT / "backend"
for _p in (str(_REPO_ROOT), str(_BACKEND), str(_REPO_ROOT / "gprMax")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from backend.schema import (  # noqa: E402
    ExtractedLayers,
    ExtractedLayerParams,
    DatasetConfig,
    ExtractedWaveform,
    ExtractedAntenna,
    CylinderTargetRange,
    SampledSample,
)
from dataset_sampling.layer_sampler import (  # noqa: E402
    sample_layers,
    MAX_TARGET_ATTEMPTS,
    _draw_target,
)
from dataset_sampling.peplinski_derive import derive_samples  # noqa: E402
from dataset_sampling.global_derive import derive_global  # noqa: E402
from dataset_sampling.target_placement import (  # noqa: E402
    validate_and_place,
    _placement_failures,
    _clearance,
    _r_floor,
)


# ── fixtures / helpers ──────────────────────────────────────────────────────

def _layers():
    L = ExtractedLayerParams(
        name="topsoil", thickness_m_min=0.2, thickness_m_max=0.5,
        sand_pct_min=30, sand_pct_max=45, clay_pct_min=6, clay_pct_max=18,
        theta_v_min=0.05, theta_v_max=0.20,
        bulk_density_gcm3_min=1.4, bulk_density_gcm3_max=1.6,
        particle_density_gcm3_min=2.6, particle_density_gcm3_max=2.7,
    )
    return ExtractedLayers(num_layers=1, layers=[L])


def _cfg():
    return DatasetConfig(num_samples=6)


def _wf():
    return ExtractedWaveform(waveform_center_freq_hz=0.7e9, waveform_name="w")


def _ant():
    return ExtractedAntenna(tx_rx_offset_m=0.1)


def _target_range():
    return CylinderTargetRange(
        x_center_min_m=0.3, x_center_max_m=0.9,
        depth_min_m=0.1, depth_max_m=0.35,
        radius_min_m=0.02, radius_max_m=0.06,
    )


def _build_grid(target_range, n=6, seed=11):
    """Run draw -> aggregate -> derive and return (grid, samples, cfg, layers)."""
    ext, cfg = _layers(), _cfg()
    samples, _w = sample_layers(ext, n, seed=seed, target_range=target_range)
    _derived, agg = derive_samples(samples, cfg, _wf())
    grid = derive_global(
        cfg, _wf(), _ant(), ext, None, agg.eps_r_max, agg.eps_r_min,
        smallest_feature_global_m=agg.smallest_feature_global_m,
        largest_extent_global_m=agg.largest_extent_global_m,
        deepest_target_bottom_global_m=agg.deepest_target_bottom_global_m,
    )
    return grid, samples, cfg, ext


def _sample_with_target(template_sample, sample_id, x, depth, r):
    """Clone a sample's layers (as dicts, avoiding cross-module instance issues)
    and attach a specific target."""
    return SampledSample.model_validate({
        "sample_id": sample_id,
        "layers": [l.model_dump() for l in template_sample.layers],
        "target": {"kind": "cylinder", "name": "target", "material": "pec",
                   "x_center_m": x, "depth_m": depth, "radius_m": r},
    })


# ── tests ───────────────────────────────────────────────────────────────────

def test_redraw_on_pml_hit():
    tr = _target_range()
    grid, samples, cfg, _ = _build_grid(tr)
    clr = _clearance(grid, cfg)
    # x sits inside the (pml+15) clearance -> invalid placement
    s = _sample_with_target(samples[0], 1, x=clr * 0.5, depth=0.2, r=0.04)
    assert _placement_failures(s.target, grid, cfg)  # precondition: it IS invalid

    res = validate_and_place([s], grid, cfg, tr, seed=1)
    assert len(res.surviving) == 1 and not res.dropped
    assert res.n_redrawn == 1
    # the re-drawn target is now valid (clears the PML+gap, fully buried)
    assert not _placement_failures(res.surviving[0].target, grid, cfg)


def test_immediate_drop_when_impossible():
    tr = _target_range()
    grid, samples, cfg, _ = _build_grid(tr)
    # Shrink the domain so even r_floor cannot fit, and force a large lower bound.
    tiny = grid.model_copy(update={
        "domain_x_m": _clearance(grid, cfg) * 2 + grid.dx_m,  # no room for any radius
    })
    impossible_range = tr.model_copy(update={"radius_min_m": 0.5, "radius_max_m": 0.6})
    s = _sample_with_target(samples[0], 1, x=tiny.domain_x_m / 2, depth=0.2, r=0.5)

    res = validate_and_place([s], tiny, cfg, impossible_range, seed=1)
    assert not res.surviving
    assert len(res.dropped) == 1
    # dropped via the short-circuit, NOT after MAX_TARGET_ATTEMPTS redraws
    assert "without redraw" in res.dropped[0]["reason"]
    assert "redraws" not in res.dropped[0]["reason"]


def test_shrink_to_recover():
    tr = _target_range()
    grid, samples, cfg, _ = _build_grid(tr)
    # A radius far too big for clearance, but radius_max allows a fit -> must shrink.
    big_r = grid.domain_x_m  # absurd
    s = _sample_with_target(samples[0], 1, x=grid.domain_x_m / 2, depth=0.2, r=big_r)
    assert _placement_failures(s.target, grid, cfg)  # precondition: invalid

    res = validate_and_place([s], grid, cfg, tr, seed=3)
    assert len(res.surviving) == 1 and res.n_redrawn == 1
    new_t = res.surviving[0].target
    assert not _placement_failures(new_t, grid, cfg)
    # recovered by shrinking, within the grid floor and the user's max
    assert _r_floor(grid, tr) - 1e-9 <= new_t.radius_m <= tr.radius_max_m + 1e-9
    assert new_t.radius_m < big_r


def test_grid_stays_global_under_target_variation():
    # NOTE: this is NECESSARY but NOT SUFFICIENT — a feature that ignored targets
    # entirely would also produce one grid for all samples and pass this test. The
    # complementary proof that target corners actually FEED the grid lives in
    # test_target_corners_actually_size_the_grid below.
    tr = _target_range()
    grid, samples, cfg, _ = _build_grid(tr, n=8, seed=5)

    # targets genuinely vary across samples
    radii = {round(s.target.radius_m, 6) for s in samples}
    assert len(radii) > 1, "expected varied target radii across samples"

    before = (grid.dx_m, grid.domain_x_m, grid.domain_y_m, grid.time_window_s)
    # placement re-draws some targets but must NOT change the (already-derived) grid
    validate_and_place(list(samples), grid, cfg, tr, seed=7)
    after = (grid.dx_m, grid.domain_x_m, grid.domain_y_m, grid.time_window_s)
    assert before == after, "global grid must not change when targets are re-placed"
    # every surviving sample shares the one global grid (integer cell counts)
    for dim in (grid.domain_x_m, grid.domain_y_m):
        ratio = dim / grid.dx_m
        assert abs(ratio - round(ratio)) < 1e-6


def test_target_corners_actually_size_the_grid():
    """The target corners must measurably change the grid vs a no-target grid.

    Same eps aggregate + config; ONLY the target corner args fed to derive_global
    vary. A small target must give a finer Δx; a deep one a deeper/taller domain;
    a wide one a wider domain. This rules out the silent no-op where the feature
    leaves the grid untouched (which would still pass grid-stays-global).
    """
    ext, cfg, wf, ant = _layers(), _cfg(), _wf(), _ant()
    # eps corners from a no-target draw; reused for every variant so only the
    # target corners differ between grids.
    samples, _w = sample_layers(ext, 6, seed=5, target_range=None)
    _d, agg = derive_samples(samples, cfg, wf)
    assert agg.smallest_feature_global_m is None  # no-target aggregate has no corners

    def g(sf, le, db):
        return derive_global(
            cfg, wf, ant, ext, None, agg.eps_r_max, agg.eps_r_min,
            smallest_feature_global_m=sf, largest_extent_global_m=le,
            deepest_target_bottom_global_m=db,
        )

    base = g(None, None, None)

    # a small target (2*radius/10 below the wavelength Δx) must tighten Δx
    fine = g(0.02, 0.02, 0.10)
    assert fine.dx_m < base.dx_m - 1e-12, (
        f"small target must tighten Δx (base {base.dx_m}, fine {fine.dx_m})"
    )

    # a deep target must deepen depth_z and therefore the vertical domain
    deepest_bottom = 1.50
    deep = g(0.08, 0.08, deepest_bottom)
    assert deep.depth_z_m > base.depth_z_m + 1e-9
    assert deep.domain_y_m > base.domain_y_m + 1e-9
    # AND the deepest target must actually clear the bottom PML by `clearance`
    # (guards the depth_z = deepest_bottom + max(0, clearance - pad) formula
    # against a regression that under-accounts for the bottom pad).
    from backend.validation_tools_new import PML_GAP_CELLS as _GAP
    clearance = (_cfg().pml_cells + _GAP) * deep.dx_m
    target_bottom = deep.ground_y_m - deepest_bottom
    assert target_bottom >= clearance - 1e-9, (
        f"deepest target bottom {target_bottom:.5f} m sits inside the bottom "
        f"PML+gap clearance {clearance:.5f} m"
    )

    # a wide target must widen the horizontal domain
    wide = g(2.0, 2.0, 0.20)
    assert wide.domain_x_m > base.domain_x_m + 1e-9


def test_box_sphere_stub_raises():
    import random

    class _BoxTargetRange:  # stand-in for a future, unsupported target type
        name = "box"
        material = "pec"

    with pytest.raises(NotImplementedError):
        _draw_target(_BoxTargetRange(), random.Random(0))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
