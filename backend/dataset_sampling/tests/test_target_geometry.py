"""
Tests for variable buried-object geometry (cylinders + boxes) on a fixed
global grid.

Covers the invariants that make the feature safe:
  - a dynamic object landing in the PML is re-drawn into the valid region,
  - an object that can NEVER fit (domain too small for its floor size) is
    dropped immediately (no wasted redraws),
  - an object that fails only on size shrinks to recover (per-kind floors),
  - object variation does NOT change the global grid (one grid for all samples),
  - the aggregated corners actually FEED the grid (dx / domain / depth),
  - STATIC objects (min == max ranges) draw identically into every sample, are
    skipped by placement, auto-widen the domain via the symmetric x-halfwidth
    corner, and hard-fail the global gate when widening cannot help,
  - a dynamic failure drops the WHOLE sample,
  - positional target/spec pairing asserts kind equality.

Run: pytest backend/dataset_sampling/tests/test_target_geometry.py
"""
import random
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
    ExtractedTargetRanges,
    DatasetConfig,
    ExtractedWaveform,
    ExtractedAntenna,
    BoxTargetRange,
    CylinderTargetRange,
    SampledSample,
)
from dataset_sampling.layer_sampler import sample_layers  # noqa: E402
from dataset_sampling.peplinski_derive import derive_samples  # noqa: E402
from dataset_sampling.global_derive import derive_global  # noqa: E402
from dataset_sampling import target_shapes  # noqa: E402
from dataset_sampling.global_validation import validate_global  # noqa: E402
from dataset_sampling.target_placement import (  # noqa: E402
    validate_and_place,
    _clearance,
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


def _cyl_range(**overrides):
    kw = dict(
        x_offset_min_m=-0.3, x_offset_max_m=0.3,
        depth_min_m=0.1, depth_max_m=0.35,
        radius_min_m=0.02, radius_max_m=0.06,
    )
    kw.update(overrides)
    return CylinderTargetRange(**kw)


def _box_range(**overrides):
    kw = dict(
        x_offset_min_m=-0.25, x_offset_max_m=0.25,
        depth_min_m=0.15, depth_max_m=0.35,
        width_min_m=0.05, width_max_m=0.15,
        height_min_m=0.04, height_max_m=0.08,
    )
    kw.update(overrides)
    return BoxTargetRange(**kw)


def _tr(cylinders=(), boxes=()):
    return ExtractedTargetRanges(cylinders=list(cylinders), boxes=list(boxes))


def _build_grid(target_ranges, n=6, seed=11):
    """Run draw -> aggregate -> derive and return (grid, samples, cfg, layers)."""
    ext, cfg = _layers(), _cfg()
    samples, _w = sample_layers(ext, n, seed=seed, target_ranges=target_ranges)
    _derived, agg = derive_samples(samples, cfg, _wf(), target_ranges=target_ranges)
    grid = derive_global(
        cfg, _wf(), _ant(), ext, agg.eps_r_max, agg.eps_r_min,
        smallest_feature_global_m=agg.smallest_feature_global_m,
        largest_extent_global_m=agg.largest_extent_global_m,
        deepest_target_bottom_global_m=agg.deepest_target_bottom_global_m,
        static_x_halfwidth_global_m=agg.static_x_halfwidth_global_m,
    )
    return grid, samples, cfg, ext


def _sample_with_targets(template_sample, sample_id, targets):
    """Clone a sample's layers (as dicts, avoiding cross-module instance issues)
    and attach specific targets (list of dicts)."""
    return SampledSample.model_validate({
        "sample_id": sample_id,
        "layers": [l.model_dump() for l in template_sample.layers],
        "targets": targets,
    })


def _cyl_target(x_off, depth, r, name="target"):
    return {"kind": "cylinder", "name": name, "material": "pec",
            "x_offset_m": x_off, "depth_m": depth, "radius_m": r}


def _box_target(x_off, depth, w, h, name="target"):
    return {"kind": "box", "name": name, "material": "pec",
            "x_offset_m": x_off, "depth_m": depth, "width_m": w, "height_m": h}


# ── dynamic placement (redraw / drop) ───────────────────────────────────────

def test_redraw_on_pml_hit():
    tr = _tr(cylinders=[_cyl_range()])
    grid, samples, cfg, _ = _build_grid(tr)
    clr = _clearance(grid, cfg)
    # x offset puts the center inside the LEFT (pml+15) clearance -> invalid
    bad_off = -(grid.domain_x_m / 2.0) + clr * 0.5
    s = _sample_with_targets(samples[0], 1, [_cyl_target(bad_off, 0.2, 0.04)])
    assert target_shapes.placement_failures(s.targets[0], grid, cfg)  # precondition

    res = validate_and_place([s], grid, cfg, tr, seed=1)
    assert len(res.surviving) == 1 and not res.dropped
    assert res.n_redrawn == 1
    assert not target_shapes.placement_failures(res.surviving[0].targets[0], grid, cfg)


def test_immediate_drop_when_impossible():
    tr = _tr(cylinders=[_cyl_range()])
    grid, samples, cfg, _ = _build_grid(tr)
    # Shrink the domain so even the floor radius cannot fit, and force a large
    # lower bound.
    tiny = grid.model_copy(update={
        "domain_x_m": _clearance(grid, cfg) * 2 + grid.dx_m,  # no room for any radius
    })
    impossible = _tr(cylinders=[_cyl_range(radius_min_m=0.5, radius_max_m=0.6)])
    s = _sample_with_targets(samples[0], 1, [_cyl_target(0.0, 0.2, 0.5)])

    res = validate_and_place([s], tiny, cfg, impossible, seed=1)
    assert not res.surviving
    assert len(res.dropped) == 1
    # dropped via the short-circuit, NOT after MAX_TARGET_ATTEMPTS redraws
    assert "without redraw" in res.dropped[0]["reason"]
    assert "redraws" not in res.dropped[0]["reason"]


def test_shrink_to_recover():
    tr = _tr(cylinders=[_cyl_range()])
    grid, samples, cfg, _ = _build_grid(tr)
    spec = tr.cylinders[0]
    # A radius far too big for clearance, but radius_max allows a fit -> shrink.
    big_r = grid.domain_x_m  # absurd
    s = _sample_with_targets(samples[0], 1, [_cyl_target(0.0, 0.2, big_r)])
    assert target_shapes.placement_failures(s.targets[0], grid, cfg)

    res = validate_and_place([s], grid, cfg, tr, seed=3)
    assert len(res.surviving) == 1 and res.n_redrawn == 1
    new_t = res.surviving[0].targets[0]
    assert not target_shapes.placement_failures(new_t, grid, cfg)
    # recovered by shrinking, within the grid floor and the user's max
    # (radius is a HALF-extent: floor = max(5*dx, radius_min) = 10 cells across)
    r_floor = max(5.0 * grid.dx_m, spec.radius_min_m)
    assert r_floor - 1e-9 <= new_t.radius_m <= spec.radius_max_m + 1e-9
    assert new_t.radius_m < big_r


def test_box_redraw_shrinks_within_floors():
    tr = _tr(boxes=[_box_range()])
    grid, samples, cfg, _ = _build_grid(tr)
    spec = tr.boxes[0]
    # Absurdly wide box: must shrink to recover, within per-side floors.
    s = _sample_with_targets(
        samples[0], 1, [_box_target(0.0, 0.25, grid.domain_x_m, 0.06)]
    )
    assert target_shapes.placement_failures(s.targets[0], grid, cfg)

    res = validate_and_place([s], grid, cfg, tr, seed=3)
    assert len(res.surviving) == 1 and res.n_redrawn == 1
    new_t = res.surviving[0].targets[0]
    assert not target_shapes.placement_failures(new_t, grid, cfg)
    # box sides are FULL extents: floors are 10*dx (vs 5*dx for the radius,
    # which is a half-extent — both encode the same >=10-cells rule)
    w_floor = max(10.0 * grid.dx_m, spec.width_min_m)
    h_floor = max(10.0 * grid.dx_m, spec.height_min_m)
    assert w_floor - 1e-9 <= new_t.width_m <= spec.width_max_m + 1e-9
    assert h_floor - 1e-9 <= new_t.height_m <= spec.height_max_m + 1e-9


def test_dynamic_failure_drops_whole_sample():
    tr = _tr(cylinders=[_cyl_range()])
    grid, samples, cfg, _ = _build_grid(tr)
    # Two specs: a placeable cylinder + one whose floor can never fit the grid.
    specs = _tr(cylinders=[
        _cyl_range(),
        _cyl_range(radius_min_m=grid.domain_x_m, radius_max_m=grid.domain_x_m + 0.1),
    ])
    good = _cyl_target(0.0, 0.2, 0.04, name="ok")
    assert not target_shapes.placement_failures(
        SampledSample.model_validate(
            {"sample_id": 9, "layers": [l.model_dump() for l in samples[0].layers],
             "targets": [good]}
        ).targets[0], grid, cfg)
    bad = _cyl_target(0.0, 0.2, grid.domain_x_m, name="huge")
    s = _sample_with_targets(samples[0], 1, [good, bad])

    res = validate_and_place([s], grid, cfg, specs, seed=1)
    # ONE failing dynamic object drops the WHOLE sample.
    assert not res.surviving
    assert len(res.dropped) == 1
    assert "object #1" in res.dropped[0]["reason"]
    assert "huge" in res.dropped[0]["reason"]


def test_pairing_asserts_kind_per_index():
    tr = _tr(cylinders=[_cyl_range()])
    grid, samples, cfg, _ = _build_grid(tr)
    # Drawn kind at index 0 is a box, but the canonical spec there is a cylinder.
    s = _sample_with_targets(samples[0], 1, [_box_target(0.0, 0.25, 0.1, 0.05)])
    with pytest.raises(AssertionError, match="canonical ordering"):
        validate_and_place([s], grid, cfg, tr, seed=1)


# ── draws & aggregation ─────────────────────────────────────────────────────

def test_box_and_cylinder_draw_within_ranges():
    tr = _tr(cylinders=[_cyl_range()], boxes=[_box_range()])
    _grid, samples, _cfg_, _ = _build_grid(tr, n=8, seed=5)
    for s in samples:
        assert len(s.targets) == 2
        cyl, box = s.targets  # canonical order: cylinders, then boxes
        assert cyl.kind == "cylinder" and box.kind == "box"
        c_spec, b_spec = tr.cylinders[0], tr.boxes[0]
        assert c_spec.x_offset_min_m <= cyl.x_offset_m <= c_spec.x_offset_max_m
        assert c_spec.depth_min_m <= cyl.depth_m <= c_spec.depth_max_m
        assert c_spec.radius_min_m <= cyl.radius_m <= c_spec.radius_max_m
        assert b_spec.x_offset_min_m <= box.x_offset_m <= b_spec.x_offset_max_m
        assert b_spec.width_min_m <= box.width_m <= b_spec.width_max_m
        assert b_spec.height_min_m <= box.height_m <= b_spec.height_max_m
    # negative offsets are legal and actually drawn
    offs = [t.x_offset_m for s in samples for t in s.targets]
    assert any(o < 0 for o in offs)


def test_multi_object_aggregation_worst_case():
    ext, cfg, wf = _layers(), _cfg(), _wf()
    base_samples, _w = sample_layers(ext, 2, seed=5)
    fixed = [
        [_cyl_target(0.1, 0.30, 0.03), _box_target(-0.1, 0.20, 0.05, 0.02)],
        [_cyl_target(0.0, 0.10, 0.05), _box_target(0.2, 0.40, 0.30, 0.10)],
    ]
    samples = [
        _sample_with_targets(base_samples[i], i + 1, tgts)
        for i, tgts in enumerate(fixed)
    ]
    _d, agg = derive_samples(samples, cfg, wf)
    # smallest in-plane feature: box min(w,h)=0.02 beats every cylinder 2r
    assert agg.smallest_feature_global_m == pytest.approx(0.02)
    # largest x extent: the 0.30 m wide box
    assert agg.largest_extent_global_m == pytest.approx(0.30)
    # deepest bottom: box depth 0.40 + h/2 = 0.45
    assert agg.deepest_target_bottom_global_m == pytest.approx(0.45)


def test_feature_math_is_grid_independent():
    """smallest_feature/largest_extent/bottom_depth take ONLY the target — the
    thin-z (one cell) extent structurally cannot leak into the feature."""
    from backend.schema import SampledTarget
    thin_box = SampledTarget.model_validate(_box_target(0.0, 0.3, 0.5, 0.04))
    assert target_shapes.smallest_feature(thin_box) == pytest.approx(0.04)
    assert target_shapes.largest_extent(thin_box) == pytest.approx(0.5)
    assert target_shapes.bottom_depth(thin_box) == pytest.approx(0.32)


# ── one grid for all samples / corners feed the grid ────────────────────────

def test_grid_stays_global_under_target_variation():
    # NOTE: this is NECESSARY but NOT SUFFICIENT — a feature that ignored targets
    # entirely would also produce one grid for all samples and pass this test. The
    # complementary proof that target corners actually FEED the grid lives in
    # test_target_corners_actually_size_the_grid below.
    tr = _tr(cylinders=[_cyl_range()])
    grid, samples, cfg, _ = _build_grid(tr, n=8, seed=5)

    # targets genuinely vary across samples
    radii = {round(s.targets[0].radius_m, 6) for s in samples}
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
    samples, _w = sample_layers(ext, 6, seed=5)
    _d, agg = derive_samples(samples, cfg, wf)
    assert agg.smallest_feature_global_m is None  # no-target aggregate has no corners

    def g(sf, le, db, sxh=None):
        return derive_global(
            cfg, wf, ant, ext, agg.eps_r_max, agg.eps_r_min,
            smallest_feature_global_m=sf, largest_extent_global_m=le,
            deepest_target_bottom_global_m=db,
            static_x_halfwidth_global_m=sxh,
        )

    base = g(None, None, None)

    # a small target (feature/10 below the wavelength Δx) must tighten Δx
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

    # a pinned STATIC object's halfwidth must widen the domain SYMMETRICALLY
    # (2*(halfwidth + clearance)), covering left- and right-pinned objects alike
    pinned = g(0.08, 0.08, 0.20, sxh=1.2)
    clearance_p = (_cfg().pml_cells + _GAP) * pinned.dx_m
    assert pinned.domain_x_m >= 2.0 * (1.2 + clearance_p) - 1e-9
    assert pinned.domain_x_m > g(0.08, 0.08, 0.20).domain_x_m + 1e-9


# ── static objects ──────────────────────────────────────────────────────────

def _static_cyl(x_off=-0.4, depth=0.25, r=0.04):
    return _cyl_range(
        x_offset_min_m=x_off, x_offset_max_m=x_off,
        depth_min_m=depth, depth_max_m=depth,
        radius_min_m=r, radius_max_m=r,
    )


def test_static_detection():
    assert _static_cyl().is_static
    assert not _cyl_range().is_static
    # PARTIALLY degenerate = dynamic
    assert not _cyl_range(depth_min_m=0.2, depth_max_m=0.2).is_static


def test_static_object_identical_across_samples():
    tr = _tr(cylinders=[_static_cyl()], boxes=[_box_range()])
    _grid, samples, _c, _ = _build_grid(tr, n=6, seed=3)
    statics = [s.targets[0] for s in samples]
    first = statics[0]
    for t in statics[1:]:
        assert t == first, "static object must draw identically in every sample"
    # the ranged box still varies
    widths = {round(s.targets[1].width_m, 6) for s in samples}
    assert len(widths) > 1


def test_static_left_pinned_widens_domain():
    static = _static_cyl(x_off=-0.4)
    grid_with, _s, cfg, _ = _build_grid(_tr(cylinders=[static]), n=4, seed=3)
    # halfwidth = |x_offset| + extent/2 = 0.4 + 0.04; widening is symmetric
    clearance = _clearance(grid_with, cfg)
    assert grid_with.domain_x_m >= 2.0 * (0.44 + clearance) - 1e-9
    # and the pinned object actually PLACES on the derived grid (gate would pass)
    t = target_shapes.draw_target(static, random.Random(0))
    assert not target_shapes.placement_failures(t, grid_with, cfg)


def test_static_object_skipped_by_placement():
    static = _static_cyl()
    tr = _tr(cylinders=[static])
    grid, samples, cfg, _ = _build_grid(tr, n=3, seed=3)
    # Artificially misplace the static object (inside the left PML clearance):
    # placement must NOT touch it — static objects are gate-validated only.
    bad_off = -(grid.domain_x_m / 2.0) + _clearance(grid, cfg) * 0.5
    misplaced = _cyl_target(bad_off, static.depth_min_m, static.radius_min_m)
    s = _sample_with_targets(samples[0], 1, [misplaced])
    assert target_shapes.placement_failures(s.targets[0], grid, cfg)

    res = validate_and_place([s], grid, cfg, tr, seed=1)
    assert len(res.surviving) == 1 and not res.dropped and res.n_redrawn == 0
    assert res.surviving[0].targets[0].x_offset_m == pytest.approx(bad_off)


def test_static_object_fails_global_gate():
    # A static cylinder with depth < radius is not fully buried — a violation
    # that domain widening can never fix; it must surface as a gate ERROR.
    shallow = _static_cyl(x_off=0.0, depth=0.01, r=0.05)
    tr = _tr(cylinders=[shallow])
    grid, _s, cfg, ext = _build_grid(tr, n=3, seed=3)

    report = validate_global(grid, cfg, _wf(), _ant(), ext, target_ranges=tr)
    static_errors = [e for e in report.errors if "[static_target_placement]" in e]
    assert static_errors, f"expected a static-placement gate error, got: {report.errors}"
    assert "never repositioned" in static_errors[0]

    # sanity: a well-placed static object produces NO static gate errors
    ok = _tr(cylinders=[_static_cyl(x_off=0.1, depth=0.25, r=0.04)])
    grid_ok, _s2, cfg2, ext2 = _build_grid(ok, n=3, seed=3)
    report_ok = validate_global(grid_ok, cfg2, _wf(), _ant(), ext2, target_ranges=ok)
    assert not [e for e in report_ok.errors if "[static_target_placement]" in e]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
