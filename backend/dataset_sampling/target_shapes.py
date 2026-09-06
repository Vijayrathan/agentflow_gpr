"""
Per-kind geometry math for buried targets (cylinder, box) in ONE place, shared
by the sampler (draw), the Peplinski-derive aggregation (grid corners), the
global-validation gate (static objects) and the per-sample placement pass
(dynamic objects). No file I/O here.

Conventions (see backend/schema.py target-range block):
  - x_offset_m: SIGNED offset of the object's center from the domain's
    horizontal center; absolute x = domain_x/2 + x_offset (grid required).
  - depth_m: depth of the object's CENTER below the ground surface;
    y_center = ground_y - depth (grid required).
  - Feature/extent/bottom helpers take ONLY the target — no grid argument —
    which structurally guarantees the thin-z extent (one cell) can never leak
    into the feature computation. 2D features use the active plane; finite 3D length and crossline sides also constrain spacing.

Dispatch is on the `kind` field, never isinstance: the codebase has dual
import paths (`schema` vs `backend.schema`) that resolve to distinct module
objects, so isinstance checks are unreliable across them.
"""
from __future__ import annotations

import random
from typing import List, Tuple

from backend.schema import (
    DatasetConfig,
    ExtractedTargetRanges,
    GlobalDerived,
    SampledTarget,
)
from backend.validation_tools_new import validate_target


def _round(x: float, ndigits: int = 6) -> float:
    return round(x, ndigits)


# ---------------------------------------------------------------------------
# Range-spec helpers (pre-grid: operate on the collected min/max ranges)
# ---------------------------------------------------------------------------

def iter_ranges(tr: ExtractedTargetRanges) -> List:
    """The CANONICAL object order: cylinders in list order, then boxes.

    Every consumer (sampler, placement re-pairing, viz) relies on this order —
    samples[i].targets[j] corresponds positionally to iter_ranges(tr)[j].
    """
    return list(tr.cylinders) + list(tr.boxes)


def draw_target(spec, rng: random.Random, independent_seed=None) -> SampledTarget:
    """Draw one concrete target from its range spec (grid-independent).

    A static spec (all min == max) draws identically every time — degenerate
    uniform draws need no special-casing.
    """
    def draw(field):
        from backend.dataset_sampling.contract import stream_seed
        lo, hi = getattr(spec, field + "_min_m"), getattr(spec, field + "_max_m")
        if lo == hi:
            if independent_seed is None:
                rng.uniform(lo, hi)  # retain the legacy stream consumption
            return lo
        if independent_seed is not None:
            return random.Random(stream_seed(independent_seed, field)).uniform(lo, hi)
        return _round(rng.uniform(lo, hi))

    base = dict(
        kind=spec.kind,
        name=spec.name,
        material=spec.material,
        x_offset_m=draw("x_offset"),
        depth_m=draw("depth"),
    )
    if spec.z_offset_min_m is not None:
        base["z_offset_m"] = draw("z_offset")
    if spec.kind == "cylinder":
        if spec.length_min_m is not None:
            base.update(length_m=draw("length"), cylinder_axis=spec.cylinder_axis)
        return SampledTarget(
            **base,
            radius_m=draw("radius"),
        )
    if spec.kind == "box":
        if spec.crossline_size_min_m is not None:
            base["crossline_size_m"] = draw("crossline_size")
        return SampledTarget(
            **base,
            width_m=draw("width"),
            height_m=draw("height"),
        )
    raise ValueError(f"Unknown target kind '{spec.kind}'")


def spec_extent_x_max(spec) -> float:
    """Worst-case x extent of a range spec (for static-footprint aggregation)."""
    if spec.kind == "cylinder":
        return 2.0 * spec.radius_max_m
    if spec.kind == "box":
        return spec.width_max_m
    raise ValueError(f"Unknown target kind '{spec.kind}'")


def static_x_halfwidth(tr: ExtractedTargetRanges):
    """max(|x_offset| + extent/2) over STATIC specs, or None when there are none.

    Center-relative offsets make this symmetric: widening domain_x to
    2*(halfwidth + clearance) accommodates both left- and right-pinned fixed
    objects. Dynamic objects are excluded — redraw handles their positioning.
    """
    vals = [
        max(abs(s.x_offset_min_m), abs(s.x_offset_max_m)) + spec_extent_x_max(s) / 2.0
        for s in iter_ranges(tr)
        if s.is_static
    ]
    return max(vals) if vals else None


# ---------------------------------------------------------------------------
# Drawn-target helpers (feature math — NO grid argument, in-plane only)
# ---------------------------------------------------------------------------

def half_extents(t: SampledTarget) -> Tuple[float, float]:
    """In-plane half extents (hx, hy) around the object center."""
    if t.length_m is not None or t.crossline_size_m is not None:
        return half_extents_3d(t)[:2]
    if t.kind == "cylinder":
        return t.radius_m, t.radius_m
    if t.kind == "box":
        return t.width_m / 2.0, t.height_m / 2.0
    raise ValueError(f"Unknown target kind '{t.kind}'")


def half_extents_3d(t):
    if t.kind == "box":
        return t.width_m / 2, t.height_m / 2, (t.crossline_size_m / 2 if t.crossline_size_m else 0)
    extents = [t.radius_m] * 3
    extents["xyz".index(t.cylinder_axis or "z")] = t.length_m / 2 if t.length_m else 0
    return tuple(extents)


def smallest_feature(t: SampledTarget) -> float:
    """Smallest in-plane dimension (feeds the >=10-cells Δx tightening).
    Cylinder: the diameter. Box: the smaller side. Thin z NEVER enters."""
    if t.kind == "cylinder":
        return min(2.0 * t.radius_m, t.length_m) if t.length_m is not None else 2.0 * t.radius_m
    if t.kind == "box":
        return min(t.width_m, t.height_m, t.crossline_size_m or float("inf"))
    raise ValueError(f"Unknown target kind '{t.kind}'")


def largest_extent(t: SampledTarget) -> float:
    """Horizontal (x) extent — widens domain_x."""
    return 2 * half_extents(t)[0]


def bottom_depth(t: SampledTarget) -> float:
    """Depth of the object's BOTTOM below ground — deepens depth_z."""
    _, hy = half_extents(t)
    return t.depth_m + hy


# ---------------------------------------------------------------------------
# Grid-dependent helpers (absolute placement)
# ---------------------------------------------------------------------------

def target_bbox(t: SampledTarget, grid: GlobalDerived):
    """Absolute (min, max) bbox tuples (x, y, z); thin z spans one cell."""
    hx, hy = half_extents(t)
    x_abs = grid.domain_x_m / 2.0 + t.x_offset_m
    y_center = grid.ground_y_m - t.depth_m
    if grid.dimensionality == "3D":
        hx, hy, hz = half_extents_3d(t)
        z_center = grid.domain_z_m / 2 + t.z_offset_m
        return ((x_abs - hx, y_center - hy, z_center - hz),
                (x_abs + hx, y_center + hy, z_center + hz))
    bbox_min = (x_abs - hx, y_center - hy, 0.0)
    bbox_max = (x_abs + hx, y_center + hy, grid.dx_m)
    return bbox_min, bbox_max


def placement_failures(
    t: SampledTarget, grid: GlobalDerived, cfg: DatasetConfig
) -> List[str]:
    """Placement problems for one drawn target ([] when valid).

    validate_target errors (outside domain, ANY face) AND warnings (within the
    (pml+gap) clearance, or fewer than 10 cells across) both count as failures;
    plus the fully-buried check (object top at or below the ground surface).
    Shared by the global-validation gate (static objects) and the per-sample
    placement pass (dynamic objects).
    """
    if cfg.contract_version >= 2:
        from backend.dataset_sampling.scene import resolve_target, validate_resolved_target
        try:
            return validate_resolved_target(resolve_target(t, grid, cfg), grid, cfg)
        except ValueError as exc:
            return [str(exc)]
    bbox_min, bbox_max = target_bbox(t, grid)
    e, w = validate_target(
        name=t.name,
        min_dimension_m=smallest_feature(t),
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        domain=(grid.domain_x_m, grid.domain_y_m, grid.domain_z_m),
        max_cell_m=grid.dx_m,
        pml_cells=cfg.pml_cells,
    )
    reasons = list(e) + list(w)
    _, hy = half_extents(t)
    if t.depth_m < hy - 1e-9:  # top of object above the ground surface
        reasons.append(
            f"target '{t.name}' ({t.kind}) not fully buried "
            f"(center depth {t.depth_m:.4f} m < half height {hy:.4f} m)"
        )
    return reasons
