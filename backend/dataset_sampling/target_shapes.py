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
    into the feature computation. All features are IN-PLANE dimensions.

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


def draw_target(spec, rng: random.Random) -> SampledTarget:
    """Draw one concrete target from its range spec (grid-independent).

    A static spec (all min == max) draws identically every time — degenerate
    uniform draws need no special-casing.
    """
    base = dict(
        kind=spec.kind,
        name=spec.name,
        material=spec.material,
        x_offset_m=_round(rng.uniform(spec.x_offset_min_m, spec.x_offset_max_m)),
        depth_m=_round(rng.uniform(spec.depth_min_m, spec.depth_max_m)),
    )
    if spec.kind == "cylinder":
        return SampledTarget(
            **base,
            radius_m=_round(rng.uniform(spec.radius_min_m, spec.radius_max_m)),
        )
    if spec.kind == "box":
        return SampledTarget(
            **base,
            width_m=_round(rng.uniform(spec.width_min_m, spec.width_max_m)),
            height_m=_round(rng.uniform(spec.height_min_m, spec.height_max_m)),
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
    if t.kind == "cylinder":
        return t.radius_m, t.radius_m
    if t.kind == "box":
        return t.width_m / 2.0, t.height_m / 2.0
    raise ValueError(f"Unknown target kind '{t.kind}'")


def smallest_feature(t: SampledTarget) -> float:
    """Smallest in-plane dimension (feeds the >=10-cells Δx tightening).
    Cylinder: the diameter. Box: the smaller side. Thin z NEVER enters."""
    if t.kind == "cylinder":
        return 2.0 * t.radius_m
    if t.kind == "box":
        return min(t.width_m, t.height_m)
    raise ValueError(f"Unknown target kind '{t.kind}'")


def largest_extent(t: SampledTarget) -> float:
    """Horizontal (x) extent — widens domain_x."""
    if t.kind == "cylinder":
        return 2.0 * t.radius_m
    if t.kind == "box":
        return t.width_m
    raise ValueError(f"Unknown target kind '{t.kind}'")


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
    bbox_min, bbox_max = target_bbox(t, grid)
    e, w = validate_target(
        name=t.name,
        min_dimension_m=smallest_feature(t),
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        domain=(grid.domain_x_m, grid.domain_y_m, grid.dx_m),
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
