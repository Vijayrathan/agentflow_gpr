"""
STAGE 7 — per-sample buried-target placement validation (+ redraw / drop).

Runs AFTER the global derive (it needs the domain) but the grid is FIXED here:
redraws must fit the existing Δx / domain and NEVER trigger a re-derive. For each
sample that drew a buried target we:

  1. Build the absolute target bbox (cylinder: centre (x_center, ground_y - depth)
     ± radius; the thin out-of-plane axis spans one cell) and validate it with
     validate_target PLUS a fully-buried check (target top <= surface).
  2. On failure, re-draw THAT target's geometry only (radius shrink + reposition)
     within the radius-dependent valid envelope (domain minus the (pml+15) gap),
     up to MAX_TARGET_ATTEMPTS. If even the smallest grid-valid radius (r_floor)
     cannot fit, DROP immediately (no wasted attempts).
  3. If it still fails after the cap, DROP the sample and log (sample_id, reason).

Dropped samples REDUCE the dataset size N (no backfill). Surviving samples (with
updated targets) are written back to the manifest; dropped ones are logged.

Grid-faithful radius floor: any radius >= 5*Δx resolves to >=10 cells on the
fixed grid, so r_floor = max(5*Δx, radius_min_m). Because the grid was tightened
by smallest_feature_global/10, every initial target already has radius >= r_floor,
so re-draw is shrink/reposition only — the impossible case is purely positional.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from backend.schema import (
    DatasetConfig,
    GlobalDerived,
    SampledSample,
    SampledTarget,
    CylinderTargetRange,
)
from backend.validation_tools_new import PML_GAP_CELLS, validate_target
from dataset_sampling.layer_sampler import MAX_TARGET_ATTEMPTS

# tiny inward margin (in cells) so envelope edges sit strictly inside the (pml+15)
# gap despite floating-point noise at the boundary.
_EDGE_TOL_CELLS = 1e-6


def _clearance(grid: GlobalDerived, cfg: DatasetConfig) -> float:
    return (cfg.pml_cells + PML_GAP_CELLS) * grid.dx_m


def _target_bbox(target: SampledTarget, grid: GlobalDerived):
    """Absolute (min, max) bbox tuples (x, y, z) for a cylinder disc in 2D."""
    y_center = grid.ground_y_m - target.depth_m
    r = target.radius_m
    bbox_min = (target.x_center_m - r, y_center - r, 0.0)
    bbox_max = (target.x_center_m + r, y_center + r, grid.dx_m)  # thin z = 1 cell
    return bbox_min, bbox_max


def _placement_failures(
    target: SampledTarget, grid: GlobalDerived, cfg: DatasetConfig
) -> List[str]:
    """Return the list of placement problems ([] when the target is valid).

    validate_target errors (outside domain) AND warnings (within the (pml+15)
    gap, or fewer than 10 cells across) both count as placement failures here;
    plus an explicit fully-buried check (target top must stay below the surface).
    """
    bbox_min, bbox_max = _target_bbox(target, grid)
    e, w = validate_target(
        name=target.name,
        min_dimension_m=2.0 * target.radius_m,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        domain=(grid.domain_x_m, grid.domain_y_m, grid.dx_m),
        max_cell_m=grid.dx_m,
        pml_cells=cfg.pml_cells,
    )
    reasons = list(e) + list(w)
    # fully buried: top of the disc must be at or below the ground surface.
    if (grid.ground_y_m - target.depth_m) + target.radius_m > grid.ground_y_m + 1e-9:
        reasons.append(
            f"target '{target.name}' not fully buried "
            f"(depth {target.depth_m:.4f} < radius {target.radius_m:.4f})"
        )
    return reasons


def _r_floor(grid: GlobalDerived, target_range: CylinderTargetRange) -> float:
    """Smallest grid-valid radius: >=10 cells across AND >= the user's lower bound."""
    return max(5.0 * grid.dx_m, target_range.radius_min_m)


def _feasible_at_r_floor(
    grid: GlobalDerived, cfg: DatasetConfig, r_floor: float
) -> Tuple[bool, bool]:
    """Can ANY centre fit at the smallest radius? (x_feasible, y_feasible).

    x: domain_x - 2*clearance - 2*r_floor >= 0
    y: a depth in [r_floor (fully buried), ground_y - clearance - r_floor] exists
       <=> ground_y - clearance - 2*r_floor >= 0
    """
    clearance = _clearance(grid, cfg)
    x_ok = grid.domain_x_m - 2.0 * clearance - 2.0 * r_floor >= 0.0
    y_ok = grid.ground_y_m - clearance - 2.0 * r_floor >= 0.0
    return x_ok, y_ok


def _redraw_target(
    target: SampledTarget,
    grid: GlobalDerived,
    cfg: DatasetConfig,
    target_range: CylinderTargetRange,
    rng: random.Random,
) -> Tuple[Optional[SampledTarget], int]:
    """Re-draw a target into the valid envelope. Returns (new_target|None, attempts).

    Radius is shrunk within [r_floor, original_radius] (the grid guarantees
    original >= r_floor); centre is drawn within the radius-dependent envelope
    (domain minus the (pml+15) gap, fully buried). None if no attempt succeeded.
    """
    clearance = _clearance(grid, cfg)
    r_floor = _r_floor(grid, target_range)
    # INVARIANT (not just a comment): the global derive tightens Δx by
    # smallest_feature_global/10, so every drawn target already resolves to >=10
    # cells and thus original_radius >= r_floor. The shrink range [r_floor,
    # original_radius] therefore can't invert. If a future change to the Δx
    # derivation or the draw breaks this, fail LOUDLY here rather than silently
    # producing an inverted/empty interval or a radius above the user's max.
    # (tol covers the 6-decimal rounding applied to drawn radii.)
    assert target.radius_m >= r_floor - 1e-6, (
        f"placement invariant violated: original radius {target.radius_m:.6f} m < "
        f"r_floor {r_floor:.6f} m (= max(5*Δx={5*grid.dx_m:.6f}, "
        f"radius_min={target_range.radius_min_m:.6f})). The global derive's "
        "smallest_feature tightening no longer guarantees the 10-cell rule for "
        "drawn targets — re-check global_derive.derive_global before trusting this."
    )
    r_hi = min(target.radius_m, target_range.radius_max_m)
    tol = _EDGE_TOL_CELLS * grid.dx_m
    attempts = 0
    for _ in range(MAX_TARGET_ATTEMPTS):
        attempts += 1
        radius = rng.uniform(r_floor, r_hi) if r_hi > r_floor else r_floor
        x_lo, x_hi = clearance + radius + tol, grid.domain_x_m - clearance - radius - tol
        d_lo, d_hi = radius + tol, grid.ground_y_m - clearance - radius - tol
        if x_lo > x_hi or d_lo > d_hi:
            continue  # this radius is too big for the domain — try a smaller one
        candidate = SampledTarget(
            kind=target.kind,
            name=target.name,
            material=target.material,
            x_center_m=round(rng.uniform(x_lo, x_hi), 6),
            depth_m=round(rng.uniform(d_lo, d_hi), 6),
            radius_m=round(radius, 6),
        )
        if not _placement_failures(candidate, grid, cfg):
            return candidate, attempts
    return None, attempts


@dataclass
class PlacementResult:
    surviving: List[SampledSample] = field(default_factory=list)
    dropped: List[dict] = field(default_factory=list)   # {sample_id, reason}
    n_redrawn: int = 0
    n_unchanged: int = 0


def validate_and_place(
    samples: List[SampledSample],
    grid: GlobalDerived,
    cfg: DatasetConfig,
    target_range: Optional[CylinderTargetRange],
    seed: Optional[int] = 1234,
) -> PlacementResult:
    """Validate each sample's target against the FIXED grid; redraw or drop."""
    result = PlacementResult()
    rng = random.Random(seed)
    r_floor = _r_floor(grid, target_range) if target_range is not None else None

    for s in samples:
        # No target on this sample -> nothing grid-dependent to check; keep it.
        if s.target is None:
            result.surviving.append(s)
            result.n_unchanged += 1
            continue

        if not _placement_failures(s.target, grid, cfg):
            result.surviving.append(s)
            result.n_unchanged += 1
            continue

        # Failed placement. If even r_floor cannot fit, drop immediately.
        if target_range is not None:
            x_ok, y_ok = _feasible_at_r_floor(grid, cfg, r_floor)
            if not (x_ok and y_ok):
                axes = ", ".join(a for a, ok in (("x", x_ok), ("y/depth", y_ok)) if not ok)
                result.dropped.append({
                    "sample_id": s.sample_id,
                    "reason": (
                        f"impossible to fit even at r_floor={r_floor:.4f} m on {axes} "
                        f"(domain {grid.domain_x_m:.3f}x{grid.domain_y_m:.3f} m, "
                        f"ground_y={grid.ground_y_m:.3f} m) — dropped without redraw"
                    ),
                })
                continue

        new_target = None
        if target_range is not None:
            new_target, _attempts = _redraw_target(s.target, grid, cfg, target_range, rng)

        if new_target is None:
            result.dropped.append({
                "sample_id": s.sample_id,
                "reason": (
                    f"could not place target within {MAX_TARGET_ATTEMPTS} redraws "
                    f"(last failures: {'; '.join(_placement_failures(s.target, grid, cfg))})"
                ),
            })
            continue

        s.target = new_target
        result.surviving.append(s)
        result.n_redrawn += 1

    return result


# ---------------------------------------------------------------------------
# File-IO orchestration (reads/writes the dataset manifests)
# ---------------------------------------------------------------------------

def _resolve(output_dir: str, filename: str) -> Path:
    path = Path(output_dir)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent.parent / path
    return path / filename


def run_placement(
    output_dir: str,
    cfg: DatasetConfig,
    grid: GlobalDerived,
    target_range: Optional[CylinderTargetRange],
    seed: Optional[int] = 1234,
    samples_filename: str = "sampled_layers.json",
    dropped_filename: str = "dropped_targets.json",
) -> PlacementResult:
    """Load the sampled manifest, place/redraw/drop targets, rewrite the manifest.

    Preserves the manifest's existing `warnings`, replaces `samples` with the
    survivors, updates `num_samples`, and records dropped samples both inline and
    in a separate `dropped_targets.json`.
    """
    samples_path = _resolve(output_dir, samples_filename)
    with open(samples_path, encoding="utf-8") as f:
        payload = json.load(f)
    samples = [SampledSample.model_validate(s) for s in payload["samples"]]

    result = validate_and_place(samples, grid, cfg, target_range, seed=seed)

    payload["num_samples"] = len(result.surviving)
    payload["samples"] = [s.model_dump() for s in result.surviving]
    payload["dropped_targets"] = result.dropped
    with open(samples_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    dropped_path = _resolve(output_dir, dropped_filename)
    with open(dropped_path, "w", encoding="utf-8") as f:
        json.dump({"dropped_targets": result.dropped}, f, indent=2)

    return result
