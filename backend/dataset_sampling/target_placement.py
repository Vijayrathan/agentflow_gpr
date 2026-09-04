"""
STAGE 7 — per-sample buried-target placement validation (+ redraw / drop).

Runs AFTER the global derive (it needs the domain) but the grid is FIXED here:
redraws must fit the existing Δx / domain and NEVER trigger a re-derive. Each
sample carries `targets` — one drawn object per collected range spec, in the
canonical order (cylinders, then boxes). For every DYNAMIC (ranged) object we:

  1. Validate its absolute placement (shared target_shapes.placement_failures:
     in-domain, PML+gap clearance, >=10 cells across, fully buried).
  2. On failure, re-draw THAT object's geometry only (size shrink + reposition)
     within its size-dependent valid envelope (domain minus the (pml+15) gap),
     up to MAX_TARGET_ATTEMPTS. If even the smallest grid-valid size cannot
     fit, DROP immediately (no wasted attempts).
  3. If ANY dynamic object of a sample still fails after the cap, DROP the
     WHOLE sample and log (sample_id, reason) — surviving files always contain
     the full object set the user specified.

STATIC objects (range min == max on every field) are SKIPPED here entirely:
they are identical in every sample and were placement-validated ONCE by the
global-validation gate — this pass never repositions, resizes or drops them.

Dropped samples REDUCE the dataset size N (no backfill). Surviving samples
(with updated targets) are written back to the manifest; dropped ones logged.

Grid-faithful size floors (the >=10-cells-across rule, per kind):
  cylinder: r_floor = max(5*Δx, radius_min_m) — the radius is a HALF-extent,
            so 5*Δx puts 10 cells across the DIAMETER.
  box:      w_floor = max(10*Δx, width_min_m), h_floor = max(10*Δx, height_min_m)
            — box sides are FULL extents, so 10*Δx puts 10 cells across the
            side. The 5*Δx vs 10*Δx difference is CORRECT (half- vs full-
            dimension); do not "fix" one to match the other.
Because the grid was tightened by smallest_feature_global/10, every drawn
object already satisfies its floor, so re-draw is shrink/reposition only —
the impossible case is purely positional.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from backend.schema import (
    DatasetConfig,
    ExtractedTargetRanges,
    GlobalDerived,
    SampledSample,
    SampledTarget,
)
from backend.validation_tools_new import PML_GAP_CELLS
from dataset_sampling.layer_sampler import MAX_TARGET_ATTEMPTS
from dataset_sampling.target_shapes import iter_ranges, placement_failures

# tiny inward margin (in cells) so envelope edges sit strictly inside the (pml+15)
# gap despite floating-point noise at the boundary.
_EDGE_TOL_CELLS = 1e-6


def _clearance(grid: GlobalDerived, cfg: DatasetConfig) -> float:
    return (cfg.pml_cells + PML_GAP_CELLS) * grid.dx_m


def _floor_half_extents(grid: GlobalDerived, spec) -> Tuple[float, float]:
    """Smallest grid-valid HALF extents (hx, hy) for a range spec.

    cylinder: r_floor = max(5*Δx, radius_min_m). The radius is a half-extent:
              5*Δx ⇒ 10 cells across the diameter (the >=10-cells rule).
    box:      sides are FULL extents: floors are max(10*Δx, side_min) ⇒ 10 cells
              across the side; halved here. NOT an inconsistency with 5*Δx above.
    """
    if spec.kind == "cylinder":
        r_floor = max(5.0 * grid.dx_m, spec.radius_min_m)
        return r_floor, r_floor
    if spec.kind == "box":
        w_floor = max(10.0 * grid.dx_m, spec.width_min_m)
        h_floor = max(10.0 * grid.dx_m, spec.height_min_m)
        return w_floor / 2.0, h_floor / 2.0
    raise ValueError(f"Unknown target kind '{spec.kind}'")


def _feasible_at_floor(
    grid: GlobalDerived, cfg: DatasetConfig, spec
) -> Tuple[bool, bool]:
    """Can ANY center fit at the smallest grid-valid size? (x_feasible, y_feasible).

    x: domain_x - 2*clearance - 2*hx_floor >= 0
    y: a depth in [hy_floor (fully buried), ground_y - clearance - hy_floor]
       exists <=> ground_y - clearance - 2*hy_floor >= 0
    """
    hx_f, hy_f = _floor_half_extents(grid, spec)
    clearance = _clearance(grid, cfg)
    x_ok = grid.domain_x_m - 2.0 * clearance - 2.0 * hx_f >= 0.0
    y_ok = grid.ground_y_m - clearance - 2.0 * hy_f >= 0.0
    return x_ok, y_ok


def _assert_size_invariant(target: SampledTarget, grid: GlobalDerived, spec) -> None:
    """INVARIANT (not just a comment): the global derive tightens Δx by
    smallest_feature_global/10, so every drawn object already resolves to >=10
    cells and its dimensions sit at or above the grid floors. If a future
    change to the Δx derivation or the draw breaks this, fail LOUDLY rather
    than silently producing an inverted/empty shrink interval.
    (tol covers the 6-decimal rounding applied to drawn sizes.)"""
    tol = 1e-6
    if spec.kind == "cylinder":
        r_floor = max(5.0 * grid.dx_m, spec.radius_min_m)
        assert target.radius_m >= r_floor - tol, (
            f"placement invariant violated: radius {target.radius_m:.6f} m < "
            f"r_floor {r_floor:.6f} m (= max(5*Δx={5*grid.dx_m:.6f}, "
            f"radius_min={spec.radius_min_m:.6f})). The global derive's "
            "smallest_feature tightening no longer guarantees the 10-cell rule "
            "for drawn targets — re-check global_derive.derive_global."
        )
        return
    w_floor = max(10.0 * grid.dx_m, spec.width_min_m)
    h_floor = max(10.0 * grid.dx_m, spec.height_min_m)
    assert target.width_m >= w_floor - tol and target.height_m >= h_floor - tol, (
        f"placement invariant violated: box {target.width_m:.6f}x"
        f"{target.height_m:.6f} m below floors {w_floor:.6f}x{h_floor:.6f} m "
        f"(= max(10*Δx={10*grid.dx_m:.6f}, side_min)). The global derive's "
        "smallest_feature tightening no longer guarantees the 10-cell rule "
        "for drawn targets — re-check global_derive.derive_global."
    )


def _redraw_target(
    target: SampledTarget,
    grid: GlobalDerived,
    cfg: DatasetConfig,
    spec,
    rng: random.Random,
) -> Tuple[Optional[SampledTarget], int]:
    """Re-draw a DYNAMIC object into the valid envelope. Returns
    (new_target|None, attempts).

    Sizes shrink within [floor, min(original, user's max)] (the grid guarantees
    original >= floor); the center is drawn within the size-dependent envelope
    (domain minus the (pml+15) gap on every face, fully buried). The x position
    is drawn as an OFFSET from the domain center. None if no attempt succeeded.
    """
    clearance = _clearance(grid, cfg)
    _assert_size_invariant(target, grid, spec)
    tol = _EDGE_TOL_CELLS * grid.dx_m
    half_x = grid.domain_x_m / 2.0
    attempts = 0
    for _ in range(MAX_TARGET_ATTEMPTS):
        attempts += 1
        if spec.kind == "cylinder":
            # radius is a half-extent: floor 5*Δx = 10 cells across the diameter
            r_floor = max(5.0 * grid.dx_m, spec.radius_min_m)
            r_hi = min(target.radius_m, spec.radius_max_m)
            radius = rng.uniform(r_floor, r_hi) if r_hi > r_floor else r_floor
            hx = hy = radius
            size_fields = {"radius_m": round(radius, 6)}
        else:  # box — sides are full extents: floors 10*Δx = 10 cells across the side
            w_floor = max(10.0 * grid.dx_m, spec.width_min_m)
            h_floor = max(10.0 * grid.dx_m, spec.height_min_m)
            w_hi = min(target.width_m, spec.width_max_m)
            h_hi = min(target.height_m, spec.height_max_m)
            w = rng.uniform(w_floor, w_hi) if w_hi > w_floor else w_floor
            h = rng.uniform(h_floor, h_hi) if h_hi > h_floor else h_floor
            hx, hy = w / 2.0, h / 2.0
            size_fields = {"width_m": round(w, 6), "height_m": round(h, 6)}

        off_max = half_x - clearance - hx - tol   # symmetric offset envelope
        d_lo, d_hi = hy + tol, grid.ground_y_m - clearance - hy - tol
        if off_max < 0.0 or d_lo > d_hi:
            continue  # this size is too big for the domain — try a smaller one
        candidate = SampledTarget(
            kind=target.kind,
            name=target.name,
            material=target.material,
            x_offset_m=round(rng.uniform(-off_max, off_max), 6),
            depth_m=round(rng.uniform(d_lo, d_hi), 6),
            **size_fields,
        )
        if not placement_failures(candidate, grid, cfg):
            return candidate, attempts
    return None, attempts


@dataclass
class PlacementResult:
    surviving: List[SampledSample] = field(default_factory=list)
    dropped: List[dict] = field(default_factory=list)   # {sample_id, reason}
    n_redrawn: int = 0        # count of redrawn TARGETS (not samples)
    n_unchanged: int = 0      # count of samples left fully untouched


def _pair_targets(sample: SampledSample, specs: List) -> List[tuple]:
    """Positionally pair sample.targets[j] with the j-th canonical range spec.

    Load-bearing: whole-sample drop is the ONLY drop path, so the list always
    matches the canonical spec order. Assert KIND equality per index (not just
    length) — a mismatch would silently mis-apply per-kind math."""
    assert len(sample.targets) == len(specs), (
        f"sample {sample.sample_id}: {len(sample.targets)} drawn target(s) vs "
        f"{len(specs)} range spec(s) — manifest and target_ranges are out of sync "
        "(stale sampled_layers.json?)"
    )
    for j, (t, spec) in enumerate(zip(sample.targets, specs)):
        assert t.kind == spec.kind, (
            f"sample {sample.sample_id} target #{j}: drawn kind '{t.kind}' != "
            f"spec kind '{spec.kind}' — canonical ordering violated"
        )
    return list(zip(sample.targets, specs))


def validate_and_place(
    samples: List[SampledSample],
    grid: GlobalDerived,
    cfg: DatasetConfig,
    target_ranges: Optional[ExtractedTargetRanges],
    seed: Optional[int] = 1234,
) -> PlacementResult:
    """Validate each sample's dynamic objects against the FIXED grid; redraw or
    drop the whole sample. Static objects are skipped (gate-validated once)."""
    result = PlacementResult()
    rng = random.Random(seed)
    specs = iter_ranges(target_ranges) if target_ranges is not None else []

    for s in samples:
        if not s.targets:
            result.surviving.append(s)
            result.n_unchanged += 1
            continue

        pairs = _pair_targets(s, specs)
        new_targets = list(s.targets)
        redrawn_here = 0
        drop_reason: Optional[str] = None

        for j, (t, spec) in enumerate(pairs):
            if spec.is_static:
                continue  # identical in every sample; validated at the global gate
            if not placement_failures(t, grid, cfg):
                continue

            # Failed placement. If even the floor size cannot fit, drop now.
            x_ok, y_ok = _feasible_at_floor(grid, cfg, spec)
            if not (x_ok and y_ok):
                axes = ", ".join(a for a, ok in (("x", x_ok), ("y/depth", y_ok)) if not ok)
                drop_reason = (
                    f"object #{j} '{t.name}' ({t.kind}): impossible to fit even at "
                    f"its floor size on {axes} (domain {grid.domain_x_m:.3f}x"
                    f"{grid.domain_y_m:.3f} m, ground_y={grid.ground_y_m:.3f} m) — "
                    "dropped without redraw"
                )
                break

            new_target, _attempts = _redraw_target(t, grid, cfg, spec, rng)
            if new_target is None:
                drop_reason = (
                    f"object #{j} '{t.name}' ({t.kind}): could not place within "
                    f"{MAX_TARGET_ATTEMPTS} redraws (last failures: "
                    f"{'; '.join(placement_failures(t, grid, cfg))})"
                )
                break
            new_targets[j] = new_target
            redrawn_here += 1

        if drop_reason is not None:
            # ANY dynamic object failing drops the WHOLE sample: surviving files
            # must always contain the full object set the user specified.
            result.dropped.append({"sample_id": s.sample_id, "reason": drop_reason})
            continue

        if redrawn_here:
            s.targets = new_targets
            result.n_redrawn += redrawn_here
        else:
            result.n_unchanged += 1
        result.surviving.append(s)

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
    target_ranges: Optional[ExtractedTargetRanges],
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

    result = validate_and_place(samples, grid, cfg, target_ranges, seed=seed)

    payload["num_samples"] = len(result.surviving)
    payload["samples"] = [s.model_dump() for s in result.surviving]
    payload["dropped_targets"] = result.dropped
    with open(samples_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    dropped_path = _resolve(output_dir, dropped_filename)
    with open(dropped_path, "w", encoding="utf-8") as f:
        json.dump({"dropped_targets": result.dropped}, f, indent=2)

    return result
