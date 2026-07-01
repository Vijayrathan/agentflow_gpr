"""
Layer sampling for the gprMax Peplinski dataset pipeline.

Runs immediately AFTER the layer-extraction stage: draws `num_samples` concrete
parameter sets over the per-layer ranges collected in ExtractedLayers. Sand,
clay, thickness and both densities are drawn uniformly; silt is the derived
texture-closure label (100 - sand - clay). theta_v is NOT drawn — its (min, max)
envelope is passed straight through, because #soil_peplinski consumes a moisture
BAND, not a scalar.

Each draw is validated with validate_sampled_layer (TIER 2); infeasible draws
(silt < 0, or theta_v_max above the sample's own porosity) are rejected and
re-drawn. The N samples are written to a JSON manifest in the dataset directory
for the downstream derive/emit stages.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

from backend.schema import (
    ExtractedLayers,
    ExtractedLayerParams,
    SampledLayer,
    SampledSample,
    SampledTarget,
    CylinderTargetRange,
)
from backend.validation_tools_new import validate_sampled_layer

# Max attempts when re-drawing a target's geometry against the (later) global
# grid in the per-sample placement pass (Stage 7). Defined here next to the
# draw it governs; imported by the placement module.
MAX_TARGET_ATTEMPTS = 20


def _round(x: float, ndigits: int = 6) -> float:
    return round(x, ndigits)


def _draw_target(
    target_range: CylinderTargetRange,
    rng: random.Random,
) -> SampledTarget:
    """Draw one concrete buried target from its range (cylinder only).

    Grid-independent: only the user's ranges constrain the draw here (no domain
    exists yet). Position/depth are re-validated and, if needed, re-drawn against
    the derived domain downstream (Stage 7).
    """
    # Dispatch on class name (not isinstance) so the check is robust to the
    # codebase's dual import paths (`schema` vs `backend.schema` resolve to
    # distinct module objects). Box/sphere range types would be stubbed here.
    if type(target_range).__name__ != "CylinderTargetRange":
        raise NotImplementedError(
            f"Only cylinder buried targets are supported; got "
            f"{type(target_range).__name__} (box/sphere are future work)."
        )
    return SampledTarget(
        kind="cylinder",
        name=target_range.name,
        material=target_range.material,
        x_center_m=_round(rng.uniform(target_range.x_center_min_m, target_range.x_center_max_m)),
        depth_m=_round(rng.uniform(target_range.depth_min_m, target_range.depth_max_m)),
        radius_m=_round(rng.uniform(target_range.radius_min_m, target_range.radius_max_m)),
    )


def _normalise(msg: str) -> str:
    """Collapse the varying numbers out of a message so per-draw warnings that
    differ only by their drawn value group into a single counted entry."""
    return re.sub(r"[-+]?\d*\.?\d+", "#", msg)


def _sample_one_layer(
    layer: ExtractedLayerParams,
    rng: random.Random,
    enforce_validity: bool,
    max_retries: int = 200,
) -> Tuple[SampledLayer, List[str]]:
    """Draw one feasible concrete layer from a range spec.

    theta_v is fixed to the layer's (min, max) envelope (not drawn), so the band
    is validated once up front. sand/clay/thickness/densities are drawn uniformly
    and re-drawn on rejection (silt < 0, or theta_v_max exceeding the sample's own
    porosity computed from the drawn densities).

    Returns the layer together with any non-blocking warnings for the accepted
    draw (e.g. texture outside the Peplinski calibration range), so they can be
    surfaced rather than silently dropped.
    """
    tv_min, tv_max = layer.theta_v_min, layer.theta_v_max
    if tv_min >= tv_max:
        name = f"'{layer.name}'" if layer.name else "(unnamed)"
        raise ValueError(
            f"Layer {name}: theta_v_min ({tv_min}) >= theta_v_max ({tv_max}); "
            "#soil_peplinski needs a real moisture band (min < max)."
        )

    for _ in range(max_retries):
        thickness = rng.uniform(layer.thickness_m_min, layer.thickness_m_max)
        sand = rng.uniform(layer.sand_pct_min, layer.sand_pct_max)
        clay = rng.uniform(layer.clay_pct_min, layer.clay_pct_max)
        silt = 100.0 - sand - clay
        if silt < 0.0:
            continue  # texture closure infeasible for this draw — redraw

        bulk = rng.uniform(layer.bulk_density_gcm3_min, layer.bulk_density_gcm3_max)
        particle = rng.uniform(
            layer.particle_density_gcm3_min, layer.particle_density_gcm3_max
        )

        errors, warnings = validate_sampled_layer(
            sand=sand, silt=silt, clay=clay,
            theta_v_min=tv_min, theta_v_max=tv_max,
            bulk_density=bulk, particle_density=particle,
            enforce_validity=enforce_validity,
        )
        if errors:
            continue  # e.g. theta_v_max > drawn porosity — redraw densities/texture

        sampled = SampledLayer(
            name=layer.name,
            thickness_m=_round(thickness),
            sand_pct=_round(sand),
            clay_pct=_round(clay),
            silt_pct=_round(silt),
            theta_v_min=tv_min,
            theta_v_max=tv_max,
            bulk_density_gcm3=_round(bulk),
            particle_density_gcm3=_round(particle),
        )
        return sampled, warnings

    name = f"'{layer.name}'" if layer.name else "(unnamed)"
    raise ValueError(
        f"Could not draw a feasible sample for layer {name} after {max_retries} "
        "retries — the ranges may be too tight (check that the drawn densities "
        "can support theta_v_max as pore space, and that sand+clay can be <= 100)."
    )


def sample_layers(
    extracted: ExtractedLayers,
    num_samples: int,
    seed: Optional[int] = None,
    enforce_validity: bool = True,
    target_range: Optional[CylinderTargetRange] = None,
) -> Tuple[List[SampledSample], List[str]]:
    """Draw `num_samples` concrete parameter sets over the extracted layer ranges.

    When `target_range` is given, a buried target is drawn per sample alongside
    the soil (grid-independent draw; placement against the domain is validated
    in Stage 7). Returns (samples, warnings). Non-blocking warnings from the
    accepted draws (e.g. texture outside the Peplinski calibration range) are
    grouped across all draws and counted, so they surface here at draw time.
    """
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")

    rng = random.Random(seed)
    samples: List[SampledSample] = []
    warn_counts: Counter = Counter()
    n_layer_records = 0
    for i in range(1, num_samples + 1):
        layers: List[SampledLayer] = []
        for layer in extracted.layers:
            sampled, warnings = _sample_one_layer(layer, rng, enforce_validity)
            layers.append(sampled)
            n_layer_records += 1
            for m in warnings:
                warn_counts[_normalise(m)] += 1
        target = _draw_target(target_range, rng) if target_range is not None else None
        samples.append(SampledSample(sample_id=i, layers=layers, target=target))

    warnings_summary = [
        f"[{n}/{n_layer_records} sample-layers] {msg}"
        for msg, n in warn_counts.items()
    ]
    return samples, warnings_summary


def write_samples(
    samples: List[SampledSample],
    output_dir: str,
    warnings: Optional[List[str]] = None,
    filename: str = "sampled_layers.json",
) -> str:
    """Write the drawn samples to a JSON manifest in the dataset directory.

    Relative output_dir paths are resolved against the project root so the
    manifest lands in the same place regardless of the current working directory.
    """
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parent.parent.parent / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / filename
    payload = {
        "num_samples": len(samples),
        "warnings": warnings or [],
        "samples": [s.model_dump() for s in samples],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return str(path)


def read_samples(
    output_dir: str,
    filename: str = "sampled_layers.json",
) -> List[SampledSample]:
    """Load the drawn samples written by write_samples."""
    path = Path(output_dir)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent.parent / path
    path = path / filename
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [SampledSample.model_validate(s) for s in data["samples"]]


def sample_and_write(
    extracted: ExtractedLayers,
    num_samples: int,
    output_dir: str,
    seed: Optional[int] = None,
    enforce_validity: bool = True,
    filename: str = "sampled_layers.json",
    target_range: Optional[CylinderTargetRange] = None,
) -> Tuple[List[SampledSample], str, List[str]]:
    """Sample the layer ranges (and optional buried target) and persist the draws.

    Returns (samples, json_path, warnings).
    """
    samples, warnings = sample_layers(
        extracted, num_samples, seed=seed, enforce_validity=enforce_validity,
        target_range=target_range,
    )
    path = write_samples(samples, output_dir, warnings=warnings, filename=filename)
    return samples, path, warnings
