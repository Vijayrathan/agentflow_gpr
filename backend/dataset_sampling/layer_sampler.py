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
    ExtractedTargetRanges,
    SampledLayer,
    SampledSample,
)
from backend.validation_tools_new import validate_sampled_layer
from backend.dataset_sampling.target_shapes import draw_target, iter_ranges

# Max attempts when re-drawing a target's geometry against the (later) global
# grid in the per-sample placement pass (Stage 7). Defined here next to the
# draw it governs; imported by the placement module.
MAX_TARGET_ATTEMPTS = 20


def _round(x: float, ndigits: int = 6) -> float:
    return round(x, ndigits)


def _normalise(msg: str) -> str:
    """Collapse the varying numbers out of a message so per-draw warnings that
    differ only by their drawn value group into a single counted entry."""
    return re.sub(r"[-+]?\d*\.?\d+", "#", msg)


def _sample_one_layer(
    layer: ExtractedLayerParams,
    rng: random.Random,
    enforce_validity: bool,
    max_retries: int = 200,
    nbins: int = 50,
    preserve_precision: bool = False,
    rejection_log=None,
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
            if rejection_log is not None:
                rejection_log.append("texture closure: sand + clay > 100")
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
            if rejection_log is not None:
                rejection_log.extend(errors)
            continue  # e.g. theta_v_max > drawn porosity — redraw densities/texture
        actual_wet = tv_max + (tv_max - tv_min) / (2 * (nbins - 1))
        if actual_wet > min(0.30, 1 - bulk / particle):
            if rejection_log is not None:
                rejection_log.append("shifted native wet bin exceeds porosity/calibration limit")
            continue

        quantize = (lambda v: v) if preserve_precision else _round
        sampled = SampledLayer(
            name=layer.name,
            thickness_m=quantize(thickness),
            sand_pct=quantize(sand),
            clay_pct=quantize(clay),
            silt_pct=quantize(silt),
            theta_v_min=tv_min,
            theta_v_max=tv_max,
            bulk_density_gcm3=quantize(bulk),
            particle_density_gcm3=quantize(particle),
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
    target_ranges: Optional[ExtractedTargetRanges] = None,
    dataset_config=None,
) -> Tuple[List[SampledSample], List[str]]:
    """Draw `num_samples` concrete parameter sets over the extracted layer ranges.

    When `target_ranges` holds objects, every object is drawn per sample in the
    canonical order (cylinders, then boxes) — a static (min==max) range draws
    identically into every sample. Grid-independent; placement against the
    domain is validated downstream (static: global gate; dynamic: Stage 7).
    Returns (samples, warnings). Non-blocking warnings from the accepted draws
    (e.g. texture outside the Peplinski calibration range) are grouped across
    all draws and counted, so they surface here at draw time.
    """
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")

    from backend.dataset_sampling.contract import stream_seed, validate_capabilities
    if dataset_config is not None:
        validate_capabilities(dataset_config, target_ranges=target_ranges)
    version = dataset_config.contract_version if dataset_config else 1
    nbins = dataset_config.fractal_nbins if dataset_config else 50
    if version >= 2:
        seed = dataset_config.seed if seed is None else seed
    specs = iter_ranges(target_ranges) if target_ranges is not None else []
    rng = random.Random(seed)
    samples: List[SampledSample] = []
    warn_counts: Counter = Counter()
    n_layer_records = 0
    for i in range(1, num_samples + 1):
        layers: List[SampledLayer] = []
        layer_rejections = []
        for j, layer in enumerate(extracted.layers):
            rejected = []
            layer_rng = random.Random(stream_seed(seed, i, "layer", j)) if version >= 2 else rng
            sampled, warnings = _sample_one_layer(layer, layer_rng, enforce_validity,
                                                   nbins=nbins, preserve_precision=version >= 2, rejection_log=rejected)
            layer_rejections.append({"layer_index": j, "reasons": dict(Counter(rejected)), "attempt_limit": 200})
            layers.append(sampled)
            n_layer_records += 1
            for m in warnings:
                warn_counts[_normalise(m)] += 1
        targets = [draw_target(spec, rng, independent_seed=stream_seed(seed, i, "target", j) if version >= 2 else None)
                   for j, spec in enumerate(specs)]
        provenance = {"seed": seed, "random_policy": "independent-identity-v2" if version >= 2 else "legacy-sequential-v1",
                      "original_targets": [t.model_dump() for t in targets],
                      "soil_draw_rejections": layer_rejections,
                      "moisture_policy": "supplied band passed through unchanged"}
        samples.append(SampledSample(sample_id=i, layers=layers, targets=targets, provenance=provenance))

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
    target_ranges: Optional[ExtractedTargetRanges] = None,
    dataset_config=None,
) -> Tuple[List[SampledSample], str, List[str]]:
    """Sample the layer ranges (and any buried objects) and persist the draws.

    Returns (samples, json_path, warnings).
    """
    samples, warnings = sample_layers(
        extracted, num_samples, seed=seed, enforce_validity=enforce_validity,
        target_ranges=target_ranges, dataset_config=dataset_config,
    )
    path = write_samples(samples, output_dir, warnings=warnings, filename=filename)
    return samples, path, warnings
