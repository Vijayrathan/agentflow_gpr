"""Centralised validation for dataset sampling.

Three categories:
1. Constants (MODEL_CONSTRAINTS, VALID_SALINITY_CLASSES)
2. Range validation (pre-sampling: validate ranges, check feasibility, clamp to model)
3. Concrete value validation (post-sampling: validate sampled values)
4. Completeness checks (pre-resolution: check required fields present)
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from backend.schema import (
    ExtractedLayerParams,
    ExtractedLayers,
    ExtractedAntennaWaveform,
    ExtractedModelConfig,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_CONSTRAINTS = {
    "peplinski": {
        "theta_v_max": 0.30,
        "sand_pct": (15, 50),
        "silt_pct": (35, 65),
        "clay_pct": (5, 20),
    },
    "dobson": {"theta_v_max": 0.50},
    "mironov": {"theta_v_max": 0.45},
    # crim has no restrictions
}

VALID_SALINITY_CLASSES = {"fresh", "slightly_saline", "brackish", "saline"}


# ---------------------------------------------------------------------------
# Concrete value validation (post-sampling)
# ---------------------------------------------------------------------------

def validate_sampled_layer(
    sand: float,
    silt: float,
    clay: float,
    theta_v: float,
    bd: Optional[float],
    pd: Optional[float],
    model: str,
) -> Optional[str]:
    """Return an error string if the sampled concrete values are invalid,
    or None if everything is OK.

    Enforces texture sum, density ordering, porosity vs theta_v,
    and model-specific bounds.
    """
    # 1. Texture sum (should be guaranteed by _sample_texture, but double-check)
    p_sum = sand + silt + clay
    if abs(p_sum - 100.0) > 0.01:
        return f"sand+silt+clay={p_sum:.2f}, must equal 100"

    # 2. Density cross-checks
    if bd is not None and pd is not None:
        if bd >= pd:
            return (
                f"bulk_density ({bd:.3f}) must be < particle_density ({pd:.3f})"
            )
        porosity = 1.0 - (bd / pd)
        if not (0.0 < porosity < 1.0):
            return f"derived porosity ({porosity:.3f}) must be in (0, 1)"
        if theta_v > porosity:
            return (
                f"theta_v ({theta_v:.3f}) exceeds porosity ({porosity:.3f}); "
                "soil cannot hold more water than its pore space"
            )

    # 3. Model-specific constraints
    constraints = MODEL_CONSTRAINTS.get(model.lower(), {})

    tv_max = constraints.get("theta_v_max")
    if tv_max is not None and theta_v > tv_max:
        return f"{model}: theta_v ({theta_v:.3f}) exceeds max ({tv_max})"

    sand_range = constraints.get("sand_pct")
    if sand_range and not (sand_range[0] <= sand <= sand_range[1]):
        return f"{model}: sand_pct ({sand:.1f}) outside {sand_range[0]}-{sand_range[1]}%"

    silt_range = constraints.get("silt_pct")
    if silt_range and not (silt_range[0] <= silt <= silt_range[1]):
        return f"{model}: silt_pct ({silt:.1f}) outside {silt_range[0]}-{silt_range[1]}%"

    clay_range = constraints.get("clay_pct")
    if clay_range and not (clay_range[0] <= clay <= clay_range[1]):
        return f"{model}: clay_pct ({clay:.1f}) outside {clay_range[0]}-{clay_range[1]}%"

    return None


# ---------------------------------------------------------------------------
# Range clamping & feasibility (used during sampling)
# ---------------------------------------------------------------------------

def clamp_texture_to_model(
    sand_lo: float,
    sand_hi: float,
    silt_lo: float,
    silt_hi: float,
    clay_lo: float,
    clay_hi: float,
    model: str,
) -> Tuple[float, float, float, float, float, float]:
    """Clamp texture ranges to model-specific validity bounds.

    Returns (sand_lo, sand_hi, silt_lo, silt_hi, clay_lo, clay_hi).
    """
    constraints = MODEL_CONSTRAINTS.get(model.lower(), {})
    if "sand_pct" in constraints:
        m_lo, m_hi = constraints["sand_pct"]
        sand_lo, sand_hi = max(sand_lo, m_lo), min(sand_hi, m_hi)
    if "silt_pct" in constraints:
        m_lo, m_hi = constraints["silt_pct"]
        silt_lo, silt_hi = max(silt_lo, m_lo), min(silt_hi, m_hi)
    if "clay_pct" in constraints:
        m_lo, m_hi = constraints["clay_pct"]
        clay_lo, clay_hi = max(clay_lo, m_lo), min(clay_hi, m_hi)
    return sand_lo, sand_hi, silt_lo, silt_hi, clay_lo, clay_hi


def validate_texture_feasibility(
    sand_lo: float,
    sand_hi: float,
    silt_lo: float,
    silt_hi: float,
    clay_lo: float,
    clay_hi: float,
    model: str,
) -> None:
    """Raise ValueError if texture ranges cannot produce sand+silt+clay=100."""
    if sand_lo > sand_hi or silt_lo > silt_hi or clay_lo > clay_hi:
        raise ValueError(
            f"Texture ranges are infeasible for model '{model}': "
            f"sand [{sand_lo}-{sand_hi}], silt [{silt_lo}-{silt_hi}], "
            f"clay [{clay_lo}-{clay_hi}]"
        )
    lower_sum = sand_lo + silt_lo + clay_lo
    upper_sum = sand_hi + silt_hi + clay_hi
    if lower_sum > 100 + 1e-6:
        raise ValueError(
            f"Texture lower bounds sum to {lower_sum:.1f} > 100; "
            "impossible to satisfy sand+silt+clay=100"
        )
    if upper_sum < 100 - 1e-6:
        raise ValueError(
            f"Texture upper bounds sum to {upper_sum:.1f} < 100; "
            "impossible to reach sand+silt+clay=100"
        )


def clamp_theta_v_to_model(
    tv_lo: float,
    tv_hi: float,
    model: str,
) -> Tuple[float, float]:
    """Clamp theta_v range to model-specific max.

    Returns (tv_lo, tv_hi).
    Raises ValueError if no overlap exists.
    """
    constraints = MODEL_CONSTRAINTS.get(model.lower(), {})
    tv_max = constraints.get("theta_v_max")
    if tv_max is not None:
        tv_hi = min(tv_hi, tv_max)
        if tv_lo > tv_hi:
            raise ValueError(
                f"theta_v range [{tv_lo}, {tv_hi}] has no overlap "
                f"with {model} max ({tv_max})"
            )
    return tv_lo, tv_hi


# ---------------------------------------------------------------------------
# Extracted layer range validation (pre-resolution)
# ---------------------------------------------------------------------------

def validate_extracted_layer_params(layer: ExtractedLayerParams) -> List[str]:
    """Validate extracted layer parameter ranges.

    Returns a list of error strings (empty if valid).
    Does NOT do density point-range normalization — that is a
    transformation step handled by the resolver.
    """
    errors: List[str] = []

    # Thickness
    t_min = layer.thickness_m_min
    t_max = layer.thickness_m_max
    if t_min is None or t_max is None:
        errors.append("thickness_m_min and thickness_m_max are required")
    else:
        if t_min <= 0:
            errors.append("thickness_m_min must be > 0")
        if t_max < t_min:
            errors.append("thickness_m_max must be >= thickness_m_min")

    # Sand/silt/clay ranges
    for param, lo, hi in [
        ("sand_pct", layer.sand_pct_min, layer.sand_pct_max),
        ("silt_pct", layer.silt_pct_min, layer.silt_pct_max),
        ("clay_pct", layer.clay_pct_min, layer.clay_pct_max),
    ]:
        if lo is None or hi is None:
            errors.append(f"{param}_min and {param}_max are required")
        else:
            if not (0 <= lo <= 100):
                errors.append(f"{param}_min must be in [0, 100]")
            if not (0 <= hi <= 100):
                errors.append(f"{param}_max must be in [0, 100]")
            if hi < lo:
                errors.append(f"{param}_max must be >= {param}_min")

    # Theta_v ranges
    tv_min = layer.theta_v_min
    tv_max = layer.theta_v_max
    if tv_min is None or tv_max is None:
        errors.append("theta_v_min and theta_v_max are required")
    else:
        if not (0 <= tv_min <= 1):
            errors.append("theta_v_min must be in [0, 1]")
        if not (0 <= tv_max <= 1):
            errors.append("theta_v_max must be in [0, 1]")
        if tv_max < tv_min:
            errors.append("theta_v_max must be >= theta_v_min")

    # Optional density ranges — normalize single-sided to point range for validation
    bd_min = layer.bulk_density_gcm3_min
    bd_max = layer.bulk_density_gcm3_max
    if bd_min is not None and bd_max is None:
        bd_max = bd_min
    elif bd_max is not None and bd_min is None:
        bd_min = bd_max
    if bd_min is not None:
        if bd_min <= 0:
            errors.append("bulk_density_gcm3_min must be > 0")
        if bd_max < bd_min:
            errors.append("bulk_density_gcm3_max must be >= bulk_density_gcm3_min")

    pd_min = layer.particle_density_gcm3_min
    pd_max = layer.particle_density_gcm3_max
    if pd_min is not None and pd_max is None:
        pd_max = pd_min
    elif pd_max is not None and pd_min is None:
        pd_min = pd_max
    if pd_min is not None:
        if pd_min <= 0:
            errors.append("particle_density_gcm3_min must be > 0")
        if pd_max < pd_min:
            errors.append("particle_density_gcm3_max must be >= particle_density_gcm3_min")

    # Salinity classes
    sal_classes = layer.salinity_classes
    if sal_classes is not None:
        invalid = [c for c in sal_classes if c not in VALID_SALINITY_CLASSES]
        if invalid:
            errors.append(
                f"Invalid salinity_classes: {invalid}. "
                f"Valid values: {sorted(VALID_SALINITY_CLASSES)}"
            )

    return errors


# ---------------------------------------------------------------------------
# Completeness checks (pre-resolution)
# ---------------------------------------------------------------------------

def check_layer_completeness(layer: ExtractedLayerParams, index: int) -> List[str]:
    """Check if a raw extracted layer has enough range data to be resolved."""
    problems = []

    if layer.thickness_m_min is None or layer.thickness_m_max is None:
        problems.append("thickness_m_min and thickness_m_max")

    has_sand = layer.sand_pct_min is not None and layer.sand_pct_max is not None
    has_silt = layer.silt_pct_min is not None and layer.silt_pct_max is not None
    has_clay = layer.clay_pct_min is not None and layer.clay_pct_max is not None
    if not (has_sand and has_silt and has_clay):
        problems.append("sand_pct_min/max, silt_pct_min/max, clay_pct_min/max (all required)")

    has_theta = layer.theta_v_min is not None and layer.theta_v_max is not None
    if not has_theta:
        problems.append("theta_v_min and theta_v_max")

    return [f"- Layer {index}: {', '.join(problems)}"] if problems else []


def layer_has_any_range(layer: ExtractedLayerParams) -> bool:
    """Return True if at least one ranged parameter has min < max (genuine variability)."""
    pairs = [
        (layer.thickness_m_min, layer.thickness_m_max),
        (layer.sand_pct_min, layer.sand_pct_max),
        (layer.silt_pct_min, layer.silt_pct_max),
        (layer.clay_pct_min, layer.clay_pct_max),
        (layer.theta_v_min, layer.theta_v_max),
        (layer.bulk_density_gcm3_min, layer.bulk_density_gcm3_max),
        (layer.particle_density_gcm3_min, layer.particle_density_gcm3_max),
    ]
    return any(lo is not None and hi is not None and hi > lo for lo, hi in pairs)


def check_layers_stage_complete(layers: ExtractedLayers) -> List[str]:
    """Check whether the layers stage has all required information."""
    missing: List[str] = []
    if not layers.layers:
        missing.append("- At least one soil layer is required")
    else:
        for i, layer in enumerate(layers.layers, 1):
            missing.extend(check_layer_completeness(layer, i))
    return missing


def check_antenna_waveform_stage_complete(
    antenna_wf: ExtractedAntennaWaveform,
) -> List[str]:
    """Check whether the antenna/waveform stage has all required information."""
    # Lazy import to avoid circular dependency with resolvers
    from resolvers import resolve_antenna_waveform

    missing: List[str] = []
    antenna, waveform = resolve_antenna_waveform(antenna_wf)
    if antenna is None:
        missing.append("- antenna.tx_rx_offset_m (TX-RX antenna offset in meters)")
    if waveform is None:
        missing.append("- waveform.center_freq_hz (waveform center frequency in Hz)")
    return missing


def check_model_stage_complete(model: ExtractedModelConfig) -> List[str]:
    """Check whether the model/domain stage has all required information."""
    # Lazy import to avoid circular dependency with resolvers
    from resolvers import resolve_model_config

    missing: List[str] = []

    if model.num_samples is None or model.num_samples < 1:
        missing.append("- num_samples (number of .in files to generate, e.g. '100 samples')")

    model_dict = resolve_model_config(model)
    model_field_labels = {
        "model": "model (dielectric model: 'crim', 'peplinski', 'dobson', or 'mironov')",
        "title": "title (simulation title)",
        "source_height_m": "source_height_m (antenna height above ground in meters)",
        "domain_xy_m": "domain_xy_m (domain_x and domain_y in meters)",
        "cells_per_wavelength": "cells_per_wavelength (mesh resolution)",
        "max_cell_m": "max_cell_m (maximum cell size in meters)",
        "temperature_c": "temperature_c (temperature in Celsius)",
        "enforce_validity": "enforce_validity (true/false)",
    }
    for field, label in model_field_labels.items():
        if model_dict.get(field) is None:
            missing.append(f"- {label}")
    return missing
