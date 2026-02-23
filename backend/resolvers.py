"""
Resolver functions that convert user-friendly extraction results (from subagents)
into fully-resolved schemas ready for gprMax file generation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from schema import (
    GprSchema,
    LayerSchema,
    WaveformSchema,
    AntennaSchema,
    ExtractedLayerParams,
    ExtractedLayers,
    ExtractedAntennaWaveform,
    ExtractedModelConfig,
    ExtractedOptionalParams,
    AggregatedExtraction,
)
from soil_setup import (
    ANTENNA_PRESET_TO_FREQ_HZ,
    ANTENNA_PRESET_TO_TXRX_OFFSET_M,
    QUALITY_TO_MESH,
)

logger = logging.getLogger(__name__)

VALID_SALINITY_CLASSES = {"fresh", "slightly_saline", "brackish", "saline"}


# ---------------------------------------------------------------------------
# Resolved layer range dataclass
# ---------------------------------------------------------------------------

@dataclass
class ResolvedLayerRange:
    """Validated min/max ranges for a single layer, ready for sampling."""
    name: Optional[str]
    thickness_m_min: float
    thickness_m_max: float
    sand_pct_min: float
    sand_pct_max: float
    silt_pct_min: float
    silt_pct_max: float
    clay_pct_min: float
    clay_pct_max: float
    theta_v_min: float
    theta_v_max: float
    bulk_density_gcm3_min: Optional[float]
    bulk_density_gcm3_max: Optional[float]
    particle_density_gcm3_min: Optional[float]
    particle_density_gcm3_max: Optional[float]
    organic_fraction: float          # single value, defaults to 0.0
    salinity_classes: Optional[List[str]]
    porewater_sigma_Sm: Optional[float]


# ---------------------------------------------------------------------------
# Layer resolver — returns a ResolvedLayerRange (validation only, no sampling)
# ---------------------------------------------------------------------------

def resolve_layer(layer: ExtractedLayerParams) -> ResolvedLayerRange:
    """Validate and promote ExtractedLayerParams into a ResolvedLayerRange."""
    errors = []

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

    # Optional density ranges — if only one side is given, treat as a point range (min == max)
    bd_min = layer.bulk_density_gcm3_min
    bd_max = layer.bulk_density_gcm3_max
    if bd_min is not None and bd_max is None:
        bd_max = bd_min   # single value → point range
    elif bd_max is not None and bd_min is None:
        bd_min = bd_max   # single value → point range
    if bd_min is not None:  # both sides now set (or both None)
        if bd_min <= 0:
            errors.append("bulk_density_gcm3_min must be > 0")
        if bd_max < bd_min:
            errors.append("bulk_density_gcm3_max must be >= bulk_density_gcm3_min")

    pd_min = layer.particle_density_gcm3_min
    pd_max = layer.particle_density_gcm3_max
    if pd_min is not None and pd_max is None:
        pd_max = pd_min   # single value → point range
    elif pd_max is not None and pd_min is None:
        pd_min = pd_max   # single value → point range
    if pd_min is not None:  # both sides now set (or both None)
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

    if errors:
        name_label = f"layer '{layer.name}'" if layer.name else "layer"
        raise ValueError(f"Validation errors for {name_label}: " + "; ".join(errors))

    return ResolvedLayerRange(
        name=layer.name,
        thickness_m_min=t_min,
        thickness_m_max=t_max,
        sand_pct_min=layer.sand_pct_min,
        sand_pct_max=layer.sand_pct_max,
        silt_pct_min=layer.silt_pct_min,
        silt_pct_max=layer.silt_pct_max,
        clay_pct_min=layer.clay_pct_min,
        clay_pct_max=layer.clay_pct_max,
        theta_v_min=tv_min,
        theta_v_max=tv_max,
        bulk_density_gcm3_min=bd_min,
        bulk_density_gcm3_max=bd_max,
        particle_density_gcm3_min=pd_min,
        particle_density_gcm3_max=pd_max,
        organic_fraction=layer.organic_fraction if layer.organic_fraction is not None else 0.0,
        salinity_classes=sal_classes,
        porewater_sigma_Sm=layer.porewater_sigma_Sm,
    )


def resolve_layers(extracted: ExtractedLayers) -> List[ResolvedLayerRange]:
    """Resolve all extracted layers into ResolvedLayerRange objects."""
    return [resolve_layer(layer) for layer in extracted.layers]


# ---------------------------------------------------------------------------
# Antenna + waveform resolver
# ---------------------------------------------------------------------------

def resolve_antenna_waveform(
    extracted: ExtractedAntennaWaveform,
) -> Tuple[Optional[AntennaSchema], Optional[WaveformSchema]]:
    """Resolve extracted antenna/waveform into final schemas."""

    # Frequency: explicit override > preset lookup
    center_freq = extracted.waveform_center_freq_hz
    if center_freq is None and extracted.antenna_preset:
        center_freq = ANTENNA_PRESET_TO_FREQ_HZ.get(extracted.antenna_preset)

    # TX-RX offset: explicit override > preset lookup
    tx_rx_offset = extracted.tx_rx_offset_m
    if tx_rx_offset is None and extracted.antenna_preset:
        tx_rx_offset = ANTENNA_PRESET_TO_TXRX_OFFSET_M.get(extracted.antenna_preset)

    antenna = AntennaSchema(
        kind=extracted.antenna_kind or "hertzian_dipole",
        axis=extracted.antenna_axis or "x",
        tx_rx_offset_m=tx_rx_offset,
        source_start_time=extracted.source_start_time,
        source_end_time=extracted.source_end_time,
    ) if tx_rx_offset is not None else None

    waveform = WaveformSchema(
        kind=extracted.waveform_kind or "ricker",
        amplitude=extracted.waveform_amplitude or 1.0,
        center_freq_hz=center_freq,
        name=extracted.waveform_name or "default_waveform",
    ) if center_freq is not None else None

    return antenna, waveform


# ---------------------------------------------------------------------------
# Model config resolver
# ---------------------------------------------------------------------------

def resolve_model_config(extracted: ExtractedModelConfig) -> dict:
    """Resolve extracted model config into a flat dict of GprSchema fields.
    Returns a dict so callers can see which fields are still None."""

    # Mesh settings: explicit override > quality preset
    cells_per_wavelength = extracted.cells_per_wavelength
    max_cell_m = extracted.max_cell_m
    if extracted.quality and extracted.quality in QUALITY_TO_MESH:
        preset_cpw, preset_max_cell = QUALITY_TO_MESH[extracted.quality]
        if cells_per_wavelength is None:
            cells_per_wavelength = float(preset_cpw)
        if max_cell_m is None:
            max_cell_m = preset_max_cell

    # Domain: explicit override > survey_length/max_depth with margins
    domain_x = extracted.domain_x
    if domain_x is None and extracted.survey_length_m is not None:
        domain_x = extracted.survey_length_m + 2.0

    domain_y = extracted.domain_y
    if domain_y is None and extracted.max_depth_m is not None:
        domain_y = extracted.max_depth_m + 1.0

    return {
        "model": extracted.model,
        "title": extracted.title,
        "source_height_m": extracted.source_height_m,
        "domain_x": domain_x,
        "domain_y": domain_y,
        "cells_per_wavelength": cells_per_wavelength,
        "max_cell_m": max_cell_m,
        "temperature_c": extracted.temperature_c,
        "enforce_validity": extracted.enforce_validity,
    }


# ---------------------------------------------------------------------------
# Merge two AggregatedExtraction objects (incremental state updates)
# ---------------------------------------------------------------------------

def _merge_flat_model(existing, new):
    """Merge two Pydantic models field-by-field; non-None new values override."""
    merged = {}
    for field_name in existing.model_fields:
        new_val = getattr(new, field_name)
        old_val = getattr(existing, field_name)
        merged[field_name] = new_val if new_val is not None else old_val
    return type(existing)(**merged)


def _merge_layer(existing_layer: ExtractedLayerParams, new_layer: ExtractedLayerParams) -> ExtractedLayerParams:
    """Merge two ExtractedLayerParams field-by-field; non-None new values override."""
    merged = {}
    for field_name in existing_layer.model_fields:
        new_val = getattr(new_layer, field_name)
        old_val = getattr(existing_layer, field_name)
        merged[field_name] = new_val if new_val is not None else old_val
    return ExtractedLayerParams(**merged)


def merge_aggregations(
    existing: AggregatedExtraction | None,
    new: AggregatedExtraction,
) -> AggregatedExtraction:
    """Merge a new (partial) extraction into the existing session state.

    - Flat schemas (antenna_waveform, model_params, optional_params): field-by-field,
      non-None new values override existing values.
    - Layers:
      - Same layer count as existing → merge each layer field-by-field (partial updates
        preserved; only fields the user explicitly mentioned are overwritten).
      - Different layer count and > 0 → full replace (user described new layers).
      - 0 layers extracted → keep existing.
    """
    if existing is None:
        return new

    # Layers: merge field-by-field when count matches, replace when count changes
    if new.layers.num_layers > 0 and new.layers.layers:
        if len(new.layers.layers) == len(existing.layers.layers):
            # Same number of layers — merge each layer individually so a partial update
            # (e.g. user only mentions sand range) keeps all other fields intact.
            merged_layer_list = [
                _merge_layer(old_l, new_l)
                for old_l, new_l in zip(existing.layers.layers, new.layers.layers)
            ]
            merged_layers = ExtractedLayers(
                num_layers=len(merged_layer_list),
                layers=merged_layer_list,
            )
        else:
            # Layer count changed — full replace (user described a different set of layers)
            merged_layers = new.layers
    else:
        merged_layers = existing.layers

    merged_antenna_wf = _merge_flat_model(existing.antenna_waveform, new.antenna_waveform)
    merged_model = _merge_flat_model(existing.model_params, new.model_params)
    merged_optional = _merge_flat_model(existing.optional_params, new.optional_params)

    return AggregatedExtraction(
        layers=merged_layers,
        antenna_waveform=merged_antenna_wf,
        model_params=merged_model,
        optional_params=merged_optional,
    )


# ---------------------------------------------------------------------------
# Merge all extractions into GprSchema (or return missing-field info)
# ---------------------------------------------------------------------------

def _check_layer_completeness(layer: ExtractedLayerParams, index: int) -> List[str]:
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


def _layer_has_any_range(layer: ExtractedLayerParams) -> bool:
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


def merge_extractions(
    layers_result: ExtractedLayers,
    antenna_wf_result: ExtractedAntennaWaveform,
    model_result: ExtractedModelConfig,
    optional_result: ExtractedOptionalParams,
) -> Tuple[Optional[GprSchema], List[str]]:
    """Merge 4 subagent outputs into a GprSchema.

    Completeness is checked on the raw extracted data *before* calling the
    resolvers, so incomplete inputs never hit strict Pydantic schemas.

    Returns (schema, missing) where *missing* lists human-readable descriptions
    of any fields that could not be resolved.  If missing is empty the schema
    is complete and ready for validation / file generation.
    """
    missing: List[str] = []

    # --- num_samples ---
    if model_result.num_samples is None or model_result.num_samples < 1:
        missing.append("- num_samples (number of .in files to generate, e.g. '100 samples')")

    # --- Layers (check raw data, don't resolve yet) ---
    num_samples = model_result.num_samples or 0
    if not layers_result.layers:
        missing.append("- At least one soil layer is required")
    else:
        for i, layer in enumerate(layers_result.layers, 1):
            missing.extend(_check_layer_completeness(layer, i))

        # When generating >1 sample, require at least one layer to have genuine ranges.
        # If every parameter in every layer has min == max, all samples would be identical.
        if num_samples > 1:
            any_layer_has_range = any(
                _layer_has_any_range(layer) for layer in layers_result.layers
            )
            if not any_layer_has_range:
                missing.append(
                    "- At least one layer parameter must be a range (min < max) when generating "
                    f"multiple samples (you requested {num_samples}). "
                    "Example: 'thickness 0.2 to 0.5m', 'sand 40 to 70%', 'theta_v 0.05 to 0.25'. "
                    "Without ranges all generated files will be identical."
                )

    # --- Antenna + Waveform (safe — returns None when incomplete) ---
    antenna, waveform = resolve_antenna_waveform(antenna_wf_result)
    if antenna is None:
        missing.append("- antenna.tx_rx_offset_m (or antenna_preset like 'generic_400MHz')")
    if waveform is None:
        missing.append("- waveform.center_freq_hz (or antenna_preset like 'generic_400MHz')")

    # --- Model config (safe — returns dict with None values) ---
    model_dict = resolve_model_config(model_result)
    model_field_labels = {
        "model": "model (dielectric model: 'crim', 'peplinski', 'dobson', or 'mironov')",
        "title": "title (simulation title)",
        "source_height_m": "source_height_m (antenna height above ground in meters)",
        "domain_x": "domain_x (or survey_length_m)",
        "domain_y": "domain_y (or max_depth_m)",
        "cells_per_wavelength": "cells_per_wavelength (or quality: 'fast'/'balanced'/'high_accuracy')",
        "max_cell_m": "max_cell_m (or quality)",
        "temperature_c": "temperature_c (temperature in Celsius)",
        "enforce_validity": "enforce_validity (true/false)",
    }
    for field, label in model_field_labels.items():
        if model_dict.get(field) is None:
            missing.append(f"- {label}")

    # Return early with missing list — resolvers are NOT called for layers
    if missing:
        return None, missing

    # Everything present — safe to resolve layers into ResolvedLayerRange then build a
    # template GprSchema with a placeholder single layer for validation purposes.
    # The real per-sample layers are built by dataset_generator.
    resolved_ranges = resolve_layers(layers_result)

    # Build a representative GprSchema using the midpoint of each range for validation.
    # dataset_generator.py will build per-sample GprSchema objects.
    def _mid(lo, hi):
        return (lo + hi) / 2.0 if lo is not None and hi is not None else None

    template_layers = []
    for r in resolved_ranges:
        sand_mid = _mid(r.sand_pct_min, r.sand_pct_max)
        silt_mid = _mid(r.silt_pct_min, r.silt_pct_max)
        clay_mid = _mid(r.clay_pct_min, r.clay_pct_max)
        # Normalise to 100
        total = sand_mid + silt_mid + clay_mid
        if total > 0:
            sand_mid = round(100 * sand_mid / total, 6)
            silt_mid = round(100 * silt_mid / total, 6)
            clay_mid = round(100 - sand_mid - silt_mid, 6)
        template_layers.append(LayerSchema(
            name=r.name,
            thickness_m=_mid(r.thickness_m_min, r.thickness_m_max),
            sand_pct=sand_mid,
            silt_pct=silt_mid,
            clay_pct=clay_mid,
            theta_v=_mid(r.theta_v_min, r.theta_v_max),
            bulk_density_gcm3=_mid(r.bulk_density_gcm3_min, r.bulk_density_gcm3_max),
            particle_density_gcm3=_mid(r.particle_density_gcm3_min, r.particle_density_gcm3_max),
            organic_fraction=r.organic_fraction,
            salinity_class=r.salinity_classes[0] if r.salinity_classes else None,
            porewater_sigma_Sm=r.porewater_sigma_Sm,
        ))

    gpr = GprSchema(
        model=model_dict["model"],
        title=model_dict["title"],
        source_height_m=model_dict["source_height_m"],
        domain_x=model_dict["domain_x"],
        domain_y=model_dict["domain_y"],
        cells_per_wavelength=model_dict["cells_per_wavelength"],
        max_cell_m=model_dict["max_cell_m"],
        temperature_c=model_dict["temperature_c"],
        enforce_validity=model_dict["enforce_validity"],
        waveform=waveform,
        antenna=antenna,
        layers=template_layers,
        surface_roughness=optional_result.surface_roughness,
        snapshots=optional_result.snapshots,
        rx_array=optional_result.rx_array,
        pml_cells=optional_result.pml_cells,
        num_threads=optional_result.num_threads,
        output_dir=optional_result.output_dir,
    )

    return gpr, []
