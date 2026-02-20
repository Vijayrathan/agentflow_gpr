"""
Resolver functions that convert user-friendly extraction results (from subagents)
into fully-resolved schemas ready for gprMax file generation.

Uses lookup tables from soil_setup to fill defaults when users provide
descriptive terms like texture_class="sandy_loam" instead of raw percentages.
"""
from __future__ import annotations

import logging
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
    TEXTURE_DEFAULTS,
    THETA_V_BY_TEXTURE_AND_STATE,
    PARTICLE_DENSITY_DEFAULT,
    ORGANIC_FRACTION_BY_LEVEL,
    BULK_DENSITY_PRIOR,
    SALINITY_CLASS_MAP,
    POREWATER_SIGMA_PRIOR,
    ANTENNA_PRESET_TO_FREQ_HZ,
    ANTENNA_PRESET_TO_TXRX_OFFSET_M,
    QUALITY_TO_MESH,
    _texture_bucket,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer resolver
# ---------------------------------------------------------------------------

def resolve_layer(layer: ExtractedLayerParams) -> LayerSchema:
    """Resolve a single extracted layer into a fully-specified LayerSchema."""

    # Texture fractions: prefer explicit overrides, fall back to texture_class lookup
    if layer.sand_pct is not None and layer.silt_pct is not None and layer.clay_pct is not None:
        sand, silt, clay = layer.sand_pct, layer.silt_pct, layer.clay_pct
    elif layer.texture_class and layer.texture_class in TEXTURE_DEFAULTS:
        sand, silt, clay = TEXTURE_DEFAULTS[layer.texture_class]
    else:
        sand, silt, clay = None, None, None

    # Volumetric water content
    if layer.theta_v is not None:
        theta_v = layer.theta_v
    elif layer.texture_class and layer.moisture_state:
        table = THETA_V_BY_TEXTURE_AND_STATE.get(layer.texture_class, {})
        theta_v = table.get(layer.moisture_state)
    else:
        theta_v = None

    # Organic fraction
    if layer.organic_fraction is not None:
        organic_fraction = layer.organic_fraction
    elif layer.organic_level and layer.organic_level in ORGANIC_FRACTION_BY_LEVEL:
        organic_fraction = ORGANIC_FRACTION_BY_LEVEL[layer.organic_level]
    else:
        organic_fraction = 0.0

    # Particle density
    if layer.particle_density_gcm3 is not None:
        particle_density = layer.particle_density_gcm3
    else:
        organic_lvl = layer.organic_level or "none"
        particle_density = 2.30 if organic_lvl == "high_peaty" else PARTICLE_DENSITY_DEFAULT

    # Bulk density
    if layer.bulk_density_gcm3 is not None:
        bulk_density = layer.bulk_density_gcm3
    elif layer.texture_class:
        organic_lvl = layer.organic_level or "none"
        compaction = layer.compaction_level or "normal"
        bucket = _texture_bucket(layer.texture_class, organic_lvl)
        bulk_density = BULK_DENSITY_PRIOR.get(bucket, {}).get(compaction, 1.35)
    else:
        bulk_density = None

    # Salinity class
    if layer.salinity_class is not None:
        salinity_class = layer.salinity_class
    elif layer.salinity_environment and layer.salinity_environment in SALINITY_CLASS_MAP:
        salinity_class = SALINITY_CLASS_MAP[layer.salinity_environment]
    else:
        salinity_class = None

    # Porewater conductivity
    if layer.porewater_sigma_Sm is not None:
        porewater_sigma = layer.porewater_sigma_Sm
    elif layer.salinity_environment and layer.salinity_environment in POREWATER_SIGMA_PRIOR:
        porewater_sigma = POREWATER_SIGMA_PRIOR[layer.salinity_environment]
    else:
        porewater_sigma = None

    return LayerSchema(
        name=layer.name,
        thickness_m=layer.thickness_m,
        sand_pct=sand,
        silt_pct=silt,
        clay_pct=clay,
        theta_v=theta_v,
        bulk_density_gcm3=bulk_density,
        particle_density_gcm3=particle_density,
        organic_fraction=organic_fraction,
        salinity_class=salinity_class,
        porewater_sigma_Sm=porewater_sigma,
    )


def resolve_layers(extracted: ExtractedLayers) -> List[LayerSchema]:
    """Resolve all extracted layers."""
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


def merge_aggregations(
    existing: AggregatedExtraction | None,
    new: AggregatedExtraction,
) -> AggregatedExtraction:
    """Merge a new (partial) extraction into the existing session state.

    - Flat schemas (antenna_waveform, model_params, optional_params): field-by-field,
      non-None new values override existing values.
    - Layers: if the new extraction has num_layers > 0 and non-empty layers list,
      replace entirely; otherwise keep existing layers.
    """
    if existing is None:
        return new

    # Layers: replace if new extraction provides them
    if new.layers.num_layers > 0 and new.layers.layers:
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
    """Check if a raw extracted layer has enough data to be resolved."""
    problems = []
    if layer.thickness_m is None:
        problems.append("thickness_m")

    has_pcts = (layer.sand_pct is not None and layer.silt_pct is not None and layer.clay_pct is not None)
    has_texture = (layer.texture_class is not None and layer.texture_class in TEXTURE_DEFAULTS)
    if not has_pcts and not has_texture:
        problems.append("sand_pct/silt_pct/clay_pct (or texture_class)")

    has_theta = layer.theta_v is not None
    has_moisture = (layer.texture_class is not None and layer.moisture_state is not None)
    if not has_theta and not has_moisture:
        problems.append("theta_v (or texture_class + moisture_state)")

    return [f"- Layer {index}: {', '.join(problems)}"] if problems else []


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

    # --- Layers (check raw data, don't resolve yet) ---
    if not layers_result.layers:
        missing.append("- At least one soil layer is required")
    else:
        for i, layer in enumerate(layers_result.layers, 1):
            missing.extend(_check_layer_completeness(layer, i))

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

    # Everything present — safe to resolve layers into strict LayerSchema
    resolved_layers = resolve_layers(layers_result)

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
        layers=resolved_layers,
        surface_roughness=optional_result.surface_roughness,
        snapshots=optional_result.snapshots,
        rx_array=optional_result.rx_array,
        pml_cells=optional_result.pml_cells,
        num_threads=optional_result.num_threads,
        output_dir=optional_result.output_dir,
    )

    return gpr, []
