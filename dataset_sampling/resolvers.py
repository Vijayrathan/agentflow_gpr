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
    ExtractedAdvancedParams,
)
from validation import (
    validate_extracted_layer_params,
    check_layer_completeness,
    layer_has_any_range,
    check_antenna_waveform_stage_complete,
    check_model_stage_complete,
)

logger = logging.getLogger(__name__)


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
    errors = validate_extracted_layer_params(layer)
    if errors:
        name_label = f"layer '{layer.name}'" if layer.name else "layer"
        raise ValueError(f"Validation errors for {name_label}: " + "; ".join(errors))

    # Density point-range normalization (single value → min == max)
    bd_min = layer.bulk_density_gcm3_min
    bd_max = layer.bulk_density_gcm3_max
    if bd_min is not None and bd_max is None:
        bd_max = bd_min
    elif bd_max is not None and bd_min is None:
        bd_min = bd_max

    pd_min = layer.particle_density_gcm3_min
    pd_max = layer.particle_density_gcm3_max
    if pd_min is not None and pd_max is None:
        pd_max = pd_min
    elif pd_max is not None and pd_min is None:
        pd_min = pd_max

    return ResolvedLayerRange(
        name=layer.name,
        thickness_m_min=layer.thickness_m_min,
        thickness_m_max=layer.thickness_m_max,
        sand_pct_min=layer.sand_pct_min,
        sand_pct_max=layer.sand_pct_max,
        silt_pct_min=layer.silt_pct_min,
        silt_pct_max=layer.silt_pct_max,
        clay_pct_min=layer.clay_pct_min,
        clay_pct_max=layer.clay_pct_max,
        theta_v_min=layer.theta_v_min,
        theta_v_max=layer.theta_v_max,
        bulk_density_gcm3_min=bd_min,
        bulk_density_gcm3_max=bd_max,
        particle_density_gcm3_min=pd_min,
        particle_density_gcm3_max=pd_max,
        organic_fraction=layer.organic_fraction if layer.organic_fraction is not None else 0.0,
        salinity_classes=layer.salinity_classes,
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

    center_freq = extracted.waveform_center_freq_hz
    tx_rx_offset = extracted.tx_rx_offset_m

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

    cells_per_wavelength = extracted.cells_per_wavelength
    max_cell_m = extracted.max_cell_m
    domain_x = extracted.domain_x
    domain_y = extracted.domain_y

    domain_xy_m = (domain_x, domain_y) if domain_x is not None and domain_y is not None else None

    return {
        "model": extracted.model,
        "title": extracted.title,
        "source_height_m": extracted.source_height_m,
        "domain_xy_m": domain_xy_m,
        "top_air_extra_m": extracted.top_air_extra_m,
        "cells_per_wavelength": cells_per_wavelength,
        "max_cell_m": max_cell_m,
        "rx_same_height": extracted.rx_same_height,
        "temperature_c": extracted.temperature_c,
        "enforce_validity": extracted.enforce_validity,
        "salinity_defaults_Sm": tuple(extracted.salinity_defaults_Sm) if extracted.salinity_defaults_Sm else None,
    }



# ---------------------------------------------------------------------------
# Merge all extractions into GprSchema (or return missing-field info)
# ---------------------------------------------------------------------------

def merge_extractions(
    layers_result: ExtractedLayers,
    antenna_wf_result: ExtractedAntennaWaveform,
    model_result: ExtractedModelConfig,
    advanced_result: ExtractedAdvancedParams,
) -> Tuple[Optional[GprSchema], Optional[List[ResolvedLayerRange]], List[str]]:
    """Merge 4 subagent outputs into a GprSchema.

    Completeness is checked on the raw extracted data *before* calling the
    resolvers, so incomplete inputs never hit strict Pydantic schemas.

    Returns (schema, resolved_layer_ranges, missing) where *missing* lists
    human-readable descriptions of any fields that could not be resolved.
    If missing is empty the schema and resolved ranges are ready for generation.
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
            missing.extend(check_layer_completeness(layer, i))

        # When generating >1 sample, require at least one layer to have genuine ranges.
        # If every parameter in every layer has min == max, all samples would be identical.
        if num_samples > 1:
            any_range = any(
                layer_has_any_range(layer) for layer in layers_result.layers
            )
            if not any_range:
                missing.append(
                    "- At least one layer parameter must be a range (min < max) when generating "
                    f"multiple samples (you requested {num_samples}). "
                    "Example: 'thickness 0.2 to 0.5m', 'sand 40 to 70%', 'theta_v 0.05 to 0.25'. "
                    "Without ranges all generated files will be identical."
                )

    # --- Antenna + Waveform ---
    missing.extend(check_antenna_waveform_stage_complete(antenna_wf_result))

    # --- Model config ---
    missing.extend(check_model_stage_complete(model_result))

    # Return early with missing list — resolvers are NOT called for layers
    if missing:
        return None, None, missing

    # Everything present — safe to resolve layers into ResolvedLayerRange then build a
    # template GprSchema with a placeholder single layer for validation purposes.
    # The real per-sample layers are built by dataset_generator.
    resolved_ranges = resolve_layers(layers_result)
    antenna, waveform = resolve_antenna_waveform(antenna_wf_result)
    model_dict = resolve_model_config(model_result)

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

    # Compose objects list from extracted cylinders, boxes, spheres
    objects = None
    obj_parts = []
    if advanced_result.cylinders:
        obj_parts.extend(advanced_result.cylinders)
    if advanced_result.boxes:
        obj_parts.extend(advanced_result.boxes)
    if advanced_result.spheres:
        obj_parts.extend(advanced_result.spheres)
    if obj_parts:
        objects = obj_parts

    gpr = GprSchema(
        model=model_dict["model"],
        title=model_dict["title"],
        source_height_m=model_dict["source_height_m"],
        domain_xy_m=model_dict["domain_xy_m"],
        top_air_extra_m=model_dict["top_air_extra_m"],
        cells_per_wavelength=model_dict["cells_per_wavelength"],
        max_cell_m=model_dict["max_cell_m"],
        rx_same_height=model_dict["rx_same_height"],
        temperature_c=model_dict["temperature_c"],
        enforce_validity=model_dict["enforce_validity"],
        salinity_defaults_Sm=model_dict["salinity_defaults_Sm"],
        waveform=waveform,
        antenna=antenna,
        layers=template_layers,
        objects=objects,
        surface_roughness=advanced_result.surface_roughness,
        snapshots=advanced_result.snapshots,
        rx_array=advanced_result.rx_array,
        pml_cells=advanced_result.pml_cells,
        num_threads=advanced_result.num_threads,
        output_dir=advanced_result.output_dir,
    )

    return gpr, resolved_ranges, []
