from __future__ import annotations

from typing import List
from .soil_structure_schema import *
from .soil_lookup_table import *


# -----------------------------
# 4) RESOLVER / MAPPER
# -----------------------------

def resolve_to_full(sim: UserSimulationSimple) -> SimulationFull:
    # --- Antenna + waveform ---
    center_freq = sim.waveform.center_freq_hz_override or ANTENNA_PRESET_TO_FREQ_HZ[sim.antenna.preset]
    txrx_offset = sim.antenna.tx_rx_offset_m_override or ANTENNA_PRESET_TO_TXRX_OFFSET_M[sim.antenna.preset]
    waveform_ampl = sim.waveform.amplitude or 1.0

    antenna_full = AntennaFull(
        kind=str(sim.antenna.preset),  # or map to your internal kind strings
        axis=sim.antenna.axis,
        tx_rx_offset_m=txrx_offset,
    )
    waveform_full = WaveformFull(
        kind=sim.waveform.kind,
        amplitude=waveform_ampl,
        center_freq_hz=center_freq,
        name=sim.waveform.name,
    )

    # --- Model domain + meshing ---
    cells_per_wavelength, max_cell_cap = QUALITY_TO_MESH[sim.model.quality]
    if sim.model.cells_per_wavelength_override is not None:
        cells_per_wavelength = sim.model.cells_per_wavelength_override
    if sim.model.max_cell_m_override is not None:
        max_cell_m = sim.model.max_cell_m_override
    else:
        # Cap by quality and also by wavelength rule-of-thumb
        # wavelength ~ v / f ; v in soil is ~ c / sqrt(eps_r); without eps_r we just apply cap.
        max_cell_m = max_cell_cap

    # Domains: default build from length/depth with margins
    domain_x = sim.model.domain_x_override or (sim.model.survey_length_m + 2.0)  # 1m margin each side
    domain_y = sim.model.domain_y_override or (sim.model.max_depth_m + 1.0)      # 1m below depth

    model_full = ModelFull(
        model=sim.model.model,
        title=sim.model.title,
        source_height_m=sim.model.antenna_height_m,
        domain_x=domain_x,
        domain_y=domain_y,
        cells_per_wavelength=cells_per_wavelength,
        max_cell_m=max_cell_m,
        temperature_c=sim.model.temperature_c,
        enforce_validity=sim.model.enforce_validity,
    )

    # --- Layers ---
    layers_full: List[LayerFull] = []
    for layer in sim.layers:
        # texture fractions
        if layer.sand_pct_override is not None:
            sand, silt, clay = float(layer.sand_pct_override), float(layer.silt_pct_override), float(layer.clay_pct_override)
        else:
            sand, silt, clay = TEXTURE_DEFAULTS[layer.texture_class]

        # water content
        if layer.theta_v_override is not None:
            theta_v = float(layer.theta_v_override)
        else:
            theta_v = THETA_V_BY_TEXTURE_AND_STATE[layer.texture_class][layer.moisture_state]

        # organic
        organic_fraction = ORGANIC_FRACTION_BY_LEVEL[layer.organic_level]

        # particle density
        particle_density = (
            float(layer.particle_density_override_gcm3)
            if layer.particle_density_override_gcm3 is not None
            else (2.30 if layer.organic_level == "high_peaty" else PARTICLE_DENSITY_DEFAULT)
        )

        # bulk density
        if layer.bulk_density_override_gcm3 is not None:
            bulk_density = float(layer.bulk_density_override_gcm3)
        else:
            bucket = _texture_bucket(layer.texture_class, layer.organic_level)
            bulk_density = BULK_DENSITY_PRIOR[bucket][layer.compaction_level]

        # salinity + sigma
        salinity_class = SALINITY_CLASS_MAP[layer.salinity_environment]
        if layer.porewater_sigma_override_Sm is not None:
            porewater_sigma = float(layer.porewater_sigma_override_Sm)
        else:
            porewater_sigma = POREWATER_SIGMA_PRIOR[layer.salinity_environment]

        layers_full.append(
            LayerFull(
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
                name=layer.name,
            )
        )

    return SimulationFull(
        layers=layers_full,
        waveform=waveform_full,
        antenna=antenna_full,
        model=model_full,
    )
