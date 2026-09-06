"""Native Peplinski derivation and dataset-wide material/geometry bounds.

Derive every instantiated bin with gprMax before a grid exists. Labels use the
resolved peak frequency. Version 2 checks the complex design-band response and
uses a conservative all-frequency phase-index bound including Debye relaxation
and conductivity. Persist coefficient tables/digests for native build parity.
Historical v1 spatial planning continues to consume its scalar peak-epsilon
corners. No alternate mixture model or scalar-moisture path is introduced.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import List, Tuple

# Put the inner gprMax package root on the path so we reuse its Peplinski routine
# instead of reimplementing the mixing model.
_GPRMAX_ROOT = Path(__file__).resolve().parent.parent.parent / "gprMax"
if str(_GPRMAX_ROOT) not in sys.path:
    sys.path.insert(0, str(_GPRMAX_ROOT))

from gprMax.materials import PeplinskiSoil, Material  # noqa: E402
from gprMax.constants import e0, c
import numpy as np

from backend.schema import (
    DatasetConfig,
    ExtractedWaveform,
    SampledSample,
    DerivedLayer,
    DerivedSample,
    GlobalEpsAggregate,
)
from backend.validation_tools_new import peak_frequency
from backend.dataset_sampling import target_shapes


class _GridStub:
    """Minimal stand-in for a gprMax Grid.

    calculate_debye_properties only reads len(G.materials) and appends to it, so a
    throwaway object with an empty materials list is all it needs — no finalized
    grid (dt/dx) is required to derive permittivity.
    """

    def __init__(self):
        self.materials: list = []


def material_coefficients(material):
    return {"er_inf": float(material.er), "sigma_native_Sm": float(material.se),
            "delta_er": [float(v) for v in material.deltaer],
            "tau_s": [float(v) for v in material.tau]}


def native_material_table(name, sand_pct, clay_pct, bulk_density, particle_density,
                          theta_v_min, theta_v_max, nbins):
    values = (sand_pct, clay_pct, bulk_density, particle_density, theta_v_min, theta_v_max)
    if not all(math.isfinite(v) for v in values) or nbins < 2:
        raise ValueError("Native soil requires finite inputs and at least two bins")
    if not (0 <= sand_pct <= 100 and 0 <= clay_pct <= 100 and sand_pct + clay_pct <= 100):
        raise ValueError("Native soil texture must close with nonnegative fractions")
    if not (0 < bulk_density < particle_density and 0 <= theta_v_min < theta_v_max):
        raise ValueError("Native soil requires ordered positive densities and a nonzero moisture band")
    actual_wet = theta_v_max + (theta_v_max - theta_v_min) / (2 * (nbins - 1))
    cap = min(0.30, 1 - bulk_density / particle_density)
    if actual_wet > cap + 1e-14:
        raise ValueError(f"Native wettest bin {actual_wet:.8g} exceeds porosity/calibration cap {cap:.8g}")
    soil = PeplinskiSoil(name or "soil", sand_pct / 100, clay_pct / 100,
                        bulk_density, particle_density, (theta_v_min, theta_v_max))
    grid = _GridStub()
    old_maxpoles = Material.maxpoles
    try:
        soil.calculate_debye_properties(nbins, grid, name or "soil")
    finally:
        Material.maxpoles = old_maxpoles  # previews must not mutate solver global state
    if len(grid.materials) != nbins:
        raise ValueError("Native material count differs from requested bins")
    for m in grid.materials:
        numbers = [m.er, m.se, *m.deltaer, *m.tau]
        if not all(np.isfinite(v) for v in numbers) or m.er <= 0 or m.se < 0 or any(v < 0 for v in m.deltaer) or any(v <= 0 for v in m.tau):
            raise ValueError(f"Nonfinite, nonpassive or invalid native soil material {m.ID}")
    return grid.materials


def derive_layer_properties(
    name: str,
    sand_pct: float,
    clay_pct: float,
    bulk_density: float,
    particle_density: float,
    theta_v_min: float,
    theta_v_max: float,
    nbins: int,
    freq_hz: float,
) -> Tuple[float, float, float, float]:
    """Return (eps_r_dry, eps_r_wet, sigma_dry, sigma_wet) for one sampled layer.

    eps_* is the real part of gprMax's in-band permittivity on the edge bins —
    driest bin (first) and wettest bin (last) — never the stored infinite-frequency
    m.er. sigma_* is the edge bins' effective conductivity (S/m) exactly as gprMax
    stores it on the material (`Material.se`), which is frequency-independent; it
    is a label, not a sizing input.
    """
    mats = native_material_table(name, sand_pct, clay_pct, bulk_density, particle_density,
                                 theta_v_min, theta_v_max, nbins)
    if not mats:
        raise ValueError(f"Peplinski derive produced no materials for layer '{name}'")

    eps_dry = mats[0].calculate_er(freq_hz).real    # driest bin
    eps_wet = mats[-1].calculate_er(freq_hz).real   # wettest bin
    return eps_dry, eps_wet, float(mats[0].se), float(mats[-1].se)


def derive_layer_eps(
    name: str,
    sand_pct: float,
    clay_pct: float,
    bulk_density: float,
    particle_density: float,
    theta_v_min: float,
    theta_v_max: float,
    nbins: int,
    freq_hz: float,
) -> Tuple[float, float]:
    """Return (eps_r_dry, eps_r_wet) — the sizing-relevant half of
    `derive_layer_properties`. Kept as the narrow entry point for the live
    visualization preview."""
    eps_dry, eps_wet, _, _ = derive_layer_properties(
        name, sand_pct, clay_pct, bulk_density, particle_density,
        theta_v_min, theta_v_max, nbins, freq_hz,
    )
    return eps_dry, eps_wet


def derive_samples(
    samples: List[SampledSample],
    dataset_config: DatasetConfig,
    waveform: ExtractedWaveform,
    target_ranges=None,
) -> Tuple[List[DerivedSample], GlobalEpsAggregate]:
    """Derive in-band eps_r edges for every sampled layer and aggregate the
    global eps_r corners across all sample-layers.

    `target_ranges` (ExtractedTargetRanges | None) is used ONLY for the static
    x-footprint corner: static (min==max) objects have an exactly known
    horizontal position, so their |x_offset| + extent/2 halfwidth lets the
    global derive widen domain_x to fit them (dynamic objects reposition via
    redraw instead)."""
    nbins = dataset_config.fractal_nbins
    from backend.dataset_sampling.contract import digest, validate_capabilities
    from backend.dataset_sampling.numerics import excitation
    validate_capabilities(dataset_config, target_ranges=target_ranges, waveform=waveform)
    spectrum = excitation(dataset_config, waveform) if dataset_config.contract_version >= 2 else None
    specs = target_shapes.iter_ranges(target_ranges) if target_ranges else []
    x_halfwidth = z_halfwidth = 0.0
    index_bound = 1.0
    minimum_layer = float("inf")
    min_tau = float("inf")
    freq = peak_frequency(
        waveform.waveform_center_freq_hz, dataset_config.center_freq_is_peak
    )
    design_band = spectrum["design_band_hz"] if spectrum else [freq, freq]

    derived: List[DerivedSample] = []
    eps_max = float("-inf")
    eps_min = float("inf")
    # Buried-target corners, aggregated in the SAME pass (size-only — the target
    # material does NOT feed the eps corners). All position-independent here.
    #
    # GHOST CORNER (conscious, conservative approximation): each quantity is
    # aggregated INDEPENDENTLY across samples, so the worst-case feeding the grid
    # is a synthetic target that may not correspond to any single drawn sample
    # (smallest radius from one sample, deepest bottom from another, etc.). This
    # mirrors the eps corners (eps_r_max from the wettest sample, eps_r_min from
    # the driest) and over-sizes the global grid slightly — it is always at least
    # as safe as the true per-sample worst case, which is exactly what one global
    # grid for all samples requires.
    smallest_feature = float("inf")   # min in-plane feature -> tightens dx
    largest_extent = float("-inf")    # max x extent         -> enlarges domain_x
    deepest_bottom = float("-inf")    # max bottom depth     -> enlarges depth_z
    have_target = False
    for s in samples:
        dlayers: List[DerivedLayer] = []
        for layer in s.layers:
            mats = native_material_table(
                layer.name,
                layer.sand_pct,
                layer.clay_pct,
                layer.bulk_density_gcm3,
                layer.particle_density_gcm3,
                layer.theta_v_min,
                layer.theta_v_max,
                nbins,
            )
            eps_values = [m.calculate_er(freq).real for m in mats]
            eps_dry, eps_wet = eps_values[0], eps_values[-1]
            sig_dry, sig_wet = float(mats[0].se), float(mats[-1].se)
            table = [material_coefficients(m) for m in mats]
            frequencies = np.linspace(*design_band, 129) if spectrum else [freq]
            phase_lambda = float("inf")
            attenuation_length = float("inf")
            for m in mats:
                min_tau = min(min_tau, *m.tau)
                # Triangle inequality for passive Debye terms at ALL f>=design_low:
                # |eps(f)| <= er_inf + sum(delta_er) + sigma/(2*pi*f_low*eps0).
                # Re(sqrt(eps)) <= sqrt(|eps|). This conservative bound (not
                # frequency sampling alone) guarantees the resolution budget.
                bound = m.er + sum(m.deltaer) + m.se / (2 * math.pi * design_band[0] * e0)
                index_bound = max(index_bound, math.sqrt(bound))
                for f in frequencies:
                    response = complex(m.calculate_er(f))
                    if not np.isfinite(response) or response.real <= 0 or response.imag > 1e-12:
                        raise ValueError(f"Invalid/passivity-violating native response at {f} Hz")
                    refractive = np.sqrt(response)
                    phase_lambda = min(phase_lambda, c / (f * refractive.real))
                    alpha = 2 * math.pi * f * abs(refractive.imag) / c
                    if alpha > 0:
                        attenuation_length = min(attenuation_length, 1 / alpha)
            provenance = {"model": "native-Peplinski", "nbins": nbins,
                          "moisture_band": [layer.theta_v_min, layer.theta_v_max],
                          "wettest_bin": layer.theta_v_max + (layer.theta_v_max - layer.theta_v_min) / (2 * (nbins - 1)),
                          "evaluation_frequency_hz": freq, "table_digest": digest(table),
                          "table": table, "design_band_hz": design_band,
                          "sampled_phase_wavelength_min_m": float(phase_lambda),
                          "sampled_attenuation_length_min_m": float(attenuation_length) if math.isfinite(attenuation_length) else None}
            dlayers.append(
                DerivedLayer(
                    name=layer.name,
                    eps_r_dry=eps_dry,
                    eps_r_wet=eps_wet,
                    sigma_dry=sig_dry,
                    sigma_wet=sig_wet,
                    material_provenance=provenance,
                )
            )
            eps_max = max(eps_max, *eps_values)
            eps_min = min(eps_min, *eps_values)
            minimum_layer = min(minimum_layer, layer.thickness_m)
        for j, t in enumerate(s.targets):
            have_target = True
            smallest_feature = min(smallest_feature, target_shapes.smallest_feature(t))
            largest_extent = max(largest_extent, target_shapes.largest_extent(t))
            deepest_bottom = max(deepest_bottom, target_shapes.bottom_depth(t))
            if dataset_config.contract_version >= 2:
                hx, hy, hz = target_shapes.half_extents_3d(t)
                spec = specs[j] if specs else None
                x_extent = max(abs(spec.x_offset_min_m), abs(spec.x_offset_max_m)) if spec else abs(t.x_offset_m)
                x_halfwidth = max(x_halfwidth, x_extent + hx)
                deepest_bottom = max(deepest_bottom, (spec.depth_max_m if spec else t.depth_m) + hy)
                if dataset_config.dimensionality == "3D":
                    z_extent = max(abs(spec.z_offset_min_m), abs(spec.z_offset_max_m)) if spec else abs(t.z_offset_m)
                    z_halfwidth = max(z_halfwidth, z_extent + hz)
        derived.append(DerivedSample(sample_id=s.sample_id, layers=dlayers))

    static_halfwidth = (
        target_shapes.static_x_halfwidth(target_ranges)
        if target_ranges is not None
        else None
    )
    if dataset_config.contract_version >= 2:
        static_halfwidth = x_halfwidth or None
    aggregate = GlobalEpsAggregate(
        eps_r_max=eps_max,
        eps_r_min=eps_min,
        num_samples=len(samples),
        frequency_hz=freq,
        nbins=nbins,
        smallest_feature_global_m=smallest_feature if have_target else None,
        largest_extent_global_m=largest_extent if have_target else None,
        deepest_target_bottom_global_m=deepest_bottom if have_target else None,
        static_x_halfwidth_global_m=static_halfwidth,
        z_halfwidth_global_m=z_halfwidth or None,
        spectral_lambda_min_m=c / (design_band[1] * index_bound) if spectrum else None,
        spectral_index_max=index_bound if spectrum else None,
        min_layer_thickness_m=minimum_layer if math.isfinite(minimum_layer) else None,
        min_relaxation_time_s=min_tau if math.isfinite(min_tau) else None,
    )
    return derived, aggregate


def write_derived(
    derived: List[DerivedSample],
    aggregate: GlobalEpsAggregate,
    output_dir: str,
    filename: str = "derived_layers.json",
) -> str:
    """Write the per-sample derived eps_r and the global aggregate to a manifest."""
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parent.parent.parent / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / filename
    payload = {
        "eps_r_aggregate": aggregate.model_dump(),
        "samples": [d.model_dump() for d in derived],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return str(path)


def read_aggregate(
    output_dir: str,
    filename: str = "derived_layers.json",
) -> GlobalEpsAggregate:
    """Load the global eps_r aggregate written by write_derived."""
    path = Path(output_dir)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent.parent / path
    path = path / filename
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return GlobalEpsAggregate.model_validate(data["eps_r_aggregate"])


def derive_and_write(
    samples: List[SampledSample],
    dataset_config: DatasetConfig,
    waveform: ExtractedWaveform,
    output_dir: str,
    filename: str = "derived_layers.json",
    target_ranges=None,
) -> Tuple[List[DerivedSample], GlobalEpsAggregate, str]:
    """Derive in-band eps_r for all samples and persist the manifest.

    Returns (derived, aggregate, json_path).
    """
    derived, aggregate = derive_samples(
        samples, dataset_config, waveform, target_ranges=target_ranges
    )
    path = write_derived(derived, aggregate, output_dir, filename=filename)
    return derived, aggregate, path
