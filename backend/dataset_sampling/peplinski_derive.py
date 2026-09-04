"""
Per-sample Peplinski derive (STAGE 6).

After the antenna stage, every parameter needed to compute soil permittivity is
known. We derive the in-band relative permittivity for each drawn sample using
gprMax's OWN Peplinski routine (gprMax.materials.PeplinskiSoil), so the sizing
eps_r matches — by construction — the eps gprMax will build at model-build time
from #soil_peplinski. We do NOT reimplement the mixing model.

sigma is recorded alongside eps as a LABEL only (it is what gprMax itself writes
onto the edge-bin materials, `Material.se`). It plays no part in sizing: only the
real, in-band eps_r enters the wavelength / grid budget, and the global aggregate
carries eps corners only.

Procedure per sampled layer:
  1. Build PeplinskiSoil(name, sand_frac, clay_frac, rho_b, rho_s,
     (theta_v_min, theta_v_max)) — sand/clay as FRACTIONS, moisture as the BAND.
  2. Hand it a throwaway grid stub. calculate_debye_properties only reads
     len(G.materials) and appends to it; it never touches dt/dx. The G argument
     does NOT imply a finalized grid must exist first.
  3. calculate_debye_properties(nbins, G, name) populates G.materials with nbins
     Debye materials spanning the moisture band (dry -> wet).
  4. Evaluate calculate_er(f).real on the edge bins. Do NOT read m.er directly —
     that is the infinite-frequency value and understates in-band eps, which would
     make the grid too coarse. calculate_er folds the Debye relaxation and the
     conductivity term back in to give the true in-band eps.
       * driest bin (first) -> smallest eps -> largest lambda_max -> domain size
       * wettest bin (last) -> largest eps  -> smallest lambda_min -> global dx
       (gprMax shifts materials to bin midpoints, so the wettest bin sits half a
        bin above theta_v_max.)

Aggregate the wettest across all sample-layers into eps_r_max, the driest into
eps_r_min. Free space (eps=1) is folded in later at the global-derive stage.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Tuple

# Put the inner gprMax package root on the path so we reuse its Peplinski routine
# instead of reimplementing the mixing model.
_GPRMAX_ROOT = Path(__file__).resolve().parent.parent.parent / "gprMax"
if str(_GPRMAX_ROOT) not in sys.path:
    sys.path.insert(0, str(_GPRMAX_ROOT))

from gprMax.materials import PeplinskiSoil  # noqa: E402

from backend.schema import (
    DatasetConfig,
    ExtractedWaveform,
    SampledSample,
    DerivedLayer,
    DerivedSample,
    GlobalEpsAggregate,
)
from backend.validation_tools_new import peak_frequency
from dataset_sampling import target_shapes


class _GridStub:
    """Minimal stand-in for a gprMax Grid.

    calculate_debye_properties only reads len(G.materials) and appends to it, so a
    throwaway object with an empty materials list is all it needs — no finalized
    grid (dt/dx) is required to derive permittivity.
    """

    def __init__(self):
        self.materials: list = []


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
    soil = PeplinskiSoil(
        name or "soil",
        sand_pct / 100.0,    # gprMax expects fractions
        clay_pct / 100.0,
        bulk_density,
        particle_density,
        (theta_v_min, theta_v_max),   # full moisture band, not a scalar
    )
    grid = _GridStub()
    soil.calculate_debye_properties(nbins, grid, name or "soil")

    mats = grid.materials
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
    freq = peak_frequency(
        waveform.waveform_center_freq_hz, dataset_config.center_freq_is_peak
    )

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
            eps_dry, eps_wet, sig_dry, sig_wet = derive_layer_properties(
                layer.name,
                layer.sand_pct,
                layer.clay_pct,
                layer.bulk_density_gcm3,
                layer.particle_density_gcm3,
                layer.theta_v_min,
                layer.theta_v_max,
                nbins,
                freq,
            )
            dlayers.append(
                DerivedLayer(
                    name=layer.name,
                    eps_r_dry=eps_dry,
                    eps_r_wet=eps_wet,
                    sigma_dry=sig_dry,
                    sigma_wet=sig_wet,
                )
            )
            eps_max = max(eps_max, eps_wet)
            eps_min = min(eps_min, eps_dry)
        for t in s.targets:
            have_target = True
            smallest_feature = min(smallest_feature, target_shapes.smallest_feature(t))
            largest_extent = max(largest_extent, target_shapes.largest_extent(t))
            deepest_bottom = max(deepest_bottom, target_shapes.bottom_depth(t))
        derived.append(DerivedSample(sample_id=s.sample_id, layers=dlayers))

    static_halfwidth = (
        target_shapes.static_x_halfwidth(target_ranges)
        if target_ranges is not None
        else None
    )
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
