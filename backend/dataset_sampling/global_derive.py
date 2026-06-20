"""
STAGE 7 — global derive.

Runs ONCE over the whole sampling space, after the per-sample Peplinski eps derive
(STAGE 6) and after all collect stages. Turns the aggregated eps_r corners + the
waveform / antenna / layers / advanced params into the ONE global grid, domain,
depth and time window shared by every sample, so all N input files sit on the same
Yee grid and are directly comparable for ML.

Order is strict: f_peak -> Wang band (Peplinski gate) -> lambda -> dx -> domain ->
depth -> dt -> time window. eps corners come from STAGE 6:
  eps_r_max = max wettest-bin eps  -> smallest lambda_min -> finest dx
  eps_r_min = min driest-bin eps   -> largest  lambda_max -> biggest domain
Free space (eps=1) is folded into eps_min here (air sets the largest lambda).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from backend.schema import (
    DatasetConfig,
    ExtractedWaveform,
    ExtractedAntenna,
    ExtractedLayers,
    ExtractedAdvancedParams,
    GlobalDerived,
)
from backend.validation_tools_new import (
    C0,
    WANG_FLOW_OVER_FP,
    WANG_FHIGH_OVER_FP,
    WANG_FCENTRE_OVER_FP,
    PEPLINSKI_FREQ_HZ,
)

PEPLINSKI_FMIN_HZ, PEPLINSKI_FMAX_HZ = PEPLINSKI_FREQ_HZ


def _smallest_feature_m(adv: Optional[ExtractedAdvancedParams]) -> Optional[float]:
    """Smallest target diameter across cylinders/spheres, used to tighten dx so
    the smallest feature resolves to >=10 cells. None when there are no targets."""
    if adv is None:
        return None
    feats = []
    for c in (adv.cylinders or []):
        if c.radius:
            feats.append(2 * c.radius)
    for s in (adv.spheres or []):
        if s.radius:
            feats.append(2 * s.radius)
    return min(feats) if feats else None


def _target_footprint_x_m(adv: Optional[ExtractedAdvancedParams]) -> Optional[float]:
    """Hook for real geometry x-extents; None when no targets contribute."""
    return None


def derive_global(
    cfg: DatasetConfig,
    wf: ExtractedWaveform,
    ant: ExtractedAntenna,
    layers: ExtractedLayers,
    adv: Optional[ExtractedAdvancedParams],
    eps_r_max_over_samples: float,
    eps_r_min_over_samples: float,
) -> GlobalDerived:
    """Size the single global grid/domain/depth/time window from the aggregated
    eps_r corners. eps_r_max/min come from the STAGE 6 derive across all samples."""
    # 7a. peak frequency (what #waveform actually takes).
    if cfg.center_freq_is_peak:
        f_peak = wf.waveform_center_freq_hz
    else:                       # input was Wang's band-centre -> back out the peak
        f_peak = wf.waveform_center_freq_hz / WANG_FCENTRE_OVER_FP

    # 7b. Wang band edges -> Peplinski validity gate. Already enforced upstream by
    #     the post-antenna validation gate; re-asserted here defensively.
    f_min = WANG_FLOW_OVER_FP * f_peak
    f_max = WANG_FHIGH_OVER_FP * f_peak
    bandwidth = f_max - f_min
    gate_ok = (f_min >= PEPLINSKI_FMIN_HZ) and (f_max <= PEPLINSKI_FMAX_HZ)
    if not gate_ok:
        raise ValueError(
            f"Peplinski gate FAIL: band [{f_min/1e6:.1f}, {f_max/1e6:.1f}] MHz "
            f"outside [{PEPLINSKI_FMIN_HZ/1e6:.0f}, {PEPLINSKI_FMAX_HZ/1e6:.0f}] MHz "
            f"(peak={f_peak/1e6:.1f} MHz)"
        )

    # 7c. highest SIGNIFICANT frequency for the resolution check (NOT f_max).
    f_high = cfg.high_freq_factor * wf.waveform_center_freq_hz

    # 7d. global eps corners (include free space = 1 for the air region above).
    eps_max = eps_r_max_over_samples
    eps_min = min(eps_r_min_over_samples, 1.0)   # air sets the largest lambda

    # 7e. global Δx from lambda_min (finest grid). Tighten for the smallest target.
    lambda_min = C0 / (f_high * (eps_max ** 0.5))
    dx = lambda_min / cfg.cells_per_wavelength
    smallest_feat = _smallest_feature_m(adv)
    if smallest_feat is not None:
        dx = min(dx, smallest_feat / 10.0)       # >=10 cells across smallest feature

    # 7f. lambda_max + surface dimension (1.5*lambda_max).
    lambda_max = C0 / (f_min * (eps_min ** 0.5))
    surface_xy = 1.5 * lambda_max

    # 7g. GLOBAL depth: deepest possible stack OR range-resolution floor.
    max_stack = sum(L.thickness_m_max for L in layers.layers)
    range_res = C0 / (2.0 * bandwidth * (eps_max ** 0.5))   # slowest medium
    depth_z = max(max_stack, range_res)

    # 7h. antenna height: collected if given, else derived >= lambda_max/2.
    source_height = (
        ant.source_height_m if ant.source_height_m is not None else (lambda_max / 2.0)
    )

    # 7i. domain. y is the vertical (air gap + soil + buffers + PML).
    pad = (cfg.pml_cells + cfg.buffer_cells) * dx
    domain_y = pad + source_height + depth_z + pad
    target_footprint = _target_footprint_x_m(adv)
    domain_x = max(surface_xy, (target_footprint or 0.0) + ant.tx_rx_offset_m + 2 * pad)

    # 7j. CFL time step (gprMax sets at the limit; uniform Δ).
    n_dim = 2.0 if cfg.dimensionality == "2D" else 3.0
    dt = dx / (C0 * (n_dim ** 0.5))

    # 7k. time window: two-way travel to the deepest reflector in the SLOWEST
    #     medium + a pulse-length margin.
    v_min = C0 / (eps_max ** 0.5)
    pulse_margin = 2.0 / f_peak
    time_window = 2.0 * (source_height + depth_z) / v_min + pulse_margin

    return GlobalDerived(
        f_peak_hz=f_peak, f_min_hz=f_min, f_max_hz=f_max, bandwidth_hz=bandwidth,
        f_high_hz=f_high, eps_r_max_global=eps_max, eps_r_min_global=eps_min,
        dx_m=dx, lambda_min_m=lambda_min, lambda_max_m=lambda_max,
        surface_xy_m=surface_xy, source_height_m=source_height, depth_z_m=depth_z,
        domain_x_m=domain_x, domain_y_m=domain_y, dt_s=dt, time_window_s=time_window,
        peplinski_gate_ok=gate_ok,
    )


def write_global(
    grid: GlobalDerived,
    output_dir: str,
    filename: str = "global_derive.json",
) -> str:
    """Persist the global derive to a JSON manifest in the dataset directory."""
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parent.parent.parent / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(grid.model_dump(), f, indent=2)
    return str(path)


def derive_and_write_global(
    cfg: DatasetConfig,
    wf: ExtractedWaveform,
    ant: ExtractedAntenna,
    layers: ExtractedLayers,
    adv: Optional[ExtractedAdvancedParams],
    eps_r_max_over_samples: float,
    eps_r_min_over_samples: float,
    output_dir: str,
    filename: str = "global_derive.json",
):
    """Derive the global grid/domain/depth/time window and persist it.

    Returns (global_derived, json_path).
    """
    grid = derive_global(
        cfg, wf, ant, layers, adv,
        eps_r_max_over_samples, eps_r_min_over_samples,
    )
    path = write_global(grid, output_dir, filename=filename)
    return grid, path
