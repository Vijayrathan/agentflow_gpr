"""
STAGE 7 — global derive.

Runs ONCE over the whole sampling space, after the per-sample Peplinski eps derive
(STAGE 6) and after all collect stages. Turns the aggregated eps_r corners + the
waveform / antenna / layers + the buried-target corners into the ONE global grid,
domain, depth and time window shared by every sample, so all N input files sit on
the same Yee grid and are directly comparable for ML. The target corners (from the
STAGE 6 aggregation over all drawn objects) are the ONLY geometry input.

Order is strict: f_peak -> Wang band (Peplinski gate) -> lambda -> dx -> domain ->
depth -> dt -> time window. eps corners come from STAGE 6:
  eps_r_max = max wettest-bin eps  -> smallest lambda_min -> finest dx
  eps_r_min = min driest-bin eps   -> largest  lambda_max -> biggest domain
Free space (eps=1) is folded into eps_min here (air sets the largest lambda).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

from backend.schema import (
    DatasetConfig,
    ExtractedWaveform,
    ExtractedAntenna,
    ExtractedLayers,
    GlobalDerived,
)
from backend.validation_tools_new import (
    C0,
    WANG_FLOW_OVER_FP,
    WANG_FHIGH_OVER_FP,
    WANG_FCENTRE_OVER_FP,
    PEPLINSKI_FREQ_HZ,
    PML_GAP_CELLS,
)

PEPLINSKI_FMIN_HZ, PEPLINSKI_FMAX_HZ = PEPLINSKI_FREQ_HZ


def derive_global(
    cfg: DatasetConfig,
    wf: ExtractedWaveform,
    ant: ExtractedAntenna,
    layers: ExtractedLayers,
    eps_r_max_over_samples: float,
    eps_r_min_over_samples: float,
    smallest_feature_global_m: Optional[float] = None,
    largest_extent_global_m: Optional[float] = None,
    deepest_target_bottom_global_m: Optional[float] = None,
    static_x_halfwidth_global_m: Optional[float] = None,
    enforce_peplinski_gate: bool = True,
) -> GlobalDerived:
    """Size the single global grid/domain/depth/time window from the aggregated
    eps_r corners and the buried-target corners.

    eps_r_max/min come from the STAGE 6 derive across all samples. The target
    corners (size/extent/deepest bottom, plus the static x halfwidth) also come
    from that same pass and are size-only: they tighten Δx, widen domain_x, and
    deepen depth_z so the global grid fits the worst-case per-sample target while
    staying identical for every sample. `clearance = (pml_cells + PML_GAP_CELLS)*Δx`
    is the object/source PML gap and is DISTINCT from the domain
    `pad = (pml_cells + buffer_cells)*Δx`."""
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
    if enforce_peplinski_gate and not gate_ok:
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

    # 7e. global Δx from lambda_min (finest grid). Tighten so the smallest
    #     in-plane target feature (aggregated over ALL drawn objects in STAGE 6)
    #     resolves to >=10 cells.
    lambda_min = C0 / (f_high * (eps_max ** 0.5))
    dx = lambda_min / cfg.cells_per_wavelength
    if smallest_feature_global_m is not None:
        dx = min(dx, smallest_feature_global_m / 10.0)
    # Δx is now FINAL. clearance (object/source PML gap) is distinct from `pad`.
    clearance = (cfg.pml_cells + PML_GAP_CELLS) * dx
    pad = (cfg.pml_cells + cfg.buffer_cells) * dx

    # 7f. lambda_max + surface dimension (1.5*lambda_max).
    lambda_max = C0 / (f_min * (eps_min ** 0.5))
    surface_xy = 1.5 * lambda_max

    # 7g. GLOBAL depth: deepest stack OR range-resolution floor OR deep enough that
    #     the deepest+largest buried target still clears the BOTTOM PML gap.
    #     target bottom = ground_y - deepest_bottom, ground_y = pad + depth_z;
    #     require target_bottom >= clearance. The bottom `pad` already supplies
    #     (pml+buffer), so add only the shortfall to the (pml+15) clearance, i.e.
    #     max(0, clearance - pad)  (0 when buffer already covers the gap).
    max_stack = sum(L.thickness_m_max for L in layers.layers)
    range_res = C0 / (2.0 * bandwidth * (eps_max ** 0.5))   # slowest medium
    depth_z = max(max_stack, range_res)
    if deepest_target_bottom_global_m is not None:
        target_depth_floor = deepest_target_bottom_global_m + max(0.0, clearance - pad)
        depth_z = max(depth_z, target_depth_floor)

    # 7h. antenna height: collected if given, else derived >= lambda_max/2.
    source_height = (
        ant.source_height_m if ant.source_height_m is not None else (lambda_max / 2.0)
    )

    # 7i. domain. y is the vertical: bottom pad | soil | air gap | TOP gap.
    #     The top gap must clear the antenna (a source) from the top PML by the
    #     (pml+15) gap, so it is `clearance`, not the (pml+buffer) `pad` — the
    #     antenna would otherwise sit only buffer(10) cells from the top PML.
    ground_y = pad + depth_z                       # ground surface (pre-snap, fixed)
    domain_y = pad + depth_z + source_height + max(pad, clearance)
    # x margins use `clearance` (not `pad`) so the static Tx/Rx clear the side PMLs
    # by the (pml+15) gap.
    domain_x = max(surface_xy, ant.tx_rx_offset_m + 2 * clearance)
    # widen x so the widest buried target fits horizontally with PML+gap clearance
    if largest_extent_global_m is not None:
        domain_x = max(domain_x, largest_extent_global_m + 2 * clearance)
    # widen x so STATIC (pinned) objects fit at their exact position. Offsets are
    # center-relative, so the halfwidth (max |x_offset| + extent/2) is symmetric:
    # widening covers both left- and right-pinned objects. Dynamic objects don't
    # need this — the per-sample placement pass repositions them via redraw.
    if static_x_halfwidth_global_m is not None:
        domain_x = max(domain_x, 2.0 * (static_x_halfwidth_global_m + clearance))

    # snap domains UP to whole cells so every sample shares one integer Yee grid;
    # snapping extends the TOP only (bottom face at y=0 and ground_y stay fixed).
    domain_x = math.ceil(domain_x / dx) * dx
    domain_y = math.ceil(domain_y / dx) * dx

    # static Tx/Rx (derived ONCE; identical for every sample), after the snap.
    x_mid = domain_x / 2.0
    tx_x = x_mid - ant.tx_rx_offset_m / 2.0
    rx_x = x_mid + ant.tx_rx_offset_m / 2.0
    tx_y = ground_y + source_height
    rx_y = tx_y if (ant.rx_same_height is None or ant.rx_same_height) else tx_y

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
        ground_y_m=ground_y, tx_x_m=tx_x, rx_x_m=rx_x, tx_y_m=tx_y, rx_y_m=rx_y,
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


def read_global(
    output_dir: str,
    filename: str = "global_derive.json",
) -> GlobalDerived:
    """Load the global derive written by write_global."""
    path = Path(output_dir)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent.parent / path
    path = path / filename
    with open(path, encoding="utf-8") as f:
        return GlobalDerived.model_validate(json.load(f))


def derive_and_write_global(
    cfg: DatasetConfig,
    wf: ExtractedWaveform,
    ant: ExtractedAntenna,
    layers: ExtractedLayers,
    eps_r_max_over_samples: float,
    eps_r_min_over_samples: float,
    output_dir: str,
    filename: str = "global_derive.json",
    smallest_feature_global_m: Optional[float] = None,
    largest_extent_global_m: Optional[float] = None,
    deepest_target_bottom_global_m: Optional[float] = None,
    static_x_halfwidth_global_m: Optional[float] = None,
    enforce_peplinski_gate: bool = True,
):
    """Derive the global grid/domain/depth/time window and persist it.

    Returns (global_derived, json_path).
    """
    grid = derive_global(
        cfg, wf, ant, layers,
        eps_r_max_over_samples, eps_r_min_over_samples,
        smallest_feature_global_m=smallest_feature_global_m,
        largest_extent_global_m=largest_extent_global_m,
        deepest_target_bottom_global_m=deepest_target_bottom_global_m,
        static_x_halfwidth_global_m=static_x_halfwidth_global_m,
        enforce_peplinski_gate=enforce_peplinski_gate,
    )
    path = write_global(grid, output_dir, filename=filename)
    return grid, path
