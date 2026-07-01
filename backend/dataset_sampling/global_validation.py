"""
STAGE 6 — global validation (TIER 3).

Runs once, right after the global derive, on the single shared grid. It applies
the TIER 3 validators from validation_tools_new in CASCADE-GATED phases ordered
cheapest-and-most-fundamental first, so one upstream failure does not spray a
dozen downstream errors that all stem from it:

  Phase 1  grid fundamentals / domain fit   (resolution, stability, integer-cell
           alignment, PML-vs-domain) — if this fails the cell counts feeding the
           later checks are meaningless, so STOP here.
  Phase 2  placement & stratigraphy         (static Tx/Rx clearance, source-height
           vs domain, layer stack, time window, rx-array step) — STOP on error.
  Phase 3  feasibility (warnings, last)      (memory, CFL iteration count).

Per-sample TARGET placement is NOT validated here — it is grid-dependent but
per-sample, so it lives in STAGE 7 (target_placement.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from backend.schema import (
    DatasetConfig,
    ExtractedWaveform,
    ExtractedAntenna,
    ExtractedLayers,
    ExtractedAdvancedParams,
    GlobalDerived,
)
from backend.validation_tools_new import (
    PML_GAP_CELLS,
    validate_global_grid,
    validate_debye_tau_vs_dt,
    validate_domain_alignment,
    validate_pml_vs_domain,
    validate_antenna_placement,
    validate_layer_thickness_and_stack,
    validate_time_window,
    validate_rxarray_step,
    validate_memory,
    validate_cfl_and_iterations,
)


@dataclass
class GlobalValidationReport:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _add(report: GlobalValidationReport, name: str, result) -> None:
    e, w = result
    report.errors += [f"[{name}] {m}" for m in e]
    report.warnings += [f"[{name}] {m}" for m in w]


def _check_source_height_vs_domain(
    grid: GlobalDerived, clearance: float
) -> tuple:
    """Source height >= lambda_max/2 AND the antenna still clears the TOP PML gap.

    The top-gap check uses `clearance` ((pml+15)·Δx), the same object/source gap
    the derive sizes the top with and that validate_antenna_placement enforces —
    NOT the (pml+buffer) `pad`. These two can conflict in a thin domain (a tall
    source height can push the Tx into the top PML+gap), so check BOTH explicitly
    for a clear diagnosis rather than relying only on the generic placement message.
    """
    e: List[str] = []
    min_h = 0.5 * grid.lambda_max_m
    if grid.source_height_m < min_h - 1e-12:
        e.append(
            f"source_height {grid.source_height_m:.4f} m < lambda_max/2 = {min_h:.4f} m"
        )
    if grid.tx_y_m + clearance > grid.domain_y_m + 1e-9:
        e.append(
            f"antenna top (tx_y {grid.tx_y_m:.4f} + clearance {clearance:.4f}) exceeds "
            f"domain_y {grid.domain_y_m:.4f} m — thin domain; raise domain or lower source height"
        )
    return e, []


def validate_global(
    grid: GlobalDerived,
    cfg: DatasetConfig,
    wf: ExtractedWaveform,
    ant: ExtractedAntenna,
    layers: ExtractedLayers,
    adv=None,  # ExtractedAdvancedParams | None (unused for now; kept for parity)
) -> GlobalValidationReport:
    """Validate the single global grid in cascade-gated phases."""
    report = GlobalValidationReport()

    # 2D geometry: uniform cell; the thin (out-of-plane) axis is one cell.
    # Only `clearance` (the (pml+15) object/source gap) is needed here — the
    # (pml+buffer) `pad` has no meaning for the validators' margins.
    dx = dy = dz = grid.dx_m
    dom_x, dom_y, dom_z = grid.domain_x_m, grid.domain_y_m, dx
    clearance = (cfg.pml_cells + PML_GAP_CELLS) * dx

    # ── Phase 1 — grid fundamentals / domain fit (gate hard) ──────────────────
    _add(report, "global_grid", validate_global_grid(
        max_cell_m=dx,
        center_freq_hz=wf.waveform_center_freq_hz,
        waveform_kind=wf.waveform_kind or "ricker",
        eps_r_max=grid.eps_r_max_global,
        cells_per_wavelength=cfg.cells_per_wavelength,
    ))
    _add(report, "debye_tau_vs_dt", validate_debye_tau_vs_dt(dx, dy, dz))
    _add(report, "domain_alignment", validate_domain_alignment(dom_x, dom_y, dom_z, dx, dy, dz))
    _add(report, "pml_vs_domain", validate_pml_vs_domain(dom_x, dom_y, dom_z, dx, dy, dz, pml_cells=cfg.pml_cells))
    if report.errors:
        report.warnings.append(
            "Stopped after domain-fit phase — fix these before placement/feasibility "
            "checks (their cell counts depend on a valid grid)."
        )
        return report

    # ── Phase 2 — placement & stratigraphy (gate) ─────────────────────────────
    _add(report, "antenna_placement", validate_antenna_placement(
        tx_x_m=grid.tx_x_m, rx_x_m=grid.rx_x_m, tx_vertical_m=grid.tx_y_m,
        domain_x_m=dom_x, domain_vertical_m=dom_y, max_cell_m=dx,
        ground_vertical_m=grid.ground_y_m, source_height_m=grid.source_height_m,
        lambda_max_air_m=grid.lambda_max_m, pml_cells=cfg.pml_cells,
    ))
    _add(report, "source_height_vs_domain", _check_source_height_vs_domain(grid, clearance))
    _add(report, "layer_thickness_and_stack", validate_layer_thickness_and_stack(
        layer_names=[L.name or f"layer_{i+1}" for i, L in enumerate(layers.layers)],
        thicknesses_m=[L.thickness_m_max for L in layers.layers],
        max_cell_m=dx, global_depth_m=grid.depth_z_m,
    ))
    _add(report, "time_window", validate_time_window(
        grid.time_window_s, grid.depth_z_m, grid.eps_r_max_global,
    ))
    if ant.rx_array is not None:
        rx = ant.rx_array
        _add(report, "rxarray_step", validate_rxarray_step(rx.dx, rx.dy, rx.dz, dx, dy, dz))
    if report.errors:
        report.warnings.append(
            "Stopped after placement/stratigraphy phase — fix these before the "
            "feasibility checks."
        )
        return report

    # ── Phase 3 — feasibility (warnings, last) ────────────────────────────────
    _add(report, "memory", validate_memory(dom_x, dom_y, dom_z, dx, dy, dz))
    _add(report, "cfl_and_iterations", validate_cfl_and_iterations(dx, dy, dz, grid.time_window_s))

    return report
