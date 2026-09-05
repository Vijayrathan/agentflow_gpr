"""
Validation ruleset for the gprMax Peplinski dataset pipeline.

Organised by the LEVEL at which a check can run, defined by what information is
available at that point — not by "pre/post sampling" (sampling now happens
inside extraction).

  TIER 0  SCHEMA          pydantic construction-time invariants (in
                          gpr_dataset_schema.py). NOT repeated here. See the
                          "ALREADY IN SCHEMA" notes to avoid duplicate checks.
  TIER 1  COLLECT         per collect-agent, envelope/feasibility checks that
                          need only that agent's fields. Safe to expose as agent
                          tools.
  TIER 2  PER-SAMPLE      run after each draw inside extraction; needs concrete
                          sampled values.
  TIER 3  GLOBAL DERIVE   run once after sampling + waveform + antenna, using the
                          worst-case eps_r from gprMax's OWN Peplinski routine
                          (calculate_debye_properties -> calculate_er) and the
                          derived global grid.
  TIER 4  EMISSION        run by the .in writer module (elsewhere): names,
                          reserved collisions, essential commands, snapshot window.

Peplinski-only. gprMax computes eps/sigma and the Debye poles itself, so NO
mixing-model maths lives here. Multi-model (CRIM/Dobson/Mironov) constraints and
the temperature/water-Debye path have been removed — they no longer affect the
simulation.

Each validator returns (errors: list[str], warnings: list[str]).
Use _passed()/_format() to render. Wrap TIER 1 ones as LangChain @tool if needed.
"""
from __future__ import annotations
import math
from typing import List, Optional, Sequence, Tuple

# ── Constants ────────────────────────────────────────────────────────────────
C0 = 299_792_458.0

VALID_WAVEFORMS = {
    "gaussian", "gaussiandot", "gaussiandotnorm", "gaussiandotdot",
    "gaussiandotdotnorm", "ricker", "gaussianprime", "gaussiandoubleprime",
    "sine", "contsine",
}

# Peplinski (1995) — the ONLY model gprMax supports natively.
PEPLINSKI_FREQ_HZ = (0.30e9, 1.30e9)          # model validity window
PEPLINSKI_THETA_V_MAX = 0.30                  # calibration moisture ceiling
PEPLINSKI_TEXTURE_PCT = {                     # 1995 Table I calibration soils
    "sand": (15.0, 50.0), "silt": (35.0, 65.0), "clay": (5.0, 20.0),
}

# Wang (2015) Ricker band edges (for the VALIDITY gate). f = peak frequency.
WANG_FLOW_OVER_FP = 0.481623
WANG_FHIGH_OVER_FP = 1.636567
WANG_FCENTRE_OVER_FP = 1.059095

# Highest SIGNIFICANT frequency factor (for the RESOLUTION check, ~-40 dB).
# Single source of truth — never defaults to a no-op (the old 1.0 disabled the
# safety). This is a DIFFERENT quantity from the Wang -6 dB band edges above.
HIGH_FREQ_FACTOR = {
    "ricker": 2.5, "gaussiandotdot": 2.5, "gaussiandotdotnorm": 2.5,
    "gaussianprime": 2.5, "gaussiandoubleprime": 3.0,
    "gaussiandot": 2.0, "gaussiandotnorm": 2.0, "gaussian": 2.0,
    "sine": 1.2, "contsine": 1.2,
}
DEFAULT_HIGH_FREQ_FACTOR = 2.5  # conservative fallback, NOT 1.0

# gprMax Peplinski emits Debye materials all with this fixed relaxation time.
PEPLINSKI_WATER_TAU_S = 9.231e-12

# gprMax reserved material identifiers.
RESERVED_MATERIAL_NAMES = {"pec", "free_space", "grass", "water"}
BUILTIN_MATERIAL_NAMES = {"pec", "free_space"}

PML_GAP_CELLS = 15          # gprMax guidance: sources/targets >=15 cells from PML
RESISTANCE_MAX_OHM = 376.73  # free-space impedance, exclusive upper bound


def _format(errors: List[str], warnings: List[str]) -> str:
    if errors:
        return "VALIDATION FAILED: " + "; ".join(errors)
    if warnings:
        return "VALIDATION PASSED (warnings: " + "; ".join(warnings) + ")"
    return "VALIDATION PASSED"


# ── frequency helpers (shared) ───────────────────────────────────────────────
def peak_frequency(center_freq_hz: float, center_is_peak: bool) -> float:
    """gprMax #waveform takes the PEAK frequency. If the collected center is
    Wang's band-centre instead, convert. (Decision lives in DatasetConfig.)"""
    return center_freq_hz if center_is_peak else center_freq_hz / WANG_FCENTRE_OVER_FP


def ricker_band_hz(center_freq_hz: float, center_is_peak: bool) -> Tuple[float, float]:
    fp = peak_frequency(center_freq_hz, center_is_peak)
    return WANG_FLOW_OVER_FP * fp, WANG_FHIGH_OVER_FP * fp


def high_significant_freq_hz(center_freq_hz: float, waveform_kind: str) -> float:
    return center_freq_hz * HIGH_FREQ_FACTOR.get(waveform_kind.lower(), DEFAULT_HIGH_FREQ_FACTOR)


# ═══════════════════════════════════════════════════════════════════════════
# TIER 1 — COLLECT-STAGE (envelope / feasibility; one agent's fields only)
# ═══════════════════════════════════════════════════════════════════════════




def validate_waveform_and_peplinski_gate(
    kind: str, center_freq_hz: float, center_is_peak: bool,
    amplitude: Optional[float] = None,
) -> Tuple[List[str], List[str]]:
    """Waveform sanity + the CORRECTED Peplinski gate.

    FIX: gate the DERIVED band edges (Wang), not the centre frequency. Checking
    only the centre lets a band like [397,1350] MHz pass while breaching 1300.
    """
    e: List[str] = []
    if kind.lower() not in VALID_WAVEFORMS:
        e.append(f"unsupported waveform '{kind}'")
    if center_freq_hz is None or center_freq_hz <= 0:
        e.append("center_freq_hz must be > 0")
        return e, []
    if amplitude is not None and not math.isfinite(amplitude):
        e.append("amplitude must be finite")

    f_min, f_max = ricker_band_hz(center_freq_hz, center_is_peak)
    lo, hi = PEPLINSKI_FREQ_HZ
    if f_min < lo or f_max > hi:
        fp = peak_frequency(center_freq_hz, center_is_peak)
        e.append(
            f"Peplinski gate FAIL: derived band [{f_min/1e6:.0f},{f_max/1e6:.0f}] MHz "
            f"outside [{lo/1e6:.0f},{hi/1e6:.0f}] MHz (peak={fp/1e6:.0f} MHz)"
        )
    return e, []


def validate_antenna_config(
    kind: str, axis: str,
    resistance: Optional[float] = None,
    source_start_time: Optional[float] = None,
    source_end_time: Optional[float] = None,
) -> Tuple[List[str], List[str]]:
    """Antenna sanity NOT already covered by the schema's _resistance_rules.
    (Schema enforces: resistance required + 0<R<376.73 for transmission_line OR
    voltage_source. Here we add axis, the source-timing PAIR rule, and a softer
    resistance finite/range echo for hertzian cases where schema stays silent.)"""
    e: List[str] = []
    if kind not in {"hertzian_dipole", "voltage_source", "transmission_line"}:
        e.append(f"unsupported antenna kind {kind!r}")
    if axis.lower() not in {"x", "y", "z"}:
        e.append("axis must be x, y or z")
    # transmission_line AND voltage_source both need resistance (schema enforces).
    # No 50-100 ohm "recommended" warning — that heuristic is not grounded.
    if resistance is not None:
        if not math.isfinite(resistance) or resistance <= 0 or resistance >= RESISTANCE_MAX_OHM:
            e.append(f"resistance must satisfy 0 < R < {RESISTANCE_MAX_OHM} ohm")
    # gprMax timing is a pair [start, end]. end-alone -> start defaults to 0 (ok);
    # start-alone is the genuine error.
    if source_start_time is not None and source_end_time is None:
        e.append("source_start_time without source_end_time (gprMax needs the pair)")
    if source_start_time is not None and source_end_time is not None and source_start_time >= source_end_time:
        e.append("source_start_time must be < source_end_time")
    return e, []


# ═══════════════════════════════════════════════════════════════════════════
# TIER 2 — PER-SAMPLE (after each draw, inside extraction)
# ═══════════════════════════════════════════════════════════════════════════

def validate_sampled_layer(
    sand: float, silt: float, clay: float,
    theta_v_min: float, theta_v_max: float,
    bulk_density: float, particle_density: float,
    enforce_validity: bool = True,
) -> Tuple[List[str], List[str]]:
    """Concrete drawn values. theta_v is a BAND (min,max) passed to #soil_peplinski.
    Note: gprMax instantiates materials at bin MIDPOINTS, so the wettest material
    sits half a bin above theta_v_max — bind the porosity cap on the band top with
    a small margin in your sampler if you draw near the limit."""
    e: List[str] = []; w: List[str] = []

    if abs(sand + silt + clay - 100.0) > 0.01:
        e.append(f"sand+silt+clay={sand+silt+clay:.2f}, must be 100")
    if theta_v_min >= theta_v_max:
        e.append("theta_v_min must be < theta_v_max (real moisture band)")  # ALSO in schema; cheap to re-assert

    if bulk_density >= particle_density:
        e.append(f"bulk_density {bulk_density:.3f} >= particle_density {particle_density:.3f}")
    else:
        porosity = 1.0 - bulk_density / particle_density
        if not (0.0 < porosity < 1.0):
            e.append(f"derived porosity {porosity:.3f} not in (0,1)")
        elif theta_v_max > porosity:
            e.append(f"theta_v_max {theta_v_max:.3f} > porosity {porosity:.3f} (water > pore space)")

    if enforce_validity:
        if theta_v_max > PEPLINSKI_THETA_V_MAX:
            e.append(f"theta_v_max {theta_v_max:.3f} > Peplinski max {PEPLINSKI_THETA_V_MAX}")
        for name, val in [("sand", sand), ("silt", silt), ("clay", clay)]:
            c_lo, c_hi = PEPLINSKI_TEXTURE_PCT[name]
            if not (c_lo <= val <= c_hi):
                w.append(f"{name} {val:.1f}% outside Peplinski calibration [{c_lo:.0f},{c_hi:.0f}]%")
    return e, w


# ═══════════════════════════════════════════════════════════════════════════
# TIER 3 — GLOBAL DERIVE (run ONCE; eps_r_max from gprMax's own Peplinski)
# ═══════════════════════════════════════════════════════════════════════════

def validate_global_grid(
    max_cell_m: float, center_freq_hz: float, waveform_kind: str, eps_r_max: float,
    cells_per_wavelength: int = 10,
) -> Tuple[List[str], List[str]]:
    """lambda/10 using the highest SIGNIFICANT frequency and the worst-case
    (highest) eps_r aggregated from the wettest Peplinski bin across all samples."""
    e: List[str] = []
    if min(max_cell_m, center_freq_hz, eps_r_max) <= 0:
        return ["max_cell_m, center_freq_hz, eps_r_max must all be > 0"], []
    f_high = high_significant_freq_hz(center_freq_hz, waveform_kind)
    lambda_min = C0 / (f_high * math.sqrt(eps_r_max))
    max_allowed = lambda_min / cells_per_wavelength
    if max_cell_m > max_allowed + 1e-12:
        e.append(f"max_cell_m {max_cell_m:.5f} > lambda_min/{cells_per_wavelength} = {max_allowed:.5f} "
                 f"(f_high={f_high:.3e} Hz, eps_r_max={eps_r_max:.2f})")
    return e, []


def validate_domain_alignment(
    domain_x_m: float, domain_y_m: float, domain_z_m: float,
    dx: float, dy: float, dz: float,
) -> Tuple[List[str], List[str]]:
    e: List[str] = []
    for label, dim, cell in [("x", domain_x_m, dx), ("y", domain_y_m, dy), ("z", domain_z_m, dz)]:
        if cell <= 0:
            e.append(f"d{label} must be > 0"); continue
        r = dim / cell
        if abs(r - round(r)) > 1e-9:
            e.append(f"domain_{label} {dim} not an integer multiple of d{label} {cell} (ratio {r:.6f})")
    return e, []


def validate_cfl_and_iterations(
    dx: float, dy: float, dz: float, time_window_s: float, max_iterations: int = 50_000,
) -> Tuple[List[str], List[str]]:
    """gprMax sets dt at the CFL limit automatically; this checks the resulting
    iteration count is feasible."""
    if min(dx, dy, dz) <= 0 or time_window_s <= 0:
        return ["dx,dy,dz,time_window must be > 0"], []
    dt = 1.0 / (C0 * math.sqrt(1/dx**2 + 1/dy**2 + 1/dz**2))
    n = math.ceil(time_window_s / dt)
    w = [f"dt={dt:.3e} s, iterations={n:,} for window {time_window_s:.3e} s"]
    if n > max_iterations:
        w.append(f"iteration count {n:,} is high — coarsen grid or shorten window")
    return [], w


def validate_debye_tau_vs_dt(
    dx: float, dy: float, dz: float, tau_s: float = PEPLINSKI_WATER_TAU_S,
) -> Tuple[List[str], List[str]]:
    """gprMax requires Debye tau > dt. Peplinski's tau is fixed at watertau."""
    if min(dx, dy, dz) <= 0:
        return ["dx,dy,dz must be > 0"], []
    dt = 1.0 / (C0 * math.sqrt(1/dx**2 + 1/dy**2 + 1/dz**2))
    if tau_s <= dt:
        return [f"Debye tau {tau_s:.3e} s <= dt {dt:.3e} s (unstable); coarsen grid to raise dt"], []
    return [], []


def validate_pml_vs_domain(
    domain_x_m: float, domain_y_m: float, domain_z_m: float,
    dx: float, dy: float, dz: float, pml_cells: int = 10,
) -> Tuple[List[str], List[str]]:
    if min(dx, dy, dz) <= 0:
        return ["dx,dy,dz must be > 0"], []
    e: List[str] = []
    is_2d = domain_z_m <= dz * 1.5
    axes = [("x", domain_x_m, dx), ("y", domain_y_m, dy)] + ([] if is_2d else [("z", domain_z_m, dz)])
    for label, dim, cell in axes:
        n = round(dim / cell)
        if 2 * pml_cells >= n:
            e.append(f"{label}: 2*pml_cells({2*pml_cells}) >= domain cells({n})")
    return e, []


def validate_memory(
    domain_x_m: float, domain_y_m: float, domain_z_m: float,
    dx: float, dy: float, dz: float,
    available_ram_bytes: int = 32 * 1024**3, bytes_per_cell: int = 146,
) -> Tuple[List[str], List[str]]:
    if min(dx, dy, dz) <= 0:
        return ["dx,dy,dz must be > 0"], []
    nx, ny, nz = round(domain_x_m/dx), round(domain_y_m/dy), round(domain_z_m/dz)
    cells = nx * ny * nz
    est = cells * bytes_per_cell + 50_000_000
    if est > available_ram_bytes:
        return [f"~{est/1024**3:.1f} GiB > RAM {available_ram_bytes/1024**3:.1f} GiB "
                f"(grid {nx}x{ny}x{nz}={cells:,})"], []
    return [], []


def validate_time_window(
    time_window_s: float, depth_m: float, eps_r_max: float,
) -> Tuple[List[str], List[str]]:
    """Window must cover two-way travel to the deepest reflector in the SLOWEST
    medium. NOTE: this is the simulation window, NOT source_end_time."""
    if eps_r_max <= 0:
        return ["eps_r_max must be > 0"], []
    if time_window_s <= 0:
        return ["time_window_s must be > 0"], []
    v_min = C0 / math.sqrt(eps_r_max)
    t2 = 2.0 * depth_m / v_min
    if time_window_s < t2:
        return [f"time_window {time_window_s:.3e} s < two-way travel {t2:.3e} s "
                f"(depth {depth_m} m, eps_r_max {eps_r_max:.2f})"], []
    return [], []


def validate_antenna_placement(
    tx_x_m: float, rx_x_m: float, tx_vertical_m: float,
    domain_x_m: float, domain_vertical_m: float, max_cell_m: float,
    ground_vertical_m: float, source_height_m: float, lambda_max_air_m: float,
    pml_cells: int = 10,
) -> Tuple[List[str], List[str]]:
    """PML clearance on all faces + source height >= lambda_max/2 above ground.
    lambda_max_air_m = c / f_min (Wang lower edge), in air (eps_r=1).

    Axis-neutral: the caller states which physical axis is vertical via
    `*_vertical_m` (this project uses y); the thin axis keeps its own name."""
    e: List[str] = []
    if max_cell_m <= 0:
        return ["max_cell_m must be > 0"], []
    margin = (pml_cells + PML_GAP_CELLS) * max_cell_m
    for label, x in [("Tx", tx_x_m), ("Rx", rx_x_m)]:
        if x < margin or (domain_x_m - x) < margin:
            e.append(f"{label} x={x:.4f} within PML+{PML_GAP_CELLS} margin {margin:.4f} m")
    if tx_vertical_m > domain_vertical_m - margin or tx_vertical_m < margin:
        e.append(f"Tx vertical={tx_vertical_m:.4f} within PML+{PML_GAP_CELLS} margin of a vertical face")
    min_h = 0.5 * lambda_max_air_m
    if source_height_m < min_h - 1e-12:
        e.append(f"source_height {source_height_m:.4f} m < lambda_max/2 = {min_h:.4f} m")
    return e, []


def validate_layer_thickness_and_stack(
    layer_names: Sequence[str], thicknesses_m: Sequence[float],
    max_cell_m: float, global_depth_m: float, min_cells: int = 3,
) -> Tuple[List[str], List[str]]:
    """Each layer spans >= min_cells; total stack must fit the GLOBAL depth box."""
    e: List[str] = []; w: List[str] = []
    if max_cell_m <= 0:
        return ["max_cell_m must be > 0"], []
    for name, t in zip(layer_names, thicknesses_m):
        cells = t / max_cell_m
        if cells < min_cells:
            w.append(f"layer '{name}' is {cells:.1f} cells (< {min_cells})")
    total = sum(thicknesses_m)
    if total > global_depth_m + 1e-9:
        e.append(f"layer stack {total:.4f} m exceeds global depth {global_depth_m:.4f} m")
    return e, w


def validate_target(
    name: str, min_dimension_m: float,
    bbox_min: Tuple[float, float, float], bbox_max: Tuple[float, float, float],
    domain: Tuple[float, float, float], max_cell_m: float,
    pml_cells: int = 10, min_cells_across: int = 10,
) -> Tuple[List[str], List[str]]:
    """Buried target: resolved by >= min_cells_across, inside domain, clear of PML."""
    e: List[str] = []; w: List[str] = []
    if max_cell_m <= 0:
        return ["max_cell_m must be > 0"], []
    if min_dimension_m / max_cell_m < min_cells_across:
        w.append(f"target '{name}' is {min_dimension_m/max_cell_m:.1f} cells across (< {min_cells_across})")
    margin = (pml_cells + PML_GAP_CELLS) * max_cell_m
    is_2d = domain[2] <= max_cell_m * 1.5
    faces = [("x", bbox_min[0], domain[0]-bbox_max[0]), ("y", bbox_min[1], domain[1]-bbox_max[1])]
    if not is_2d:
        faces.append(("z", bbox_min[2], domain[2]-bbox_max[2]))
    for label, lo, hi in faces:
        if lo < 0 or hi < 0:
            e.append(f"target '{name}' outside domain on {label}")
        elif lo < margin or hi < margin:
            w.append(f"target '{name}' within PML+{PML_GAP_CELLS} margin on {label}")
    return e, w


def validate_rxarray_step(
    rx_dx: float, rx_dy: float, rx_dz: float, dx: float, dy: float, dz: float,
) -> Tuple[List[str], List[str]]:
    e: List[str] = []
    for label, step, cell in [("dx", rx_dx, dx), ("dy", rx_dy, dy), ("dz", rx_dz, dz)]:
        if step > 0 and step < cell:
            e.append(f"rx_array {label} {step:.5g} < cell {cell:.5g} (must be >= cell or 0)")
    return e, []


# ═══════════════════════════════════════════════════════════════════════════
# TIER 4 — EMISSION (run by the .in writer module, not the agents)
# ═══════════════════════════════════════════════════════════════════════════

def validate_material_names(names: Sequence[str]) -> Tuple[List[str], List[str]]:
    """No whitespace, unique, and not colliding with gprMax reserved identifiers."""
    e: List[str] = []
    seen: dict[str, int] = {}
    for i, name in enumerate(names):
        if not name or not name.strip():
            e.append(f"material name #{i} is empty")
        elif any(ws in name for ws in (" ", "\t")):
            e.append(f"material name '{name}' contains whitespace (gprMax splits on spaces)")
        if name.lower() in RESERVED_MATERIAL_NAMES:
            e.append(f"'{name}' collides with reserved identifier {sorted(RESERVED_MATERIAL_NAMES)}")
        key = name.lower()
        if key in seen:
            e.append(f"duplicate material name '{name}' (#{seen[key]} and #{i})")
        else:
            seen[key] = i
    return e, []


def validate_essential_commands(
    has_domain: bool, has_dx_dy_dz: bool, has_time_window: bool,
) -> Tuple[List[str], List[str]]:
    """The three mandatory gprMax commands. time_window is its OWN command —
    distinct from any source_end_time."""
    missing = [n for n, ok in [("#domain", has_domain), ("#dx_dy_dz", has_dx_dy_dz),
                               ("#time_window", has_time_window)] if not ok]
    return ([f"missing mandatory commands: {', '.join(missing)}"] if missing else []), []


def validate_snapshot_in_window(
    snapshot_time_s: float, time_window_s: float,
) -> Tuple[List[str], List[str]]:
    if snapshot_time_s <= 0:
        return [], []  # no snapshot configured
    if time_window_s > 0 and snapshot_time_s > time_window_s:
        return [f"snapshot time {snapshot_time_s:.3e} s exceeds window {time_window_s:.3e} s"], []
    return [], []
