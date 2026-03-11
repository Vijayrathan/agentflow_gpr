"""Validation tools for gprMax simulation schemas.

Each function validates a specific component and is decorated as a LangChain
tool so it can be passed directly to a DeepAgent.
"""

import math
from typing import Annotated, List, Optional
from langchain_core.tools import tool

# Physical constants (duplicated from physics_modelling.py to keep self-contained)
C0 = 299_792_458.0               # Speed of light in vacuum (m/s)
EPS0 = 8.854187817e-12            # Vacuum permittivity (F/m)
MU0 = 4 * math.pi * 1e-7         # Vacuum permeability (H/m)


VALID_WAVEFORMS = {
    "gaussian",
    "gaussiandot",
    "gaussiandotnorm",
    "gaussiandotdot",
    "gaussiandotdotnorm",
    "ricker",
    "gaussianprime",
    "gaussiandoubleprime",
    "sine",
    "contsine",
}

# Model-specific frequency bands (Hz)
_MODEL_FREQ_BANDS = {
    "peplinski": (0.3e9, 1.3e9),
    "dobson": (1.4e9, 18e9),
    "mironov": (0.6e9, 18e9),
}


def _result(errors: list, warnings: list) -> str:
    """Format validation result with errors and/or warnings."""
    if errors:
        return "VALIDATION FAILED: " + "; ".join(errors)
    if warnings:
        return "VALIDATION PASSED (warnings: " + "; ".join(warnings) + ")"
    return "VALIDATION PASSED"


# ---------------------------------------------------------------------------
# Soil layer validation
# ---------------------------------------------------------------------------

@tool
def validate_layer(
    organic_fraction: Annotated[Optional[float], "Organic fraction (0-1)"] = None,
    porewater_sigma_Sm: Annotated[Optional[float], "Porewater conductivity in S/m"] = None,
) -> str:
    """Validate non-range soil layer parameters.

    Range-based parameters (thickness, texture percentages, theta_v, densities,
    porosity) are validated at sampling time in the dataset generator, not here.
    """
    errors: list[str] = []

    if organic_fraction is not None and organic_fraction < 0:
        errors.append("organic_fraction must be >= 0")

    if porewater_sigma_Sm is not None and porewater_sigma_Sm < 0:
        errors.append("porewater_sigma_Sm must be >= 0")

    return _result(errors, [])


# ---------------------------------------------------------------------------
# Waveform validation
# ---------------------------------------------------------------------------

@tool
def validate_waveform(
    kind: Annotated[str, "Waveform type (e.g. 'ricker', 'gaussian')"],
    center_freq_hz: Annotated[Optional[float], "Centre frequency in Hz (must be > 0)"] = None,
    amplitude: Annotated[Optional[float], "Waveform amplitude (must be finite)"] = None,
    model: Annotated[Optional[str], "Dielectric model name; if provided, warns when freq is outside model band"] = None,
) -> str:
    """Validate waveform parameters."""
    errors: list[str] = []
    warnings: list[str] = []

    if kind not in VALID_WAVEFORMS:
        errors.append(
            f"Unsupported waveform kind '{kind}'. "
            f"Must be one of: {', '.join(sorted(VALID_WAVEFORMS))}"
        )

    if center_freq_hz is not None:
        if center_freq_hz <= 0:
            errors.append("center_freq_hz must be > 0")

    if amplitude is not None:
        if not math.isfinite(amplitude):
            errors.append("amplitude must be finite")

    if model is not None and center_freq_hz is not None and center_freq_hz > 0:
        band = _MODEL_FREQ_BANDS.get(model.lower())
        if band is not None:
            f_lo, f_hi = band
            if not (f_lo <= center_freq_hz <= f_hi):
                warnings.append(
                    f"center_freq_hz ({center_freq_hz:.3e} Hz) is outside the "
                    f"{model} validity band ({f_lo:.1e}–{f_hi:.1e} Hz)"
                )

    return _result(errors, warnings)


# ---------------------------------------------------------------------------
# Dielectric model validation
# ---------------------------------------------------------------------------

@tool
def validate_model(
    model: Annotated[str, "Dielectric model name (peplinski, dobson, mironov, crim)"],
    f0: Annotated[float, "Centre frequency in Hz"],
) -> str:
    """Validate the dielectric model name and that the centre frequency falls
    within the model's validity band.

    Range-based parameters (theta_v, texture percentages) are validated at
    sampling time in the dataset generator, not here.
    """
    errors: list[str] = []
    model_lower = model.lower()
    band = _MODEL_FREQ_BANDS.get(model_lower)
    if band is not None:
        f_lo, f_hi = band
        if not (f_lo <= f0 <= f_hi):
            errors.append(
                f"{model} valid for {f_lo:.1e}–{f_hi:.1e} Hz "
                f"(got {f0:.3e} Hz)"
            )
    elif model_lower != "crim":
        errors.append(f"Unknown model '{model}'")
    return _result(errors, [])


# ---------------------------------------------------------------------------
# Custom material validation
# ---------------------------------------------------------------------------

@tool
def validate_custom_material(
    eps_r: Annotated[float, "Relative permittivity"],
    sigma: Annotated[float, "Electrical conductivity (S/m)"],
    mu_r: Annotated[float, "Relative permeability"],
    sigma_m: Annotated[float, "Magnetic conductivity (Ohm/m)"],
) -> str:
    """Validate custom material electromagnetic properties."""
    errors: list[str] = []
    if eps_r < 1.0:
        errors.append("eps_r must be >= 1.0")
    if sigma < 0.0:
        errors.append("sigma must be >= 0")
    if mu_r < 1.0:
        errors.append("mu_r must be >= 1.0")
    if sigma_m < 0.0:
        errors.append("sigma_m must be >= 0")
    return _result(errors, [])


# ---------------------------------------------------------------------------
# Antenna validation
# ---------------------------------------------------------------------------

@tool
def validate_antenna(
    kind: Annotated[str, "Antenna type ('hertzian_dipole' or 'voltage_source')"],
    axis: Annotated[str, "Polarisation axis ('x', 'y', or 'z')"],
    source_start_time: Annotated[Optional[float], "Source start time (s)"] = None,
    source_end_time: Annotated[Optional[float], "Source end time (s)"] = None,
    resistance: Annotated[Optional[float], "Feed resistance (ohms); required for voltage_source"] = None,
    tx_rx_offset_m: Annotated[Optional[float], "Tx-Rx offset in metres"] = None,
    cell_size_m: Annotated[Optional[float], "Cell size in metres (for offset check)"] = None,
) -> str:
    """Validate antenna configuration."""
    errors: list[str] = []
    warnings: list[str] = []

    if kind.lower() not in {"hertzian_dipole", "voltage_source"}:
        errors.append("Only 'hertzian_dipole' and 'voltage_source' are supported")
    if axis.lower() not in {"x", "y", "z"}:
        errors.append("Axis must be 'x', 'y', or 'z'")
    if source_start_time is not None and source_end_time is not None:
        if source_start_time >= source_end_time:
            errors.append("source_start_time must be < source_end_time")

    # Resistance checks
    if resistance is not None:
        if not math.isfinite(resistance):
            errors.append("resistance must be finite")
        elif resistance < 0:
            errors.append("resistance must be >= 0")
    elif kind.lower() == "voltage_source":
        warnings.append("voltage_source typically requires a resistance parameter")

    # Tx-Rx offset vs cell size
    if tx_rx_offset_m is not None and cell_size_m is not None:
        if cell_size_m > 0 and tx_rx_offset_m < cell_size_m:
            errors.append(
                f"tx_rx_offset_m ({tx_rx_offset_m:.6g}) must be >= cell_size_m ({cell_size_m:.6g})"
            )

    return _result(errors, warnings)


# ---------------------------------------------------------------------------
# Geometry object validation
# ---------------------------------------------------------------------------

@tool
def validate_cylinder(
    name: Annotated[str, "Cylinder name"],
    radius: Annotated[float, "Cylinder radius in metres"],
    material: Annotated[str, "Material identifier"],
    has_custom_material: Annotated[bool, "Whether a custom material is defined"] = False,
    x1: Annotated[Optional[float], "X start coordinate"] = None,
    y1: Annotated[Optional[float], "Y start coordinate"] = None,
    z1: Annotated[Optional[float], "Z start coordinate"] = None,
    x2: Annotated[Optional[float], "X end coordinate"] = None,
    y2: Annotated[Optional[float], "Y end coordinate"] = None,
    z2: Annotated[Optional[float], "Z end coordinate"] = None,
    domain_x_m: Annotated[Optional[float], "Domain X extent for bounds check"] = None,
    domain_y_m: Annotated[Optional[float], "Domain Y extent for bounds check"] = None,
    domain_z_m: Annotated[Optional[float], "Domain Z extent for bounds check"] = None,
) -> str:
    """Validate a cylinder geometry object."""
    errors: list[str] = []
    if radius <= 0:
        errors.append(f"Cylinder '{name}': radius must be positive")
    if not material:
        errors.append(f"Cylinder '{name}': material must be a non-empty string")
    if material == "custom" and not has_custom_material:
        errors.append(f"Cylinder '{name}': custom_material required when material='custom'")

    # Domain bounds checks
    coords = {"x": (x1, x2, domain_x_m), "y": (y1, y2, domain_y_m), "z": (z1, z2, domain_z_m)}
    for dim, (c1, c2, dom) in coords.items():
        if dom is not None:
            for label, val in [(f"{dim}1", c1), (f"{dim}2", c2)]:
                if val is not None and (val < 0 or val > dom):
                    errors.append(
                        f"Cylinder '{name}': {label}={val:.6g} is outside domain [0, {dom:.6g}]"
                    )

    return _result(errors, [])


@tool
def validate_box(
    name: Annotated[str, "Box name"],
    x1: Annotated[float, "X start coordinate"],
    y1: Annotated[float, "Y start coordinate"],
    z1: Annotated[float, "Z start coordinate"],
    x2: Annotated[float, "X end coordinate"],
    y2: Annotated[float, "Y end coordinate"],
    z2: Annotated[float, "Z end coordinate"],
    material: Annotated[str, "Material identifier"],
    has_custom_material: Annotated[bool, "Whether a custom material is defined"] = False,
    domain_x_m: Annotated[Optional[float], "Domain X extent for bounds check"] = None,
    domain_y_m: Annotated[Optional[float], "Domain Y extent for bounds check"] = None,
    domain_z_m: Annotated[Optional[float], "Domain Z extent for bounds check"] = None,
) -> str:
    """Validate a box geometry object."""
    errors: list[str] = []
    if x2 <= x1:
        errors.append(f"Box '{name}': x2 must be > x1")
    if y2 <= y1:
        errors.append(f"Box '{name}': y2 must be > y1")
    if z2 <= z1:
        errors.append(f"Box '{name}': z2 must be > z1")
    if not material:
        errors.append(f"Box '{name}': material must be a non-empty string")
    if material == "custom" and not has_custom_material:
        errors.append(f"Box '{name}': custom_material required when material='custom'")

    # Domain bounds checks
    bounds = {"x": (x1, x2, domain_x_m), "y": (y1, y2, domain_y_m), "z": (z1, z2, domain_z_m)}
    for dim, (c1, c2, dom) in bounds.items():
        if dom is not None:
            if c1 < 0 or c1 > dom:
                errors.append(f"Box '{name}': {dim}1={c1:.6g} is outside domain [0, {dom:.6g}]")
            if c2 < 0 or c2 > dom:
                errors.append(f"Box '{name}': {dim}2={c2:.6g} is outside domain [0, {dom:.6g}]")

    return _result(errors, [])


@tool
def validate_sphere(
    name: Annotated[str, "Sphere name"],
    radius: Annotated[float, "Sphere radius in metres"],
    material: Annotated[str, "Material identifier"],
    has_custom_material: Annotated[bool, "Whether a custom material is defined"] = False,
    cx: Annotated[Optional[float], "Centre X coordinate"] = None,
    cy: Annotated[Optional[float], "Centre Y coordinate"] = None,
    cz: Annotated[Optional[float], "Centre Z coordinate"] = None,
    domain_x_m: Annotated[Optional[float], "Domain X extent for bounds check"] = None,
    domain_y_m: Annotated[Optional[float], "Domain Y extent for bounds check"] = None,
    domain_z_m: Annotated[Optional[float], "Domain Z extent for bounds check"] = None,
) -> str:
    """Validate a sphere geometry object."""
    errors: list[str] = []
    if radius <= 0:
        errors.append(f"Sphere '{name}': radius must be positive")
    if not material:
        errors.append(f"Sphere '{name}': material must be a non-empty string")
    if material == "custom" and not has_custom_material:
        errors.append(f"Sphere '{name}': custom_material required when material='custom'")

    # Domain bounds checks (centre +/- radius)
    sphere_bounds = {"x": (cx, domain_x_m), "y": (cy, domain_y_m), "z": (cz, domain_z_m)}
    for dim, (centre, dom) in sphere_bounds.items():
        if centre is not None and dom is not None and radius > 0:
            if centre - radius < 0:
                errors.append(
                    f"Sphere '{name}': c{dim}-radius={centre - radius:.6g} extends below 0"
                )
            if centre + radius > dom:
                errors.append(
                    f"Sphere '{name}': c{dim}+radius={centre + radius:.6g} exceeds domain {dim}={dom:.6g}"
                )

    return _result(errors, [])


# ---------------------------------------------------------------------------
# Surface roughness validation
# ---------------------------------------------------------------------------

@tool
def validate_surface(
    fractal_dim: Annotated[float, "Fractal dimension"],
    weight_x: Annotated[float, "Weight in X direction"],
    weight_y: Annotated[float, "Weight in Y direction"],
    amplitude_m: Annotated[float, "Roughness amplitude in metres"],
    add_water: Annotated[bool, "Whether water layer is added"],
    water_depth_m: Annotated[float, "Water depth in metres"] = 0.005,
) -> str:
    """Validate surface roughness configuration."""
    errors: list[str] = []
    if fractal_dim < 0:
        errors.append("fractal_dim must be >= 0")
    if weight_x < 0:
        errors.append("weight_x must be >= 0")
    if weight_y < 0:
        errors.append("weight_y must be >= 0")
    if amplitude_m <= 0:
        errors.append("amplitude_m must be > 0")
    if add_water and water_depth_m >= amplitude_m:
        errors.append("water_depth_m must be < amplitude_m when add_water=True")
    if add_water and water_depth_m <= 0:
        errors.append("water_depth_m must be > 0 when add_water=True")
    return _result(errors, [])


# ---------------------------------------------------------------------------
# Receiver array validation
# ---------------------------------------------------------------------------

@tool
def validate_rxarray(
    x1: Annotated[float, "X start"], y1: Annotated[float, "Y start"], z1: Annotated[float, "Z start"],
    x2: Annotated[float, "X end"], y2: Annotated[float, "Y end"], z2: Annotated[float, "Z end"],
    dx: Annotated[float, "X step"], dy: Annotated[float, "Y step"], dz: Annotated[float, "Z step"],
) -> str:
    """Validate receiver array configuration."""
    errors: list[str] = []
    if dx <= 0:
        errors.append("dx must be > 0")
    if dy <= 0:
        errors.append("dy must be > 0")
    if dz <= 0:
        errors.append("dz must be > 0")
    return _result(errors, [])


# ---------------------------------------------------------------------------
# Snapshot validation
# ---------------------------------------------------------------------------

@tool
def validate_snapshot(
    time_s: Annotated[float, "Snapshot time in seconds"],
) -> str:
    """Validate snapshot configuration."""
    if time_s <= 0:
        return "VALIDATION FAILED: time_s must be > 0"
    return "VALIDATION PASSED"


# ---------------------------------------------------------------------------
# Mesh / discretisation validation
# ---------------------------------------------------------------------------

@tool
def validate_mesh(
    max_cell_m: Annotated[float, "Maximum cell size in metres (dx, dy, or dz)"],
    center_freq_hz: Annotated[float, "Waveform centre frequency in Hz"],
    domain_x_m: Annotated[float, "Domain width in metres"],
    domain_y_m: Annotated[float, "Domain depth/height in metres"],
    eps_r_max: Annotated[float, "Estimated max relative permittivity in the model"] = 10.0,
    domain_z_m: Annotated[Optional[float], "Domain Z extent in metres (checked for integer cell multiple)"] = None,
    pml_cells: Annotated[Optional[int], "Number of PML cells (must be >= 0)"] = None,
) -> str:
    """Validate that the cell size satisfies the Nyquist spatial-sampling
    criterion and that domain dimensions are exact integer multiples of the
    cell size."""
    errors: list[str] = []

    # Minimum wavelength in the highest-permittivity medium
    lambda_min = C0 / (center_freq_hz * math.sqrt(eps_r_max))
    max_allowed = lambda_min / 10.0
    if max_cell_m > max_allowed:
        errors.append(
            f"max_cell_m ({max_cell_m:.6f}) exceeds lambda_min/10 "
            f"({max_allowed:.6f} m, lambda_min={lambda_min:.4f} m at "
            f"eps_r_max={eps_r_max})"
        )

    # Grid dimensions must be exact integer multiples of cell size
    tol = 1e-9
    for label, dim_m in [("domain_x_m", domain_x_m), ("domain_y_m", domain_y_m)]:
        ratio = dim_m / max_cell_m
        if abs(ratio - round(ratio)) > tol:
            errors.append(
                f"{label} ({dim_m}) is not an integer multiple of "
                f"max_cell_m ({max_cell_m}); ratio = {ratio:.6f}"
            )

    if domain_z_m is not None:
        nz = domain_z_m / max_cell_m
        if abs(nz - round(nz)) > tol:
            errors.append(
                f"domain_z_m ({domain_z_m}) is not an integer multiple of "
                f"max_cell_m ({max_cell_m}); ratio = {nz:.6f}"
            )

    if pml_cells is not None and pml_cells < 0:
        errors.append("pml_cells must be >= 0")

    return _result(errors, [])


# ---------------------------------------------------------------------------
# Time window validation
# ---------------------------------------------------------------------------

@tool
def validate_time_window(
    source_end_time_s: Annotated[float, "Simulation time window in seconds"],
    domain_depth_m: Annotated[float, "Total depth of the subsurface domain in metres"],
    eps_r_max: Annotated[float, "Estimated max relative permittivity"] = 10.0,
) -> str:
    """Validate that the time window is long enough for EM waves to propagate
    to the maximum depth and return (two-way travel time)."""
    errors: list[str] = []
    if source_end_time_s <= 0:
        errors.append("source_end_time_s must be > 0")
    v_min = C0 / math.sqrt(eps_r_max)
    min_time = 2.0 * domain_depth_m / v_min
    if source_end_time_s < min_time:
        errors.append(
            f"source_end_time ({source_end_time_s:.3e} s) is shorter than the "
            f"two-way travel time ({min_time:.3e} s) for depth {domain_depth_m} m "
            f"at eps_r_max={eps_r_max}"
        )
    return _result(errors, [])


# ---------------------------------------------------------------------------
# Antenna placement validation
# ---------------------------------------------------------------------------

@tool
def validate_antenna_placement(
    tx_x_m: Annotated[float, "Transmitter x-coordinate in metres"],
    rx_x_m: Annotated[float, "Receiver x-coordinate in metres"],
    domain_x_m: Annotated[float, "Domain width in metres"],
    max_cell_m: Annotated[float, "Cell size in metres"],
    min_edge_cells: Annotated[int, "Minimum cells from domain edge (including PML)"] = 15,
) -> str:
    """Validate that Tx and Rx antennas are at least min_edge_cells cells
    away from the domain boundaries to avoid PML interference."""
    errors: list[str] = []
    margin_m = min_edge_cells * max_cell_m
    for label, x in [("Tx", tx_x_m), ("Rx", rx_x_m)]:
        if x < margin_m:
            errors.append(
                f"{label} at x={x:.4f} m is only {x / max_cell_m:.1f} cells from "
                f"the left edge (need >= {min_edge_cells})"
            )
        dist_right = domain_x_m - x
        if dist_right < margin_m:
            errors.append(
                f"{label} at x={x:.4f} m is only {dist_right / max_cell_m:.1f} cells "
                f"from the right edge (need >= {min_edge_cells})"
            )
    return _result(errors, [])


# ---------------------------------------------------------------------------
# Material references validation
# ---------------------------------------------------------------------------

@tool
def validate_material_references(
    materials_used: Annotated[List[str], "Material identifiers referenced by geometry objects"],
    has_custom_material: Annotated[List[bool], "Whether each object defines a custom_material"],
) -> str:
    """Validate that all materials referenced by geometry objects are either
    gprMax builtins or backed by a custom material definition."""
    BUILTINS = {"pec", "free_space"}
    errors: list[str] = []
    for i, (mat, has_custom) in enumerate(zip(materials_used, has_custom_material)):
        if mat in BUILTINS:
            continue
        if mat == "custom" and has_custom:
            continue
        if mat == "custom" and not has_custom:
            errors.append(f"Object {i}: material='custom' but no custom_material defined")
        elif not has_custom:
            errors.append(
                f"Object {i}: material '{mat}' is not a gprMax builtin "
                f"({', '.join(sorted(BUILTINS))}) and has no custom_material"
            )
    return _result(errors, [])


# ---------------------------------------------------------------------------
# Essential parameters validation
# ---------------------------------------------------------------------------

@tool
def validate_essential_params(
    has_domain: Annotated[bool, "Whether domain dimensions are specified"],
    has_dx_dy_dz: Annotated[bool, "Whether cell sizes (dx, dy, dz) are specified"],
    has_time_window: Annotated[bool, "Whether the simulation time window is specified"],
) -> str:
    """Validate that the three essential gprMax simulation parameters are present."""
    missing: list[str] = []
    if not has_domain:
        missing.append("domain (domain_x, domain_y)")
    if not has_dx_dy_dz:
        missing.append("dx_dy_dz (cell sizes)")
    if not has_time_window:
        missing.append("time_window (source_end_time)")
    if missing:
        return "VALIDATION FAILED: Missing essential parameters: " + ", ".join(missing)
    return "VALIDATION PASSED"


# ---------------------------------------------------------------------------
# Temperature validation
# ---------------------------------------------------------------------------

@tool
def validate_temperature(
    temperature_c: Annotated[float, "Simulation temperature in degrees Celsius"],
) -> str:
    """Validate simulation temperature is physically reasonable."""
    errors: list[str] = []
    warnings: list[str] = []
    if temperature_c <= -50:
        errors.append("temperature_c must be > -50")
    if temperature_c >= 100:
        errors.append("temperature_c must be < 100")
    if not errors and abs(temperature_c - 20.0) > 15:
        warnings.append(
            f"temperature_c ({temperature_c:.1f}) deviates significantly from 20C reference; "
            "Debye water relaxation parameters may lose accuracy"
        )
    return _result(errors, warnings)


# ---------------------------------------------------------------------------
# Domain geometry validation
# ---------------------------------------------------------------------------

@tool
def validate_domain_geometry(
    domain_x_m: Annotated[float, "Domain width in metres"],
    domain_y_m: Annotated[float, "Domain depth/height in metres"],
    num_layers: Annotated[int, "Declared number of layers"],
    actual_layer_count: Annotated[int, "Actual number of layers provided"],
) -> str:
    """Validate domain dimensions and layer count consistency.

    Range-based layer checks (thickness vs domain, thickness vs cell size)
    are validated at sampling time in the dataset generator, not here.
    """
    errors: list[str] = []

    if domain_x_m <= 0:
        errors.append("domain_x_m must be > 0")
    if domain_y_m <= 0:
        errors.append("domain_y_m must be > 0")

    if num_layers != actual_layer_count:
        errors.append(
            f"num_layers ({num_layers}) does not match actual layer count "
            f"({actual_layer_count})"
        )

    return _result(errors, [])


# ---------------------------------------------------------------------------
# CFL stability validation
# ---------------------------------------------------------------------------

@tool
def validate_cfl(
    dx: Annotated[float, "Cell size in x (metres)"],
    dy: Annotated[float, "Cell size in y (metres)"],
    dz: Annotated[float, "Cell size in z (metres)"],
    dt: Annotated[float, "Time step in seconds"],
) -> str:
    """Validate FDTD CFL stability condition: dt <= 1/(c * sqrt(1/dx^2 + 1/dy^2 + 1/dz^2))."""
    errors: list[str] = []

    for label, val in [("dx", dx), ("dy", dy), ("dz", dz), ("dt", dt)]:
        if val <= 0:
            errors.append(f"{label} must be > 0")

    if not errors:
        dt_max = 1.0 / (C0 * math.sqrt(1.0 / dx**2 + 1.0 / dy**2 + 1.0 / dz**2))
        if dt > dt_max:
            errors.append(
                f"CFL violation: dt ({dt:.6e} s) exceeds dt_max ({dt_max:.6e} s); "
                "simulation will be numerically unstable"
            )

    return _result(errors, [])


# ---------------------------------------------------------------------------
# Simulation metadata validation
# ---------------------------------------------------------------------------

@tool
def validate_simulation_metadata(
    title: Annotated[Optional[str], "Simulation title (must be single-line)"] = None,
    num_threads: Annotated[Optional[int], "Number of threads (must be > 0)"] = None,
    output_dir: Annotated[Optional[str], "Output directory path (must be non-empty)"] = None,
) -> str:
    """Validate simulation metadata parameters."""
    errors: list[str] = []

    if title is not None:
        if not title.strip():
            errors.append("title must be a non-empty string")
        elif "\n" in title:
            errors.append("title must be single-line (no newlines)")

    if num_threads is not None and num_threads <= 0:
        errors.append("num_threads must be > 0")

    if output_dir is not None and not output_dir.strip():
        errors.append("output_dir must be a non-empty string")

    return _result(errors, [])




