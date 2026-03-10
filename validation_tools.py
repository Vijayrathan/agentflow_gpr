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
    thickness_m: Annotated[float, "Layer thickness in metres"],
    sand_pct: Annotated[float, "Sand percentage (0-100) or fraction (0-1) if texture_as_fraction=True"],
    silt_pct: Annotated[float, "Silt percentage (0-100) or fraction (0-1) if texture_as_fraction=True"],
    clay_pct: Annotated[float, "Clay percentage (0-100) or fraction (0-1) if texture_as_fraction=True"],
    theta_v: Annotated[float, "Volumetric water content (0.0-1.0)"],
    bulk_density_gcm3: Annotated[Optional[float], "Bulk density in g/cm³"] = None,
    particle_density_gcm3: Annotated[Optional[float], "Particle density in g/cm³"] = None,
    porosity: Annotated[Optional[float], "Explicit porosity (0-1); cross-checked against derived if densities also provided"] = None,
    organic_fraction: Annotated[Optional[float], "Organic fraction (0-1)"] = None,
    porewater_sigma_Sm: Annotated[Optional[float], "Porewater conductivity in S/m"] = None,
    texture_as_fraction: Annotated[bool, "If True, texture values are 0-1 fractions, auto-normalised to 0-100"] = False,
) -> str:
    """Validate a soil layer's physical parameters."""
    errors: list[str] = []
    warnings: list[str] = []

    # --- Fraction normalisation ---
    if texture_as_fraction:
        if all(v <= 1.0 for v in (sand_pct, silt_pct, clay_pct)):
            sand_pct *= 100.0
            silt_pct *= 100.0
            clay_pct *= 100.0
        else:
            errors.append(
                "texture_as_fraction=True but at least one texture value > 1.0; "
                "provide values in 0-1 range or set texture_as_fraction=False"
            )

    # --- Basic bounds ---
    if thickness_m <= 0:
        errors.append("Layer thickness must be > 0")
    if sand_pct < 0:
        errors.append("sand_pct must be >= 0")
    if silt_pct < 0:
        errors.append("silt_pct must be >= 0")
    if clay_pct < 0:
        errors.append("clay_pct must be >= 0")

    # --- Texture sum ---
    p_sum = sand_pct + silt_pct + clay_pct
    if abs(p_sum - 100.0) > 1e-6:
        errors.append(f"Sand + silt + clay must sum to 100 (got {p_sum:.2f})")

    # --- Theta_v ---
    if not (0.0 <= theta_v <= 1.0):
        errors.append("theta_v must be between 0.0 and 1.0")

    # --- Density checks ---
    if bulk_density_gcm3 is not None and bulk_density_gcm3 <= 0:
        errors.append("bulk_density_gcm3 must be > 0 if provided")
    if particle_density_gcm3 is not None and particle_density_gcm3 <= 0:
        errors.append("particle_density_gcm3 must be > 0 if provided")

    # --- Cross-density checks ---
    derived_porosity = None
    if (bulk_density_gcm3 is not None and bulk_density_gcm3 > 0
            and particle_density_gcm3 is not None and particle_density_gcm3 > 0):
        if bulk_density_gcm3 >= particle_density_gcm3:
            errors.append(
                f"bulk_density ({bulk_density_gcm3}) must be < particle_density "
                f"({particle_density_gcm3})"
            )
        else:
            derived_porosity = 1.0 - (bulk_density_gcm3 / particle_density_gcm3)
            if not (0.0 < derived_porosity < 1.0):
                errors.append(f"Derived porosity ({derived_porosity:.3f}) must be in (0, 1)")
            elif theta_v > derived_porosity:
                errors.append(
                    f"theta_v ({theta_v:.3f}) must be <= porosity ({derived_porosity:.3f}); "
                    "soil cannot hold more water than its pore space"
                )

    # --- Explicit porosity ---
    if porosity is not None:
        if not (0.0 < porosity < 1.0):
            errors.append(f"Explicit porosity ({porosity:.3f}) must be in (0, 1)")
        else:
            if theta_v > porosity:
                errors.append(
                    f"theta_v ({theta_v:.3f}) must be <= explicit porosity ({porosity:.3f}); "
                    "soil cannot hold more water than its pore space"
                )
            if derived_porosity is not None and abs(porosity - derived_porosity) > 0.05:
                warnings.append(
                    f"Explicit porosity ({porosity:.3f}) differs from density-derived "
                    f"porosity ({derived_porosity:.3f}) by more than 0.05"
                )

    # --- Organic fraction ---
    if organic_fraction is not None and organic_fraction < 0:
        errors.append("organic_fraction must be >= 0")

    # --- Porewater conductivity ---
    if porewater_sigma_Sm is not None and porewater_sigma_Sm < 0:
        errors.append("porewater_sigma_Sm must be >= 0")

    return _result(errors, warnings)


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
    theta_v: Annotated[float, "Volumetric water content (0.0-1.0)"],
    sand_pct: Annotated[float, "Sand percentage (0-100)"],
    silt_pct: Annotated[float, "Silt percentage (0-100)"],
    clay_pct: Annotated[float, "Clay percentage (0-100)"],
) -> str:
    """Validate that layer properties fall within the valid range for the
    chosen dielectric model."""
    errors: list[str] = []
    if model == "peplinski":
        if not (0.3e9 <= f0 <= 1.3e9):
            errors.append("Peplinski valid for ~0.3-1.3 GHz")
        if not (0.0 <= theta_v <= 0.30):
            errors.append("Peplinski moisture valid ~0-0.30")
        if not (15 <= sand_pct <= 50 and 5 <= clay_pct <= 20 and 35 <= silt_pct <= 65):
            errors.append("Peplinski texture ranges: sand 15-50%, clay 5-20%, silt 35-65%")
    elif model == "dobson":
        if not (1.4e9 <= f0 <= 18e9):
            errors.append("Dobson valid for ~1.4-18 GHz")
        if not (0.0 <= theta_v <= 0.50):
            errors.append("Dobson moisture valid ~0-0.50")
    elif model == "mironov":
        if not (0.6e9 <= f0 <= 18e9):
            errors.append("Mironov valid for ~0.6-18 GHz")
        if not (0.0 <= theta_v <= 0.45):
            errors.append("Mironov moisture valid ~0-0.45")
    elif model == "crim":
        pass
    else:
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
    layer_thicknesses_m: Annotated[List[float], "List of layer thicknesses in metres"],
    num_layers: Annotated[int, "Declared number of layers"],
    cell_size_m: Annotated[Optional[float], "Cell size in metres (for layer resolvability check)"] = None,
) -> str:
    """Validate domain dimensions, layer count, layer fit, and layer resolvability."""
    errors: list[str] = []
    warnings: list[str] = []

    if domain_x_m <= 0:
        errors.append("domain_x_m must be > 0")
    if domain_y_m <= 0:
        errors.append("domain_y_m must be > 0")

    if num_layers != len(layer_thicknesses_m):
        errors.append(
            f"num_layers ({num_layers}) does not match actual layer count "
            f"({len(layer_thicknesses_m)})"
        )

    total_thickness = sum(layer_thicknesses_m)
    if domain_y_m > 0 and total_thickness > domain_y_m + 1e-9:
        errors.append(
            f"Total layer thickness ({total_thickness:.6g} m) exceeds "
            f"domain_y_m ({domain_y_m:.6g} m)"
        )

    if cell_size_m is not None and cell_size_m > 0:
        for i, t in enumerate(layer_thicknesses_m):
            if t < cell_size_m:
                errors.append(
                    f"Layer {i}: thickness ({t:.6g} m) is less than cell size "
                    f"({cell_size_m:.6g} m); layer is unresolvable"
                )

    return _result(errors, warnings)


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


# ---------------------------------------------------------------------------
# Range consistency validation
# ---------------------------------------------------------------------------

@tool
def validate_ranges(
    range_pairs: Annotated[List[List[float]], "List of [min, max] pairs to validate"],
    range_names: Annotated[List[str], "Name for each range pair (for error messages)"],
    texture_bounds: Annotated[Optional[List[List[float]]], "[[sand_min,sand_max],[silt_min,silt_max],[clay_min,clay_max]] for texture sum checks"] = None,
) -> str:
    """Validate that min <= max for all range pairs and that texture bounds
    allow a valid sum-to-100 combination."""
    errors: list[str] = []
    warnings: list[str] = []

    if len(range_pairs) != len(range_names):
        errors.append(
            f"range_pairs length ({len(range_pairs)}) != range_names length ({len(range_names)})"
        )
        return _result(errors, warnings)

    for pair, name in zip(range_pairs, range_names):
        if len(pair) != 2:
            errors.append(f"'{name}': expected [min, max] pair, got {len(pair)} values")
            continue
        lo, hi = pair
        if lo > hi:
            errors.append(f"'{name}': min ({lo}) > max ({hi})")

    if texture_bounds is not None:
        if len(texture_bounds) != 3:
            errors.append("texture_bounds must have exactly 3 entries [sand, silt, clay]")
        else:
            sand_range, silt_range, clay_range = texture_bounds
            if all(len(r) == 2 for r in (sand_range, silt_range, clay_range)):
                lower_sum = sand_range[0] + silt_range[0] + clay_range[0]
                upper_sum = sand_range[1] + silt_range[1] + clay_range[1]
                if lower_sum > 100 + 1e-6:
                    errors.append(
                        f"Texture lower bounds sum to {lower_sum:.2f} > 100; "
                        "impossible to satisfy sand+silt+clay=100"
                    )
                if upper_sum < 100 - 1e-6:
                    errors.append(
                        f"Texture upper bounds sum to {upper_sum:.2f} < 100; "
                        "impossible to reach sand+silt+clay=100"
                    )

    return _result(errors, warnings)


# ---------------------------------------------------------------------------
# Comprehensive cross-parameter validation
# ---------------------------------------------------------------------------

@tool
def validate_cross_params(
    sand_pct: Annotated[Optional[float], "Sand percentage (0-100)"] = None,
    silt_pct: Annotated[Optional[float], "Silt percentage (0-100)"] = None,
    clay_pct: Annotated[Optional[float], "Clay percentage (0-100)"] = None,
    theta_v: Annotated[Optional[float], "Volumetric water content (0-1)"] = None,
    bulk_density_gcm3: Annotated[Optional[float], "Bulk density in g/cm³"] = None,
    particle_density_gcm3: Annotated[Optional[float], "Particle density in g/cm³"] = None,
    porosity: Annotated[Optional[float], "Explicit porosity (0-1)"] = None,
    max_cell_m: Annotated[Optional[float], "Cell size in metres"] = None,
    center_freq_hz: Annotated[Optional[float], "Centre frequency in Hz"] = None,
    eps_r_max: Annotated[Optional[float], "Max relative permittivity"] = None,
    domain_x_m: Annotated[Optional[float], "Domain X dimension in metres"] = None,
    domain_y_m: Annotated[Optional[float], "Domain Y dimension in metres"] = None,
    num_layers: Annotated[Optional[int], "Declared layer count"] = None,
    actual_layer_count: Annotated[Optional[int], "Actual number of layers provided"] = None,
    model: Annotated[Optional[str], "Dielectric model name"] = None,
) -> str:
    """Comprehensive cross-parameter consistency check. Only validates
    relationships between parameters that are both provided (non-None)."""
    errors: list[str] = []
    warnings: list[str] = []

    # 1. Porosity derivation cross-check
    derived_porosity = None
    if bulk_density_gcm3 is not None and particle_density_gcm3 is not None:
        if bulk_density_gcm3 > 0 and particle_density_gcm3 > 0:
            if bulk_density_gcm3 >= particle_density_gcm3:
                errors.append(
                    f"bulk_density ({bulk_density_gcm3}) must be < particle_density "
                    f"({particle_density_gcm3})"
                )
            else:
                derived_porosity = 1.0 - (bulk_density_gcm3 / particle_density_gcm3)
                if porosity is not None and abs(porosity - derived_porosity) > 0.05:
                    errors.append(
                        f"Explicit porosity ({porosity:.3f}) differs from density-derived "
                        f"porosity ({derived_porosity:.3f}) by more than 0.05"
                    )

    # 2. theta_v vs porosity
    effective_porosity = porosity if porosity is not None else derived_porosity
    if theta_v is not None and effective_porosity is not None:
        if theta_v > effective_porosity:
            errors.append(
                f"theta_v ({theta_v:.3f}) exceeds porosity ({effective_porosity:.3f}); "
                "soil cannot hold more water than its pore space"
            )

    # 3. Texture sum
    if sand_pct is not None and silt_pct is not None and clay_pct is not None:
        p_sum = sand_pct + silt_pct + clay_pct
        if abs(p_sum - 100.0) > 1e-6:
            errors.append(f"Sand + silt + clay must sum to 100 (got {p_sum:.2f})")

    # 4. Nyquist cell size
    if max_cell_m is not None and center_freq_hz is not None and eps_r_max is not None:
        if center_freq_hz > 0 and eps_r_max >= 1.0:
            lambda_min = C0 / (center_freq_hz * math.sqrt(eps_r_max))
            max_allowed = lambda_min / 10.0
            if max_cell_m > max_allowed:
                errors.append(
                    f"max_cell_m ({max_cell_m:.6f}) exceeds lambda_min/10 "
                    f"({max_allowed:.6f} m) at eps_r_max={eps_r_max}"
                )

    # 5. Domain divisibility
    if max_cell_m is not None and max_cell_m > 0:
        tol = 1e-9
        if domain_x_m is not None:
            nx = domain_x_m / max_cell_m
            if abs(nx - round(nx)) > tol:
                errors.append(
                    f"domain_x_m ({domain_x_m}) is not an integer multiple of "
                    f"max_cell_m ({max_cell_m}); ratio = {nx:.6f}"
                )
        if domain_y_m is not None:
            ny = domain_y_m / max_cell_m
            if abs(ny - round(ny)) > tol:
                errors.append(
                    f"domain_y_m ({domain_y_m}) is not an integer multiple of "
                    f"max_cell_m ({max_cell_m}); ratio = {ny:.6f}"
                )

    # 6. Layer count match
    if num_layers is not None and actual_layer_count is not None:
        if num_layers != actual_layer_count:
            errors.append(
                f"num_layers ({num_layers}) != actual layer count ({actual_layer_count})"
            )

    # 7. Peplinski-specific bounds
    if model is not None and model.lower() == "peplinski":
        if sand_pct is not None and not (15 <= sand_pct <= 50):
            errors.append(f"Peplinski: sand_pct ({sand_pct}) outside valid range 15-50%")
        if clay_pct is not None and not (5 <= clay_pct <= 20):
            errors.append(f"Peplinski: clay_pct ({clay_pct}) outside valid range 5-20%")
        if silt_pct is not None and not (35 <= silt_pct <= 65):
            errors.append(f"Peplinski: silt_pct ({silt_pct}) outside valid range 35-65%")
        if theta_v is not None and not (0.0 <= theta_v <= 0.30):
            errors.append(f"Peplinski: theta_v ({theta_v}) outside valid range 0-0.30")
        if center_freq_hz is not None and not (0.3e9 <= center_freq_hz <= 1.3e9):
            errors.append(
                f"Peplinski: center_freq_hz ({center_freq_hz:.3e}) outside valid range 0.3-1.3 GHz"
            )

    return _result(errors, warnings)
