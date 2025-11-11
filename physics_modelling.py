"""
 deterministic core (v2): Physics-first generator of gprMax .in files.

Implements 7 upgrades aligned with the review paper:
1) Model switch with guardrails: 'crim', 'peplinski', 'dobson', 'mironov'.
   - Frequency and moisture/texture validity checks for each model.
2) Prefer porosity from densities (P = 1 - bulk/particle) when provided; fall back to texture heuristic.
3) Water dielectric with temperature and split bound/free (Mironov-style option).
4) Strict enforcement of model validity windows by default.
5) Propagation helpers: skin depth and unambiguous range (for stepped-frequency designs).
6) Loss mapping transparency: prefer model-provided sigma; otherwise sigma = ωε0 Im{εr}.
7) Organic and salinity flags: organic raises σ and εr (when wet); salinity sets porewater σ.

Note: Peplinski/Dobson here are currently routed through CRIM but protected by their
validity windows and ready for later exact formulas. Mironov path implements a practical
bound/free split with two Debye poles.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import cmath
import math
import cmath

# Physical constants
EPS0 = 8.854187817e-12  # F/m
MU0 = 4 * math.pi * 1e-7
C0 = 1.0 / math.sqrt(EPS0 * MU0)

# ---------------------------
# Utility and physics helpers
# ---------------------------

def cpow(z: complex, p: float) -> complex:
    """Complex power with real exponent p, safe for small magnitudes."""
    if abs(z) == 0:
        return 0j
    r, phi = cmath.polar(z)
    return cmath.rect(r ** p, p * phi)

def water_permittivity_debye(f_hz: float, temp_c: float = 20.0, sigma_ion: float = 0.0) -> complex:
    """Single-pole Debye water with coarse temperature and ionic conductivity dependence."""
    eps_s_20 = 80.1
    eps_inf = 4.9
    tau_20 = 9.23e-12
    # very coarse linearized temperature trends near room temp
    d_epss_dT = -0.4
    d_tau_dT = -0.15e-12
    eps_s = eps_s_20 + d_epss_dT * (temp_c - 20.0)
    tau = max(tau_20 + d_tau_dT * (temp_c - 20.0), 2e-12)
    omega = 2 * math.pi * f_hz
    debye = eps_inf + (eps_s - eps_inf) / (1.0 + 1j * omega * tau)
    if sigma_ion > 0 and omega > 0:
        debye -= 1j * sigma_ion / (omega * EPS0)
    return debye


def solid_permittivity_from_texture(sand_pct: float, silt_pct: float, clay_pct: float) -> float:
    """Representative solids permittivity from texture (quartz/silicates/clays)."""
    s = max(sand_pct, 0.0)
    si = max(silt_pct, 0.0)
    c = max(clay_pct, 0.0)
    total = s + si + c
    if total <= 0:
        return 5.0
    s_norm, si_norm, c_norm = s / total, si / total, c / total
    eps_sand = 4.5
    eps_silt = 5.5
    eps_clay = 7.0
    return eps_sand * s_norm + eps_silt * si_norm + eps_clay * c_norm


def estimate_porosity(sand_pct: float, silt_pct: float, clay_pct: float) -> float:
    """Fallback porosity estimate from texture (bounds 0.30..0.60)."""
    s = max(min(sand_pct, 100.0), 0.0)
    c = max(min(clay_pct, 100.0), 0.0)
    base = 0.42
    adj = 0.0012 * (c - s * 0.4)
    n = base + adj
    return float(min(max(n, 0.30), 0.60))


def porosity_from_densities(bulk_gcm3: float, particle_gcm3: float = 2.65) -> float:
    """Porosity P = 1 - bulk/particle (dimensionless)."""
    return float(min(max(1.0 - (bulk_gcm3 / particle_gcm3), 0.0), 0.8))


def crim_mixture(theta_v: float, porosity: float, eps_solid: float, eps_water: complex) -> complex:
    """CRIM: sqrt(eps_eff) = φ_air*1 + φ_w*sqrt(ε_w) + φ_s*sqrt(ε_solid)."""
    porosity = float(min(max(porosity, 0.0), 0.9))
    theta_v = float(min(max(theta_v, 0.0), porosity))
    phi_w = theta_v
    phi_air = max(porosity - theta_v, 0.0)
    phi_s = 1.0 - porosity
    n_eff = phi_air * 1.0 + phi_w * cmath.sqrt(eps_water) + phi_s * math.sqrt(eps_solid)
    return n_eff * n_eff


def mironov_mixture(theta_v: float, porosity: float, eps_solid: float, f_hz: float, temp_c: float,
                    clay_frac_pct: float, sigma_free: float = 0.0) -> Tuple[complex, Optional[float]]:
    """Practical Mironov-style split: bound and free water Debye poles.

    Returns (ε_eff, σ_model_if_any). σ_model reflects free-water conductivity contribution only.
    """
    porosity = float(min(max(porosity, 0.0), 0.9))
    theta_v = float(min(max(theta_v, 0.0), porosity))
    # bound water fraction proportional to clay; cap by moisture and empirical max
    max_bound = min(0.45, 0.005 * clay_frac_pct)
    theta_bound = min(theta_v, max_bound)
    theta_free = max(theta_v - theta_bound, 0.0)
    # free water Debye (with ionic σ)
    eps_w_free = water_permittivity_debye(f_hz, temp_c, sigma_ion=sigma_free)
    # bound water Debye (slower, lower εs)
    eps_s_bound = 50.0
    eps_inf_bound = 4.9
    tau_bound = 30e-12
    omega = 2 * math.pi * f_hz
    eps_w_bound = eps_inf_bound + (eps_s_bound - eps_inf_bound) / (1.0 + 1j * omega * tau_bound)
    # CRIM-like in n-space with split water
    phi_air = max(porosity - theta_v, 0.0)
    phi_s = 1.0 - porosity
    n_eff = (phi_air * 1.0
             + theta_bound * cmath.sqrt(eps_w_bound)
             + theta_free * cmath.sqrt(eps_w_free)
             + phi_s * math.sqrt(eps_solid))
    eps_eff = n_eff * n_eff
    sigma_model = None
    if sigma_free > 0.0:
        sigma_model = sigma_free * (theta_free / max(theta_v, 1e-9))
    return eps_eff, sigma_model


def peplinski_mixture(theta_v: float, rho_b_gcm3: float, rho_s_gcm3: float,
                      sand_pct: float, clay_pct: float,
                      f_hz: float, temp_c: float,
                      eps_solid: Optional[float] = None) -> complex:
    """True Peplinski (1995) implementation over 0.3-1.3 GHz.

    Steps:
    1) Compute Dobson (1985) complex mixture in α-space with α=0.65 and β'=1.2748 − 0.00519·Sand% − 0.00152·Clay%.
    2) Apply Peplinski linear correction to the REAL part: ε' = 1.156·ε'_L − 0.68.
    3) Replace effective conductivity with Peplinski σ_eff = 0.0467 + 0.2204·ρ_b − 0.4111·Sand% + 0.6614·Clay%  [S/m].
       Add this as −j σ_eff/(ω ε0) to the complex permittivity (avoid double counting ionic σ).

    Notes:
    - Sand% and Clay% are in PERCENT by weight (as in the paper and our LayerSpec).
    - ρ_b, ρ_s are in g/cm^3.
    - Water permittivity uses Debye without ionic conductivity term here; σ_eff accounts for losses.
    - eps_solid: if None, use ε_s = (1.01 + 0.44·ρ_s)^2 − 0.062 (Dobson '85).
    """
    # Parameters
    alpha = 0.65
    # β' depends on texture (percent by weight)
    beta_p = 1.2748 - 0.00519 * sand_pct - 0.00152 * clay_pct
    # Solids permittivity
    if eps_solid is None:
        eps_solid = (1.01 + 0.44 * rho_s_gcm3) ** 2 - 0.062
    # Water permittivity (no ionic term here to avoid double counting)
    eps_w = water_permittivity_debye(f_hz, temp_c, sigma_ion=0.0)

    # Dobson complex mixture in α-space
    theta_v = float(max(min(theta_v, 1.0), 0.0))
    X = (1.0
         + (rho_b_gcm3 / rho_s_gcm3) * (eps_solid ** alpha - 1.0)
         + (theta_v ** beta_p) * cpow(eps_w, alpha)
         - theta_v)
    eps_L = cpow(X, 1.0 / alpha)  # complex ε from Dobson

    # Peplinski corrections
    eps_L_real = eps_L.real
    eps_real_corr = 1.156 * eps_L_real - 0.68  # Eq.(9) correction to real part

    # Effective conductivity σ_eff (Eq.(10)), sand/clay in percent, ρb in g/cm3
    sigma_eff = 0.0467 + 0.2204 * rho_b_gcm3 - 0.4111 * (sand_pct / 100.0) * 100 + 0.6614 * (clay_pct / 100.0) * 100
    # The last line simplifies to 0.0467 + 0.2204 ρb − 0.4111 Sand% + 0.6614 Clay%

    # Build final complex ε: use corrected real part and combine imaginary part from Dobson water loss with σ_eff
    omega = 2 * math.pi * f_hz
    eps_imag_total = eps_L.imag + sigma_eff / (omega * EPS0)
    return complex(eps_real_corr, eps_imag_total)


def eps_to_sigma(eps_eff: complex, f_hz: float) -> tuple[float, float]:
    eps_r_real = float(max(eps_eff.real, 1.0))
    eps_r_imag = float(eps_eff.imag)
    omega = 2 * math.pi * f_hz
    sigma = omega * EPS0 * abs(eps_r_imag)  # use abs, never clip to zero
    return eps_r_real, sigma


# Propagation helpers (design guidance)

def attenuation_constants(eps_r: float, sigma: float, f_hz: float) -> Tuple[float, float]:
    """Return (alpha [Np/m], beta [rad/m]) for a plane wave in a lossy dielectric."""
    omega = 2 * math.pi * f_hz
    eps = eps_r * EPS0
    term = math.sqrt(1 + (sigma / (omega * eps)) ** 2)
    alpha = omega * math.sqrt(MU0 * eps) * math.sqrt(0.5 * (term - 1.0))
    beta = omega * math.sqrt(MU0 * eps) * math.sqrt(0.5 * (term + 1.0))
    return alpha, beta


def skin_depth(eps_r: float, sigma: float, f_hz: float) -> float:
    """1/e amplitude depth (≈ 1/alpha)."""
    alpha, _ = attenuation_constants(eps_r, sigma, f_hz)
    return float("inf") if alpha == 0.0 else 1.0 / alpha


def rmax_unambiguous(delta_f_hz: float, eps_r: float) -> float:
    """Rmax = c / (2 sqrt(εr) Δf) for stepped-frequency designs."""
    return C0 / (2.0 * math.sqrt(max(eps_r, 1.0)) * max(delta_f_hz, 1e-12))


# ---------------------------
# Data structures
# ---------------------------

@dataclass
class LayerSpec:
    thickness_m: float
    sand_pct: float
    silt_pct: float
    clay_pct: float
    theta_v: float  # volumetric water content 0..1
    porosity: Optional[float] = None
    bulk_density_gcm3: Optional[float] = None
    particle_density_gcm3: Optional[float] = None  # default 2.65 if not provided
    organic_fraction: float = 0.0  # 0..1
    salinity_class: Optional[str] = None  # 'fresh'|'brackish'|'saline'
    porewater_sigma_Sm: Optional[float] = None  # overrides salinity_class if set
    name: Optional[str] = None

    def validate(self) -> None:
        if self.thickness_m <= 0:
            raise ValueError("Layer thickness must be > 0")
        p_sum = self.sand_pct + self.silt_pct + self.clay_pct
        if abs(p_sum - 100.0) > 1e-6:
            raise ValueError("Sand + silt + clay percentages must sum to 100")
        if not (0.0 <= self.theta_v <= 1.0):
            raise ValueError("theta_v must be 0..1 (volumetric)")
        if self.bulk_density_gcm3 is not None and self.bulk_density_gcm3 <= 0:
            raise ValueError("bulk_density_gcm3 must be > 0 if provided")
        if self.particle_density_gcm3 is not None and self.particle_density_gcm3 <= 0:
            raise ValueError("particle_density_gcm3 must be > 0 if provided")


@dataclass
class WaveformSpec:
    kind: str  # 'ricker' | 'gaussian'
    amplitude: float
    center_freq_hz: float
    name: str = "wf"

    def gprmax_line(self) -> str:
        k = self.kind.lower()
        if k not in {"ricker", "gaussian"}:
            raise ValueError("Unsupported waveform kind. Use 'ricker' or 'gaussian'.")
        return f"#waveform: {k} {self.amplitude:g} {self.center_freq_hz:g} {self.name}"


@dataclass
class AntennaSpec:
    kind: str  # 'hertzian_dipole'
    axis: str  # 'x'|'y'|'z'
    tx_rx_offset_m: float = 0.05

    def validate(self) -> None:
        if self.kind.lower() not in {"hertzian_dipole"}:
            raise ValueError("Only 'hertzian_dipole' is supported")
        if self.axis.lower() not in {"x", "y", "z"}:
            raise ValueError("Axis must be 'x','y','z'")


@dataclass
class ModelSpec:
    title: str
    layers: List[LayerSpec]
    waveform: WaveformSpec
    antenna: AntennaSpec
    source_height_m: float
    domain_xy_m: Tuple[float, float] = (0.6, 0.4)
    top_air_extra_m: Optional[float] = None
    cells_per_wavelength: int = 15
    max_cell_m: float = 0.005
    rx_same_height: bool = True
    temperature_c: float = 20.0
    # model selection and validity
    model: str = "crim"  # 'crim'|'peplinski'|'dobson'|'mironov'
    enforce_validity: bool = True
    # default porewater conductivity per salinity class
    salinity_defaults_Sm: Tuple[float, float, float] = (0.0, 0.5, 3.0)  # fresh, brackish, saline

    def build(self) -> str:
        if not self.layers:
            raise ValueError("At least one layer is required")
        for L in self.layers:
            L.validate()

        f0 = self.waveform.center_freq_hz

        def check_validity(model: str, f0: float, L: LayerSpec) -> None:
            theta = L.theta_v
            sand, silt, clay = L.sand_pct, L.silt_pct, L.clay_pct
            if model == "peplinski":
                if not (0.3e9 <= f0 <= 1.3e9):
                    raise ValueError("Peplinski valid for ~0.3–1.3 GHz")
                if not (0.0 <= theta <= 0.30):
                    raise ValueError("Peplinski moisture valid ~0–0.30")
                if not (15 <= sand <= 50 and 5 <= clay <= 20 and 35 <= silt <= 65):
                    raise ValueError("Peplinski texture ranges: sand 15–50, clay 5–20, silt 35–65 %")
            elif model == "dobson":
                if not (1.4e9 <= f0 <= 18e9):
                    raise ValueError("Dobson valid for ~1.4–18 GHz")
                if not (0.0 <= theta <= 0.50):
                    raise ValueError("Dobson moisture valid ~0–0.50")
            elif model == "mironov":
                if not (0.6e9 <= f0 <= 18e9):
                    raise ValueError("Mironov valid for ~0.6–18 GHz")
                if not (0.0 <= theta <= 0.45):
                    raise ValueError("Mironov moisture valid ~0–0.45")
            elif model == "crim":
                pass
            else:
                raise ValueError("Unknown model selection")

        eps_r_list: List[float] = []
        sigma_list: List[float] = []
        mat_names: List[str] = []

        for i, L in enumerate(self.layers, 1):
            sigma_model = None  
            if self.enforce_validity:
                check_validity(self.model, f0, L)

            # Porosity
            if L.porosity is not None:
                n = L.porosity
            elif L.bulk_density_gcm3 is not None:
                pd = L.particle_density_gcm3 if L.particle_density_gcm3 is not None else 2.65
                n = porosity_from_densities(L.bulk_density_gcm3, pd)
            else:
                n = estimate_porosity(L.sand_pct, L.silt_pct, L.clay_pct)

            eps_s = solid_permittivity_from_texture(L.sand_pct, L.silt_pct, L.clay_pct)

            # Pore water conductivity from salinity class if not explicitly set
            sigma_pore = L.porewater_sigma_Sm if L.porewater_sigma_Sm is not None else None
            if sigma_pore is None and L.salinity_class:
                sc = L.salinity_class.lower()
                if sc == "fresh":
                    sigma_pore = self.salinity_defaults_Sm[0]
                elif sc == "brackish":
                    sigma_pore = self.salinity_defaults_Sm[1]
                elif sc == "saline":
                    sigma_pore = self.salinity_defaults_Sm[2]
            if sigma_pore is None:
                sigma_pore = 0.0

            # Compute ε_eff
            if self.model == "crim":
                eps_w = water_permittivity_debye(f0, self.temperature_c, sigma_ion=sigma_pore)
                eps_eff = crim_mixture(L.theta_v, n, eps_s, eps_w)
                sigma_model = None
            elif self.model == "mironov":
                eps_eff, sigma_model = mironov_mixture(L.theta_v, n, eps_s, f0, self.temperature_c, L.clay_pct, sigma_free=sigma_pore)
            elif self.model == "peplinski":
                # True Peplinski 0.3–1.3 GHz implementation (ε' linear correction + σ_eff replacement)
                # Prefer densities to get ρ_b and ρ_s; fall back to defaults if needed.
                rho_b = L.bulk_density_gcm3 if L.bulk_density_gcm3 is not None else 1.5
                rho_s = L.particle_density_gcm3 if L.particle_density_gcm3 is not None else 2.65
                eps_eff = peplinski_mixture(
                    theta_v=L.theta_v,
                    rho_b_gcm3=rho_b,
                    rho_s_gcm3=rho_s,
                    sand_pct=L.sand_pct,
                    clay_pct=L.clay_pct,
                    f_hz=f0,
                    temp_c=self.temperature_c,
                    eps_solid=eps_s,
                )
                sigma_model = None  # handled inside via σ_eff
            elif self.model == "dobson":
                # Dobson (1985) via the same α-space blend, without Peplinski corrections
                alpha = 0.65
                beta_p = 1.2748 - 0.00519 * L.sand_pct - 0.00152 * L.clay_pct
                eps_w = water_permittivity_debye(f0, self.temperature_c, sigma_ion=0.0)
                X = (1.0
                     + ( (L.bulk_density_gcm3 if L.bulk_density_gcm3 is not None else 1.5) / (L.particle_density_gcm3 if L.particle_density_gcm3 is not None else 2.65) ) * (eps_s ** alpha - 1.0)
                     + (L.theta_v ** beta_p) * cpow(eps_w, alpha)
                     - L.theta_v)
                eps_eff = cpow(X, 1.0 / alpha)
                sigma_model = None
            else:
                raise ValueError("Unknown model selection")

            eps_r, sigma_eff = eps_to_sigma(eps_eff, f0)
            sigma_final = sigma_model if (sigma_model is not None and sigma_model > 0) else sigma_eff

            # Organic matter heuristic adjustments
            if L.organic_fraction > 0.0:
                wetness = L.theta_v / max(n, 1e-6) if n > 0 else 0.0
                eps_r *= 1.0 + 0.05 * L.organic_fraction * min(wetness, 1.0)
                sigma_final *= 1.0 + 0.3 * L.organic_fraction

            eps_r_list.append(eps_r)
            sigma_list.append(max(sigma_final, 0.0))
            mat_names.append(L.name if L.name else f"layer{i}")

        # Geometry extents
        total_layers_thick = sum(L.thickness_m for L in self.layers)
        air_top = max(self.source_height_m + 6 * self.max_cell_m, 0.05)
        if self.top_air_extra_m is not None:
            air_top = max(air_top, self.top_air_extra_m)
        z_extent = air_top + total_layers_thick
        x_extent, y_extent = self.domain_xy_m

        # Grid spacing from min wavelength
        eps_r_max = max(eps_r_list)
        lambda_min = C0 / (f0 * math.sqrt(eps_r_max))
        dx_candidate = min(lambda_min / self.cells_per_wavelength, self.max_cell_m)
        dx = dy = dz = dx_candidate

        # Time window: two-way through full depth in slowest medium plus margin
        v_min = C0 / math.sqrt(eps_r_max)
        t_two_way = 2 * z_extent / v_min
        t_margin = 0.2 * t_two_way
        time_window = t_two_way + t_margin

        # Source/receiver placement
        x0 = 0.5 * x_extent
        y0 = 0.5 * y_extent
        z_tx = air_top + self.source_height_m
        if z_tx >= z_extent:
            raise ValueError("Source height exceeds model z-extent; increase top air or reduce height")
        rx_x = x0 + self.antenna.tx_rx_offset_m
        rx_y = y0
        rx_z = z_tx if self.rx_same_height else z_tx

        # Build .in
        lines: List[str] = []
        lines.append(f"#title: {self.title}")
        lines.append(f"#domain: {x_extent:.6g} {y_extent:.6g} {z_extent:.6g}")
        lines.append(f"#dx_dy_dz: {dx:.6g} {dy:.6g} {dz:.6g}")
        lines.append(f"#time_window: {time_window:.6g}")
        lines.append("")

        for eps_r, sigma, name in zip(eps_r_list, sigma_list, mat_names):
            lines.append(f"#material: {eps_r:.8g} {sigma:.8g} 1 0 {name}")

        lines.append("")

        lines.append(self.waveform.gprmax_line())
        k = self.antenna.kind.lower()
        axis = self.antenna.axis.lower()
        if k == "hertzian_dipole":
            lines.append(f"#hertzian_dipole: {axis} {x0:.6g} {y0:.6g} {z_tx:.6g} {self.waveform.name}")
            lines.append(f"#rx: {rx_x:.6g} {rx_y:.6g} {rx_z:.6g}")
        lines.append("")

        z_cur = air_top
        for L, name in zip(self.layers, mat_names):
            z1 = z_cur
            z2 = z_cur + L.thickness_m
            lines.append(f"#box: 0 0 {z1:.6g} {x_extent:.6g} {y_extent:.6g} {z2:.6g} {name}")
            z_cur = z2

        lines.append("")
        lines.append(f"#geometry_view: 0 0 0 {x_extent:.6g} {y_extent:.6g} {z_extent:.6g} {dx:.6g} {dy:.6g} {dz:.6g} model_view n")

        return "\n".join(lines)


def generate_gprmax_input_file(
    # LayerSpec parameters (for each layer in layers list)
    layer_thicknesses_m: List[float],
    layer_sand_pcts: List[float],
    layer_silt_pcts: List[float],
    layer_clay_pcts: List[float],
    layer_theta_vs: List[float],
    layer_porosities: Optional[List[Optional[float]]] = None,
    layer_bulk_densities_gcm3: Optional[List[Optional[float]]] = None,
    layer_particle_densities_gcm3: Optional[List[Optional[float]]] = None,
    layer_organic_fractions: Optional[List[float]] = None,
    layer_salinity_classes: Optional[List[Optional[str]]] = None,
    layer_porewater_sigmas_Sm: Optional[List[Optional[float]]] = None,
    layer_names: Optional[List[Optional[str]]] = None,
    # WaveformSpec parameters
    waveform_kind: str = "ricker",
    waveform_amplitude: float = 1.0,
    waveform_center_freq_hz: float = 1.5e9,
    waveform_name: str = "wf",
    # AntennaSpec parameters
    antenna_kind: str = "hertzian_dipole",
    antenna_axis: str = "z",
    antenna_tx_rx_offset_m: float = 0.05,
    # ModelSpec parameters
    model_title: str = "Layered soil with selectable dielectric model",
    source_height_m: float = 0.07,
    domain_xy_m: Tuple[float, float] = (0.6, 0.4),
    top_air_extra_m: Optional[float] = None,
    cells_per_wavelength: int = 15,
    max_cell_m: float = 0.005,
    rx_same_height: bool = True,
    temperature_c: float = 20.0,
    model: str = "crim",
    enforce_validity: bool = True,
    salinity_defaults_Sm: Tuple[float, float, float] = (0.0, 0.5, 3.0),
    output_filename: str = "generated.in",
):
    """
    Generate a gprMax input file with all parameters from LayerSpec, WaveformSpec, AntennaSpec, and ModelSpec.
    
    Args:
        layer_thicknesses_m: List of layer thicknesses in meters (required for each layer)
        layer_sand_pcts: List of sand percentages (required for each layer)
        layer_silt_pcts: List of silt percentages (required for each layer)
        layer_clay_pcts: List of clay percentages (required for each layer)
        layer_theta_vs: List of volumetric water contents 0..1 (required for each layer)
        layer_porosities: Optional list of porosities (None to auto-calculate)
        layer_bulk_densities_gcm3: Optional list of bulk densities in g/cm³
        layer_particle_densities_gcm3: Optional list of particle densities in g/cm³ (default 2.65)
        layer_organic_fractions: Optional list of organic fractions 0..1 (default 0.0)
        layer_salinity_classes: Optional list of salinity classes ('fresh'|'brackish'|'saline')
        layer_porewater_sigmas_Sm: Optional list of porewater conductivities in S/m
        layer_names: Optional list of layer names
        waveform_kind: Waveform type ('ricker' or 'gaussian')
        waveform_amplitude: Waveform amplitude
        waveform_center_freq_hz: Center frequency in Hz
        waveform_name: Waveform name
        antenna_kind: Antenna type ('hertzian_dipole')
        antenna_axis: Antenna axis ('x', 'y', or 'z')
        antenna_tx_rx_offset_m: Transmitter-receiver offset in meters
        model_title: Model title string
        source_height_m: Source height above ground in meters
        domain_xy_m: Domain size (x, y) in meters
        top_air_extra_m: Extra air space at top (optional)
        cells_per_wavelength: Grid resolution (cells per wavelength)
        max_cell_m: Maximum cell size in meters
        rx_same_height: Whether receiver is at same height as transmitter
        temperature_c: Temperature in Celsius
        model: Dielectric model ('crim', 'peplinski', 'dobson', 'mironov')
        enforce_validity: Whether to enforce model validity windows
        salinity_defaults_Sm: Default conductivities for fresh/brackish/saline (S/m)
        output_filename: Output filename for the .in file
    """
    # Validate that all required layer lists have the same length
    num_layers = len(layer_thicknesses_m)
    if not all(len(lst) == num_layers for lst in [
        layer_sand_pcts, layer_silt_pcts, layer_clay_pcts, layer_theta_vs
    ]):
        raise ValueError("All required layer parameter lists must have the same length")
    
    # Build layers list
    layers = []
    for i in range(num_layers):
        layer_kwargs = {
            "thickness_m": layer_thicknesses_m[i],
            "sand_pct": layer_sand_pcts[i],
            "silt_pct": layer_silt_pcts[i],
            "clay_pct": layer_clay_pcts[i],
            "theta_v": layer_theta_vs[i],
        }
        
        # Add optional parameters if provided
        if layer_porosities is not None and i < len(layer_porosities):
            layer_kwargs["porosity"] = layer_porosities[i]
        if layer_bulk_densities_gcm3 is not None and i < len(layer_bulk_densities_gcm3):
            layer_kwargs["bulk_density_gcm3"] = layer_bulk_densities_gcm3[i]
        if layer_particle_densities_gcm3 is not None and i < len(layer_particle_densities_gcm3):
            layer_kwargs["particle_density_gcm3"] = layer_particle_densities_gcm3[i]
        if layer_organic_fractions is not None and i < len(layer_organic_fractions):
            layer_kwargs["organic_fraction"] = layer_organic_fractions[i]
        if layer_salinity_classes is not None and i < len(layer_salinity_classes):
            layer_kwargs["salinity_class"] = layer_salinity_classes[i]
        if layer_porewater_sigmas_Sm is not None and i < len(layer_porewater_sigmas_Sm):
            layer_kwargs["porewater_sigma_Sm"] = layer_porewater_sigmas_Sm[i]
        if layer_names is not None and i < len(layer_names):
            layer_kwargs["name"] = layer_names[i]
        
        layers.append(LayerSpec(**layer_kwargs))
    
    # Create WaveformSpec
    wf = WaveformSpec(
        kind=waveform_kind,
        amplitude=waveform_amplitude,
        center_freq_hz=waveform_center_freq_hz,
        name=waveform_name,
    )
    
    # Create AntennaSpec
    ant = AntennaSpec(
        kind=antenna_kind,
        axis=antenna_axis,
        tx_rx_offset_m=antenna_tx_rx_offset_m,
    )
    
    # Create ModelSpec
    spec = ModelSpec(
        title=model_title,
        layers=layers,
        waveform=wf,
        antenna=ant,
        source_height_m=source_height_m,
        domain_xy_m=domain_xy_m,
        top_air_extra_m=top_air_extra_m,
        cells_per_wavelength=cells_per_wavelength,
        max_cell_m=max_cell_m,
        rx_same_height=rx_same_height,
        temperature_c=temperature_c,
        model=model,
        enforce_validity=enforce_validity,
        salinity_defaults_Sm=salinity_defaults_Sm,
    )
    
    # Build and write the file
    text = spec.build()
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Wrote {output_filename}")

    # # helper
    # eps_r_example, sigma_example = 9.0, 0.02
    # print("Skin depth @1.5 GHz:", skin_depth(eps_r_example, sigma_example, 1.5e9), "m")
    # print("Rmax (Δf=500 MHz):", rmax_unambiguous(5e8, eps_r_example), "m")
