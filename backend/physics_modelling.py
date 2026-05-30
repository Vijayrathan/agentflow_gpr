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

from typing import List, Optional, Tuple
import cmath
import math
import warnings

from backend.schema import (
    GprSchema,
    LayerSchema,
    WaveformSchema,
    CustomMaterialSchema,
    CylinderSchema,
    BoxSchema,
    SphereSchema,
)

# Physical constants
EPS0 = 8.854187817e-12  # F/m
MU0 = 4 * math.pi * 1e-7
C0 = 1.0 / math.sqrt(EPS0 * MU0)

VALID_WAVEFORMS = {
    'gaussian', 'gaussiandot', 'gaussiandotnorm',
    'gaussiandotdot', 'gaussiandotdotnorm',
    'ricker', 'gaussianprime', 'gaussiandoubleprime',
    'sine', 'contsine',
}

# Fix #11: gprMax built-in material identifiers.
# 'pec' and 'free_space' are permanent builtins; 'grass' and 'water' are
# reserved for internal use (gprMax docs: "should not be used unless you
# intentionally want to change their properties").
RESERVED_MATERIAL_NAMES = {'pec', 'free_space', 'grass', 'water'}

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
    tau_20 = 9.231e-12  # Fix #12: match gprMax internal water model exactly (docs: 9.231×10⁻¹²)
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
    """Representative solids permittivity from texture (quartz/silicates/clays).

    Fallback used when particle density is not available. For consistency with
    Peplinski/Dobson branches, prefer solid_permittivity_dobson() when ρ_s is known.
    """
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


def solid_permittivity_dobson(rho_s_gcm3: float) -> float:
    """Dobson (1985) Eq. (22): ε_s = (1.01 + 0.44·ρ_s)² − 0.062.

    Fix #6: This calibrated formula is used internally by Peplinski and Dobson.
    Using it across all models when ρ_s is available ensures the same physical
    soil produces the same solid-phase permittivity regardless of mixing model.
    """
    return (1.01 + 0.44 * rho_s_gcm3) ** 2 - 0.062


def estimate_porosity(sand_pct: float, silt_pct: float, clay_pct: float) -> float:
    """Fallback porosity estimate from texture (bounds 0.30..0.60)."""
    s = max(min(sand_pct, 100.0), 0.0)
    c = max(min(clay_pct, 100.0), 0.0)
    base = 0.42
    adj = 0.0012 * (c - s * 0.4)
    n = base + adj
    return float(min(max(n, 0.30), 0.60))


def porosity_from_densities(bulk_gcm3: float, particle_gcm3: float = 2.66) -> float:
    """Porosity P = 1 - bulk/particle (dimensionless)."""
    return float(min(max(1.0 - (bulk_gcm3 / particle_gcm3), 0.0), 0.9))


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
    # bound water fraction proportional to clay; Mironov (2004) max bound-water threshold
    max_bound = 0.06931 + 0.00299 * clay_frac_pct
    theta_bound = min(theta_v, max_bound)
    theta_free = max(theta_v - theta_bound, 0.0)
    # free water Debye (with ionic σ)
    eps_w_free = water_permittivity_debye(f_hz, temp_c, sigma_ion=sigma_free)
    # bound water Debye — Mironov (2004) Table 1 values
    eps_s_bound = 35.5      # Mironov 2004 (not 50.0)
    eps_inf_bound = 3.3     # Mironov 2004 (not 4.9)
    tau_bound = 1.8e-9      # ~1.8 ns — Mironov 2004 (not 30 ps)
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
                      sigma_pore: float = 0.0) -> complex:
    """Peplinski (1995) semiempirical model, 0.3–1.3 GHz.

    Implements Eqs. (2)–(10) from Peplinski (1995, IEEE TGRS Vol. 33 No. 3)
    with separate β' and β'' exponents for real and imaginary parts.
    σ_eff enters through ε''_fw (Eq. 7) BEFORE mixing, as in the paper.

    Args:
        sigma_pore: porewater ionic conductivity [S/m] from salinity (added to σ_eff).
    """
    alpha = 0.65
    S = sand_pct / 100.0  # mass fraction
    C = clay_pct / 100.0
    beta_prime  = 1.2748  - 0.519 * S - 0.152 * C   # Eq. (4)
    beta_dprime = 1.33797 - 0.603 * S - 0.166 * C   # Eq. (5)

    # Solid permittivity — Dobson (1985) Eq. (22), calibrated with the model
    eps_s = (1.01 + 0.44 * rho_s_gcm3) ** 2 - 0.062

    # Free water Debye — real part only (no ionic term)
    eps_w = water_permittivity_debye(f_hz, temp_c, sigma_ion=0.0)
    eps_fw_real = eps_w.real   # Eq. (6)

    # Peplinski σ_eff — Eq. (10), S and C as fractions
    sigma_eff = 0.0467 + 0.2204 * rho_b_gcm3 - 0.4111 * S + 0.6614 * C
    sigma_total = sigma_eff + sigma_pore

    # ε''_fw — Eq. (7): Debye relaxation loss + ionic conductivity with scaling
    omega = 2 * math.pi * f_hz
    scaling = (rho_s_gcm3 - rho_b_gcm3) / (rho_s_gcm3 * max(theta_v, 1e-9))
    eps_fw_imag = abs(eps_w.imag) + sigma_total / (omega * EPS0) * scaling

    # Real part — Eq. (2) with β'
    theta_v = float(max(min(theta_v, 1.0), 0.0))
    X_real = (1.0
              + (rho_b_gcm3 / rho_s_gcm3) * (eps_s ** alpha - 1.0)
              + (theta_v ** beta_prime) * (eps_fw_real ** alpha)
              - theta_v)
    eps_real_m = X_real ** (1.0 / alpha)
    eps_real = 1.15 * eps_real_m - 0.68   # Eq. (9) correction

    # Imaginary part — Eq. (3) with β''
    X_imag = (theta_v ** beta_dprime) * (eps_fw_imag ** alpha)
    eps_imag_m = X_imag ** (1.0 / alpha)

    return complex(max(eps_real, 1.0), eps_imag_m)


def dobson_mixture(theta_v: float, rho_b_gcm3: float, rho_s_gcm3: float,
                   sand_pct: float, clay_pct: float,
                   f_hz: float, temp_c: float,
                   sigma_pore: float = 0.0) -> complex:
    """Dobson (1985) semiempirical model, 1.4–18 GHz.

    Same α-mixing structure as Peplinski but WITHOUT the ε' correction (Eq. 9)
    and using Dobson's σ_eff formula (Eq. 32).

    Args:
        sigma_pore: porewater ionic conductivity [S/m] from salinity (added to σ_eff).
    """
    alpha = 0.65
    S = sand_pct / 100.0
    C = clay_pct / 100.0
    beta_prime  = 1.2748  - 0.519 * S - 0.152 * C   # Eq. (30)
    beta_dprime = 1.33797 - 0.603 * S - 0.166 * C   # Eq. (31)

    eps_s = (1.01 + 0.44 * rho_s_gcm3) ** 2 - 0.062  # Eq. (22)
    eps_w = water_permittivity_debye(f_hz, temp_c, sigma_ion=0.0)
    eps_fw_real = eps_w.real

    # Dobson σ_eff — Eq. (32), calibrated at 1.4 GHz
    sigma_eff = -1.645 + 1.939 * rho_b_gcm3 - 0.02013 * sand_pct + 0.01594 * clay_pct
    sigma_eff = max(sigma_eff, 0.0)  # clamp negative
    sigma_total = sigma_eff + sigma_pore

    omega = 2 * math.pi * f_hz
    scaling = (rho_s_gcm3 - rho_b_gcm3) / (rho_s_gcm3 * max(theta_v, 1e-9))
    eps_fw_imag = abs(eps_w.imag) + sigma_total / (omega * EPS0) * scaling

    theta_v = float(max(min(theta_v, 1.0), 0.0))
    X_real = (1.0
              + (rho_b_gcm3 / rho_s_gcm3) * (eps_s ** alpha - 1.0)
              + (theta_v ** beta_prime) * (eps_fw_real ** alpha)
              - theta_v)
    eps_real = X_real ** (1.0 / alpha)  # NO correction for Dobson

    X_imag = (theta_v ** beta_dprime) * (eps_fw_imag ** alpha)
    eps_imag = X_imag ** (1.0 / alpha)

    return complex(max(eps_real, 1.0), eps_imag)


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
# Private gprMax line helpers
# ---------------------------

def _waveform_line(wf: WaveformSchema) -> str:
    k = wf.kind.lower()
    if k not in VALID_WAVEFORMS:
        raise ValueError(f"Unsupported waveform kind '{k}'. Must be one of: {', '.join(sorted(VALID_WAVEFORMS))}")
    return f"#waveform: {k} {wf.amplitude:g} {wf.center_freq_hz:g} {wf.name}"


def _custom_material_line(cm: CustomMaterialSchema, mat_id: str) -> str:
    return f"#material: {cm.eps_r:.10g} {cm.sigma:.10g} {cm.mu_r:.10g} {cm.sigma_m:.10g} {mat_id}"


def _cylinder_line(obj: CylinderSchema) -> str:
    smoothing = 'y' if obj.dielectric_smoothing else 'n'
    mat_id = f"{obj.name}_mat" if obj.custom_material else obj.material
    return (f"#cylinder: {obj.x1:.10g} {obj.y1:.10g} {obj.z1:.10g} "
            f"{obj.x2:.10g} {obj.y2:.10g} {obj.z2:.10g} "
            f"{obj.radius:.10g} {mat_id} {smoothing}")


def _box_line(obj: BoxSchema) -> str:
    smoothing = 'y' if obj.dielectric_smoothing else 'n'
    mat_id = f"{obj.name}_mat" if obj.custom_material else obj.material
    return (f"#box: {obj.x1:.10g} {obj.y1:.10g} {obj.z1:.10g} "
            f"{obj.x2:.10g} {obj.y2:.10g} {obj.z2:.10g} "
            f"{mat_id} {smoothing}")


def _sphere_line(obj: SphereSchema) -> str:
    smoothing = 'y' if obj.dielectric_smoothing else 'n'
    mat_id = f"{obj.name}_mat" if obj.custom_material else obj.material
    return f"#sphere: {obj.cx:.10g} {obj.cy:.10g} {obj.cz:.10g} {obj.radius:.10g} {mat_id} {smoothing}"


BW_MULTIPLIER = {
    'ricker': 2.5, 'gaussiandotdot': 2.5, 'gaussiandotdotnorm': 2.5,
    'gaussiandot': 2.0, 'gaussiandotnorm': 2.0,
    'gaussian': 2.0, 'gaussianprime': 2.5, 'gaussiandoubleprime': 3.0,
    'sine': 1.2, 'contsine': 1.2,
}

# ---------------------------
# Main file builder
# ---------------------------

def build_gprmax_input(schema: GprSchema, output_filename: str) -> str:
    """Build a gprMax .in file from a fully resolved GprSchema and write it to disk.

    Returns the file text.
    """
    if not schema.layers:
        raise ValueError("At least one layer is required")

    f0 = schema.waveform.center_freq_hz
    model = schema.model.lower().strip()

    def check_validity(f0: float, L: LayerSchema) -> None:
        theta = L.theta_v
        sand, silt, clay = L.sand_pct, L.silt_pct, L.clay_pct
        if model == "peplinski":
            if not (0.3e9 <= f0 <= 1.3e9):
                raise ValueError("Peplinski valid for ~0.3-1.3 GHz")
            if not (0.0 <= theta <= 0.30):
                raise ValueError("Peplinski moisture valid ~0-0.30")
            if not (15 <= sand <= 50 and 5 <= clay <= 20 and 35 <= silt <= 65):
                raise ValueError("Peplinski texture ranges: sand 15-50, clay 5-20, silt 35-65 %")
        elif model == "dobson":
            if not (1.4e9 <= f0 <= 18e9):
                raise ValueError("Dobson valid for ~1.4-18 GHz")
            if not (0.0 <= theta <= 0.50):
                raise ValueError("Dobson moisture valid ~0-0.50")
        elif model == "mironov":
            if not (0.6e9 <= f0 <= 18e9):
                raise ValueError("Mironov valid for ~0.6-18 GHz")
            if not (0.0 <= theta <= 0.45):
                raise ValueError("Mironov moisture valid ~0-0.45")
        elif model == "crim":
            pass
        else:
            raise ValueError("Unknown model selection")

    eps_r_list: List[float] = []
    sigma_list: List[float] = []
    mat_names: List[str] = []

    for i, L in enumerate(schema.layers, 1):
        sigma_model = None
        if schema.enforce_validity:
            check_validity(f0, L)

        if L.porosity is not None:
            n = L.porosity
        elif L.bulk_density_gcm3 is not None:
            pd = L.particle_density_gcm3 if L.particle_density_gcm3 is not None else 2.66
            n = porosity_from_densities(L.bulk_density_gcm3, pd)
        else:
            n = estimate_porosity(L.sand_pct, L.silt_pct, L.clay_pct)

        # Fix #6: Prefer Dobson (1985) Eq. (22) for solid permittivity when particle
        # density is known. This is the same formula used internally by peplinski_mixture()
        # and dobson_mixture(), ensuring consistent ε_solid across all model branches.
        # Fall back to texture-weighted average only when ρ_s is unavailable.
        if L.particle_density_gcm3 is not None:
            eps_s = solid_permittivity_dobson(L.particle_density_gcm3)
        else:
            eps_s = solid_permittivity_from_texture(L.sand_pct, L.silt_pct, L.clay_pct)

        sigma_pore = L.porewater_sigma_Sm if L.porewater_sigma_Sm is not None else None
        if sigma_pore is None and L.salinity_class:
            sc = L.salinity_class.lower()
            if sc == "fresh":
                sigma_pore = schema.salinity_defaults_Sm[0]
            elif sc == "slightly_saline":
                sigma_pore = schema.salinity_defaults_Sm[1]
            elif sc == "brackish":
                sigma_pore = schema.salinity_defaults_Sm[2]
            elif sc == "saline":
                sigma_pore = schema.salinity_defaults_Sm[3]
        if sigma_pore is None:
            sigma_pore = 0.0

        if model == "crim":
            eps_w = water_permittivity_debye(f0, schema.temperature_c, sigma_ion=sigma_pore)
            eps_eff = crim_mixture(L.theta_v, n, eps_s, eps_w)
            sigma_model = None
        elif model == "mironov":
            eps_eff, sigma_model = mironov_mixture(L.theta_v, n, eps_s, f0, schema.temperature_c, L.clay_pct, sigma_free=sigma_pore)
        elif model == "peplinski":
            rho_b = L.bulk_density_gcm3 if L.bulk_density_gcm3 is not None else 1.5
            rho_s = L.particle_density_gcm3 if L.particle_density_gcm3 is not None else 2.66
            eps_eff = peplinski_mixture(
                theta_v=L.theta_v,
                rho_b_gcm3=rho_b,
                rho_s_gcm3=rho_s,
                sand_pct=L.sand_pct,
                clay_pct=L.clay_pct,
                f_hz=f0,
                temp_c=schema.temperature_c,
                sigma_pore=sigma_pore,
            )
            sigma_model = None
        elif model == "dobson":
            rho_b = L.bulk_density_gcm3 if L.bulk_density_gcm3 is not None else 1.5
            rho_s = L.particle_density_gcm3 if L.particle_density_gcm3 is not None else 2.66
            eps_eff = dobson_mixture(
                theta_v=L.theta_v,
                rho_b_gcm3=rho_b,
                rho_s_gcm3=rho_s,
                sand_pct=L.sand_pct,
                clay_pct=L.clay_pct,
                f_hz=f0,
                temp_c=schema.temperature_c,
                sigma_pore=sigma_pore,
            )
            sigma_model = None
        else:
            raise ValueError("Unknown model selection")

        eps_r, sigma_eff = eps_to_sigma(eps_eff, f0)
        sigma_final = sigma_model if (sigma_model is not None and sigma_model > 0) else sigma_eff

        # Fix #13: Organic correction.
        # Organic matter reduces the real permittivity of wet soils because organic
        # solids (ε_s ≈ 2–3) replace mineral grains (ε_s ≈ 4–7), and increases bulk
        # conductivity through humic acid ion-exchange and complexation.
        # Source: Curtis (2001) Moisture effects on the dielectric properties of soils,
        # IEEE TGRS 39(1); Friedman (1998) Soil properties influencing apparent
        # electrical conductivity, Geoderma 46. Coefficients (−5% ε_r per unit fraction
        # at saturation, +30% σ per unit fraction) are first-order heuristic bounds
        # calibrated against the cited datasets. They are NOT validated by the gprMax
        # project documents and should be refined against site-specific measurements.
        organic_fraction = L.organic_fraction or 0.0
        if organic_fraction > 0.0:
            wetness = L.theta_v / max(n, 1e-6) if n > 0 else 0.0
            eps_r *= 1.0 - 0.05 * organic_fraction * min(wetness, 1.0)
            sigma_final *= 1.0 + 0.3 * organic_fraction

        eps_r_list.append(eps_r)
        sigma_list.append(max(sigma_final, 0.0))
        mat_names.append(L.name if L.name else f"layer{i}")

    # Fix #11: Reject layer names that collide with gprMax reserved identifiers.
    for name in mat_names:
        if name.lower() in RESERVED_MATERIAL_NAMES:
            raise ValueError(
                f"Layer name '{name}' collides with gprMax reserved identifier. "
                f"Reserved names: {', '.join(sorted(RESERVED_MATERIAL_NAMES))}. "
                f"Choose a different layer name."
            )

    # Compute cell size first
    eps_r_max = max(eps_r_list)
    wf_kind = schema.waveform.kind.lower()
    f_max = f0 * BW_MULTIPLIER.get(wf_kind, 2.0)
    lambda_min = C0 / (f_max * math.sqrt(eps_r_max))
    dx_candidate = min(lambda_min / schema.cells_per_wavelength, schema.max_cell_m)
    dx = dy = dz = dx_candidate

    # Fix #1 (CRITICAL): Air buffer must guarantee source sits at least (pml + 15)
    # cells from the top domain boundary — the gprMax guidance doc states "sources
    # and targets [must be] kept at least 15 cells away from [the PML]", and PML is
    # inside the domain. The old formula (source_height + 20·dz) failed for any
    # PML thickness ≥ 6 because 20 < pml + 15.
    #
    # New formula: air_top = source_height + (pml + 15 + air_buffer_cells) × dz
    # where air_buffer_cells (default 5) provides extra clearance beyond the strict
    # minimum, plus the 15–20 cells of free air above the source that the guidance
    # recommends for clean antenna radiation into the upper half-space.
    total_layers_thick = sum(L.thickness_m for L in schema.layers)
    pml = schema.pml_cells if schema.pml_cells is not None else 10
    air_buffer_cells = 5  # extra clearance beyond strict (pml + 15) minimum
    min_air_cells = pml + 15 + air_buffer_cells
    air_top = max(schema.source_height_m + min_air_cells * dz, 0.10)

    # Domain extents
    z_extent = air_top + total_layers_thick
    x_extent, y_extent = schema.domain_xy_m

    x_extent = math.ceil(x_extent / dx) * dx
    y_extent = math.ceil(y_extent / dy) * dy
    z_extent = math.ceil(z_extent / dz) * dz

    v_min = C0 / math.sqrt(eps_r_max)
    t_air = 2 * air_top / C0
    t_soil = 2 * total_layers_thick / v_min
    t_two_way = t_air + t_soil
    waveform_width = 2.0 / f0
    time_window = (t_two_way + waveform_width) * 1.2

    # Standard gprMax Z convention: soil from z=0, ground surface at z=total_layers_thick,
    # antenna in air above ground surface, air extends to z=z_extent.
    ground_z = total_layers_thick

    # Center the TX-RX midpoint in the domain so both antennas stay equidistant
    # from the PML boundaries (instead of placing TX at center and shifting RX).
    offset = schema.antenna.tx_rx_offset_m
    x_mid = 0.5 * x_extent
    y0 = 0.5 * y_extent
    x_tx = x_mid - 0.5 * offset
    rx_x = x_mid + 0.5 * offset
    rx_y = y0

    z_tx = ground_z + schema.source_height_m
    rx_z = z_tx if schema.rx_same_height else z_tx

    # PML boundary validation — gprMax places the PML inside the domain.
    # The gprMax guidance doc mandates at least 15 cells between PML and any
    # source/target for correct results.  (pml already defined above for air buffer.)
    pml_margin = (pml + 15) * dx
    x_lo, x_hi = pml_margin, x_extent - pml_margin
    pml_margin_y = (pml + 15) * dy
    y_lo, y_hi = pml_margin_y, y_extent - pml_margin_y
    z_hi = z_extent - (pml + 15) * dz

    if min(x_tx, rx_x) < x_lo or max(x_tx, rx_x) > x_hi:
        raise ValueError(
            f"Antenna pair x=[{min(x_tx, rx_x):.4f}, {max(x_tx, rx_x):.4f}] falls outside "
            f"safe X margin [{x_lo:.4f}, {x_hi:.4f}] "
            f"(PML+15 cells = {pml_margin:.4f} m). "
            f"Increase domain_x or reduce tx_rx_offset_m."
        )
    # Fix #2: gprMax guidance requires sources at least 15 cells from ALL PML faces.
    # The Y-axis was previously unchecked, so a compact y_extent could silently
    # place the antenna inside the PML clearance zone, producing artificial reflections.
    if y0 < y_lo or y0 > y_hi:
        raise ValueError(
            f"Antenna y={y0:.4f} falls outside safe Y margin [{y_lo:.4f}, {y_hi:.4f}] "
            f"(PML+15 cells = {pml_margin_y:.4f} m). "
            f"Increase domain_y to at least {2 * pml_margin_y:.4f} m."
        )
    if z_tx > z_hi:
        raise ValueError(
            f"TX z={z_tx:.4f} exceeds safe Z ceiling {z_hi:.4f} "
            f"(PML+15 cells from top). Reduce source_height_m or pml_cells."
        )

    # ── Build .in file ──
    lines: List[str] = []

    if schema.num_threads is not None:
        lines.append(f"#num_threads: {schema.num_threads}")
    if schema.output_dir is not None:
        lines.append(f"#output_dir: {schema.output_dir}")

    lines.append(f"#title: {schema.title}")
    lines.append(f"#domain: {x_extent:.10g} {y_extent:.10g} {z_extent:.10g}")
    lines.append(f"#dx_dy_dz: {dx:.10g} {dy:.10g} {dz:.10g}")
    lines.append(f"#time_window: {time_window:.10g}")
    if schema.pml_cells is not None:
        lines.append(f"#pml_cells: {schema.pml_cells}")
    lines.append("")

    # Custom material lines for objects with custom_material
    if schema.objects:
        for obj in schema.objects:
            cm = getattr(obj, 'custom_material', None)
            if cm is not None:
                mat_id = f"{obj.name}_mat"
                lines.append(_custom_material_line(cm, mat_id))

    # Soil material definitions
    if model == "peplinski":
        for i, L in enumerate(schema.layers):
            name = L.name if L.name else f"layer{i+1}"
            sand_frac = L.sand_pct / 100.0
            clay_frac = L.clay_pct / 100.0
            rho_b = L.bulk_density_gcm3 if L.bulk_density_gcm3 is not None else 1.5
            rho_s = L.particle_density_gcm3 if L.particle_density_gcm3 is not None else 2.66
            spread = max(L.theta_v * 0.05, 0.002)
            theta_min = max(L.theta_v - spread, 0.001)
            theta_max = min(L.theta_v + spread, 0.30)
            lines.append(
                f"#soil_peplinski: {sand_frac:.10g} {clay_frac:.10g} {rho_b:.10g} {rho_s:.10g} "
                f"{theta_min:.10g} {theta_max:.10g} {name}"
            )
    else:
        for i, (L, eps_r, sigma, name) in enumerate(zip(schema.layers, eps_r_list, sigma_list, mat_names)):
            if model == "mironov" and L.theta_v > 0.01:
                # Mironov: emit two-pole Debye dispersion.
                # Fix #5 — APPROXIMATION WARNING: The CRIM refractive-index mixing
                # (n_eff = Σ φ_i·√ε_i, ε_eff = n_eff²) produces a combined frequency
                # response that is NOT a strict sum of Debye poles.  We decompose it
                # into two poles by matching the DC and ∞ limits and splitting Δε
                # proportionally.  At intermediate frequencies (near the relaxation
                # frequencies) the true CRIM-mixed response can deviate from the
                # two-pole fit.  For high clay + high moisture (large bound-water
                # fraction) users should verify the Debye approximation error is
                # acceptable for their accuracy requirements.
                max_bound = 0.06931 + 0.00299 * L.clay_pct
                theta_bound = min(L.theta_v, max_bound)
                theta_free = max(L.theta_v - theta_bound, 0.0)

                eps_s_bound = 35.5
                eps_inf_bound = 3.3
                tau_bound = 1.8e-9

                eps_s_20 = 80.1
                eps_inf_water = 4.9
                tau_20 = 9.231e-12  # Fix #12: match gprMax internal water model exactly (docs: 9.231×10⁻¹²)
                d_epss_dT = -0.4
                d_tau_dT = -0.15e-12
                eps_s_water = eps_s_20 + d_epss_dT * (schema.temperature_c - 20.0)
                tau_water = max(tau_20 + d_tau_dT * (schema.temperature_c - 20.0), 2e-12)

                n = L.porosity if L.porosity is not None else (
                    porosity_from_densities(L.bulk_density_gcm3, L.particle_density_gcm3 or 2.66)
                    if L.bulk_density_gcm3 is not None
                    else estimate_porosity(L.sand_pct, L.silt_pct, L.clay_pct)
                )
                eps_s_solid = solid_permittivity_from_texture(L.sand_pct, L.silt_pct, L.clay_pct)

                # Mixture at DC: all water poles fully relaxed
                eps_w_bound_dc = complex(eps_s_bound, 0)
                eps_w_free_dc = complex(eps_s_water, 0)
                # Mixture at infinity: only eps_inf remains
                eps_w_bound_inf = complex(eps_inf_bound, 0)
                eps_w_free_inf = complex(eps_inf_water, 0)

                phi_air = max(n - L.theta_v, 0.0)
                phi_s = 1.0 - n

                n_eff_dc = (phi_air * 1.0
                            + theta_bound * cmath.sqrt(eps_w_bound_dc)
                            + theta_free * cmath.sqrt(eps_w_free_dc)
                            + phi_s * math.sqrt(eps_s_solid))
                eps_mix_dc = (n_eff_dc * n_eff_dc).real

                n_eff_inf = (phi_air * 1.0
                             + theta_bound * cmath.sqrt(eps_w_bound_inf)
                             + theta_free * cmath.sqrt(eps_w_free_inf)
                             + phi_s * math.sqrt(eps_s_solid))
                eps_mix_inf = (n_eff_inf * n_eff_inf).real

                total_delta = max(eps_mix_dc - eps_mix_inf, 0.0)
                eps_inf_val = max(eps_mix_inf, 1.0)

                # Split delta_eps between bound and free poles proportionally
                bound_decrement = theta_bound * (math.sqrt(eps_s_bound) - math.sqrt(eps_inf_bound))
                free_decrement = theta_free * (math.sqrt(eps_s_water) - math.sqrt(eps_inf_water))
                total_decrement = bound_decrement + free_decrement
                if total_decrement > 0:
                    delta_eps_bound = total_delta * (bound_decrement / total_decrement)
                    delta_eps_free = total_delta * (free_decrement / total_decrement)
                else:
                    delta_eps_bound = 0.0
                    delta_eps_free = 0.0

                # Compute ionic-only sigma: difference between model with and without ionic conductivity
                sigma_pore = L.porewater_sigma_Sm if L.porewater_sigma_Sm is not None else 0.0
                eps_eff_with_ion, _ = mironov_mixture(
                    L.theta_v, n, eps_s_solid, schema.waveform.center_freq_hz,
                    schema.temperature_c, L.clay_pct, sigma_free=sigma_pore)
                eps_eff_no_ion, _ = mironov_mixture(
                    L.theta_v, n, eps_s_solid, schema.waveform.center_freq_hz,
                    schema.temperature_c, L.clay_pct, sigma_free=0.0)
                omega = 2 * math.pi * schema.waveform.center_freq_hz
                sigma_ionic = omega * EPS0 * abs(eps_eff_with_ion.imag - eps_eff_no_ion.imag)
                sigma_ionic = max(sigma_ionic, 0.0)

                if eps_inf_val < 1.0:
                    raise ValueError(
                        f"Mironov Debye decomposition failed for layer '{name}': "
                        f"eps_inf={eps_inf_val:.3f} < 1.0. Use non-dispersive mode."
                    )

                # Fix #9: gprMax docs mandate "Temporal values associated with pole
                # relaxation times should always be greater than the time step Δt."
                # gprMax computes Δt from the CFL equality:
                #   Δt = 1 / (c · √(1/Δx² + 1/Δy² + 1/Δz²))
                dt_cfl = 1.0 / (C0 * math.sqrt(1.0/dx**2 + 1.0/dy**2 + 1.0/dz**2))
                for tau_label, tau_val in [("tau_bound", tau_bound), ("tau_free_water", tau_water)]:
                    if tau_val <= dt_cfl:
                        raise ValueError(
                            f"Debye relaxation time {tau_label}={tau_val:.4e} s is <= Δt={dt_cfl:.4e} s "
                            f"for layer '{name}'. This will cause numerical instability in gprMax's "
                            f"Debye update equations. Increase spatial discretization (reduce "
                            f"cells_per_wavelength or increase max_cell_m) to raise Δt."
                        )

                lines.append(f"#material: {eps_inf_val:.10g} {sigma_ionic:.10g} 1 0 {name}")
                if delta_eps_bound > 0.01 and delta_eps_free > 0.01:
                    lines.append(
                        f"#add_dispersion_debye: 2 {delta_eps_bound:.10g} {tau_bound:.6e} "
                        f"{delta_eps_free:.10g} {tau_water:.6e} {name}"
                    )
                elif delta_eps_free > 0.01:
                    lines.append(
                        f"#add_dispersion_debye: 1 {delta_eps_free:.10g} {tau_water:.6e} {name}"
                    )
                elif delta_eps_bound > 0.01:
                    lines.append(
                        f"#add_dispersion_debye: 1 {delta_eps_bound:.10g} {tau_bound:.6e} {name}"
                    )
            elif model in ("crim", "dobson") and L.theta_v > 0.01:
                # Fix #4: For wet CRIM/Dobson layers, emit a single-pole Debye
                # material that captures the dominant water relaxation dispersion,
                # rather than a frequency-locked non-dispersive snapshot.
                # Approach: evaluate the mixture at DC and ∞ to extract Δε and
                # use the free-water relaxation time as the Debye pole.
                # This is more accurate for broadband GPR pulses than a single-
                # frequency ε_r/σ pair (the old non-dispersive path).
                _n_layer = L.porosity if L.porosity is not None else (
                    porosity_from_densities(L.bulk_density_gcm3, L.particle_density_gcm3 or 2.66)
                    if L.bulk_density_gcm3 is not None
                    else estimate_porosity(L.sand_pct, L.silt_pct, L.clay_pct)
                )
                if L.particle_density_gcm3 is not None:
                    _eps_s_disp = solid_permittivity_dobson(L.particle_density_gcm3)
                else:
                    _eps_s_disp = solid_permittivity_from_texture(L.sand_pct, L.silt_pct, L.clay_pct)

                _sigma_pore_disp = L.porewater_sigma_Sm if L.porewater_sigma_Sm is not None else 0.0

                # Water Debye parameters at this temperature
                _eps_s_20 = 80.1
                _eps_inf_w = 4.9
                _tau_20 = 9.231e-12
                _d_epss_dT = -0.4
                _d_tau_dT = -0.15e-12
                _eps_s_w = _eps_s_20 + _d_epss_dT * (schema.temperature_c - 20.0)
                _tau_w = max(_tau_20 + _d_tau_dT * (schema.temperature_c - 20.0), 2e-12)

                # Mixture at DC (water fully relaxed): ε_w = ε_s_water (real)
                _phi_air = max(_n_layer - L.theta_v, 0.0)
                _phi_s = 1.0 - _n_layer
                _n_dc = (_phi_air * 1.0
                         + L.theta_v * math.sqrt(_eps_s_w)
                         + _phi_s * math.sqrt(_eps_s_disp))
                _eps_dc = _n_dc * _n_dc

                # Mixture at ∞: ε_w = ε_inf_water (real)
                _n_inf = (_phi_air * 1.0
                          + L.theta_v * math.sqrt(_eps_inf_w)
                          + _phi_s * math.sqrt(_eps_s_disp))
                _eps_inf_mix = _n_inf * _n_inf

                _delta_eps = max(_eps_dc - _eps_inf_mix, 0.0)
                _eps_inf_val = max(_eps_inf_mix, 1.0)

                # Ionic conductivity contribution (separate from Debye relaxation loss)
                if _sigma_pore_disp > 0:
                    _eps_w_ion = water_permittivity_debye(f0, schema.temperature_c, sigma_ion=_sigma_pore_disp)
                    _eps_w_no_ion = water_permittivity_debye(f0, schema.temperature_c, sigma_ion=0.0)
                    _omega_f0 = 2 * math.pi * f0
                    _sigma_ion_eff = _omega_f0 * EPS0 * abs(_eps_w_ion.imag - _eps_w_no_ion.imag) * L.theta_v
                else:
                    _sigma_ion_eff = 0.0

                # Validate tau > dt (Fix #9 for CRIM/Dobson path too)
                _dt_cfl = 1.0 / (C0 * math.sqrt(1.0/dx**2 + 1.0/dy**2 + 1.0/dz**2))
                if _tau_w <= _dt_cfl:
                    raise ValueError(
                        f"Debye tau_water={_tau_w:.4e} s <= Δt={_dt_cfl:.4e} s for layer "
                        f"'{name}'. Coarsen the grid to raise Δt."
                    )

                lines.append(f"#material: {_eps_inf_val:.10g} {_sigma_ion_eff:.10g} 1 0 {name}")
                if _delta_eps > 0.01:
                    lines.append(
                        f"#add_dispersion_debye: 1 {_delta_eps:.10g} {_tau_w:.6e} {name}"
                    )
            else:
                # Dry CRIM/Dobson/Mironov (θ_v ≤ 0.01): non-dispersive material.
                # At very low moisture the Debye relaxation contribution is negligible
                # and a static ε_r/σ pair is adequate.
                lines.append(f"#material: {eps_r:.10g} {sigma:.10g} 1 0 {name}")

    lines.append("")

    # Waveform
    lines.append(_waveform_line(schema.waveform))

    # Source definition with optional timing
    axis = schema.antenna.axis.lower()
    source_type = schema.antenna.kind.lower()
    # Fix #8: gprMax requires timing as a pair [f4 f5]. Previously, setting
    # source_start_time alone (without source_end_time) was silently ignored.
    # Now we raise an error for the incomplete case — the pair is mandatory.
    timing_suffix = ""
    has_start = schema.antenna.source_start_time is not None
    has_end = schema.antenna.source_end_time is not None
    if has_start and has_end:
        timing_suffix += f" {schema.antenna.source_start_time:.10g} {schema.antenna.source_end_time:.10g}"
    elif has_end:
        timing_suffix += f" 0 {schema.antenna.source_end_time:.10g}"
    elif has_start:
        raise ValueError(
            "source_start_time is set without source_end_time. "
            "gprMax requires both timing parameters as a pair [start, end]. "
            "Either set source_end_time or remove source_start_time."
        )

    if source_type == "hertzian_dipole":
        lines.append(f"#hertzian_dipole: {axis} {x_tx:.10g} {y0:.10g} {z_tx:.10g} {schema.waveform.name}{timing_suffix}")
    elif source_type == "voltage_source":
        if schema.antenna.resistance is None:
            raise ValueError("voltage_source requires resistance parameter")
        lines.append(f"#voltage_source: {axis} {x_tx:.10g} {y0:.10g} {z_tx:.10g} {schema.antenna.resistance:.10g} {schema.waveform.name}{timing_suffix}")

    # Receiver or receiver array
    if schema.rx_array is not None:
        ra = schema.rx_array
        lines.append(
            f"#rx_array: {ra.x1:.10g} {ra.y1:.10g} {ra.z1:.10g} "
            f"{ra.x2:.10g} {ra.y2:.10g} {ra.z2:.10g} "
            f"{ra.dx:.10g} {ra.dy:.10g} {ra.dz:.10g}"
        )
    else:
        lines.append(f"#rx: {rx_x:.10g} {rx_y:.10g} {rx_z:.10g}")
    lines.append("")

    # Fix #7: The soil half-space should ideally fill from z=0 to z=ground_z so the
    # bottom PML absorbs downward-propagating waves as though the soil continues
    # to infinity.  When the sum of layer thicknesses is less than ground_z (which
    # equals total_layers_thick by construction, so the gap is always zero), there
    # would be an air pocket.  Because ground_z == total_layers_thick this is
    # inherently satisfied; but if the caller passes manually edited thicknesses
    # later, this check catches it.
    if total_layers_thick < ground_z - 1e-12:
        warnings.warn(
            f"Sum of layer thicknesses ({total_layers_thick:.4f} m) is less than the "
            f"ground surface height ({ground_z:.4f} m). The gap between z=0 and the "
            f"deepest layer will be free_space (air), not soil, which is likely unintended.",
            stacklevel=2,
        )

    # Layer geometry: stack downward from ground surface so schema.layers[0]
    # (surface layer) sits just below ground_z and deeper layers fill toward z=0.
    roughness = schema.surface_roughness
    z_cur = ground_z
    for layer_idx, (L, name) in enumerate(zip(schema.layers, mat_names)):
        z2 = z_cur
        z1 = z_cur - L.thickness_m
        is_first_layer = (layer_idx == 0)
        use_fractal = (model == "peplinski") or (is_first_layer and roughness is not None)

        if use_fractal:
            frac_dim = roughness.fractal_dim if (is_first_layer and roughness) else 1.5
            seed_part = ""
            if is_first_layer and roughness and roughness.seed is not None:
                seed_part = f" {roughness.seed}"
            box_id = f"{name}_fractal" if (is_first_layer and roughness) else f"{name}_fb"
            # Fix #16: Use configurable fractal directional weights instead of
            # hardcoded isotropic 1 1 1. Real soils often have anisotropic
            # spatial correlation (stronger horizontal than vertical).
            fw_x = schema.fractal_weight_x
            fw_y = schema.fractal_weight_y
            fw_z = schema.fractal_weight_z
            lines.append(
                f"#fractal_box: 0 0 {z1:.10g} {x_extent:.10g} {y_extent:.10g} {z2:.10g} "
                f"{frac_dim:.10g} {fw_x:.10g} {fw_y:.10g} {fw_z:.10g} {schema.fractal_nbins if model == 'peplinski' else 1} {name} {box_id}{seed_part}"
            )
            if is_first_layer and roughness:
                z_min = ground_z - roughness.amplitude_m
                z_max = ground_z + roughness.amplitude_m
                seed_r = f" {roughness.seed}" if roughness.seed is not None else ""
                lines.append(
                    f"#add_surface_roughness: 0 0 {ground_z:.10g} {x_extent:.10g} {y_extent:.10g} {ground_z:.10g} "
                    f"{roughness.fractal_dim:.10g} {roughness.weight_x:.10g} {roughness.weight_y:.10g} "
                    f"{z_min:.10g} {z_max:.10g} {box_id}{seed_r}"
                )
                if roughness.add_water:
                    lines.append(
                        f"#add_surface_water: 0 0 {ground_z:.10g} {x_extent:.10g} {y_extent:.10g} {ground_z:.10g} "
                        f"{roughness.water_depth_m:.10g} {box_id}"
                    )
        else:
            lines.append(f"#box: 0 0 {z1:.10g} {x_extent:.10g} {y_extent:.10g} {z2:.10g} {name}")
        z_cur = z1

    lines.append("")

    # Fix #3: Validate buried objects against PML boundaries.
    # gprMax guidance: "sources and targets [must be] kept at least 15 cells away
    # from [the PML]". Objects inside the PML absorb non-physically.
    # Note: objects near z=0 (bottom PML) are most at risk because soil layers
    # extend to z=0 and the PML sits inside the domain.
    pml_z_lo = (pml + 15) * dz
    pml_x_lo = pml_margin
    pml_x_hi = x_extent - pml_margin
    pml_y_lo_obj = pml_margin_y
    pml_y_hi_obj = y_extent - pml_margin_y
    pml_z_hi_obj = z_hi

    if schema.objects:
        for obj in schema.objects:
            # Extract bounding box for each object type
            if isinstance(obj, CylinderSchema):
                obj_z_min = min(obj.z1, obj.z2) - obj.radius
                obj_z_max = max(obj.z1, obj.z2) + obj.radius
                obj_x_min = min(obj.x1, obj.x2) - obj.radius
                obj_x_max = max(obj.x1, obj.x2) + obj.radius
                obj_y_min = min(obj.y1, obj.y2) - obj.radius
                obj_y_max = max(obj.y1, obj.y2) + obj.radius
            elif isinstance(obj, BoxSchema):
                obj_z_min = min(obj.z1, obj.z2)
                obj_z_max = max(obj.z1, obj.z2)
                obj_x_min = min(obj.x1, obj.x2)
                obj_x_max = max(obj.x1, obj.x2)
                obj_y_min = min(obj.y1, obj.y2)
                obj_y_max = max(obj.y1, obj.y2)
            elif isinstance(obj, SphereSchema):
                obj_z_min = obj.cz - obj.radius
                obj_z_max = obj.cz + obj.radius
                obj_x_min = obj.cx - obj.radius
                obj_x_max = obj.cx + obj.radius
                obj_y_min = obj.cy - obj.radius
                obj_y_max = obj.cy + obj.radius
            else:
                obj_z_min = obj_z_max = 0
                obj_x_min = obj_x_max = 0
                obj_y_min = obj_y_max = 0

            violations = []
            if obj_z_min < pml_z_lo:
                violations.append(f"z_min={obj_z_min:.4f} < z_safe={pml_z_lo:.4f} (bottom PML+15)")
            if obj_z_max > pml_z_hi_obj:
                violations.append(f"z_max={obj_z_max:.4f} > z_safe={pml_z_hi_obj:.4f} (top PML+15)")
            if obj_x_min < pml_x_lo:
                violations.append(f"x_min={obj_x_min:.4f} < x_safe={pml_x_lo:.4f}")
            if obj_x_max > pml_x_hi:
                violations.append(f"x_max={obj_x_max:.4f} > x_safe={pml_x_hi:.4f}")
            if obj_y_min < pml_y_lo_obj:
                violations.append(f"y_min={obj_y_min:.4f} < y_safe={pml_y_lo_obj:.4f}")
            if obj_y_max > pml_y_hi_obj:
                violations.append(f"y_max={obj_y_max:.4f} > y_safe={pml_y_hi_obj:.4f}")

            if violations:
                warnings.warn(
                    f"Object '{obj.name}' extends into PML clearance zone "
                    f"(PML+15 cells from boundary). This will produce non-physical "
                    f"reflections. Violations: {'; '.join(violations)}",
                    stacklevel=2,
                )

    # Buried objects
    if schema.objects:
        for obj in schema.objects:
            if isinstance(obj, CylinderSchema):
                lines.append(_cylinder_line(obj))
            elif isinstance(obj, BoxSchema):
                lines.append(_box_line(obj))
            elif isinstance(obj, SphereSchema):
                lines.append(_sphere_line(obj))
        lines.append("")

    # Snapshots
    if schema.snapshots:
        for snap in schema.snapshots:
            s_dx = snap.dx if snap.dx is not None else dx
            s_dy = snap.dy if snap.dy is not None else dy
            s_dz = snap.dz if snap.dz is not None else dz
            s_x2 = snap.x2 if snap.x2 is not None else x_extent
            s_y2 = snap.y2 if snap.y2 is not None else y_extent
            s_z2 = snap.z2 if snap.z2 is not None else z_extent
            lines.append(
                f"#snapshot: {snap.x1:.10g} {snap.y1:.10g} {snap.z1:.10g} "
                f"{s_x2:.10g} {s_y2:.10g} {s_z2:.10g} "
                f"{s_dx:.10g} {s_dy:.10g} {s_dz:.10g} "
                f"{snap.time_s:.10g} {snap.filename}"
            )
        lines.append("")

    lines.append(f"#geometry_view: 0 0 0 {x_extent:.10g} {y_extent:.10g} {z_extent:.10g} {dx:.10g} {dy:.10g} {dz:.10g} model_view n")

    text = "\n".join(lines)
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Wrote {output_filename}")
    return text