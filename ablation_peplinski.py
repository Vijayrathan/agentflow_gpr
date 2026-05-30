"""
Ablation: gprMax built-in soil_peplinski vs. our peplinski_mixture.

Reproduces what gprMax's PeplinskiSoil.calculate_debye_properties() does,
then evaluates the resulting Debye material at the actual center frequency,
and compares to our peplinski_mixture() output.

Usage:
    python ablation_peplinski.py
"""
from __future__ import annotations

import math
import cmath

# ── Physical constants (match gprMax) ─────────────────────────────────────────
EPS0 = 8.854187817e-12
C0   = 299792458.0

# gprMax water constants (materials.py lines 33-36)
WATER_ER  = 80.1
WATER_ERI = 4.9
WATER_TAU = 9.231e-12
WATER_DELTA_ER = WATER_ER - WATER_ERI  # 75.2


# ─────────────────────────────────────────────────────────────────────────────
# gprMax built-in: reproduce PeplinskiSoil.calculate_debye_properties()
# ─────────────────────────────────────────────────────────────────────────────

def gprmax_peplinski_debye(
    sand_frac: float,   # 0-1
    clay_frac: float,   # 0-1
    rho_b: float,       # bulk density g/cm³
    rho_s: float,       # sand particle density g/cm³
    theta_v: float,     # volumetric water fraction
) -> dict:
    """
    Reproduces what gprMax computes for a single theta_v bin mid-point.
    Returns dict with: er_inf, delta_er, tau, sigma, and the Debye-evaluated
    complex epsilon at any queried frequency.
    """
    S = sand_frac
    C = clay_frac
    a = 0.65

    # Solid permittivity — Dobson (1985) Eq. (22)
    es = (1.01 + 0.44 * rho_s) ** 2 - 0.062

    # Fitting exponents
    b1 = 1.2748 - 0.519 * S - 0.152 * C  # beta' (real part)
    b2 = 1.33797 - 0.603 * S - 0.166 * C  # beta'' (imag part)

    # ── Water evaluated at FIXED 1.3 GHz (gprMax hardcodes this) ─────────────
    f_fixed = 1.3e9
    w_fixed = 2 * math.pi * f_fixed
    erealw = WATER_ERI + WATER_DELTA_ER / (1 + (w_fixed * WATER_TAU) ** 2)

    # ── Effective conductivity (0.3–1.3 GHz formula, gprMax line 292) ─────────
    sigf = 0.0467 + 0.2204 * rho_b - 0.411 * S + 0.6614 * C

    # ── Real part (Eq. 2 then Eq. 9 correction) ───────────────────────────────
    er = (1.0
          + (rho_b / rho_s) * (es ** a - 1.0)
          + (theta_v ** b1) * (erealw ** a)
          - theta_v) ** (1.0 / a)
    er = 1.15 * er - 0.68                  # correction (Eq. 9)

    # ── Permittivity at infinite frequency ────────────────────────────────────
    eri = er - (theta_v ** (b2 / a)) * WATER_DELTA_ER

    # ── Effective conductivity in the Debye material ──────────────────────────
    sig = theta_v ** (b2 / a) * (sigf * (rho_s - rho_b)) / (rho_s * theta_v)

    return {
        "er_inf":    eri,
        "delta_er":  er - eri,
        "tau":       WATER_TAU,
        "sigma_dc":  sig,
        # The "static" er gprMax would read (before Debye dispersion at runtime)
        "er_static": er,
    }


def gprmax_debye_at_freq(debye: dict, f: float) -> tuple[float, float]:
    """
    Evaluate gprMax's Debye material at frequency f.
    Returns (eps_r_real, sigma_eff) matching what the FDTD solver sees.
    """
    omega = 2 * math.pi * f
    tau = debye["tau"]
    eps_complex = (debye["er_inf"]
                   + debye["delta_er"] / (1 + 1j * omega * tau))
    eps_r = eps_complex.real
    # Loss → sigma:  ε'' = Im{ε} = -delta_er * omega*tau / (1 + (omega*tau)^2)
    # But gprMax *also* carries a separate DC sigma term.
    sigma_from_debye = omega * EPS0 * abs(eps_complex.imag)
    sigma_total = sigma_from_debye + debye["sigma_dc"]
    return eps_r, sigma_total


# ─────────────────────────────────────────────────────────────────────────────
# Our solver: import from backend
# ─────────────────────────────────────────────────────────────────────────────

from backend.physics_modelling import (
    peplinski_mixture,
    water_permittivity_debye,
    eps_to_sigma,
)


# ─────────────────────────────────────────────────────────────────────────────
# Run ablation
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(
    sand_pct: float,
    clay_pct: float,
    rho_b: float,
    rho_s: float,
    theta_v: float,
    f_hz: float,
    temp_c: float = 20.0,
    label: str = "",
):
    print(f"\n{'='*70}")
    if label:
        print(f"  {label}")
    print(f"{'='*70}")
    print(f"  Inputs:  sand={sand_pct}%  clay={clay_pct}%  ρ_b={rho_b} g/cm³  "
          f"ρ_s={rho_s} g/cm³  θ_v={theta_v}  f={f_hz/1e9:.3f} GHz  T={temp_c}°C")
    print()

    # ── gprMax path ───────────────────────────────────────────────────────────
    debye = gprmax_peplinski_debye(
        sand_frac=sand_pct / 100.0,
        clay_frac=clay_pct / 100.0,
        rho_b=rho_b,
        rho_s=rho_s,
        theta_v=theta_v,
    )
    gpm_eps_r, gpm_sigma = gprmax_debye_at_freq(debye, f_hz)

    print("  gprMax #soil_peplinski (Debye evaluated at f0):")
    print(f"    er_inf    = {debye['er_inf']:.4f}")
    print(f"    delta_er  = {debye['delta_er']:.4f}")
    print(f"    sigma_dc  = {debye['sigma_dc']:.6f} S/m")
    print(f"    → ε_r     = {gpm_eps_r:.4f}  σ_eff = {gpm_sigma:.6f} S/m")
    print()

    # ── Our path ──────────────────────────────────────────────────────────────
    eps_eff = peplinski_mixture(
        theta_v=theta_v,
        rho_b_gcm3=rho_b,
        rho_s_gcm3=rho_s,
        sand_pct=sand_pct,
        clay_pct=clay_pct,
        f_hz=f_hz,
        temp_c=temp_c,
        sigma_pore=0.0,
    )
    our_eps_r, our_sigma = eps_to_sigma(eps_eff, f_hz)

    print("  Our peplinski_mixture (evaluated at f0):")
    print(f"    ε_eff     = {eps_eff.real:.4f} - {abs(eps_eff.imag):.4f}j")
    print(f"    → ε_r     = {our_eps_r:.4f}  σ_eff = {our_sigma:.6f} S/m")
    print()

    # ── Delta ─────────────────────────────────────────────────────────────────
    d_eps = abs(our_eps_r - gpm_eps_r)
    d_sig = abs(our_sigma - gpm_sigma)
    pct_eps = 100 * d_eps / max(gpm_eps_r, 1e-9)
    pct_sig = 100 * d_sig / max(gpm_sigma, 1e-9)
    print(f"  Δε_r   = {d_eps:.4f}  ({pct_eps:.1f}%)")
    print(f"  Δσ     = {d_sig:.6f} S/m  ({pct_sig:.1f}%)")


if __name__ == "__main__":
    # Canonical Peplinski example from gprMax docs:
    # sand=0.5, clay=0.5, ρ_b=2.0 g/cm³, ρ_s=2.66 g/cm³, θ_v range 0.001-0.25
    # We use the mid-point θ_v ≈ 0.125 as the single reference value.

    BASE = dict(sand_pct=50.0, clay_pct=50.0, rho_b=2.0, rho_s=2.66)

    print("\nPeplinski Ablation: gprMax built-in vs. our solver")
    print("Reproduces gprMax docs example (sand=0.5, clay=0.5, ρ_b=2.0, ρ_s=2.66)")

    # ── Sweep across moisture values ──────────────────────────────────────────
    for theta in [0.05, 0.10, 0.15, 0.20, 0.25]:
        run_ablation(
            **BASE,
            theta_v=theta,
            f_hz=1.0e9,
            label=f"θ_v={theta:.2f}, f=1.0 GHz (mid Peplinski band)",
        )

    # ── Sweep across frequencies (within Peplinski range) ────────────────────
    for f_ghz in [0.3, 0.6, 0.9, 1.3]:
        run_ablation(
            **BASE,
            theta_v=0.15,
            f_hz=f_ghz * 1e9,
            label=f"θ_v=0.15, f={f_ghz} GHz",
        )

    # ── Typical GPR sandy soil ─────────────────────────────────────────────────
    run_ablation(
        sand_pct=40.0, clay_pct=10.0, rho_b=1.5, rho_s=2.66,
        theta_v=0.15, f_hz=1.0e9,
        label="Sandy loam: sand=40%, clay=10%, ρ_b=1.5, θ_v=0.15, f=1.0 GHz",
    )
