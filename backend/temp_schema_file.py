"""
gprMax synthetic-dataset parameter schema (corrected).

Goal of the project: produce N labelled gprMax input files for ML training.
A sampler draws per-sample values over the declared ranges; everything that the
physics fixes (Peplinski eps/sig, peak frequency, wavelengths, grid, domain,
depth, time window) is DERIVED, never collected.

We use ONE GLOBAL grid + ONE GLOBAL domain + ONE GLOBAL depth for all N samples,
so every input file is on the same Yee grid and the outputs are directly
comparable for ML. The grid is sized from the WORST-CASE corner of the sampling
space (see STAGE 7).

────────────────────────────────────────────────────────────────────────────
PIPELINE / EXECUTION ORDER
────────────────────────────────────────────────────────────────────────────
  STAGE 0  DatasetConfig            (collect, top-level orchestration)
  STAGE 1  ExtractedLayers          (collect)  ─┐
  STAGE 2  ExtractedWaveform        (collect)   ├ independent collect-stages;
  STAGE 3  ExtractedAntenna         (collect)   │ order among them is free
  STAGE 4  ExtractedAdvancedParams  (collect)  ─┘
  GATE     Peplinski validity check  (needs derived band ∩ chosen soil model)
  STAGE 5  sampler draws N parameter sets over the ranges in STAGE 1/2
  STAGE 6  PER-SAMPLE derive: Peplinski eps/sig (+ silt label)
  STAGE 7  GLOBAL derive: fp → band → lambda → dx → domain → depth → dt → tw
  STAGE 8  emit N input files on the single global grid

Source grounding for the constants/rules used below:
  - Wang (2015), "Frequencies of the Ricker wavelet": band-edge & centre ratios.
  - Peplinski et al. (1995) / gprMax docs: #soil_peplinski valid 0.3–1.3 GHz.
  - gprMax examples_simple_2D: lambda/10 rule; highest significant freq is
    ~2–3x the centre freq at the -40 dB level; PML default 10 cells + ~10-cell
    buffer; resolve smallest feature to >=10 cells.
  - Khosravi Largani et al. (FDTD medium-dimension guidelines): surface ~1.5*lambda_max,
    antenna height > lambda_max/2, depth > range resolution.
  - gprMax input_file_commands: #transmission_line/#voltage_source resistance
    must satisfy 0 < R < 376.73 ohm; #waveform amplitude is a required field.
"""

from __future__ import annotations
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, model_validator

C0 = 299_792_458.0  # speed of light, m/s

# Wang (2015) Ricker constants (Lambert-W based, amplitude-spectrum -6 dB band):
WANG_FLOW_OVER_FP   = 0.481623   # f_l1 / f_peak
WANG_FHIGH_OVER_FP  = 1.636567   # f_l2 / f_peak
WANG_FCENTRE_OVER_FP = 1.059095  # f_central / f_peak  (geometric band centre)

PEPLINSKI_FMIN_HZ = 0.30e9
PEPLINSKI_FMAX_HZ = 1.30e9
MAX_TL_RESISTANCE_OHM = 376.73   # exclusive upper bound



# ---------------------------------------------------------------------------
# Core component schemas (used in final GprSchema)
# ---------------------------------------------------------------------------

class CustomMaterialSchema(BaseModel):
    eps_r: float
    sigma: float = 0.0
    mu_r: float = 1.0
    sigma_m: float = 0.0


class CylinderSchema(BaseModel):
    name: str
    x1: float
    y1: float
    z1: float
    x2: float
    y2: float
    z2: float
    radius: float
    material: str = 'pec'
    custom_material: Optional[CustomMaterialSchema] = None
    dielectric_smoothing: bool = True


class BoxSchema(BaseModel):
    name: str
    x1: float
    y1: float
    z1: float
    x2: float
    y2: float
    z2: float
    material: str = 'pec'
    custom_material: Optional[CustomMaterialSchema] = None
    dielectric_smoothing: bool = True


class SphereSchema(BaseModel):
    name: str
    cx: float
    cy: float
    cz: float
    radius: float
    material: str = 'pec'
    custom_material: Optional[CustomMaterialSchema] = None
    dielectric_smoothing: bool = True


class SurfaceRoughnessConfigSchema(BaseModel):
    fractal_dim: float = 1.5
    weight_x: float = 1.0
    weight_y: float = 1.0
    amplitude_m: float = 0.01
    add_water: bool = False
    water_depth_m: float = 0.005
    seed: Optional[int] = None


class RxArrayConfigSchema(BaseModel):
    x1: float
    y1: float
    z1: float
    x2: float
    y2: float
    z2: float
    dx: float
    dy: float
    dz: float


class SnapshotConfigSchema(BaseModel):
    time_s: float
    filename: str
    dx: Optional[float] = None
    dy: Optional[float] = None
    dz: Optional[float] = None
    x1: float = 0.0
    y1: float = 0.0
    z1: float = 0.0
    x2: Optional[float] = None
    y2: Optional[float] = None
    z2: Optional[float] = None



# ═════════════════════════════════════════════════════════════════════════════
# STAGE 0 — DATASET / RUN ORCHESTRATION  (collect)
#   Holds everything that has no physics dependency but drives the run.
#   FIX #4: num_samples, output_dir, num_threads, pml_cells, model/title naming,
#   and fractal_nbins now have a home (they were homeless before).
# ═════════════════════════════════════════════════════════════════════════════
class DatasetConfig(BaseModel):
    num_samples: int = Field(..., gt=0,
        description="Number of input files / data samples to generate.")
    model_basename: str = "soil_sample"   # -> #title and output filename stem
    output_dir: str = "./dataset"
    num_threads: Optional[int] = None      # OpenMP threads; None -> gprMax default

    # FDTD boundary / grid policy (enter the GLOBAL derive in STAGE 7)
    pml_cells: int = 10                    # gprMax default; sized into the domain
    buffer_cells: int = 10                 # extra cells between PML and objects
    cells_per_wavelength: int = 10         # lambda/10 rule-of-thumb
    dimensionality: Literal["2D", "3D"] = "2D"

    # Resolution policy: highest SIGNIFICANT frequency = factor * centre freq.
    # FIX #6: this is for the lambda_min / Δx check ONLY. Do NOT use Wang's -6 dB
    # fmax here — that is for the Peplinski gate. Significant content sits ~2–3x
    # the centre at -40 dB (examples_simple_2D).
    high_freq_factor: float = 3.0

    # FIX #5: does waveform_center_freq_hz mean the PEAK freq (what gprMax's
    # #waveform takes) or Wang's BAND-CENTRE freq? Pin it down ONCE here.
    center_freq_is_peak: bool = True

    # Soil build: #soil_peplinski via #fractal_box needs a material count.
    fractal_nbins: int = 50                # SUGGESTION: was missing


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 1 — LAYERS / SOIL  (collect)
# ═════════════════════════════════════════════════════════════════════════════
class ExtractedLayerParams(BaseModel):
    """One layer. Ranges are sampled per-sample in STAGE 5.

    NOTE on sampler/schema consistency (FIX #8): the sampler must draw EVERY
    field declared as a range here (sand, clay, theta_v, both densities,
    thickness). If a quantity is meant to be fixed, set min == max.
    """
    name: Optional[str] = None

    # FIX #2: per-layer thickness — REQUIRED for stratified models. Without it
    # the layer interfaces are undefined. Global depth is derived from the
    # worst-case stack of these (STAGE 7g).
    thickness_m_min: float
    thickness_m_max: float

    # Texture (percent). FIX #9: closure is over THREE fractions:
    # sand + silt + clay = 100. We store sand & clay (Peplinski's only texture
    # inputs); silt is computed as a LABEL in STAGE 6a.
    sand_pct_min: float
    sand_pct_max: float
    clay_pct_min: float
    clay_pct_max: float

    # Volumetric water content (fraction 0–1) — this is the per-layer ENVELOPE.
    # gprMax's PeplinskiSoil takes a (min, max) RANGE for watervolfraction, not a
    # scalar, and builds a series of dispersive materials across it. So theta_v is
    # never a single sampled value. Per sample, the sampler draws a sub-band
    # (theta_v_min_s, theta_v_max_s) INSIDE this envelope (see SampledLayer); that
    # sub-band is what is passed to #soil_peplinski. theta_v therefore VARIES per
    # sample by varying the drawn sub-band.
    # FIX #3: capped by porosity n = 1 - rho_b/rho_s. Binding check is per-sample
    # in STAGE 6b (against the sample's own densities); here we only check the
    # envelope is feasible at the loosest possible porosity.
    theta_v_min: float
    theta_v_max: float

    # Densities (g/cm^3). Must satisfy rho_bulk < rho_particle (porosity > 0).
    bulk_density_gcm3_min: float
    bulk_density_gcm3_max: float
    particle_density_gcm3_min: float
    particle_density_gcm3_max: float

    @model_validator(mode="after")
    def _check_ranges(self):
        for lo, hi, label in [
            (self.thickness_m_min, self.thickness_m_max, "thickness"),
            (self.sand_pct_min, self.sand_pct_max, "sand_pct"),
            (self.clay_pct_min, self.clay_pct_max, "clay_pct"),
            (self.theta_v_min, self.theta_v_max, "theta_v"),
            (self.bulk_density_gcm3_min, self.bulk_density_gcm3_max, "bulk_density"),
            (self.particle_density_gcm3_min, self.particle_density_gcm3_max, "particle_density"),
        ]:
            if lo > hi:
                raise ValueError(f"{label}: min ({lo}) > max ({hi})")
        # FIX #9: texture closure feasibility — sand+clay must be able to be <=100
        if self.sand_pct_min + self.clay_pct_min > 100.0:
            raise ValueError("sand_pct_min + clay_pct_min > 100 (no room for silt)")
        # FIX #3: envelope feasibility. The TOP of the moisture envelope must fit
        # the LOOSEST achievable porosity (min bulk, max particle); otherwise no
        # density the sampler could draw would ever support theta_v_max.
        n_max = 1.0 - (self.bulk_density_gcm3_min / self.particle_density_gcm3_max)
        if self.theta_v_max > n_max:
            raise ValueError(
                f"theta_v_max ({self.theta_v_max}) exceeds loosest porosity ({n_max:.3f}); "
                "water content cannot exceed pore space even at max porosity")
        return self


class ExtractedLayers(BaseModel):
    num_layers: int
    layers: List[ExtractedLayerParams]

    @model_validator(mode="after")
    def _count_matches(self):
        # SUGGESTION: keep num_layers and the list in sync.
        if self.num_layers != len(self.layers):
            raise ValueError(f"num_layers ({self.num_layers}) != len(layers) ({len(self.layers)})")
        return self


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 2 — WAVEFORM  (collect)
#   Collect ONLY. No lambda/Δx here — those need soil eps and so live in the
#   GLOBAL derive (STAGE 7). FIX #1.
# ═════════════════════════════════════════════════════════════════════════════
class ExtractedWaveform(BaseModel):
    waveform_kind: Optional[str] = "ricker"
    waveform_amplitude: float = 1.0   # required #waveform field; 1.0 by convention
    waveform_center_freq_hz: float    # interpreted per DatasetConfig.center_freq_is_peak
    waveform_name: str

    # source timing (optional): moved here from any band logic — these gate WHEN
    # the source is on/off and only resolve AFTER the time window (STAGE 7k).
    source_start_time: Optional[float] = None   # delay
    source_end_time: Optional[float] = None      # removal


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 3 — ANTENNA  (collect)
# ═════════════════════════════════════════════════════════════════════════════
class ExtractedAntenna(BaseModel):
    antenna_kind: Optional[str] = "hertzian_dipole"

    # SUGGESTION: polarisation is conventionally perpendicular to the B-scan
    # survey direction (paper used Y; 2D gprMax example used z). Don't set blindly.
    antenna_axis: Optional[str] = "y"

    tx_rx_offset_m: float

    # FIX #10: resistance is required for BOTH transmission_line AND voltage_source,
    # bounded 0 < R < 376.73 ohm.
    resistance: Optional[float] = None

    rx_same_height: Optional[bool] = True

    # FIX #10: source_height_m must exist for "same height" to mean anything.
    # If left None it is DERIVED in STAGE 7h as >= lambda_max/2.
    source_height_m: Optional[float] = None

    rx_array: Optional[RxArrayConfigSchema] = None

    @model_validator(mode="after")
    def _resistance_rules(self):
        needs_R = self.antenna_kind in ("transmission_line", "voltage_source")
        if needs_R:
            if self.resistance is None:
                raise ValueError(f"resistance required for antenna_kind='{self.antenna_kind}'")
            if not (0.0 < self.resistance < MAX_TL_RESISTANCE_OHM):
                raise ValueError(f"resistance must be 0 < R < {MAX_TL_RESISTANCE_OHM} ohm")
        return self


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 4 — ADVANCED / GEOMETRY  (collect)
# ═════════════════════════════════════════════════════════════════════════════
class ExtractedAdvancedParams(BaseModel):
    surface_roughness: Optional[SurfaceRoughnessConfigSchema] = None
    snapshots: Optional[List[SnapshotConfigSchema]] = None
    cylinders: Optional[List[CylinderSchema]] = None
    boxes: Optional[List[BoxSchema]] = None
    spheres: Optional[List[SphereSchema]] = None


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 5 — SAMPLER OUTPUT (one drawn sample)
#
# theta_v VARIES PER SAMPLE as a drawn sub-band (theta_v_min, theta_v_max) inside
# the layer envelope, because PeplinskiSoil consumes a range, not a scalar.
#
# SAMPLER POLICY (your sampler must implement one of these; flag your choice):
#   - draw a centre c in [env_min, env_max] and a half-width w  -> (c-w, c+w), or
#   - draw two ordered values in [env_min, env_max].
# Either way, require theta_v_min < theta_v_max (a real band; a degenerate band
# defeats the fractal moisture distribution).
#
# FEASIBILITY (important sampler dependency): theta_v_max is capped by the
# sample's OWN porosity n = 1 - rho_b/rho_s. Since densities are ALSO drawn,
# draw the densities first, compute n, then cap theta_v_max <= n (or reject the
# draw). Drawing theta_v and densities independently will produce infeasible
# samples.
# ═════════════════════════════════════════════════════════════════════════════
class SampledLayer(BaseModel):
    name: Optional[str] = None
    thickness_m: float
    sand_pct: float
    clay_pct: float
    theta_v_min: float          # per-sample moisture band, lower edge
    theta_v_max: float          # per-sample moisture band, upper edge
    bulk_density_gcm3: float
    particle_density_gcm3: float

    @model_validator(mode="after")
    def _band_ok(self):
        if self.theta_v_min >= self.theta_v_max:
            raise ValueError("theta_v_min must be < theta_v_max (need a real moisture band)")
        return self

class SampledSample(BaseModel):
    sample_id: int
    layers: List[SampledLayer]


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 6 — PER-SAMPLE DERIVE  (Peplinski eps/sig + silt label)
# ═════════════════════════════════════════════════════════════════════════════
class DerivedLayer(BaseModel):
    name: Optional[str]
    thickness_m: float
    silt_pct: float              # 6a: LABEL only — NOT a Peplinski input
    theta_v_min: float           # passed through to #soil_peplinski as the range
    theta_v_max: float
    eps_r_dry: float             # eps at theta_v_min (driest edge -> smallest eps)
    eps_r_wet: float             # eps at theta_v_max (wettest edge -> largest eps)
    sigma_dry: float
    sigma_wet: float

def derive_per_sample(sample: SampledLayer, gprmax_peplinski_eps) -> DerivedLayer:
    """STAGE 6 — runs for every sampled layer of every sample.

    `gprmax_peplinski_eps(sand_fraction, clay_fraction, bulk_density,
    sand_particle_density, water_volumetric_fraction) -> (eps_r, sigma)` wraps
    gprMax's own Peplinski calculation evaluated at a SINGLE moisture value. The
    actual #soil_peplinski command still receives the full (min, max) band and
    builds its fractal material series internally; here we only need the eps
    BAND EDGES to size the global grid (STAGE 7).
    """
    # 6a. silt label (texture closure over three fractions)
    silt_pct = 100.0 - sample.sand_pct - sample.clay_pct

    # 6b. physical-validity guards on the drawn point
    if sample.sand_pct + sample.clay_pct > 100.0:
        raise ValueError("sand+clay > 100 in drawn sample")
    if sample.bulk_density_gcm3 >= sample.particle_density_gcm3:
        raise ValueError("bulk_density >= particle_density (porosity <= 0)")
    porosity = 1.0 - sample.bulk_density_gcm3 / sample.particle_density_gcm3
    # binding per-sample porosity cap: the WET edge of the band must fit pore space
    if sample.theta_v_max > porosity:
        raise ValueError(
            f"theta_v_max {sample.theta_v_max} > porosity {porosity:.3f} "
            "(wet edge of moisture band exceeds pore space)")

    # 6c. percent -> fraction for the gprMax command
    sand_frac = sample.sand_pct / 100.0
    clay_frac = sample.clay_pct / 100.0

    # 6d. eps/sig at the two band edges. eps is monotonic in water content, so
    #     the dry/wet edges bracket the permittivity spread of this soil.
    eps_dry, sig_dry = gprmax_peplinski_eps(
        sand_fraction=sand_frac, clay_fraction=clay_frac,
        bulk_density=sample.bulk_density_gcm3,
        sand_particle_density=sample.particle_density_gcm3,
        water_volumetric_fraction=sample.theta_v_min,
    )
    eps_wet, sig_wet = gprmax_peplinski_eps(
        sand_fraction=sand_frac, clay_fraction=clay_frac,
        bulk_density=sample.bulk_density_gcm3,
        sand_particle_density=sample.particle_density_gcm3,
        water_volumetric_fraction=sample.theta_v_max,
    )

    # # 6e. representative moisture label for ML (band midpoint)
    # theta_v_label = 0.5 * (sample.theta_v_min + sample.theta_v_max)

    return DerivedLayer(
        name=sample.name, thickness_m=sample.thickness_m, silt_pct=silt_pct,
        theta_v_min=sample.theta_v_min, theta_v_max=sample.theta_v_max,
        eps_r_dry=eps_dry, eps_r_wet=eps_wet, sigma_dry=sig_dry, sigma_wet=sig_wet,
    )


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 7 — GLOBAL DERIVE  (ONE grid / domain / depth for ALL samples)
#   Order is strict: fp → band(gate) → lambda → dx → domain → depth → dt → tw.
# ═════════════════════════════════════════════════════════════════════════════
class GlobalDerived(BaseModel):
    f_peak_hz: float
    f_min_hz: float          # Wang lower band edge (for Peplinski gate)
    f_max_hz: float          # Wang upper band edge (for Peplinski gate)
    bandwidth_hz: float
    f_high_hz: float         # significant high freq (for resolution)
    eps_r_max_global: float
    eps_r_min_global: float
    dx_m: float              # global Δx = Δy = Δz
    lambda_min_m: float
    lambda_max_m: float
    surface_xy_m: float
    source_height_m: float
    depth_z_m: float
    domain_x_m: float
    domain_y_m: float        # total height incl. air gap + buffers + PML
    dt_s: float
    time_window_s: float
    peplinski_gate_ok: bool


def derive_global(cfg: DatasetConfig,
                  wf: ExtractedWaveform,
                  ant: ExtractedAntenna,
                  layers: ExtractedLayers,
                  adv: ExtractedAdvancedParams,
                  eps_r_max_over_samples: float,
                  eps_r_min_over_samples: float) -> GlobalDerived:
    """STAGE 7 — runs ONCE over the whole sampling space.

    eps_r_max/min_over_samples are aggregated from STAGE 6 across all N samples
    (plus any fixed target materials). With per-sample moisture BANDS, take:
      - eps_r_max_over_samples = max over samples of DerivedLayer.eps_r_wet
        (wettest edge of the wettest soil -> highest eps -> smallest lambda_min
        -> finest Δx; resolution-safe for every sample)
      - eps_r_min_over_samples = min over samples of DerivedLayer.eps_r_dry
        (driest edge -> lowest eps -> largest lambda_max -> biggest domain)
    """
    # 7a. peak frequency (what #waveform actually takes). FIX #5.
    if cfg.center_freq_is_peak:
        f_peak = wf.waveform_center_freq_hz
    else:                       # input was Wang's band-centre -> back out the peak
        f_peak = wf.waveform_center_freq_hz / WANG_FCENTRE_OVER_FP

    # 7b. Wang band edges -> Peplinski validity GATE (0.3–1.3 GHz).
    f_min = WANG_FLOW_OVER_FP  * f_peak
    f_max = WANG_FHIGH_OVER_FP * f_peak
    bandwidth = f_max - f_min
    gate_ok = (f_min >= PEPLINSKI_FMIN_HZ) and (f_max <= PEPLINSKI_FMAX_HZ)
    if not gate_ok:
        raise ValueError(
            f"Peplinski gate FAIL: band [{f_min/1e6:.1f}, {f_max/1e6:.1f}] MHz "
            f"outside [300, 1300] MHz. (peak={f_peak/1e6:.1f} MHz)")

    # 7c. highest SIGNIFICANT frequency for the resolution check (NOT f_max). FIX #6.
    f_high = cfg.high_freq_factor * wf.waveform_center_freq_hz

    # 7d. global eps corners (include free space = 1 for the air region above)
    eps_max = eps_r_max_over_samples
    eps_min = min(eps_r_min_over_samples, 1.0)   # air sets the largest lambda

    # 7e. global Δx from lambda_min (finest grid). Tighten for smallest target.
    lambda_min = C0 / (f_high * (eps_max ** 0.5))
    dx = lambda_min / cfg.cells_per_wavelength
    smallest_feat = _smallest_feature_m(adv)
    if smallest_feat is not None:
        dx = min(dx, smallest_feat / 10.0)       # resolve smallest feature to >=10 cells

    # 7f. lambda_max + surface dimension (1.5*lambda_max, Khosravi et al.).
    lambda_max = C0 / (f_min * (eps_min ** 0.5))
    surface_xy = 1.5 * lambda_max

    # 7g. GLOBAL depth: deepest possible stack OR range-resolution floor.
    #     Single grid -> use the WORST (deepest) stack across the sampling space.
    max_stack = sum(L.thickness_m_max for L in layers.layers)
    range_res = C0 / (2.0 * bandwidth * (eps_max ** 0.5))   # slowest medium
    depth_z = max(max_stack, range_res)

    # 7h. antenna height: collected if given, else derived >= lambda_max/2.
    source_height = ant.source_height_m if ant.source_height_m is not None else (lambda_max / 2.0)

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


def _smallest_feature_m(adv: ExtractedAdvancedParams) -> Optional[float]:
    feats = []
    for c in (adv.cylinders or []):
        if c.radius_m: feats.append(2 * c.radius_m)
    for s in (adv.spheres or []):
        if s.radius_m: feats.append(2 * s.radius_m)
    return min(feats) if feats else None

def _target_footprint_x_m(adv: ExtractedAdvancedParams) -> Optional[float]:
    # Hook for your real geometry extents; returns None if no targets.
    return None


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 8 — top-level container handed to the input-file writer.
# ═════════════════════════════════════════════════════════════════════════════
class PipelineBundle(BaseModel):
    dataset: DatasetConfig
    layers: ExtractedLayers
    waveform: ExtractedWaveform
    antenna: ExtractedAntenna
    advanced: ExtractedAdvancedParams
    grid: Optional[GlobalDerived] = None   # filled after STAGE 7