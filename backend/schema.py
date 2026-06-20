from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Dict, Any, Tuple, Literal

# Resistance bound for #transmission_line / #voltage_source (exclusive upper bound).
MAX_TL_RESISTANCE_OHM = 376.73


# ---------------------------------------------------------------------------
# Core component schemas (used in final GprSchema)
# ---------------------------------------------------------------------------

class CustomMaterialSchema(BaseModel):
    eps_r: float
    sigma: float = 0.0
    mu_r: float = 1.0
    sigma_m: float = 0.0


class WaveformSchema(BaseModel):
    kind: str
    amplitude: float
    center_freq_hz: float
    name: str


class AntennaSchema(BaseModel):
    kind: str
    axis: str
    tx_rx_offset_m: float
    resistance: Optional[float] = None  # required when kind="voltage_source"
    source_start_time: Optional[float] = None
    source_end_time: Optional[float] = None


class LayerSchema(BaseModel):
    name: Optional[str] = None
    thickness_m: float
    sand_pct: float
    silt_pct: float
    clay_pct: float
    theta_v: float
    bulk_density_gcm3: Optional[float] = None
    particle_density_gcm3: Optional[float] = None
    porosity: Optional[float] = None
    organic_fraction: Optional[float] = None
    salinity_class: Optional[str] = None
    porewater_sigma_Sm: Optional[float] = None


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


# ---------------------------------------------------------------------------
# Final simulation schema (fully resolved, ready for file generation)
# ---------------------------------------------------------------------------

class GprSchema(BaseModel):
    model: str
    title: str
    source_height_m: float
    domain_xy_m: Tuple[float, float] = (0.6, 0.4)
    cells_per_wavelength: int = 15
    max_cell_m: float = 0.005
    rx_same_height: bool = True
    temperature_c: float = 20.0
    enforce_validity: bool = True
    salinity_defaults_Sm: Tuple[float, float, float, float] = (0.0, 0.1, 1.0, 3.5)
    waveform: WaveformSchema
    antenna: AntennaSchema
    layers: List[LayerSchema]
    objects: Optional[List[CylinderSchema | BoxSchema | SphereSchema]] = None
    surface_roughness: Optional[SurfaceRoughnessConfigSchema] = None
    snapshots: Optional[List[SnapshotConfigSchema]] = None
    rx_array: Optional[RxArrayConfigSchema] = None
    pml_cells: Optional[int] = None
    num_threads: Optional[int] = None
    output_dir: Optional[str] = None
    # Fix #10: gprMax docs example uses 50 bins; 3 gave only 3 discrete moisture
    # levels defeating the purpose of fractal heterogeneity.
    fractal_nbins: int = 50
    # Fix #16: fractal box directional weights (default isotropic).
    # Real soils often have stronger horizontal than vertical correlation;
    # weight_z < 1.0 emphasises horizontal layering.
    fractal_weight_x: float = 1.0
    fractal_weight_y: float = 1.0
    fractal_weight_z: float = 1.0


# ---------------------------------------------------------------------------
# Subagent extraction schemas (collect-only)
#
# These are the ONLY parameters the extraction agents collect. Everything the
# physics fixes (Peplinski eps/sig, peak frequency, wavelengths, grid, domain,
# depth, time window) is DERIVED downstream, never collected here. See
# temp_schema_file.py for the full pipeline / derive-stage documentation.
#
# Sections (one per agent):
#   dataset_config   -> DatasetConfig
#   layers           -> ExtractedLayers
#   waveform         -> ExtractedWaveform
#   antenna          -> ExtractedAntenna
#   advanced_params  -> ExtractedAdvancedParams
# ---------------------------------------------------------------------------

class DatasetConfig(BaseModel):
    """STAGE 0 — dataset / run orchestration (collect).

    Holds everything that has no physics dependency but drives the run.
    """
    num_samples: int = Field(..., gt=0,
        description="Number of input files / data samples to generate (NOT time samples).")
    model_basename: str = "soil_sample"   # -> #title and output filename stem
    output_dir: str = "./dataset"
    num_threads: Optional[int] = None      # OpenMP threads; None -> gprMax default

    # FDTD boundary / grid policy (enter the GLOBAL derive downstream)
    pml_cells: int = 10                    # gprMax default; sized into the domain
    buffer_cells: int = 10                 # extra cells between PML and objects
    cells_per_wavelength: int = 10         # lambda/10 rule-of-thumb
    dimensionality: Literal["2D", "3D"] = "2D"

    # Resolution policy: highest SIGNIFICANT frequency = factor * centre freq.
    high_freq_factor: float = 3.0

    # Does waveform_center_freq_hz mean the PEAK freq (what gprMax's #waveform
    # takes) or Wang's BAND-CENTRE freq? Pin it down once here.
    center_freq_is_peak: bool = True

    # Soil build: #soil_peplinski via #fractal_box needs a material count.
    fractal_nbins: int = 50


class ExtractedLayerParams(BaseModel):
    """One soil layer (STAGE 1). Ranges are sampled per-sample downstream.

    The sampler must draw EVERY field declared as a range here. If a quantity
    is meant to be fixed, set min == max. Silt is NOT collected — it is a
    derived label (100 - sand - clay) computed downstream.
    """
    name: Optional[str] = None

    # Per-layer thickness — REQUIRED for stratified models.
    thickness_m_min: float
    thickness_m_max: float

    # Texture (percent). Closure is over three fractions: sand + silt + clay = 100.
    # We store sand & clay (Peplinski's only texture inputs); silt is derived.
    sand_pct_min: float
    sand_pct_max: float
    clay_pct_min: float
    clay_pct_max: float

    # Volumetric water content (fraction 0-1) — the per-layer ENVELOPE. Per
    # sample, the sampler draws a sub-band inside this envelope and passes it to
    # #soil_peplinski (which consumes a (min, max) range, not a scalar).
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
        # texture closure feasibility — sand+clay must be able to be <=100
        if self.sand_pct_min + self.clay_pct_min > 100.0:
            raise ValueError("sand_pct_min + clay_pct_min > 100 (no room for silt)")
        # envelope feasibility: the TOP of the moisture envelope must fit the
        # LOOSEST achievable porosity (min bulk, max particle).
        n_max = 1.0 - (self.bulk_density_gcm3_min / self.particle_density_gcm3_max)
        if self.theta_v_max > n_max:
            raise ValueError(
                f"theta_v_max ({self.theta_v_max}) exceeds loosest porosity ({n_max:.3f}); "
                "water content cannot exceed pore space even at max porosity")
        return self


# compute peplinski eps and sig from the gprmax module itself

class ExtractedLayers(BaseModel):
    """Output of the layer extraction subagent (STAGE 1)."""
    num_layers: int
    layers: List[ExtractedLayerParams]

    @model_validator(mode="after")
    def _count_matches(self):
        if self.num_layers != len(self.layers):
            raise ValueError(f"num_layers ({self.num_layers}) != len(layers) ({len(self.layers)})")
        return self


class ExtractedWaveform(BaseModel):
    """Output of the waveform extraction subagent (STAGE 2)."""

    waveform_kind: Optional[str] = "ricker"
    waveform_amplitude: float = 1.0  # required #waveform field; 1.0 by convention
    waveform_center_freq_hz: float  # required: centre frequency in Hz
    waveform_name: str  # required: descriptive name for the waveform

    # Source timing (optional): gates WHEN the source is on/off; resolves AFTER
    # the time window downstream.
    source_start_time: Optional[float] = None   # delay
    source_end_time: Optional[float] = None      # removal


#compute fp

#compute lambda_min, max and cells_per_wavelength


class ExtractedAntenna(BaseModel):
    """Output of the antenna extraction subagent (STAGE 3)."""
    antenna_kind: Optional[str] = "hertzian_dipole"
    antenna_axis: Optional[str] = "x"
    tx_rx_offset_m: float  # required: Tx-Rx offset in metres
    resistance: Optional[float] = None  # required for transmission_line / voltage_source
    rx_same_height: Optional[bool] = True  # Rx at same height as Tx

    # source_height_m must exist for "same height" to mean anything. If left
    # None it is DERIVED downstream as >= lambda_max/2.
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


class ExtractedAdvancedParams(BaseModel):
    """Output of the optional / advanced parameters extraction subagent (STAGE 4)."""
    surface_roughness: Optional[SurfaceRoughnessConfigSchema] = None
    snapshots: Optional[List[SnapshotConfigSchema]] = None
    cylinders: Optional[List[CylinderSchema]] = None
    boxes: Optional[List[BoxSchema]] = None
    spheres: Optional[List[SphereSchema]] = None


#the smallest target feature must resolve to ≥10 cells, which can tighten Δx below λmin/10, and target extents enlarge the domain

# ---------------------------------------------------------------------------
# Aggregated extraction result (returned by the coordinator before resolving)
# ---------------------------------------------------------------------------

class AggregatedExtraction(BaseModel):
    """All five subagent outputs bundled together."""
    dataset_config: DatasetConfig
    layers: ExtractedLayers
    waveform: ExtractedWaveform
    antenna: ExtractedAntenna
    advanced_params: ExtractedAdvancedParams

# ---------------------------------------------------------------------------
# Sampler output (STAGE 5) — one concrete draw over the layer ranges
# ---------------------------------------------------------------------------

class SampledLayer(BaseModel):
    """One concrete layer drawn from an ExtractedLayerParams range.

    sand/clay/thickness and both densities are drawn uniformly; silt_pct is the
    derived texture-closure label (100 - sand - clay). theta_v is NOT drawn — the
    per-layer (min, max) envelope is passed straight through because
    #soil_peplinski consumes a moisture BAND, not a scalar.
    """
    name: Optional[str] = None
    thickness_m: float
    sand_pct: float
    clay_pct: float
    silt_pct: float                 # derived label: 100 - sand - clay
    theta_v_min: float              # per-layer moisture band, passed through
    theta_v_max: float
    bulk_density_gcm3: float
    particle_density_gcm3: float

    @model_validator(mode="after")
    def _band_ok(self):
        if self.theta_v_min >= self.theta_v_max:
            raise ValueError("theta_v_min must be < theta_v_max (need a real moisture band)")
        return self


class SampledSample(BaseModel):
    """One drawn sample: concrete values for every layer."""
    sample_id: int
    layers: List[SampledLayer]


# ---------------------------------------------------------------------------
# Per-sample Peplinski derive (STAGE 6) — sizing permittivity from gprMax's OWN
# routine. sigma is intentionally NOT derived: gprMax writes the actual eps/sigma
# materials at model-build time, and only the real, in-band eps_r enters the
# wavelength / grid budget.
# ---------------------------------------------------------------------------

class DerivedLayer(BaseModel):
    """In-band relative permittivity edges for one sampled layer, evaluated with
    gprMax's PeplinskiSoil at the operating frequency (calculate_er(f).real)."""
    name: Optional[str] = None
    eps_r_dry: float    # driest bin  -> smallest eps -> largest lambda_max (domain)
    eps_r_wet: float    # wettest bin -> largest eps  -> smallest lambda_min (dx)


class DerivedSample(BaseModel):
    sample_id: int
    layers: List[DerivedLayer]


class GlobalEpsAggregate(BaseModel):
    """eps_r corners aggregated across all sampled layers (soil only; free space
    eps=1 is folded in later at the global-derive stage)."""
    eps_r_max: float    # max wettest-bin eps over all sample-layers -> finest dx
    eps_r_min: float    # min driest-bin eps over all sample-layers  -> largest domain
    num_samples: int
    frequency_hz: float
    nbins: int


# ---------------------------------------------------------------------------
# Global derive (STAGE 7) — ONE grid / domain / depth / time window for ALL
# samples, sized from the aggregated eps_r corners + waveform/antenna/layers.
# ---------------------------------------------------------------------------

class GlobalDerived(BaseModel):
    f_peak_hz: float
    f_min_hz: float          # Wang lower band edge (Peplinski gate)
    f_max_hz: float          # Wang upper band edge (Peplinski gate)
    bandwidth_hz: float
    f_high_hz: float         # highest significant freq (resolution check)
    eps_r_max_global: float
    eps_r_min_global: float
    dx_m: float              # global Δx = Δy = Δz
    lambda_min_m: float
    lambda_max_m: float
    surface_xy_m: float
    source_height_m: float
    depth_z_m: float
    domain_x_m: float
    domain_y_m: float        # vertical: air gap + soil + buffers + PML
    dt_s: float
    time_window_s: float
    peplinski_gate_ok: bool


# ---------------------------------------------------------------------------
# Dataset generation result schemas
# ---------------------------------------------------------------------------

class SampledLayerValues(BaseModel):
    """Concrete single-valued layer produced by sampling from a ResolvedLayerRange.

    bulk_density_gcm3 and particle_density_gcm3 are always populated: either with
    the user-supplied sampled value or with the physics model fallback (1.5 / 2.66)
    so the manifest is a complete record of what was used in the dielectric computation.
    """
    name: Optional[str] = None
    thickness_m: float
    sand_pct: float
    silt_pct: float
    clay_pct: float
    theta_v: float
    bulk_density_gcm3: float          # user-supplied or fallback (1.5 g/cm³)
    particle_density_gcm3: float      # user-supplied or fallback (2.66 g/cm³)
    porosity: Optional[float] = None  # user-supplied; None when derived from densities/texture
    organic_fraction: float
    salinity_class: Optional[str] = None
    porewater_sigma_Sm: Optional[float] = None


class SampleRecord(BaseModel):
    """One generated .in file and its sampled parameter values."""
    sample_index: int
    filename: str
    filepath: str
    layers: List[SampledLayerValues]


class DatasetGenerationResult(BaseModel):
    """Summary returned after batch generation."""
    status: str          # "complete" | "partial" | "error"
    dataset_name: str
    output_dir: str
    num_requested: int
    num_generated: int
    num_failed: int
    manifest_csv_path: str
    manifest_json_path: str
    samples: List[SampleRecord]
    errors: List[str]
