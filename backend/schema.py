from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Dict, Any, Tuple, Literal

# Resistance bound for #transmission_line / #voltage_source (exclusive upper bound).
MAX_TL_RESISTANCE_OHM = 376.73

# The pipeline emits 2D TMz models as a one-cell-thick z domain. A scalar
# gprMax #pml_cells value would also apply to z and fail because nz == 1, so the
# extracted pml_cells value is treated as the in-plane PML thickness and z faces
# are forced to zero in the effective gprMax command.
THIN_2D_NZ_CELLS = 1


def validate_gprmax_pml_profile(
    pml_cells: Tuple[int, int, int, int, int, int],
    *,
    nx: Optional[int] = None,
    ny: Optional[int] = None,
    nz: Optional[int] = None,
) -> None:
    """Validate a six-face gprMax PML profile against known axis cell counts.

    gprMax rejects a PML when the opposing PML faces consume the whole axis. For
    symmetric PML values this is the familiar 2*pml >= n_axis condition.
    Unknown axes are skipped because their cell counts are derived downstream.
    """
    axis_specs = (
        ("x", pml_cells[0], pml_cells[3], nx),
        ("y", pml_cells[1], pml_cells[4], ny),
        ("z", pml_cells[2], pml_cells[5], nz),
    )
    for axis, lower, upper, cells in axis_specs:
        if lower < 0 or upper < 0:
            raise ValueError(f"PML cells for {axis} faces must be >= 0")
        if cells is not None and lower + upper >= cells:
            raise ValueError(
                f"gprMax rejects PML on {axis}: lower+upper PML cells "
                f"({lower + upper}) >= n{axis} ({cells}); for symmetric PML this "
                "is the 2*pml_cells >= n_axis rule"
            )


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
# ENTIRE_MIGRATION.md for the full pipeline / derive-stage documentation.
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
    pml_cells: int = Field(10, ge=0)       # gprMax default; in-plane for 2D
    buffer_cells: int = Field(10, ge=0)    # extra cells between PML and objects
    cells_per_wavelength: int = Field(10, gt=0)  # lambda/10 rule-of-thumb
    dimensionality: Literal["2D", "3D"] = "2D"

    # Resolution policy: highest SIGNIFICANT frequency = factor * centre freq.
    high_freq_factor: float = Field(3.0, gt=0)

    # Does waveform_center_freq_hz mean the PEAK freq (what gprMax's #waveform
    # takes) or Wang's BAND-CENTRE freq? Pin it down once here.
    center_freq_is_peak: bool = True

    # Soil build: #soil_peplinski via #fractal_box needs a material count.
    fractal_nbins: int = Field(50, gt=0)

    def gprmax_pml_cells(self) -> Tuple[int, int, int, int, int, int]:
        """Six-face #pml_cells tuple in gprMax order: x0 y0 z0 xmax ymax zmax."""
        p = self.pml_cells
        if self.dimensionality == "2D":
            return (p, p, 0, p, p, 0)
        return (p, p, p, p, p, p)

    @model_validator(mode="after")
    def _check_pml_policy(self):
        known_nz = THIN_2D_NZ_CELLS if self.dimensionality == "2D" else None
        validate_gprmax_pml_profile(self.gprmax_pml_cells(), nz=known_nz)
        return self


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
    """Output of the optional / advanced parameters extraction stage.

    Geometry objects were REMOVED from this section: all buried objects (fixed
    or sampled) are collected as ranges in the target stage (ExtractedTargetRanges;
    a static object is a degenerate min==max range)."""
    surface_roughness: Optional[SurfaceRoughnessConfigSchema] = None
    snapshots: Optional[List[SnapshotConfigSchema]] = None


# ---------------------------------------------------------------------------
# Variable buried-target ranges (collected by the target stage, right after
# layers and BEFORE the soil draw). Object geometry is DRAWN per sample over
# these ranges; only size/extent affect the global grid (material never feeds
# the eps_r corners — targets are PEC-only).
#
# Coordinate frame (domain-independent — the domain does not exist yet at
# collection time):
#   x_offset — SIGNED offset of the object's center from the domain's
#              horizontal center (= the Tx/Rx midpoint). 0 = under the antenna
#              midpoint, negative = left. Resolved to absolute x as
#              domain_x/2 + x_offset once the global derive fixes the grid.
#   depth    — depth of the object's CENTER below the ground surface (all
#              kinds), resolved as y_center = ground_y - depth.
#
# STATIC objects: a range with min == max on EVERY field draws identically
# into every sample (no special machinery). Static objects are never
# redrawn/dropped per sample — their placement is validated ONCE at the
# global-validation gate.
#
# Spheres are deliberately NOT supported: in the 2D one-cell-z domain a real
# sphere (r >> dx/2) would extend outside the domain in z. Deferred until a
# 3D emitter exists.
# ---------------------------------------------------------------------------

class _TargetRangeBase(BaseModel):
    """Shared fields + static detection for buried-target sampling ranges."""
    name: str = "target"
    material: Literal["pec"] = "pec"   # PEC-only: size-only grid effect, no eps
    x_offset_min_m: float              # signed offset from domain center
    x_offset_max_m: float
    depth_min_m: float                 # depth of CENTER below ground surface
    depth_max_m: float

    def _range_pairs(self) -> List[Tuple[float, float]]:
        """The geometric (min, max) pairs; subclasses extend."""
        return [
            (self.x_offset_min_m, self.x_offset_max_m),
            (self.depth_min_m, self.depth_max_m),
        ]

    @property
    def is_static(self) -> bool:
        """True when EVERY range is degenerate (min == max): the object draws
        identically into every sample. Partially-degenerate objects are dynamic."""
        return all(lo == hi for lo, hi in self._range_pairs())

    @model_validator(mode="after")
    def _check_base_ranges(self):
        for lo, hi, label in [
            (self.x_offset_min_m, self.x_offset_max_m, "x_offset"),
            (self.depth_min_m, self.depth_max_m, "depth"),
        ]:
            if lo > hi:
                raise ValueError(f"{label}: min ({lo}) > max ({hi})")
        return self


class CylinderTargetRange(_TargetRangeBase):
    """Buried cylinder range (2D: a disc of the given radius in the x-y plane,
    axis along the thin z)."""
    kind: Literal["cylinder"] = "cylinder"
    radius_min_m: float
    radius_max_m: float

    def _range_pairs(self) -> List[Tuple[float, float]]:
        return super()._range_pairs() + [(self.radius_min_m, self.radius_max_m)]

    @model_validator(mode="after")
    def _check_ranges(self):
        if self.radius_min_m > self.radius_max_m:
            raise ValueError(
                f"radius: min ({self.radius_min_m}) > max ({self.radius_max_m})")
        if self.radius_min_m <= 0:
            raise ValueError(f"radius_min_m must be > 0 (got {self.radius_min_m})")
        return self


class BoxTargetRange(_TargetRangeBase):
    """Buried rectangular box range. width = x extent, height = y extent;
    depth is the depth of the box CENTER below ground (same rule as cylinder)."""
    kind: Literal["box"] = "box"
    width_min_m: float
    width_max_m: float
    height_min_m: float
    height_max_m: float

    def _range_pairs(self) -> List[Tuple[float, float]]:
        return super()._range_pairs() + [
            (self.width_min_m, self.width_max_m),
            (self.height_min_m, self.height_max_m),
        ]

    @model_validator(mode="after")
    def _check_ranges(self):
        for lo, hi, label in [
            (self.width_min_m, self.width_max_m, "width"),
            (self.height_min_m, self.height_max_m, "height"),
        ]:
            if lo > hi:
                raise ValueError(f"{label}: min ({lo}) > max ({hi})")
        if self.width_min_m <= 0:
            raise ValueError(f"width_min_m must be > 0 (got {self.width_min_m})")
        if self.height_min_m <= 0:
            raise ValueError(f"height_min_m must be > 0 (got {self.height_min_m})")
        return self


class ExtractedTargetRanges(BaseModel):
    """Output of the target stage: zero or more buried objects, each range-based.
    Skip = save an empty payload {} (both lists default empty)."""
    cylinders: List[CylinderTargetRange] = Field(default_factory=list)
    boxes: List[BoxTargetRange] = Field(default_factory=list)

    @property
    def has_targets(self) -> bool:
        return bool(self.cylinders or self.boxes)


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


class SampledTarget(BaseModel):
    """One concrete buried object drawn for a sample (cylinder or box).

    x_offset_m is the SIGNED offset of the center from the domain's horizontal
    center; depth_m is the depth of the CENTER below the ground surface. Both
    resolve to absolute coordinates only once the global derive fixes the grid
    (x_abs = domain_x/2 + x_offset, y_center = ground_y - depth).
    """
    kind: Literal["cylinder", "box"]
    name: str = "target"
    material: str = "pec"
    x_offset_m: float
    depth_m: float
    radius_m: Optional[float] = None    # cylinder
    width_m: Optional[float] = None     # box: x extent
    height_m: Optional[float] = None    # box: y extent

    @model_validator(mode="after")
    def _kind_fields(self):
        if self.kind == "cylinder" and self.radius_m is None:
            raise ValueError("cylinder target requires radius_m")
        if self.kind == "box" and (self.width_m is None or self.height_m is None):
            raise ValueError("box target requires width_m and height_m")
        return self


class SampledSample(BaseModel):
    """One drawn sample: concrete values for every layer (+ drawn objects, in
    the canonical range order: cylinders first, then boxes)."""
    sample_id: int
    layers: List[SampledLayer]
    targets: List[SampledTarget] = Field(default_factory=list)


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
    eps=1 is folded in later at the global-derive stage).

    The target corners (size/extent) are aggregated in the SAME per-sample pass:
    they are size-only — the target material does NOT feed eps_r_max/min.
    """
    eps_r_max: float    # max wettest-bin eps over all sample-layers -> finest dx
    eps_r_min: float    # min driest-bin eps over all sample-layers  -> largest domain
    num_samples: int
    frequency_hz: float
    nbins: int
    # Buried-target corners (None when no targets were drawn). Per kind:
    # cylinder feature/extent = 2r, bottom = depth + r; box feature = min(w, h),
    # extent = w, bottom = depth + h/2 (in-plane dimensions only; thin z never
    # enters the feature).
    smallest_feature_global_m: Optional[float] = None   # min feature -> tightens dx
    largest_extent_global_m: Optional[float] = None      # max x extent -> enlarges domain_x
    deepest_target_bottom_global_m: Optional[float] = None  # max bottom depth -> enlarges depth_z
    # STATIC objects only: max(|x_offset| + extent/2). Center-relative offsets
    # make this symmetric, so widening domain_x to 2*(halfwidth + clearance)
    # accommodates both left- and right-pinned fixed objects.
    static_x_halfwidth_global_m: Optional[float] = None


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
    domain_x_m: float        # snapped up to an integer number of cells
    domain_y_m: float        # vertical: air gap + soil + buffers + PML (snapped)
    dt_s: float
    time_window_s: float
    peplinski_gate_ok: bool
    # Static geometry — derived ONCE, identical for every sample:
    ground_y_m: float        # ground surface y = bottom pad + depth_z
    tx_x_m: float            # transmitter x (= domain_x/2 - tx_rx_offset/2)
    rx_x_m: float            # receiver x   (= domain_x/2 + tx_rx_offset/2)
    tx_y_m: float            # transmitter height = ground_y + source_height
    rx_y_m: float            # receiver height (= tx_y if rx_same_height)


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
