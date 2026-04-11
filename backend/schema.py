from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Tuple


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
    top_air_extra_m: Optional[float] = None
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
    fractal_nbins: int = 3


# ---------------------------------------------------------------------------
# Subagent extraction schemas
# Each subagent returns one of these typed classes. Fields are Optional so
# partial extraction is possible; resolvers fill defaults from lookup tables.
# ---------------------------------------------------------------------------

class ExtractedLayerParams(BaseModel):
    """Single layer as extracted by the layer subagent.
    Only explicit numeric min/max ranges are accepted for the core parameters."""
    name: Optional[str] = None

    # Thickness range (required)
    thickness_m_min: Optional[float] = None
    thickness_m_max: Optional[float] = None

    # Texture fractions — ranges (required; must sum to 100 after sampling)
    sand_pct_min: Optional[float] = None
    sand_pct_max: Optional[float] = None
    silt_pct_min: Optional[float] = None
    silt_pct_max: Optional[float] = None
    clay_pct_min: Optional[float] = None
    clay_pct_max: Optional[float] = None

    # Volumetric water content range (required for physics models)
    theta_v_min: Optional[float] = None
    theta_v_max: Optional[float] = None

    # Density ranges
    bulk_density_gcm3_min: Optional[float] = None
    bulk_density_gcm3_max: Optional[float] = None
    particle_density_gcm3_min: Optional[float] = None
    particle_density_gcm3_max: Optional[float] = None

    # Categorical — list of allowed classes
    salinity_classes: Optional[List[str]] = None  # e.g. ["fresh", "brackish"]

    # Porosity override range (if not provided, derived from densities or texture)
    porosity_min: Optional[float] = None
    porosity_max: Optional[float] = None

    # Single values (not ranged)
    organic_fraction: Optional[float] = None
    porewater_sigma_Sm: Optional[float] = None


class ExtractedLayers(BaseModel):
    """Output of the layer extraction subagent."""
    num_layers: int
    layers: List[ExtractedLayerParams]


class ExtractedAntennaWaveform(BaseModel):
    """Output of the antenna + waveform extraction subagent."""
    antenna_kind: Optional[str] = "hertzian_dipole"
    antenna_axis: Optional[str] = "x"
    tx_rx_offset_m: float  # required: Tx-Rx offset in metres
    resistance: Optional[float] = None  # required when antenna_kind="voltage_source"
    source_start_time: Optional[float] = None
    source_end_time: Optional[float] = None

    waveform_kind: Optional[str] = "ricker"
    waveform_amplitude: float  # required: signal amplitude (e.g. 1.0)
    waveform_center_freq_hz: float  # required: centre frequency in Hz
    waveform_name: str  # required: descriptive name for the waveform


class ExtractedModelConfig(BaseModel):
    """Output of the model / domain extraction subagent."""
    model: str  # required: dielectric model (peplinski, dobson, mironov, crim)
    title: str  # required: simulation title
    source_height_m: float  # required: antenna height above ground (m)
    domain_x: float  # required: domain width (m)
    domain_y: float  # required: domain depth (m)
    top_air_extra_m: Optional[float] = None  # extra air space above source
    cells_per_wavelength: float  # required: cells per minimum wavelength
    max_cell_m: float  # required: maximum cell size (m)
    rx_same_height: Optional[bool] = True  # Rx at same height as Tx
    temperature_c: float  # required: ambient temperature (°C)
    enforce_validity: Optional[bool] = None
    num_samples: int  # required: number of samples to generate


class ExtractedAdvancedParams(BaseModel):
    """Output of the optional / advanced parameters extraction subagent."""
    surface_roughness: Optional[SurfaceRoughnessConfigSchema] = None
    snapshots: Optional[List[SnapshotConfigSchema]] = None
    rx_array: Optional[RxArrayConfigSchema] = None
    cylinders: Optional[List[CylinderSchema]] = None
    boxes: Optional[List[BoxSchema]] = None
    spheres: Optional[List[SphereSchema]] = None
    pml_cells: Optional[int] = None
    num_threads: Optional[int] = None
    output_dir: Optional[str] = None
    fractal_nbins: Optional[int] = None


# ---------------------------------------------------------------------------
# Aggregated extraction result (returned by the coordinator before resolving)
# ---------------------------------------------------------------------------

class AggregatedExtraction(BaseModel):
    """All four subagent outputs bundled together."""
    layers: ExtractedLayers
    antenna_waveform: ExtractedAntennaWaveform
    model_params: ExtractedModelConfig
    optional_params: ExtractedAdvancedParams

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
