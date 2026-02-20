from pydantic import BaseModel
from typing import Optional, List, Dict, Any


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
    domain_x: float
    domain_y: float
    cells_per_wavelength: float
    max_cell_m: float
    temperature_c: float
    enforce_validity: bool
    waveform: WaveformSchema
    antenna: AntennaSchema
    layers: List[LayerSchema]
    surface_roughness: Optional[SurfaceRoughnessConfigSchema] = None
    snapshots: Optional[List[SnapshotConfigSchema]] = None
    rx_array: Optional[RxArrayConfigSchema] = None
    pml_cells: Optional[int] = None
    num_threads: Optional[int] = None
    output_dir: Optional[str] = None


# ---------------------------------------------------------------------------
# Subagent extraction schemas
# Each subagent returns one of these typed classes. Fields are Optional so
# partial extraction is possible; resolvers fill defaults from lookup tables.
# ---------------------------------------------------------------------------

class ExtractedLayerParams(BaseModel):
    """Single layer as extracted by the layer subagent.
    Accepts both user-friendly terms (texture_class, moisture_state) and
    raw numeric overrides (sand_pct, theta_v).  The resolver merges them."""
    name: Optional[str] = None
    thickness_m: Optional[float] = None

    # User-friendly descriptors (resolved via lookup tables)
    texture_class: Optional[str] = None
    moisture_state: Optional[str] = None
    organic_level: Optional[str] = None
    salinity_environment: Optional[str] = None
    compaction_level: Optional[str] = None

    # Expert numeric overrides (take precedence over descriptors)
    sand_pct: Optional[float] = None
    silt_pct: Optional[float] = None
    clay_pct: Optional[float] = None
    theta_v: Optional[float] = None
    bulk_density_gcm3: Optional[float] = None
    particle_density_gcm3: Optional[float] = None
    organic_fraction: Optional[float] = None
    salinity_class: Optional[str] = None
    porewater_sigma_Sm: Optional[float] = None


class ExtractedLayers(BaseModel):
    """Output of the layer extraction subagent."""
    num_layers: int
    layers: List[ExtractedLayerParams]


class ExtractedAntennaWaveform(BaseModel):
    """Output of the antenna + waveform extraction subagent."""
    antenna_kind: Optional[str] = "hertzian_dipole"
    antenna_axis: Optional[str] = "x"
    antenna_preset: Optional[str] = None
    tx_rx_offset_m: Optional[float] = None
    source_start_time: Optional[float] = None
    source_end_time: Optional[float] = None

    waveform_kind: Optional[str] = "ricker"
    waveform_amplitude: Optional[float] = None
    waveform_center_freq_hz: Optional[float] = None
    waveform_name: Optional[str] = None


class ExtractedModelConfig(BaseModel):
    """Output of the model / domain extraction subagent."""
    model: Optional[str] = None
    title: Optional[str] = None
    quality: Optional[str] = None
    source_height_m: Optional[float] = None
    survey_length_m: Optional[float] = None
    max_depth_m: Optional[float] = None
    domain_x: Optional[float] = None
    domain_y: Optional[float] = None
    cells_per_wavelength: Optional[float] = None
    max_cell_m: Optional[float] = None
    temperature_c: Optional[float] = None
    enforce_validity: Optional[bool] = None


class ExtractedOptionalParams(BaseModel):
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


# ---------------------------------------------------------------------------
# Aggregated extraction result (returned by the coordinator before resolving)
# ---------------------------------------------------------------------------

class AggregatedExtraction(BaseModel):
    """All four subagent outputs bundled together."""
    layers: ExtractedLayers
    antenna_waveform: ExtractedAntennaWaveform
    model_params: ExtractedModelConfig
    optional_params: ExtractedOptionalParams
