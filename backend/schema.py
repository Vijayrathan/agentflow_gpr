from pydantic import BaseModel
from typing import Optional, List, Dict, Any    

class WaveformSchema(BaseModel):
    kind: str
    amplitude: float
    center_freq_hz: float
    name: str


class AntennaSchema(BaseModel):
    kind: str
    axis: str
    tx_rx_offset_m: float


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
# Parameter extraction schema
class ExtractedParameters(BaseModel):
    """Schema for extracted parameters from user query"""
    num_layers: Optional[int] = None
    layers: Optional[List[Dict[str, Any]]] = None
    waveform: Optional[Dict[str, Any]] = None
    antenna: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    title: Optional[str] = None
    source_height_m: Optional[float] = None
    domain_x: Optional[float] = None
    domain_y: Optional[float] = None
    cells_per_wavelength: Optional[float] = None
    max_cell_m: Optional[float] = None
    temperature_c: Optional[float] = None
    enforce_validity: Optional[bool] = None
