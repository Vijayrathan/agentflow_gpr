from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal

class WaveformSchema(BaseModel):
    kind: str
    amplitude: float
    center_freq_hz: float
    name: str


class AntennaSchema(BaseModel):
    kind: str
    axis: str
    tx_rx_offset_m: float
    source_type: Literal["hertzian_dipole", "voltage_source"] = Field(default="hertzian_dipole", description="Type of source")
    resistance: Optional[float] = Field(default=None, description="Resistance for voltage source (Ohms)")
    
    def validate_source_params(self):
        """Validate that voltage_source has resistance"""
        if self.source_type == "voltage_source" and self.resistance is None:
            raise ValueError("voltage_source requires resistance parameter")
        return self


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


class CylinderObjectSchema(BaseModel):
    """Schema for cylindrical objects (pipes, rods, etc.)"""
    name: str
    x1: float
    y1: float
    z1: float
    x2: float
    y2: float
    z2: float
    radius: float
    material: str
    dielectric_smoothing: bool = True


class BoxObjectSchema(BaseModel):
    """Schema for box objects (tanks, containers, etc.)"""
    name: str
    x1: float
    y1: float
    z1: float
    x2: float
    y2: float
    z2: float
    material: str
    dielectric_smoothing: bool = True


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
    objects: Optional[List[CylinderObjectSchema | BoxObjectSchema]] = None
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
    objects: Optional[List[Dict[str, Any]]] = None
