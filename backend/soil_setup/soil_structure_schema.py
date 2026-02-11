

from __future__ import annotations

from typing import List, Literal, Optional, Dict, Tuple,Any
from pydantic import BaseModel, Field, model_validator
from soil_setup.soil_lookup_table import *

# -----------------------------
# 1) USER-FACING SCHEMAS
# -----------------------------

USDAClass = Literal[
    "sand",
    "loamy_sand",
    "sandy_loam",
    "loam",
    "silt_loam",
    "silt",
    "sandy_clay_loam",
    "clay_loam",
    "silty_clay_loam",
    "sandy_clay",
    "silty_clay",
    "clay",
]

MoistureState = Literal["dry", "normal", "wet", "saturated"]
OrganicLevel = Literal["none", "low", "moderate", "high_peaty"]
SalinityEnv = Literal["fresh", "slightly_saline", "brackish", "seawater"]
CompactionLevel = Literal["loose", "normal", "compacted"]

Axis = Literal["x", "y", "z"]
Quality = Literal["fast", "balanced", "high_accuracy"]

AntennaPreset = Literal[
    "generic_200MHz",
    "generic_400MHz",
    "generic_800MHz",
    "generic_1GHz",
    "generic_1.2GHz",
    "generic_1.5GHz"
]


class UserLayerSimple(BaseModel):
    """
    What your GPT-4.1 extractor should fill.
    """
    name: str = Field(default="layer_1")
    thickness_m: float = Field(gt=0.0)

    texture_class: USDAClass
    moisture_state: MoistureState = "normal"

    # Optional "easy" knobs
    organic_level: OrganicLevel = "none"
    salinity_environment: SalinityEnv = "fresh"
    compaction_level: CompactionLevel = "normal"

    # Optional expert overrides (your extractor can fill these if user provides numbers)
    theta_v_override: Optional[float] = Field(default=None, ge=0.0, le=0.9)
    sand_pct_override: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    silt_pct_override: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    clay_pct_override: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    bulk_density_override_gcm3: Optional[float] = Field(default=None, ge=0.8, le=2.2)
    particle_density_override_gcm3: Optional[float] = Field(default=None, ge=2.0, le=3.0)
    porewater_sigma_override_Sm: Optional[float] = Field(default=None, ge=0.0, le=10.0)

    @model_validator(mode="after")
    def _check_overrides_sum(self):
        # If user overrides texture fractions, enforce sum ~ 100 (tolerant)
        s = self.sand_pct_override
        si = self.silt_pct_override
        c = self.clay_pct_override
        if any(v is not None for v in (s, si, c)):
            if not all(v is not None for v in (s, si, c)):
                raise ValueError("If overriding texture fractions, provide sand/silt/clay together.")
            total = float(s + si + c)
            if abs(total - 100.0) > 1e-6:
                raise ValueError(f"Overridden sand/silt/clay must sum to 100. Got {total}.")
        return self


class UserWaveformSimple(BaseModel):
    # Keep minimal; mostly derived from antenna preset
    kind: Literal["ricker", "gaussian_derivative"] = "ricker"
    amplitude: Optional[float] = Field(default=None, gt=0.0)  # default resolver sets 1.0
    center_freq_hz_override: Optional[float] = Field(default=None, gt=0.0)
    name: str = "default_waveform"


class UserAntennaSimple(BaseModel):
    preset: AntennaPreset = "generic_400MHz"
    axis: Axis = "x"
    # optional expert override
    tx_rx_offset_m_override: Optional[float] = Field(default=None, gt=0.0)


SoilDielectricModel = Literal["peplinski", "dobson", "mironov", "crim"]


class UserModelSimple(BaseModel):
    # Soil dielectric model - REQUIRED, no default
    # - peplinski: Native gprMax support, 0.3-1.3 GHz, strict texture constraints
    # - dobson: Higher frequencies 1.4-18 GHz, requires Python block
    # - mironov: Wide range 0.6-18 GHz, bound/free water, requires Python block
    # - crim: No constraints, simple mixing, requires Python block
    model: SoilDielectricModel = Field(
        ...,  # Required field, no default
        description="Soil dielectric model: peplinski (native), dobson, mironov, or crim"
    )
    title: str = "GPR Simulation"
    quality: Quality = "balanced"

    # Let users specify in intuitive terms
    survey_length_m: float = Field(default=10.0, gt=0.0)
    max_depth_m: float = Field(default=2.0, gt=0.0)

    # Common antenna configurations
    antenna_height_m: float = Field(default=0.02, ge=0.0, le=1.0)  # ground-coupled default
    temperature_c: float = Field(default=20.0, ge=-20.0, le=60.0)

    enforce_validity: bool = True

    # Optional expert overrides
    domain_x_override: Optional[float] = Field(default=None, gt=0.0)
    domain_y_override: Optional[float] = Field(default=None, gt=0.0)
    cells_per_wavelength_override: Optional[int] = Field(default=None, ge=8, le=40)
    max_cell_m_override: Optional[float] = Field(default=None, gt=0.0)


class UserCylinderObject(BaseModel):
    """
    Represents a cylindrical object (pipes, rods, etc.)
    Cylinder is defined by two face centers and a radius
    """
    name: str = Field(default="cylinder_1")
    x1: float = Field(description="X coordinate of first face center (m)")
    y1: float = Field(description="Y coordinate of first face center (m)")
    z1: float = Field(description="Z coordinate of first face center (m)")
    x2: float = Field(description="X coordinate of second face center (m)")
    y2: float = Field(description="Y coordinate of second face center (m)")
    z2: float = Field(description="Z coordinate of second face center (m)")
    radius: float = Field(gt=0.0, description="Radius of cylinder (m)")
    material: Literal["pec", "free_space"] = Field(default="pec", description="Material type")
    dielectric_smoothing: bool = Field(default=True, description="Enable dielectric smoothing at boundaries")


class UserBoxObject(BaseModel):
    """
    Represents a box object (tanks, buried containers, etc.)
    Box is defined by two corner points (lower-left and upper-right)
    """
    name: str = Field(default="box_1")
    x1: float = Field(description="X coordinate of lower corner (m)")
    y1: float = Field(description="Y coordinate of lower corner (m)")
    z1: float = Field(description="Z coordinate of lower corner (m)")
    x2: float = Field(description="X coordinate of upper corner (m)")
    y2: float = Field(description="Y coordinate of upper corner (m)")
    z2: float = Field(description="Z coordinate of upper corner (m)")
    material: Literal["pec", "free_space"] = Field(default="pec", description="Material type")
    dielectric_smoothing: bool = Field(default=True, description="Enable dielectric smoothing at boundaries")
    
    @model_validator(mode="after")
    def _check_box_coordinates(self):
        """Ensure x2>x1, y2>y1, z2>z1"""
        if self.x2 <= self.x1:
            raise ValueError(f"x2 ({self.x2}) must be greater than x1 ({self.x1})")
        if self.y2 <= self.y1:
            raise ValueError(f"y2 ({self.y2}) must be greater than y1 ({self.y1})")
        if self.z2 <= self.z1:
            raise ValueError(f"z2 ({self.z2}) must be greater than z1 ({self.z1})")
        return self


class UserInputSimulation(BaseModel):
    layers: List[UserLayerSimple]
    antenna: UserAntennaSimple = UserAntennaSimple()
    waveform: UserWaveformSimple = UserWaveformSimple()
    model: UserModelSimple  # Required - no default since soil dielectric model must be specified
    objects: Optional[List[UserCylinderObject | UserBoxObject]] = Field(default=None, description="Optional buried objects")

    @model_validator(mode="after")
    def _check_layers(self):
        if len(self.layers) == 0:
            raise ValueError("At least one layer is required.")
        return self


# -----------------------------
# 2) YOUR CURRENT "FULL" OUTPUT SCHEMA (WHAT YOU NEED)
# -----------------------------

class LayerFull(BaseModel):
    thickness_m: float
    sand_pct: float
    silt_pct: float
    clay_pct: float
    theta_v: float
    bulk_density_gcm3: float
    particle_density_gcm3: float
    organic_fraction: float
    salinity_class: str
    porewater_sigma_Sm: float
    name: str


class WaveformFull(BaseModel):
    kind: str
    amplitude: float
    center_freq_hz: float
    name: str


class AntennaFull(BaseModel):
    kind: str
    axis: Axis
    tx_rx_offset_m: float
    source_type: str = "hertzian_dipole"  # 'hertzian_dipole' or 'voltage_source'
    resistance: Optional[float] = None  # For voltage_source (Ohms)


class ModelFull(BaseModel):
    model: str
    title: str
    source_height_m: float
    domain_x: float
    domain_y: float
    cells_per_wavelength: int
    max_cell_m: float
    temperature_c: float
    enforce_validity: bool


class CylinderObjectFull(BaseModel):
    """Full specification for a cylindrical object"""
    name: str
    x1: float
    y1: float
    z1: float
    x2: float
    y2: float
    z2: float
    radius: float
    material: str  # 'pec' or other material ID
    dielectric_smoothing: bool


class BoxObjectFull(BaseModel):
    """Full specification for a box object"""
    name: str
    x1: float
    y1: float
    z1: float
    x2: float
    y2: float
    z2: float
    material: str  # 'pec' or other material ID
    dielectric_smoothing: bool


class SimulationFull(BaseModel):
    layers: List[LayerFull]
    waveform: WaveformFull
    antenna: AntennaFull
    model: ModelFull
    objects: Optional[List[CylinderObjectFull | BoxObjectFull]] = Field(default=None)
