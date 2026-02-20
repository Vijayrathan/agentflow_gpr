

from __future__ import annotations

from typing import List, Literal, Optional, Dict, Tuple
from pydantic import BaseModel, Field, model_validator
from .soil_lookup_table import *

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


class UserModelSimple(BaseModel):
    model: str = "gprMax"
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


class UserSimulationSimple(BaseModel):
    layers: List[UserLayerSimple]
    antenna: UserAntennaSimple = UserAntennaSimple()
    waveform: UserWaveformSimple = UserWaveformSimple()
    model: UserModelSimple = UserModelSimple()

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


class SimulationFull(BaseModel):
    layers: List[LayerFull]
    waveform: WaveformFull
    antenna: AntennaFull
    model: ModelFull


# -----------------------------
# 5) EXAMPLE
# -----------------------------
if __name__ == "__main__":
    user = UserSimulationSimple(
        layers=[
            UserLayerSimple(
                name="topsoil",
                thickness_m=0.3,
                texture_class="sandy_loam",
                moisture_state="normal",
                organic_level="low",
                salinity_environment="fresh",
                compaction_level="normal",
            ),
            UserLayerSimple(
                name="subsoil",
                thickness_m=1.7,
                texture_class="clay_loam",
                moisture_state="wet",
                organic_level="none",
                salinity_environment="fresh",
                compaction_level="compacted",
            ),
        ],
        antenna=UserAntennaSimple(preset="generic_400MHz", axis="x"),
        waveform=UserWaveformSimple(kind="ricker", name="ricker_default"),
        model=UserModelSimple(
            title="Example site",
            survey_length_m=15.0,
            max_depth_m=2.5,
            quality="balanced",
            antenna_height_m=0.02,
        ),
    )

    full = resolve_to_full(user)
    print(full.model_dump_json(indent=2))
