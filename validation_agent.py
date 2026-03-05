"""Validation tools for gprMax simulation schemas.

Each function validates a specific component and is decorated as a LangChain
tool so it can be passed directly to a DeepAgent.
"""

from typing import Annotated, Optional
import os
from langchain_core.tools import tool

import dotenv
from deepagents import create_deep_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from rag import rag_search
from prompt_library import LAYER_RAG_SUBAGENT_PROMPT,LAYER_AGENT_PROMPT
from schema import GprSchema
from parameters_global_state import post_parameters,get_parameters,patch_parameters
dotenv.load_dotenv()


# Initialize the model
llm = ChatOpenAI(
    model="gpt-4.1",
    api_key=os.getenv("OPENAI_API_KEY"),
)

VALID_WAVEFORMS = {
    "gaussian",
    "gaussiandot",
    "gaussiandotnorm",
    "gaussiandotdot",
    "gaussiandotdotnorm",
    "ricker",
    "gaussianprime",
    "gaussiandoubleprime",
    "sine",
    "contsine",
}


@tool
def validate_layer(
    thickness_m: Annotated[float, "Layer thickness in metres"],
    sand_pct: Annotated[float, "Sand percentage (0-100)"],
    silt_pct: Annotated[float, "Silt percentage (0-100)"],
    clay_pct: Annotated[float, "Clay percentage (0-100)"],
    theta_v: Annotated[float, "Volumetric water content (0.0-1.0)"],
    bulk_density_gcm3: Annotated[Optional[float], "Bulk density in g/cm³"] = None,
    particle_density_gcm3: Annotated[Optional[float], "Particle density in g/cm³"] = None,
) -> str:
    """Validate a soil layer's physical parameters."""
    errors = []
    if thickness_m <= 0:
        errors.append("Layer thickness must be > 0")
    p_sum = sand_pct + silt_pct + clay_pct
    if abs(p_sum - 100.0) > 1e-6:
        errors.append(f"Sand + silt + clay must sum to 100 (got {p_sum:.2f})")
    if not (0.0 <= theta_v <= 1.0):
        errors.append("theta_v must be between 0.0 and 1.0")
    if bulk_density_gcm3 is not None and bulk_density_gcm3 <= 0:
        errors.append("bulk_density_gcm3 must be > 0 if provided")
    if particle_density_gcm3 is not None and particle_density_gcm3 <= 0:
        errors.append("particle_density_gcm3 must be > 0 if provided")
    if errors:
        return "VALIDATION FAILED: " + "; ".join(errors)
    return "VALIDATION PASSED"


@tool
def validate_waveform(
    kind: Annotated[str, "Waveform type (e.g. 'ricker', 'gaussian')"],
) -> str:
    """Validate that the waveform kind is supported by gprMax."""
    if kind not in VALID_WAVEFORMS:
        return (
            f"VALIDATION FAILED: Unsupported waveform kind '{kind}'. "
            f"Must be one of: {', '.join(sorted(VALID_WAVEFORMS))}"
        )
    return "VALIDATION PASSED"


@tool
def validate_model(
    model: Annotated[str, "Dielectric model name (peplinski, dobson, mironov, crim)"],
    f0: Annotated[float, "Centre frequency in Hz"],
    theta_v: Annotated[float, "Volumetric water content (0.0-1.0)"],
    sand_pct: Annotated[float, "Sand percentage (0-100)"],
    silt_pct: Annotated[float, "Silt percentage (0-100)"],
    clay_pct: Annotated[float, "Clay percentage (0-100)"],
) -> str:
    """Validate that layer properties fall within the valid range for the
    chosen dielectric model."""
    errors = []
    if model == "peplinski":
        if not (0.3e9 <= f0 <= 1.3e9):
            errors.append("Peplinski valid for ~0.3-1.3 GHz")
        if not (0.0 <= theta_v <= 0.30):
            errors.append("Peplinski moisture valid ~0-0.30")
        if not (15 <= sand_pct <= 50 and 5 <= clay_pct <= 20 and 35 <= silt_pct <= 65):
            errors.append("Peplinski texture ranges: sand 15-50%, clay 5-20%, silt 35-65%")
    elif model == "dobson":
        if not (1.4e9 <= f0 <= 18e9):
            errors.append("Dobson valid for ~1.4-18 GHz")
        if not (0.0 <= theta_v <= 0.50):
            errors.append("Dobson moisture valid ~0-0.50")
    elif model == "mironov":
        if not (0.6e9 <= f0 <= 18e9):
            errors.append("Mironov valid for ~0.6-18 GHz")
        if not (0.0 <= theta_v <= 0.45):
            errors.append("Mironov moisture valid ~0-0.45")
    elif model == "crim":
        pass
    else:
        errors.append(f"Unknown model '{model}'")
    if errors:
        return "VALIDATION FAILED: " + "; ".join(errors)
    return "VALIDATION PASSED"


@tool
def validate_custom_material(
    eps_r: Annotated[float, "Relative permittivity"],
    sigma: Annotated[float, "Electrical conductivity (S/m)"],
    mu_r: Annotated[float, "Relative permeability"],
    sigma_m: Annotated[float, "Magnetic conductivity (Ohm/m)"],
) -> str:
    """Validate custom material electromagnetic properties."""
    errors = []
    if eps_r < 1.0:
        errors.append("eps_r must be >= 1.0")
    if sigma < 0.0:
        errors.append("sigma must be >= 0")
    if mu_r < 1.0:
        errors.append("mu_r must be >= 1.0")
    if sigma_m < 0.0:
        errors.append("sigma_m must be >= 0")
    if errors:
        return "VALIDATION FAILED: " + "; ".join(errors)
    return "VALIDATION PASSED"


@tool
def validate_antenna(
    kind: Annotated[str, "Antenna type ('hertzian_dipole' or 'voltage_source')"],
    axis: Annotated[str, "Polarisation axis ('x', 'y', or 'z')"],
    source_start_time: Annotated[Optional[float], "Source start time (s)"] = None,
    source_end_time: Annotated[Optional[float], "Source end time (s)"] = None,
) -> str:
    """Validate antenna configuration."""
    errors = []
    if kind.lower() not in {"hertzian_dipole", "voltage_source"}:
        errors.append("Only 'hertzian_dipole' and 'voltage_source' are supported")
    if axis.lower() not in {"x", "y", "z"}:
        errors.append("Axis must be 'x', 'y', or 'z'")
    if source_start_time is not None and source_end_time is not None:
        if source_start_time >= source_end_time:
            errors.append("source_start_time must be < source_end_time")
    if errors:
        return "VALIDATION FAILED: " + "; ".join(errors)
    return "VALIDATION PASSED"


@tool
def validate_cylinder(
    name: Annotated[str, "Cylinder name"],
    radius: Annotated[float, "Cylinder radius in metres"],
    material: Annotated[str, "Material identifier"],
    has_custom_material: Annotated[bool, "Whether a custom material is defined"] = False,
) -> str:
    """Validate a cylinder geometry object."""
    errors = []
    if radius <= 0:
        errors.append(f"Cylinder '{name}': radius must be positive")
    if not material:
        errors.append(f"Cylinder '{name}': material must be a non-empty string")
    if material == "custom" and not has_custom_material:
        errors.append(f"Cylinder '{name}': custom_material required when material='custom'")
    if errors:
        return "VALIDATION FAILED: " + "; ".join(errors)
    return "VALIDATION PASSED"


@tool
def validate_box(
    name: Annotated[str, "Box name"],
    x1: Annotated[float, "X start coordinate"],
    y1: Annotated[float, "Y start coordinate"],
    z1: Annotated[float, "Z start coordinate"],
    x2: Annotated[float, "X end coordinate"],
    y2: Annotated[float, "Y end coordinate"],
    z2: Annotated[float, "Z end coordinate"],
    material: Annotated[str, "Material identifier"],
    has_custom_material: Annotated[bool, "Whether a custom material is defined"] = False,
) -> str:
    """Validate a box geometry object."""
    errors = []
    if x2 <= x1:
        errors.append(f"Box '{name}': x2 must be > x1")
    if y2 <= y1:
        errors.append(f"Box '{name}': y2 must be > y1")
    if z2 <= z1:
        errors.append(f"Box '{name}': z2 must be > z1")
    if not material:
        errors.append(f"Box '{name}': material must be a non-empty string")
    if material == "custom" and not has_custom_material:
        errors.append(f"Box '{name}': custom_material required when material='custom'")
    if errors:
        return "VALIDATION FAILED: " + "; ".join(errors)
    return "VALIDATION PASSED"


@tool
def validate_sphere(
    name: Annotated[str, "Sphere name"],
    radius: Annotated[float, "Sphere radius in metres"],
    material: Annotated[str, "Material identifier"],
    has_custom_material: Annotated[bool, "Whether a custom material is defined"] = False,
) -> str:
    """Validate a sphere geometry object."""
    errors = []
    if radius <= 0:
        errors.append(f"Sphere '{name}': radius must be positive")
    if not material:
        errors.append(f"Sphere '{name}': material must be a non-empty string")
    if material == "custom" and not has_custom_material:
        errors.append(f"Sphere '{name}': custom_material required when material='custom'")
    if errors:
        return "VALIDATION FAILED: " + "; ".join(errors)
    return "VALIDATION PASSED"


@tool
def validate_surface(
    fractal_dim: Annotated[float, "Fractal dimension"],
    weight_x: Annotated[float, "Weight in X direction"],
    weight_y: Annotated[float, "Weight in Y direction"],
    amplitude_m: Annotated[float, "Roughness amplitude in metres"],
    add_water: Annotated[bool, "Whether water layer is added"],
    water_depth_m: Annotated[float, "Water depth in metres"] = 0.005,
) -> str:
    """Validate surface roughness configuration."""
    errors = []
    if fractal_dim < 0:
        errors.append("fractal_dim must be >= 0")
    if weight_x < 0:
        errors.append("weight_x must be >= 0")
    if weight_y < 0:
        errors.append("weight_y must be >= 0")
    if amplitude_m <= 0:
        errors.append("amplitude_m must be > 0")
    if add_water and water_depth_m >= amplitude_m:
        errors.append("water_depth_m must be < amplitude_m when add_water=True")
    if add_water and water_depth_m <= 0:
        errors.append("water_depth_m must be > 0 when add_water=True")
    if errors:
        return "VALIDATION FAILED: " + "; ".join(errors)
    return "VALIDATION PASSED"


@tool
def validate_rxarray(
    x1: Annotated[float, "X start"], y1: Annotated[float, "Y start"], z1: Annotated[float, "Z start"],
    x2: Annotated[float, "X end"], y2: Annotated[float, "Y end"], z2: Annotated[float, "Z end"],
    dx: Annotated[float, "X step"], dy: Annotated[float, "Y step"], dz: Annotated[float, "Z step"],
) -> str:
    """Validate receiver array configuration."""
    errors = []
    if dx <= 0:
        errors.append("dx must be > 0")
    if dy <= 0:
        errors.append("dy must be > 0")
    if dz <= 0:
        errors.append("dz must be > 0")
    if errors:
        return "VALIDATION FAILED: " + "; ".join(errors)
    return "VALIDATION PASSED"


@tool
def validate_snapshot(
    time_s: Annotated[float, "Snapshot time in seconds"],
) -> str:
    """Validate snapshot configuration."""
    if time_s <= 0:
        return "VALIDATION FAILED: time_s must be > 0"
    return "VALIDATION PASSED"


agent = create_deep_agent(
    model=llm,
    system_prompt=VALIDATION_AGENT_PROMPT,
    checkpointer=InMemorySaver(),
    tools=[ patch_parameters]
)
