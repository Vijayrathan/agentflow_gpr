
from __future__ import annotations
from typing import List, Literal, Optional, Dict, Tuple



# -----------------------------
# 3) LOOKUP TABLES (DEFAULT PRIORS)
# -----------------------------

# Representative point inside each USDA class (not for classification; for forward sim defaults)
# These are standard USDA texture class midpoints.
TEXTURE_DEFAULTS: Dict[str, Tuple[float, float, float]] = {
    "sand": (92.0, 5.0, 3.0),
    "loamy_sand": (80.0, 12.0, 8.0),
    "sandy_loam": (60.0, 30.0, 10.0),
    "loam": (40.0, 39.0, 21.0),
    "silt_loam": (15.0, 65.0, 20.0),
    "silt": (10.0, 85.0, 5.0),
    "sandy_clay_loam": (55.0, 15.0, 30.0),
    "clay_loam": (33.0, 32.0, 35.0),
    "silty_clay_loam": (10.0, 55.0, 35.0),
    "sandy_clay": (55.0, 5.0, 40.0),
    "silty_clay": (10.0, 45.0, 45.0),
    "clay": (20.0, 20.0, 60.0),
}

# Moisture priors by broad bucket (you can refine; keep user-friendly)
# Values are typical midpoints for simulation defaults.
THETA_V_BY_TEXTURE_AND_STATE: Dict[str, Dict[str, float]] = {
    # sandy-ish
    "sand": {"dry": 0.05, "normal": 0.12, "wet": 0.22, "saturated": 0.30},
    "loamy_sand": {"dry": 0.06, "normal": 0.14, "wet": 0.24, "saturated": 0.32},
    "sandy_loam": {"dry": 0.08, "normal": 0.18, "wet": 0.28, "saturated": 0.36},

    # loam-ish
    "loam": {"dry": 0.10, "normal": 0.22, "wet": 0.32, "saturated": 0.42},
    "silt_loam": {"dry": 0.12, "normal": 0.25, "wet": 0.35, "saturated": 0.45},
    "silt": {"dry": 0.12, "normal": 0.26, "wet": 0.36, "saturated": 0.46},

    # clay-ish
    "sandy_clay_loam": {"dry": 0.14, "normal": 0.28, "wet": 0.38, "saturated": 0.48},
    "clay_loam": {"dry": 0.15, "normal": 0.30, "wet": 0.40, "saturated": 0.50},
    "silty_clay_loam": {"dry": 0.16, "normal": 0.32, "wet": 0.42, "saturated": 0.52},
    "sandy_clay": {"dry": 0.16, "normal": 0.32, "wet": 0.42, "saturated": 0.52},
    "silty_clay": {"dry": 0.18, "normal": 0.35, "wet": 0.45, "saturated": 0.55},
    "clay": {"dry": 0.18, "normal": 0.35, "wet": 0.45, "saturated": 0.55},
}

# Particle density: constant for mineral soils; reduce for organic-rich soils a bit
PARTICLE_DENSITY_DEFAULT = 2.65  # g/cm3

ORGANIC_FRACTION_BY_LEVEL: Dict[str, float] = {
    "none": 0.00,
    "low": 0.02,
    "moderate": 0.05,
    "high_peaty": 0.20,
}

# Bulk density by texture bucket + compaction (simple prior)
# These are modeling defaults; in reality it varies widely.
BULK_DENSITY_PRIOR: Dict[str, Dict[str, float]] = {
    "sandy": {"loose": 1.40, "normal": 1.55, "compacted": 1.70},
    "loamy": {"loose": 1.20, "normal": 1.35, "compacted": 1.55},
    "clayey": {"loose": 1.10, "normal": 1.25, "compacted": 1.45},
    "organic": {"loose": 0.75, "normal": 0.90, "compacted": 1.05},
}

def _texture_bucket(texture_class: str, organic_level: str) -> str:
    if organic_level == "high_peaty":
        return "organic"
    if texture_class in ("sand", "loamy_sand", "sandy_loam"):
        return "sandy"
    if texture_class in ("loam", "silt_loam", "silt"):
        return "loamy"
    return "clayey"

SALINITY_CLASS_MAP: Dict[str, str] = {
    "fresh": "fresh",
    "slightly_saline": "slightly_saline",
    "brackish": "brackish",
    "seawater": "seawater",
}

# Porewater conductivity sigma (S/m) coarse priors by environment
POREWATER_SIGMA_PRIOR: Dict[str, float] = {
    "fresh": 0.01,
    "slightly_saline": 0.1,
    "brackish": 1.0,
    "seawater": 4.0,
}

ANTENNA_PRESET_TO_FREQ_HZ: Dict[str, float] = {
    "generic_200MHz": 200e6,
    "generic_400MHz": 400e6,
    "generic_800MHz": 800e6,
    "generic_1GHz": 1_000e6,
    "generic_1.2GHz": 1_200e6,
    "generic_1.5GHz": 1_500e6,
}

ANTENNA_PRESET_TO_TXRX_OFFSET_M: Dict[str, float] = {
    # Simple geometry prior; adjust to your instrument library later
    "generic_200MHz": 0.25,
    "generic_400MHz": 0.15,
    "generic_800MHz": 0.08,
    "generic_1GHz": 0.05,
    "generic_1.2GHz": 0.04,
    "generic_1.5GHz": 0.03,
}

QUALITY_TO_MESH: Dict[str, Tuple[int, float]] = {
    # (cells_per_wavelength, max_cell_m_cap)
    "fast": (10, 0.03),
    "balanced": (15, 0.02),
    "high_accuracy": (20, 0.01),
}

# -----------------------------
# 4) DIELECTRIC MODEL CONSTRAINTS
# -----------------------------

# Soil dielectric model types
SoilDielectricModel = Literal["peplinski", "dobson", "mironov", "crim"]

# Model validity constraints
# Each model has specific validity ranges for frequency, moisture, and texture
MODEL_CONSTRAINTS: Dict[str, Dict[str, any]] = {
    "peplinski": {
        "name": "Peplinski (1995)",
        "freq_hz_range": (0.3e9, 1.3e9),
        "theta_v_range": (0.0, 0.30),
        "sand_pct_range": (15.0, 50.0),
        "clay_pct_range": (5.0, 20.0),
        "silt_pct_range": (35.0, 65.0),
        "description": "Semi-empirical model for GPR frequencies (0.3-1.3 GHz). Native gprMax support.",
        "gprmax_native": True,
        "gprmax_command": "#soil_peplinski",
    },
    "dobson": {
        "name": "Dobson (1985)",
        "freq_hz_range": (1.4e9, 18e9),
        "theta_v_range": (0.0, 0.50),
        "sand_pct_range": None,  # No texture constraints
        "clay_pct_range": None,
        "silt_pct_range": None,
        "description": "Four-component mixing model for higher frequencies (1.4-18 GHz). Requires Python block.",
        "gprmax_native": False,
        "gprmax_command": None,
    },
    "mironov": {
        "name": "Mironov (2009)",
        "freq_hz_range": (0.6e9, 18e9),  # 0.6-18 GHz per original paper
        "theta_v_range": (0.0, 0.45),
        "sand_pct_range": None,  # No texture constraints
        "clay_pct_range": None,
        "silt_pct_range": None,
        "description": "Generalized refractive mixing model with bound/free water. Wide frequency range (0.6-18 GHz). Requires Python block.",
        "gprmax_native": False,
        "gprmax_command": None,
    },
    "crim": {
        "name": "CRIM (Complex Refractive Index Model)",
        "freq_hz_range": (0.0, float('inf')),  # No frequency constraints
        "theta_v_range": (0.0, 0.90),  # Porosity-limited
        "sand_pct_range": None,  # No texture constraints
        "clay_pct_range": None,
        "silt_pct_range": None,
        "description": "Simple volumetric mixing model. No frequency/texture constraints. Good for exploratory simulations. Requires Python block.",
        "gprmax_native": False,
        "gprmax_command": None,
    },
}


def get_model_description(model: str) -> str:
    """Get a user-friendly description of a soil dielectric model."""
    if model in MODEL_CONSTRAINTS:
        return MODEL_CONSTRAINTS[model]["description"]
    return f"Unknown model: {model}"


def check_model_compatibility(
    model: str,
    texture_class: str,
    moisture_state: str,
    center_freq_hz: float,
) -> Tuple[bool, Optional[str]]:
    """
    Check if a soil dielectric model is compatible with the given parameters.
    
    Returns:
        (is_compatible, error_message) - error_message is None if compatible
    """
    if model not in MODEL_CONSTRAINTS:
        return False, f"Unknown model: {model}"
    
    constraints = MODEL_CONSTRAINTS[model]
    errors = []
    
    # Check frequency range
    freq_min, freq_max = constraints["freq_hz_range"]
    if not (freq_min <= center_freq_hz <= freq_max):
        freq_min_ghz = freq_min / 1e9
        freq_max_ghz = freq_max / 1e9
        freq_ghz = center_freq_hz / 1e9
        errors.append(
            f"Frequency {freq_ghz:.2f} GHz outside valid range "
            f"({freq_min_ghz:.1f}-{freq_max_ghz:.1f} GHz)"
        )
    
    # Check moisture range
    theta_v = THETA_V_BY_TEXTURE_AND_STATE.get(texture_class, {}).get(moisture_state)
    if theta_v is not None:
        theta_min, theta_max = constraints["theta_v_range"]
        if not (theta_min <= theta_v <= theta_max):
            errors.append(
                f"Moisture {theta_v:.2f} outside valid range "
                f"({theta_min:.2f}-{theta_max:.2f})"
            )
    
    # Check texture constraints (only for Peplinski)
    if constraints.get("sand_pct_range") is not None:
        sand, silt, clay = TEXTURE_DEFAULTS.get(texture_class, (0, 0, 0))
        
        sand_min, sand_max = constraints["sand_pct_range"]
        clay_min, clay_max = constraints["clay_pct_range"]
        silt_min, silt_max = constraints["silt_pct_range"]
        
        if not (sand_min <= sand <= sand_max):
            errors.append(f"Sand {sand:.0f}% outside range ({sand_min:.0f}-{sand_max:.0f}%)")
        if not (clay_min <= clay <= clay_max):
            errors.append(f"Clay {clay:.0f}% outside range ({clay_min:.0f}-{clay_max:.0f}%)")
        if not (silt_min <= silt <= silt_max):
            errors.append(f"Silt {silt:.0f}% outside range ({silt_min:.0f}-{silt_max:.0f}%)")
    
    if errors:
        return False, "; ".join(errors)
    
    return True, None


def get_compatible_models(
    texture_class: str,
    moisture_state: str,
    center_freq_hz: float,
) -> List[str]:
    """
    Get list of compatible soil dielectric models for given parameters.
    """
    compatible = []
    for model_name in MODEL_CONSTRAINTS:
        is_ok, _ = check_model_compatibility(
            model_name, texture_class, moisture_state, center_freq_hz
        )
        if is_ok:
            compatible.append(model_name)
    return compatible


def get_user_friendly_model_error(
    model: str,
    texture_class: str,
    moisture_state: str,
    center_freq_hz: float,
) -> str:
    """
    Generate a user-friendly error message when model is incompatible.
    """
    is_compatible, error_msg = check_model_compatibility(
        model, texture_class, moisture_state, center_freq_hz
    )
    
    if is_compatible:
        return ""
    
    # Get compatible alternatives
    compatible = get_compatible_models(texture_class, moisture_state, center_freq_hz)
    
    message = f"The '{model}' model is not compatible with your settings:\n{error_msg}\n\n"
    
    if compatible:
        message += f"Compatible alternatives: {', '.join(compatible)}"
    else:
        message += "Consider adjusting your frequency or using CRIM model (no constraints)."
    
    return message


# -----------------------------
# 5) MATERIAL TYPES FOR OBJECTS
# -----------------------------

# PEC (Perfect Electric Conductor) - infinite conductivity
# Used for metal objects like pipes, tanks, rebar, etc.
# In gprMax, PEC is a special built-in material with ID 'pec'
SPECIAL_MATERIALS: Dict[str, Dict[str, any]] = {
    "pec": {
        "name": "Perfect Electric Conductor",
        "conductivity": float('inf'),
        "relative_permittivity": 1.0,
        "relative_permeability": 1.0,
        "description": "Metallic objects (pipes, tanks, rebar, etc.)",
        "gprmax_id": "pec"
    },
    "free_space": {
        "name": "Free Space",
        "conductivity": 0.0,
        "relative_permittivity": 1.0,
        "relative_permeability": 1.0,
        "description": "Air/vacuum",
        "gprmax_id": "free_space"
    }
}
