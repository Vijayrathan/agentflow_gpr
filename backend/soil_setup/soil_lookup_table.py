from __future__ import annotations

from typing import List, Literal, Optional, Dict, Tuple


# -----------------------------
# 3) LOOKUP TABLES (DEFAULT PRIORS)
# -----------------------------

# Representative point inside each USDA class (not for classification; for forward sim defaults)
# You can tune these later. These are "safe" midpoints, not hard truths.
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
}

ANTENNA_PRESET_TO_TXRX_OFFSET_M: Dict[str, float] = {
    # Simple geometry prior; adjust to your instrument library later
    "generic_200MHz": 0.25,
    "generic_400MHz": 0.15,
    "generic_800MHz": 0.08,
    "generic_1GHz": 0.05,
}

QUALITY_TO_MESH: Dict[str, Tuple[int, float]] = {
    # (cells_per_wavelength, max_cell_m_cap)
    "fast": (10, 0.03),
    "balanced": (15, 0.02),
    "high_accuracy": (20, 0.01),
}
