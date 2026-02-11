"""
Validation Error Translator - Mediator between technical validation and user-friendly feedback.

This module bridges the gap between:
1. User-friendly input terms (sand, loam, wet, dry, etc.)
2. Technical parameter names (theta_v, sand_pct, bulk_density_gcm3, etc.)

It translates cryptic validation errors into actionable, user-understandable messages.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from soil_setup.soil_lookup_table import (
    TEXTURE_DEFAULTS,
    THETA_V_BY_TEXTURE_AND_STATE,
    BULK_DENSITY_PRIOR,
    ORGANIC_FRACTION_BY_LEVEL,
    POREWATER_SIGMA_PRIOR,
    ANTENNA_PRESET_TO_FREQ_HZ,
    QUALITY_TO_MESH,
)


@dataclass
class UserFriendlyError:
    """A user-friendly error with context and suggestions."""
    field: str                    # User-friendly field name
    message: str                  # User-friendly error message
    user_input: Optional[str]     # What the user originally provided
    suggestion: Optional[str]     # Actionable suggestion to fix
    valid_options: Optional[List[str]] = None  # Valid choices if applicable


# Mapping from technical parameter names to user-friendly descriptions
PARAM_TO_USER_FRIENDLY: Dict[str, Dict[str, str]] = {
    # Layer parameters
    "thickness_m": {
        "name": "Layer Thickness",
        "description": "How thick the soil layer is (in meters)",
        "example": "e.g., '0.5 meters' or '50 cm'",
    },
    "texture_class": {
        "name": "Soil Type",
        "description": "The type of soil (sand, loam, clay, etc.)",
        "example": "e.g., 'sandy loam', 'clay', 'silt'",
    },
    "moisture_state": {
        "name": "Moisture Level",
        "description": "How wet the soil is",
        "example": "e.g., 'dry', 'normal', 'wet', or 'saturated'",
    },
    "organic_level": {
        "name": "Organic Content",
        "description": "Amount of organic matter in the soil",
        "example": "e.g., 'none', 'low', 'moderate', or 'high/peaty'",
    },
    "salinity_environment": {
        "name": "Salinity",
        "description": "Salt content in the soil water",
        "example": "e.g., 'fresh', 'slightly saline', 'brackish'",
    },
    "compaction_level": {
        "name": "Soil Compaction",
        "description": "How compacted/dense the soil is",
        "example": "e.g., 'loose', 'normal', or 'compacted'",
    },
    
    # Technical parameters (derived from user inputs)
    "theta_v": {
        "name": "Water Content",
        "description": "Volumetric water content (derived from moisture level)",
        "derived_from": "moisture_state + texture_class",
    },
    "sand_pct": {
        "name": "Sand Percentage",
        "description": "Percentage of sand in soil (derived from soil type)",
        "derived_from": "texture_class",
    },
    "silt_pct": {
        "name": "Silt Percentage", 
        "description": "Percentage of silt in soil (derived from soil type)",
        "derived_from": "texture_class",
    },
    "clay_pct": {
        "name": "Clay Percentage",
        "description": "Percentage of clay in soil (derived from soil type)",
        "derived_from": "texture_class",
    },
    "bulk_density_gcm3": {
        "name": "Soil Density",
        "description": "How dense the soil is (derived from compaction and soil type)",
        "derived_from": "compaction_level + texture_class",
    },
    "porewater_sigma_Sm": {
        "name": "Water Conductivity",
        "description": "Electrical conductivity of soil water (derived from salinity)",
        "derived_from": "salinity_environment",
    },
    
    # Model parameters
    "survey_length_m": {
        "name": "Survey Length",
        "description": "How long the GPR survey line is (in meters)",
        "example": "e.g., '10 meters' or '20m'",
    },
    "max_depth_m": {
        "name": "Maximum Depth",
        "description": "How deep the simulation should go (in meters)",
        "example": "e.g., '2 meters' or '3m'",
    },
    "antenna_height_m": {
        "name": "Antenna Height",
        "description": "Height of antenna above ground (in meters)",
        "example": "e.g., '0.02m' for ground-coupled, '0.5m' for air-launched",
    },
    "temperature_c": {
        "name": "Temperature",
        "description": "Soil temperature in Celsius",
        "example": "e.g., '20°C' or '25 degrees'",
    },
    "quality": {
        "name": "Simulation Quality",
        "description": "Trade-off between speed and accuracy",
        "example": "'fast', 'balanced', or 'high_accuracy'",
    },
    
    # Antenna parameters
    "preset": {
        "name": "Antenna Frequency",
        "description": "The GPR antenna frequency",
        "example": "e.g., '400 MHz', '800 MHz', '1 GHz'",
    },
    "axis": {
        "name": "Antenna Orientation",
        "description": "Direction the antenna is oriented",
        "example": "'x', 'y', or 'z'",
    },
}


# Valid options for categorical fields
VALID_OPTIONS: Dict[str, List[str]] = {
    "texture_class": [
        "sand", "loamy_sand", "sandy_loam", "loam", "silt_loam", "silt",
        "sandy_clay_loam", "clay_loam", "silty_clay_loam", "sandy_clay", 
        "silty_clay", "clay"
    ],
    "moisture_state": ["dry", "normal", "wet", "saturated"],
    "organic_level": ["none", "low", "moderate", "high_peaty"],
    "salinity_environment": ["fresh", "slightly_saline", "brackish", "seawater"],
    "compaction_level": ["loose", "normal", "compacted"],
    "quality": ["fast", "balanced", "high_accuracy"],
    "preset": ["generic_200MHz", "generic_400MHz", "generic_800MHz", "generic_1GHz", 
               "generic_1.2GHz", "generic_1.5GHz"],
    "axis": ["x", "y", "z"],
}


# User-friendly names for texture classes
TEXTURE_FRIENDLY_NAMES: Dict[str, str] = {
    "sand": "Sand",
    "loamy_sand": "Loamy Sand",
    "sandy_loam": "Sandy Loam",
    "loam": "Loam",
    "silt_loam": "Silt Loam",
    "silt": "Silt",
    "sandy_clay_loam": "Sandy Clay Loam",
    "clay_loam": "Clay Loam",
    "silty_clay_loam": "Silty Clay Loam",
    "sandy_clay": "Sandy Clay",
    "silty_clay": "Silty Clay",
    "clay": "Clay",
}


class ValidationErrorTranslator:
    """
    Translates technical validation errors into user-friendly messages.
    
    This class understands:
    - The mapping between user inputs (sand, wet, etc.) and derived parameters
    - How to suggest fixes based on the user's original terminology
    - Physical constraints and their implications
    """
    
    def __init__(self, user_context: Optional[Dict[str, Any]] = None):
        """
        Initialize translator with optional user context.
        
        Args:
            user_context: Dict containing user's original inputs for reference
        """
        self.user_context = user_context or {}
    
    def translate_error(
        self, 
        param_path: str, 
        error_message: str,
        layer_data: Optional[Dict] = None
    ) -> UserFriendlyError:
        """
        Translate a single validation error to user-friendly format.
        
        Args:
            param_path: The parameter path (e.g., "layer_0.thickness_m")
            error_message: The technical error message
            layer_data: Optional layer data for context
            
        Returns:
            UserFriendlyError with user-friendly message and suggestions
        """
        # Parse the parameter path
        parts = param_path.split(".")
        
        # Determine if this is a layer-specific error
        layer_num = None
        param_name = parts[-1]
        
        if len(parts) >= 2 and parts[0].startswith("layer"):
            try:
                layer_num = int(parts[0].replace("layer_", "")) + 1
            except ValueError:
                layer_num = None
            param_name = parts[-1]
        
        # Get user-friendly info for this parameter
        param_info = PARAM_TO_USER_FRIENDLY.get(param_name, {})
        friendly_name = param_info.get("name", param_name.replace("_", " ").title())
        
        # Build the field name with layer context
        if layer_num:
            field = f"Layer {layer_num} - {friendly_name}"
        else:
            field = friendly_name
        
        # Translate the error message
        translated_message, suggestion = self._translate_message(
            param_name, error_message, layer_data
        )
        
        # Get valid options if applicable
        valid_options = VALID_OPTIONS.get(param_name)
        
        # Get user's original input if available
        user_input = None
        if layer_data:
            user_input = layer_data.get(param_name)
        
        return UserFriendlyError(
            field=field,
            message=translated_message,
            user_input=str(user_input) if user_input else None,
            suggestion=suggestion,
            valid_options=valid_options,
        )
    
    def _translate_message(
        self, 
        param_name: str, 
        error_message: str,
        layer_data: Optional[Dict] = None
    ) -> Tuple[str, Optional[str]]:
        """
        Translate a technical error message to user-friendly language.
        
        Returns:
            Tuple of (translated_message, suggestion)
        """
        error_lower = error_message.lower()
        
        # Handle thickness errors
        if param_name == "thickness_m":
            if "greater than 0" in error_lower or "gt=0" in error_lower:
                return (
                    "The layer thickness must be a positive number.",
                    "Please specify how thick this layer is (e.g., '0.5 meters' or '50 cm')."
                )
            if "required" in error_lower:
                return (
                    "Please specify how thick this soil layer is.",
                    "For example: '30 cm thick' or '0.5 meters'."
                )
        
        # Handle texture class errors
        if param_name == "texture_class":
            if "required" in error_lower or "missing" in error_lower:
                return (
                    "Please specify what type of soil this layer is.",
                    "Common options: sand, sandy loam, loam, clay loam, clay, etc."
                )
            if "invalid" in error_lower or "not in" in error_lower:
                user_value = layer_data.get("texture_class") if layer_data else "unknown"
                friendly_options = [TEXTURE_FRIENDLY_NAMES.get(t, t) for t in VALID_OPTIONS["texture_class"]]
                return (
                    f"'{user_value}' is not a recognized soil type.",
                    f"Please use one of: {', '.join(friendly_options[:6])}... (and more)"
                )
        
        # Handle moisture state errors
        if param_name == "moisture_state":
            if "invalid" in error_lower:
                return (
                    "The moisture level you specified isn't recognized.",
                    "Please use: 'dry', 'normal', 'wet', or 'saturated'."
                )
        
        # Handle derived parameter errors (theta_v, sand_pct, etc.)
        if param_name in ["theta_v", "theta_v_override"]:
            if "0" in error_lower and "0.9" in error_lower:
                return (
                    "The water content value is outside the valid range (0-90%).",
                    "Try adjusting the moisture level (dry/normal/wet/saturated) instead."
                )
        
        if param_name in ["sand_pct", "silt_pct", "clay_pct"]:
            if "sum" in error_lower or "100" in error_lower:
                return (
                    "The sand, silt, and clay percentages must add up to 100%.",
                    "Tip: Just specify the soil type (e.g., 'sandy loam') and we'll use standard values."
                )
        
        # Handle model parameter errors
        if param_name == "survey_length_m":
            if "greater than 0" in error_lower:
                return (
                    "The survey length must be a positive number.",
                    "Please specify how long your GPR survey line should be (e.g., '10 meters')."
                )
        
        if param_name == "max_depth_m":
            if "greater than 0" in error_lower:
                return (
                    "The maximum depth must be a positive number.",
                    "Please specify how deep you want to simulate (e.g., '2 meters')."
                )
        
        if param_name == "antenna_height_m":
            if "between" in error_lower or "0" in error_lower and "1" in error_lower:
                return (
                    "The antenna height should be between 0 and 1 meter.",
                    "Use 0.02m for ground-coupled antennas, or up to 1m for air-launched."
                )
        
        if param_name == "temperature_c":
            if "-20" in error_lower or "60" in error_lower:
                return (
                    "The temperature should be between -20°C and 60°C.",
                    "Typical soil temperatures are between 10°C and 30°C."
                )
        
        # Handle antenna preset errors
        if param_name == "preset":
            if "invalid" in error_lower:
                return (
                    "The antenna frequency you specified isn't available.",
                    "Available options: 200 MHz, 400 MHz, 800 MHz, 1 GHz."
                )
        
        # Default translation
        return (error_message, None)
    
    def translate_all_errors(
        self, 
        validation_errors: Dict[str, str],
        state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, UserFriendlyError]:
        """
        Translate all validation errors to user-friendly format.
        
        Args:
            validation_errors: Dict of param_path -> error_message
            state: Optional full state for context
            
        Returns:
            Dict of param_path -> UserFriendlyError
        """
        translated = {}
        layers = state.get("layers", []) if state else []
        
        for param_path, error_message in validation_errors.items():
            # Get layer data if this is a layer error
            layer_data = None
            if param_path.startswith("layer_"):
                try:
                    layer_idx = int(param_path.split(".")[0].replace("layer_", ""))
                    if layer_idx < len(layers):
                        layer_data = layers[layer_idx]
                except (ValueError, IndexError):
                    pass
            
            translated[param_path] = self.translate_error(
                param_path, error_message, layer_data
            )
        
        return translated
    
    def format_user_message(
        self, 
        translated_errors: Dict[str, UserFriendlyError]
    ) -> str:
        """
        Format translated errors into a user-friendly message.
        
        Args:
            translated_errors: Dict of translated errors
            
        Returns:
            Formatted string message for the user
        """
        if not translated_errors:
            return "All parameters look good!"
        
        lines = ["I found some issues with your inputs:\n"]
        
        for param_path, error in translated_errors.items():
            lines.append(f"**{error.field}**")
            lines.append(f"  • Issue: {error.message}")
            
            if error.user_input:
                lines.append(f"  • You provided: '{error.user_input}'")
            
            if error.suggestion:
                lines.append(f"  • Suggestion: {error.suggestion}")
            
            if error.valid_options and len(error.valid_options) <= 6:
                options_str = ", ".join(error.valid_options)
                lines.append(f"  • Valid options: {options_str}")
            
            lines.append("")  # Empty line between errors
        
        lines.append("Please provide corrected values for the issues above.")
        
        return "\n".join(lines)
    
    @staticmethod
    def explain_derived_parameter(param_name: str, layer_data: Dict) -> str:
        """
        Explain how a derived parameter was calculated from user inputs.
        
        Args:
            param_name: The derived parameter name
            layer_data: The layer data with user inputs
            
        Returns:
            Explanation string
        """
        texture = layer_data.get("texture_class", "unknown")
        moisture = layer_data.get("moisture_state", "normal")
        compaction = layer_data.get("compaction_level", "normal")
        organic = layer_data.get("organic_level", "none")
        salinity = layer_data.get("salinity_environment", "fresh")
        
        if param_name == "theta_v":
            theta = THETA_V_BY_TEXTURE_AND_STATE.get(texture, {}).get(moisture, "N/A")
            return (
                f"Water content ({theta}) was calculated from:\n"
                f"  • Soil type: {TEXTURE_FRIENDLY_NAMES.get(texture, texture)}\n"
                f"  • Moisture level: {moisture}\n"
                f"To change this, adjust the moisture level or provide a specific override."
            )
        
        if param_name in ["sand_pct", "silt_pct", "clay_pct"]:
            defaults = TEXTURE_DEFAULTS.get(texture, (0, 0, 0))
            return (
                f"Texture fractions were derived from soil type '{TEXTURE_FRIENDLY_NAMES.get(texture, texture)}':\n"
                f"  • Sand: {defaults[0]}%\n"
                f"  • Silt: {defaults[1]}%\n"
                f"  • Clay: {defaults[2]}%\n"
                f"To change these, either pick a different soil type or provide specific percentages."
            )
        
        if param_name == "bulk_density_gcm3":
            return (
                f"Soil density was calculated from:\n"
                f"  • Soil type: {TEXTURE_FRIENDLY_NAMES.get(texture, texture)}\n"
                f"  • Compaction: {compaction}\n"
                f"  • Organic content: {organic}\n"
                f"To change this, adjust compaction level or provide a specific density value."
            )
        
        if param_name == "porewater_sigma_Sm":
            sigma = POREWATER_SIGMA_PRIOR.get(salinity, "N/A")
            return (
                f"Water conductivity ({sigma} S/m) was derived from salinity: {salinity}\n"
                f"To change this, adjust the salinity environment or provide a specific value."
            )
        
        return f"Parameter '{param_name}' is derived from your input settings."


def translate_validation_errors(
    validation_errors: Dict[str, str],
    state: Optional[Dict[str, Any]] = None
) -> str:
    """
    Convenience function to translate validation errors to a user-friendly message.
    
    Args:
        validation_errors: Dict of param_path -> error_message
        state: Optional full state for context
        
    Returns:
        User-friendly error message string
    """
    translator = ValidationErrorTranslator()
    translated = translator.translate_all_errors(validation_errors, state)
    return translator.format_user_message(translated)

