"""
Validator Node - Real-time parameter validation.

This node:
1. Checks parameter completeness
2. Validates physics constraints
3. Provides specific feedback on errors (translated to user-friendly language)
4. Routes to collection, resolver, or end based on validation results
"""

from typing import Dict, Any, List, Tuple
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from langgraph_state import SimulationState, update_collected_params
from soil_setup.soil_structure_schema import (
    UserLayerSimple,
    UserAntennaSimple,
    UserWaveformSimple,
    UserModelSimple,
    UserInputSimulation,
)
from soil_setup.validation_translator import (
    ValidationErrorTranslator,
    translate_validation_errors,
)
from soil_setup.soil_lookup_table import (
    check_model_compatibility,
    get_user_friendly_model_error,
    get_compatible_models,
    ANTENNA_PRESET_TO_FREQ_HZ,
)
from init import logger


def validator_node(state: SimulationState) -> Dict[str, Any]:
    """
    Validate collected parameters in real-time.
    
    Args:
        state: Current simulation state
    
    Returns:
        Updated state with validation results
    """
    logger.info("[VALIDATOR] Starting parameter validation...")
    
    collection_stage = state.get("collection_stage", "model")
    messages = state.get("messages", [])
    
    # Update collected params set
    collected = update_collected_params(state)
    
    validation_errors = {}
    parameters_complete = False
    parameters_valid = False
    
    # Stage-specific validation
    if collection_stage == "model":
        valid, errors = _validate_model_params(state)
        if not valid:
            validation_errors.update(errors)
        else:
            # Model stage complete
            logger.info("[VALIDATOR] Model parameters valid")
    
    elif collection_stage == "layers":
        valid, errors = _validate_layer_params(state)
        if not valid:
            validation_errors.update(errors)
        else:
            logger.info("[VALIDATOR] Layer parameters valid")
    
    elif collection_stage == "antenna":
        valid, errors = _validate_antenna_params(state)
        if not valid:
            validation_errors.update(errors)
        else:
            logger.info("[VALIDATOR] Antenna parameters valid")
    
    elif collection_stage == "objects":
        valid, errors = _validate_object_params(state)
        if not valid:
            validation_errors.update(errors)
        else:
            logger.info("[VALIDATOR] Object parameters valid")
    
    elif collection_stage == "complete":
        # Final validation - check everything
        valid, errors = _validate_all_params(state)
        if not valid:
            validation_errors.update(errors)
        else:
            parameters_complete = True
            parameters_valid = True
            logger.info("[VALIDATOR] All parameters complete and valid!")
    
    # Build response based on validation results
    updates = {
        "collected_params": collected,
        "validation_errors": validation_errors,
        "parameters_complete": parameters_complete,
        "parameters_valid": parameters_valid,
    }
    
    # If there are validation errors, provide user-friendly feedback
    if validation_errors:
        # Use the translator to convert technical errors to user-friendly messages
        error_message = translate_validation_errors(validation_errors, state)
        
        updates["messages"] = messages + [AIMessage(content=error_message)]
        updates["awaiting_user_input"] = True
    
    return updates


def _validate_model_params(state: SimulationState) -> Tuple[bool, Dict[str, str]]:
    """Validate model/simulation setup parameters."""
    
    model = state.get("model", {})
    errors = {}
    
    # Check required fields
    if not model.get("title"):
        errors["model.title"] = "Simulation title is required"
    
    if model.get("survey_length_m") is not None:
        if model["survey_length_m"] <= 0:
            errors["model.survey_length_m"] = "Survey length must be greater than 0"
    else:
        errors["model.survey_length_m"] = "Survey length is required"
    
    if model.get("max_depth_m") is not None:
        if model["max_depth_m"] <= 0:
            errors["model.max_depth_m"] = "Max depth must be greater than 0"
    else:
        errors["model.max_depth_m"] = "Max depth is required"
    
    if model.get("antenna_height_m") is not None:
        if model["antenna_height_m"] < 0 or model["antenna_height_m"] > 1.0:
            errors["model.antenna_height_m"] = "Antenna height must be between 0 and 1.0 meters"
    
    if model.get("temperature_c") is not None:
        if model["temperature_c"] < -20 or model["temperature_c"] > 60:
            errors["model.temperature_c"] = "Temperature must be between -20 and 60°C"
    
    quality = model.get("quality", "balanced")
    if quality not in ["fast", "balanced", "high_accuracy"]:
        errors["model.quality"] = f"Quality must be 'fast', 'balanced', or 'high_accuracy', got '{quality}'"
    
    return len(errors) == 0, errors


def _validate_layer_params(state: SimulationState) -> Tuple[bool, Dict[str, str]]:
    """Validate layer parameters."""
    
    num_layers = state.get("num_layers")
    layers = state.get("layers", [])
    errors = {}
    
    if num_layers is None or num_layers <= 0:
        errors["num_layers"] = "Number of layers must be at least 1"
        return False, errors
    
    if len(layers) != num_layers:
        errors["layers"] = f"Expected {num_layers} layer(s), but have data for {len(layers)}"
        return False, errors
    
    # Validate each layer
    valid_textures = [
        "sand", "loamy_sand", "sandy_loam", "loam", "silt_loam", "silt",
        "sandy_clay_loam", "clay_loam", "silty_clay_loam", "sandy_clay", "silty_clay", "clay"
    ]
    valid_moisture = ["dry", "normal", "wet", "saturated"]
    
    for i, layer in enumerate(layers):
        layer_prefix = f"layer_{i}"
        
        # Check required fields
        if not layer.get("thickness_m"):
            errors[f"{layer_prefix}.thickness_m"] = f"Layer {i+1} thickness is required"
        elif layer["thickness_m"] <= 0:
            errors[f"{layer_prefix}.thickness_m"] = f"Layer {i+1} thickness must be > 0"
        
        if not layer.get("texture_class"):
            errors[f"{layer_prefix}.texture_class"] = f"Layer {i+1} texture class is required"
        elif layer["texture_class"] not in valid_textures:
            errors[f"{layer_prefix}.texture_class"] = (
                f"Layer {i+1} texture '{layer['texture_class']}' is invalid. "
                f"Must be one of: {', '.join(valid_textures)}"
            )
        
        moisture = layer.get("moisture_state", "normal")
        if moisture not in valid_moisture:
            errors[f"{layer_prefix}.moisture_state"] = (
                f"Layer {i+1} moisture '{moisture}' is invalid. "
                f"Must be one of: {', '.join(valid_moisture)}"
            )
        
        # Validate overrides if present
        if layer.get("theta_v_override") is not None:
            theta = layer["theta_v_override"]
            if theta < 0 or theta > 0.9:
                errors[f"{layer_prefix}.theta_v_override"] = (
                    f"Layer {i+1} volumetric water content must be between 0 and 0.9"
                )
        
        # Check texture fraction overrides
        if any(layer.get(f"{k}_pct_override") is not None for k in ["sand", "silt", "clay"]):
            sand = layer.get("sand_pct_override")
            silt = layer.get("silt_pct_override")
            clay = layer.get("clay_pct_override")
            
            if not all(v is not None for v in [sand, silt, clay]):
                errors[f"{layer_prefix}.texture_overrides"] = (
                    f"Layer {i+1}: If overriding texture fractions, must provide sand, silt, AND clay percentages"
                )
            else:
                total = sand + silt + clay
                if abs(total - 100.0) > 0.01:
                    errors[f"{layer_prefix}.texture_overrides"] = (
                        f"Layer {i+1}: Sand + silt + clay must sum to 100% (got {total:.1f}%)"
                    )
    
    return len(errors) == 0, errors


def _validate_antenna_params(state: SimulationState) -> Tuple[bool, Dict[str, str]]:
    """Validate antenna parameters."""
    
    antenna = state.get("antenna", {})
    errors = {}
    
    valid_presets = [
        "generic_200MHz", "generic_400MHz", "generic_800MHz", 
        "generic_1GHz", "generic_1.2GHz", "generic_1.5GHz"
    ]
    valid_axes = ["x", "y", "z"]
    
    preset = antenna.get("preset")
    if not preset:
        errors["antenna.preset"] = "Antenna frequency preset is required"
    elif preset not in valid_presets:
        errors["antenna.preset"] = (
            f"Antenna preset '{preset}' is invalid. "
            f"Must be one of: {', '.join(valid_presets)}"
        )
    
    axis = antenna.get("axis", "x")
    if axis not in valid_axes:
        errors["antenna.axis"] = f"Antenna axis '{axis}' is invalid. Must be one of: {', '.join(valid_axes)}"
    
    if antenna.get("tx_rx_offset_m_override") is not None:
        if antenna["tx_rx_offset_m_override"] <= 0:
            errors["antenna.tx_rx_offset_m_override"] = "TX-RX offset must be > 0"
    
    return len(errors) == 0, errors


def _validate_object_params(state: SimulationState) -> Tuple[bool, Dict[str, str]]:
    """Validate buried object parameters."""
    
    objects = state.get("objects", [])
    errors = {}
    
    if not objects:
        # Objects are optional
        return True, {}
    
    for i, obj in enumerate(objects):
        obj_prefix = f"object_{i}"
        obj_type = obj.get("type", "unknown")
        
        # Common validations
        if obj_type == "cylinder":
            if not all(k in obj for k in ["x1", "y1", "z1", "x2", "y2", "z2", "radius"]):
                errors[f"{obj_prefix}"] = f"Cylinder {i+1} missing required coordinates or radius"
            elif obj.get("radius", 0) <= 0:
                errors[f"{obj_prefix}.radius"] = f"Cylinder {i+1} radius must be > 0"
        
        elif obj_type == "box":
            if not all(k in obj for k in ["x1", "y1", "z1", "x2", "y2", "z2"]):
                errors[f"{obj_prefix}"] = f"Box {i+1} missing required coordinates"
            else:
                # Check that x2 > x1, y2 > y1, z2 > z1
                if obj.get("x2", 0) <= obj.get("x1", 0):
                    errors[f"{obj_prefix}.x"] = f"Box {i+1} x2 must be > x1"
                if obj.get("y2", 0) <= obj.get("y1", 0):
                    errors[f"{obj_prefix}.y"] = f"Box {i+1} y2 must be > y1"
                if obj.get("z2", 0) <= obj.get("z1", 0):
                    errors[f"{obj_prefix}.z"] = f"Box {i+1} z2 must be > z1"
        
        # Validate material
        material = obj.get("material", "pec")
        if material not in ["pec", "free_space"]:
            errors[f"{obj_prefix}.material"] = (
                f"Object {i+1} material '{material}' is invalid. Must be 'pec' or 'free_space'"
            )
    
    return len(errors) == 0, errors


def _validate_all_params(state: SimulationState) -> Tuple[bool, Dict[str, str]]:
    """Final validation of all parameters."""
    
    errors = {}
    
    # Validate all sections
    valid_model, model_errors = _validate_model_params(state)
    if not valid_model:
        errors.update(model_errors)
    
    valid_layers, layer_errors = _validate_layer_params(state)
    if not valid_layers:
        errors.update(layer_errors)
    
    valid_antenna, antenna_errors = _validate_antenna_params(state)
    if not valid_antenna:
        errors.update(antenna_errors)
    
    if state.get("objects"):
        valid_objects, object_errors = _validate_object_params(state)
        if not valid_objects:
            errors.update(object_errors)
    
    # Check waveform (usually auto-set, minimal validation)
    waveform = state.get("waveform", {})
    if not waveform.get("kind"):
        errors["waveform.kind"] = "Waveform kind is required"
    
    # Validate model compatibility with soil textures AND frequency
    # This catches errors early with user-friendly messages
    model_data = state.get("model", {})
    layers = state.get("layers", [])
    antenna = state.get("antenna", {})
    
    dielectric_model = model_data.get("model")
    if not dielectric_model:
        errors["model.model"] = "Soil dielectric model is required. Please select: peplinski, mironov, dobson, or crim"
        return False, errors
    
    antenna_preset = antenna.get("preset", "generic_400MHz")
    freq_hz = ANTENNA_PRESET_TO_FREQ_HZ.get(antenna_preset, 400e6)
    
    # Check model-frequency compatibility first (before layer checks)
    compatible_models = get_compatible_models("loam", "normal", freq_hz)  # Use generic texture for freq check
    if dielectric_model not in compatible_models:
        freq_mhz = freq_hz / 1e6
        
        # Build helpful error message
        if dielectric_model == "mironov" and freq_hz < 0.6e9:
            errors["model_frequency"] = (
                f"**Mironov model** requires frequency ≥ 600 MHz.\n"
                f"Your antenna is set to {freq_mhz:.0f} MHz.\n\n"
                f"Options:\n"
                f"1. Change antenna to 800 MHz or higher\n"
                f"2. Switch to **CRIM** model (works at any frequency)\n"
                f"3. Switch to **Peplinski** model (0.3-1.3 GHz, but has texture constraints)"
            )
        elif dielectric_model == "dobson" and freq_hz < 1.4e9:
            errors["model_frequency"] = (
                f"**Dobson model** requires frequency ≥ 1.4 GHz.\n"
                f"Your antenna is set to {freq_mhz:.0f} MHz.\n\n"
                f"Options:\n"
                f"1. Change antenna to 1.5 GHz or higher\n"
                f"2. Switch to **CRIM** model (works at any frequency)\n"
                f"3. Switch to **Mironov** model (0.6-18 GHz)"
            )
        elif dielectric_model == "peplinski" and (freq_hz < 0.3e9 or freq_hz > 1.3e9):
            errors["model_frequency"] = (
                f"**Peplinski model** requires frequency 0.3-1.3 GHz.\n"
                f"Your antenna is set to {freq_mhz:.0f} MHz.\n\n"
                f"Options:\n"
                f"1. Change antenna to 400 MHz - 1 GHz range\n"
                f"2. Switch to **CRIM** model (works at any frequency)"
            )
        else:
            errors["model_frequency"] = (
                f"Model '{dielectric_model}' is not compatible with {freq_mhz:.0f} MHz.\n"
                f"Compatible models at this frequency: {', '.join(compatible_models)}"
            )
    
    # Check layer-specific compatibility (texture constraints for Peplinski)
    for i, layer in enumerate(layers):
        texture_class = layer.get("texture_class")
        moisture_state = layer.get("moisture_state", "normal")
        
        if texture_class:
            is_compatible, error_msg = check_model_compatibility(
                dielectric_model, texture_class, moisture_state, freq_hz
            )
            
            if not is_compatible and "model_frequency" not in errors:
                # Only show texture errors if frequency is OK
                friendly_error = get_user_friendly_model_error(
                    dielectric_model, texture_class, moisture_state, freq_hz
                )
                errors[f"model_compatibility_layer_{i}"] = friendly_error
    
    return len(errors) == 0, errors


def route_after_validation(state: SimulationState) -> str:
    """
    Conditional routing after validation.
    
    Args:
        state: Current simulation state
    
    Returns:
        Next node name
    """
    collection_stage = state.get("collection_stage", "model")
    validation_errors = state.get("validation_errors", {})
    parameters_complete = state.get("parameters_complete", False)
    parameters_valid = state.get("parameters_valid", False)
    
    # If there are validation errors, return to parameter collection
    if validation_errors:
        logger.info("[VALIDATOR] Validation errors found, returning to parameter collection")
        return "parameter_collection"
    
    # If parameters are complete and valid, move to resolution
    if parameters_complete and parameters_valid:
        logger.info("[VALIDATOR] All parameters valid, moving to resolution")
        return "resolver"
    
    # If still collecting parameters, continue collection
    if collection_stage != "complete":
        logger.info(f"[VALIDATOR] Stage {collection_stage} validated, continuing collection")
        return "parameter_collection"
    
    # Default: wait for more input
    logger.info("[VALIDATOR] Waiting for user input")
    return "__end__"

