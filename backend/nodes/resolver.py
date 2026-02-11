"""
Resolver Node - Converts user-friendly parameters to full simulation schema.

This node:
1. Takes UserInputSimulation from state
2. Calls resolve_to_full() to convert to ExtractedParameters
3. Stores resolved parameters in state
4. Translates any resolution errors to user-friendly messages
"""

from typing import Dict, Any
from langchain_core.messages import AIMessage

from langgraph_state import SimulationState
from soil_setup.soil_structure_schema import (
    UserInputSimulation,
    UserLayerSimple,
    UserAntennaSimple,
    UserWaveformSimple,
    UserModelSimple,
    UserCylinderObject,
    UserBoxObject,
)
from soil_setup.soil_structure_form import resolve_to_full
from soil_setup.validation_translator import ValidationErrorTranslator
from init import logger


def resolver_node(state: SimulationState) -> Dict[str, Any]:
    """
    Resolve user-friendly parameters to full simulation schema.
    
    Args:
        state: Current simulation state
    
    Returns:
        Updated state with resolved_params
    """
    logger.info("[RESOLVER] Converting user parameters to full schema...")
    
    messages = state.get("messages", [])
    
    try:
        # Build UserInputSimulation from state
        layers_data = state.get("layers", [])
        antenna_data = state.get("antenna", {})
        waveform_data = state.get("waveform", {})
        model_data = state.get("model", {})
        objects_data = state.get("objects", [])
        
        # Convert layers to UserLayerSimple objects
        user_layers = []
        for layer_dict in layers_data:
            try:
                # Set defaults for optional fields
                layer_dict.setdefault("name", f"layer_{len(user_layers) + 1}")
                layer_dict.setdefault("moisture_state", "normal")
                layer_dict.setdefault("organic_level", "none")
                layer_dict.setdefault("salinity_environment", "fresh")
                layer_dict.setdefault("compaction_level", "normal")
                
                user_layer = UserLayerSimple(**layer_dict)
                user_layers.append(user_layer)
                logger.info(f"[RESOLVER] Converted layer: {user_layer.name}")
            except Exception as e:
                logger.error(f"[RESOLVER] Error converting layer {len(user_layers) + 1}: {e}")
                raise
        
        # Convert antenna to UserAntennaSimple
        antenna_data.setdefault("preset", "generic_400MHz")
        antenna_data.setdefault("axis", "x")
        user_antenna = UserAntennaSimple(**antenna_data)
        logger.info(f"[RESOLVER] Converted antenna: {user_antenna.preset}")
        
        # Convert waveform to UserWaveformSimple
        waveform_data.setdefault("kind", "ricker")
        waveform_data.setdefault("name", "default_waveform")
        user_waveform = UserWaveformSimple(**waveform_data)
        logger.info(f"[RESOLVER] Converted waveform: {user_waveform.kind}")
        
        # Convert model to UserModelSimple
        # Note: 'model' (soil dielectric model) is REQUIRED - no default
        if "model" not in model_data or model_data.get("model") is None:
            raise ValueError(
                "Soil dielectric model not specified. "
                "Please select one of: peplinski, mironov, dobson, or crim"
            )
        model_data.setdefault("quality", "balanced")
        model_data.setdefault("antenna_height_m", 0.02)
        model_data.setdefault("temperature_c", 20.0)
        model_data.setdefault("enforce_validity", True)
        user_model = UserModelSimple(**model_data)
        logger.info(f"[RESOLVER] Converted model: {user_model.title} (soil model: {user_model.model})")
        
        # Convert objects (if any)
        user_objects = []
        if objects_data:
            for obj_dict in objects_data:
                obj_type = obj_dict.pop("type", None)
                obj_dict.setdefault("dielectric_smoothing", True)
                
                try:
                    if obj_type == "cylinder":
                        obj_dict.setdefault("material", "pec")
                        obj_dict.setdefault("name", f"cylinder_{len(user_objects) + 1}")
                        user_obj = UserCylinderObject(**obj_dict)
                        user_objects.append(user_obj)
                        logger.info(f"[RESOLVER] Converted cylinder object: {user_obj.name}")
                    elif obj_type == "box":
                        obj_dict.setdefault("material", "pec")
                        obj_dict.setdefault("name", f"box_{len(user_objects) + 1}")
                        user_obj = UserBoxObject(**obj_dict)
                        user_objects.append(user_obj)
                        logger.info(f"[RESOLVER] Converted box object: {user_obj.name}")
                except Exception as e:
                    logger.error(f"[RESOLVER] Error converting object: {e}")
                    # Continue with other objects
        
        # Build UserInputSimulation
        user_simulation = UserInputSimulation(
            layers=user_layers,
            antenna=user_antenna,
            waveform=user_waveform,
            model=user_model,
            objects=user_objects if user_objects else None,
        )
        
        logger.info("[RESOLVER] Built UserInputSimulation successfully")
        
        # Resolve to full schema
        resolved_params = resolve_to_full(user_simulation)
        
        logger.info("[RESOLVER] Resolved to ExtractedParameters successfully")
        logger.info(f"[RESOLVER] Model: {resolved_params.model}")
        logger.info(f"[RESOLVER] Layers: {len(resolved_params.layers)}")
        logger.info(f"[RESOLVER] Antenna: {resolved_params.antenna}")
        logger.info(f"[RESOLVER] Waveform: {resolved_params.waveform}")
        if resolved_params.objects:
            logger.info(f"[RESOLVER] Objects: {len(resolved_params.objects)}")
        
        # ExtractedParameters now contains dicts, so we can use them directly
        # Just need to extract the nested fields properly
        resolved_dict = {
            "model": resolved_params.model,
            "title": resolved_params.title,
            "layers": resolved_params.layers,  # Already dicts
            "antenna": resolved_params.antenna,  # Already dict
            "waveform": resolved_params.waveform,  # Already dict
            "num_layers": resolved_params.num_layers or len(resolved_params.layers),
            "source_height_m": resolved_params.source_height_m,
            "domain_x": resolved_params.domain_x,
            "domain_y": resolved_params.domain_y,
            "cells_per_wavelength": resolved_params.cells_per_wavelength,
            "max_cell_m": resolved_params.max_cell_m,
            "temperature_c": resolved_params.temperature_c,
            "enforce_validity": resolved_params.enforce_validity,
        }
        
        if resolved_params.objects:
            resolved_dict["objects"] = resolved_params.objects  # Already dicts
        
        # Success message
        # Access dict values since resolved_params now contains dicts
        antenna_kind = resolved_params.antenna.get("kind", "unknown") if isinstance(resolved_params.antenna, dict) else resolved_params.antenna
        waveform_kind = resolved_params.waveform.get("kind", "unknown") if isinstance(resolved_params.waveform, dict) else resolved_params.waveform
        waveform_freq = resolved_params.waveform.get("center_freq_hz", 0) if isinstance(resolved_params.waveform, dict) else 0
        
        # Get survey_length and max_depth from model_data in state
        survey_length = model_data.get("survey_length_m", "N/A")
        max_depth = model_data.get("max_depth_m", "N/A")
        
        success_message = (
            "Parameters resolved successfully! All calculations complete.\n\n"
            "Summary:\n"
            f"- Model: {resolved_params.model}\n"
            f"- Title: {resolved_params.title}\n"
            f"- Layers: {len(resolved_params.layers)}\n"
            f"- Survey length: {survey_length} m\n"
            f"- Max depth: {max_depth} m\n"
            f"- Antenna: {antenna_kind}\n"
            f"- Waveform: {waveform_kind} @ {waveform_freq/1e6:.0f} MHz\n"
            "\nProceeding to generate the input file..."
        )
        
        return {
            "resolved_params": resolved_dict,
            "messages": messages + [AIMessage(content=success_message)],
        }
    
    except Exception as e:
        logger.error(f"[RESOLVER] Error resolving parameters: {e}", exc_info=True)
        
        # Use translator to provide user-friendly error message
        translator = ValidationErrorTranslator()
        
        # Try to extract meaningful context from the error
        error_str = str(e)
        
        # Build a user-friendly error message
        if "thickness" in error_str.lower():
            friendly_error = (
                "There's an issue with one of your layer thicknesses.\n\n"
                "Please make sure each layer has a positive thickness value "
                "(e.g., '0.5 meters' or '30 cm')."
            )
        elif "texture" in error_str.lower() or "sand" in error_str.lower() or "clay" in error_str.lower():
            friendly_error = (
                "There's an issue with the soil type specification.\n\n"
                "Please use standard soil types like: sand, sandy loam, loam, "
                "clay loam, clay, etc."
            )
        elif "moisture" in error_str.lower() or "theta" in error_str.lower():
            friendly_error = (
                "There's an issue with the moisture level.\n\n"
                "Please use: 'dry', 'normal', 'wet', or 'saturated'."
            )
        elif "antenna" in error_str.lower() or "frequency" in error_str.lower():
            friendly_error = (
                "There's an issue with the antenna configuration.\n\n"
                "Available frequencies: 200 MHz, 400 MHz, 800 MHz, 1 GHz."
            )
        else:
            friendly_error = (
                f"There was an issue processing your inputs:\n\n"
                f"{error_str}\n\n"
                "Please review your inputs and try again. If you're unsure, "
                "try using simpler terms like 'sandy loam' for soil type "
                "and 'wet' for moisture level."
            )
        
        return {
            "resolved_params": None,
            "validation_errors": {"resolution": error_str},
            "messages": messages + [AIMessage(content=friendly_error)],
        }

