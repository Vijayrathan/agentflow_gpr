"""
Parameter Collection Node - Incrementally collects simulation parameters.

This node:
1. Extracts parameters from user's message using LLM with structured output
2. Merges new parameters with existing state
3. Focuses on grouped parameters (model, layers, antenna, objects)
4. Provides contextual guidance based on use case
"""

from typing import Dict, Any, Optional
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError
import json
import re

from langgraph_state import SimulationState, merge_layer_params
from soil_setup.soil_structure_schema import (
    UserLayerSimple,
    UserAntennaSimple,
    UserWaveformSimple,
    UserModelSimple,
    UserCylinderObject,
    UserBoxObject,
)
from init import openai_api_key, openai_model, logger


def extract_json_from_response(response_text: str) -> dict:
    """
    Extract JSON from LLM response, handling markdown code blocks.
    
    Args:
        response_text: Raw LLM response text
    
    Returns:
        Parsed JSON dictionary
    """
    # Try to parse directly first
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass
    
    # Try to extract from markdown code block
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try to find any JSON object in the text
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    # If all else fails, return empty dict
    logger.warning(f"[PARAM_COLLECTION] Could not extract JSON from response: {response_text[:200]}")
    return {}


def parameter_collection_node(state: SimulationState) -> Dict[str, Any]:
    """
    Extract and merge parameters from user's message.
    
    Args:
        state: Current simulation state
    
    Returns:
        Updated state with new parameters merged
    """
    logger.info("[PARAM_COLLECTION] Starting parameter extraction...")
    
    messages = state.get("messages", [])
    collection_stage = state.get("collection_stage", "model")
    current_focus = state.get("current_focus", "model_basics")
    
    if not messages:
        return {}
    
    latest_message = messages[-1]
    if not isinstance(latest_message, HumanMessage):
        return {}
    
    user_input = latest_message.content
    logger.info(f"[PARAM_COLLECTION] Stage: {collection_stage}, Focus: {current_focus}")
    logger.info(f"[PARAM_COLLECTION] User input: {user_input[:150]}...")
    
    llm = ChatOpenAI(model=openai_model, api_key=openai_api_key, temperature=0)
    
    # Clear awaiting_user_input since we're now processing their input
    updates = {
        "awaiting_user_input": False
    }
    
    # Stage 1: Model parameters
    if collection_stage == "model":
        updates.update(_extract_model_params(llm, user_input, state))
    
    # Stage 2: Layer count and layer parameters
    elif collection_stage == "layers":
        updates.update(_extract_layer_params(llm, user_input, state))
    
    # Stage 3: Antenna parameters
    elif collection_stage == "antenna":
        updates.update(_extract_antenna_params(llm, user_input, state))
    
    # Stage 4: Waveform parameters (usually auto-set from antenna)
    elif collection_stage == "waveform":
        updates.update(_extract_waveform_params(llm, user_input, state))
    
    # Stage 5: Objects (if needed)
    elif collection_stage == "objects":
        updates.update(_extract_object_params(llm, user_input, state))
    
    # Stage 6: Complete - but user wants to modify something
    elif collection_stage == "complete":
        # Use LLM to intelligently detect what the user wants to modify
        modification_updates = _detect_and_apply_modification(llm, user_input, state)
        if modification_updates:
            updates.update(modification_updates)
        else:
            logger.info("[PARAM_COLLECTION] No modifications detected at complete stage")
    
    logger.info(f"[PARAM_COLLECTION] Extracted updates: {list(updates.keys())}")
    
    return updates


def _extract_model_params(llm: ChatOpenAI, user_input: str, state: SimulationState) -> Dict[str, Any]:
    """Extract model/simulation setup parameters."""
    
    existing_model = state.get("model", {})
    use_case_desc = state.get("use_case_description", "GPR simulation")
    current_focus = state.get("current_focus", "model_basics")
    
    # Check if we're specifically asking for soil dielectric model
    if current_focus == "soil_model_selection":
        return _extract_soil_model_selection(llm, user_input, state)
    
    prompt = f"""Extract GPR simulation model parameters from the user's input.

User is creating: {use_case_desc}

User Input: "{user_input}"

Extract ANY of the following parameters that are mentioned:
- title: Simulation title/name (string)
- quality: "fast", "balanced", or "high_accuracy" (default: "balanced")
- survey_length_m: Length of survey line in meters (float, > 0)
- max_depth_m: Maximum depth to simulate in meters (float, > 0)
- antenna_height_m: Height of antenna above ground in meters (float, >= 0, typical: 0.02 for ground-coupled)
- temperature_c: Temperature in Celsius (float, default: 20.0)

Current values: {json.dumps(existing_model, indent=2)}

If parameter not mentioned, return null for it.

Respond with ONLY valid JSON:
{{
  "title": "string or null",
  "quality": "string or null",
  "survey_length_m": number or null,
  "max_depth_m": number or null,
  "antenna_height_m": number or null,
  "temperature_c": number or null
}}
"""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        logger.info(f"[PARAM_COLLECTION] Model extraction LLM response: {response.content[:300]}")
        extracted = extract_json_from_response(response.content)
        
        # Merge with existing
        model_dict = {**existing_model}
        for key, value in extracted.items():
            if value is not None:
                model_dict[key] = value
        
        logger.info(f"[PARAM_COLLECTION] Model params: {model_dict}")
        
        # Check if basic model params are complete (not including soil model yet)
        required_model_fields = ["title", "survey_length_m", "max_depth_m"]
        basic_model_complete = all(model_dict.get(field) is not None for field in required_model_fields)
        
        updates = {"model": model_dict}
        
        # If basic model params complete, ask for soil dielectric model
        if basic_model_complete and model_dict.get("model") is None:
            # Ask about soil dielectric model
            soil_model_message = (
                "Great! Now I need to know which **soil dielectric model** to use for the simulation.\n\n"
                "Available models:\n\n"
                "1. **Peplinski** - Native gprMax support\n"
                "   - Best for: Standard GPR frequencies (0.3-1.3 GHz)\n"
                "   - ⚠️ Limitations: Strict texture constraints (only loam/silt_loam compatible)\n\n"
                "2. **Mironov** ⭐ Recommended for higher frequencies\n"
                "   - Best for: Wide frequency range (0.6-18 GHz)\n"
                "   - Features: Accounts for bound/free water in soil\n"
                "   - ✓ No texture constraints - works with all soil types\n\n"
                "3. **Dobson** - For higher frequencies\n"
                "   - Best for: Higher frequencies (1.4-18 GHz)\n"
                "   - ✓ No texture constraints\n\n"
                "4. **CRIM** - Simple, no constraints\n"
                "   - Best for: Exploratory simulations, any frequency\n"
                "   - ✓ No frequency or texture constraints\n\n"
                "Which model would you like to use? (e.g., 'mironov', 'peplinski', 'crim', etc.)"
            )
            updates["current_focus"] = "soil_model_selection"
            updates["awaiting_user_input"] = True
            updates["messages"] = state["messages"] + [AIMessage(content=soil_model_message)]
        elif basic_model_complete and model_dict.get("model") is not None:
            # All model params complete, move to layers
            next_message = (
                "Great! Now let's define the soil layers.\n\n"
                "How many layers do you want to simulate? "
                "(For example: 2 layers, 3 layers, etc.)"
            )
            updates["collection_stage"] = "layers"
            updates["current_focus"] = "layer_count"
            updates["awaiting_user_input"] = True
            updates["messages"] = state["messages"] + [AIMessage(content=next_message)]
        
        return updates
    
    except Exception as e:
        logger.error(f"[PARAM_COLLECTION] Error extracting model params: {e}", exc_info=True)
        return {}


def _extract_soil_model_selection(llm: ChatOpenAI, user_input: str, state: SimulationState) -> Dict[str, Any]:
    """Extract soil dielectric model selection from user input."""
    
    existing_model = state.get("model", {})
    
    prompt = f"""Extract the soil dielectric model selection from the user's input.

User Input: "{user_input}"

The user is selecting a soil dielectric model. Valid options are:
- "peplinski" - Native gprMax support, 0.3-1.3 GHz, strict texture constraints
- "mironov" - Wide frequency range 0.6-18 GHz, bound/free water model
- "dobson" - Higher frequencies 1.4-18 GHz
- "crim" - Simple mixing model, no constraints (works at any frequency)

Map user input to one of these exact values:
- "peplinski", "pep", "1" → "peplinski"
- "mironov", "mir", "2" → "mironov"
- "dobson", "dob", "3" → "dobson"
- "crim", "4", "simple" → "crim"

Respond with ONLY valid JSON:
{{
  "model": "peplinski" or "mironov" or "dobson" or "crim" or null
}}
"""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        logger.info(f"[PARAM_COLLECTION] Soil model extraction LLM response: {response.content[:300]}")
        extracted = extract_json_from_response(response.content)
        
        selected_model = extracted.get("model")
        
        if selected_model and selected_model in ["peplinski", "mironov", "dobson", "crim"]:
            # Valid model selected
            model_dict = {**existing_model, "model": selected_model}
            
            # Provide model-specific guidance
            if selected_model == "peplinski":
                model_note = (
                    f"Using **Peplinski** model (native gprMax support).\n\n"
                    f"⚠️ Note: This model has strict texture constraints:\n"
                    f"- Sand: 15-50%\n"
                    f"- Clay: 5-20%\n"
                    f"- Silt: 35-65%\n"
                    f"- Frequency: 0.3-1.3 GHz\n\n"
                    f"Only 'loam' and 'silt_loam' textures are fully compatible.\n\n"
                )
            elif selected_model == "mironov":
                model_note = (
                    f"Using **Mironov** model.\n\n"
                    f"✓ Wide frequency range (0.6-18 GHz)\n"
                    f"✓ No texture constraints\n"
                    f"✓ Accounts for bound/free water\n\n"
                )
            elif selected_model == "dobson":
                model_note = (
                    f"Using **Dobson** model.\n\n"
                    f"✓ Higher frequency range (1.4-18 GHz)\n"
                    f"✓ No texture constraints\n\n"
                )
            else:  # crim
                model_note = (
                    f"Using **CRIM** model (Complex Refractive Index Model).\n\n"
                    f"✓ No frequency constraints\n"
                    f"✓ No texture constraints\n"
                    f"✓ Simple volumetric mixing\n\n"
                )
            
            next_message = (
                model_note +
                "Now let's define the soil layers.\n\n"
                "How many layers do you want to simulate? "
                "(For example: 2 layers, 3 layers, etc.)"
            )
            
            return {
                "model": model_dict,
                "collection_stage": "layers",
                "current_focus": "layer_count",
                "awaiting_user_input": True,
                "messages": state["messages"] + [AIMessage(content=next_message)],
            }
        else:
            # Invalid or unclear selection
            clarify_message = (
                "I didn't catch which model you'd like to use. Please select one:\n\n"
                "- **peplinski** - Native gprMax, strict constraints\n"
                "- **mironov** - Recommended, wide frequency range\n"
                "- **dobson** - Higher frequencies\n"
                "- **crim** - Simple, no constraints\n\n"
                "Just type the model name (e.g., 'mironov')."
            )
            return {
                "awaiting_user_input": True,
                "messages": state["messages"] + [AIMessage(content=clarify_message)],
            }
    
    except Exception as e:
        logger.error(f"[PARAM_COLLECTION] Error extracting soil model: {e}", exc_info=True)
        return {}


def _extract_all_layers_at_once(llm: ChatOpenAI, user_input: str, state: SimulationState, num_layers: int) -> Dict[str, Any]:
    """Extract all layer specifications when user provides them all at once."""
    
    prompt = f"""Extract ALL soil layer parameters from the user's input. They are providing specifications for {num_layers} layer(s).

User Input: "{user_input}"

Extract parameters for EACH layer mentioned:
- thickness_m: Thickness in meters (float, > 0)
- texture_class: USDA texture class - MUST be one of: "sand", "loamy_sand", "sandy_loam", "loam", "silt_loam", "silt", "sandy_clay_loam", "clay_loam", "silty_clay_loam", "sandy_clay", "silty_clay", "clay"
- moisture_state: "dry", "normal", "wet", or "saturated"
- organic_level: "none", "low", "moderate", or "high_peaty" (default: "none")
- salinity_environment: "fresh", "slightly_saline", "brackish", or "seawater" (default: "fresh")
- compaction_level: "loose", "normal", or "compacted" (default: "normal")

Respond with ONLY valid JSON - an array of layer objects:
{{
  "layers": [
    {{
      "thickness_m": number,
      "texture_class": "string",
      "moisture_state": "string",
      "organic_level": "string",
      "salinity_environment": "string",
      "compaction_level": "string"
    }},
    ...
  ]
}}

Return exactly {num_layers} layer objects in order (Layer 1, Layer 2, etc.).
"""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        logger.info(f"[PARAM_COLLECTION] Multi-layer extraction LLM response: {response.content[:300]}")
        extracted = extract_json_from_response(response.content)
        
        layers_data = extracted.get("layers", [])
        if layers_data and len(layers_data) == num_layers:
            # Successfully extracted all layers
            logger.info(f"[PARAM_COLLECTION] Extracted {len(layers_data)} layers")
            
            # Add names if not provided
            for i, layer in enumerate(layers_data):
                if not layer.get("name"):
                    layer["name"] = f"layer_{i + 1}"
            
            # Move to antenna stage since all layers are complete
            antenna_message = (
                "All layers configured!\n\n"
                "Now let's set up the antenna. Which frequency would you like?\n"
                "- 200 MHz (deep penetration, lower resolution)\n"
                "- 400 MHz (balanced)\n"
                "- 800 MHz (higher resolution, shallower)\n"
                "- 1 GHz (very high resolution, very shallow)\n\n"
                "You can also specify the antenna axis (x, y, or z) if needed."
            )
            
            return {
                "layers": layers_data,
                "collection_stage": "antenna",
                "current_focus": "antenna_config",
                "awaiting_user_input": True,
                "messages": state["messages"] + [AIMessage(content=antenna_message)],
            }
        else:
            logger.warning(f"[PARAM_COLLECTION] Expected {num_layers} layers, got {len(layers_data)}")
            # Fall back to single-layer extraction
            return {}
    
    except Exception as e:
        logger.error(f"[PARAM_COLLECTION] Error extracting all layers: {e}", exc_info=True)
        return {}


def _extract_layer_modifications(llm: ChatOpenAI, user_input: str, state: SimulationState) -> Dict[str, Any]:
    """
    Extract layer modifications when user wants to update existing layers.
    Handles texture percentages (sand/silt/clay) and other layer properties.
    """
    existing_layers = state.get("layers", [])
    num_layers = state.get("num_layers", len(existing_layers))
    
    if not existing_layers:
        return {}
    
    user_input_lower = user_input.lower()
    
    # Check if user wants to update all layers or specific ones
    update_all_layers = any(phrase in user_input_lower for phrase in ["both", "all layers", "for both", "all of them"])
    
    prompt = f"""Extract layer modification parameters from the user's input.

User Input: "{user_input}"

Current layers ({num_layers} total): {json.dumps(existing_layers, indent=2)}

The user wants to modify layer parameters. Extract ANY of the following:

For texture percentages (sand/silt/clay):
- sand_pct_override: Sand percentage 0-100 (float)
- silt_pct_override: Silt percentage 0-100 (float)  
- clay_pct_override: Clay percentage 0-100 (float)

For other properties:
- thickness_m: Thickness in meters (float, > 0)
- texture_class: USDA texture class (string)
- moisture_state: "dry", "normal", "wet", or "saturated"
- organic_level: "none", "low", "moderate", or "high_peaty"
- salinity_environment: "fresh", "slightly_saline", "brackish", or "seawater"
- compaction_level: "loose", "normal", or "compacted"

IMPORTANT:
- If user says "both layers" or "all layers", set apply_to_all: true
- If user specifies layer numbers (e.g., "layer 1", "layer 2"), extract which layers
- Extract the actual numeric values for sand/silt/clay percentages

Respond with ONLY valid JSON:
{{
  "apply_to_all": boolean,
  "target_layers": [0, 1, ...] or null (0-indexed layer numbers, null if apply_to_all is true),
  "updates": {{
    "sand_pct_override": number or null,
    "silt_pct_override": number or null,
    "clay_pct_override": number or null,
    "thickness_m": number or null,
    "texture_class": "string or null",
    "moisture_state": "string or null",
    "organic_level": "string or null",
    "salinity_environment": "string or null",
    "compaction_level": "string or null"
  }}
}}
"""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        logger.info(f"[PARAM_COLLECTION] Layer modification extraction LLM response: {response.content[:300]}")
        extracted = extract_json_from_response(response.content)
        
        apply_to_all = extracted.get("apply_to_all", update_all_layers)
        target_layers = extracted.get("target_layers", [])
        updates_dict = extracted.get("updates", {})
        
        # Remove null values
        updates_dict = {k: v for k, v in updates_dict.items() if v is not None}
        
        if not updates_dict:
            logger.warning("[PARAM_COLLECTION] No valid updates extracted from modification request")
            return {}
        
        # Apply updates to layers
        updated_layers = [layer.copy() for layer in existing_layers]
        
        if apply_to_all:
            # Update all layers
            logger.info(f"[PARAM_COLLECTION] Applying updates to all {len(updated_layers)} layers: {updates_dict}")
            for layer in updated_layers:
                layer.update(updates_dict)
        elif target_layers:
            # Update specific layers
            logger.info(f"[PARAM_COLLECTION] Applying updates to layers {target_layers}: {updates_dict}")
            for layer_idx in target_layers:
                if 0 <= layer_idx < len(updated_layers):
                    updated_layers[layer_idx].update(updates_dict)
        else:
            # Default to first layer if unclear
            logger.info(f"[PARAM_COLLECTION] Applying updates to layer 0 (default): {updates_dict}")
            updated_layers[0].update(updates_dict)
        
        logger.info(f"[PARAM_COLLECTION] Updated layers: {updated_layers}")
        
        return {
            "layers": updated_layers,
        }
    
    except Exception as e:
        logger.error(f"[PARAM_COLLECTION] Error extracting layer modifications: {e}", exc_info=True)
        return {}


def _detect_and_apply_modification(llm: ChatOpenAI, user_input: str, state: SimulationState) -> Dict[str, Any]:
    """
    Use LLM to intelligently detect what parameter the user wants to modify.
    This handles modifications at the 'complete' stage for ANY parameter type.
    """
    existing_model = state.get("model", {})
    existing_layers = state.get("layers", [])
    existing_antenna = state.get("antenna", {})
    
    # Build context about current state
    current_state_summary = {
        "model": {
            "title": existing_model.get("title"),
            "soil_model": existing_model.get("model"),
            "survey_length_m": existing_model.get("survey_length_m"),
            "max_depth_m": existing_model.get("max_depth_m"),
        },
        "antenna": {
            "preset": existing_antenna.get("preset"),
            "axis": existing_antenna.get("axis"),
        },
        "num_layers": len(existing_layers),
        "layers_summary": [
            {
                "name": l.get("name"),
                "texture_class": l.get("texture_class"),
                "thickness_m": l.get("thickness_m"),
                "moisture_state": l.get("moisture_state"),
            }
            for l in existing_layers
        ],
    }
    
    prompt = f"""Analyze what the user wants to modify in their GPR simulation.

User Input: "{user_input}"

Current simulation state:
{json.dumps(current_state_summary, indent=2)}

Determine which category the user wants to modify:
1. "antenna" - if user mentions frequency, MHz, GHz, antenna, or wants to change frequency
2. "layers" - if user mentions layer, soil, texture, thickness, moisture, sand, silt, clay
3. "model" - if user mentions title, survey length, depth, domain, soil dielectric model (peplinski/mironov/dobson/crim)
4. "objects" - if user mentions object, cylinder, box, pipe, tank
5. "none" - if unclear what they want to modify

Also extract the specific value they want to change to, if mentioned.

Respond with ONLY valid JSON:
{{
  "category": "antenna" or "layers" or "model" or "objects" or "none",
  "specific_change": {{
    "field": "field name if identifiable",
    "value": "new value if mentioned"
  }}
}}
"""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        logger.info(f"[PARAM_COLLECTION] Modification detection LLM response: {response.content[:300]}")
        extracted = extract_json_from_response(response.content)
        
        category = extracted.get("category", "none")
        specific_change = extracted.get("specific_change", {})
        
        logger.info(f"[PARAM_COLLECTION] Detected modification category: {category}")
        
        updates = {}
        
        if category == "antenna":
            logger.info("[PARAM_COLLECTION] User wants to modify antenna parameters")
            # Handle antenna frequency changes
            antenna_updates = _extract_antenna_modification(llm, user_input, state, specific_change)
            updates.update(antenna_updates)
            if antenna_updates.get("antenna"):
                updates["parameters_complete"] = False
                updates["awaiting_user_input"] = False
                
        elif category == "layers":
            logger.info("[PARAM_COLLECTION] User wants to modify layer parameters")
            layer_updates = _extract_layer_modifications(llm, user_input, state)
            updates.update(layer_updates)
            if layer_updates.get("layers"):
                updates["parameters_complete"] = False
                
        elif category == "model":
            logger.info("[PARAM_COLLECTION] User wants to modify model parameters")
            model_updates = _extract_model_modification(llm, user_input, state, specific_change)
            updates.update(model_updates)
            if model_updates.get("model"):
                updates["parameters_complete"] = False
                
        elif category == "objects":
            logger.info("[PARAM_COLLECTION] User wants to modify object parameters")
            object_updates = _extract_object_params(llm, user_input, state)
            updates.update(object_updates)
            if object_updates.get("objects"):
                updates["parameters_complete"] = False
        else:
            logger.info("[PARAM_COLLECTION] Could not determine modification category")
            
        return updates
        
    except Exception as e:
        logger.error(f"[PARAM_COLLECTION] Error detecting modification: {e}", exc_info=True)
        return {}


def _extract_antenna_modification(llm: ChatOpenAI, user_input: str, state: SimulationState, specific_change: Dict) -> Dict[str, Any]:
    """
    Extract antenna modifications, including custom frequency values.
    """
    existing_antenna = state.get("antenna", {})
    
    prompt = f"""Extract antenna modification from the user's input.

User Input: "{user_input}"

Current antenna settings: {json.dumps(existing_antenna, indent=2)}

The user wants to change antenna settings. Extract the raw frequency value:

- frequency_mhz: The frequency in MHz (convert GHz to MHz: 1 GHz = 1000 MHz)
  Examples:
  - "200 MHz" → 200
  - "400MHz" → 400
  - "650 MHz" → 650
  - "750 MHz" → 750
  - "800 MHz" → 800
  - "1 GHz" → 1000
  - "1.5 GHz" → 1500
  - "650Ghz" → 650 (this is likely a typo for 650 MHz)
  
- axis: Antenna polarization axis - "x", "y", or "z"

Respond with ONLY valid JSON:
{{
  "frequency_mhz": number or null,
  "axis": "string or null"
}}
"""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        logger.info(f"[PARAM_COLLECTION] Antenna modification LLM response: {response.content[:300]}")
        extracted = extract_json_from_response(response.content)
        
        # Merge with existing
        antenna_dict = {**existing_antenna}
        
        # Map frequency to preset
        freq_mhz = extracted.get("frequency_mhz")
        if freq_mhz is not None:
            # Handle potential GHz typo (e.g., "650Ghz" should be 650 MHz)
            if freq_mhz > 10000:  # Likely meant MHz, not GHz
                freq_mhz = freq_mhz / 1000
            antenna_dict["preset"] = _map_frequency_to_preset(freq_mhz)
            logger.info(f"[PARAM_COLLECTION] Mapped {freq_mhz} MHz to preset: {antenna_dict['preset']}")
        
        # Set axis if provided
        if extracted.get("axis"):
            antenna_dict["axis"] = extracted["axis"]
        
        logger.info(f"[PARAM_COLLECTION] Updated antenna params: {antenna_dict}")
        
        # Generate confirmation message
        preset = antenna_dict.get("preset", "generic_400MHz")
        freq_map = {
            "generic_200MHz": "200 MHz",
            "generic_400MHz": "400 MHz",
            "generic_800MHz": "800 MHz",
            "generic_1GHz": "1 GHz",
            "generic_1.2GHz": "1.2 GHz",
            "generic_1.5GHz": "1.5 GHz",
        }
        freq_str = freq_map.get(preset, preset)
        
        confirm_message = (
            f"Updated antenna frequency to **{freq_str}**.\n\n"
            f"Re-validating parameters..."
        )
        
        return {
            "antenna": antenna_dict,
            "messages": state["messages"] + [AIMessage(content=confirm_message)],
        }
        
    except Exception as e:
        logger.error(f"[PARAM_COLLECTION] Error extracting antenna modification: {e}", exc_info=True)
        return {}


def _extract_model_modification(llm: ChatOpenAI, user_input: str, state: SimulationState, specific_change: Dict) -> Dict[str, Any]:
    """
    Extract model/simulation parameter modifications.
    """
    existing_model = state.get("model", {})
    
    prompt = f"""Extract model/simulation parameter modifications from the user's input.

User Input: "{user_input}"

Current model settings: {json.dumps(existing_model, indent=2)}

Extract ANY of the following parameters that the user wants to change:
- model: Soil dielectric model - "peplinski", "mironov", "dobson", or "crim"
- title: Simulation title/name
- survey_length_m: Survey length in meters
- max_depth_m: Maximum depth in meters
- quality: "fast", "balanced", or "high_accuracy"
- temperature_c: Temperature in Celsius

Respond with ONLY valid JSON (null for unchanged fields):
{{
  "model": "string or null",
  "title": "string or null",
  "survey_length_m": number or null,
  "max_depth_m": number or null,
  "quality": "string or null",
  "temperature_c": number or null
}}
"""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        logger.info(f"[PARAM_COLLECTION] Model modification LLM response: {response.content[:300]}")
        extracted = extract_json_from_response(response.content)
        
        # Merge with existing
        model_dict = {**existing_model}
        changes_made = []
        for key, value in extracted.items():
            if value is not None:
                model_dict[key] = value
                changes_made.append(f"{key}: {value}")
        
        if not changes_made:
            return {}
        
        logger.info(f"[PARAM_COLLECTION] Updated model params: {model_dict}")
        
        # Generate confirmation message
        confirm_message = (
            f"Updated model parameters:\n" +
            "\n".join(f"- {change}" for change in changes_made) +
            "\n\nRe-validating parameters..."
        )
        
        return {
            "model": model_dict,
            "messages": state["messages"] + [AIMessage(content=confirm_message)],
            "awaiting_user_input": False,
        }
        
    except Exception as e:
        logger.error(f"[PARAM_COLLECTION] Error extracting model modification: {e}", exc_info=True)
        return {}


def _extract_layer_params(llm: ChatOpenAI, user_input: str, state: SimulationState) -> Dict[str, Any]:
    """Extract layer count and individual layer parameters."""
    
    num_layers = state.get("num_layers")
    existing_layers = state.get("layers", [])
    current_focus = state.get("current_focus", "layer_count")
    
    # Check if user is providing complete layer specifications for all layers at once
    # (e.g., "Layer 1: ..., Layer 2: ...")
    user_input_lower = user_input.lower()
    if num_layers and ("layer 1" in user_input_lower or "l1" in user_input_lower):
        # User might be providing all layers at once - try to extract all
        logger.info("[PARAM_COLLECTION] Detecting multi-layer specification")
        return _extract_all_layers_at_once(llm, user_input, state, num_layers)
    
    # First, check if we need to get layer count
    if num_layers is None or current_focus == "layer_count":
        prompt = f"""Extract the number of soil layers from the user's input.

User Input: "{user_input}"

Extract:
- num_layers: Integer number of layers mentioned (e.g., "2 layers" → 2, "three layers" → 3)

Respond with ONLY valid JSON:
{{
  "num_layers": number or null
}}
"""
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            logger.info(f"[PARAM_COLLECTION] Layer count extraction LLM response: {response.content[:300]}")
            extracted = extract_json_from_response(response.content)
            
            if extracted.get("num_layers"):
                num_layers = int(extracted["num_layers"])
                logger.info(f"[PARAM_COLLECTION] Extracted num_layers: {num_layers}")
                
                # Initialize empty layers
                layers = [{}for _ in range(num_layers)]
                
                # Ask about first layer
                next_message = (
                    f"Perfect! We'll set up {num_layers} layer(s).\n\n"
                    f"Let's start with **Layer 1**:\n"
                    f"- Thickness (in meters, e.g., 0.5)\n"
                    f"- Soil texture (e.g., sand, loam, clay, silt)\n"
                    f"- Moisture state (dry, normal, wet, or saturated)\n\n"
                    f"You can provide these all at once or one at a time."
                )
                
                return {
                    "num_layers": num_layers,
                    "layers": layers,
                    "current_focus": "layer_0",
                    "awaiting_user_input": True,
                    "messages": state["messages"] + [AIMessage(content=next_message)],
                }
        
        except Exception as e:
            logger.error(f"[PARAM_COLLECTION] Error extracting layer count: {e}", exc_info=True)
        
        return {}
    
    # Extract individual layer parameters
    if current_focus and current_focus.startswith("layer_"):
        try:
            layer_index = int(current_focus.split("_")[1])
        except (IndexError, ValueError):
            layer_index = 0
        
        prompt = f"""Extract soil layer parameters from the user's input for Layer {layer_index + 1}.

User Input: "{user_input}"

Extract ANY of the following parameters:
- name: Layer name (string, optional)
- thickness_m: Thickness in meters (float, > 0)
- texture_class: USDA texture class - MUST be one of: "sand", "loamy_sand", "sandy_loam", "loam", "silt_loam", "silt", "sandy_clay_loam", "clay_loam", "silty_clay_loam", "sandy_clay", "silty_clay", "clay"
- moisture_state: "dry", "normal", "wet", or "saturated" (default: "normal")
- organic_level: "none", "low", "moderate", or "high_peaty" (default: "none")
- salinity_environment: "fresh", "slightly_saline", "brackish", or "seawater" (default: "fresh")
- compaction_level: "loose", "normal", or "compacted" (default: "normal")

Current layer {layer_index + 1} values: {json.dumps(existing_layers[layer_index] if layer_index < len(existing_layers) else {}, indent=2)}

Respond with ONLY valid JSON (null for unmentioned fields):
{{
  "name": "string or null",
  "thickness_m": number or null,
  "texture_class": "string or null",
  "moisture_state": "string or null",
  "organic_level": "string or null",
  "salinity_environment": "string or null",
  "compaction_level": "string or null"
}}
"""
        
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            logger.info(f"[PARAM_COLLECTION] Layer {layer_index + 1} extraction LLM response: {response.content[:300]}")
            extracted = extract_json_from_response(response.content)
            
            # Merge with existing layer
            current_layer = existing_layers[layer_index] if layer_index < len(existing_layers) else {}
            for key, value in extracted.items():
                if value is not None:
                    current_layer[key] = value
            
            # Update layers list
            updated_layers = merge_layer_params(existing_layers, current_layer, layer_index)
            
            logger.info(f"[PARAM_COLLECTION] Layer {layer_index + 1} params: {current_layer}")
            
            # Check if this layer is complete
            required_layer_fields = ["thickness_m", "texture_class"]
            layer_complete = all(current_layer.get(field) is not None for field in required_layer_fields)
            
            updates = {"layers": updated_layers}
            
            if layer_complete:
                # Move to next layer or finish layer collection
                if layer_index + 1 < num_layers:
                    next_layer_message = (
                        f"Layer {layer_index + 1} configured!\n\n"
                        f"Now for **Layer {layer_index + 2}**:\n"
                        f"- Thickness (meters)\n"
                        f"- Soil texture\n"
                        f"- Moisture state (optional, default: normal)\n"
                    )
                    updates["current_focus"] = f"layer_{layer_index + 1}"
                    updates["awaiting_user_input"] = True
                    updates["messages"] = state["messages"] + [AIMessage(content=next_layer_message)]
                else:
                    # All layers done, move to antenna
                    antenna_message = (
                        "All layers configured!\n\n"
                        "Now let's set up the antenna. Which frequency would you like?\n"
                        "- 200 MHz (deep penetration, lower resolution)\n"
                        "- 400 MHz (balanced)\n"
                        "- 800 MHz (higher resolution, shallower)\n"
                        "- 1 GHz (very high resolution, very shallow)\n\n"
                        "You can also specify the antenna axis (x, y, or z) if needed."
                    )
                    updates["collection_stage"] = "antenna"
                    updates["current_focus"] = "antenna_config"
                    updates["awaiting_user_input"] = True
                    updates["messages"] = state["messages"] + [AIMessage(content=antenna_message)]
            
            return updates
        
        except Exception as e:
            logger.error(f"[PARAM_COLLECTION] Error extracting layer params: {e}", exc_info=True)
    
    return {}


def _map_frequency_to_preset(freq_mhz: float) -> str:
    """Map a frequency in MHz to the closest antenna preset."""
    if freq_mhz <= 300:
        return "generic_200MHz"
    elif freq_mhz <= 550:
        return "generic_400MHz"
    elif freq_mhz <= 900:
        return "generic_800MHz"
    elif freq_mhz <= 1100:
        return "generic_1GHz"
    elif freq_mhz <= 1350:
        return "generic_1.2GHz"
    else:
        return "generic_1.5GHz"


def _extract_antenna_params(llm: ChatOpenAI, user_input: str, state: SimulationState) -> Dict[str, Any]:
    """Extract antenna parameters."""
    
    existing_antenna = state.get("antenna", {})
    
    prompt = f"""Extract antenna parameters from the user's input.

User Input: "{user_input}"

Extract the frequency value mentioned by the user (in MHz or GHz) and the axis if mentioned.

- frequency_mhz: The frequency in MHz (convert GHz to MHz: 1 GHz = 1000 MHz)
  Examples:
  - "200 MHz" → 200
  - "400MHz" → 400
  - "650 MHz" → 650
  - "800 MHz" → 800
  - "1 GHz" → 1000
  - "1.5 GHz" → 1500
  - "650Ghz" → 650000 (but this is likely a typo for 650 MHz, so use 650)
  
- axis: Antenna polarization axis - "x", "y", or "z"

Current values: {json.dumps(existing_antenna, indent=2)}

Respond with ONLY valid JSON:
{{
  "frequency_mhz": number or null,
  "axis": "string or null"
}}
"""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        logger.info(f"[PARAM_COLLECTION] Antenna extraction LLM response: {response.content[:300]}")
        extracted = extract_json_from_response(response.content)
        
        # Merge with existing
        antenna_dict = {**existing_antenna}
        
        # Map frequency to preset
        freq_mhz = extracted.get("frequency_mhz")
        if freq_mhz is not None:
            # Handle potential GHz typo (e.g., "650Ghz" should be 650 MHz)
            if freq_mhz > 10000:  # Likely meant MHz, not GHz
                freq_mhz = freq_mhz / 1000
            antenna_dict["preset"] = _map_frequency_to_preset(freq_mhz)
            logger.info(f"[PARAM_COLLECTION] Mapped {freq_mhz} MHz to preset: {antenna_dict['preset']}")
        
        # Set axis if provided
        if extracted.get("axis"):
            antenna_dict["axis"] = extracted["axis"]
        
        # Set defaults if needed
        if not antenna_dict.get("preset"):
            antenna_dict["preset"] = "generic_400MHz"
        if not antenna_dict.get("axis"):
            antenna_dict["axis"] = "x"
        
        logger.info(f"[PARAM_COLLECTION] Antenna params: {antenna_dict}")
        
        # Auto-fill waveform based on antenna
        waveform_dict = state.get("waveform", {})
        if not waveform_dict:
            waveform_dict = {
                "kind": "ricker",
                "name": "default_waveform"
            }
        
        # Check if we need objects
        use_case = state.get("use_case", "")
        needs_objects = any(keyword in use_case for keyword in ["buried", "pipe", "tank", "object"])
        
        updates = {
            "antenna": antenna_dict,
            "waveform": waveform_dict,
        }
        
        if needs_objects:
            objects_message = (
                "Antenna configured!\n\n"
                "I noticed your simulation involves buried objects. "
                "Would you like to add any objects (pipes, tanks, etc.)?\n\n"
                "For a cylinder/pipe, specify: position (x, y, z coordinates), radius, and length\n"
                "For a box/tank, specify: corner coordinates\n\n"
                "Or say 'skip' to proceed without objects."
            )
            updates["collection_stage"] = "objects"
            updates["current_focus"] = "objects_definition"
            updates["awaiting_user_input"] = True
            updates["messages"] = state["messages"] + [AIMessage(content=objects_message)]
        else:
            # Move directly to validation
            updates["collection_stage"] = "complete"
            updates["parameters_complete"] = True
            updates["awaiting_user_input"] = False
        
        return updates
    
    except Exception as e:
        logger.error(f"[PARAM_COLLECTION] Error extracting antenna params: {e}", exc_info=True)
        return {}


def _extract_waveform_params(llm: ChatOpenAI, user_input: str, state: SimulationState) -> Dict[str, Any]:
    """Extract waveform parameters (rarely needed, mostly auto-set)."""
    # Usually auto-set from antenna, minimal extraction needed
    return {}


def _extract_object_params(llm: ChatOpenAI, user_input: str, state: SimulationState) -> Dict[str, Any]:
    """Extract buried object parameters."""
    
    user_input_lower = user_input.lower()
    
    # Check if user wants to skip objects
    if any(keyword in user_input_lower for keyword in ["skip", "no objects", "none"]) and "not sure" not in user_input_lower:
        return {
            "collection_stage": "complete",
            "parameters_complete": True,
            "awaiting_user_input": False,
        }
    
    # Get simulation context for smart defaults
    survey_length = state.get("model", {}).get("survey_length_m", 5.0)
    max_depth = state.get("model", {}).get("max_depth_m", 2.0)
    
    # Extract object specifications
    prompt = f"""Extract buried object specifications from the user's input.

User Input: "{user_input}"

Simulation Context:
- Survey length: {survey_length} meters
- Max depth: {max_depth} meters

Extract cylindrical objects (pipes, rods):
- name: Object name (extract from input or default to "cylinder")
- radius: Cylinder radius in meters (REQUIRED - extract from input)
- length: Cylinder length in meters (extract from input if mentioned)
- x1, y1, z1: First face center coordinates (meters) - if not specified, use center of domain
- x2, y2, z2: Second face center coordinates (meters) - calculate from length if not specified
- material: "pec" (metal) or "free_space" (void) - default to "pec"

Extract box objects (tanks, containers):
- name: Object name (extract from input or default to "box")
- x1, y1, z1: Lower corner coordinates (meters) - if not specified, use center of domain
- x2, y2, z2: Upper corner coordinates (meters) - if not specified, estimate from context
- material: "pec" (metal) or "free_space" (void) - default to "pec"

IMPORTANT: If coordinates are not specified but radius/length ARE specified, provide reasonable defaults:
- Place cylinder horizontally in center of domain at mid-depth
- x1 = survey_length/2 - length/2, y1 = 0, z1 = max_depth/2
- x2 = survey_length/2 + length/2, y2 = 0, z2 = max_depth/2

Respond with ONLY valid JSON:
{{
  "cylinders": [
    {{"name": "string", "x1": number, "y1": number, "z1": number, "x2": number, "y2": number, "z2": number, "radius": number, "material": "pec"}}
  ],
  "boxes": [
    {{"name": "string", "x1": number, "y1": number, "z1": number, "x2": number, "y2": number, "z2": number, "material": "pec"}}
  ],
  "needs_clarification": boolean (true if critical info like radius/dimensions is missing)
}}

If no objects can be extracted, return empty arrays and set needs_clarification to true.
"""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        logger.info(f"[PARAM_COLLECTION] Object extraction LLM response: {response.content[:300]}")
        extracted = extract_json_from_response(response.content)
        
        objects_list = []
        needs_clarification = extracted.get("needs_clarification", False)
        
        # Add cylinders
        for cyl in extracted.get("cylinders", []):
            objects_list.append({
                "type": "cylinder",
                **cyl
            })
        
        # Add boxes
        for box in extracted.get("boxes", []):
            objects_list.append({
                "type": "box",
                **box
            })
        
        if objects_list:
            logger.info(f"[PARAM_COLLECTION] Extracted {len(objects_list)} object(s)")
            # Confirm with user about the defaults used
            obj_summary = []
            for obj in objects_list:
                if obj["type"] == "cylinder":
                    obj_summary.append(
                        f"- Cylinder: radius={obj['radius']}m, "
                        f"from ({obj['x1']:.2f}, {obj['y1']:.2f}, {obj['z1']:.2f}) "
                        f"to ({obj['x2']:.2f}, {obj['y2']:.2f}, {obj['z2']:.2f})"
                    )
                else:
                    obj_summary.append(
                        f"- Box: from ({obj['x1']:.2f}, {obj['y1']:.2f}, {obj['z1']:.2f}) "
                        f"to ({obj['x2']:.2f}, {obj['y2']:.2f}, {obj['z2']:.2f})"
                    )
            
            confirm_message = (
                "I've configured the following object(s):\n\n" +
                "\n".join(obj_summary) +
                "\n\nProceeding with parameter validation..."
            )
            
            return {
                "objects": objects_list,
                "collection_stage": "complete",
                "parameters_complete": True,
                "awaiting_user_input": False,
                "messages": state["messages"] + [AIMessage(content=confirm_message)],
            }
        elif needs_clarification:
            # LLM indicated it needs more info
            clarify_message = (
                "I need more information about the object. Please provide:\n"
                "- For cylinders: radius (required) and optionally length and position\n"
                "- For boxes: dimensions or corner coordinates\n\n"
                "Or say 'skip' to proceed without objects."
            )
            return {
                "messages": state["messages"] + [AIMessage(content=clarify_message)],
                "awaiting_user_input": True,
            }
        else:
            # No objects extracted, assume user wants to skip
            logger.info("[PARAM_COLLECTION] No objects extracted, proceeding without objects")
            return {
                "collection_stage": "complete",
                "parameters_complete": True,
                "awaiting_user_input": False,
            }
    
    except Exception as e:
        logger.error(f"[PARAM_COLLECTION] Error extracting objects: {e}", exc_info=True)
        return {}


def route_after_parameter_collection(state: SimulationState) -> str:
    """
    Conditional routing after parameter collection.
    
    Args:
        state: Current simulation state
    
    Returns:
        Next node name: "validator" or "__end__"
    """
    awaiting_user_input = state.get("awaiting_user_input", False)
    
    # If we just asked the user a question, wait for their response
    if awaiting_user_input:
        logger.info("[PARAM_COLLECTION] Waiting for user input")
        return "__end__"
    
    # Otherwise, proceed to validation
    logger.info("[PARAM_COLLECTION] Proceeding to validation")
    return "validator"

