import logging
import dotenv
import os
import openai
from typing import List, Optional
import json
from init import logger, openai_client, openai_model

from langchain.tools import tool

@tool
async def extraction_agent(initial_input: str, user_responses: Optional[List[str]] = None):
    """
    Run the workflow to extract parameters and generate gprMax input file.
    
    Args:
        initial_input: Initial user query
        user_responses: Optional list of subsequent user responses (for iterative input collection)
    
    Returns:
        dict with either:
        - "status": "complete", "output": generated file info
        - "status": "incomplete", "missing_params": formatted missing parameters message
    """
    logger.info("[TOOL CALL] extraction_agent - Extracting parameters from user input")
    logger.debug(f"[TOOL CALL] extraction_agent - Input: {initial_input[:200]}..." if len(initial_input) > 200 else f"[TOOL CALL] extraction_agent - Input: {initial_input}")
    system_prompt="""You are a parameter extraction assistant. Extract all parameters mentioned in the user's query about gprMax simulation setup.
        
        Extract the following information:
        - Number of layers and their properties (thickness_m, sand_pct, silt_pct, clay_pct, theta_v, bulk_density_gcm3, particle_density_gcm3, organic_fraction, salinity_class, porewater_sigma_Sm, name)
        - Waveform properties (kind, amplitude, center_freq_hz, name)
        - Antenna properties (kind, axis, tx_rx_offset_m)
        - Model properties (model, title, source_height_m, domain_x, domain_y, cells_per_wavelength, max_cell_m, temperature_c, enforce_validity)
        
        Return ONLY the parameters that are explicitly mentioned. Do not make up any parameters. Strictly leave the fields as None if not mentioned."""
    
    response = openai_client.responses.create(
    model=openai_model,
    input=f"{system_prompt} \n\n User query: {initial_input}",
        )
    
    # Handle different types of response.output
    if not response.output:
        result = None
    elif isinstance(response.output, list):
        # If output is a list, take the first element if available
        if len(response.output) > 0:
            first_item = response.output[0]
            # Check if it's a Pydantic model
            if hasattr(first_item, 'model_dump'):
                result = first_item.model_dump()
            elif isinstance(first_item, dict):
                result = first_item
            else:
                # Convert to dict if possible
                result = dict(first_item) if hasattr(first_item, '__dict__') else str(first_item)
        else:
            result = None
    elif hasattr(response.output, 'model_dump'):
        # It's a Pydantic model
        result = response.output.model_dump()
    elif isinstance(response.output, dict):
        # Already a dict
        result = response.output
    else:
        # Fallback: try to convert to dict or string
        result = dict(response.output) if hasattr(response.output, '__dict__') else str(response.output)
    logger.info("[TOOL RESULT] extraction_agent - Parameter extraction completed")
    logger.debug(f"[TOOL RESULT] extraction_agent - Extracted parameters: {json.dumps(result, indent=2, default=str)}")
    return result

