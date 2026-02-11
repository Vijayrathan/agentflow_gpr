"""
Generator Node - Creates the gprMax .in file.

This node:
1. Takes resolved_params (ExtractedParameters) from state
2. Calls generate_gprmax_input_file() from physics_modelling.py
3. Stores file_path and file_content in state
"""

from typing import Dict, Any
from langchain_core.messages import AIMessage
from pathlib import Path
import time

from langgraph_state import SimulationState
from schema import GprSchema, WaveformSchema, AntennaSchema, LayerSchema, CylinderObjectSchema, BoxObjectSchema
from physics_modelling import generate_gprmax_input_file, CylinderObject, BoxObject
from sim_setup_agent import get_workspace_directory
from init import logger


def generator_node(state: SimulationState) -> Dict[str, Any]:
    """
    Generate gprMax input file from resolved parameters.
    
    Args:
        state: Current simulation state
    
    Returns:
        Updated state with file generation results
    """
    logger.info("[GENERATOR] Generating gprMax input file...")
    
    messages = state.get("messages", [])
    resolved_params = state.get("resolved_params")
    
    if not resolved_params:
        error_message = "Cannot generate file: Parameters not resolved. Please complete parameter collection first."
        return {
            "file_generated": False,
            "messages": messages + [AIMessage(content=error_message)],
        }
    
    try:
        # Prepare workspace directory
        workspace_dir = get_workspace_directory()
        generated_files_dir = workspace_dir / "generated_files"
        generated_files_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        timestamp = int(time.time() * 1000)
        session_id = str(hash(str(messages[-1].content if messages else "")))[:8]
        output_filename = str(generated_files_dir / f"generated_session_{timestamp}_{session_id}.in")
        
        logger.info(f"[GENERATOR] Output file: {output_filename}")
        
        # Extract parameters for generate_gprmax_input_file
        layers = resolved_params.get("layers", [])
        antenna = resolved_params.get("antenna", {})
        waveform = resolved_params.get("waveform", {})
        objects = resolved_params.get("objects", [])
        
        # Prepare layer data
        layer_thicknesses_m = [layer["thickness_m"] for layer in layers]
        layer_sand_pcts = [layer["sand_pct"] for layer in layers]
        layer_silt_pcts = [layer["silt_pct"] for layer in layers]
        layer_clay_pcts = [layer["clay_pct"] for layer in layers]
        layer_theta_vs = [layer["theta_v"] for layer in layers]
        layer_bulk_densities_gcm3 = [layer["bulk_density_gcm3"] for layer in layers]
        layer_particle_densities_gcm3 = [layer["particle_density_gcm3"] for layer in layers]
        layer_organic_fractions = [layer["organic_fraction"] for layer in layers]
        layer_salinity_classes = [layer["salinity_class"] for layer in layers]
        layer_porewater_sigmas_Sm = [layer["porewater_sigma_Sm"] for layer in layers]
        layer_names = [layer["name"] for layer in layers]
        
        logger.info(f"[GENERATOR] Layers: {len(layers)}")
        logger.info(f"[GENERATOR] Layer names: {layer_names}")
        
        # Prepare objects
        objects_list = None
        if objects:
            objects_list = []
            for obj in objects:
                if obj.get("type") == "cylinder":
                    cylinder = CylinderObject(
                        name=obj["name"],
                        x1=obj["x1"],
                        y1=obj["y1"],
                        z1=obj["z1"],
                        x2=obj["x2"],
                        y2=obj["y2"],
                        z2=obj["z2"],
                        radius=obj["radius"],
                        material=obj["material"],
                        dielectric_smoothing=obj.get("dielectric_smoothing", True),
                    )
                    objects_list.append(cylinder)
                    logger.info(f"[GENERATOR] Added cylinder: {obj['name']}")
                elif obj.get("type") == "box":
                    box = BoxObject(
                        name=obj["name"],
                        x1=obj["x1"],
                        y1=obj["y1"],
                        z1=obj["z1"],
                        x2=obj["x2"],
                        y2=obj["y2"],
                        z2=obj["z2"],
                        material=obj["material"],
                        dielectric_smoothing=obj.get("dielectric_smoothing", True),
                    )
                    objects_list.append(box)
                    logger.info(f"[GENERATOR] Added box: {obj['name']}")
        
        # Generate the input file
        file_content = generate_gprmax_input_file(
            layer_thicknesses_m=layer_thicknesses_m,
            layer_sand_pcts=layer_sand_pcts,
            layer_silt_pcts=layer_silt_pcts,
            layer_clay_pcts=layer_clay_pcts,
            layer_theta_vs=layer_theta_vs,
            layer_bulk_densities_gcm3=layer_bulk_densities_gcm3,
            layer_particle_densities_gcm3=layer_particle_densities_gcm3,
            layer_organic_fractions=layer_organic_fractions,
            layer_salinity_classes=layer_salinity_classes,
            layer_porewater_sigmas_Sm=layer_porewater_sigmas_Sm,
            layer_names=layer_names,
            waveform_kind=waveform["kind"],
            waveform_amplitude=waveform["amplitude"],
            waveform_center_freq_hz=waveform["center_freq_hz"],
            waveform_name=waveform["name"],
            antenna_kind=antenna["kind"],
            antenna_axis=antenna["axis"],
            antenna_tx_rx_offset_m=antenna["tx_rx_offset_m"],
            antenna_source_type=antenna.get("source_type", "hertzian_dipole"),
            antenna_resistance=antenna.get("resistance"),
            objects=objects_list,
            model_title=resolved_params.get("title", "GPR Simulation"),
            source_height_m=resolved_params.get("source_height_m", 0.02),
            domain_xy_m=(resolved_params.get("domain_x", 10.0), resolved_params.get("domain_y", 2.0)),
            cells_per_wavelength=int(resolved_params.get("cells_per_wavelength", 10)),
            max_cell_m=resolved_params.get("max_cell_m", 0.1),
            rx_same_height=True,
            temperature_c=resolved_params.get("temperature_c", 20.0),
            model=resolved_params.get("model", "gprMax"),
            enforce_validity=resolved_params.get("enforce_validity", True),
            output_filename=output_filename,
        )
        
        logger.info(f"[GENERATOR] File generated successfully: {output_filename}")
        logger.info(f"[GENERATOR] File size: {len(file_content)} characters")
        
        # Success message with file content preview
        success_message = (
            f"✅ **GPR Input File Generated Successfully!**\n\n"
            f"**File:** `{Path(output_filename).name}`\n"
            f"**Location:** `{output_filename}`\n\n"
            f"**Simulation Details:**\n"
            f"- Title: {resolved_params.get('title')}\n"
            f"- Model: {resolved_params.get('model')}\n"
            f"- Layers: {len(layers)}\n"
            f"- Domain: {resolved_params.get('domain_x')}m × {resolved_params.get('domain_y')}m\n"
            f"- Antenna: {antenna['kind']} @ {waveform['center_freq_hz']/1e6:.0f} MHz\n"
            f"- Temperature: {resolved_params.get('temperature_c')}°C\n"
            f"\nInput Parameters File:\n```\n{file_content}\n```"
        )
        
        return {
            "file_generated": True,
            "file_path": output_filename,
            "file_content": file_content,
            "messages": messages + [AIMessage(content=success_message)],
        }
    
    except Exception as e:
        logger.error(f"[GENERATOR] Error generating file: {e}", exc_info=True)
        
        error_message = (
            f"❌ **Error Generating File**\n\n"
            f"Error: {str(e)}\n\n"
            f"Please review the parameters and try again."
        )
        
        return {
            "file_generated": False,
            "file_path": None,
            "file_content": None,
            "validation_errors": {"generation": str(e)},
            "messages": messages + [AIMessage(content=error_message)],
        }

