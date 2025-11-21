from pydantic_ai import Agent
from pydantic import BaseModel
from physics_modelling import generate_gprmax_input_file
from typing import List, Optional, Dict, Any
import os
import dotenv
import asyncio
import openai
import json
import re
import logging
import huggingface_hub
import subprocess
import shutil
from pathlib import Path
from schema import GprSchema, WaveformSchema, AntennaSchema, LayerSchema, ExtractedParameters

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

dotenv.load_dotenv()

# HF_TOKEN = os.getenv("HF_TOKEN")
# huggingface_hub.login(token=HF_TOKEN)
# qwen_model = "Qwen/Qwen3-8B"
openai_api_key = os.getenv("OPENAI_API_KEY")

openai_model = "gpt-4.1"

openai_client = openai.OpenAI(api_key=openai_api_key)

def run_gprmax_simulation_tool(input_file: str) -> str:
    """
    Run gprMax simulation with the given input file.
    
    First checks if gprMax is installed. If not, clones the repository,
    installs it using conda, and then runs the simulation.
    
    Args:
        input_file: Path to the gprMax input file (.in file)
    
    Returns:
        str: Success message or error message
    """
    logger.info(f"[TOOL CALL] run_gprmax_simulation_tool - Running simulation with input file: {input_file}")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        error_msg = f"Input file not found: {input_file}"
        logger.error(f"[TOOL RESULT] run_gprmax_simulation_tool - {error_msg}")
        return error_msg
    
    # Check if gprMax is installed by trying to run it as a module
    gprmax_installed = False
    try:
        # Try to execute the module (it may show usage/help or error, but module exists)
        result = subprocess.run(
            ["python", "-m", "gprMax"],
            capture_output=True,
            text=True,
            timeout=10
        )
        # Check stderr for "No module named" error
        if result.stderr and ("No module named" in result.stderr or "ModuleNotFoundError" in result.stderr):
            logger.debug("gprMax module not found")
            gprmax_installed = False
        else:
            # Module exists (may have other errors, but module is installed)
            gprmax_installed = True
            logger.debug("gprMax module found")
    except subprocess.TimeoutExpired:
        logger.debug("gprMax check timed out")
        gprmax_installed = False
    except FileNotFoundError:
        logger.debug("python command not found")
        gprmax_installed = False
    except Exception as e:
        logger.debug(f"gprMax check error: {e}")
        gprmax_installed = False
    
    if not gprmax_installed:
        logger.info("gprMax not found. Installing gprMax...")
        
        # Check if conda is available
        conda_available = shutil.which("conda") is not None
        if not conda_available:
            error_msg = "conda is not available. Please install conda to use gprMax."
            logger.error(f"[TOOL RESULT] run_gprmax_simulation_tool - {error_msg}")
            return error_msg
        
        # Clone repository if it doesn't exist
        gprmax_dir = Path("gprMax")
        if not gprmax_dir.exists():
            logger.info("Cloning gprMax repository...")
            try:
                clone_result = subprocess.run(
                    ["git", "clone", "https://github.com/gprMax/gprMax.git"],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minutes timeout for clone
                )
                if clone_result.returncode != 0:
                    error_msg = f"Failed to clone gprMax repository: {clone_result.stderr}"
                    logger.error(f"[TOOL RESULT] run_gprmax_simulation_tool - {error_msg}")
                    return error_msg
                logger.info("Successfully cloned gprMax repository")
            except subprocess.TimeoutExpired:
                error_msg = "Timeout while cloning gprMax repository"
                logger.error(f"[TOOL RESULT] run_gprmax_simulation_tool - {error_msg}")
                return error_msg
            except Exception as e:
                error_msg = f"Error cloning gprMax repository: {str(e)}"
                logger.error(f"[TOOL RESULT] run_gprmax_simulation_tool - {error_msg}")
                return error_msg
        
        # Change to gprMax directory
        original_dir = os.getcwd()
        try:
            os.chdir(gprmax_dir)
            
            # Check if conda environment already exists
            env_exists_result = subprocess.run(
                ["conda", "env", "list"],
                capture_output=True,
                text=True
            )
            env_exists = "gprMax" in env_exists_result.stdout
            
            if not env_exists:
                logger.info("Creating conda environment from conda_env.yml...")
                # Create conda environment
                env_result = subprocess.run(
                    ["conda", "env", "create", "-f", "conda_env.yml"],
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minutes timeout for environment creation
                )
                if env_result.returncode != 0:
                    error_msg = f"Failed to create conda environment: {env_result.stderr}"
                    logger.error(f"[TOOL RESULT] run_gprmax_simulation_tool - {error_msg}")
                    os.chdir(original_dir)
                    return error_msg
                logger.info("Successfully created conda environment")
            
            # Build and install gprMax
            logger.info("Building and installing gprMax...")
            
            # Use conda run to execute commands in the gprMax environment
            build_result = subprocess.run(
                ["conda", "run", "-n", "gprMax", "python", "setup.py", "build"],
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout for build
            )
            if build_result.returncode != 0:
                error_msg = f"Failed to build gprMax: {build_result.stderr}"
                logger.error(f"[TOOL RESULT] run_gprmax_simulation_tool - {error_msg}")
                os.chdir(original_dir)
                return error_msg
            
            install_result = subprocess.run(
                ["conda", "run", "-n", "gprMax", "python", "setup.py", "install"],
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout for install
            )
            if install_result.returncode != 0:
                error_msg = f"Failed to install gprMax: {install_result.stderr}"
                logger.error(f"[TOOL RESULT] run_gprmax_simulation_tool - {error_msg}")
                os.chdir(original_dir)
                return error_msg
            
            logger.info("Successfully installed gprMax")
            
            # Return to original directory
            os.chdir(original_dir)
            
        except subprocess.TimeoutExpired as e:
            error_msg = f"Timeout during gprMax installation: {str(e)}"
            logger.error(f"[TOOL RESULT] run_gprmax_simulation_tool - {error_msg}")
            os.chdir(original_dir)
            return error_msg
        except Exception as e:
            error_msg = f"Error during gprMax installation: {str(e)}"
            logger.error(f"[TOOL RESULT] run_gprmax_simulation_tool - {error_msg}")
            os.chdir(original_dir)
            return error_msg
    
    # Run the simulation
    logger.info(f"Running gprMax simulation with input file: {input_file}")
    try:
        # Use absolute path for input file
        abs_input_file = os.path.abspath(input_file)
        
        # Check if we need to use conda run (if we just installed it)
        if not gprmax_installed:
            # Run using conda environment
            sim_result = subprocess.run(
                ["conda", "run", "-n", "gprMax", "python", "-m", "gprMax", abs_input_file],
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout for simulation
            )
        else:
            # Run directly
            sim_result = subprocess.run(
                ["python", "-m", "gprMax", abs_input_file],
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout for simulation
            )
        
        if sim_result.returncode != 0:
            error_msg = f"gprMax simulation failed:\nSTDOUT: {sim_result.stdout}\nSTDERR: {sim_result.stderr}"
            logger.error(f"[TOOL RESULT] run_gprmax_simulation_tool - {error_msg}")
            return error_msg
        
        success_msg = f"Successfully ran gprMax simulation with input file: {input_file}\nSTDOUT: {sim_result.stdout}"
        logger.info(f"[TOOL RESULT] run_gprmax_simulation_tool - {success_msg}")
        return success_msg
        
    except subprocess.TimeoutExpired:
        error_msg = "Simulation timed out after 1 hour"
        logger.error(f"[TOOL RESULT] run_gprmax_simulation_tool - {error_msg}")
        return error_msg
    except Exception as e:
        error_msg = f"Error running gprMax simulation: {str(e)}"
        logger.error(f"[TOOL RESULT] run_gprmax_simulation_tool - {error_msg}")
        return error_msg

def generate_gprmax_input_file_tool(gpr_data: GprSchema) -> str:
    """Generate gprMax input file from complete GprSchema"""
    logger.info(f"[TOOL CALL] generate_gprmax_input_file_tool - Generating gprMax input file for: {gpr_data.title}")
    # Convert GprSchema to the format expected by generate_gprmax_input_file
    
    layer_thicknesses_m = [layer.thickness_m for layer in gpr_data.layers]
    layer_sand_pcts = [layer.sand_pct for layer in gpr_data.layers]
    layer_silt_pcts = [layer.silt_pct for layer in gpr_data.layers]
    layer_clay_pcts = [layer.clay_pct for layer in gpr_data.layers]
    layer_theta_vs = [layer.theta_v for layer in gpr_data.layers]
    layer_bulk_densities_gcm3 = [layer.bulk_density_gcm3 for layer in gpr_data.layers]
    layer_particle_densities_gcm3 = [layer.particle_density_gcm3 for layer in gpr_data.layers]
    layer_organic_fractions = [layer.organic_fraction if layer.organic_fraction is not None else 0.0 for layer in gpr_data.layers]
    layer_salinity_classes = [layer.salinity_class for layer in gpr_data.layers]
    layer_porewater_sigmas_Sm = [layer.porewater_sigma_Sm for layer in gpr_data.layers]
    layer_names = [layer.name for layer in gpr_data.layers]
    
    text = generate_gprmax_input_file(
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
        waveform_kind=gpr_data.waveform.kind,
        waveform_amplitude=gpr_data.waveform.amplitude,
        waveform_center_freq_hz=gpr_data.waveform.center_freq_hz,
        waveform_name=gpr_data.waveform.name,
        antenna_kind=gpr_data.antenna.kind,
        antenna_axis=gpr_data.antenna.axis,
        antenna_tx_rx_offset_m=gpr_data.antenna.tx_rx_offset_m,
        model_title=gpr_data.title,
        source_height_m=gpr_data.source_height_m,
        domain_xy_m=(gpr_data.domain_x, gpr_data.domain_y),
        cells_per_wavelength=int(gpr_data.cells_per_wavelength),
        max_cell_m=gpr_data.max_cell_m,
        rx_same_height=True,
        temperature_c=gpr_data.temperature_c,
        model=gpr_data.model,
        enforce_validity=gpr_data.enforce_validity,
    )
    result_msg = f"Successfully generated gprMax input file for: {gpr_data.title} \n\n {text}"
    logger.info(f"[TOOL RESULT] generate_gprmax_input_file_tool - {result_msg}")
    return result_msg



def check_input_completeness(extracted: ExtractedParameters) -> tuple[bool, str]:
    """Check if all required parameters are provided. Returns (is_complete, missing_params_message)"""
    logger.info("[TOOL CALL] check_input_completeness - Checking if all required parameters are provided")
    missing = []
    
    # Check global parameters
    if extracted.model is None:
        missing.append("- model (dielectric model: 'crim', 'peplinski', 'dobson', or 'mironov')")
    if extracted.title is None:
        missing.append("- title (simulation title)")
    if extracted.source_height_m is None:
        missing.append("- source_height_m (source height in meters)")
    if extracted.domain_x is None:
        missing.append("- domain_x (domain size in x direction, meters)")
    if extracted.domain_y is None:
        missing.append("- domain_y (domain size in y direction, meters)")
    if extracted.cells_per_wavelength is None:
        missing.append("- cells_per_wavelength (grid resolution)")
    if extracted.max_cell_m is None:
        missing.append("- max_cell_m (maximum cell size in meters)")
    if extracted.temperature_c is None:
        missing.append("- temperature_c (temperature in Celsius)")
    if extracted.enforce_validity is None:
        missing.append("- enforce_validity (boolean)")
    
    # Check waveform
    if extracted.waveform is None:
        missing.append("- waveform (kind, amplitude, center_freq_hz, name)")
    else:
        # Handle both dict and object access
        if isinstance(extracted.waveform, dict):
            if "kind" not in extracted.waveform or extracted.waveform["kind"] is None:
                missing.append("- waveform.kind ('ricker' or 'gaussian')")
            if "amplitude" not in extracted.waveform or extracted.waveform["amplitude"] is None:
                missing.append("- waveform.amplitude")
            if "center_freq_hz" not in extracted.waveform or extracted.waveform["center_freq_hz"] is None:
                missing.append("- waveform.center_freq_hz (center frequency in Hz)")
            if "name" not in extracted.waveform or extracted.waveform["name"] is None:
                missing.append("- waveform.name")
        else:
            if not hasattr(extracted.waveform, "kind") or getattr(extracted.waveform, "kind", None) is None:
                missing.append("- waveform.kind ('ricker' or 'gaussian')")
            if not hasattr(extracted.waveform, "amplitude") or getattr(extracted.waveform, "amplitude", None) is None:
                missing.append("- waveform.amplitude")
            if not hasattr(extracted.waveform, "center_freq_hz") or getattr(extracted.waveform, "center_freq_hz", None) is None:
                missing.append("- waveform.center_freq_hz (center frequency in Hz)")
            if not hasattr(extracted.waveform, "name") or getattr(extracted.waveform, "name", None) is None:
                missing.append("- waveform.name")
    
    # Check antenna
    if extracted.antenna is None:
        missing.append("- antenna (kind, axis, tx_rx_offset_m)")
    else:
        # Handle both dict and object access
        if isinstance(extracted.antenna, dict):
            if "kind" not in extracted.antenna or extracted.antenna["kind"] is None:
                missing.append("- antenna.kind ('hertzian_dipole')")
            if "axis" not in extracted.antenna or extracted.antenna["axis"] is None:
                missing.append("- antenna.axis ('x', 'y', or 'z')")
            if "tx_rx_offset_m" not in extracted.antenna or extracted.antenna["tx_rx_offset_m"] is None:
                missing.append("- antenna.tx_rx_offset_m (transmitter-receiver offset in meters)")
        else:
            if not hasattr(extracted.antenna, "kind") or getattr(extracted.antenna, "kind", None) is None:
                missing.append("- antenna.kind ('hertzian_dipole')")
            if not hasattr(extracted.antenna, "axis") or getattr(extracted.antenna, "axis", None) is None:
                missing.append("- antenna.axis ('x', 'y', or 'z')")
            if not hasattr(extracted.antenna, "tx_rx_offset_m") or getattr(extracted.antenna, "tx_rx_offset_m", None) is None:
                missing.append("- antenna.tx_rx_offset_m (transmitter-receiver offset in meters)")
    
    # Check layers
    if extracted.num_layers is None or extracted.num_layers <= 0:
        missing.append("- num_layers (number of layers, must be > 0)")
    elif extracted.layers is None or len(extracted.layers) != extracted.num_layers:
        missing.append(f"- layers (need {extracted.num_layers} layer(s) with complete data)")
    else:
        for i, layer in enumerate(extracted.layers, 1):
            layer_missing = []
            # Handle both dict and object access
            if isinstance(layer, dict):
                if "thickness_m" not in layer or layer["thickness_m"] is None:
                    layer_missing.append("thickness_m")
                if "sand_pct" not in layer or layer["sand_pct"] is None:
                    layer_missing.append("sand_pct")
                if "silt_pct" not in layer or layer["silt_pct"] is None:
                    layer_missing.append("silt_pct")
                if "clay_pct" not in layer or layer["clay_pct"] is None:
                    layer_missing.append("clay_pct")
                if "theta_v" not in layer or layer["theta_v"] is None:
                    layer_missing.append("theta_v (volumetric water content)")
            else:
                if not hasattr(layer, "thickness_m") or getattr(layer, "thickness_m", None) is None:
                    layer_missing.append("thickness_m")
                if not hasattr(layer, "sand_pct") or getattr(layer, "sand_pct", None) is None:
                    layer_missing.append("sand_pct")
                if not hasattr(layer, "silt_pct") or getattr(layer, "silt_pct", None) is None:
                    layer_missing.append("silt_pct")
                if not hasattr(layer, "clay_pct") or getattr(layer, "clay_pct", None) is None:
                    layer_missing.append("clay_pct")
                if not hasattr(layer, "theta_v") or getattr(layer, "theta_v", None) is None:
                    layer_missing.append("theta_v (volumetric water content)")
            
            if layer_missing:
                missing.append(f"- Layer {i}: {', '.join(layer_missing)}")
    
    if missing:
        missing_msg = "\n".join(missing)
        logger.info(f"[TOOL RESULT] check_input_completeness - Parameters incomplete. Missing: {len(missing)} parameter(s)")
        logger.debug(f"[TOOL RESULT] check_input_completeness - Missing details: {missing_msg}")
        return False, missing_msg
    logger.info("[TOOL RESULT] check_input_completeness - All required parameters are present")
    return True, ""


def format_missing_params_message(missing_params: str) -> str:
    """Format missing parameters into a user-friendly message"""
    return f"""The following parameters are missing or incomplete:

{missing_params}

Ask the user to provide all the missing information to proceed with generating the gprMax input file."""


def format_validation_errors_message(validation_errors: list[str]) -> str:
    """Format validation errors into a user-friendly message"""
    return f"""Parameter validation failed. Please correct the following errors:

{validation_errors}

Ask the user to provide corrected values for the parameters mentioned above."""


def validate_gpr_parameters(gpr_data: GprSchema) -> tuple[bool, str]:
    """
    Validate all parameters according to physics_modelling.py validity rules.
    
    Returns:
        (is_valid, error_message) - if is_valid is False, error_message contains validation errors
    """
    logger.info(f"[TOOL CALL] validate_gpr_parameters - Validating parameters for: {gpr_data.title}")
    errors = []
    
    # 1. Check model is valid
    valid_models = {'crim', 'peplinski', 'dobson', 'mironov'}
    if gpr_data.model.lower() not in valid_models:
        errors.append(f"Invalid model '{gpr_data.model}'. Must be one of: {', '.join(valid_models)}")
    
    # 2. Check at least one layer exists
    if not gpr_data.layers or len(gpr_data.layers) == 0:
        errors.append("At least one layer is required")
        return False, "\n".join(errors)
    
    # 3. Validate each layer (LayerSpec.validate rules)
    for i, layer in enumerate(gpr_data.layers, 1):
        layer_errors = []
        
        # thickness_m must be > 0
        if layer.thickness_m <= 0:
            layer_errors.append("thickness_m must be > 0")
        
        # sand + silt + clay must sum to 100
        p_sum = layer.sand_pct + layer.silt_pct + layer.clay_pct
        if abs(p_sum - 100.0) > 1e-6:
            layer_errors.append(f"sand_pct + silt_pct + clay_pct must sum to 100 (got {p_sum:.2f})")
        
        # theta_v must be 0..1
        if not (0.0 <= layer.theta_v <= 1.0):
            layer_errors.append(f"theta_v must be between 0.0 and 1.0 (got {layer.theta_v})")
        
        # bulk_density_gcm3 must be > 0 if provided
        if layer.bulk_density_gcm3 is not None and layer.bulk_density_gcm3 <= 0:
            layer_errors.append("bulk_density_gcm3 must be > 0 if provided")
        
        # particle_density_gcm3 must be > 0 if provided
        if layer.particle_density_gcm3 is not None and layer.particle_density_gcm3 <= 0:
            layer_errors.append("particle_density_gcm3 must be > 0 if provided")
        
        if layer_errors:
            errors.append(f"Layer {i} errors: {'; '.join(layer_errors)}")
    
    # 4. Validate waveform (WaveformSpec rules)
    waveform_kind_lower = gpr_data.waveform.kind.lower()
    if waveform_kind_lower not in {'ricker', 'gaussian'}:
        errors.append(f"Invalid waveform.kind '{gpr_data.waveform.kind}'. Must be 'ricker' or 'gaussian'")
    
    # 5. Validate antenna (AntennaSpec.validate rules)
    antenna_kind_lower = gpr_data.antenna.kind.lower()
    if antenna_kind_lower not in {'hertzian_dipole'}:
        errors.append(f"Invalid antenna.kind '{gpr_data.antenna.kind}'. Only 'hertzian_dipole' is supported")
    
    antenna_axis_lower = gpr_data.antenna.axis.lower()
    if antenna_axis_lower not in {'x', 'y', 'z'}:
        errors.append(f"Invalid antenna.axis '{gpr_data.antenna.axis}'. Must be 'x', 'y', or 'z'")
    
    # 6. Model-specific validity checks (from check_validity in ModelSpec.build)
    if gpr_data.enforce_validity:
        f0 = gpr_data.waveform.center_freq_hz
        model_lower = gpr_data.model.lower()
        
        for i, layer in enumerate(gpr_data.layers, 1):
            theta = layer.theta_v
            sand = layer.sand_pct
            silt = layer.silt_pct
            clay = layer.clay_pct
            
            if model_lower == "peplinski":
                if not (0.3e9 <= f0 <= 1.3e9):
                    errors.append(f"Layer {i}: Peplinski model requires frequency between 0.3-1.3 GHz (got {f0/1e9:.2f} GHz)")
                if not (0.0 <= theta <= 0.30):
                    errors.append(f"Layer {i}: Peplinski model requires moisture content between 0-0.30 (got {theta:.3f})")
                if not (15 <= sand <= 50 and 5 <= clay <= 20 and 35 <= silt <= 65):
                    errors.append(f"Layer {i}: Peplinski model requires sand 15-50%, clay 5-20%, silt 35-65% (got sand={sand:.1f}%, clay={clay:.1f}%, silt={silt:.1f}%)")
            
            elif model_lower == "dobson":
                if not (1.4e9 <= f0 <= 18e9):
                    errors.append(f"Layer {i}: Dobson model requires frequency between 1.4-18 GHz (got {f0/1e9:.2f} GHz)")
                if not (0.0 <= theta <= 0.50):
                    errors.append(f"Layer {i}: Dobson model requires moisture content between 0-0.50 (got {theta:.3f})")
            
            elif model_lower == "mironov":
                if not (0.6e9 <= f0 <= 18e9):
                    errors.append(f"Layer {i}: Mironov model requires frequency between 0.6-18 GHz (got {f0/1e9:.2f} GHz)")
                if not (0.0 <= theta <= 0.45):
                    errors.append(f"Layer {i}: Mironov model requires moisture content between 0-0.45 (got {theta:.3f})")
            
            # CRIM has no restrictions
    
    # 7. Check source height constraint (from ModelSpec.build)
    # This is approximate - we'd need to compute z_extent exactly, but we can check basic constraints
    total_layers_thick = sum(layer.thickness_m for layer in gpr_data.layers)
    air_top = max(gpr_data.source_height_m + 6 * gpr_data.max_cell_m, 0.05)
    z_extent = air_top + total_layers_thick
    z_tx = air_top + gpr_data.source_height_m
    
    if z_tx >= z_extent:
        errors.append(f"Source height ({gpr_data.source_height_m} m) would exceed model z-extent. Consider reducing source_height_m or increasing domain size.")
    
    # 8. Check domain dimensions are positive
    if gpr_data.domain_x <= 0:
        errors.append("domain_x must be > 0")
    if gpr_data.domain_y <= 0:
        errors.append("domain_y must be > 0")
    
    # 9. Check cells_per_wavelength is positive
    if gpr_data.cells_per_wavelength <= 0:
        errors.append("cells_per_wavelength must be > 0")
    
    # 10. Check max_cell_m is positive
    if gpr_data.max_cell_m <= 0:
        errors.append("max_cell_m must be > 0")
    
    if errors:
        error_msg = "Validation errors found:\n" + "\n".join(f"  - {err}" for err in errors)
        logger.warning(f"[TOOL RESULT] validate_gpr_parameters - Validation failed with {len(errors)} error(s)")
        logger.debug(f"[TOOL RESULT] validate_gpr_parameters - Validation errors: {error_msg}")
        return False, error_msg
    
    logger.info("[TOOL RESULT] validate_gpr_parameters - All parameters are valid")
    return True, ""


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
    result = response.output.model_dump() if response.output else None
    logger.info("[TOOL RESULT] extraction_agent - Parameter extraction completed")
    logger.debug(f"[TOOL RESULT] extraction_agent - Extracted parameters: {json.dumps(result, indent=2, default=str)}")
    return result


async def central_agent(initial_input: str):
    """
    Interactive workflow that loops until all inputs are complete.
    
    Args:
        initial_input: Initial user query
        get_user_input_func: Optional function to get user input. If None, will return on first missing params.
                            Should be a callable that takes a prompt string and returns user input string.
    
    Returns:
        AgentRunResult with thought_process attribute containing structured thought process data
    """
    central_agent = Agent(
        name="Central Agent",
        system_prompt="""You are a agent that coordinates the workflows.

        You will be given a user query and you will need to extract the parameters from the user query
        You will then need to check if the parameters are complete and valid.
        If the parameters are complete and valid, you will need to generate the gprmax input file using the generate_gprmax_input_file_tool.
        If the parameters are not complete or valid, you will need to ask the user for the missing or incorrect parameters. 
        Repeat the process until the parameters are complete and valid and then generate the gprmax input file using the generate_gprmax_input_file_tool.


        """,
        model=openai_model,
        tools=[generate_gprmax_input_file_tool,check_input_completeness,validate_gpr_parameters,extraction_agent],
    )
    
    try:
        # Track thought process steps
        thought_process = []
        
        # Run the agent and capture the result
        central_agent_result = await central_agent.run(initial_input)
        
        # Extract thought process from messages
        # Try all_messages first, then fallback to new_messages
        messages_to_process = []
        if hasattr(central_agent_result, 'all_messages'):
            all_messages = central_agent_result.all_messages
            # Check if it's a method (callable) or a property
            if callable(all_messages):
                try:
                    messages_to_process = all_messages()
                except:
                    pass
            elif all_messages:
                messages_to_process = all_messages
        
        if not messages_to_process and hasattr(central_agent_result, 'new_messages'):
            new_messages = central_agent_result.new_messages
            # Check if it's a method (callable) or a property
            if callable(new_messages):
                try:
                    messages_to_process = new_messages()
                except:
                    pass
            elif new_messages:
                messages_to_process = new_messages
        
        for msg in messages_to_process:
            msg_type = type(msg).__name__
            
            # Debug: log message structure to understand tool call format
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                logger.debug(f"Found tool_calls in message type {msg_type}: {len(msg.tool_calls)} tool calls")
                if msg.tool_calls:
                    logger.debug(f"First tool_call type: {type(msg.tool_calls[0])}, dir: {[x for x in dir(msg.tool_calls[0]) if not x.startswith('_')]}")
            
            # Extract message content
            if hasattr(msg, 'content') and msg.content:
                content = str(msg.content)
                role = getattr(msg, 'role', 'unknown')
                # Only add assistant/user messages, skip system messages
                if role in ['assistant', 'user']:
                    step = {
                        'type': 'message',
                        'role': role,
                        'content': content
                    }
                    thought_process.append(step)
            
            # Extract tool calls
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    # Try multiple ways to get tool name
                    tool_name = 'unknown'
                    try:
                        # First, try direct attributes
                        if hasattr(tool_call, 'name'):
                            tool_name = str(tool_call.name)
                        elif hasattr(tool_call, 'tool_name'):
                            tool_name = str(tool_call.tool_name)
                        elif hasattr(tool_call, 'function'):
                            # Some formats have function.name
                            func = tool_call.function
                            if hasattr(func, 'name'):
                                tool_name = str(func.name)
                        elif isinstance(tool_call, dict):
                            tool_name = str(tool_call.get('name', tool_call.get('tool_name', 'unknown')))
                        else:
                            # Try to get all attributes and look for name-like ones
                            attrs = [attr for attr in dir(tool_call) if not attr.startswith('_')]
                            logger.debug(f"Tool call attributes: {attrs}")
                            # Try common attribute names
                            for attr_name in ['name', 'tool_name', 'function_name', 'tool']:
                                if hasattr(tool_call, attr_name):
                                    attr_value = getattr(tool_call, attr_name)
                                    if attr_value:
                                        tool_name = str(attr_value)
                                        break
                    except Exception as e:
                        logger.debug(f"Error extracting tool name: {e}, tool_call type: {type(tool_call)}")
                    
                    # Try multiple ways to get tool arguments
                    tool_args = {}
                    try:
                        if hasattr(tool_call, 'args'):
                            tool_args = tool_call.args
                        elif hasattr(tool_call, 'arguments'):
                            tool_args = tool_call.arguments
                        elif hasattr(tool_call, 'function'):
                            # Some formats have function.arguments
                            func = tool_call.function
                            if hasattr(func, 'arguments'):
                                if isinstance(func.arguments, str):
                                    try:
                                        tool_args = json.loads(func.arguments)
                                    except:
                                        tool_args = {}
                                else:
                                    tool_args = func.arguments
                        elif isinstance(tool_call, dict):
                            tool_args = tool_call.get('args', tool_call.get('arguments', {}))
                    except Exception as e:
                        logger.debug(f"Error extracting tool args: {e}")
                    
                    step = {
                        'type': 'tool_call',
                        'tool_name': tool_name,
                        'args': tool_args
                    }
                    thought_process.append(step)
            
            # Extract tool results
            if hasattr(msg, 'tool_result') and msg.tool_result:
                tool_result = msg.tool_result
                step = {
                    'type': 'tool_result',
                    'result': str(tool_result)
                }
                thought_process.append(step)
        
        # Attach thought process to result
        if hasattr(central_agent_result, '__dict__'):
            central_agent_result.thought_process = thought_process
        else:
            # If it's a dataclass, we'll handle it in app.py
            pass
        
        return central_agent_result, thought_process
    except Exception as e:
        logger.error(f"[CENTRAL AGENT] Error during workflow execution: {str(e)}", exc_info=True)
        raise 
    


if __name__ == "__main__":
  async def main():
    try:
      inp = """
title=my_simulation,
enforce_validity=True,
We need to create a simulation. Create a 3 layer simulation with each layer with following config
* thickness=0.4 , sand percentage=60, silt percentage= 30, clay percentage= 10, moisture content= 0.10, bulk density= 1.3, particle_density=2.65, salinity class=fresh, name=l1
* thickness=0.6 , sand percentage=35, silt percentage= 40, clay percentage= 25, moisture content= 0.18, bulk density= 1.5, particle_density=2.65, salinity class=brackish, name=l2
* thickness=1.0 , sand percentage=20, silt percentage= 40, clay percentage= 40, moisture content= 0.25, bulk density= 1.6, organic fraction=0.2 ,particle_density=2.65, name=l3

Waveform= Ricker with 1.0 amplitude, 1.5e9 center frequency and name of my_ricker,Antenna= hertzian_dipole with axis as z and tx_rx_offset of 0.08
source_height=0.07,domain_xy= 0.8 and 0.4,cells per wavelength= 15,max cells= 0.003,temperature = 20.0,model= mironov
      """
      result, thought_process = await central_agent(inp)
      print(result.output)
    except Exception as e:
      print(f"Error: {e}")
      import traceback
      traceback.print_exc()
  
  asyncio.run(main())