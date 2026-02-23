from pydantic_ai import Agent
from pydantic import BaseModel
from physics_modelling import (
    generate_gprmax_input_file, VALID_WAVEFORMS,
    CylinderObject, BoxObject, SphereObject, CustomMaterial,
    SurfaceRoughnessConfig, RxArrayConfig, SnapshotConfig,
)
from typing import List, Optional, Dict, Any, Tuple
import os
import dotenv
import asyncio
import openai
import json
import logging
import subprocess
import shutil
import sys
from pathlib import Path
from schema import (
    GprSchema, WaveformSchema, AntennaSchema, LayerSchema,
    ExtractedLayers, ExtractedLayerParams,
    ExtractedAntennaWaveform,
    ExtractedModelConfig,
    ExtractedOptionalParams,
    AggregatedExtraction,
    SampledLayerValues, SampleRecord, DatasetGenerationResult,
)
from resolvers import merge_extractions, merge_aggregations, resolve_layers
from rag import GeophysicsRAG

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

def call_rag(query: str) -> List[str]:
    """
    Call the RAG model to retrieve relevant information from the database.
    """
    rag = GeophysicsRAG(mode="inference")
    results = rag.search(query)
    relevant_docs = []
    for i, (doc, score) in enumerate(results, 1):
        if score > 0.5:
            relevant_docs.append(doc)
    return relevant_docs

def get_workspace_directory() -> Path:
    """
    Get the workspace directory outside Flask's working directory.
    This prevents file changes from triggering Flask's hot reload.
    
    Returns:
        Path: Absolute path to the workspace directory
    """
    # Use environment variable if set, otherwise use /tmp or a directory outside project
    workspace_base = os.getenv("GPR_WORKSPACE_DIR", None)
    
    if workspace_base:
        workspace_dir = Path(workspace_base)
    else:
        # Default to /tmp/intelligent_gpr_workspace (outside Flask working directory)
        workspace_dir = Path("/tmp/intelligent_gpr_workspace")
    
    # Create directory if it doesn't exist
    workspace_dir.mkdir(parents=True, exist_ok=True)
    return workspace_dir

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
        
        # Get workspace directory (outside Flask working directory)
        workspace_dir = get_workspace_directory()
        
        # Clone repository to workspace directory if it doesn't exist
        gprmax_dir = workspace_dir / "gprMax"
        if not gprmax_dir.exists():
            logger.info(f"Cloning gprMax repository to {gprmax_dir}...")
            # Change to workspace directory for cloning
            original_dir = os.getcwd()
            try:
                os.chdir(workspace_dir)
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
            finally:
                # Always restore original directory
                os.chdir(original_dir)
        
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
    
    # Run the simulation with live streaming to terminal
    logger.info(f"Running gprMax simulation with input file: {input_file}")
    try:
        # Use absolute path for input file
        abs_input_file = os.path.abspath(input_file)
        
        # Build the command with unbuffered Python flag
        # Set environment variable for unbuffered output
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        env['PYTHONIOENCODING'] = 'utf-8'
        
        if not gprmax_installed:
            # Get conda python executable path to avoid conda run buffering
            # Try to find the python in the conda environment
            conda_python = None
            try:
                # Get conda python path
                conda_info = subprocess.run(
                    ["conda", "info", "--base"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if conda_info.returncode == 0:
                    conda_base = conda_info.stdout.strip()
                    conda_python = os.path.join(conda_base, "envs", "gprMax", "bin", "python")
                    if not os.path.exists(conda_python):
                        # Try Windows path
                        conda_python = os.path.join(conda_base, "envs", "gprMax", "python.exe")
                    if not os.path.exists(conda_python):
                        conda_python = None
            except:
                pass
            
            if conda_python and os.path.exists(conda_python):
                # Use python directly from conda env (avoids conda run buffering)
                cmd = [conda_python, "-u", "-m", "gprMax", abs_input_file]
            else:
                # Fallback to conda run with -u flag
                cmd = ["conda", "run", "-n", "gprMax", "python", "-u", "-m", "gprMax", abs_input_file]
        else:
            cmd = ["python", "-u", "-m", "gprMax", abs_input_file]
        
        # Use Popen to stream output in real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr into stdout
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True,
            env=env
        )
        
        # Capture output and stream to terminal simultaneously
        combined_output = ""
        print("\n" + "="*80, flush=True)
        print("gprMax Simulation Output (Live):", flush=True)
        print("="*80 + "\n", flush=True)
        
        try:
            # Read line by line and print immediately
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    # Process has finished
                    break
                if output:
                    # Print to terminal immediately (no buffering)
                    print(output, end='', flush=True)
                    # Also capture for return value
                    combined_output += output
                    # Log important lines
                    line_lower = output.lower()
                    if any(keyword in line_lower for keyword in ['error', 'warning', 'completed', 'failed']):
                        logger.info(f"[GPRMAX] {output.strip()}")
            
            # Get return code
            return_code = process.poll()
            
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            error_msg = "Simulation timed out after 1 hour"
            logger.error(f"[TOOL RESULT] run_gprmax_simulation_tool - {error_msg}")
            print(f"\n{error_msg}\n")
            return error_msg
        
        print("\n" + "="*80)
        print("Simulation Complete")
        print("="*80 + "\n")
        sys.stdout.flush()
        
        if return_code != 0:
            error_msg = f"gprMax simulation failed (exit code: {process.returncode})\n\n{combined_output}"
            logger.error(f"[TOOL RESULT] run_gprmax_simulation_tool - {error_msg}")
            return error_msg
        
        # Return the full simulation logs
        if combined_output:
            return combined_output
        else:
            # Fallback if no output captured
            return "Simulation completed successfully, but no output was captured."
        
    except subprocess.TimeoutExpired:
        error_msg = "Simulation timed out after 1 hour"
        logger.error(f"[TOOL RESULT] run_gprmax_simulation_tool - {error_msg}")
        return error_msg
    except Exception as e:
        error_msg = f"Error running gprMax simulation: {str(e)}"
        logger.error(f"[TOOL RESULT] run_gprmax_simulation_tool - {error_msg}")
        return error_msg

# Global variable to store the current output filename for the tool
_current_output_filename = None

def generate_gprmax_input_file_tool(gpr_data: GprSchema) -> str:
    """Generate gprMax input file from complete GprSchema"""
    global _current_output_filename
    logger.info(f"[TOOL CALL] generate_gprmax_input_file_tool - Generating gprMax input file for: {gpr_data.title}")

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

    output_filename = _current_output_filename if _current_output_filename else "generated.in"

    # Build objects list from schema (not yet wired through GprSchema but ready for future)
    objects = None

    # Convert surface_roughness schema to dataclass
    surface_roughness = None
    if gpr_data.surface_roughness is not None:
        sr = gpr_data.surface_roughness
        surface_roughness = SurfaceRoughnessConfig(
            fractal_dim=sr.fractal_dim,
            weight_x=sr.weight_x,
            weight_y=sr.weight_y,
            amplitude_m=sr.amplitude_m,
            add_water=sr.add_water,
            water_depth_m=sr.water_depth_m,
            seed=sr.seed,
        )

    # Convert rx_array schema to dataclass
    rx_array = None
    if gpr_data.rx_array is not None:
        ra = gpr_data.rx_array
        rx_array = RxArrayConfig(
            x1=ra.x1, y1=ra.y1, z1=ra.z1,
            x2=ra.x2, y2=ra.y2, z2=ra.z2,
            dx=ra.dx, dy=ra.dy, dz=ra.dz,
        )

    # Convert snapshots schema to dataclass list
    snapshots = None
    if gpr_data.snapshots is not None:
        snapshots = []
        for s in gpr_data.snapshots:
            snapshots.append(SnapshotConfig(
                time_s=s.time_s, filename=s.filename,
                dx=s.dx, dy=s.dy, dz=s.dz,
                x1=s.x1, y1=s.y1, z1=s.z1,
                x2=s.x2, y2=s.y2, z2=s.z2,
            ))

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
        antenna_source_start_time=gpr_data.antenna.source_start_time,
        antenna_source_end_time=gpr_data.antenna.source_end_time,
        model_title=gpr_data.title,
        source_height_m=gpr_data.source_height_m,
        domain_xy_m=(gpr_data.domain_x, gpr_data.domain_y),
        cells_per_wavelength=int(gpr_data.cells_per_wavelength),
        max_cell_m=gpr_data.max_cell_m,
        rx_same_height=True,
        temperature_c=gpr_data.temperature_c,
        model=gpr_data.model,
        enforce_validity=gpr_data.enforce_validity,
        output_filename=output_filename,
        objects=objects,
        pml_cells=gpr_data.pml_cells,
        num_threads=gpr_data.num_threads,
        output_dir=gpr_data.output_dir,
        surface_roughness=surface_roughness,
        snapshots=snapshots,
        rx_array=rx_array,
    )
    result_msg = f"Successfully generated gprMax input file for: {gpr_data.title} \n\n {text}"
    logger.info(f"[TOOL RESULT] generate_gprmax_input_file_tool - {result_msg}")
    return result_msg



def check_input_completeness(aggregated: AggregatedExtraction) -> tuple[bool, str]:
    """Check if all required parameters are present after subagent extraction.

    Uses the resolver's merge_extractions which returns a list of missing fields.
    Returns (is_complete, missing_params_message).
    """
    logger.info("[TOOL CALL] check_input_completeness - Checking if all required parameters are provided")

    _, missing = merge_extractions(
        aggregated.layers,
        aggregated.antenna_waveform,
        aggregated.model_params,
        aggregated.optional_params,
    )

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
    if waveform_kind_lower not in VALID_WAVEFORMS:
        errors.append(f"Invalid waveform.kind '{gpr_data.waveform.kind}'. Must be one of: {', '.join(sorted(VALID_WAVEFORMS))}")
    
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
    total_layers_thick = sum(layer.thickness_m for layer in gpr_data.layers)
    air_top = max(gpr_data.source_height_m + 15 * gpr_data.max_cell_m, 0.10)
    z_extent = air_top + total_layers_thick
    z_tx = air_top - gpr_data.source_height_m

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

    # 11. pml_cells >= 2 if set
    if gpr_data.pml_cells is not None and gpr_data.pml_cells < 2:
        errors.append("pml_cells must be >= 2 if set")

    # 12. Surface roughness validations
    if gpr_data.surface_roughness is not None:
        sr = gpr_data.surface_roughness
        if sr.amplitude_m <= 0:
            errors.append("surface_roughness.amplitude_m must be > 0")
        if sr.fractal_dim < 0:
            errors.append("surface_roughness.fractal_dim must be >= 0")
        if sr.add_water and sr.water_depth_m >= sr.amplitude_m:
            errors.append("surface_roughness.water_depth_m must be < amplitude_m when add_water=True")

    # 13. Antenna source timing
    if gpr_data.antenna.source_start_time is not None and gpr_data.antenna.source_end_time is not None:
        if gpr_data.antenna.source_start_time >= gpr_data.antenna.source_end_time:
            errors.append("antenna.source_start_time must be < source_end_time")

    if errors:
        error_msg = "Validation errors found:\n" + "\n".join(f"  - {err}" for err in errors)
        logger.warning(f"[TOOL RESULT] validate_gpr_parameters - Validation failed with {len(errors)} error(s)")
        logger.debug(f"[TOOL RESULT] validate_gpr_parameters - Validation errors: {error_msg}")
        return False, error_msg
    
    logger.info("[TOOL RESULT] validate_gpr_parameters - All parameters are valid")
    return True, ""


# ---------------------------------------------------------------------------
# Subagent definitions (4 focused extraction agents)
# ---------------------------------------------------------------------------

_layer_extraction_agent = Agent(
    name="Layer Extraction Subagent",
    system_prompt="""You are a soil-layer parameter extraction assistant for gprMax GPR simulations.
Extract ONLY the soil layer PARAMETER RANGES from the user's message. Ignore antenna, waveform, model, and other parameters.

CONTEXT: You may receive a "CURRENT PARAMETERS" block showing previously collected values.
- Extract ONLY new or changed values from the user's latest message.
- If the user is NOT mentioning any layer information at all, return num_layers=0 and layers=[].
- If the user asks to modify a specific layer (e.g. "change layer 2 thickness to 0.2-0.6"), return
  the FULL updated layer list with that modification applied (use existing values from CURRENT PARAMETERS
  for unchanged fields).
- If the user provides entirely new layers, return those and ignore previous layers.

For each layer, extract EXPLICIT MIN and MAX numeric values for:
  REQUIRED:
    - thickness_m_min, thickness_m_max: layer thickness range in meters (must be > 0)
    - sand_pct_min, sand_pct_max: sand percentage range (0-100)
    - silt_pct_min, silt_pct_max: silt percentage range (0-100)
    - clay_pct_min, clay_pct_max: clay percentage range (0-100)
    - theta_v_min, theta_v_max: volumetric water content range (0.0 to 1.0)

  OPTIONAL:
    - bulk_density_gcm3_min, bulk_density_gcm3_max: bulk density range in g/cm³
      (if a single value is given, set BOTH min and max to that value)
    - particle_density_gcm3_min, particle_density_gcm3_max: particle density range in g/cm³
      (if a single value is given, set BOTH min and max to that value)
    - salinity_classes: list of allowed salinity classes
      (valid values: "fresh", "slightly_saline", "brackish", "saline")
    - organic_fraction: single value (0.0 to 1.0), NOT a range
    - porewater_sigma_Sm: porewater conductivity in S/m, single value
    - name: layer name/label

EXAMPLES:
  "thickness 0.2 to 0.5m"       → thickness_m_min=0.2, thickness_m_max=0.5
  "thickness 0.4m"               → thickness_m_min=0.4, thickness_m_max=0.4
  "sand between 40 and 70%"      → sand_pct_min=40, sand_pct_max=70
  "sand 60%"                     → sand_pct_min=60, sand_pct_max=60
  "water content 0.05–0.3"       → theta_v_min=0.05, theta_v_max=0.3
  "water content 0.15"           → theta_v_min=0.15, theta_v_max=0.15
  "bulk density 1.3"             → bulk_density_gcm3_min=1.3, bulk_density_gcm3_max=1.3
  "bulk density 1.2 to 1.6"      → bulk_density_gcm3_min=1.2, bulk_density_gcm3_max=1.6
  "particle density 2.65"        → particle_density_gcm3_min=2.65, particle_density_gcm3_max=2.65
  "fresh or brackish"            → salinity_classes=["fresh","brackish"]
  "fresh water"                  → salinity_classes=["fresh"]

IMPORTANT:
- For ALL numeric fields (required AND optional), if the user gives a single value set min=max=that value.
- Prefer extracting RANGES (min < max) when the user explicitly states a range.
- Do NOT invent values not mentioned by the user. Leave unmentioned fields as None.
- Do NOT use texture_class, moisture_state, organic_level, compaction_level, or salinity_environment.
  Only extract explicit numeric ranges.

Count the number of layers and set num_layers accordingly.
Return ONLY parameters that are explicitly mentioned or modified. Leave unmentioned fields as None.""",
    model=openai_model,
    output_type=ExtractedLayers,
)

_antenna_waveform_extraction_agent = Agent(
    name="Antenna & Waveform Extraction Subagent",
    system_prompt="""You are an antenna and waveform parameter extraction assistant for gprMax GPR simulations.
Extract ONLY antenna and waveform information from the user's message. Ignore soil layers, model config, and other parameters.

CONTEXT: You may receive a "CURRENT PARAMETERS" block showing previously collected values.
- Extract ONLY new or changed values from the user's latest message.
- If the user is NOT mentioning any antenna/waveform information, leave ALL fields as None.
- The merge system will preserve existing values for fields you leave as None.

ANTENNA parameters:
  - antenna_kind: antenna type, currently only 'hertzian_dipole' is supported
  - antenna_axis: polarization axis — 'x', 'y', or 'z'
  - antenna_preset: shortcut preset — one of: generic_200MHz, generic_400MHz, generic_800MHz, generic_1GHz
    (presets auto-resolve frequency and tx_rx_offset if not explicitly given)
  - tx_rx_offset_m: transmitter-receiver separation in meters
  - source_start_time: optional source delay in seconds
  - source_end_time: optional source cutoff time in seconds

WAVEFORM parameters:
  - waveform_kind: one of: gaussian, gaussiandot, gaussiandotnorm, gaussiandotdot,
    gaussiandotdotnorm, ricker, gaussianprime, gaussiandoubleprime, sine, contsine
  - waveform_amplitude: signal amplitude (default 1.0)
  - waveform_center_freq_hz: center frequency in Hz (e.g. 400e6 for 400 MHz)
  - waveform_name: identifier for the waveform

If the user mentions a frequency (e.g. "400 MHz antenna"), set waveform_center_freq_hz.
If the user mentions a preset name, set antenna_preset.
Return ONLY parameters that are explicitly mentioned or changed. Leave unmentioned fields as None.""",
    model=openai_model,
    output_type=ExtractedAntennaWaveform,
)

_model_extraction_agent = Agent(
    name="Model & Domain Extraction Subagent",
    system_prompt="""You are a model/domain configuration extraction assistant for gprMax GPR simulations.
Extract ONLY simulation model and domain configuration from the user's message. Ignore soil layers, antenna, waveform, and buried objects.

CONTEXT: You may receive a "CURRENT PARAMETERS" block showing previously collected values.
- Extract ONLY new or changed values from the user's latest message.
- If the user is NOT mentioning any model/domain information, leave ALL fields as None.
- The merge system will preserve existing values for fields you leave as None.

Parameters to extract:
  - model: dielectric mixing model — one of: crim, peplinski, dobson, mironov
  - title: simulation title string
  - quality: simulation quality preset — one of: fast, balanced, high_accuracy
    (auto-resolves cells_per_wavelength and max_cell_m if not explicitly given)
  - source_height_m: antenna height above ground surface in meters
  - survey_length_m: total survey length in meters (user-friendly alternative to domain_x)
  - max_depth_m: maximum investigation depth in meters (user-friendly alternative to domain_y)
  - domain_x: explicit domain size in x direction in meters
  - domain_y: explicit domain size in y direction in meters
  - cells_per_wavelength: grid resolution (typical: 10-20)
  - max_cell_m: maximum cell size in meters
  - temperature_c: temperature in Celsius
  - enforce_validity: whether to check model-specific frequency/moisture validity constraints (boolean)
  - num_samples: number of simulation .in files to generate for the dataset
    (e.g. "100 samples", "generate 50 files", "I want 200 simulations", "create 10 examples")

Return ONLY parameters that are explicitly mentioned or changed. Leave unmentioned fields as None.""",
    model=openai_model,
    output_type=ExtractedModelConfig,
)

_optional_params_extraction_agent = Agent(
    name="Optional Parameters Extraction Subagent",
    system_prompt="""You are an optional/advanced parameter extraction assistant for gprMax GPR simulations.
Extract ONLY optional and advanced parameters from the user's message. Ignore soil layers, antenna, waveform, and core model config.

CONTEXT: You may receive a "CURRENT PARAMETERS" block showing previously collected values.
- Extract ONLY new or changed values from the user's latest message.
- If the user is NOT mentioning any optional/advanced parameters, leave ALL fields as None.
- The merge system will preserve existing values for fields you leave as None.

Parameters to extract:

BURIED OBJECTS:
  - cylinders: list of {name, x1, y1, z1, x2, y2, z2, radius, material, custom_material, dielectric_smoothing}
  - boxes: list of {name, x1, y1, z1, x2, y2, z2, material, custom_material, dielectric_smoothing}
  - spheres: list of {name, cx, cy, cz, radius, material, custom_material, dielectric_smoothing}
  material defaults to 'pec' (perfect electrical conductor). custom_material is {eps_r, sigma, mu_r, sigma_m}.

SURFACE ROUGHNESS:
  - surface_roughness: {fractal_dim, weight_x, weight_y, amplitude_m, add_water, water_depth_m, seed}

RECEIVER ARRAY:
  - rx_array: {x1, y1, z1, x2, y2, z2, dx, dy, dz}

SNAPSHOTS:
  - snapshots: list of {time_s, filename, dx, dy, dz, x1, y1, z1, x2, y2, z2}

SIMULATION SETTINGS:
  - pml_cells: PML absorbing boundary thickness (integer >= 2)
  - num_threads: OpenMP thread count (integer)
  - output_dir: output directory path (string)

Return ONLY parameters that are explicitly mentioned or changed. Leave everything as None if not mentioned.
It is perfectly normal for ALL fields to be None if the user didn't mention any advanced features.""",
    model=openai_model,
    output_type=ExtractedOptionalParams,
)


# ---------------------------------------------------------------------------
# Extraction coordinator (dispatches to 4 subagents in parallel)
# ---------------------------------------------------------------------------

def _build_state_context(state: AggregatedExtraction) -> str:
    """Serialise current parameter state into a context block for subagent prompts."""
    dump = state.model_dump(exclude_none=True)
    return json.dumps(dump, indent=2, default=str)


async def extraction_agent(
    user_input: str,
    current_state: Optional[AggregatedExtraction] = None,
) -> Tuple[AggregatedExtraction, Optional[GprSchema], List[str]]:
    """Extract parameters from user input by dispatching to 4 focused subagents.

    If *current_state* is provided the agents receive it as read-only context so
    they can interpret relative modifications (e.g. "change layer 2 thickness").
    The new extraction is merged into the existing state via merge_aggregations().

    Returns (updated_state, gpr_schema_or_none, missing_fields).
    """
    logger.info("[EXTRACTION] Dispatching to 4 subagents in parallel")
    log_input = user_input[:200] + "..." if len(user_input) > 200 else user_input
    logger.debug(f"[EXTRACTION] Input: {log_input}")

    # Build the prompt: optional state context + user message
    if current_state is not None:
        context_block = _build_state_context(current_state)
        prompt = (
            f"=== CURRENT PARAMETERS (read-only context) ===\n"
            f"{context_block}\n"
            f"=== END CURRENT PARAMETERS ===\n\n"
            f"User message:\n{user_input}"
        )
    else:
        prompt = user_input

    # Run all 4 subagents concurrently
    layers_result, antenna_wf_result, model_result, optional_result = await asyncio.gather(
        _layer_extraction_agent.run(prompt),
        _antenna_waveform_extraction_agent.run(prompt),
        _model_extraction_agent.run(prompt),
        _optional_params_extraction_agent.run(prompt),
    )

    layers: ExtractedLayers = layers_result.output
    antenna_wf: ExtractedAntennaWaveform = antenna_wf_result.output
    model_cfg: ExtractedModelConfig = model_result.output
    optional_params: ExtractedOptionalParams = optional_result.output

    logger.info("[EXTRACTION] All 4 subagents completed")
    # Log density extraction per layer so failures are visible in server logs
    for i, lyr in enumerate(layers.layers, 1):
        bd_min, bd_max = lyr.bulk_density_gcm3_min, lyr.bulk_density_gcm3_max
        pd_min, pd_max = lyr.particle_density_gcm3_min, lyr.particle_density_gcm3_max
        bd_str = f"{bd_min}–{bd_max}" if bd_min is not None else "NOT extracted"
        pd_str = f"{pd_min}–{pd_max}" if pd_min is not None else "NOT extracted"
        logger.info(
            f"[EXTRACTION] Layer {i} density — bulk: {bd_str}, particle: {pd_str}"
        )

    new_extraction = AggregatedExtraction(
        layers=layers,
        antenna_waveform=antenna_wf,
        model_params=model_cfg,
        optional_params=optional_params,
    )

    # Merge into existing state
    merged = merge_aggregations(current_state, new_extraction)
    logger.info("[EXTRACTION] State merged")

    # Resolve merged state into GprSchema
    gpr_schema, missing = merge_extractions(
        merged.layers,
        merged.antenna_waveform,
        merged.model_params,
        merged.optional_params,
    )

    if missing:
        logger.info(f"[EXTRACTION] Incomplete — {len(missing)} field(s) missing")
    else:
        logger.info("[EXTRACTION] Complete — all parameters resolved")

    return merged, gpr_schema, missing


# ---------------------------------------------------------------------------
# Deterministic workflow functions (replace LLM-orchestrated central_agent)
# ---------------------------------------------------------------------------

def _setup_output_path(user_id: Optional[str] = None) -> str:
    """Set up the output file path and configure the global filename."""
    global _current_output_filename

    workspace_dir = get_workspace_directory()
    generated_files_dir = workspace_dir / "generated_files"
    generated_files_dir.mkdir(parents=True, exist_ok=True)

    if user_id:
        output_file_path = str(generated_files_dir / f"generated_{user_id}.in")
    else:
        output_file_path = str(generated_files_dir / "generated.in")

    _current_output_filename = output_file_path
    logger.info(f"[WORKFLOW] Output file path: {output_file_path}")
    return output_file_path


async def simulate_workflow(
    user_input: str,
    user_id: Optional[str] = None,
    current_state: Optional[AggregatedExtraction] = None,
) -> dict:
    """Deterministic simulation pipeline: extract -> merge -> validate -> generate.

    Accepts and returns parameter state so callers can persist it across turns.

    Returns a dict with:
      - status: "incomplete" | "invalid" | "complete" | "error"
      - message: human-readable response
      - params: updated AggregatedExtraction dict (always present)
      - (optional) missing_params, validation_errors, file_path, file_content, gpr_schema
    """
    global _current_output_filename

    try:
        # 1. Set up output file path
        output_file_path = _setup_output_path(user_id)

        # 2. Extract & merge parameters
        logger.info("[WORKFLOW] Step 1: Extracting parameters")
        merged_state, gpr_schema, missing = await extraction_agent(
            user_input, current_state=current_state
        )

        params_dump = merged_state.model_dump()

        if missing:
            _current_output_filename = None
            missing_msg = "\n".join(missing)
            message = (
                "The following parameters are missing or incomplete:\n\n"
                f"{missing_msg}\n\n"
                "Please provide the missing information to proceed."
            )
            return {
                "status": "incomplete",
                "message": message,
                "missing_params": missing_msg,
                "params": params_dump,
            }

        # 3. Validate physics constraints
        logger.info("[WORKFLOW] Step 2: Validating parameters")
        is_valid, errors = validate_gpr_parameters(gpr_schema)

        if not is_valid:
            _current_output_filename = None
            message = (
                "Parameter validation failed. Please correct the following errors:\n\n"
                f"{errors}\n\n"
                "Please provide corrected values."
            )
            return {
                "status": "invalid",
                "message": message,
                "validation_errors": errors,
                "params": params_dump,
            }

        # 4. Generate dataset (.in files + manifest)
        logger.info("[WORKFLOW] Step 3: Generating dataset")
        from dataset_generator import generate_dataset

        num_samples = merged_state.model_params.num_samples or 1
        dataset_name = gpr_schema.title or f"dataset_{user_id or 'default'}"
        title_prefix = dataset_name

        resolved_ranges = resolve_layers(merged_state.layers)

        # Build a concise density summary so the user can confirm extraction
        density_lines = []
        for i, r in enumerate(resolved_ranges, 1):
            parts = []
            if r.bulk_density_gcm3_min is not None:
                if r.bulk_density_gcm3_min == r.bulk_density_gcm3_max:
                    parts.append(f"bulk density={r.bulk_density_gcm3_min:.3f} g/cm³")
                else:
                    parts.append(
                        f"bulk density={r.bulk_density_gcm3_min:.3f}–"
                        f"{r.bulk_density_gcm3_max:.3f} g/cm³"
                    )
            else:
                parts.append("bulk density=not provided (fallback: 1.5 g/cm³ for Peplinski/Dobson; texture-based porosity for CRIM/Mironov)")
            if r.particle_density_gcm3_min is not None:
                if r.particle_density_gcm3_min == r.particle_density_gcm3_max:
                    parts.append(f"particle density={r.particle_density_gcm3_min:.3f} g/cm³")
                else:
                    parts.append(
                        f"particle density={r.particle_density_gcm3_min:.3f}–"
                        f"{r.particle_density_gcm3_max:.3f} g/cm³"
                    )
            else:
                parts.append("particle density=not provided (fallback: 2.65 g/cm³)")
            density_lines.append(f"  Layer {i}: {', '.join(parts)}")
        density_summary = "\n".join(density_lines)
        logger.info(f"[WORKFLOW] Density parameter summary:\n{density_summary}")

        dataset_result = generate_dataset(
            resolved_layer_ranges=resolved_ranges,
            gpr_schema_template=gpr_schema,
            num_samples=num_samples,
            dataset_name=dataset_name,
            title_prefix=title_prefix,
        )

        _current_output_filename = None

        if dataset_result.num_generated == 0:
            status = "error"
            message = (
                f"Dataset generation failed — no files were produced.\n\n"
                f"Errors:\n" + "\n".join(dataset_result.errors)
            )
        elif dataset_result.num_failed > 0:
            status = "partial"
            message = (
                f"Generated {dataset_result.num_generated}/{num_samples} files "
                f"({dataset_result.num_failed} failed) in {dataset_result.output_dir}\n\n"
                f"Manifest: {dataset_result.manifest_csv_path}\n\n"
                f"Density parameters used:\n{density_summary}\n\n"
                + (f"Errors:\n" + "\n".join(dataset_result.errors) if dataset_result.errors else "")
            )
        else:
            status = "complete"
            message = (
                f"Generated {dataset_result.num_generated}/{num_samples} files "
                f"in {dataset_result.output_dir}\n\n"
                f"Manifest CSV: {dataset_result.manifest_csv_path}\n"
                f"Manifest JSON: {dataset_result.manifest_json_path}\n\n"
                f"Density parameters used:\n{density_summary}"
            )

        return {
            "status": status,
            "message": message,
            "dataset_result": dataset_result.model_dump(),
            "density_summary": density_summary,
            "params": params_dump,
        }

    except Exception as e:
        _current_output_filename = None
        logger.error(f"[WORKFLOW] Error in simulate_workflow: {e}", exc_info=True)
        return {
            "status": "error",
            "message": f"An error occurred during simulation setup: {str(e)}",
            "params": current_state.model_dump() if current_state else None,
        }


async def qa_workflow(query: str) -> dict:
    """RAG-based Q&A pipeline: retrieve relevant docs then synthesise an answer.

    Returns a dict with:
      - status: "complete"
      - message: the synthesised answer
      - sources: list of retrieved document excerpts
    """
    try:
        logger.info(f"[QA WORKFLOW] Retrieving docs for: {query[:100]}")
        docs = call_rag(query)

        if not docs:
            return {
                "status": "complete",
                "message": "No relevant information was found in the knowledge base for your question.",
                "sources": [],
            }

        context = "\n\n---\n\n".join(docs)
        logger.info(f"[QA WORKFLOW] Synthesising answer from {len(docs)} documents")

        response = openai_client.chat.completions.create(
            model=openai_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a knowledgeable geophysics and Ground Penetrating Radar (GPR) assistant. "
                        "Answer the user's question using ONLY the provided context. "
                        "If the context does not contain enough information, say so honestly. "
                        "Cite relevant details from the context in your answer."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}",
                },
            ],
        )
        answer = response.choices[0].message.content

        return {
            "status": "complete",
            "message": answer,
            "sources": docs,
        }

    except Exception as e:
        logger.error(f"[QA WORKFLOW] Error: {e}", exc_info=True)
        return {
            "status": "error",
            "message": f"An error occurred while answering your question: {str(e)}",
            "sources": [],
        }

async def runner_agent(input_file: str) -> str:
    """
    Agent that runs gprMax simulation with the given input file.
    
    Args:
        input_file: Path to the gprMax input file (.in file)
    
    Returns:
        str: Result message from the simulation (success or error message)
    """
    runner_agent = Agent(
        name="Runner Agent",
        system_prompt="""You are a simulation runner agent. Your task is to run gprMax simulations.

        When given an input file path, you should call the run_gprmax_simulation_tool with that file path.
        The tool will handle running the simulation and return the result.
        """,
        model=openai_model,
        tools=[run_gprmax_simulation_tool],
    )
    
    try:
        logger.info(f"[RUNNER AGENT] Starting simulation for input file: {input_file}")
        
        # Create a prompt for the agent to run the simulation
        prompt = f"Run the gprMax simulation with input file: {input_file}"
        
        # Run the agent and capture the result
        runner_agent_result = await runner_agent.run(prompt)
        
        # Extract the output from the result
        result_output = ""
        if hasattr(runner_agent_result, 'output') and runner_agent_result.output:
            result_output = str(runner_agent_result.output)
        elif hasattr(runner_agent_result, 'messages') and runner_agent_result.messages:
            # Get the last message from the agent
            result_output = str(runner_agent_result.messages[-1]) if runner_agent_result.messages else ""
        
        # Fallback if still empty
        if not result_output:
            result_output = "Simulation completed, but no output message was generated."
        
        logger.info(f"[RUNNER AGENT] Simulation completed. Result: {result_output[:200]}...")
        return result_output
        
    except Exception as e:
        error_msg = f"Error running gprMax simulation: {str(e)}"
        logger.error(f"[RUNNER AGENT] {error_msg}", exc_info=True)
        return error_msg


if __name__ == "__main__":
    async def main():
        try:
            inp = """
Generate 5 samples. title=test_dataset, enforce_validity=False.
Layer 1: thickness 0.2 to 0.4m, sand 50 to 70%, silt 15 to 35%, clay 5 to 20%, water content 0.05 to 0.25, fresh or brackish, name=topsoil.
Layer 2: thickness 0.3 to 0.6m, sand 20 to 40%, silt 30 to 50%, clay 15 to 35%, water content 0.10 to 0.30, name=subsoil.
400 MHz ricker waveform, tx_rx_offset 0.08m.
CRIM model, domain 0.6 x 0.4m, source height 0.07m, cells per wavelength 15, max cell 0.003, temperature 20.
            """
            result = await simulate_workflow(inp, user_id="test_user")
            print(f"Status: {result['status']}")
            print(f"Message: {result['message'][:500]}")
            if result.get("dataset_result"):
                dr = result["dataset_result"]
                print(f"Generated: {dr['num_generated']}/{dr['num_requested']} files")
                print(f"Output dir: {dr['output_dir']}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    asyncio.run(main())