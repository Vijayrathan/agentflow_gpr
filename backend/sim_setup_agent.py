

from schema import GprSchema, WaveformSchema, AntennaSchema, LayerSchema, ExtractedParameters

from init import logger, openai_client, openai_model

from physics_modelling import generate_gprmax_input_file

from pathlib import Path
import os
from langchain.tools import tool

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


_current_output_filename = None


@tool
def generate_gprmax_input_file_tool(gpr_data: GprSchema) -> str:
    """Generate gprMax input file from complete GprSchema"""
    global _current_output_filename
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
    
    # Use the global output filename if set, otherwise use default
    output_filename = _current_output_filename if _current_output_filename else "generated.in"
    
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
        output_filename=output_filename,
    )
    result_msg = f"Successfully generated gprMax input file for: {gpr_data.title} \n\n {text}"
    logger.info(f"[TOOL RESULT] generate_gprmax_input_file_tool - {result_msg}")
    return result_msg


@tool
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

@tool
def format_missing_params_message(missing_params: str) -> str:
    """Format missing parameters into a user-friendly message"""
    return f"""The following parameters are missing or incomplete:

{missing_params}

Ask the user to provide all the missing information to proceed with generating the gprMax input file."""

@tool
def format_validation_errors_message(validation_errors: list[str]) -> str:
    """Format validation errors into a user-friendly message"""
    return f"""Parameter validation failed. Please correct the following errors:

{validation_errors}

Ask the user to provide corrected values for the parameters mentioned above."""

@tool
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

