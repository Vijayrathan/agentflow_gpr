from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pydantic import BaseModel, ValidationError
from physics_modelling import generate_gprmax_input_file
from typing import List, Optional, Dict, Any, Callable
import os
import dotenv
import asyncio
import json
import re
import huggingface_hub
from schema import GprSchema, WaveformSchema, AntennaSchema, LayerSchema, ExtractedParameters

dotenv.load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

huggingface_hub.login(token=HUGGINGFACE_API_KEY)

def generate_gprmax_input_file_tool(gpr_data: GprSchema) -> str:
    """Generate GPRMax input file from complete GprSchema"""
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
    
    generate_gprmax_input_file(
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
    return f"Successfully generated GPRMax input file for: {gpr_data.title}"



def ask_user_for_inputs(missing_params: str) -> str:
    """Ask user for missing parameters"""
    return f"Please provide the following missing inputs:\n{missing_params}"


class QwenExtractor:
    """Extracts parameters from user query using Qwen3-8B model"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-8B-Instruct"):
        """Initialize Qwen model and tokenizer"""
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        print(f"Loading Qwen model: {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        if not torch.cuda.is_available():
            self.model = self.model.to(self.device)
        print("Model loaded successfully.")
    
    def extract_parameters(self, user_query: str, conversation_history: Optional[List[str]] = None) -> ExtractedParameters:
        """
        Extract parameters from user query using Qwen model.
        
        Args:
            user_query: User's query string
            conversation_history: Optional list of previous user responses
            
        Returns:
            ExtractedParameters object
        """
        # Build the prompt with schema information
        system_prompt = """You are a parameter extraction assistant. Extract all parameters mentioned in the user's query about GPRMax simulation setup.

Extract the following information:
- Number of layers and their properties (thickness_m, sand_pct, silt_pct, clay_pct, theta_v, bulk_density_gcm3, particle_density_gcm3, organic_fraction, salinity_class, porewater_sigma_Sm, name)
- Waveform properties (kind, amplitude, center_freq_hz, name)
- Antenna properties (kind, axis, tx_rx_offset_m)
- Model properties (model, title, source_height_m, domain_x, domain_y, cells_per_wavelength, max_cell_m, temperature_c, enforce_validity)

Return ONLY the parameters that are explicitly mentioned. Do not make up any parameters. Strictly leave the fields as None if not mentioned.

Return your response as a valid JSON object matching this schema:
{
  "num_layers": <int or null>,
  "layers": [{"thickness_m": <float>, "sand_pct": <float>, "silt_pct": <float>, "clay_pct": <float>, "theta_v": <float>, ...}],
  "waveform": {"kind": <str>, "amplitude": <float>, "center_freq_hz": <float>, "name": <str>},
  "antenna": {"kind": <str>, "axis": <str>, "tx_rx_offset_m": <float>},
  "model": <str>,
  "title": <str>,
  "source_height_m": <float>,
  "domain_x": <float>,
  "domain_y": <float>,
  "cells_per_wavelength": <float>,
  "max_cell_m": <float>,
  "temperature_c": <float>,
  "enforce_validity": <bool>
}

Return ONLY the JSON object, no additional text."""

        # Combine conversation history if provided
        full_query = user_query
        if conversation_history:
            full_query += "\n\nAdditional user responses:\n" + "\n".join(conversation_history)
        
        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_query}
        ]
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize and generate
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=2048,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        generated_text = self.tokenizer.decode(
            generated_ids[0][model_inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # Extract JSON from response
        json_text = self._extract_json(generated_text)
        
        # Parse JSON and create ExtractedParameters
        try:
            json_data = json.loads(json_text)
            return ExtractedParameters(**json_data)
        except (json.JSONDecodeError, ValidationError) as e:
            # If parsing fails, try to create with minimal data
            print(f"Warning: Failed to parse extracted JSON: {e}")
            print(f"Generated text: {generated_text[:500]}")
            return ExtractedParameters()
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON object from text response"""
        # Try to find JSON in the response
        # Look for { ... } pattern
        start_idx = text.find('{')
        if start_idx == -1:
            return "{}"
        
        # Find matching closing brace
        brace_count = 0
        end_idx = start_idx
        for i in range(start_idx, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        
        if end_idx > start_idx:
            return text[start_idx:end_idx]
        return "{}"


def check_input_completeness(extracted: ExtractedParameters) -> tuple[bool, str]:
    """Check if all required parameters are provided. Returns (is_complete, missing_params_message)"""
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
        return False, missing_msg
    return True, ""


def format_missing_params_message(missing_params: str) -> str:
    """Format missing parameters into a user-friendly message"""
    return f"""The following parameters are missing or incomplete:

{missing_params}

Please provide all the missing information to proceed with generating the GPRMax input file."""


def format_validation_errors_message(validation_error: str) -> str:
    """Format validation errors into a user-friendly message"""
    return f"""Parameter validation failed. Please correct the following errors:

{validation_error}

Please provide corrected values for the parameters mentioned above."""


def validate_gpr_parameters(gpr_data: GprSchema) -> tuple[bool, str]:
    """
    Validate all parameters according to physics_modelling.py validity rules.
    
    Returns:
        (is_valid, error_message) - if is_valid is False, error_message contains validation errors
    """
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
        return False, error_msg
    
    return True, ""


def _convert_extracted_to_gpr_schema(extracted_params: ExtractedParameters) -> GprSchema:
    """Convert ExtractedParameters to GprSchema format"""
    # Build layers
    layers = []
    for layer_data in extracted_params.layers or []:
        # Handle both dict and object access
        if isinstance(layer_data, dict):
            layer = LayerSchema(
                name=layer_data.get("name"),
                thickness_m=layer_data["thickness_m"],
                sand_pct=layer_data["sand_pct"],
                silt_pct=layer_data["silt_pct"],
                clay_pct=layer_data["clay_pct"],
                theta_v=layer_data["theta_v"],
                bulk_density_gcm3=layer_data.get("bulk_density_gcm3"),
                particle_density_gcm3=layer_data.get("particle_density_gcm3"),
                organic_fraction=layer_data.get("organic_fraction"),
                salinity_class=layer_data.get("salinity_class"),
                porewater_sigma_Sm=layer_data.get("porewater_sigma_Sm"),
            )
        else:
            # If it's already a Pydantic model or object
            layer = LayerSchema(
                name=getattr(layer_data, "name", None),
                thickness_m=layer_data.thickness_m,
                sand_pct=layer_data.sand_pct,
                silt_pct=layer_data.silt_pct,
                clay_pct=layer_data.clay_pct,
                theta_v=layer_data.theta_v,
                bulk_density_gcm3=getattr(layer_data, "bulk_density_gcm3", None),
                particle_density_gcm3=getattr(layer_data, "particle_density_gcm3", None),
                organic_fraction=getattr(layer_data, "organic_fraction", None),
                salinity_class=getattr(layer_data, "salinity_class", None),
                porewater_sigma_Sm=getattr(layer_data, "porewater_sigma_Sm", None),
            )
        layers.append(layer)
    
    # Build waveform - handle both dict and object access
    waveform_dict = extracted_params.waveform
    if isinstance(waveform_dict, dict):
        waveform = WaveformSchema(
            kind=waveform_dict["kind"],
            amplitude=waveform_dict["amplitude"],
            center_freq_hz=waveform_dict["center_freq_hz"],
            name=waveform_dict["name"],
        )
    else:
        waveform = WaveformSchema(
            kind=waveform_dict.kind,
            amplitude=waveform_dict.amplitude,
            center_freq_hz=waveform_dict.center_freq_hz,
            name=waveform_dict.name,
        )
    
    # Build antenna - handle both dict and object access
    antenna_dict = extracted_params.antenna
    if isinstance(antenna_dict, dict):
        antenna = AntennaSchema(
            kind=antenna_dict["kind"],
            axis=antenna_dict["axis"],
            tx_rx_offset_m=antenna_dict["tx_rx_offset_m"],
        )
    else:
        antenna = AntennaSchema(
            kind=antenna_dict.kind,
            axis=antenna_dict.axis,
            tx_rx_offset_m=antenna_dict.tx_rx_offset_m,
        )
    
    # Build complete GprSchema
    gpr_data = GprSchema(
        model=extracted_params.model,
        title=extracted_params.title,
        source_height_m=extracted_params.source_height_m,
        domain_x=extracted_params.domain_x,
        domain_y=extracted_params.domain_y,
        cells_per_wavelength=extracted_params.cells_per_wavelength,
        max_cell_m=extracted_params.max_cell_m,
        temperature_c=extracted_params.temperature_c,
        enforce_validity=extracted_params.enforce_validity if extracted_params.enforce_validity is not None else True,
        waveform=waveform,
        antenna=antenna,
        layers=layers,
    )
    
    return gpr_data


class CentralOrchestratorAgent:
    """
    Central orchestrator agent that manages the complete workflow:
    1. Takes user query
    2. Calls extraction function (LLM parsing)
    3. Calls completeness checker
    4. Calls validity checker
    5. If both pass, calls generate file function
    6. Otherwise asks user to fix issues and re-runs loop
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-8B-Instruct"):
        """Initialize the orchestrator with Qwen extractor"""
        self.extractor = QwenExtractor(model_name)
    
    async def process_query(
        self,
        user_query: str,
        conversation_history: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Process a user query through the complete workflow.
        
        Args:
            user_query: User's query string
            conversation_history: Optional list of previous user responses
            
        Returns:
            dict with status and result information:
            - "status": "complete", "incomplete", "validation_error", or "error"
            - Additional fields based on status
        """
        try:
            # Step 1: Extract parameters from user query
            extracted_params = self.extractor.extract_parameters(
                user_query,
                conversation_history
            )
            
            # Step 2: Check input completeness
            is_complete, missing_params = check_input_completeness(extracted_params)
            
            if not is_complete:
                # Parameters incomplete, ask user for missing inputs
                missing_msg = format_missing_params_message(missing_params)
                user_message = ask_user_for_inputs(missing_msg)
                
                return {
                    "status": "incomplete",
                    "missing_params": missing_msg,
                    "user_message": user_message
                }
            
            # Step 3: All parameters are complete, convert to GprSchema
            try:
                gpr_data = _convert_extracted_to_gpr_schema(extracted_params)
            except Exception as e:
                return {
                    "status": "error",
                    "error": f"Failed to convert extracted parameters: {str(e)}"
                }
            
            # Step 4: Validate parameters before generating the file
            is_valid, validation_error = validate_gpr_parameters(gpr_data)
            if not is_valid:
                validation_msg = format_validation_errors_message(validation_error)
                user_message = ask_user_for_inputs(validation_msg)
                return {
                    "status": "validation_error",
                    "error": validation_error,
                    "validation_message": validation_msg,
                    "user_message": user_message
                }
            
            # Step 5: Generate the file
            result_message = generate_gprmax_input_file_tool(gpr_data)
            
            return {
                "status": "complete",
                "output": result_message,
                "data": gpr_data.model_dump()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def run_interactive_loop(
        self,
        initial_input: str,
        get_user_input_func: Optional[Callable[[str], str]] = None
    ) -> Dict[str, Any]:
        """
        Run interactive loop until all inputs are complete and valid.
        
        Args:
            initial_input: Initial user query
            get_user_input_func: Optional function to get user input.
                                Should be a callable that takes a prompt string and returns user input string.
        
        Returns:
            dict with workflow result
        """
        user_responses = []
        max_iterations = 10  # Prevent infinite loops
        
        for iteration in range(max_iterations):
            result = await self.process_query(initial_input, user_responses if user_responses else None)
            
            if result["status"] == "complete":
                print("All parameters complete!")
                print(result["output"])
                return result
            elif result["status"] == "incomplete":
                print(f"\nMissing parameters (iteration {iteration + 1}):")
                print(result["user_message"])
                
                # If we have a function to get user input, continue the loop
                if get_user_input_func:
                    try:
                        user_response = get_user_input_func(result["user_message"])
                        if user_response:
                            user_responses.append(user_response)
                            continue
                    except Exception as e:
                        print(f"Error getting user input: {e}")
                
                # Otherwise, return the missing params message
                return result
            elif result["status"] == "validation_error":
                print(f"\nValidation error (iteration {iteration + 1}):")
                print(result.get("user_message", result.get("validation_message", "Unknown validation error")))
                
                # If we have a function to get user input, continue the loop to allow corrections
                if get_user_input_func:
                    try:
                        user_response = get_user_input_func(result.get("user_message", "Please correct the validation errors above."))
                        if user_response:
                            user_responses.append(user_response)
                            continue
                    except Exception as e:
                        print(f"Error getting user input: {e}")
                
                # Otherwise, return the validation error message
                return result
            elif result["status"] == "error":
                print(f"Error: {result['error']}")
                return result
        
        print("✗ Maximum iterations reached. Please try again with more complete input.")
        return {"status": "max_iterations_reached"}


# Global orchestrator instance (lazy initialization)
_orchestrator: Optional[CentralOrchestratorAgent] = None

def get_orchestrator(model_name: str = "Qwen/Qwen2.5-8B-Instruct") -> CentralOrchestratorAgent:
    """Get or create the global orchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = CentralOrchestratorAgent(model_name)
    return _orchestrator


# Main code entrypoint - Legacy functions for backward compatibility
async def run_extraction_workflow(initial_input: str, user_responses: Optional[List[str]] = None):
    """
    Run the workflow to extract parameters and generate GPRMax input file.
    Uses the central orchestrator agent.
    
    Args:
        initial_input: Initial user query
        user_responses: Optional list of subsequent user responses (for iterative input collection)
    
    Returns:
        dict with either:
        - "status": "complete", "output": generated file info
        - "status": "incomplete", "missing_params": formatted missing parameters message
    """
    orchestrator = get_orchestrator()
    return await orchestrator.process_query(initial_input, user_responses)


async def run_interactive_workflow(initial_input: str, get_user_input_func=None):
    """
    Interactive workflow that loops until all inputs are complete.
    Uses the central orchestrator agent.
    
    Args:
        initial_input: Initial user query
        get_user_input_func: Optional function to get user input. If None, will return on first missing params.
                            Should be a callable that takes a prompt string and returns user input string.
    
    Returns:
        dict with workflow result
    """
    orchestrator = get_orchestrator()
    return await orchestrator.run_interactive_loop(initial_input, get_user_input_func)


# # Alias for backward compatibility with app.py
# async def run_workflow(initial_input: str, user_responses: Optional[List[str]] = None):
#     """Alias for run_extraction_workflow for backward compatibility"""
#     return await run_extraction_workflow(initial_input, user_responses)


if __name__ == "__main__":
  async def main():
    try:
      inp = """
I want to simulate a model with 2 layers.
      """
      result = await run_interactive_workflow(inp)
      
      if result.get("status") == "complete":
        with open("output.json", "w") as f:
          json.dump(result["data"], f, indent=2)
        print("\n Output saved to output.json")
    except Exception as e:
      print(f"Error: {e}")
      import traceback
      traceback.print_exc()
  
  asyncio.run(main())
