"""
LangGraph state schema for GPR simulation workflow.

This module defines the complete state structure that persists across
conversation turns, tracking parameter collection, validation, and file generation.
"""

from __future__ import annotations
from typing import TypedDict, Annotated, Sequence, Optional, List, Dict, Any, Set
from langchain_core.messages import BaseMessage
import operator

from soil_setup.soil_structure_schema import (
    UserLayerSimple,
    UserAntennaSimple,
    UserWaveformSimple,
    UserModelSimple,
    UserCylinderObject,
    UserBoxObject,
)
from schema import ExtractedParameters


class SimulationState(TypedDict):
    """
    Complete state for the LangGraph GPR simulation workflow.
    
    This state tracks:
    - Conversation history
    - Mode routing (simulation vs RAG)
    - Use case identification
    - Incremental parameter collection
    - Real-time validation
    - Final resolution and file generation
    """
    
    # ============================================================================
    # CONVERSATION TRACKING
    # ============================================================================
    messages: Annotated[Sequence[BaseMessage], operator.add]
    """Conversation history - automatically appended"""
    
    # ============================================================================
    # MODE ROUTING
    # ============================================================================
    mode: Optional[str]
    """Current mode: 'simulation' or 'rag'"""
    
    # ============================================================================
    # USE CASE IDENTIFICATION
    # ============================================================================
    use_case: Optional[str]
    """User's simulation scenario (e.g., 'agriculture_field_with_buried_pipe')"""
    
    use_case_confirmed: bool
    """Whether user has confirmed the identified use case"""
    
    use_case_description: Optional[str]
    """Human-readable description of the use case"""
    
    # ============================================================================
    # PARAMETER COLLECTION (User-Friendly Schema)
    # ============================================================================
    
    # Layers - progressively filled
    num_layers: Optional[int]
    """Number of soil layers in the simulation"""
    
    layers: Optional[List[Dict[str, Any]]]
    """List of UserLayerSimple dicts - incrementally filled"""
    
    # Antenna configuration
    antenna: Optional[Dict[str, Any]]
    """UserAntennaSimple dict"""
    
    # Waveform configuration
    waveform: Optional[Dict[str, Any]]
    """UserWaveformSimple dict"""
    
    # Model configuration
    model: Optional[Dict[str, Any]]
    """UserModelSimple dict"""
    
    # Optional objects (pipes, tanks, etc.)
    objects: Optional[List[Dict[str, Any]]]
    """List of UserCylinderObject or UserBoxObject dicts"""
    
    # ============================================================================
    # WORKFLOW TRACKING
    # ============================================================================
    required_params: List[str]
    """List of parameter names required for this use case"""
    
    collected_params: Set[str]
    """Set of parameter names that have been filled"""
    
    validation_errors: Dict[str, str]
    """Current validation issues: {param_name: error_message}"""
    
    current_focus: Optional[str]
    """Which parameter group we're currently collecting"""
    
    collection_stage: Optional[str]
    """Current stage: 'use_case', 'model', 'layers', 'antenna', 'objects', 'complete'"""
    
    # ============================================================================
    # RESOLUTION & GENERATION
    # ============================================================================
    resolved_params: Optional[Dict[str, Any]]
    """ExtractedParameters dict - full resolved simulation parameters"""
    
    file_generated: bool
    """Whether the .in file has been generated"""
    
    file_path: Optional[str]
    """Path to the generated .in file"""
    
    file_content: Optional[str]
    """Content of the generated .in file"""
    
    # ============================================================================
    # HELPER FLAGS
    # ============================================================================
    parameters_complete: bool
    """Whether all required parameters have been collected"""
    
    parameters_valid: bool
    """Whether all parameters pass validation"""
    
    awaiting_user_input: bool
    """Whether we're waiting for user to provide more information"""
    
    last_question: Optional[str]
    """Last question asked to the user (for context)"""


def create_initial_state() -> SimulationState:
    """
    Create an initial empty state for a new conversation.
    
    Returns:
        SimulationState with default values
    """
    return SimulationState(
        messages=[],
        mode=None,
        use_case=None,
        use_case_confirmed=False,
        use_case_description=None,
        num_layers=None,
        layers=None,
        antenna=None,
        waveform=None,
        model=None,
        objects=None,
        required_params=[],
        collected_params=set(),
        validation_errors={},
        current_focus=None,
        collection_stage=None,
        resolved_params=None,
        file_generated=False,
        file_path=None,
        file_content=None,
        parameters_complete=False,
        parameters_valid=False,
        awaiting_user_input=False,
        last_question=None,
    )


def merge_layer_params(
    existing_layers: Optional[List[Dict[str, Any]]],
    new_layer_data: Dict[str, Any],
    layer_index: int
) -> List[Dict[str, Any]]:
    """
    Merge new layer parameters with existing ones.
    
    Args:
        existing_layers: Current layers list
        new_layer_data: New parameters for a specific layer
        layer_index: Which layer to update (0-based)
    
    Returns:
        Updated layers list
    """
    if existing_layers is None:
        existing_layers = []
    
    # Ensure we have enough layer dicts
    while len(existing_layers) <= layer_index:
        existing_layers.append({})
    
    # Merge new data into the layer
    existing_layers[layer_index].update(new_layer_data)
    
    return existing_layers


def update_collected_params(state: SimulationState) -> Set[str]:
    """
    Scan the state and determine which parameters have been collected.
    
    Args:
        state: Current simulation state
    
    Returns:
        Set of collected parameter names
    """
    collected = set()
    
    # Model parameters
    if state.get("model"):
        model = state["model"]
        if model.get("title"):
            collected.add("model.title")
        if model.get("quality"):
            collected.add("model.quality")
        if model.get("survey_length_m"):
            collected.add("model.survey_length_m")
        if model.get("max_depth_m"):
            collected.add("model.max_depth_m")
        if model.get("antenna_height_m") is not None:
            collected.add("model.antenna_height_m")
        if model.get("temperature_c") is not None:
            collected.add("model.temperature_c")
    
    # Antenna parameters
    if state.get("antenna"):
        antenna = state["antenna"]
        if antenna.get("preset"):
            collected.add("antenna.preset")
        if antenna.get("axis"):
            collected.add("antenna.axis")
    
    # Waveform parameters
    if state.get("waveform"):
        waveform = state["waveform"]
        if waveform.get("kind"):
            collected.add("waveform.kind")
        if waveform.get("name"):
            collected.add("waveform.name")
    
    # Layer parameters
    if state.get("num_layers"):
        collected.add("num_layers")
        
        if state.get("layers"):
            for i, layer in enumerate(state["layers"]):
                if layer.get("thickness_m"):
                    collected.add(f"layer.{i}.thickness_m")
                if layer.get("texture_class"):
                    collected.add(f"layer.{i}.texture_class")
                if layer.get("moisture_state"):
                    collected.add(f"layer.{i}.moisture_state")
    
    # Objects (if present)
    if state.get("objects"):
        collected.add("objects")
    
    return collected


def get_required_params_for_use_case(use_case: str) -> List[str]:
    """
    Determine which parameters are required based on the use case.
    
    Args:
        use_case: Identified use case string
    
    Returns:
        List of required parameter names
    """
    # Base required parameters for all simulations
    base_required = [
        "model.title",
        "model.quality",
        "model.survey_length_m",
        "model.max_depth_m",
        "model.antenna_height_m",
        "antenna.preset",
        "antenna.axis",
        "num_layers",
    ]
    
    # Use case specific requirements
    if "buried" in use_case.lower() or "pipe" in use_case.lower() or "tank" in use_case.lower():
        base_required.append("objects")
    
    # Note: Layer-specific requirements are dynamic based on num_layers
    # They will be added once num_layers is known
    
    return base_required

