"""
Use Case Node - Identifies user's simulation scenario and required parameters.

This node:
1. Extracts the use case from user's initial query
2. Asks for confirmation
3. Determines which parameters are needed based on the use case
"""

from typing import Dict, Any
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from langgraph_state import SimulationState, get_required_params_for_use_case
from init import openai_api_key, openai_model, logger


def use_case_node(state: SimulationState) -> Dict[str, Any]:
    """
    Identify and confirm the user's simulation use case.
    
    Args:
        state: Current simulation state
    
    Returns:
        Updated state with use case information
    """
    logger.info("[USE_CASE] Processing use case identification...")
    
    messages = state.get("messages", [])
    use_case_confirmed = state.get("use_case_confirmed", False)
    
    # If use case already confirmed, don't reset collection stage
    if use_case_confirmed:
        logger.info("[USE_CASE] Use case already confirmed, proceeding to parameter collection")
        # Only set collection_stage if it doesn't exist yet
        updates = {}
        if not state.get("collection_stage"):
            updates["collection_stage"] = "model"
            updates["current_focus"] = "model_basics"
        return updates
    
    # Get latest user message
    if not messages:
        return {
            "use_case_confirmed": False,
            "awaiting_user_input": True,
        }
    
    latest_message = messages[-1]
    if not isinstance(latest_message, HumanMessage):
        return {"use_case_confirmed": False}
    
    user_query = latest_message.content.lower()
    
    # Check if user is confirming a previously identified use case
    existing_use_case = state.get("use_case")
    if existing_use_case and any(keyword in user_query for keyword in ["yes", "correct", "right", "confirm", "yep", "yeah"]):
        logger.info(f"[USE_CASE] User confirmed use case: {existing_use_case}")
        
        required_params = get_required_params_for_use_case(existing_use_case)
        
        response_message = (
            f"Great! Let's set up your {state.get('use_case_description', 'simulation')}. "
            f"I'll guide you through the required parameters.\n\n"
            f"**Step 1: Basic Simulation Setup**\n\n"
            f"Please provide:\n"
            f"1. **Title**: A name for your simulation (e.g., 'Field Survey with Buried Cylinder')\n"
            f"2. **Survey Length**: How long is your survey line? (in meters, e.g., 5.0)\n"
            f"3. **Max Depth**: How deep should we simulate? (in meters, e.g., 2.0)\n"
            f"4. **Antenna Height**: Height above ground (in meters, typical: 0.02 for ground-coupled)\n\n"
            f"You can provide all at once or one at a time."
        )
        
        return {
            "use_case_confirmed": True,
            "required_params": required_params,
            "collection_stage": "model",
            "current_focus": "model_basics",
            "awaiting_user_input": True,
            "messages": messages + [AIMessage(content=response_message)],
        }
    
    # Check if user is rejecting the use case
    if existing_use_case and any(keyword in user_query for keyword in ["no", "not", "wrong", "incorrect", "different"]):
        logger.info("[USE_CASE] User rejected use case, re-extracting...")
        # Fall through to re-extract
    
    # Extract use case from query
    llm = ChatOpenAI(model=openai_model, api_key=openai_api_key, temperature=0)
    
    extraction_prompt = f"""Analyze this GPR simulation request and extract the use case:

User Request: "{latest_message.content}"

Identify:
1. What type of environment/scenario? (e.g., agricultural field, urban area, road survey, archaeological site)
2. What are they looking for? (e.g., buried pipes, utilities, soil layers, subsurface features)
3. Any special objects or features mentioned? (e.g., metal pipes, tanks, voids, roots)

Provide:
1. use_case_key: A short identifier (e.g., "agriculture_buried_pipe", "urban_utility_survey", "soil_layering_basic")
2. use_case_description: A friendly description (e.g., "agricultural field survey with buried metal pipe detection")
3. has_buried_objects: true/false - whether they mention specific buried objects

Format your response as:
use_case_key: <key>
use_case_description: <description>
has_buried_objects: <true/false>
"""
    
    try:
        response = llm.invoke([HumanMessage(content=extraction_prompt)])
        response_text = response.content.strip()
        
        # Parse response
        lines = response_text.split('\n')
        use_case_key = ""
        use_case_description = ""
        has_buried_objects = False
        
        for line in lines:
            if line.startswith("use_case_key:"):
                use_case_key = line.split(":", 1)[1].strip()
            elif line.startswith("use_case_description:"):
                use_case_description = line.split(":", 1)[1].strip()
            elif line.startswith("has_buried_objects:"):
                has_buried_objects = "true" in line.lower()
        
        if not use_case_key:
            use_case_key = "general_gpr_survey"
            use_case_description = "general GPR simulation"
        
        logger.info(f"[USE_CASE] Extracted use case: {use_case_key}")
        logger.info(f"[USE_CASE] Description: {use_case_description}")
        logger.info(f"[USE_CASE] Has buried objects: {has_buried_objects}")
        
        # Ask for confirmation
        confirmation_message = (
            f"I understand you want to create a simulation for: **{use_case_description}**.\n\n"
            f"Is that correct? (Please reply with 'yes' to confirm or describe what you'd like differently)"
        )
        
        return {
            "use_case": use_case_key,
            "use_case_description": use_case_description,
            "use_case_confirmed": False,
            "awaiting_user_input": True,
            "last_question": "use_case_confirmation",
            "messages": messages + [AIMessage(content=confirmation_message)],
        }
    
    except Exception as e:
        logger.error(f"[USE_CASE] Error extracting use case: {e}", exc_info=True)
        
        # Fallback: assume general simulation
        fallback_message = (
            "I'll help you create a GPR simulation. Let's start by collecting the necessary parameters.\n\n"
            "First, tell me about your simulation setup: What's the title and what are you trying to simulate?"
        )
        
        return {
            "use_case": "general_gpr_survey",
            "use_case_description": "general GPR simulation",
            "use_case_confirmed": True,
            "required_params": get_required_params_for_use_case("general_gpr_survey"),
            "collection_stage": "model",
            "current_focus": "model_basics",
            "messages": messages + [AIMessage(content=fallback_message)],
        }


def route_after_use_case(state: SimulationState) -> str:
    """
    Conditional routing after use case node.
    
    Args:
        state: Current simulation state
    
    Returns:
        Next node name: "parameter_collection" or "__end__"
    """
    if state.get("use_case_confirmed"):
        return "parameter_collection"
    else:
        # Wait for user confirmation
        return "__end__"

