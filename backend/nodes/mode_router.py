"""
Mode Router Node - Determines if user wants simulation or RAG query.

This node analyzes the user's message to route to either:
- Simulation mode: User wants to create/modify a simulation
- RAG mode: User is asking a question about GPR/geophysics
"""

from typing import Dict, Any
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from langgraph_state import SimulationState
from init import openai_api_key, openai_model, logger


def mode_router_node(state: SimulationState) -> Dict[str, Any]:
    """
    Route user query to simulation or RAG mode.
    
    Args:
        state: Current simulation state
    
    Returns:
        Updated state with mode set
    """
    logger.info("[MODE_ROUTER] Analyzing user intent...")
    
    # Get the latest user message
    messages = state.get("messages", [])
    if not messages:
        logger.warning("[MODE_ROUTER] No messages in state")
        return {"mode": "simulation", "messages": messages}
    
    latest_message = messages[-1]
    if not isinstance(latest_message, HumanMessage):
        # If last message is not from user, default to simulation
        logger.info("[MODE_ROUTER] Latest message not from user, defaulting to simulation mode")
        return {"mode": "simulation"}
    
    user_query = latest_message.content
    logger.info(f"[MODE_ROUTER] Analyzing query: {user_query[:100]}...")
    
    # Quick pattern matching for obvious simulation requests
    query_lower = user_query.lower().strip()
    simulation_patterns = [
        "i want to simulate",
        "i want to create a simulation",
        "i need to simulate",
        "simulate a",
        "simulate an",
        "create a simulation",
        "setup a simulation",
        "generate a simulation",
        "build a simulation",
        "make a simulation",
        "run a simulation",
    ]
    
    # If query starts with any simulation pattern, immediately classify as simulation
    for pattern in simulation_patterns:
        if query_lower.startswith(pattern):
            logger.info(f"[MODE_ROUTER] Matched simulation pattern: '{pattern}', classifying as simulation")
            return {"mode": "simulation"}
    
    # Use LLM to classify intent for ambiguous cases
    llm = ChatOpenAI(model=openai_model, api_key=openai_api_key, temperature=0)
    
    classification_prompt = f"""Analyze the user's query and determine their intent:

User Query: "{user_query}"

Classify as either:
- "simulation": User wants to create, modify, or generate a GPR simulation. This includes ANY request that starts with "I want to simulate", "simulate", "create a simulation", "setup", "configure", "build", "make", "design", "generate", or mentions creating/running a simulation scenario.
- "rag": User is asking a QUESTION about geophysics concepts, models, or best practices. This is ONLY for informational questions like "what is", "how does", "why does", "explain", "tell me about", "describe", "which model should I use".

IMPORTANT:
- "I want to simulate" = simulation (NOT rag)
- "I want to simulate X" = simulation (NOT rag)
- "simulate X" = simulation (NOT rag)
- "What is X?" = rag
- "How does X work?" = rag
- "Explain X" = rag

If the user is confirming something (yes, no, correct, that's right), classify as "simulation".

If unclear, default to "simulation".

Respond with ONLY ONE WORD: either "simulation" or "rag"
"""
    
    try:
        response = llm.invoke([HumanMessage(content=classification_prompt)])
        mode = response.content.strip().lower()
        
        # Validate response
        if mode not in ["simulation", "rag"]:
            logger.warning(f"[MODE_ROUTER] Invalid mode from LLM: {mode}, defaulting to simulation")
            mode = "simulation"
        
        logger.info(f"[MODE_ROUTER] Classified as: {mode}")
        
        return {
            "mode": mode,
        }
    
    except Exception as e:
        logger.error(f"[MODE_ROUTER] Error classifying intent: {e}", exc_info=True)
        # Default to simulation on error
        return {
            "mode": "simulation",
        }


def route_after_mode_router(state: SimulationState) -> str:
    """
    Conditional routing function after mode_router node.
    
    Args:
        state: Current simulation state
    
    Returns:
        Next node name: "use_case", "parameter_collection", or "rag"
    """
    mode = state.get("mode", "simulation")
    
    if mode == "rag":
        return "rag"
    
    # If we're already in parameter collection (use case confirmed and collection stage set),
    # skip use_case node and go directly to parameter_collection
    use_case_confirmed = state.get("use_case_confirmed", False)
    collection_stage = state.get("collection_stage")
    
    if use_case_confirmed and collection_stage:
        logger.info(f"[MODE_ROUTER] Already in collection stage '{collection_stage}', routing to parameter_collection")
        return "parameter_collection"
    
    # Otherwise, go through use_case identification
    return "use_case"

