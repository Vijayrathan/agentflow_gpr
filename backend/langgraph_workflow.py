"""
LangGraph Workflow - Assembles all nodes into a stateful graph.

This module wires together all the nodes (mode_router, use_case, parameter_collection,
validator, resolver, generator, rag) into a complete LangGraph workflow with persistence.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pathlib import Path

from langgraph_state import SimulationState
from nodes.mode_router import mode_router_node, route_after_mode_router
from nodes.use_case_node import use_case_node, route_after_use_case
from nodes.parameter_collection import parameter_collection_node, route_after_parameter_collection
from nodes.validator import validator_node, route_after_validation
from nodes.resolver import resolver_node
from nodes.generator import generator_node
from nodes.rag_node import rag_node
from init import logger


def create_workflow() -> StateGraph:
    """
    Create and assemble the complete LangGraph workflow.
    
    Returns:
        Compiled StateGraph with persistence
    """
    logger.info("[WORKFLOW] Building LangGraph workflow...")
    
    # Create the graph
    workflow = StateGraph(SimulationState)
    
    # Add all nodes
    workflow.add_node("mode_router", mode_router_node)
    workflow.add_node("use_case", use_case_node)
    workflow.add_node("parameter_collection", parameter_collection_node)
    workflow.add_node("validator", validator_node)
    workflow.add_node("resolver", resolver_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("rag", rag_node)
    
    logger.info("[WORKFLOW] Added all nodes")
    
    # Set entry point
    workflow.set_entry_point("mode_router")
    
    # Define edges
    
    # 1. Mode router -> use_case, parameter_collection, or rag
    workflow.add_conditional_edges(
        "mode_router",
        route_after_mode_router,
        {
            "use_case": "use_case",
            "parameter_collection": "parameter_collection",
            "rag": "rag"
        }
    )
    
    # 2. Use case -> parameter_collection or END (wait for confirmation)
    workflow.add_conditional_edges(
        "use_case",
        route_after_use_case,
        {
            "parameter_collection": "parameter_collection",
            "__end__": END
        }
    )
    
    # 3. Parameter collection -> validator or END (validate only if parameters were collected)
    workflow.add_conditional_edges(
        "parameter_collection",
        route_after_parameter_collection,
        {
            "validator": "validator",
            "__end__": END
        }
    )
    
    # 4. Validator -> parameter_collection, resolver, or END
    workflow.add_conditional_edges(
        "validator",
        route_after_validation,
        {
            "parameter_collection": "parameter_collection",
            "resolver": "resolver",
            "__end__": END
        }
    )
    
    # 5. Resolver -> generator (always generate after resolution)
    workflow.add_edge("resolver", "generator")
    
    # 6. Generator -> END (workflow complete)
    workflow.add_edge("generator", END)
    
    # 7. RAG -> END (return answer and wait for next query)
    workflow.add_edge("rag", END)
    
    logger.info("[WORKFLOW] Defined all edges")
    
    return workflow


def compile_workflow() -> StateGraph:
    """
    Compile the workflow with persistence.
    
    Returns:
        Compiled workflow with MemorySaver checkpointer
    """
    logger.info("[WORKFLOW] Compiling workflow with persistence...")
    
    workflow = create_workflow()
    
    # Use MemorySaver for in-memory persistence
    # This is simpler and works synchronously without needing async initialization
    memory = MemorySaver()
    
    # Compile with checkpointer
    app = workflow.compile(checkpointer=memory)
    
    logger.info("[WORKFLOW] Workflow compiled successfully with MemorySaver!")
    
    return app


# Create the compiled workflow (singleton)
langgraph_app = compile_workflow()


# Export for use in Flask app
__all__ = ["langgraph_app"]

