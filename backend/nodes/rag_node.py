"""
RAG Node - Handles knowledge queries using GeophysicsRAG.

This node:
1. Takes user's question
2. Searches the geophysics knowledge base
3. Returns relevant information
"""

from typing import Dict, Any
from langchain_core.messages import AIMessage, HumanMessage

from langgraph_state import SimulationState
from rag import GeophysicsRAG
from init import logger


# Singleton RAG instance
_rag_instance = None


def get_rag_instance() -> GeophysicsRAG:
    """Get or create RAG instance (singleton)."""
    global _rag_instance
    if _rag_instance is None:
        logger.info("[RAG] Initializing GeophysicsRAG instance...")
        _rag_instance = GeophysicsRAG(mode="inference")
    return _rag_instance


def rag_node(state: SimulationState) -> Dict[str, Any]:
    """
    Search geophysics knowledge base and answer questions.
    
    Args:
        state: Current simulation state
    
    Returns:
        Updated state with RAG response
    """
    logger.info("[RAG] Processing knowledge query...")
    
    messages = state.get("messages", [])
    
    if not messages:
        return {}
    
    latest_message = messages[-1]
    if not isinstance(latest_message, HumanMessage):
        return {}
    
    user_query = latest_message.content
    logger.info(f"[RAG] Query: {user_query[:150]}...")
    
    try:
        # Get RAG instance
        rag = get_rag_instance()
        
        # Search knowledge base
        results = rag.search(user_query, top_k=3)
        
        if not results:
            response_message = (
                "I couldn't find specific information about that in the knowledge base. "
                "However, I can help you create a GPR simulation. "
                "Would you like to start setting up a simulation instead?"
            )
            
            return {
                "messages": messages + [AIMessage(content=response_message)],
            }
        
        # Format response
        response_parts = ["Here's what I found:\n"]
        
        for i, (doc, score) in enumerate(results, 1):
            # Truncate very long documents
            doc_preview = doc[:500] if len(doc) > 500 else doc
            if len(doc) > 500:
                doc_preview += "..."
            
            response_parts.append(f"\n**Source {i}** (Relevance: {score:.2f}):\n{doc_preview}\n")
        
        response_parts.append(
            "\n---\n"
            "Would you like to know more about something specific, or would you like to create a simulation?"
        )
        
        response_message = "\n".join(response_parts)
        
        logger.info(f"[RAG] Found {len(results)} result(s)")
        
        return {
            "messages": messages + [AIMessage(content=response_message)],
        }
    
    except Exception as e:
        logger.error(f"[RAG] Error searching knowledge base: {e}", exc_info=True)
        
        error_message = (
            f"I encountered an error while searching the knowledge base: {str(e)}\n\n"
            "Would you like to try rephrasing your question, or shall we proceed with creating a simulation?"
        )
        
        return {
            "messages": messages + [AIMessage(content=error_message)],
        }

