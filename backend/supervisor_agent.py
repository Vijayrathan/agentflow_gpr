"""
Supervisor pattern architecture for GPR simulation workflow.

This module implements a supervisor agent that coordinates specialized sub-agents:
- sim_setup_agent: Handles parameter validation and gprMax input file generation
- simulation_agent: Handles running gprMax simulations
- rag_agent: Handles RAG queries for geophysics knowledge
- extraction_agent: Handles parameter extraction from user queries
"""

import os
import time
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from init import logger, openai_client, openai_model, openai_api_key
from sim_setup_agent import (
    generate_gprmax_input_file_tool,
    check_input_completeness,
    validate_gpr_parameters,
    get_workspace_directory,
)
import sim_setup_agent
from extraction_agent import extraction_agent
from simulation_agent import run_gprmax_simulation_tool
from rag import GeophysicsRAG

# Global variable to track output filename
_current_output_filename = None

# Initialize RAG system (singleton pattern)
_rag_instance = None

def get_rag_instance():
    """Get or create RAG instance (singleton)"""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = GeophysicsRAG(mode="inference")
    return _rag_instance


# ============================================================================
# SUB-AGENTS
# ============================================================================

def create_sim_setup_agent():
    """Create the simulation setup agent with its specialized tools"""
    llm = ChatOpenAI(model=openai_model, api_key=openai_api_key, temperature=0)
    
    agent = create_agent(
        model=llm,
        tools=[
            generate_gprmax_input_file_tool,
            check_input_completeness,
            validate_gpr_parameters,
        ],
        system_prompt=(
            "You are a simulation setup assistant specialized in gprMax input file generation. "
            "Your responsibilities include:\n"
            "- Validating GPR simulation parameters according to physics constraints\n"
            "- Checking parameter completeness\n"
            "- Generating gprMax input files from validated parameters\n"
            "- Providing clear feedback on validation errors or missing parameters\n\n"
            "Always confirm what was generated in your final response."
        ),
    )
    return agent


def create_simulation_agent():
    """Create the simulation agent with its specialized tools"""
    llm = ChatOpenAI(model=openai_model, api_key=openai_api_key, temperature=0)
    
    agent = create_agent(
        model=llm,
        tools=[run_gprmax_simulation_tool],
        system_prompt=(
            "You are a simulation runner assistant specialized in executing gprMax simulations. "
            "Your responsibilities include:\n"
            "- Running gprMax simulations with provided input file content\n"
            "- The run_gprmax_simulation_tool expects the input file content (text), not a file path\n"
            "- Monitoring simulation progress\n"
            "- Reporting simulation results and any errors\n\n"
            "Always provide clear feedback on simulation status and results."
        ),
    )
    return agent


def create_rag_agent():
    """Create the RAG agent for geophysics knowledge retrieval"""
    llm = ChatOpenAI(model=openai_model, api_key=openai_api_key, temperature=0)
    
    # Create a tool for RAG search
    @tool
    def search_geophysics_knowledge(query: str) -> str:
        """Search geophysics research documents for relevant information.
        
        Use this tool to find information about:
        - Dielectric models (CRIM, Peplinski, Dobson, Mironov)
        - Parameter ranges and constraints
        - Best practices for GPR simulations
        - Physical modeling concepts
        
        Args:
            query: The search query about geophysics or GPR simulation
        
        Returns:
            str: Relevant information from research documents
        """
        try:
            rag = get_rag_instance()
            # Reduce to 2 documents to avoid token limit issues
            results = rag.search(query, top_k=2)
            
            if not results:
                return "No relevant information found in the knowledge base."
            
            # Format results with truncation to prevent token limit issues
            # Limit each document to ~2000 characters to keep total under token limit
            max_chars_per_doc = 2000
            formatted_results = []
            for i, (doc, score) in enumerate(results, 1):
                # Truncate document if too long
                truncated_doc = doc[:max_chars_per_doc] if len(doc) > max_chars_per_doc else doc
                if len(doc) > max_chars_per_doc:
                    truncated_doc += "... [truncated]"
                formatted_results.append(
                    f"[Result {i}, Relevance: {score:.4f}]\n{truncated_doc}\n"
                )
            
            return "\n".join(formatted_results)
        except Exception as e:
            logger.error(f"Error in RAG search: {str(e)}", exc_info=True)
            return f"Error searching knowledge base: {str(e)}"
    
    agent = create_agent(
        model=llm,
        tools=[search_geophysics_knowledge],
        system_prompt=(
            "You are a geophysics knowledge assistant specialized in retrieving information "
            "from research documents. Your responsibilities include:\n"
            "- Searching for relevant information about GPR simulations\n"
            "- Providing information about dielectric models and their constraints\n"
            "- Answering questions about parameter ranges and best practices\n"
            "- Synthesizing information from multiple sources\n\n"
            "Always cite your sources and provide clear, accurate information."
        ),
    )
    return agent


def create_extraction_agent():
    """Create the parameter extraction agent"""
    # The extraction_agent is already a tool, so we can use it directly
    # But we'll wrap it in an agent for consistency with the pattern
    llm = ChatOpenAI(model=openai_model, api_key=openai_api_key, temperature=0)
    
    agent = create_agent(
        model=llm,
        tools=[extraction_agent],
        system_prompt=(
            "You are a parameter extraction assistant specialized in extracting GPR simulation "
            "parameters from natural language queries. Your responsibilities include:\n"
            "- Extracting all mentioned parameters from user queries\n"
            "- Identifying missing parameters\n"
            "- Structuring extracted parameters in a consistent format\n"
            "- Only extracting explicitly mentioned parameters (do not make up values)\n\n"
            "Use the extraction_agent tool to extract parameters from the user's query. "
            "Return structured parameter information that can be used by other agents."
        ),
    )
    return agent


# ============================================================================
# WRAP SUB-AGENTS AS TOOLS FOR SUPERVISOR
# ============================================================================

@tool
async def setup_simulation(request: str, runtime: ToolRuntime) -> str:
    """Setup and generate gprMax input files from parameters.
    
    Use this tool when the user wants to:
    - Generate a gprMax input file
    - Validate simulation parameters
    - Check parameter completeness
    - Create a simulation configuration
    
    Args:
        request: Natural language request about simulation setup
        runtime: Tool runtime context (automatically provided)
    
    Returns:
        str: Result of the simulation setup operation
    """
    logger.info(f"[SUPERVISOR] Routing to sim_setup_agent: {request[:200]}...")
    
    # Get original user message for context
    original_user_message = None
    if hasattr(runtime, 'state') and 'messages' in runtime.state:
        for message in runtime.state['messages']:
            if hasattr(message, 'type') and message.type == "human":
                original_user_message = message
                break
    
    # Build context-aware prompt
    if original_user_message:
        prompt = (
            f"You are assisting with the following user inquiry:\n\n"
            f"{original_user_message.content if hasattr(original_user_message, 'content') else str(original_user_message)}\n\n"
            f"You are tasked with the following sub-request:\n\n"
            f"{request}"
        )
    else:
        prompt = request
    
    agent = create_sim_setup_agent()
    result = await agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
    
    # Extract the final message
    if isinstance(result, dict):
        messages = result.get("messages", [])
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, 'content'):
                return str(last_message.content)
            return str(last_message)
        return str(result)
    
    if hasattr(result, 'messages') and result.messages:
        last_message = result.messages[-1]
        if hasattr(last_message, 'content'):
            return str(last_message.content)
        return str(last_message)
    
    return str(result)


@tool
async def run_simulation(request: str, runtime: ToolRuntime) -> str:
    """Run gprMax simulations with input files.
    
    Use this tool when the user wants to:
    - Execute a simulation
    - Run a gprMax simulation
    - Process an input file
    
    Args:
        request: Natural language request about running a simulation
        runtime: Tool runtime context (automatically provided)
    
    Returns:
        str: Simulation results or status
    """
    logger.info(f"[SUPERVISOR] Routing to simulation_agent: {request[:200]}...")
    
    agent = create_simulation_agent()
    result = await agent.ainvoke({"messages": [HumanMessage(content=request)]})
    
    # Extract the final message
    if isinstance(result, dict):
        messages = result.get("messages", [])
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, 'content'):
                return str(last_message.content)
            return str(last_message)
        return str(result)
    
    if hasattr(result, 'messages') and result.messages:
        last_message = result.messages[-1]
        if hasattr(last_message, 'content'):
            return str(last_message.content)
        return str(last_message)
    
    return str(result)


@tool
async def search_knowledge(request: str, runtime: ToolRuntime) -> str:
    """Search geophysics knowledge base for information.
    
    Use this tool when the user wants to:
    - Learn about dielectric models
    - Understand parameter constraints
    - Get information about GPR simulation concepts
    - Find best practices or research information
    
    Args:
        request: Natural language query about geophysics knowledge
        runtime: Tool runtime context (automatically provided)
    
    Returns:
        str: Relevant information from the knowledge base
    """
    logger.info(f"[SUPERVISOR] Routing to rag_agent: {request[:200]}...")
    
    agent = create_rag_agent()
    result = await agent.ainvoke({"messages": [HumanMessage(content=request)]})
    
    # Extract the final message
    if isinstance(result, dict):
        messages = result.get("messages", [])
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, 'content'):
                return str(last_message.content)
            return str(last_message)
        return str(result)
    
    if hasattr(result, 'messages') and result.messages:
        last_message = result.messages[-1]
        if hasattr(last_message, 'content'):
            return str(last_message.content)
        return str(last_message)
    
    return str(result)


@tool
async def extract_parameters(request: str, runtime: ToolRuntime) -> str:
    """Extract GPR simulation parameters from user queries.
    
    Use this tool when the user provides:
    - Initial simulation requirements
    - Parameter specifications
    - Natural language descriptions of what they want to simulate
    
    Args:
        request: Natural language request containing parameter information
        runtime: Tool runtime context (automatically provided)
    
    Returns:
        str: Extracted parameters in structured format
    """
    logger.info(f"[SUPERVISOR] Routing to extraction_agent: {request[:200]}...")
    
    # The extraction_agent tool can be called directly, but we'll use the agent wrapper
    # for consistency and better context handling
    agent = create_extraction_agent()
    result = await agent.ainvoke({"messages": [HumanMessage(content=request)]})
    
    # Extract the final message
    if isinstance(result, dict):
        messages = result.get("messages", [])
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, 'content'):
                return str(last_message.content)
            return str(last_message)
        return str(result)
    
    if hasattr(result, 'messages') and result.messages:
        last_message = result.messages[-1]
        if hasattr(last_message, 'content'):
            return str(last_message.content)
        return str(last_message)
    
    return str(result)


# ============================================================================
# SUPERVISOR AGENT
# ============================================================================

def create_supervisor_agent():
    """Create the supervisor agent that coordinates all sub-agents"""
    llm = ChatOpenAI(model=openai_model, api_key=openai_api_key, temperature=0)
    
    supervisor = create_agent(
        model=llm,
        tools=[
            setup_simulation,
            run_simulation,
            search_knowledge,
            extract_parameters,
        ],
        system_prompt=(
            "You are a supervisor agent that coordinates specialized sub-agents for GPR simulation workflows.\n\n"
            "Your role is to understand the user's request and route it to the appropriate specialist:\n\n"
            "1. **extract_parameters**: Use when the user provides initial requirements or parameter specifications.\n"
            "   - Extract parameters from natural language queries\n"
            "   - Identify what information is provided and what is missing\n\n"
            "2. **setup_simulation**: Use when you need to validate parameters or generate gprMax input files.\n"
            "   - Validate simulation parameters\n"
            "   - Check parameter completeness\n"
            "   - Generate gprMax input files\n\n"
            "3. **run_simulation**: Use when the user wants to execute a simulation.\n"
            "   - Run gprMax simulations with input files\n"
            "   - Monitor and report simulation results\n\n"
            "4. **search_knowledge**: Use when the user asks questions about geophysics concepts, models, or best practices.\n"
            "   - Answer questions about dielectric models\n"
            "   - Provide information about parameter constraints\n"
            "   - Share research-based knowledge\n\n"
            "You can call multiple agents in sequence if needed. For example:\n"
            "- First extract parameters, then setup simulation, then run simulation\n"
            "- Search knowledge to answer questions, then use that information to guide setup\n\n"
            "Always provide clear, helpful responses to the user based on the results from the sub-agents."
        ),
    )
    
    return supervisor


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def supervisor_agent(initial_input: str, user_id: Optional[str] = None):
    """
    Main entry point for the supervisor agent workflow.
    
    Args:
        initial_input: Initial user query
        user_id: Optional unique user/session ID for file naming
    
    Returns:
        tuple: (agent_result, thought_process, generated_file_path)
    """
    global _current_output_filename
    
    # Set up output filename if user_id is provided
    output_file_path = None
    if user_id:
        workspace_dir = get_workspace_directory()
        generated_files_dir = workspace_dir / "generated_files"
        generated_files_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time() * 1000)
        output_file_path = str(generated_files_dir / f"generated_session_{timestamp}_{user_id}.in")
        _current_output_filename = output_file_path
        sim_setup_agent._current_output_filename = output_file_path
        logger.info(f"[SUPERVISOR] Will save generated file to: {output_file_path}")
    else:
        workspace_dir = get_workspace_directory()
        generated_files_dir = workspace_dir / "generated_files"
        generated_files_dir.mkdir(parents=True, exist_ok=True)
        output_file_path = str(generated_files_dir / "generated.in")
        _current_output_filename = output_file_path
        sim_setup_agent._current_output_filename = output_file_path
    
    # Create supervisor agent
    supervisor = create_supervisor_agent()
    
    # Track thought process steps
    thought_process = []
    
    try:
        # Invoke the supervisor with the input
        result = await supervisor.ainvoke({"messages": [HumanMessage(content=initial_input)]})
        
        # Extract messages from result for thought process
        messages = []
        if isinstance(result, dict):
            messages = result.get("messages", result.get("output", []))
        elif hasattr(result, "messages"):
            messages = result.messages
        elif hasattr(result, "output"):
            messages = [result.output] if result.output else []
        
        # Ensure messages is a list
        if not isinstance(messages, list):
            messages = [messages] if messages else []
        
        # Process messages for thought process
        for msg in messages:
            msg_type = type(msg).__name__
            
            # Extract message content
            if hasattr(msg, 'content') and msg.content:
                content = str(msg.content)
                role = getattr(msg, 'role', 'unknown')
                if not role or role == 'unknown':
                    if 'Human' in msg_type or 'user' in msg_type.lower():
                        role = 'user'
                    elif 'AI' in msg_type or 'assistant' in msg_type.lower() or 'AIMessage' in msg_type:
                        role = 'assistant'
                    else:
                        role = 'assistant'
                
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
                    tool_name = tool_call.get('name', 'unknown') if isinstance(tool_call, dict) else getattr(tool_call, 'name', 'unknown')
                    tool_args = tool_call.get('args', {}) if isinstance(tool_call, dict) else getattr(tool_call, 'args', {})
                    
                    step = {
                        'type': 'tool_call',
                        'tool_name': str(tool_name),
                        'args': tool_args if isinstance(tool_args, dict) else {}
                    }
                    thought_process.append(step)
        
        # Extract the final output
        agent_result = type('AgentResult', (), {
            'output': messages[-1].content if messages and hasattr(messages[-1], 'content') else str(result),
            'messages': messages
        })()
        
        # Check if file was actually generated
        final_file_path = None
        if output_file_path and os.path.exists(output_file_path):
            final_file_path = output_file_path
            logger.info(f"[SUPERVISOR] File successfully generated at: {final_file_path}")
        else:
            workspace_dir = get_workspace_directory()
            default_file = workspace_dir / "generated_files" / "generated.in"
            if os.path.exists(str(default_file)):
                final_file_path = str(default_file)
                logger.info(f"[SUPERVISOR] File found at default location: {final_file_path}")
        
        # Reset global filename
        _current_output_filename = None
        sim_setup_agent._current_output_filename = None
        
        return agent_result, thought_process, final_file_path
        
    except Exception as e:
        logger.error(f"[SUPERVISOR] Error during workflow execution: {str(e)}", exc_info=True)
        _current_output_filename = None
        sim_setup_agent._current_output_filename = None
        
        # Return error result
        error_result = type('AgentResult', (), {
            'output': f"Error: {str(e)}",
            'messages': []
        })()
        return error_result, thought_process, None

