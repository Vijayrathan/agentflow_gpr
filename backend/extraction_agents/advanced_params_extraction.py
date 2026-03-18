"""
Advanced Parameters Extraction using DeepAgents.

A single agent interactively collects advanced/optional simulation parameters
from the user (geometry objects, surface roughness, snapshots, receiver arrays,
PML, threading, output directory), validates against the ExtractedAdvancedParams
schema, and writes the result to workspace/advanced_params.json.
"""

import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import dotenv
from deepagents import create_deep_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from backend.rag import rag_search
from backend.prompt_library import ADVANCED_AGENT_PROMPT, RAG_SUBAGENT_PROMPT, ADVANCED_VALIDATION_PROMPT
from backend.parameters_global_state import post_parameters, get_parameters, patch_parameters
from backend.validation_tools import (
    validate_surface, validate_sphere, validate_snapshot,
    validate_box, validate_cylinder, validate_rxarray,
    validate_custom_material, validate_material_references,
    validate_simulation_metadata,
)


dotenv.load_dotenv()


# Initialize the model
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

# ---------------------------------------------------------------------------
# RAG Sub-Agent
# ---------------------------------------------------------------------------

rag_subagent = {
    "name": "knowledge-agent",
    "description": (
        "Geophysics knowledge expert. Answers questions about PML absorbing "
        "boundaries, surface roughness, fractal dimension, snapshot outputs, "
        "receiver arrays, buried object geometries (cylinders, boxes, spheres), "
        "custom materials, and related GPR simulation topics. "
        "Searches the knowledge base first; falls back to domain expertise if needed."
    ),
    "system_prompt": RAG_SUBAGENT_PROMPT,
    "tools": [rag_search],
}

# ---------------------------------------------------------------------------
# Validation Sub-Agent
# ---------------------------------------------------------------------------

validation_subagent = {
    "name": "validation-agent",
    "description": (
        "Validates advanced simulation parameters. Checks geometry objects "
        "(cylinders, boxes, spheres) for valid dimensions and domain bounds, "
        "surface roughness, receiver arrays, snapshots, custom materials, "
        "material references, and simulation metadata. Call after collecting "
        "parameters, before storing."
    ),
    "system_prompt": ADVANCED_VALIDATION_PROMPT,
    "tools": [
        validate_surface, validate_cylinder, validate_box, validate_sphere,
        validate_rxarray, validate_snapshot, validate_custom_material,
        validate_material_references, validate_simulation_metadata,
        get_parameters,
    ],
}

# ---------------------------------------------------------------------------
# Build & Run
# ---------------------------------------------------------------------------

agent = create_deep_agent(
    model=llm,
    subagents=[rag_subagent, validation_subagent],
    system_prompt=ADVANCED_AGENT_PROMPT,
    checkpointer=InMemorySaver(),
    tools=[post_parameters, get_parameters, patch_parameters]
)

def _print_response(result: dict) -> None:
    """Print all new assistant messages and log tool calls for debugging."""
    for msg in result.get("messages", []):
        kind = type(msg).__name__
        if kind == "AIMessage" and msg.content:
            print(f"\n[Master Agent]: {msg.content}\n")
        elif kind == "ToolMessage":
            # Tool results — show which tool returned
            print(f"  [tool:{msg.name}] returned ({len(msg.content)} chars)")


if __name__ == "__main__":
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # Kick off with the initial request
    print("Starting advanced parameters extraction agent...\n")
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "I need to configure the advanced/optional parameters "
                        "for a gprMax simulation. Please begin the advanced "
                        "parameters extraction process."
                    )
                )
            ]
        },
        config=config,
    )

    _print_response(result)

    # Interactive loop: feed user replies back until the agent is done
    while True:
        user_input = input("You: ").strip()
        if not user_input or user_input.lower() in ("quit", "exit"):
            print("Exiting.")
            break

        result = agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )

        _print_response(result)
