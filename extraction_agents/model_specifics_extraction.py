"""
Model/Domain Parameter Extraction using DeepAgents.

A single agent interactively collects simulation model and domain parameters
from the user, validates against the ExtractedModelConfig schema, and writes
the result to workspace/model_config.json.
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
from rag import rag_search
from prompt_library import RAG_SUBAGENT_PROMPT, MODEL_AGENT_PROMPT, MODEL_VALIDATION_PROMPT
from parameters_global_state import post_parameters, get_parameters, patch_parameters
from validation_tools import (
    validate_model, validate_temperature, validate_time_window,
    validate_mesh, validate_essential_params, validate_cfl,
)

dotenv.load_dotenv()

# Initialize the model
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

rag_subagent = {
    "name": "knowledge-agent",
    "description": (
        "Geophysics knowledge expert. Answers questions about dielectric "
        "models (Peplinski, Dobson, Mironov, CRIM), simulation domains, "
        "mesh resolution, cells per wavelength, survey design, and related "
        "GPR topics. Searches the knowledge base first; falls back to "
        "domain expertise if needed."
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
        "Validates model and domain parameters. Checks dielectric model name and "
        "frequency band, mesh resolution (Nyquist criterion), CFL stability, "
        "time window sufficiency, temperature range, and essential parameter presence. "
        "Range-based checks (texture, moisture) are validated at sampling time. "
        "Call after collecting parameters, before storing."
    ),
    "system_prompt": MODEL_VALIDATION_PROMPT,
    "tools": [
        validate_model, validate_temperature, validate_time_window,
        validate_mesh, validate_essential_params, validate_cfl, get_parameters,
    ],
}

# ---------------------------------------------------------------------------
# Build & Run
# ---------------------------------------------------------------------------

agent = create_deep_agent(
    model=llm,
    subagents=[rag_subagent, validation_subagent],
    system_prompt=MODEL_AGENT_PROMPT,
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
    print("Starting model & domain extraction agent...\n")
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "I need to configure the simulation model and domain "
                        "parameters for a gprMax simulation. Please begin the "
                        "model/domain parameter extraction process."
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
