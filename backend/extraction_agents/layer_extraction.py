"""
Layer Parameter Extraction using DeepAgents.

A single agent interactively collects soil layer parameters from the user,
validates against the ExtractedLayerParams schema, and writes the result
to workspace/layers.json.
"""
import os
import uuid
import dotenv
from deepagents import create_deep_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from backend.rag import rag_search
from backend.prompt_library import RAG_SUBAGENT_PROMPT, LAYER_AGENT_PROMPT, LAYER_VALIDATION_PROMPT
from backend.validation_tools import validate_layer
from backend.parameters_global_state import post_parameters, get_parameters, patch_parameters
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
        "Geophysics knowledge expert. Answers questions about soil properties, "
        "dielectric models (Peplinski, Topp, etc.), GPR parameters, clay/sand/silt "
        "characteristics, bulk density, volumetric water content, and related topics. "
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
        "Validates non-range soil layer parameters (organic_fraction, "
        "porewater_sigma_Sm). Range-based parameters like texture, theta_v, "
        "and densities are validated at dataset sampling time. "
        "Call after collecting layer parameters, before storing."
    ),
    "system_prompt": LAYER_VALIDATION_PROMPT,
    "tools": [validate_layer, get_parameters],
}

# ---------------------------------------------------------------------------
# Layer Agent
# ---------------------------------------------------------------------------

agent = create_deep_agent(
    model=llm,
    subagents=[rag_subagent, validation_subagent],
    system_prompt=LAYER_AGENT_PROMPT,
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
    print("Starting layer extraction agent...\n")
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "I need to set up the soil layers for a gprMax "
                        "simulation. Please begin the layer parameter "
                        "extraction process."
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
