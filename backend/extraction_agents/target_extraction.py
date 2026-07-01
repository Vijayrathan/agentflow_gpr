"""
Buried-target range extraction (early mini-stage).

Runs right after the layer stage and BEFORE the per-sample draw. It collects the
sampling RANGES for a buried cylinder target whose geometry varies per sample
(x-position, depth, radius) and validates against the ExtractedTargetRanges /
CylinderTargetRange schema, then posts the `target_ranges` section.

Cylinders only for now (box/sphere are future). All grid-dependent placement
checks are DERIVED/validated downstream once the global grid exists — nothing
about the domain is collected here.
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
from backend.prompt_library import (
    RAG_SUBAGENT_PROMPT,
    TARGET_AGENT_PROMPT,
)
from backend.parameters_global_state import post_parameters, get_parameters, patch_parameters

dotenv.load_dotenv()

# Initialize the model
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

rag_subagent = {
    "name": "knowledge-agent",
    "description": (
        "Geophysics knowledge expert. Answers questions about buried targets, "
        "GPR survey geometry, realistic target depths/radii, and related topics. "
        "Searches the knowledge base first; falls back to domain expertise if needed."
    ),
    "system_prompt": RAG_SUBAGENT_PROMPT,
    "tools": [rag_search],
}

# ---------------------------------------------------------------------------
# Build & Run
#   No validation subagent: CylinderTargetRange constraints (each min <= max,
#   radius_min_m > 0) are schema-enforced at POST time; grid-dependent placement
#   is validated downstream once the global grid is derived (target_placement).
# ---------------------------------------------------------------------------

agent = create_deep_agent(
    model=llm,
    subagents=[rag_subagent],
    system_prompt=TARGET_AGENT_PROMPT,
    checkpointer=InMemorySaver(),
    tools=[post_parameters, get_parameters, patch_parameters],
)


def _print_response(result: dict) -> None:
    """Print all new assistant messages and log tool calls for debugging."""
    for msg in result.get("messages", []):
        kind = type(msg).__name__
        if kind == "AIMessage" and msg.content:
            print(f"\n[Master Agent]: {msg.content}\n")
        elif kind == "ToolMessage":
            print(f"  [tool:{msg.name}] returned ({len(msg.content)} chars)")


if __name__ == "__main__":
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print("Starting buried-target range agent...\n")
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "I need to configure the buried-target geometry ranges for "
                        "a gprMax simulation batch. Please begin the target range "
                        "extraction process."
                    )
                )
            ]
        },
        config=config,
    )

    _print_response(result)

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
