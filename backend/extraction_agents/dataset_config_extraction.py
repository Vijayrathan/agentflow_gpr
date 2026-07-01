"""
Dataset / run-orchestration parameter extraction using DeepAgents.

A single agent interactively collects the dataset-level configuration
(num_samples, naming, FDTD grid / boundary policy, resolution policy) and
validates against the DatasetConfig schema. Physics/domain/mesh quantities are
DERIVED downstream and are NOT collected here.
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
    DATASET_CONFIG_AGENT_PROMPT,
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
        "Geophysics knowledge expert. Answers questions about FDTD grids, "
        "cells per wavelength, PML boundaries, fractal soil material counts, "
        "and dataset/batch generation. Searches the knowledge base first; "
        "falls back to domain expertise if needed."
    ),
    "system_prompt": RAG_SUBAGENT_PROMPT,
    "tools": [rag_search],
}

# ---------------------------------------------------------------------------
# Build & Run
#   No validation subagent: DatasetConfig constraints are enforced by the
#   Pydantic schema at POST time; all physics/grid checks run downstream in the
#   deterministic stages (validation_tools_new / global_validation).
# ---------------------------------------------------------------------------

agent = create_deep_agent(
    model=llm,
    subagents=[rag_subagent],
    system_prompt=DATASET_CONFIG_AGENT_PROMPT,
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
            print(f"  [tool:{msg.name}] returned ({len(msg.content)} chars)")


if __name__ == "__main__":
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print("Starting dataset configuration agent...\n")
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "I need to configure the dataset/run parameters for a "
                        "gprMax simulation batch. Please begin the dataset "
                        "configuration process."
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
