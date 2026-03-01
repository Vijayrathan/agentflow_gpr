
import json
import os
import uuid
import dotenv
from deepagents import create_deep_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from prompt_library import ANTENNA_RAG_SUBAGENT_PROMPT,ANTENNA_AGENT_PROMPT
from langchain_openai import ChatOpenAI
from rag import rag_search

dotenv.load_dotenv()

# Initialize the model
llm = ChatOpenAI(
    model="gpt-4.1",
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
    "system_prompt": ANTENNA_RAG_SUBAGENT_PROMPT,
    "tools": [rag_search],
}


# ---------------------------------------------------------------------------
# Build & Run
# ---------------------------------------------------------------------------

agent = create_deep_agent(
    model=llm,
    subagents=[rag_subagent],
    system_prompt=ANTENNA_AGENT_PROMPT,
    checkpointer=InMemorySaver(),
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
    print("Starting antenna & waveform extraction agent...\n")
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "I need to configure the antenna and waveform for a "
                        "gprMax simulation. Please begin the antenna/waveform "
                        "parameter extraction process."
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
