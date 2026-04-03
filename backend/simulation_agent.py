"""
Simulation Error Agent — analyses gprMax simulation failures.

When run_batch_simulation encounters an error for an individual .in file,
this agent receives the error traceback and .in file content, inspects the
stored extraction parameters, and returns a diagnosis with suggested fixes.
"""

import os
import sys
import uuid
from pathlib import Path

import dotenv
from deepagents import create_deep_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.parameters_global_state import get_parameters, get_all_parameters
from backend.prompt_library import SIMULATION_AGENT_PROMPT

dotenv.load_dotenv()

# ── Agent ─────────────────────────────────────────────────────────────

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

agent = create_deep_agent(
    model=llm,
    system_prompt=SIMULATION_AGENT_PROMPT,
    checkpointer=InMemorySaver(),
    tools=[get_parameters, get_all_parameters],
)