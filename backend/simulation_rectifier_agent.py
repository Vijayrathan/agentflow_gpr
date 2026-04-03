"""
Simulation Rectifier Agent — fixes gprMax simulation failures.

Receives the error diagnosis from the Simulation Error Analyst, fetches
current parameters, determines the minimal fix, and applies it via
patch_parameters (with HITL approval before execution).
"""

import os
import sys
from pathlib import Path

import dotenv
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.parameters_global_state import (
    get_parameters,
    get_all_parameters,
    patch_parameters,
)
from backend.prompt_library import SIMULATION_RECTIFIER_PROMPT

dotenv.load_dotenv()

# -- Agent -------------------------------------------------------------------

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

agent = create_deep_agent(
    model=llm,
    system_prompt=SIMULATION_RECTIFIER_PROMPT,
    checkpointer=InMemorySaver(),
    tools=[get_parameters, get_all_parameters, patch_parameters],
    interrupt_on={"patch_parameters": True},
)
