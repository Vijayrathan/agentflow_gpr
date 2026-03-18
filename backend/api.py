"""
WebSocket API server for the GPR multi-agent pipeline.

Bridges the React frontend to the existing LangGraph agents via WebSocket,
replacing the terminal-based interaction in agentflow.py.
"""

import asyncio
import json
import logging
import sys
import uuid
from pathlib import Path
from dataclasses import dataclass, field

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from langchain_core.messages import HumanMessage

# Ensure project root is on sys.path so `backend.*` imports (used by agents) work,
# and add dataset_sampling for its bare imports (resolvers, dataset_generator).
_project_root = str(Path(__file__).resolve().parent.parent)
_ds_dir = str(Path(__file__).resolve().parent / "dataset_sampling")
for _p in [_project_root, _ds_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from parameters_global_state import start_parameter_server, BASE_URL
from extraction_agents.layer_extraction import agent as layer_agent
from extraction_agents.antenna_extraction import agent as antenna_agent
from extraction_agents.model_specifics_extraction import agent as model_agent
from extraction_agents.advanced_params_extraction import agent as advanced_agent
from dataset_sampling.dataset_generation_agent import agent as dataset_agent

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipeline definition (mirrors agentflow.py)
# ---------------------------------------------------------------------------

STAGES = [
    {
        "agent": layer_agent,
        "name": "Layer Extraction",
        "init_message": (
            "I need to set up the soil layers for a gprMax simulation. "
            "Please begin the layer parameter extraction process."
        ),
    },
    {
        "agent": antenna_agent,
        "name": "Antenna & Waveform",
        "init_message": (
            "I need to configure the antenna and waveform for a gprMax simulation. "
            "Please begin the antenna/waveform parameter extraction process."
        ),
    },
    {
        "agent": model_agent,
        "name": "Model & Domain",
        "init_message": (
            "I need to configure the simulation model and domain parameters for a "
            "gprMax simulation. Please begin the model/domain parameter extraction process."
        ),
    },
    {
        "agent": advanced_agent,
        "name": "Advanced Parameters",
        "init_message": (
            "I need to configure the advanced/optional parameters for a gprMax "
            "simulation. Please begin the advanced parameters extraction process."
        ),
    },
    {
        "agent": dataset_agent,
        "name": "Dataset Generation",
        "init_message": (
            "All parameter extractions are complete. Begin dataset generation."
        ),
    },
]

STAGE_NAMES = [s["name"] for s in STAGES]

# ---------------------------------------------------------------------------
# Per-session state
# ---------------------------------------------------------------------------


@dataclass
class SessionState:
    stage_index: int = 0
    configs: list = field(default_factory=list)
    dataset_name: str | None = None
    seen_counts: list = field(default_factory=list)

    def __post_init__(self):
        if not self.configs:
            self.configs = [
                {"configurable": {"thread_id": str(uuid.uuid4())}}
                for _ in STAGES
            ]
        if not self.seen_counts:
            self.seen_counts = [0] * len(STAGES)


sessions: dict[str, SessionState] = {}

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="GPR Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    start_parameter_server()
    logger.info("Parameter state server started on %s", BASE_URL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_responses(result: dict, seen: int = 0):
    """Extract new AI messages and detect tool calls from agent result."""
    messages = result.get("messages", [])
    ai_texts = []
    posted = False
    dataset_info = None

    for msg in messages[seen:]:
        kind = type(msg).__name__
        if kind == "AIMessage" and msg.content:
            ai_texts.append(msg.content)
        elif kind == "ToolMessage":
            if msg.name == "post_parameters":
                posted = True
            elif msg.name == "run_dataset_generation":
                try:
                    data = json.loads(msg.content)
                    if data.get("status") in ("complete", "partial"):
                        dataset_info = data
                except (json.JSONDecodeError, TypeError):
                    pass

    return ai_texts, posted, dataset_info, len(messages)


async def _invoke_agent(agent, input_messages, config):
    """Run agent.invoke in a thread to avoid blocking the event loop."""
    return await asyncio.to_thread(agent.invoke, input_messages, config)


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(ws: WebSocket, session_id: str):
    await ws.accept()

    state = sessions.setdefault(session_id, SessionState())

    # Initialize session on param server
    try:
        httpx.post(
            f"{BASE_URL}/session",
            json={"user_id": session_id, "session_id": session_id},
            timeout=5,
        )
    except Exception:
        pass

    # Send current stage info
    await ws.send_json({
        "type": "stage_change",
        "stage_index": state.stage_index,
        "stage_name": STAGE_NAMES[state.stage_index],
    })

    # Auto-send init message for the current stage if it hasn't been started
    if state.seen_counts[state.stage_index] == 0:
        await _run_agent_turn(ws, state, init=True)

    # Listen for user messages
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            if msg.get("type") == "user_message":
                await _run_agent_turn(ws, state, user_text=msg["content"])
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for session %s", session_id)
    except Exception as e:
        logger.exception("WebSocket error for session %s", session_id)
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


async def _run_agent_turn(
    ws: WebSocket, state: SessionState, user_text: str | None = None, init: bool = False
):
    """Run one turn of the current agent and handle stage transitions."""
    stage = STAGES[state.stage_index]
    agent = stage["agent"]
    config = state.configs[state.stage_index]

    if init:
        content = stage["init_message"]
    else:
        content = user_text
        # Dismiss the download card when user continues chatting
        if state.dataset_name is not None:
            state.dataset_name = None
            await ws.send_json({"type": "dataset_dismiss"})

    try:
        result = await _invoke_agent(
            agent,
            {"messages": [HumanMessage(content=content)]},
            config,
        )
    except Exception as e:
        await ws.send_json({"type": "error", "message": f"Agent error: {e}"})
        return

    ai_texts, posted, dataset_info, new_seen = _extract_responses(
        result, state.seen_counts[state.stage_index]
    )
    state.seen_counts[state.stage_index] = new_seen

    # Send all AI messages to the client
    for text in ai_texts:
        await ws.send_json({"type": "agent_message", "content": text})

    # Handle dataset generation completion
    if dataset_info and state.stage_index == len(STAGES) - 1:
        # Try to find the dataset name from the result
        dataset_name = _find_dataset_name(result)
        if dataset_name:
            state.dataset_name = dataset_name
            await ws.send_json({
                "type": "dataset_ready",
                "dataset_name": dataset_name,
                "num_generated": dataset_info.get("num_generated", 0),
            })

    # Handle stage transition
    if posted and state.stage_index < len(STAGES) - 1:
        state.stage_index += 1
        await ws.send_json({
            "type": "stage_change",
            "stage_index": state.stage_index,
            "stage_name": STAGE_NAMES[state.stage_index],
        })
        # Auto-start the next agent
        await _run_agent_turn(ws, state, init=True)


def _find_dataset_name(result: dict) -> str | None:
    """Extract the dataset_name from the most recent run_dataset_generation tool call."""
    name = None
    for msg in result.get("messages", []):
        kind = type(msg).__name__
        if kind == "AIMessage" and hasattr(msg, "tool_calls"):
            for tc in msg.tool_calls:
                if tc.get("name") == "run_dataset_generation":
                    args = tc.get("args", {})
                    if args.get("dataset_name"):
                        name = args["dataset_name"]
    return name


# ---------------------------------------------------------------------------
# File download endpoints
# ---------------------------------------------------------------------------

# Datasets are generated relative to CWD (backend/), so resolve from there.
DATASETS_DIR = Path(__file__).resolve().parent / "datasets"


@app.get("/download/manifest/{dataset_name}")
async def download_manifest(dataset_name: str):
    path = DATASETS_DIR / dataset_name / "manifest.csv"
    if not path.exists():
        return {"error": "Manifest not found", "looked_at": str(path)}
    return FileResponse(
        path,
        media_type="text/csv",
        filename=f"{dataset_name}_manifest.csv",
    )


@app.get("/download/sample/{dataset_name}")
async def download_sample(dataset_name: str):
    sample_dir = DATASETS_DIR / dataset_name / "files"
    if not sample_dir.exists():
        return {"error": "Sample directory not found", "looked_at": str(sample_dir)}
    # Return the first .in file
    in_files = sorted(sample_dir.glob("*.in"))
    if not in_files:
        return {"error": "No sample files found"}
    return FileResponse(
        in_files[0],
        media_type="text/plain",
        filename=in_files[0].name,
    )
