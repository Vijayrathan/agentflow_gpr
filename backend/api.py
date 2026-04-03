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

# Configure root logger so all modules' log output reaches the terminal
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from langchain_core.messages import HumanMessage

# Ensure project root is on sys.path so `backend.*` imports (used by agents) work,
# and add dataset_sampling for its bare imports (resolvers, dataset_generator).
_project_root = str(Path(__file__).resolve().parent.parent)
_ds_dir = str(Path(__file__).resolve().parent / "dataset_sampling")
_gprmax_root = str(Path(__file__).resolve().parent.parent / "gprMax")
for _p in [_project_root, _ds_dir, _gprmax_root]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from parameters_global_state import start_parameter_server, BASE_URL
from extraction_agents.layer_extraction import agent as layer_agent
from extraction_agents.antenna_extraction import agent as antenna_agent
from extraction_agents.model_specifics_extraction import agent as model_agent
from extraction_agents.advanced_params_extraction import agent as advanced_agent
from dataset_sampling import dataset_generation_agent as _ds_gen_mod
dataset_agent = _ds_gen_mod.agent

import httpx

from simulate import run_batch_simulation
from simulation_agent import agent as sim_error_agent
from simulation_rectifier_agent import agent as rectifier_agent
from langgraph.types import Command

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
            "All parameter extractions are complete. "
            "Start by asking the user for a dataset name, then proceed with "
            "resolve_and_validate before generating."
        ),
    },
    {
        "agent": "simulation",
        "name": "Simulation",
        "init_message": None,
    },
]

STAGE_NAMES = [s["name"] for s in STAGES]

# ---------------------------------------------------------------------------
# Per-session state
# ---------------------------------------------------------------------------


MAX_RECTIFIER_RETRIES = 3


@dataclass
class SessionState:
    session_id: str | None = None
    stage_index: int = 0
    configs: list = field(default_factory=list)
    dataset_name: str | None = None
    files_dir: str | None = None
    seen_counts: list = field(default_factory=list)
    # Rectifier feedback loop state
    rectifier_retries: int = 0
    patch_event: asyncio.Event | None = None
    patch_decision: dict | None = None
    sim_task: asyncio.Task | None = None

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
            elif msg.name == "post_dataset_to_db":
                try:
                    data = json.loads(msg.content)
                    if data.get("status") == "ok":
                        posted = True
                        # Derive files_dir from the module-level generation
                        # result (not exposed to the agent/user)
                        gen_result = _ds_gen_mod._last_generation_result
                        if gen_result and gen_result.output_dir:
                            data["files_dir"] = str(
                                Path(gen_result.output_dir) / "files"
                            )
                        dataset_info = data
                except (json.JSONDecodeError, TypeError):
                    pass

    return ai_texts, posted, dataset_info, len(messages)


async def _run_simulation_stage(ws: WebSocket, state: SessionState):
    """Run gprMax batch simulation with rectifier feedback loop.

    Stops on first error, runs diagnosis + rectifier, waits for user
    approval of the proposed patch, then loops back to dataset generation.
    Max MAX_RECTIFIER_RETRIES attempts before giving up.
    """
    files_dir = state.files_dir
    if not files_dir:
        await ws.send_json({
            "type": "agent_message",
            "content": "No dataset files directory available — cannot run simulation.",
        })
        return

    await ws.send_json({
        "type": "agent_message",
        "content": "Starting gprMax simulations on generated input files...",
    })

    # ── Run simulation (stop on first error) ──────────────────────────
    try:
        sim_result = await asyncio.to_thread(
            run_batch_simulation,
            input_dir=files_dir,
            skip_existing=False,
            stop_on_first_error=True,
        )
    except Exception as e:
        await ws.send_json({
            "type": "agent_message",
            "content": f"Simulation failed: {e}",
        })
        return

    # ── All passed — success ──────────────────────────────────────────
    if sim_result["failed"] == 0:
        summary = (
            f"Simulation complete.\n\n"
            f"- **Succeeded:** {sim_result['succeeded']}\n"
            f"- **Failed:** {sim_result['failed']}\n"
            f"- **Skipped:** {sim_result['skipped']}\n"
            f"- **Total:** {sim_result['total']}"
        )
        await ws.send_json({"type": "agent_message", "content": summary})

        # ── Extract signals from .out files & push to DB ─────────────
        await ws.send_json({
            "type": "agent_message",
            "content": "Extracting electromagnetic field signals from simulation outputs...",
        })
        try:
            from signal_extraction import extract_and_prepare_batch
            from db.db import bulk_update_signals

            extraction_result = await asyncio.to_thread(
                extract_and_prepare_batch,
                output_dir=sim_result["output_dir"],
                session_id=state.session_id,
            )

            if extraction_result["updates"]:
                await asyncio.to_thread(
                    bulk_update_signals,
                    extraction_result["updates"],
                )

            ext_summary = (
                f"Signal extraction complete.\n\n"
                f"- **Extracted:** {extraction_result['succeeded']}\n"
                f"- **Failed:** {extraction_result['failed']}"
            )
            if extraction_result["errors"]:
                ext_summary += "\n\n**Errors:**\n"
                for err in extraction_result["errors"]:
                    ext_summary += f"- `{err['filename']}`: {err['error']}\n"
            await ws.send_json({"type": "agent_message", "content": ext_summary})
        except Exception as e:
            logger.exception("Signal extraction failed")
            await ws.send_json({
                "type": "agent_message",
                "content": (
                    f"Warning: Signal extraction failed: {e}. "
                    f"Simulation outputs are saved but signals were not stored in the database."
                ),
            })

        await ws.send_json({"type": "simulation_complete", "result": sim_result})
        state.rectifier_retries = 0
        return

    # ── Simulation failed — enter rectifier loop ──────────────────────
    state.rectifier_retries += 1
    if state.rectifier_retries > MAX_RECTIFIER_RETRIES:
        await ws.send_json({
            "type": "agent_message",
            "content": (
                f"Simulation failed after **{MAX_RECTIFIER_RETRIES}** "
                f"rectification attempts. Manual intervention required."
            ),
        })
        await ws.send_json({"type": "simulation_complete", "result": sim_result})
        return

    error_info = sim_result["errors"][0]  # First (and only, due to stop)
    filename = error_info["filename"]
    error_tb = error_info["error"]

    # Read the failing .in file content from disk
    in_file_path = Path(files_dir) / filename
    try:
        in_file_content = in_file_path.read_text()
    except Exception:
        in_file_content = "(file not readable)"

    await ws.send_json({
        "type": "agent_message",
        "content": (
            f"**Simulation failed for `{filename}`** "
            f"(attempt {state.rectifier_retries}/{MAX_RECTIFIER_RETRIES})\n\n"
            f"Running error diagnosis..."
        ),
    })

    # ── Step 1: Error diagnosis agent (read-only) ─────────────────────
    diag_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    diag_prompt = (
        f"## Simulation Error\n\n"
        f"**Filename:** `{filename}`\n\n"
        f"**Error traceback:**\n```\n{error_tb}\n```\n\n"
        f"**Input file content:**\n```\n{in_file_content}\n```"
    )

    try:
        diag_result = await asyncio.to_thread(
            sim_error_agent.invoke,
            {"messages": [HumanMessage(content=diag_prompt)]},
            diag_config,
        )
        diag_parts = []
        for msg in diag_result.get("messages", []):
            if type(msg).__name__ == "AIMessage" and msg.content:
                diag_parts.append(msg.content)
        diagnosis = "\n\n".join(diag_parts) if diag_parts else "No analysis available."
    except Exception as e:
        logger.warning("Error diagnosis agent failed: %s", e)
        diagnosis = f"Error agent unavailable: {e}"

    await ws.send_json({"type": "agent_message", "content": diagnosis})

    # ── Step 2: Rectifier agent (proposes + applies patch via HITL) ───
    await ws.send_json({
        "type": "agent_message",
        "content": "Running simulation rectifier to determine parameter fix...",
    })

    rect_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    rect_prompt = (
        f"## Error Diagnosis\n\n{diagnosis}\n\n"
        f"## Failed Input File (`{filename}`)\n\n"
        f"```\n{in_file_content}\n```"
    )

    try:
        rect_result = await asyncio.to_thread(
            rectifier_agent.invoke,
            {"messages": [HumanMessage(content=rect_prompt)]},
            rect_config,
        )
    except Exception as e:
        logger.warning("Rectifier agent failed: %s", e)
        await ws.send_json({
            "type": "agent_message",
            "content": f"Rectifier agent error: {e}",
        })
        return

    # Send rectifier's explanation messages to user
    for msg in rect_result.get("messages", []):
        if type(msg).__name__ == "AIMessage" and msg.content:
            await ws.send_json({"type": "agent_message", "content": msg.content})

    # ── Step 3: Check for HITL interrupt (pending patch_parameters) ───
    graph_state = rectifier_agent.get_state(rect_config)

    if not graph_state.next:
        # No interrupt — rectifier didn't propose a patch
        await ws.send_json({
            "type": "agent_message",
            "content": (
                "The rectifier could not determine a parameter fix for this error. "
                "Manual intervention may be required."
            ),
        })
        return

    # Extract proposed patch from the interrupt's action requests
    patch_actions = []
    for task in graph_state.tasks:
        for intr in task.interrupts:
            hitl_req = intr.value
            for action in getattr(hitl_req, "action_requests", []):
                patch_actions.append({
                    "tool": getattr(action, "action", getattr(action, "name", "patch_parameters")),
                    "args": getattr(action, "args", {}),
                })

    # Send patch proposal to frontend for user approval
    await ws.send_json({
        "type": "patch_proposal",
        "patches": patch_actions,
        "retry_number": state.rectifier_retries,
        "max_retries": MAX_RECTIFIER_RETRIES,
    })

    # ── Step 4: Wait for user approval ────────────────────────────────
    state.patch_event = asyncio.Event()
    state.patch_decision = None
    await state.patch_event.wait()

    decision = state.patch_decision or {}
    approved = decision.get("approved", False)
    state.patch_event = None
    state.patch_decision = None

    if not approved:
        # User rejected — stop rectification
        await asyncio.to_thread(
            rectifier_agent.invoke,
            Command(resume={"decisions": [
                {"type": "reject", "message": "User rejected the proposed patch."}
            ]}),
            rect_config,
        )
        await ws.send_json({
            "type": "agent_message",
            "content": "Patch rejected. Simulation rectification stopped.",
        })
        return

    # ── Step 5: Resume rectifier — applies the patch ──────────────────
    await ws.send_json({
        "type": "agent_message",
        "content": "Patch approved. Applying parameter corrections...",
    })

    try:
        resume_result = await asyncio.to_thread(
            rectifier_agent.invoke,
            Command(resume={"decisions": [{"type": "approve"}]}),
            rect_config,
        )
        # Send any final messages from the rectifier
        for msg in resume_result.get("messages", []):
            if type(msg).__name__ == "AIMessage" and msg.content:
                await ws.send_json({"type": "agent_message", "content": msg.content})
    except Exception as e:
        logger.warning("Rectifier resume failed: %s", e)
        await ws.send_json({
            "type": "agent_message",
            "content": f"Failed to apply patch: {e}",
        })
        return

    # ── Step 6: Loop back to Dataset Generation stage ─────────────────
    ds_stage_index = 4  # Dataset Generation stage
    state.stage_index = ds_stage_index
    state.configs[ds_stage_index] = {"configurable": {"thread_id": str(uuid.uuid4())}}
    state.seen_counts[ds_stage_index] = 0

    await ws.send_json({
        "type": "stage_change",
        "stage_index": state.stage_index,
        "stage_name": STAGE_NAMES[state.stage_index],
    })

    await ws.send_json({
        "type": "agent_message",
        "content": (
            f"Parameters patched. Returning to dataset generation "
            f"(rectification attempt {state.rectifier_retries}/{MAX_RECTIFIER_RETRIES})..."
        ),
    })

    # Auto-start the dataset generation agent — it will regenerate
    # the dataset and eventually transition back to the simulation stage
    await _run_agent_turn(ws, state, init=True)


async def _invoke_agent(agent, input_messages, config):
    """Run agent.invoke in a thread to avoid blocking the event loop."""
    return await asyncio.to_thread(agent.invoke, input_messages, config)


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(ws: WebSocket, session_id: str):
    await ws.accept()

    state = sessions.setdefault(session_id, SessionState(session_id=session_id))

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

    # Auto-start the current stage if it hasn't been started
    if state.seen_counts[state.stage_index] == 0:
        stage = STAGES[state.stage_index]
        if stage["agent"] == "simulation":
            state.sim_task = asyncio.create_task(
                _run_simulation_stage(ws, state)
            )
        else:
            await _run_agent_turn(ws, state, init=True)

    # Listen for user messages
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            if msg.get("type") == "patch_response":
                # Forward patch approval/rejection to the simulation stage
                if state.patch_event is not None:
                    state.patch_decision = msg
                    state.patch_event.set()

            elif msg.get("type") == "user_message":
                stage = STAGES[state.stage_index]
                if stage["agent"] == "simulation":
                    if state.patch_event is not None:
                        await ws.send_json({
                            "type": "agent_message",
                            "content": "A patch is pending your approval. Please approve or reject it first.",
                        })
                    else:
                        await ws.send_json({
                            "type": "agent_message",
                            "content": "Simulation is running — please wait for it to complete.",
                        })
                else:
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

    # Handle stage transition (unified for extraction agents and dataset generation)
    if posted and state.stage_index < len(STAGES) - 1:
        # If dataset generation posted, extract files_dir and dataset_name
        if dataset_info:
            dataset_name = dataset_info.get("dataset_name")
            if dataset_name:
                state.dataset_name = dataset_name
                await ws.send_json({
                    "type": "dataset_ready",
                    "dataset_name": dataset_name,
                    "num_generated": dataset_info.get("num_generated", 0),
                })
            files_dir = dataset_info.get("files_dir")
            if files_dir:
                state.files_dir = files_dir

        state.stage_index += 1
        await ws.send_json({
            "type": "stage_change",
            "stage_index": state.stage_index,
            "stage_name": STAGE_NAMES[state.stage_index],
        })

        next_stage = STAGES[state.stage_index]
        if next_stage["agent"] == "simulation":
            state.sim_task = asyncio.create_task(
                _run_simulation_stage(ws, state)
            )
        else:
            await _run_agent_turn(ws, state, init=True)
        return


# ---------------------------------------------------------------------------
# File download endpoints
# ---------------------------------------------------------------------------

# dataset_generator.py writes to <project_root>/datasets/
DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"


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
