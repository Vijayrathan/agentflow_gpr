"""
FastAPI bridge from the frontend chatbot to the staged LangGraph pipeline.

The LangGraph workflow itself remains in `agentflow_langgraph.py`. This module
replaces the old terminal-facing orchestration with a WebSocket session
controller and persists the final generated dataset in one DB write pass.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

# Ensure imports work both under `uvicorn backend.api:app` and direct execution.
_backend_dir = str(Path(__file__).resolve().parent)
_project_root = str(Path(__file__).resolve().parent.parent)
_ds_dir = str(Path(__file__).resolve().parent / "dataset_sampling")
_gprmax_root = str(Path(__file__).resolve().parent.parent / "gprMax")
for _p in (_backend_dir, _project_root, _ds_dir, _gprmax_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agentflow_langgraph import (  # noqa: E402
    RESAMPLE_SECTIONS,
    SECTION_AGENTS,
    _captured_section,
    _remediation_message,
    _sections_from_tags,
)

# The graph file builds these nodes dynamically, so import the reusable pieces
# and deterministic nodes explicitly.
from agentflow_langgraph import (  # noqa: E402
    dataset_generation_node,
    global_derive_node,
    global_validation_node,
    layer_sampling_node,
    peplinski_derive_node,
    sample_validation_node,
    target_placement_node,
)
from db.db import ExtractionSession, batch_insert_simulations, get_session  # noqa: E402
from parameters_global_state import BASE_URL, start_parameter_server  # noqa: E402
from schema import (  # noqa: E402
    DatasetConfig,
    ExtractedAdvancedParams,
    ExtractedAntenna,
    ExtractedWaveform,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)


INIT_MESSAGES = {
    "dataset_config": (
        "I need to configure the dataset/run parameters for a gprMax "
        "simulation batch. Please begin the dataset configuration process."
    ),
    "layers": (
        "I need to set up the soil layers for a gprMax simulation. "
        "Please begin the layer parameter extraction process."
    ),
    "target_ranges": (
        "I need to configure the buried-target geometry ranges for a "
        "gprMax simulation batch. Please begin the target range extraction process."
    ),
    "waveform": (
        "I need to configure the waveform for a gprMax simulation. "
        "Please begin the waveform parameter extraction process."
    ),
    "antenna": (
        "I need to configure the antenna for a gprMax simulation. "
        "Please begin the antenna parameter extraction process."
    ),
    "advanced_params": (
        "I need to configure the advanced/optional parameters for a gprMax "
        "simulation. Please begin the advanced parameters extraction process."
    ),
}

DISPLAY_NAMES = {
    "dataset_config": "Dataset Configuration",
    "layers": "Layer Extraction",
    "target_ranges": "Buried-Target Range Extraction",
    "waveform": "Waveform Extraction",
    "antenna": "Antenna Extraction",
    "advanced_params": "Advanced Parameters Extraction",
}

GLOBAL_REMEDIATION_CHOICES = [
    "dataset_config",
    "antenna",
    "waveform",
    "layers",
    "advanced_params",
]


class FinalizeDatasetPayload(BaseModel):
    session_id: str
    user_id: str
    dataset_config: dict[str, Any]
    layers: dict[str, Any]
    target_ranges: Optional[dict[str, Any]] = None
    waveform: dict[str, Any]
    antenna: dict[str, Any]
    advanced_params: Optional[dict[str, Any]] = None
    artifacts: dict[str, Any] = Field(default_factory=dict)
    emission: dict[str, Any] = Field(default_factory=dict)


@dataclass
class ChatSession:
    session_id: str
    user_id: str
    state: dict[str, Any] = field(default_factory=lambda: {"halted": False})
    started: bool = False
    complete: bool = False
    busy: bool = False
    phase: str = "idle"
    active_section: Optional[str] = None
    active_agent: Any = None
    active_display: Optional[str] = None
    active_config: Optional[dict[str, Any]] = None
    active_seen: int = 0
    active_purpose: str = "collect"
    sample_remediation_queue: list[str] = field(default_factory=list)
    sample_remediation_errors: list[str] = field(default_factory=list)
    global_remediation_errors: list[str] = field(default_factory=list)
    global_choice_section: Optional[str] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


sessions: dict[str, ChatSession] = {}


app = FastAPI(title="GPR LangGraph Chat API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    start_parameter_server()
    logger.info("Parameter state server started on %s", BASE_URL)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/datasets/finalize")
async def finalize_dataset(payload: FinalizeDatasetPayload) -> dict[str, Any]:
    return await asyncio.to_thread(_finalize_dataset_sync, payload)


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(ws: WebSocket, session_id: str) -> None:
    await ws.accept()
    state = sessions.setdefault(session_id, _new_chat_session(session_id))

    try:
        async with state.lock:
            if not state.started:
                await _reset_parameter_state()
                state.started = True
                await ws.send_json({
                    "type": "agent_message",
                    "content": "Connected to the staged LangGraph dataset pipeline.",
                })
                await _start_agent(ws, state, "dataset_config", INIT_MESSAGES["dataset_config"])
            elif state.complete:
                await ws.send_json({
                    "type": "agent_message",
                    "content": "This dataset pipeline session is already complete.",
                })
            elif state.phase == "global_choice":
                await _send_global_choice(ws)
            elif state.phase == "agent" and state.active_agent is not None:
                await ws.send_json({
                    "type": "agent_message",
                    "content": "Reconnected. Continue with the current pipeline question.",
                })
            elif state.busy:
                await ws.send_json({
                    "type": "agent_message",
                    "content": "Reconnected while a pipeline step is still running. Please wait.",
                })
            else:
                replacement = _new_chat_session(session_id)
                sessions[session_id] = replacement
                state = replacement
                await _reset_parameter_state()
                state.started = True
                await ws.send_json({
                    "type": "agent_message",
                    "content": (
                        "The previous pipeline session was no longer resumable, "
                        "so I started a fresh dataset pipeline."
                    ),
                })
                await _start_agent(ws, state, "dataset_config", INIT_MESSAGES["dataset_config"])

        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            if msg.get("type") not in {"user_message", "choice_response"}:
                continue
            content = (msg.get("content") or msg.get("choice") or "").strip()
            if not content:
                continue
            async with state.lock:
                await _handle_user_text(ws, state, content)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for session %s", session_id)
    except Exception as exc:
        logger.exception("WebSocket error for session %s", session_id)
        with contextlib.suppress(Exception):
            await ws.send_json({"type": "error", "message": str(exc)})


def _new_chat_session(session_id: str) -> ChatSession:
    return ChatSession(session_id=session_id, user_id=session_id)


async def _reset_parameter_state() -> None:
    """Clear the agents' in-memory tool state before a new chat session."""
    try:
        await asyncio.to_thread(httpx.delete, f"{BASE_URL}/state", timeout=5)
    except Exception:
        logger.warning("Could not reset parameter server state", exc_info=True)


async def _handle_user_text(ws: WebSocket, chat: ChatSession, text: str) -> None:
    if chat.complete:
        await ws.send_json({
            "type": "agent_message",
            "content": "The dataset has already been created for this session.",
        })
        return
    if chat.busy:
        await ws.send_json({
            "type": "agent_message",
            "content": "A pipeline step is still running. Please wait for it to finish.",
        })
        return
    if chat.phase == "global_choice":
        await _handle_global_choice(ws, chat, text)
        return
    if chat.phase != "agent" or chat.active_agent is None:
        await ws.send_json({
            "type": "agent_message",
            "content": "The pipeline is not waiting for a chat reply right now.",
        })
        return

    chat.busy = True
    try:
        result = await _invoke_agent(chat.active_agent, text, chat.active_config)
        await _handle_agent_result(ws, chat, result)
    finally:
        chat.busy = False


async def _start_agent(
    ws: WebSocket,
    chat: ChatSession,
    section: str,
    init_message: str,
    purpose: str = "collect",
) -> None:
    agent, display = SECTION_AGENTS[section]
    chat.phase = "agent"
    chat.active_section = section
    chat.active_agent = agent
    chat.active_display = display
    chat.active_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    chat.active_seen = 0
    chat.active_purpose = purpose

    await ws.send_json({
        "type": "stage_change",
        "stage_name": display,
        "section": section,
    })

    chat.busy = True
    try:
        result = await _invoke_agent(agent, init_message, chat.active_config)
        await _handle_agent_result(ws, chat, result)
    finally:
        chat.busy = False


async def _invoke_agent(agent: Any, content: str, config: dict[str, Any] | None) -> dict:
    return await asyncio.to_thread(
        agent.invoke,
        {"messages": [HumanMessage(content=content)]},
        config,
    )


async def _handle_agent_result(ws: WebSocket, chat: ChatSession, result: dict) -> None:
    ai_texts, new_seen = _extract_ai_texts(result, chat.active_seen)
    chat.active_seen = new_seen
    for text in ai_texts:
        await ws.send_json({"type": "agent_message", "content": text})

    section = chat.active_section
    if section is None:
        return
    captured = _captured_section(result, section)
    if captured is None:
        return

    chat.state[section] = captured
    await ws.send_json({
        "type": "progress",
        "content": f"{chat.active_display or section} complete.",
        "section": section,
    })

    purpose = chat.active_purpose
    chat.active_section = None
    chat.active_agent = None
    chat.phase = "routing"

    if purpose == "sample_remediation":
        await _advance_sample_remediation(ws, chat)
    elif purpose == "global_remediation":
        await _advance_global_remediation(ws, chat, section)
    else:
        await _advance_after_collect(ws, chat, section)


def _extract_ai_texts(result: dict, seen: int) -> tuple[list[str], int]:
    messages = result.get("messages", [])
    texts = [
        msg.content
        for msg in messages[seen:]
        if type(msg).__name__ == "AIMessage" and getattr(msg, "content", None)
    ]
    return texts, len(messages)


async def _advance_after_collect(ws: WebSocket, chat: ChatSession, section: str) -> None:
    if section == "dataset_config":
        await _start_agent(ws, chat, "layers", INIT_MESSAGES["layers"])
    elif section == "layers":
        await _start_agent(ws, chat, "target_ranges", INIT_MESSAGES["target_ranges"])
    elif section == "target_ranges":
        await _run_deterministic(ws, chat, "Layer + Target Sampling", layer_sampling_node)
        await _start_agent(ws, chat, "waveform", INIT_MESSAGES["waveform"])
    elif section == "waveform":
        await _start_agent(ws, chat, "antenna", INIT_MESSAGES["antenna"])
    elif section == "antenna":
        await _run_sample_validation_gate(ws, chat)
    elif section == "advanced_params":
        await _run_derive_and_global_gate(ws, chat)


async def _run_sample_validation_gate(ws: WebSocket, chat: ChatSession) -> None:
    await _run_deterministic(ws, chat, "Sample Validation", sample_validation_node)
    if chat.state.get("sample_validation_passed"):
        await _start_agent(ws, chat, "advanced_params", INIT_MESSAGES["advanced_params"])
        return

    errors = chat.state.get("sample_validation_errors") or []
    await ws.send_json({
        "type": "validation_failed",
        "stage_name": "Sample Validation",
        "errors": errors,
    })
    sections = _sections_from_tags(errors, {"dataset_config", "waveform", "antenna"}) or ["waveform"]
    chat.sample_remediation_queue = list(sections)
    chat.sample_remediation_errors = list(errors)
    chat.state["sample_validation_passed"] = None
    chat.state["sample_validation_errors"] = None
    await _start_next_sample_remediation(ws, chat)


async def _start_next_sample_remediation(ws: WebSocket, chat: ChatSession) -> None:
    if not chat.sample_remediation_queue:
        await _run_sample_validation_gate(ws, chat)
        return
    section = chat.sample_remediation_queue.pop(0)
    msg = _remediation_message(
        section,
        chat.sample_remediation_errors,
        chat.state.get(section),
    )
    await _start_agent(ws, chat, section, msg, purpose="sample_remediation")


async def _advance_sample_remediation(ws: WebSocket, chat: ChatSession) -> None:
    await _start_next_sample_remediation(ws, chat)


async def _run_derive_and_global_gate(ws: WebSocket, chat: ChatSession) -> None:
    await _run_deterministic(ws, chat, "Peplinski Derive", peplinski_derive_node)
    await _run_deterministic(ws, chat, "Global Derive", global_derive_node)
    await _run_deterministic(ws, chat, "Global Validation", global_validation_node)

    if chat.state.get("global_validation_passed"):
        await _finish_dataset(ws, chat)
        return

    errors = chat.state.get("global_validation_errors") or []
    chat.global_remediation_errors = list(errors)
    chat.phase = "global_choice"
    await ws.send_json({
        "type": "validation_failed",
        "stage_name": "Global Validation",
        "errors": errors,
    })
    await _send_global_choice(ws)


async def _send_global_choice(ws: WebSocket) -> None:
    await ws.send_json({
        "type": "choice_required",
        "content": (
            "Global validation failed. Which section should be adjusted? "
            + ", ".join(GLOBAL_REMEDIATION_CHOICES)
        ),
        "choices": GLOBAL_REMEDIATION_CHOICES,
    })


async def _handle_global_choice(ws: WebSocket, chat: ChatSession, text: str) -> None:
    choice = text.strip().lower().replace(" ", "_")
    if choice in {"quit", "exit"}:
        chat.phase = "halted"
        chat.state["halted"] = True
        chat.state["halt_reason"] = "user exited during global remediation"
        await ws.send_json({"type": "agent_message", "content": "Pipeline halted."})
        return
    if choice not in GLOBAL_REMEDIATION_CHOICES:
        await ws.send_json({
            "type": "agent_message",
            "content": "Please choose one of: " + ", ".join(GLOBAL_REMEDIATION_CHOICES),
        })
        return

    chat.global_choice_section = choice
    msg = _remediation_message(choice, chat.global_remediation_errors, chat.state.get(choice))
    await _start_agent(ws, chat, choice, msg, purpose="global_remediation")


async def _advance_global_remediation(
    ws: WebSocket,
    chat: ChatSession,
    section: str,
) -> None:
    resample = section in RESAMPLE_SECTIONS
    chat.state["global_validation_passed"] = None
    chat.state["global_validation_errors"] = None
    chat.state["resample_after_global"] = resample

    if resample:
        await _run_deterministic(ws, chat, "Layer + Target Sampling", layer_sampling_node)
    await _run_derive_and_global_gate(ws, chat)


async def _finish_dataset(ws: WebSocket, chat: ChatSession) -> None:
    await _run_deterministic(ws, chat, "Per-Sample Target Placement", target_placement_node)
    await _run_deterministic(ws, chat, "Dataset Generation", dataset_generation_node)

    payload = _build_finalize_payload(chat)
    result = await finalize_dataset(FinalizeDatasetPayload.model_validate(payload))
    chat.complete = True
    chat.phase = "complete"
    await ws.send_json({
        "type": "dataset_ready",
        "content": (
            f"Dataset created and stored. Wrote {result['rows_inserted']} "
            "simulation row(s) for generated input files."
        ),
        "result": result,
    })


async def _run_deterministic(
    ws: WebSocket,
    chat: ChatSession,
    stage_name: str,
    fn: Any,
) -> None:
    chat.phase = "deterministic"
    chat.busy = True
    await ws.send_json({"type": "stage_change", "stage_name": stage_name})
    await ws.send_json({"type": "pipeline_busy", "busy": True})
    try:
        updates, output = await asyncio.to_thread(_run_node_with_output, fn, dict(chat.state))
        if updates:
            chat.state.update(updates)
        if output.strip():
            await ws.send_json({"type": "agent_message", "content": output.strip()})
    finally:
        chat.busy = False
        await ws.send_json({"type": "pipeline_busy", "busy": False})


def _run_node_with_output(fn: Any, state: dict[str, Any]) -> tuple[dict[str, Any], str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        updates = fn(state) or {}
    return updates, buf.getvalue()


def _resolve_dataset_path(output_dir: str, filename: str | None = None) -> Path:
    path = Path(output_dir)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    return path / filename if filename else path


def _read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _build_finalize_payload(chat: ChatSession) -> dict[str, Any]:
    cfg = DatasetConfig.model_validate(chat.state["dataset_config"])
    out_dir = _resolve_dataset_path(cfg.output_dir)
    sampled_path = out_dir / "sampled_layers.json"
    derived_path = out_dir / "derived_layers.json"
    global_path = out_dir / "global_derive.json"
    emitted_path = out_dir / "emitted_files.json"
    emitted = _read_json(emitted_path)

    return {
        "session_id": chat.session_id,
        "user_id": chat.user_id,
        "dataset_config": chat.state["dataset_config"],
        "layers": chat.state["layers"],
        "target_ranges": chat.state.get("target_ranges"),
        "waveform": chat.state["waveform"],
        "antenna": chat.state["antenna"],
        "advanced_params": chat.state.get("advanced_params"),
        "artifacts": {
            "output_dir": str(out_dir),
            "in_dir": emitted.get("in_dir"),
            "sampled_layers_json": str(sampled_path),
            "derived_layers_json": str(derived_path),
            "global_derive_json": str(global_path),
            "emitted_files_json": str(emitted_path),
        },
        "emission": {
            "num_requested": cfg.num_samples,
            "num_generated": emitted.get("n_written", 0),
            "num_failed": len(emitted.get("errors", [])),
            "files": emitted.get("files", []),
            "errors": emitted.get("errors", []),
        },
    }


def _coerce_uuid(value: str) -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except ValueError:
        return uuid.uuid5(uuid.NAMESPACE_URL, value)


def _finalize_dataset_sync(payload: FinalizeDatasetPayload) -> dict[str, Any]:
    session_uuid = _coerce_uuid(payload.session_id)
    cfg = DatasetConfig.model_validate(payload.dataset_config)
    wf = ExtractedWaveform.model_validate(payload.waveform)
    ant = ExtractedAntenna.model_validate(payload.antenna)
    adv = (
        ExtractedAdvancedParams.model_validate(payload.advanced_params)
        if payload.advanced_params is not None
        else None
    )

    artifacts = payload.artifacts
    sampled_path = Path(artifacts["sampled_layers_json"])
    global_path = Path(artifacts["global_derive_json"])
    emitted_path = Path(artifacts["emitted_files_json"])
    sampled_manifest = _read_json(sampled_path)
    global_derive = _read_json(global_path)
    emitted_manifest = _read_json(emitted_path)

    status = "complete" if not payload.emission.get("errors") else "partial"
    with get_session() as db:
        row = db.get(ExtractionSession, session_uuid)
        if row is None:
            row = ExtractionSession(id=session_uuid, user_id=payload.user_id)
            db.add(row)
        row.layers_ranges = {
            "layers": payload.layers,
            "target_ranges": payload.target_ranges,
        }
        row.antenna_waveform = {
            "antenna": payload.antenna,
            "waveform": payload.waveform,
        }
        row.model_config_data = {
            "dataset_config": payload.dataset_config,
            "global_derive": global_derive,
            "artifacts": artifacts,
            "emission": payload.emission,
        }
        row.advanced_params = payload.advanced_params
        row.num_samples_requested = cfg.num_samples
        row.status = status
        db.commit()

    sim_rows = _build_simulation_rows(
        session_uuid=session_uuid,
        user_id=payload.user_id,
        cfg=cfg,
        wf=wf,
        ant=ant,
        adv=adv,
        sampled_manifest=sampled_manifest,
        global_derive=global_derive,
        emitted_manifest=emitted_manifest,
    )
    inserted = batch_insert_simulations(sim_rows) if sim_rows else 0
    return {
        "status": status,
        "session_id": str(session_uuid),
        "rows_inserted": inserted,
        "num_generated": len(sim_rows),
        "output_dir": artifacts.get("output_dir"),
        "in_dir": artifacts.get("in_dir"),
    }


def _build_simulation_rows(
    *,
    session_uuid: uuid.UUID,
    user_id: str,
    cfg: DatasetConfig,
    wf: ExtractedWaveform,
    ant: ExtractedAntenna,
    adv: Optional[ExtractedAdvancedParams],
    sampled_manifest: dict[str, Any],
    global_derive: dict[str, Any],
    emitted_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    files_by_sample = {
        int(item["sample_id"]): item
        for item in emitted_manifest.get("files", [])
        if "sample_id" in item
    }
    rows: list[dict[str, Any]] = []
    for sample in sampled_manifest.get("samples", []):
        sample_id = int(sample["sample_id"])
        emitted = files_by_sample.get(sample_id)
        if emitted is None:
            continue

        cylinders = []
        if sample.get("target"):
            cylinders.append(sample["target"])
        if adv and adv.cylinders:
            cylinders.extend(c.model_dump() for c in adv.cylinders)

        rows.append({
            "id": uuid.uuid4(),
            "session_id": session_uuid,
            "user_id": user_id,
            "sample_index": sample_id,
            "antenna_kind": ant.antenna_kind or "hertzian_dipole",
            "antenna_axis": ant.antenna_axis or "x",
            "tx_rx_offset_m": ant.tx_rx_offset_m,
            "resistance": ant.resistance,
            "source_start_time": wf.source_start_time,
            "source_end_time": wf.source_end_time,
            "waveform_kind": wf.waveform_kind or "ricker",
            "waveform_amplitude": wf.waveform_amplitude,
            "waveform_center_freq_hz": wf.waveform_center_freq_hz,
            "waveform_name": wf.waveform_name,
            "model": cfg.model_basename,
            "title": cfg.model_basename,
            "source_height_m": global_derive["source_height_m"],
            "domain_x": global_derive["domain_x_m"],
            "domain_y": global_derive["domain_y_m"],
            "cells_per_wavelength": cfg.cells_per_wavelength,
            "max_cell_m": global_derive["dx_m"],
            "rx_same_height": True if ant.rx_same_height is None else ant.rx_same_height,
            "temperature_c": 20.0,
            "enforce_validity": True,
            "pml_cells": cfg.pml_cells,
            "num_threads": cfg.num_threads,
            "output_dir": cfg.output_dir,
            "layers": sample.get("layers", []),
            "num_layers": len(sample.get("layers", [])),
            "cylinders": cylinders or None,
            "boxes": [b.model_dump() for b in adv.boxes] if adv and adv.boxes else None,
            "spheres": [s.model_dump() for s in adv.spheres] if adv and adv.spheres else None,
            "surface_roughness": (
                adv.surface_roughness.model_dump()
                if adv and adv.surface_roughness
                else None
            ),
            "snapshots": (
                [s.model_dump() for s in adv.snapshots]
                if adv and adv.snapshots
                else None
            ),
            "rx_array": ant.rx_array.model_dump() if ant.rx_array else None,
            "input_file_path": emitted.get("path"),
            "output_file_path": None,
        })
    return rows
