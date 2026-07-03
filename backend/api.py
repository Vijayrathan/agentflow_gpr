"""
FastAPI bridge from the frontend chatbot to the SINGLE-AGENT pipeline.

The collection workflow lives in `agentflow_single_agent.py`: ONE deep agent
on ONE conversation thread collects every parameter section, with the
section-specific instructions injected as stage kickoff messages. This module
drives that flow over a WebSocket session (one isolated `SingleAgentSession`
per chat), runs the deterministic derive/validate stages between agent stages,
and persists the final generated dataset in one DB write pass.

Differences from the retired multi-agent bridge (agentflow_langgraph-based):
- No parameter server (port 8100) — the agent's tools write to the session's
  in-process store, which is synced into the pipeline state after each stage.
- Stage completion is read from the store (`stage_done`), not scraped from
  `post_parameters` tool calls.
- Remediation happens in the SAME conversation: validation errors are injected
  as messages and the agent agrees the fix with the user. The global-validation
  "which section?" choice menu is gone — no more `choice_required` round-trip.
- Cross-section edits are allowed at any time; if `layers` / `dataset_config`
  / `target_ranges` change after sampling ran, the samples are re-drawn before
  the derive chain (staleness check on the sampling snapshot).
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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure imports work both under `uvicorn backend.api:app` and direct execution.
_backend_dir = str(Path(__file__).resolve().parent)
_project_root = str(Path(__file__).resolve().parent.parent)
_ds_dir = str(Path(__file__).resolve().parent / "dataset_sampling")
_gprmax_root = str(Path(__file__).resolve().parent.parent / "gprMax")
for _p in (_backend_dir, _project_root, _ds_dir, _gprmax_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agentflow_single_agent import (  # noqa: E402
    RESAMPLE_SECTIONS,
    SECTION_DISPLAY,
    SingleAgentSession,
    _changed_sections,
    _samples_stale,
    dataset_generation_node,
    global_derive_node,
    global_validation_node,
    layer_sampling_node,
    peplinski_derive_node,
    sample_validation_node,
    target_placement_node,
)
from single_agent_prompts import (  # noqa: E402
    SECTION_KICKOFF,
    global_remediation_message,
    sample_remediation_message,
)
from db.db import ExtractionSession, batch_insert_simulations, get_session  # noqa: E402
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
    # ONE agent + ONE thread + its own section store for the whole chat.
    agent_session: SingleAgentSession = field(default_factory=SingleAgentSession)
    state: dict[str, Any] = field(default_factory=lambda: {"halted": False})
    started: bool = False
    complete: bool = False
    busy: bool = False
    phase: str = "idle"  # idle | agent | deterministic | routing | halted | complete
    # What the current agent turn is for. "collect" waits for `active_section`
    # to be complete in the store; the remediation purposes wait for the agent
    # to change (and complete) at least one section vs `remediation_snapshot`.
    active_purpose: str = "collect"
    active_section: Optional[str] = None
    remediation_snapshot: Optional[dict[str, Any]] = None
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
                state.started = True
                await ws.send_json({
                    "type": "agent_message",
                    "content": "Connected to the single-agent dataset pipeline.",
                })
                await _start_stage(ws, state, "dataset_config")
            elif state.complete:
                await ws.send_json({
                    "type": "agent_message",
                    "content": "This dataset pipeline session is already complete.",
                })
            elif state.phase == "agent":
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
                state.started = True
                await ws.send_json({
                    "type": "agent_message",
                    "content": (
                        "The previous pipeline session was no longer resumable, "
                        "so I started a fresh dataset pipeline."
                    ),
                })
                await _start_stage(ws, state, "dataset_config")

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
    if chat.phase != "agent":
        await ws.send_json({
            "type": "agent_message",
            "content": "The pipeline is not waiting for a chat reply right now.",
        })
        return
    if text.lower() in {"quit", "exit"}:
        chat.phase = "halted"
        chat.state["halted"] = True
        chat.state["halt_reason"] = f"user exited during {chat.active_purpose}"
        await ws.send_json({"type": "agent_message", "content": "Pipeline halted."})
        return

    chat.busy = True
    try:
        result = await asyncio.to_thread(chat.agent_session.invoke, text)
        await _handle_agent_result(ws, chat, result)
    finally:
        chat.busy = False


async def _invoke_and_handle(ws: WebSocket, chat: ChatSession, message: str) -> None:
    """Send `message` into the ONE ongoing conversation and process the turn."""
    chat.busy = True
    try:
        result = await asyncio.to_thread(chat.agent_session.invoke, message)
        await _handle_agent_result(ws, chat, result)
    finally:
        chat.busy = False


async def _start_stage(ws: WebSocket, chat: ChatSession, section: str) -> None:
    """Kick off a collection stage by injecting its section-specific
    instructions (batches, constraints, JSON schema) into the conversation."""
    chat.phase = "agent"
    chat.active_purpose = "collect"
    chat.active_section = section

    await ws.send_json({
        "type": "stage_change",
        "stage_name": SECTION_DISPLAY[section],
        "section": section,
    })
    await _invoke_and_handle(ws, chat, SECTION_KICKOFF[section])


async def _handle_agent_result(ws: WebSocket, chat: ChatSession, result: dict) -> None:
    for text in chat.agent_session.new_ai_texts(result):
        await ws.send_json({"type": "agent_message", "content": text})

    if chat.active_purpose == "collect":
        await _check_collect_done(ws, chat)
    else:
        await _check_remediation_done(ws, chat)


async def _check_collect_done(ws: WebSocket, chat: ChatSession) -> None:
    section = chat.active_section
    if section is None or not chat.agent_session.stage_done(section):
        return  # keep collecting — wait for the next user message

    _sync_sections(chat)
    await ws.send_json({
        "type": "progress",
        "content": f"{SECTION_DISPLAY[section]} complete.",
        "section": section,
    })
    chat.active_section = None
    chat.phase = "routing"
    await _advance_after_collect(ws, chat, section)


async def _check_remediation_done(ws: WebSocket, chat: ChatSession) -> None:
    """Remediation completes when the agent changed >= 1 section vs the
    snapshot taken at gate failure, and every changed section is complete."""
    changed = _changed_sections(
        chat.remediation_snapshot or {}, chat.agent_session.snapshot()
    )
    if not changed or not all(chat.agent_session.stage_done(s) for s in changed):
        return  # keep discussing — wait for the next user message

    _sync_sections(chat)
    purpose = chat.active_purpose
    chat.remediation_snapshot = None
    chat.phase = "routing"
    await ws.send_json({
        "type": "progress",
        "content": f"Updated {', '.join(sorted(changed))} — re-validating.",
    })

    if purpose == "sample_remediation":
        await _run_sample_validation_gate(ws, chat)
    else:
        chat.state["resample_after_global"] = bool(changed & RESAMPLE_SECTIONS)
        await _run_derive_chain(ws, chat)


def _sync_sections(chat: ChatSession) -> None:
    """Copy the WHOLE store into the pipeline state, so cross-section edits
    made during any stage land in state as soon as that turn completes."""
    chat.state.update(chat.agent_session.state_sync())


async def _advance_after_collect(ws: WebSocket, chat: ChatSession, section: str) -> None:
    if section == "dataset_config":
        await _start_stage(ws, chat, "layers")
    elif section == "layers":
        await _start_stage(ws, chat, "target_ranges")
    elif section == "target_ranges":
        await _run_deterministic(ws, chat, "Layer + Target Sampling", layer_sampling_node)
        await _start_stage(ws, chat, "waveform")
    elif section == "waveform":
        await _start_stage(ws, chat, "antenna")
    elif section == "antenna":
        await _run_sample_validation_gate(ws, chat)
    elif section == "advanced_params":
        await _run_derive_chain(ws, chat)


async def _run_sample_validation_gate(ws: WebSocket, chat: ChatSession) -> None:
    await _run_deterministic(ws, chat, "Sample Validation", sample_validation_node)
    if chat.state.get("sample_validation_passed"):
        await _start_stage(ws, chat, "advanced_params")
        return

    errors = chat.state.get("sample_validation_errors") or []
    await ws.send_json({
        "type": "validation_failed",
        "stage_name": "Sample Validation",
        "errors": errors,
    })
    chat.state["sample_validation_passed"] = None
    chat.state["sample_validation_errors"] = None
    await _start_remediation(
        ws, chat, "sample_remediation",
        sample_remediation_message(errors, chat.agent_session.store),
    )


async def _start_remediation(
    ws: WebSocket,
    chat: ChatSession,
    purpose: str,
    message: str,
) -> None:
    """Inject the validation errors into the SAME conversation. The agent —
    which owns every section — explains the problem and agrees the fix with
    the user; no orchestrator-side section choice is needed."""
    chat.phase = "agent"
    chat.active_purpose = purpose
    chat.active_section = None
    chat.remediation_snapshot = chat.agent_session.snapshot()
    await _invoke_and_handle(ws, chat, message)


async def _run_derive_chain(ws: WebSocket, chat: ChatSession) -> None:
    """Re-sample if needed, then derive the grid and run the global gate."""
    if chat.state.pop("resample_after_global", None) or _samples_stale(chat.state):
        await _run_deterministic(ws, chat, "Layer + Target Sampling", layer_sampling_node)

    await _run_deterministic(ws, chat, "Peplinski Derive", peplinski_derive_node)
    await _run_deterministic(ws, chat, "Global Derive", global_derive_node)
    await _run_deterministic(ws, chat, "Global Validation", global_validation_node)

    if chat.state.get("global_validation_passed"):
        await _finish_dataset(ws, chat)
        return

    errors = chat.state.get("global_validation_errors") or []
    await ws.send_json({
        "type": "validation_failed",
        "stage_name": "Global Validation",
        "errors": errors,
    })
    chat.state["global_validation_passed"] = None
    chat.state["global_validation_errors"] = None
    await _start_remediation(
        ws, chat, "global_remediation",
        global_remediation_message(errors, chat.agent_session.store),
    )


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
