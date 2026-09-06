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

Refresh/reconnect model: every chat-visible event is recorded on the session's
`transcript`, and pipeline output is routed through `chat.ws` (the CURRENT
socket) rather than the socket that started the turn — so a page refresh
mid-step neither kills the pipeline nor loses its output. A reconnect replays
the full transcript (`session_restore`), the dataset result, the busy state,
and the last scene, restoring the UI to exactly where the session left off.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import re
import shutil
import sys
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
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
    SAMPLING_INPUT_SECTIONS,
    SECTION_DISPLAY,
    SingleAgentSession,
    _changed_sections,
    _sampling_failure_errors,
    _samples_stale,
    _scoped_output_dir,
    dataset_generation_node,
    global_derive_node,
    global_validation_node,
    layer_sampling_node,
    peplinski_derive_node,
    sample_validation_node,
    target_placement_node,
)
from single_agent_prompts import (  # noqa: E402
    POST_COMPLETE_BRIEFING,
    SECTION_KICKOFF,
    global_remediation_message,
    layer_sampling_remediation_message,
    sample_remediation_message,
)
from deck_validation import extract_deck_members, validate_deck_bytes  # noqa: E402
from viz_projection import build_scene  # noqa: E402
from simulate import run_batch_simulation  # noqa: E402
import sim_similarity  # noqa: E402  (light import; qdrant_client stays lazy inside)
from db.db import (  # noqa: E402
    ExtractionSession,
    batch_insert_simulations,
    bulk_update_signals,
    count_incomplete_simulations,
    create_chat_stub,
    delete_simulations_for_session,
    get_chat_session,
    get_extraction_session,
    get_session,
    get_simulations_for_session,
    list_chat_sessions,
    set_simulation_outputs,
    upsert_chat_session,
)
from checkpointer import get_checkpointer  # noqa: E402
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
    # Which pipeline products are current for THIS session — gates what the
    # scene projection may read from disk (never a stale manifest).
    viz_flags: dict[str, bool] = field(default_factory=lambda: {
        "sampled": False, "derived": False, "grid": False,
        "placed": False, "emitted": False,
    })
    last_scene: Optional[dict[str, Any]] = None
    # The CURRENT socket for this session. All pipeline output is sent here,
    # so a page refresh mid-turn re-routes the remaining output to the new
    # connection instead of dying on the old one.
    ws: Optional[WebSocket] = None
    # Every chat-visible event (user + agent + status), replayed on reconnect
    # so a refreshed page shows the whole conversation.
    transcript: list[dict[str, Any]] = field(default_factory=list)
    # Finalize result kept for replay: repopulates the dataset tab on refresh.
    dataset_result: Optional[dict[str, Any]] = None
    # Set when the user imported a zip of .in decks (upload_dataset_zip): the
    # dataset endpoints serve THIS directory instead of the pipeline's
    # dataset_config.output_dir, until the pipeline emits its own dataset
    # (_finish_dataset clears it).
    uploaded_output_dir: Optional[str] = None
    # Forward-model (gprMax) run state. `simulating` guards against concurrent
    # runs and restores the run indicator on reconnect; the result is kept so
    # a refresh re-hydrates the "solved" state.
    simulating: bool = False
    simulation_result: Optional[dict[str, Any]] = None
    # Pending forward-model reuse recommendation (similar past session found
    # by sim_similarity). Set when /simulate returns "reuse_recommended";
    # cleared on adoption, on a forced run's completion path being irrelevant
    # (a new recommendation replaces it), and whenever the dataset itself is
    # replaced (_finish_dataset / upload).
    reuse_recommendation: Optional[dict[str, Any]] = None
    # Post-completion edit → regenerate machinery. `complete_snapshot` is the
    # store as of the last successful _finish_dataset; post-complete turns are
    # diffed against it. `regenerating` survives remediation turns: it routes
    # the sample gate's pass branch into the derive chain (advanced_params is
    # already collected) and 409s the forward model for the whole regeneration.
    complete_snapshot: Optional[dict[str, Any]] = None
    regenerating: bool = False
    post_complete_briefed: bool = False
    # Dedup for the "section incomplete — regeneration blocked" notice.
    regen_block_notice: Optional[frozenset] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


sessions: dict[str, ChatSession] = {}


# Event types that belong to the conversation record (replayed on reconnect).
# Transient signals (pipeline_busy, model_update, session_restore, error) are
# re-derived from session state instead. `user_message` events are appended
# to the transcript directly (record-only — never echoed back to the client).
RECORDED_EVENT_TYPES = {
    "agent_message",
    "stage_change",
    "progress",
    "validation_failed",
    "dataset_ready",
    "simulation_complete",
    "reuse_recommendation",
}


async def _send(chat: ChatSession, payload: dict[str, Any]) -> None:
    """Record chat-visible events, then push to the session's CURRENT socket.

    Send failures are swallowed: if the client refreshed mid-turn the payload
    is already in the transcript, so the reconnect replay delivers it — the
    pipeline itself must never die on a closed socket."""
    if payload.get("type") in RECORDED_EVENT_TYPES:
        chat.transcript.append(payload)
    ws = chat.ws
    if ws is None:
        return
    try:
        await ws.send_json(payload)
    except Exception:
        logger.info(
            "send to session %s failed (client gone?) — event kept in transcript",
            chat.session_id,
        )


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    """Warm up the RAG system in the BACKGROUND on startup so it's ready to serve
    soon after boot WITHOUT blocking the server from accepting connections.

    Loading the BGE-M3 encoder + reranker and connecting to Qdrant is slow, and if
    Qdrant is unreachable the connection can hang — doing this inline would stall
    (or deadlock) startup, so uvicorn would never finish "application startup" and
    the frontend WebSocket couldn't connect. Instead we kick it off as a background
    task and yield immediately. If a request arrives before warmup finishes,
    `_get_rag()` builds the instance lazily. Failures are swallowed — chat/pipeline
    still work and `rag_search` returns its error sentinel."""

    async def _warmup() -> None:
        try:
            # Lazy import keeps the heavy RAG deps off the module import path (and
            # out of key-free test imports); the model load runs in a worker thread.
            from backend.rag import init_rag

            logger.info("Warming up RAG system (encoder + reranker, Qdrant) in background...")
            await asyncio.to_thread(init_rag)
            logger.info("RAG ready.")
        except Exception:
            logger.exception("RAG warmup failed — lazy fallback on first use.")

    warmup_task = asyncio.create_task(_warmup())
    try:
        yield
    finally:
        warmup_task.cancel()
        with contextlib.suppress(Exception):
            await warmup_task


app = FastAPI(title="GPR LangGraph Chat API", lifespan=_lifespan)
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


def _session_output_dir(session_id: str) -> Path:
    """Resolve the on-disk dataset directory for an active chat session.
    An uploaded deck zip takes precedence until the pipeline emits its own
    dataset (see upload_dataset_zip / _finish_dataset)."""
    chat = _resolve_chat_sync(session_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Unknown session")
    if chat.uploaded_output_dir:
        out_dir = Path(chat.uploaded_output_dir)
    else:
        cfg = chat.state.get("dataset_config")
        if not cfg:
            raise HTTPException(status_code=404, detail="Session has no dataset yet")
        out_dir = _resolve_dataset_path(DatasetConfig.model_validate(cfg).output_dir)
    if not out_dir.exists():
        raise HTTPException(status_code=404, detail="Dataset directory not found")
    return out_dir


def _session_emitted_manifest(session_id: str) -> tuple[Path, dict[str, Any]]:
    out_dir = _session_output_dir(session_id)
    manifest_path = out_dir / "emitted_files.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="No generated input files yet")
    return out_dir, _read_json(manifest_path)


def _in_files_dir(out_dir: Path, manifest: dict[str, Any]) -> Path:
    in_dir = manifest.get("in_dir")
    return Path(in_dir) if in_dir else out_dir / "in_files"


@app.get("/datasets/{session_id}/files")
def list_dataset_files(session_id: str) -> dict[str, Any]:
    _out_dir, manifest = _session_emitted_manifest(session_id)
    files = [
        {"sample_id": f.get("sample_id"), "filename": f.get("filename")}
        for f in manifest.get("files", [])
        if f.get("filename")
    ]
    return {"n_written": manifest.get("n_written", len(files)), "files": files}


@app.get("/datasets/{session_id}/files/{filename}")
def get_dataset_file(session_id: str, filename: str) -> PlainTextResponse:
    out_dir, manifest = _session_emitted_manifest(session_id)
    in_dir = _in_files_dir(out_dir, manifest)
    # Only ever serve a bare filename from the session's own in_files dir.
    path = in_dir / Path(filename).name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Input file not found")
    return PlainTextResponse(path.read_text(encoding="utf-8"))


def _out_files_dir(out_dir: Path) -> Path:
    return out_dir / "out_files"


@app.get("/datasets/{session_id}/outputs")
def list_dataset_outputs(session_id: str) -> dict[str, Any]:
    """List the .in filenames whose forward-model .out file exists on disk.
    Availability is filesystem-derived so a page refresh — even mid-run —
    restores the "view outcome" buttons."""
    out_dir, manifest = _session_emitted_manifest(session_id)
    outputs_dir = _out_files_dir(out_dir)
    files = [
        f["filename"]
        for f in manifest.get("files", [])
        if f.get("filename")
        and (outputs_dir / (Path(f["filename"]).stem + ".out")).is_file()
        and _output_integrity_ok(outputs_dir, manifest, f)
    ]
    return {"files": files}


@app.get("/datasets/{session_id}/outputs/{filename}")
def get_dataset_output(session_id: str, filename: str) -> dict[str, Any]:
    """A-scan payload for one emitted file: rx1 field components + time axis
    read from the gprMax HDF5 output. `filename` is the .in name."""
    out_dir, _manifest = _session_emitted_manifest(session_id)
    entry = next((f for f in _manifest.get("files", []) if f.get("filename") == filename), None)
    if entry is None:
        raise HTTPException(status_code=404, detail="File is not in the current dataset manifest")
    # Only ever serve a bare filename's .out from the session's own dir.
    out_path = _out_files_dir(out_dir) / (Path(Path(filename).name).stem + ".out")
    if not out_path.is_file():
        raise HTTPException(status_code=404, detail="Output file not found")
    if not _output_integrity_ok(_out_files_dir(out_dir), _manifest, entry):
        raise HTTPException(status_code=422, detail="Output failed contract/receipt integrity validation")
    # Lazy import keeps h5py off the api import path.
    from signal_extraction import read_ascan

    try:
        data = read_ascan(out_path)
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"Could not read output file: {exc}"
        ) from exc
    from backend.qualification import qualification_status
    return {"filename": out_path.name, **data, "receiver": "rx1",
            "qualification": qualification_status(out_dir, _manifest) if _manifest.get("contract") else {"status": "legacy_unverified"},
            "geometry_url": f"/datasets/{session_id}/geometry/{filename}" if _manifest.get("contract") else None,
            "coordinate_frame": (_manifest.get("contract") or {}).get("coordinate_frame"),
            "field_units": {"E": "V/m", "H": "A/m"}}


def _output_integrity_ok(outputs_dir, manifest, entry):
    if not manifest.get("contract"):
        return True
    try:
        from backend.signal_extraction import validate_output
        validate_output(outputs_dir / (Path(entry["filename"]).stem + ".out"),
                        manifest["contract"], entry["resolved_scene"],
                        input_sha256=entry["input_sha256"], require_receipt=True)
        return True
    except (OSError, ValueError, KeyError):
        return False


@app.get("/datasets/{session_id}/geometry/{filename}")
def get_dataset_geometry(session_id: str, filename: str):
    out_dir, manifest = _session_emitted_manifest(session_id)
    entry = next((f for f in manifest.get("files", []) if f.get("filename") == filename), None)
    output_dir = _out_files_dir(out_dir)
    if not entry or not manifest.get("contract") or not _output_integrity_ok(output_dir, manifest, entry):
        raise HTTPException(status_code=404, detail="No verified native geometry for this sample")
    path = output_dir / (Path(filename).stem + ".geometry.h5")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Native geometry export is missing")
    return FileResponse(path, filename=path.name, media_type="application/x-hdf5")


@app.get("/datasets/{session_id}/download")
def download_dataset_zip(session_id: str) -> StreamingResponse:
    out_dir, manifest = _session_emitted_manifest(session_id)
    in_dir = _in_files_dir(out_dir, manifest)
    entries = [
        in_dir / Path(f["filename"]).name
        for f in manifest.get("files", [])
        if f.get("filename")
    ]
    entries = [p for p in entries if p.is_file()]
    if not entries:
        raise HTTPException(status_code=404, detail="No input files to download")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in entries:
            if manifest.get("contract"):
                from backend.dataset_sampling.contract import file_digest
                entry = next(f for f in manifest["files"] if f["filename"] == path.name)
                if file_digest(path) != entry["input_sha256"]:
                    raise HTTPException(status_code=409, detail="Input file differs from current contract")
            zf.write(path, arcname=path.name)
        if manifest.get("contract"):
            for name in ("dataset_contract.json", "emitted_files.json", "sampled_layers.json", "derived_layers.json", "global_derive.json", "dropped_targets.json"):
                path = out_dir / name
                if path.is_file():
                    zf.write(path, arcname=name)
    buf.seek(0)
    zip_name = f"{out_dir.name}_input_deck.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_name}"'},
    )


MAX_UPLOAD_ZIP_BYTES = 50 * 1024 * 1024


@app.post("/datasets/{session_id}/upload")
async def upload_dataset_zip(
    session_id: str, request: Request, filename: str = "uploaded_dataset.zip"
) -> dict[str, Any]:
    """Import a user zip of gprMax .in decks as the session's dataset
    ("Upload → From file…"; the raw zip is the request body, no multipart).

    Every deck must pass gprMax's own command-syntax rules (deck_validation)
    before anything lands on disk. Valid decks are written under
    ./dataset/<zip stem>/in_files with an emission-style manifest, so the
    dataset tab, the file/outcome viewers, the forward model and the zip
    download all treat the upload exactly like a generated dataset."""
    chat = await _resolve_chat(session_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Unknown session")
    if chat.simulating:
        raise HTTPException(
            status_code=409, detail="Forward model is running — wait for it to finish"
        )
    if chat.busy:
        raise HTTPException(
            status_code=409,
            detail="A pipeline step is still running — wait for it to finish",
        )
    if chat.regenerating:
        raise HTTPException(
            status_code=409,
            detail="Dataset is regenerating — wait for it to finish",
        )
    data = await request.body()
    if not data:
        raise HTTPException(
            status_code=422, detail="Empty upload — send the zip file as the request body"
        )
    if len(data) > MAX_UPLOAD_ZIP_BYTES:
        raise HTTPException(status_code=413, detail="Zip exceeds the 50 MB upload limit")

    result = await asyncio.to_thread(_import_deck_zip, chat, data, filename)
    # Recorded chat event: repopulates the dataset tab live AND on reconnect.
    await _send(chat, {
        "type": "dataset_ready",
        "content": _upload_summary(result),
        "result": result,
    })
    await _persist_chat(chat)
    return result


def _import_deck_zip(chat: ChatSession, data: bytes, zip_name: str) -> dict[str, Any]:
    """Validate every .in member and write the valid ones as the session's
    uploaded dataset. Rejected files are reported per-file, never written."""
    try:
        members, rejected = extract_deck_members(data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not members and not rejected:
        raise HTTPException(status_code=422, detail="The zip contains no .in files")

    decks: list[tuple[str, str]] = []
    for name, blob in members:
        errors = validate_deck_bytes(blob)
        if errors:
            rejected.append({"filename": name, "error": "; ".join(errors)})
        else:
            decks.append((name, blob.decode("utf-8")))

    if not decks:
        details = "; ".join(f"{r['filename']}: {r['error']}" for r in rejected[:10])
        raise HTTPException(
            status_code=422,
            detail=f"No file passed the gprMax syntax check — {details}",
        )

    # Same per-user/per-chat scoping as generated datasets — identical zip
    # names across chats or users never share a directory.
    out_dir = _resolve_dataset_path(
        _scoped_output_dir(chat.user_id, chat.session_id, Path(zip_name).stem)
    )
    in_dir = out_dir / "in_files"
    in_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for i, (name, text) in enumerate(decks):
        path = in_dir / name
        path.write_text(text, encoding="utf-8")
        files.append({"sample_id": i + 1, "filename": name, "path": str(path)})

    manifest = {
        "output_dir": str(out_dir),
        "in_dir": str(in_dir),
        "n_written": len(files),
        # Gates DB output recording: uploads have no Simulation rows and their
        # positional sample ids must never overwrite a generated dataset's.
        "source": "upload",
        "zip_name": zip_name,
        "files": files,
        "errors": [f"{r['filename']}: {r['error']}" for r in rejected],
    }
    with open(out_dir / "emitted_files.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    result = {
        "status": "uploaded" if not rejected else "partial",
        "session_id": chat.session_id,
        "output_dir": str(out_dir),
        "in_dir": str(in_dir),
        "n_written": len(files),
        "files": [
            {"sample_id": x["sample_id"], "filename": x["filename"]} for x in files
        ],
        "rejected": rejected,
    }
    chat.uploaded_output_dir = str(out_dir)
    chat.dataset_result = result  # replayed on reconnect → repopulates the tab
    chat.simulation_result = None  # previous dataset's outcomes no longer apply
    chat.reuse_recommendation = None  # recommendation was for the replaced dataset
    return result


def _upload_summary(result: dict[str, Any]) -> str:
    msg = (
        f"Imported {result['n_written']} gprMax input file(s) from the uploaded "
        "zip — they are in the Dataset tab, ready for the forward model."
    )
    rejected = result.get("rejected") or []
    if rejected:
        shown = "\n".join(f"- `{r['filename']}`: {r['error']}" for r in rejected[:10])
        more = f"\n- … and {len(rejected) - 10} more" if len(rejected) > 10 else ""
        msg += (
            f"\n\n{len(rejected)} file(s) failed the gprMax syntax check "
            f"and were skipped:\n{shown}{more}"
        )
    return msg


@app.post("/datasets/{session_id}/simulate")
async def start_forward_model(session_id: str, force: bool = False) -> dict[str, Any]:
    """Kick off the gprMax forward model on the session's emitted .in files
    ("Run forward model" button). Returns immediately; per-file progress and
    the final summary stream over the session's WebSocket.

    Unless `force=true`, a similarity check against previously simulated
    sessions runs first: on a >=threshold match the run is NOT started and
    the response is `reuse_recommended` (the frontend offers Reuse /
    Simulate-anyway; "Simulate anyway" re-POSTs with force=true)."""
    chat = await _resolve_chat(session_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Unknown session")
    out_dir, manifest = _session_emitted_manifest(session_id)
    in_dir = _in_files_dir(out_dir, manifest)
    # Run ONLY this session's emitted files — the per-dataset in_files dir can
    # still hold stale decks (re-emission after a re-sample, or another session
    # reusing the same model basename).
    filenames = [
        Path(f["filename"]).name
        for f in manifest.get("files", [])
        if f.get("filename") and (in_dir / Path(f["filename"]).name).is_file()
    ]
    if not filenames:
        raise HTTPException(status_code=404, detail="No generated input files yet")
    if chat.simulating:
        raise HTTPException(status_code=409, detail="Forward model already running")
    if chat.busy:
        raise HTTPException(
            status_code=409,
            detail="A pipeline step is still running — wait for it to finish",
        )
    # `busy` is False between remediation turns mid-regeneration, but the old
    # emitted manifest may still be on disk — never simulate it.
    if chat.regenerating:
        raise HTTPException(
            status_code=409,
            detail="Dataset is regenerating — wait for it to finish",
        )

    # Reuse gate: uploads have no comparable config, and `force` is the
    # explicit "simulate anyway" escape hatch. Any failure inside the check
    # yields None — the run must proceed normally when the similarity stack
    # is down.
    if not force and manifest.get("source") != "upload":
        rec = await asyncio.to_thread(_find_reuse_candidate, chat)
        if rec is not None:
            chat.reuse_recommendation = rec
            await _send(chat, {
                "type": "reuse_recommendation",
                "content": _reuse_summary_md(rec),
                "recommendation": rec,
            })
            await _persist_chat(chat)
            return {"status": "reuse_recommended", "recommendation": rec}

    chat.simulating = True
    # The event loop keeps only a weak ref to tasks — hold one until done.
    task = asyncio.create_task(
        _run_forward_model(chat, manifest, in_dir, out_dir / "out_files", filenames)
    )
    _simulation_tasks.add(task)
    task.add_done_callback(_simulation_tasks.discard)
    return {"status": "started", "total": len(filenames)}


_simulation_tasks: set[asyncio.Task] = set()


async def _run_forward_model(
    chat: ChatSession,
    manifest: dict[str, Any],
    in_dir: Path,
    out_dir: Path,
    filenames: list[str],
) -> None:
    """Run the gprMax batch in a worker thread, streaming per-file progress
    to the session's CURRENT socket. The chat stays usable meanwhile — the
    run only reads the emitted files and never touches the pipeline state."""
    loop = asyncio.get_running_loop()

    def on_progress(event: dict[str, Any]) -> None:
        # Called from the worker thread — marshal onto the event loop.
        asyncio.run_coroutine_threadsafe(
            _send(chat, {"type": "simulation_progress", **event}), loop
        )

    try:
        if manifest.get("contract"):
            from backend.dataset_sampling.contract import validate_release_access
            validate_release_access(DatasetConfig.model_validate(manifest["contract"]["requested"]["dataset_config"]))
        result = await asyncio.to_thread(
            run_batch_simulation,
            input_dir=in_dir,
            output_dir=out_dir,
            filenames=filenames,
            progress=on_progress,
            **({"manifest": manifest} if manifest.get("contract") else {}),
        )
    except Exception as exc:
        logger.exception("forward model failed for session %s", chat.session_id)
        chat.simulating = False
        await _send(chat, {
            "type": "simulation_complete",
            "content": f"Forward model failed to start: {exc}",
            "result": {"succeeded": 0, "failed": 0, "skipped": 0, "total": 0,
                       "errors": [{"error": str(exc)}]},
        })
        await _persist_chat(chat)
        return

    rows_updated, signals_updated = await asyncio.to_thread(
        _record_simulation_outputs, chat, manifest, result
    )
    summary = {
        "succeeded": result["succeeded"],
        "failed": result["failed"],
        "skipped": result["skipped"],
        "total": result["total"],
        "output_dir": result["output_dir"],
        "rows_updated": rows_updated,
        "signals_updated": signals_updated,
        # Which solver backend actually ran (env-resolved in simulate.py) —
        # persisted so a restored session still shows how it was produced.
        "mode": result.get("mode", "cpu"),
        "workers": result.get("workers", 1),
        # Full tracebacks stay in the server log; the chat gets one line each.
        "errors": [
            {"filename": e["filename"],
             "error": e["error"].strip().splitlines()[-1]}
            for e in result.get("errors", [])
        ],
    }
    backend_note = (
        f"{summary['mode'].upper()} solver, {summary['workers']} model(s) in parallel"
    )
    content = (
        f"Forward model complete — {summary['succeeded']} succeeded, "
        f"{summary['failed']} failed, {summary['skipped']} skipped "
        f"(of {summary['total']}; {backend_note}). Output written to "
        f"{str(out_dir).replace(_project_root + '/', '')}."
    )
    chat.simulation_result = summary
    chat.simulating = False
    await _send(chat, {
        "type": "simulation_complete",
        "content": content,
        "result": summary,
    })
    # Index this session's config for future reuse recommendations — only a
    # fully successful generated run represents a complete, adoptable dataset.
    # Never breaks the run (suppress + failure-swallowing inside).
    if (
        manifest.get("source") != "upload"
        and result["total"] > 0
        and result["failed"] == 0
    ):
        with contextlib.suppress(Exception):
            await asyncio.to_thread(
                sim_similarity.index_completed_session,
                dict(chat.state),
                session_id=str(_coerce_uuid(chat.session_id)),
                user_id=chat.user_id,
                num_samples=result["total"],
                output_dir=str(out_dir.parent),
            )
    # Runs outside any user turn — persist the summary + transcript event.
    await _persist_chat(chat)


def _record_simulation_outputs(
    chat: ChatSession,
    manifest: dict[str, Any],
    result: dict[str, Any],
) -> tuple[int, int]:
    """Persist each sample's .out path AND its extracted signal arrays onto
    its Simulation rows; returns (rows_updated, signals_updated). DB/HDF5
    failures are logged and swallowed — the simulations themselves already
    ran. The signal arrays are what make this dataset adoptable by future
    sessions (reuse copies them instead of re-simulating)."""
    if manifest.get("source") == "upload":
        # Uploaded decks have no Simulation rows; their positional sample ids
        # must not overwrite a previously generated dataset's rows.
        return (0, 0)
    sample_by_in = {
        f["filename"]: int(f["sample_id"])
        for f in manifest.get("files", [])
        if f.get("filename") and f.get("sample_id") is not None
    }
    outputs = {
        sample_by_in[o["filename"]]: o["out_file"]
        for o in result.get("outputs", [])
        if o["filename"] in sample_by_in
    }
    if not outputs:
        return (0, 0)
    session_uuid = _coerce_uuid(chat.session_id)
    try:
        rows_updated = set_simulation_outputs(session_uuid, outputs)
    except Exception:
        logger.exception(
            "recording simulation outputs failed for session %s", chat.session_id
        )
        return (0, 0)
    signals_updated = 0
    try:
        # Lazy import keeps h5py off the api import path.
        from signal_extraction import extract_and_prepare_batch

        batch = extract_and_prepare_batch(Path(result["output_dir"]), session_uuid,
                                          manifest=manifest, outputs=result.get("outputs", []))
        if batch.get("updates"):
            signals_updated = bulk_update_signals(batch["updates"])
    except Exception:
        logger.exception(
            "signal extraction failed for session %s", chat.session_id
        )
    return (rows_updated, signals_updated)


# ---------------------------------------------------------------------------
# Forward-model reuse: recommend a >=threshold-similar past session's dataset
# and, on user agreement, adopt it wholesale (files + Simulation rows +
# signals) so nothing about the current session is desynchronized.
# ---------------------------------------------------------------------------


def _find_reuse_candidate(chat: ChatSession) -> Optional[dict[str, Any]]:
    """Best ADOPTABLE >=threshold match for this session's config, or None.
    Sync (runs in a worker thread); wraps everything — a broken similarity
    stack must never block the forward model."""
    try:
        sid = str(_coerce_uuid(chat.session_id))
        matches = sim_similarity.find_similar_session(dict(chat.state), session_id=sid)
        for match in matches:
            if _reuse_candidate_adoptable(match):
                cfg = chat.state.get("dataset_config") or {}
                match["requested_samples"] = cfg.get("num_samples")
                return match
    except Exception:
        logger.exception(
            "reuse-candidate lookup failed for session %s", chat.session_id
        )
    return None


def _reuse_candidate_adoptable(match: dict[str, Any]) -> bool:
    """Stale index points must never surface: the source session's rows must
    all be simulated and its dataset directory still intact on disk."""
    try:
        src_uuid = _coerce_uuid(str(match.get("source_session_id")))
        total, incomplete = count_incomplete_simulations(src_uuid)
        if total == 0 or incomplete > 0:
            return False
        src_dir = Path(str(match.get("source_output_dir") or ""))
        return (
            src_dir.is_dir()
            and (src_dir / "emitted_files.json").is_file()
            and (src_dir / "out_files").is_dir()
        )
    except Exception:
        return False


def _reuse_summary_md(rec: dict[str, Any]) -> str:
    sid = str(rec.get("source_session_id") or "")
    when = str(rec.get("simulated_at") or "")[:10]
    requested = rec.get("requested_samples")
    n = rec.get("num_samples")
    count_note = f"{n} simulated samples"
    if requested is not None and n is not None and int(n) != int(requested):
        count_note += f" (you requested {requested})"
    lines = [
        f"**Found an existing dataset {rec['similarity_pct']}% similar to this "
        f"configuration** — session `{sid[:8]}` by `{rec.get('source_user_id')}`, "
        f"{count_note}, simulated {when or 'earlier'}.",
        "",
        "You can **reuse its samples and signals** instead of re-running the "
        "forward model, or simulate anyway (buttons next to the Run control).",
    ]
    diffs = rec.get("params_diff") or []
    if diffs:
        lines += ["", "Closest differences:"]
        for d in diffs[:5]:
            lines.append(
                f"- `{d['param']}`: yours {d['current']} vs theirs "
                f"{d['candidate']} ({round(d['sim'] * 100)}% match)"
            )
    return "\n".join(lines)


class AdoptDatasetPayload(BaseModel):
    source_session_id: str


@app.post("/datasets/{session_id}/adopt")
async def adopt_dataset(session_id: str, payload: AdoptDatasetPayload) -> dict[str, Any]:
    """Execute a pending reuse recommendation: replace this session's dataset
    (files, manifests, Simulation rows, signals) with a copy of the source
    session's. Only ever adopts the exact recommendation the simulate gate
    issued — never an arbitrary session id."""
    chat = await _resolve_chat(session_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Unknown session")
    if chat.simulating:
        raise HTTPException(
            status_code=409, detail="Forward model is running — wait for it to finish"
        )
    if chat.busy:
        raise HTTPException(
            status_code=409,
            detail="A pipeline step is still running — wait for it to finish",
        )
    if chat.regenerating:
        raise HTTPException(
            status_code=409, detail="Dataset is regenerating — wait for it to finish"
        )
    rec = chat.reuse_recommendation
    if not rec or str(rec.get("source_session_id")) != payload.source_session_id:
        raise HTTPException(
            status_code=409,
            detail="No matching reuse recommendation is pending for this session",
        )

    chat.simulating = True  # same 409 umbrella as a real run for simulate/upload
    try:
        result = await asyncio.to_thread(_adopt_dataset_sync, chat, rec)
    finally:
        chat.simulating = False
    await _send(chat, {
        "type": "dataset_ready",
        "content": (
            f"Adopted the {rec['similarity_pct']}% similar dataset from session "
            f"`{payload.source_session_id[:8]}` — {result['num_generated']} "
            "sample(s) with their simulated signals now belong to this session. "
            "No forward-model run was needed."
        ),
        "result": chat.dataset_result,
    })
    await _send(chat, {
        "type": "simulation_complete",
        "content": (
            f"Forward model skipped — reused {result['num_generated']} existing "
            "simulated output(s). The outcomes are ready in the Dataset tab."
        ),
        "result": chat.simulation_result,
    })
    await _send_model_update(chat, stage="Dataset Adoption")
    await _persist_chat(chat)
    return result


def _adopt_dataset_sync(chat: ChatSession, rec: dict[str, Any]) -> dict[str, Any]:
    """Copy the source session's dataset into this session. ALL verification
    happens before anything is deleted — a failing precondition leaves the
    current dataset untouched."""
    source_session_id = str(rec["source_session_id"])
    src_uuid = _coerce_uuid(source_session_id)
    cur_uuid = _coerce_uuid(chat.session_id)
    if src_uuid == cur_uuid:
        raise HTTPException(status_code=409, detail="Cannot adopt this session's own dataset")

    cfg_dict = chat.state.get("dataset_config")
    if not cfg_dict:
        raise HTTPException(status_code=409, detail="Session has no dataset configuration")
    cfg = DatasetConfig.model_validate(cfg_dict)
    cur_dir = _resolve_dataset_path(cfg.output_dir)

    # --- verify the source end to end (DB rows + files) -------------------
    src_dir = Path(str(rec.get("source_output_dir") or ""))
    src_row = get_extraction_session(src_uuid)
    if src_row is not None:
        recorded = ((src_row.model_config_data or {}).get("artifacts") or {}).get(
            "output_dir"
        )
        if recorded:
            src_dir = Path(recorded)
    manifest_path = src_dir / "emitted_files.json"
    if not src_dir.is_dir() or not manifest_path.is_file():
        raise HTTPException(
            status_code=409, detail="Source dataset directory no longer exists"
        )
    src_manifest = _read_json(manifest_path)
    if cfg.contract_version >= 2 or src_manifest.get("contract"):
        current = sim_similarity.current_manifest(str(cur_dir))
        source = sim_similarity.eligible_manifest(str(src_dir))
        if current is None or source is None or not sim_similarity.equivalent_contracts(current, source):
            raise HTTPException(status_code=409, detail="Reuse requires qualified outputs and exact compatible experiment/population contracts")
    if src_manifest.get("source") == "upload":
        raise HTTPException(
            status_code=409, detail="Source dataset is an upload — nothing to reuse"
        )
    src_in_dir = _in_files_dir(src_dir, src_manifest)
    src_out_dir = _out_files_dir(src_dir)
    files = [f for f in src_manifest.get("files", []) if f.get("filename")]
    if not files:
        raise HTTPException(status_code=409, detail="Source dataset has no input files")
    for f in files:
        name = Path(f["filename"]).name
        if not (src_in_dir / name).is_file() or not (
            src_out_dir / (Path(name).stem + ".out")
        ).is_file():
            raise HTTPException(
                status_code=409,
                detail=f"Source dataset is missing files for {name}",
            )
    src_rows = get_simulations_for_session(src_uuid)
    if not src_rows or any(r.simulation_completed_at is None for r in src_rows):
        raise HTTPException(
            status_code=409,
            detail="Source session's simulations are not fully complete",
        )

    # --- replace files (the adopted dataset REPLACES the drawn one) --------
    cur_in_dir = cur_dir / "in_files"
    cur_out_dir = _out_files_dir(cur_dir)
    shutil.rmtree(cur_in_dir, ignore_errors=True)
    shutil.rmtree(cur_out_dir, ignore_errors=True)
    cur_in_dir.mkdir(parents=True, exist_ok=True)
    cur_out_dir.mkdir(parents=True, exist_ok=True)
    new_files = []
    for f in files:
        name = Path(f["filename"]).name
        shutil.copy2(src_in_dir / name, cur_in_dir / name)
        out_name = Path(name).stem + ".out"
        shutil.copy2(src_out_dir / out_name, cur_out_dir / out_name)
        for suffix in (".execution.json", ".geometry.h5"):
            extra = src_out_dir / (Path(name).stem + suffix)
            if extra.exists():
                shutil.copy2(extra, cur_out_dir / extra.name)
        for snapshot in (f.get("resolved_scene") or {}).get("snapshots", []):
            relative = Path(Path(name).stem + "_snaps") / (snapshot["filename"] + ".vti")
            (cur_out_dir / relative).parent.mkdir(exist_ok=True)
            shutil.copy2(src_out_dir / relative, cur_out_dir / relative)
        entry = dict(f)
        entry["path"] = str(cur_in_dir / name)
        new_files.append(entry)
    # These manifests contain no absolute paths — copied verbatim they make
    # the samples viz and any regeneration bookkeeping consistent on disk.
    for m in ("sampled_layers.json", "derived_layers.json", "global_derive.json", "dataset_contract.json", "dropped_targets.json", "qualification.json"):
        if (src_dir / m).is_file():
            shutil.copy2(src_dir / m, cur_dir / m)
    new_manifest = dict(src_manifest)
    new_manifest.update({
        "output_dir": str(cur_dir),
        "in_dir": str(cur_in_dir),
        "files": new_files,
        "adopted_from": source_session_id,
    })
    with open(cur_dir / "emitted_files.json", "w", encoding="utf-8") as fh:
        json.dump(new_manifest, fh, indent=2)

    # --- replace Simulation rows (signals ride along) -----------------------
    row_dicts = []
    for r in src_rows:
        d = r.model_dump()
        d["id"] = uuid.uuid4()
        d["session_id"] = cur_uuid
        d["user_id"] = chat.user_id
        d["created_at"] = datetime.now(timezone.utc)
        d["output_dir"] = cfg.output_dir
        in_name = Path(r.input_file_path).name if r.input_file_path else None
        if in_name:
            d["input_file_path"] = str(cur_in_dir / in_name)
        out_name = Path(r.output_file_path).name if r.output_file_path else None
        if out_name:
            d["output_file_path"] = str(cur_out_dir / out_name)
        row_dicts.append(d)
    delete_simulations_for_session(cur_uuid)
    inserted = batch_insert_simulations(row_dicts) if row_dicts else 0
    # Healing: sessions simulated before signal extraction landed have .out
    # paths but no arrays — extract from the freshly copied outputs.
    if any(r.signal_length is None for r in src_rows):
        try:
            from signal_extraction import extract_and_prepare_batch

            batch = extract_and_prepare_batch(cur_out_dir, cur_uuid)
            if batch.get("updates"):
                bulk_update_signals(batch["updates"])
        except Exception:
            logger.exception(
                "signal healing failed after adoption for session %s",
                chat.session_id,
            )

    # --- session state -------------------------------------------------------
    result = {
        "status": "adopted",
        "session_id": chat.session_id,
        "adopted_from": source_session_id,
        "similarity_pct": rec.get("similarity_pct"),
        "rows_inserted": inserted,
        "num_generated": len(new_files),
        "output_dir": str(cur_dir),
        "in_dir": str(cur_in_dir),
        "files": [
            {"sample_id": f.get("sample_id"), "filename": f.get("filename")}
            for f in new_files
        ],
    }
    chat.dataset_result = result
    chat.simulation_result = {
        "succeeded": len(new_files),
        "failed": 0,
        "skipped": 0,
        "total": len(new_files),
        "output_dir": str(cur_out_dir),
        "rows_updated": inserted,
        "adopted_from": source_session_id,
        "errors": [],
    }
    chat.uploaded_output_dir = None
    chat.reuse_recommendation = None
    # The copied manifests are now this session's products — the scene
    # projection may read all of them.
    chat.viz_flags.update({k: True for k in chat.viz_flags})
    return result


@app.websocket("/ws/{user_id}/{session_id}")
async def websocket_endpoint(ws: WebSocket, user_id: str, session_id: str) -> None:
    await ws.accept()
    try:
        uid = _validate_user_id(user_id)
    except HTTPException:
        await ws.close(code=4400, reason="invalid user_id")
        return
    # In-memory chat, hydrated from the DB (restart / other browser), or brand
    # new (client-minted id — legacy path; new chats normally come from
    # POST /users/{uid}/chats which stubs the row first).
    state = await _resolve_chat(session_id) or sessions.setdefault(
        session_id, _new_chat_session(session_id, uid)
    )
    # Route ALL pipeline output (including a turn already in flight) here.
    state.ws = ws

    try:
        if not state.started:
            state.started = True
            async with state.lock:
                await _send(state, {
                    "type": "agent_message",
                    "content": "Connected to the single-agent dataset pipeline.",
                })
                await _start_stage(state, "dataset_config")
                # Persist AFTER the kickoff: the LLM thread now holds the
                # kickoff message, so a restart never replays it.
                await _persist_chat(state)
                # Converge the client busy flag (a refresh during this first
                # kickoff restores `busy: true` on the new socket).
                await _send(state, {"type": "pipeline_busy", "busy": False})
        else:
            # Refresh/reconnect: restore the whole UI state in one shot —
            # full chat transcript, dataset files, busy flag, then the canvas.
            # Sent WITHOUT the lock so a long-running step can't delay it.
            await ws.send_json({
                "type": "session_restore",
                "events": list(state.transcript),
                "busy": state.busy,
                "phase": state.phase,
                "complete": state.complete,
                "dataset": state.dataset_result,
                "simulating": state.simulating,
                "simulation": state.simulation_result,
                "reuse": state.reuse_recommendation,
            })
            if state.last_scene:
                await ws.send_json({"type": "model_update", "scene": state.last_scene})

        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            if msg.get("type") not in {"user_message", "choice_response"}:
                continue
            content = (msg.get("content") or msg.get("choice") or "").strip()
            if not content:
                continue
            async with state.lock:
                await _handle_user_text(state, content)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for session %s", session_id)
    except Exception as exc:
        logger.exception("WebSocket error for session %s", session_id)
        with contextlib.suppress(Exception):
            await ws.send_json({"type": "error", "message": str(exc)})
    finally:
        # Only clear the routing target if a newer socket hasn't replaced it.
        if state.ws is ws:
            state.ws = None
        # Safety-net persist; may capture a mid-turn snapshot (hydration
        # coerces busy/simulating/phase back to a consistent state).
        await _persist_chat(state)


def _new_chat_session(session_id: str, user_id: Optional[str] = None) -> ChatSession:
    user_id = user_id or session_id
    return ChatSession(
        session_id=session_id,
        user_id=user_id,
        agent_session=SingleAgentSession(
            user_id=user_id,
            session_id=session_id,
            checkpointer_factory=get_checkpointer,
        ),
    )


# ---------------------------------------------------------------------------
# Chat persistence: every ChatSession JSON field rides in one JSONB blob so a
# chat survives browser close AND backend restart. The LLM message history is
# NOT here — it lives in the LangGraph Postgres checkpointer, keyed by the
# persisted thread_id.
# ---------------------------------------------------------------------------

_PERSISTED_CHAT_FIELDS = (
    "started", "complete", "busy", "simulating", "regenerating",
    "post_complete_briefed", "phase", "active_purpose", "active_section",
    "remediation_snapshot", "complete_snapshot", "viz_flags", "last_scene",
    "transcript", "dataset_result", "simulation_result", "uploaded_output_dir",
    "reuse_recommendation", "state",
)


def _chat_title(chat: ChatSession) -> str:
    cfg = chat.agent_session.store.get("dataset_config") or {}
    if cfg.get("model_basename"):
        return str(cfg["model_basename"])[:60]
    for ev in chat.transcript:
        if ev.get("type") == "user_message" and ev.get("content"):
            return str(ev["content"])[:60]
    return "New chat"


def _chat_row_payload(chat: ChatSession) -> dict[str, Any]:
    blob: dict[str, Any] = {f: getattr(chat, f) for f in _PERSISTED_CHAT_FIELDS}
    blob["regen_block_notice"] = (
        sorted(chat.regen_block_notice) if chat.regen_block_notice else None
    )
    blob["store"] = chat.agent_session.store
    blob["seen"] = chat.agent_session.seen
    return {
        "session_id": chat.session_id,
        "user_id": chat.user_id,
        "title": _chat_title(chat),
        "thread_id": chat.agent_session.thread_id,
        "complete": chat.complete,
        "has_dataset": bool(chat.dataset_result),
        "session_state": blob,
    }


async def _persist_chat(chat: ChatSession) -> None:
    """Whole-row rewrite of the chat's durable state. Failure-swallowed —
    persistence must never break the chat loop (mirrors _send_model_update)."""
    try:
        payload = _chat_row_payload(chat)
        await asyncio.to_thread(upsert_chat_session, **payload)
    except Exception:
        logger.exception("persisting chat session %s failed", chat.session_id)


def _chat_from_row(row) -> ChatSession:
    """Rebuild a live ChatSession from its persisted row (backend restart)."""
    chat = ChatSession(
        session_id=row.id,
        user_id=row.user_id,
        agent_session=SingleAgentSession(
            user_id=row.user_id,
            session_id=row.id,
            thread_id=row.thread_id,  # durable checkpointer resumes this thread
            checkpointer_factory=get_checkpointer,
        ),
    )
    blob = row.session_state or {}
    for f in _PERSISTED_CHAT_FIELDS:
        if f in blob:
            setattr(chat, f, blob[f])
    # In-flight flags can never be true after a process restart; a
    # disconnect-persist may also have captured a mid-turn phase.
    chat.busy = False
    chat.simulating = False
    if chat.phase in {"deterministic", "routing"}:
        chat.phase = "complete" if chat.complete else "agent"
    rb = blob.get("regen_block_notice")
    chat.regen_block_notice = frozenset(rb) if rb else None
    # Mutate the store IN PLACE — save_section/get_section are closures over
    # this exact dict object; rebinding would orphan the tools.
    for k, v in (blob.get("store") or {}).items():
        chat.agent_session.store[k] = v
    chat.agent_session.seen = int(blob.get("seen") or 0)
    return chat


async def _resolve_chat(session_id: str) -> Optional[ChatSession]:
    """In-memory chat, or hydrate it from the DB, or None if unknown."""
    chat = sessions.get(session_id)
    if chat is not None:
        return chat
    try:
        row = await asyncio.to_thread(get_chat_session, session_id)
    except Exception:
        logger.exception("chat hydration lookup failed for %s", session_id)
        return None
    if row is None:
        return None
    return sessions.setdefault(session_id, _chat_from_row(row))


def _resolve_chat_sync(session_id: str) -> Optional[ChatSession]:
    """Sync twin of _resolve_chat for endpoints running in the threadpool."""
    chat = sessions.get(session_id)
    if chat is not None:
        return chat
    try:
        row = get_chat_session(session_id)
    except Exception:
        logger.exception("chat hydration lookup failed for %s", session_id)
        return None
    if row is None:
        return None
    return sessions.setdefault(session_id, _chat_from_row(row))


_USER_ID_ALNUM_RE = re.compile(r"[A-Za-z0-9]")


def _validate_user_id(raw: str) -> str:
    """Plain-string identity (no auth): trimmed, 1-64 chars, at least one
    letter/digit. The raw value is stored in the DB (ORM parametrizes); only
    the _dataset_dirname-sanitized form ever touches the filesystem."""
    value = (raw or "").strip()
    if not (1 <= len(value) <= 64) or not _USER_ID_ALNUM_RE.search(value):
        raise HTTPException(
            status_code=422,
            detail="user_id must be 1-64 characters with at least one letter or digit",
        )
    return value


@app.get("/users/{user_id}/chats")
async def list_user_chats(user_id: str) -> dict[str, Any]:
    """The DB-backed chat list — the same user id on any browser recovers it."""
    uid = _validate_user_id(user_id)
    chats = await asyncio.to_thread(list_chat_sessions, uid)
    return {"chats": chats}


@app.post("/users/{user_id}/chats")
async def create_user_chat(user_id: str) -> dict[str, Any]:
    """Mint a new chat: server-side session_id + thread_id, stub row inserted
    so the chat is listable before its first WebSocket connect."""
    uid = _validate_user_id(user_id)
    session_id = str(uuid.uuid4())
    thread_id = f"single-agent-{uuid.uuid4()}"
    await asyncio.to_thread(create_chat_stub, session_id, uid, thread_id)
    return {"session_id": session_id}


async def _handle_user_text(chat: ChatSession, text: str) -> None:
    # Record only — the client already rendered its own message locally;
    # the transcript copy is for the reconnect replay.
    chat.transcript.append({"type": "user_message", "content": text})
    try:
        if chat.busy:
            await _send(chat, {
                "type": "agent_message",
                "content": "A pipeline step is still running. Please wait for it to finish.",
            })
            return
        # "complete" still chats: the agent can answer questions about the
        # finished dataset — only pipeline advancement stops.
        if chat.phase not in {"agent", "complete"}:
            await _send(chat, {
                "type": "agent_message",
                "content": "The pipeline is not waiting for a chat reply right now.",
            })
            return
        if chat.phase == "agent" and text.lower() in {"quit", "exit"}:
            chat.phase = "halted"
            chat.state["halted"] = True
            chat.state["halt_reason"] = f"user exited during {chat.active_purpose}"
            await _send(chat, {"type": "agent_message", "content": "Pipeline halted."})
            return

        try:
            await _invoke_and_handle(chat, text)
        finally:
            # The turn (agent + any deterministic chain) is over; converge the
            # client's busy flag — needed when a refresh restored `busy: true`.
            await _send(chat, {"type": "pipeline_busy", "busy": False})
    except Exception:
        logger.exception(
            "chat turn failed for session %s; keeping websocket alive",
            chat.session_id,
        )
        chat.busy = False
        if chat.phase in {"deterministic", "routing"}:
            chat.phase = "complete" if chat.complete else "agent"
        await _send(chat, {
            "type": "agent_message",
            "content": (
                "I hit a backend problem while processing that turn. The chat "
                "is still open; adjust the last simulation values and send "
                "them again so I can continue."
            ),
        })
        await _send(chat, {"type": "pipeline_busy", "busy": False})
    finally:
        # Durable write once per turn — covers collection, remediation,
        # regeneration, _finish_dataset and the halt branch alike.
        await _persist_chat(chat)


async def _invoke_and_handle(chat: ChatSession, message: str) -> None:
    """Send `message` into the ONE ongoing conversation and process the turn."""
    chat.busy = True
    try:
        result = await asyncio.to_thread(chat.agent_session.invoke, message)
        await _handle_agent_result(chat, result)
    finally:
        chat.busy = False


async def _start_stage(chat: ChatSession, section: str) -> None:
    """Kick off a collection stage by injecting its section-specific
    instructions (batches, constraints, JSON schema) into the conversation."""
    chat.phase = "agent"
    chat.active_purpose = "collect"
    chat.active_section = section

    await _send(chat, {
        "type": "stage_change",
        "stage_name": SECTION_DISPLAY[section],
        "section": section,
    })
    await _invoke_and_handle(chat, SECTION_KICKOFF[section])


async def _send_model_update(
    chat: ChatSession, *, stage: Optional[str] = None
) -> None:
    """Project the current store/manifests into a scene and push it to the
    canvas. Display-only: any failure is logged and swallowed so the viz can
    never break the chat loop. Identical scenes are deduplicated."""
    try:
        out_dir: Optional[str] = None
        cfg_dict = chat.agent_session.store.get("dataset_config")
        if cfg_dict:
            cfg = DatasetConfig.model_validate(cfg_dict)
            out_dir = str(_resolve_dataset_path(cfg.output_dir))
        scene = await asyncio.to_thread(
            build_scene, chat.agent_session.store, chat.viz_flags, out_dir, stage
        )
    except Exception:
        logger.exception("viz projection failed for session %s", chat.session_id)
        return
    if scene is None or scene == chat.last_scene:
        return
    chat.last_scene = scene
    await _send(chat, {"type": "model_update", "scene": scene})


# Deterministic node -> viz flag updates. layer_sampling RESETS the downstream
# flags: a re-sample (staleness / global remediation) invalidates every later
# manifest until its node runs again.
_VIZ_FLAG_UPDATES: list[tuple[Any, dict[str, bool]]] = [
    (layer_sampling_node, {"sampled": True, "derived": False, "grid": False,
                           "placed": False, "emitted": False}),
    (peplinski_derive_node, {"derived": True}),
    (global_derive_node, {"grid": True}),
    (target_placement_node, {"placed": True}),
    (dataset_generation_node, {"emitted": True}),
]


async def _handle_agent_result(chat: ChatSession, result: dict) -> None:
    for text in chat.agent_session.new_ai_texts(result):
        await _send(chat, {"type": "agent_message", "content": text})

    await _send_model_update(
        chat, stage=chat.active_section or chat.active_purpose
    )

    if chat.complete:
        # Post-completion chat: relay only unless the agent edited a section
        # (user-confirmed edit) — then the dataset regenerates synchronously.
        await _check_regeneration(chat)
        return
    if chat.active_purpose == "collect":
        if not await _maybe_resample_stale(chat):
            return
        await _check_collect_done(chat)
    else:
        await _check_remediation_done(chat)


async def _maybe_resample_stale(chat: ChatSession) -> bool:
    """Eagerly re-draw samples when a cross-edit changed a sampling input
    after layer_sampling ran (e.g. the user adds the buried target during
    the waveform stage). Without this the stale draws sit on disk and on
    the canvas until the deferred check in _run_derive_chain fires after
    advanced_params."""
    snap = chat.state.get("sampling_snapshot")
    if snap is None:
        return True  # sampling hasn't run yet
    store = chat.agent_session.store
    changed = {s for s in SAMPLING_INPUT_SECTIONS if store.get(s) != snap.get(s)}
    if not changed:
        return True
    if not all(chat.agent_session.stage_done(s) for s in changed):
        return True  # mid-edit; _run_derive_chain's deferred check still covers it
    _sync_sections(chat)
    ok = await _run_layer_sampling_or_remediate(
        chat,
        {
            "kind": "return_to_agent",
            "active_purpose": chat.active_purpose,
            "active_section": chat.active_section,
        },
    )
    if not ok:
        return False
    # Still mid-collection on the active section: _run_deterministic left the
    # phase at "deterministic", which would reject the user's next chat turn.
    chat.phase = "agent"
    return True


async def _check_regeneration(chat: ChatSession) -> None:
    """Post-completion turns: diff the store against the snapshot taken at the
    last successful generation. No change => pure discussion (relay only). A
    change with every changed section complete => regenerate the dataset. An
    incomplete change NEVER touches the existing dataset — the diff persists,
    so regeneration fires on the turn that completes the section."""
    changed = _changed_sections(
        chat.complete_snapshot or {}, chat.agent_session.snapshot()
    )
    if not changed:
        return
    if chat.simulating:
        # Never re-emit under a running forward model; the persistent diff
        # re-triggers on the next turn after it finishes.
        await _send(chat, {
            "type": "agent_message",
            "content": (
                "The forward model is still running — the dataset will "
                "regenerate after it finishes (send any message then)."
            ),
        })
        return
    incomplete = frozenset(
        s for s in changed if not chat.agent_session.stage_done(s)
    )
    if incomplete:
        if incomplete != chat.regen_block_notice:  # notify once per set
            chat.regen_block_notice = incomplete
            await _send(chat, {
                "type": "progress",
                "content": (
                    f"Section(s) {', '.join(sorted(incomplete))} incomplete — "
                    "the existing dataset is kept until they are complete."
                ),
            })
        return
    chat.regen_block_notice = None
    await _start_regeneration(chat, changed)


async def _start_regeneration(chat: ChatSession, changed: set) -> None:
    """Synchronously re-run the deterministic tail after a post-complete edit.
    Same node order as the first run: gate -> (resample iff a sampling input
    changed, inside _run_derive_chain) -> derive chain -> placement ->
    emission -> finalize -> dataset_ready."""
    chat.complete = False  # remediation routing needs this off
    chat.regenerating = True
    chat.phase = "routing"
    _sync_sections(chat)
    await _send(chat, {
        "type": "progress",
        "content": (
            f"Updated {', '.join(sorted(changed))} — re-validating and "
            "regenerating the dataset."
        ),
    })
    await _run_sample_validation_gate(chat)


async def _check_collect_done(chat: ChatSession) -> None:
    section = chat.active_section
    if section is None or not chat.agent_session.stage_done(section):
        return  # keep collecting — wait for the next user message

    _sync_sections(chat)
    await _send(chat, {
        "type": "progress",
        "content": f"{SECTION_DISPLAY[section]} complete.",
        "section": section,
    })
    chat.active_section = None
    chat.phase = "routing"
    await _advance_after_collect(chat, section)


async def _check_remediation_done(chat: ChatSession) -> None:
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
    progress_tail = (
        "re-running layer sampling."
        if purpose == "sampling_remediation"
        else "re-validating."
    )
    await _send(chat, {
        "type": "progress",
        "content": f"Updated {', '.join(sorted(changed))} — {progress_tail}",
    })

    if purpose == "sample_remediation":
        await _run_sample_validation_gate(chat)
    elif purpose == "sampling_remediation":
        resume = chat.state.pop("sampling_remediation_resume", None) or {
            "kind": "start_stage",
            "section": "waveform",
        }
        ok = await _run_layer_sampling_or_remediate(chat, resume)
        if ok:
            await _resume_after_layer_sampling(chat, resume)
    else:
        chat.state["resample_after_global"] = bool(changed & RESAMPLE_SECTIONS)
        await _run_derive_chain(chat)


def _sync_sections(chat: ChatSession) -> None:
    """Copy the WHOLE store into the pipeline state, so cross-section edits
    made during any stage land in state as soon as that turn completes."""
    chat.state.update(chat.agent_session.state_sync())


async def _advance_after_collect(chat: ChatSession, section: str) -> None:
    if section == "dataset_config":
        await _start_stage(chat, "layers")
    elif section == "layers":
        await _start_stage(chat, "target_ranges")
    elif section == "target_ranges":
        ok = await _run_layer_sampling_or_remediate(
            chat, {"kind": "start_stage", "section": "waveform"}
        )
        if ok:
            await _start_stage(chat, "waveform")
    elif section == "waveform":
        await _start_stage(chat, "antenna")
    elif section == "antenna":
        await _run_sample_validation_gate(chat)
    elif section == "advanced_params":
        await _run_derive_chain(chat)


async def _run_sample_validation_gate(chat: ChatSession) -> None:
    await _run_deterministic(chat, "Sample Validation", sample_validation_node)
    if chat.state.get("sample_validation_passed"):
        if chat.regenerating:
            # Post-complete regeneration: advanced_params was already
            # collected — go straight to the derive chain (which re-samples
            # first iff a sampling input changed).
            await _run_derive_chain(chat)
        else:
            await _start_stage(chat, "advanced_params")
        return

    errors = chat.state.get("sample_validation_errors") or []
    await _send(chat, {
        "type": "validation_failed",
        "stage_name": "Sample Validation",
        "errors": errors,
    })
    chat.state["sample_validation_passed"] = None
    chat.state["sample_validation_errors"] = None
    await _start_remediation(
        chat, "sample_remediation",
        sample_remediation_message(errors, chat.agent_session.store),
    )


async def _start_remediation(
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
    await _invoke_and_handle(chat, message)


async def _run_layer_sampling_or_remediate(
    chat: ChatSession,
    resume: dict[str, Any],
) -> bool:
    """Run layer/target sampling, or route data-related failures back into
    the single-agent remediation loop without breaking the websocket turn.

    `resume` records where the pipeline should continue after the user and
    agent correct the sampling inputs."""
    try:
        await _run_deterministic(chat, "Layer + Target Sampling", layer_sampling_node)
        return True
    except Exception as exc:
        logger.exception(
            "layer sampling failed for session %s; routing to remediation",
            chat.session_id,
        )
        errors = _sampling_failure_errors(exc)
        chat.state["sampling_remediation_resume"] = dict(resume or {})
        await _send(chat, {
            "type": "validation_failed",
            "stage_name": "Layer + Target Sampling",
            "errors": errors,
        })
        await _start_remediation(
            chat,
            "sampling_remediation",
            layer_sampling_remediation_message(errors, chat.agent_session.store),
        )
        return False


async def _resume_after_layer_sampling(
    chat: ChatSession,
    resume: dict[str, Any],
) -> None:
    """Continue the path that was interrupted by sampling remediation."""
    kind = (resume or {}).get("kind")
    if kind == "start_stage":
        await _start_stage(chat, str(resume.get("section") or "waveform"))
        return
    if kind == "derive_chain":
        await _run_derive_chain(chat)
        return
    if kind == "return_to_agent":
        chat.active_purpose = str(resume.get("active_purpose") or "collect")
        chat.active_section = resume.get("active_section")
        chat.phase = "agent"
        if chat.active_purpose == "collect":
            await _check_collect_done(chat)
        return
    # Defensive default: the first sampling pass normally resumes at waveform.
    await _start_stage(chat, "waveform")


async def _run_derive_chain(chat: ChatSession) -> None:
    """Re-sample if needed, then derive the grid and run the global gate."""
    if chat.state.pop("resample_after_global", None) or _samples_stale(chat.state):
        ok = await _run_layer_sampling_or_remediate(
            chat, {"kind": "derive_chain"}
        )
        if not ok:
            return

    chat.state["global_validation_passed"] = False
    try:
        await _run_deterministic(chat, "Peplinski Derive", peplinski_derive_node)
        await _run_deterministic(chat, "Global Derive", global_derive_node)
        await _run_deterministic(chat, "Global Validation", global_validation_node)
    except ValueError as exc:
        chat.state["global_validation_errors"] = [f"[deterministic_planning] {exc}"]
        chat.viz_flags.update(grid=False, placed=False, emitted=False)

    if chat.state.get("global_validation_passed"):
        await _finish_dataset(chat)
        return

    errors = chat.state.get("global_validation_errors") or []
    await _send(chat, {
        "type": "validation_failed",
        "stage_name": "Global Validation",
        "errors": errors,
    })
    chat.state["global_validation_passed"] = None
    chat.state["global_validation_errors"] = None
    await _start_remediation(
        chat, "global_remediation",
        global_remediation_message(errors, chat.agent_session.store),
    )


async def _finish_dataset(chat: ChatSession) -> None:
    await _run_deterministic(chat, "Per-Sample Target Placement", target_placement_node)
    await _run_deterministic(chat, "Dataset Generation", dataset_generation_node)

    payload = _build_finalize_payload(chat)
    result = await finalize_dataset(FinalizeDatasetPayload.model_validate(payload))
    chat.complete = True
    chat.phase = "complete"
    # Post-completion turns must not look like collection/remediation checks.
    chat.active_purpose = "collect"
    chat.active_section = None
    chat.remediation_snapshot = None
    # The pipeline's own dataset takes over from any earlier upload.
    chat.uploaded_output_dir = None
    # Kept for reconnect replay — repopulates the dataset tab after refresh.
    chat.dataset_result = result
    # Post-completion edit machinery: baseline for the store diff, and the
    # previous forward-model results no longer apply to the new samples.
    chat.regenerating = False
    chat.complete_snapshot = chat.agent_session.snapshot()
    chat.regen_block_notice = None
    chat.simulation_result = None
    # A new dataset invalidates any pending reuse recommendation.
    chat.reuse_recommendation = None
    # Emitted filenames repeat across regenerations and the outputs listing is
    # filesystem-derived — stale .out files would attach to the new decks.
    cfg = DatasetConfig.model_validate(chat.state["dataset_config"])
    shutil.rmtree(
        _resolve_dataset_path(cfg.output_dir) / "out_files", ignore_errors=True
    )
    await _send(chat, {
        "type": "dataset_ready",
        "content": (
            f"Dataset created and stored. Wrote {result['rows_inserted']} "
            "simulation row(s) for generated input files."
        ),
        "result": result,
    })
    if not chat.post_complete_briefed:
        # One-time switch into post-completion behavior (discussion, edits
        # behind a disclaimer, restart refusal). Not re-injected after
        # regenerations — the agent already has the rules.
        chat.post_complete_briefed = True
        await _invoke_and_handle(chat, POST_COMPLETE_BRIEFING)


# Matches the CLI `_banner()` block: a ==== rule, the title line, another rule.
_BANNER_BLOCK_RE = re.compile(r"^ *={4,} *\n[^\n]*\n *={4,} *\n?", re.MULTILINE)


def _clean_stage_output(output: str) -> str:
    """Make the deterministic nodes' terminal output chat-friendly.

    The nodes print for the CLI: ==== banner blocks (the `stage_change` event
    already names the stage), `>>` routing markers, indentation, and absolute
    paths — all of which overflow or duplicate in the chat bubble.
    """
    text = _BANNER_BLOCK_RE.sub("", output)
    text = text.replace(_project_root + "/", "")
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(">>"):
            line = line[2:].strip()
        lines.append(line)
    text = "\n".join(lines)
    # Join a "Wrote ... to:" label with the bare path on the next line, but
    # leave list blocks ("Errors:" / "Warnings:" + "- ..." items) intact.
    text = re.sub(r":\n(?=\S*/\S*(?:\n|$))", ": ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


async def _run_deterministic(
    chat: ChatSession,
    stage_name: str,
    fn: Any,
) -> None:
    chat.phase = "deterministic"
    chat.busy = True
    await _send(chat, {"type": "stage_change", "stage_name": stage_name})
    await _send(chat, {"type": "pipeline_busy", "busy": True})
    try:
        updates, output = await asyncio.to_thread(_run_node_with_output, fn, dict(chat.state))
        if updates:
            chat.state.update(updates)
        output = _clean_stage_output(output)
        if output:
            await _send(chat, {"type": "agent_message", "content": output})
        for node, flag_updates in _VIZ_FLAG_UPDATES:
            if fn is node:
                chat.viz_flags.update(flag_updates)
                break
        await _send_model_update(chat, stage=stage_name)
    finally:
        chat.busy = False
        await _send(chat, {"type": "pipeline_busy", "busy": False})


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
    if cfg.contract_version >= 2:
        from backend.preflight import verify_contract
        from backend.dataset_sampling.contract import file_digest
        if not emitted_manifest.get("files") or emitted_manifest.get("errors"):
            raise ValueError("Cannot complete a dataset without a consistent accepted manifest")
        contract = emitted_manifest["contract"]
        verify_contract(contract)
        if global_derive != contract["grid"] or cfg.model_dump(exclude={"output_dir", "num_threads"}) != contract["requested"]["dataset_config"]:
            raise ValueError("Collected state/global plan differs from the emitted contract")
        for entry in emitted_manifest["files"]:
            verify_contract(contract, entry["resolved_scene"])
            if file_digest(Path(artifacts["in_dir"]) / entry["filename"]) != entry["input_sha256"]:
                raise ValueError("Input artifact changed before database finalization")
    # eps/sigma labels. Tolerated as absent (rows keep derived_layers = NULL)
    # rather than failing the finalize — a dataset without labels is still a
    # dataset, and the derive chain is what guarantees the manifest exists.
    derived_ref = artifacts.get("derived_layers_json")
    derived_manifest = (
        _read_json(Path(derived_ref))
        if derived_ref and Path(derived_ref).is_file()
        else {}
    )

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
            "dataset_contract": emitted_manifest.get("contract"),
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
        derived_manifest=derived_manifest,
        global_derive=global_derive,
        emitted_manifest=emitted_manifest,
    )
    # A re-finalize (post-complete edit regeneration) replaces the session's
    # rows — the old samples no longer exist.
    delete_simulations_for_session(session_uuid)
    inserted = batch_insert_simulations(sim_rows) if sim_rows else 0
    return {
        "status": status,
        "session_id": str(session_uuid),
        "rows_inserted": inserted,
        "num_generated": len(sim_rows),
        "output_dir": artifacts.get("output_dir"),
        "in_dir": artifacts.get("in_dir"),
        "files": [
            {"sample_id": f.get("sample_id"), "filename": f.get("filename")}
            for f in emitted_manifest.get("files", [])
            if f.get("filename")
        ],
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
    derived_manifest: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    files_by_sample = {
        int(item["sample_id"]): item
        for item in emitted_manifest.get("files", [])
        if "sample_id" in item
    }
    # Per-layer eps/sigma from the Peplinski derive, keyed the same way. The
    # layer lists are emitted in the same order by both manifests, so the entry
    # at index i labels layers[i].
    derived_by_sample = {
        int(item["sample_id"]): item.get("layers") or []
        for item in (derived_manifest or {}).get("samples", [])
        if "sample_id" in item
    }
    rows: list[dict[str, Any]] = []
    for sample in sampled_manifest.get("samples", []):
        sample_id = int(sample["sample_id"])
        emitted = files_by_sample.get(sample_id)
        if emitted is None:
            continue

        # Per-kind split of the sample's drawn objects (static objects are just
        # degenerate ranges, so they appear here too). Advanced params no
        # longer carry geometry.
        tgts = sample.get("targets") or []
        scene = emitted.get("resolved_scene")
        if scene:
            tgts = scene["targets"]
        cylinders = [t for t in tgts if t.get("kind") == "cylinder"]
        boxes = [t for t in tgts if t.get("kind") == "box"]

        rows.append({
            "id": uuid.uuid4(),
            "session_id": session_uuid,
            "user_id": user_id,
            "sample_index": sample_id,
            "antenna_kind": ant.antenna_kind,
            "antenna_axis": scene["source"]["axis"] if scene else ant.antenna_axis,
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
            "domain_z": global_derive.get("domain_z_m", global_derive["dx_m"] if cfg.dimensionality == "2D" else None),
            "dimensionality": cfg.dimensionality,
            "coordinate_frame": cfg.coordinate_frame,
            "contract_version": cfg.contract_version,
            "contract_digest": emitted.get("contract_digest"),
            "input_sha256": emitted.get("input_sha256"),
            "resolved_scene": scene,
            "requested_sample": sample,
            "qualification_status": "unqualified" if scene else "legacy_unverified",
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
            "derived_layers": derived_by_sample.get(sample_id) or None,
            "cylinders": cylinders or None,
            "boxes": boxes or None,
            "spheres": None,  # spheres unsupported (2D thin-z); column kept for schema stability
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
