import json
import os
import sys
import uuid
from pathlib import Path
from typing import Annotated, Optional

import dotenv
import httpx
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI

from deepagents import create_deep_agent
from backend.parameters_global_state import get_parameters, patch_parameters, BASE_URL
from backend.prompt_library import DATASET_GENERATION_PROMPT, DATASET_VALIDATION_PROMPT
from backend.validation_tools import (
    validate_memory_estimate,
    validate_pml_vs_domain,
    validate_domain_z_alignment,
    validate_dispersive_tau_vs_dt,
    validate_snapshot_time_range,
    validate_waveform_bandwidth,
    validate_object_resolution,
    validate_rxarray_step_vs_cell,
    validate_object_pml_distance,
)

# Add this directory (dataset_sampling) to sys.path so bare imports like
# `from resolvers import ...` and `from dataset_generator import ...` work
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.schema import (
    ExtractedLayers,
    ExtractedLayerParams,
    ExtractedAntennaWaveform,
    ExtractedModelConfig,
    ExtractedAdvancedParams,
    DatasetGenerationResult,
    SurfaceRoughnessConfigSchema,
    RxArrayConfigSchema,
)
from resolvers import merge_extractions
from dataset_generator import generate_dataset

dotenv.load_dotenv()

# Module-level state: last successful generation result (read by post_dataset_to_db)
_last_generation_result: Optional[DatasetGenerationResult] = None


# ── Helper: fetch & parse all sections from parameter server ──────────

def _fetch_and_parse():
    """Fetch all 4 sections from the parameter server and parse into schema objects.

    Returns (sections_dict, errors_list) where sections_dict maps section name
    to its parsed Pydantic object, and errors_list contains any parse failures.
    """
    # TODO: restore actual server fetch when done testing
    resp = httpx.get(f"{BASE_URL}/state", timeout=10)
    resp.raise_for_status()
    state = resp.json()
    
    section_map = {
        "layers": ExtractedLayers,
        "antenna_waveform": ExtractedAntennaWaveform,
        "model_config": ExtractedModelConfig,
        "advanced_params": ExtractedAdvancedParams,
    }
    
    parsed = {}
    errors = []
    for section, cls in section_map.items():
        data = state.get(section)
        if data is None:
            errors.append(f"Section '{section}' has no data — extraction not yet complete.")
            continue
        try:
            parsed[section] = cls.model_validate(data)
        except Exception as e:
            errors.append(f"Section '{section}' failed validation: {e}")
    
    return parsed, errors

# ── Validation Sub-Agent ───────────────────────────────────────────────

validation_subagent = {
    "name": "validation-agent",
    "description": (
        "Cross-parameter physics validation specialist. Runs 9 checks that "
        "span multiple extraction sections: memory estimate, PML vs domain, "
        "domain Z alignment, dispersive tau vs dt, snapshot time range, "
        "waveform bandwidth, object resolution, rx_array step vs cell, and "
        "object PML distance. Call after resolve_and_validate passes, before "
        "dataset generation."
    ),
    "system_prompt": DATASET_VALIDATION_PROMPT,
    "tools": [
        validate_memory_estimate,
        validate_pml_vs_domain,
        validate_domain_z_alignment,
        validate_dispersive_tau_vs_dt,
        validate_snapshot_time_range,
        validate_waveform_bandwidth,
        validate_object_resolution,
        validate_rxarray_step_vs_cell,
        validate_object_pml_distance,
        get_parameters,
    ],
}


# ── Tools ─────────────────────────────────────────────────────────────

@tool
def fetch_all_extractions() -> str:
    """Fetch and summarise all extracted parameter sections from the server.

    Returns a JSON summary of what is currently stored for each section
    (layers, antenna_waveform, model_config, advanced_params), or which
    sections are missing / invalid.
    """
    parsed, errors = _fetch_and_parse()

    summary = {}
    if "layers" in parsed:
        layers = parsed["layers"]
        summary["layers"] = {
            "num_layers": layers.num_layers,
            "layer_names": [l.name for l in layers.layers],
        }
    if "antenna_waveform" in parsed:
        aw = parsed["antenna_waveform"]
        summary["antenna_waveform"] = {
            "antenna_kind": aw.antenna_kind,
            "waveform_kind": aw.waveform_kind,
            "center_freq_hz": aw.waveform_center_freq_hz,
        }
    if "model_config" in parsed:
        mc = parsed["model_config"]
        summary["model_config"] = {
            "model": mc.model,
            "title": mc.title,
            "num_samples": mc.num_samples,
            "domain_x": mc.domain_x,
            "domain_y": mc.domain_y,
        }
    if "advanced_params" in parsed:
        summary["advanced_params"] = "populated"

    result = {"sections": summary}
    if errors:
        result["errors"] = errors

    return json.dumps(result, indent=2)


@tool
def resolve_and_validate() -> str:
    """Resolve all extracted parameters and validate readiness for dataset generation.

    Fetches sections from the server, parses them, and calls merge_extractions
    to check completeness. Returns JSON with status 'ready' or 'incomplete'
    with a list of missing fields.
    """
    parsed, parse_errors = _fetch_and_parse()

    if parse_errors:
        return json.dumps({
            "status": "incomplete",
            "missing": parse_errors,
        }, indent=2)

    required = ["layers", "antenna_waveform", "model_config", "advanced_params"]
    missing_sections = [s for s in required if s not in parsed]
    if missing_sections:
        return json.dumps({
            "status": "incomplete",
            "missing": [f"Section '{s}' not populated" for s in missing_sections],
        }, indent=2)

    _schema, _ranges, missing = merge_extractions(
        parsed["layers"],
        parsed["antenna_waveform"],
        parsed["model_config"],
        parsed["advanced_params"],
    )

    if missing:
        return json.dumps({
            "status": "incomplete",
            "missing": missing,
        }, indent=2)

    return json.dumps({
        "status": "ready",
        "num_layers": len(parsed["layers"].layers),
        "model": parsed["model_config"].model,
        "num_samples": parsed["model_config"].num_samples,
    }, indent=2)


@tool
def run_dataset_generation(
    dataset_name: Annotated[str, "Name for the output dataset directory"],
    seed: Annotated[Optional[int], "Random seed for reproducibility (optional)"] = None,
) -> str:
    """Generate the dataset from extracted parameters.

    Fetches all sections, resolves them, and generates the requested number
    of gprMax .in files. Returns a JSON summary with generation status,
    counts, and any errors.
    """
    # Fetch and parse
    parsed, parse_errors = _fetch_and_parse()
    if parse_errors:
        return json.dumps({"status": "error", "errors": parse_errors}, indent=2)

    required = ["layers", "antenna_waveform", "model_config", "advanced_params"]
    missing_sections = [s for s in required if s not in parsed]
    if missing_sections:
        return json.dumps({
            "status": "error",
            "errors": [f"Section '{s}' not populated" for s in missing_sections],
        }, indent=2)

    # Resolve
    gpr_template, resolved_ranges, missing = merge_extractions(
        parsed["layers"],
        parsed["antenna_waveform"],
        parsed["model_config"],
        parsed["advanced_params"],
    )

    if missing:
        return json.dumps({"status": "error", "errors": missing}, indent=2)

    num_samples = parsed["model_config"].num_samples

    # Generate .in files
    result = generate_dataset(
        resolved_layer_ranges=resolved_ranges,
        gpr_schema_template=gpr_template,
        num_samples=num_samples,
        dataset_name=dataset_name,
        seed=seed,
    )

    # Store for post_dataset_to_db
    global _last_generation_result
    _last_generation_result = result

    # Determine effective status (partial with >=90% is acceptable)
    effective_status = result.status
    if result.status == "partial" and result.num_generated >= 0.9 * result.num_requested:
        effective_status = "complete"

    output = {
        "status": effective_status,
        "num_requested": result.num_requested,
        "num_generated": result.num_generated,
        "num_failed": result.num_failed,
    }
    if result.errors:
        output["errors"] = result.errors

    return json.dumps(output, indent=2)


@tool
def post_dataset_to_db() -> str:
    """Persist the generated dataset to the simulations database.

    Reads the last generation result and extracted parameters, builds
    Simulation rows, and POSTs them to the database. Call only after the
    user confirms they are satisfied with the dataset.
    """
    global _last_generation_result

    if _last_generation_result is None:
        return json.dumps({"status": "error", "message": "No dataset has been generated yet. Run dataset generation first."})

    result = _last_generation_result

    # Get session info
    try:
        sess_resp = httpx.get(f"{BASE_URL}/session", timeout=10)
        sess_resp.raise_for_status()
        sess_info = sess_resp.json()
        session_id = sess_info.get("session_id")
        user_id = sess_info.get("user_id") or "cli-user"
    except Exception:
        session_id = str(uuid.uuid4())
        user_id = "cli-user"

    # Fetch all extracted sections for scalar columns
    try:
        state_resp = httpx.get(f"{BASE_URL}/state", timeout=10)
        state_resp.raise_for_status()
        state = state_resp.json()
    except Exception as e:
        return json.dumps({"status": "error", "message": f"Failed to fetch parameter state: {e}"})

    aw = state.get("antenna_waveform") or {}
    mc = state.get("model_config") or {}
    ap = state.get("advanced_params") or {}

    # Validate that all required (NOT NULL) fields were extracted
    required_aw = {
        "tx_rx_offset_m": aw.get("tx_rx_offset_m"),
        "waveform_amplitude": aw.get("waveform_amplitude"),
        "waveform_center_freq_hz": aw.get("waveform_center_freq_hz"),
        "waveform_name": aw.get("waveform_name"),
    }
    required_mc = {
        "model": mc.get("model"),
        "title": mc.get("title"),
        "source_height_m": mc.get("source_height_m"),
        "domain_x": mc.get("domain_x"),
        "domain_y": mc.get("domain_y"),
        "cells_per_wavelength": mc.get("cells_per_wavelength"),
        "max_cell_m": mc.get("max_cell_m"),
        "temperature_c": mc.get("temperature_c"),
        "num_samples": mc.get("num_samples"),
    }
    missing = [k for k, v in {**required_aw, **required_mc}.items() if v is None]
    if missing:
        return json.dumps({
            "status": "error",
            "message": f"Required fields are missing (NULL) in extracted parameters: {missing}. "
                       "These must be collected by the extraction agents before posting to DB.",
        })

    # Build simulation rows
    rows = []
    for sample in result.samples:
        row = {
            # Primary key
            "id": str(uuid.uuid4()),
            # Identity
            "session_id": session_id,
            "user_id": user_id,
            "sample_index": sample.sample_index,
            # Antenna / Waveform
            "antenna_kind": aw.get("antenna_kind") or "hertzian_dipole",
            "antenna_axis": aw.get("antenna_axis") or "x",
            "tx_rx_offset_m": aw["tx_rx_offset_m"],
            "resistance": aw.get("resistance"),
            "source_start_time": aw.get("source_start_time"),
            "source_end_time": aw.get("source_end_time"),
            "waveform_kind": aw.get("waveform_kind") or "ricker",
            "waveform_amplitude": aw["waveform_amplitude"],
            "waveform_center_freq_hz": aw["waveform_center_freq_hz"],
            "waveform_name": aw["waveform_name"],
            # Model config
            "model": mc["model"],
            "title": mc["title"],
            "source_height_m": mc["source_height_m"],
            "domain_x": mc["domain_x"],
            "domain_y": mc["domain_y"],
            "top_air_extra_m": mc.get("top_air_extra_m"),
            "cells_per_wavelength": mc["cells_per_wavelength"],
            "max_cell_m": mc["max_cell_m"],
            "rx_same_height": mc.get("rx_same_height", True),
            "temperature_c": mc["temperature_c"],
            "enforce_validity": mc.get("enforce_validity", True),
            # Advanced
            "pml_cells": ap.get("pml_cells"),
            "num_threads": ap.get("num_threads"),
            "output_dir": ap.get("output_dir"),
            # Layers (JSONB)
            "layers": [lv.model_dump() for lv in sample.layers],
            "num_layers": len(sample.layers),
            # Geometry objects (JSONB, nullable)
            "cylinders": ap.get("cylinders"),
            "boxes": ap.get("boxes"),
            "spheres": ap.get("spheres"),
            # Optional config (JSONB, nullable)
            "surface_roughness": ap.get("surface_roughness"),
            "rx_array": ap.get("rx_array"),
            "snapshots": ap.get("snapshots"),
            # File reference
            "input_file_path": sample.filepath,
        }
        rows.append(row)

    # POST to simulations endpoint
    try:
        resp = httpx.post(f"{BASE_URL}/simulations", json=rows, timeout=60)
        resp.raise_for_status()
        resp_data = resp.json()
    except Exception as e:
        return json.dumps({"status": "error", "message": f"Failed to POST simulations: {e}"})

    return json.dumps({
        "status": "ok",
        "rows_inserted": resp_data.get("rows_inserted", 0),
        "dataset_name": result.dataset_name,
        "num_generated": result.num_generated,
    }, indent=2)


@tool
def verify_simulations_db() -> str:
    """Verify that simulation rows were inserted into the database.

    Returns the total count and a sample of rows for the current session.
    """
    # Get session_id
    try:
        sess_resp = httpx.get(f"{BASE_URL}/session", timeout=10)
        sess_resp.raise_for_status()
        sess_info = sess_resp.json()
        session_id = sess_info.get("session_id")
    except Exception as e:
        return json.dumps({"status": "error", "message": f"Failed to get session info: {e}"})

    if not session_id:
        return json.dumps({"status": "error", "message": "No active session found."})

    # Query DB directly
    try:
        from db.db import get_session as get_db_session, Simulation
        from sqlmodel import select, func

        with get_db_session() as db:
            # Total count
            count_stmt = select(func.count()).where(Simulation.session_id == session_id)
            total = db.exec(count_stmt).one()

            # Sample rows
            sample_stmt = (
                select(
                    Simulation.sample_index,
                    Simulation.antenna_kind,
                    Simulation.waveform_kind,
                    Simulation.num_layers,
                )
                .where(Simulation.session_id == session_id)
                .order_by(Simulation.sample_index)
                .limit(5)
            )
            sample_rows = db.exec(sample_stmt).all()

        return json.dumps({
            "status": "ok",
            "total_rows": total,
            "sample_rows": [
                {
                    "sample_index": r[0],
                    "antenna_kind": r[1],
                    "waveform_kind": r[2],
                    "num_layers": r[3],
                }
                for r in sample_rows
            ],
        }, indent=2)

    except Exception as e:
        return json.dumps({"status": "error", "message": f"DB query failed: {e}"})


# ── Agent ─────────────────────────────────────────────────────────────

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

agent = create_deep_agent(
    model=llm,
    subagents=[validation_subagent],
    system_prompt=DATASET_GENERATION_PROMPT,
    checkpointer=InMemorySaver(),
    tools=[
        fetch_all_extractions,
        resolve_and_validate,
        run_dataset_generation,
        post_dataset_to_db,
        verify_simulations_db,
        get_parameters,
        patch_parameters,
    ],
)


# ── Standalone runner ─────────────────────────────────────────────────

def _print_response(result: dict, seen: int = 0) -> int:
    messages = result.get("messages", [])
    for msg in messages[seen:]:
        kind = type(msg).__name__
        if kind == "AIMessage" and msg.content:
            print(f"\n[Agent]: {msg.content}\n")
        elif kind == "ToolMessage":
            print(f"  [tool:{msg.name}] returned ({len(msg.content)} chars)")
    return len(messages)


if __name__ == "__main__":
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    result = agent.invoke(
        {"messages": [HumanMessage(
            content="All parameter extractions are complete. Begin dataset generation."
        )]},
        config=config,
    )
    seen = _print_response(result)

    while True:
        user_input = input("You: ").strip()
        if not user_input or user_input.lower() in ("quit", "exit"):
            break
        result = agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )
        seen = _print_response(result, seen)
