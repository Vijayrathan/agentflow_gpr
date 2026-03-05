"""
FastAPI-based global parameter state for extraction agents.

Provides:
- A FastAPI server (auto-started in a background thread) that stores
  extracted parameters for each agent section.
- LangChain @tool wrappers so agents can POST, GET, and PATCH parameters
  via HTTP against a central, durable state.

Sections:
  layers          -> ExtractedLayers
  antenna_waveform -> ExtractedAntennaWaveform
  model_config    -> ExtractedModelConfig
  advanced_params -> ExtractedAdvancedParams
"""

import threading
import time
import json
from typing import Annotated, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.tools import tool

from schema import (
    ExtractedLayers,
    ExtractedAntennaWaveform,
    ExtractedModelConfig,
    ExtractedAdvancedParams,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_HOST = "127.0.0.1"
API_PORT = 8100
BASE_URL = f"http://{API_HOST}:{API_PORT}"

# Section name -> Pydantic model class
SECTION_SCHEMAS = {
    "layers": ExtractedLayers,
    "antenna_waveform": ExtractedAntennaWaveform,
    "model_config": ExtractedModelConfig,
    "advanced_params": ExtractedAdvancedParams,
}

VALID_SECTIONS = list(SECTION_SCHEMAS.keys())

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="GPR Parameter State API")

# In-memory store: section name -> dict (serialised Pydantic model)
_store: dict[str, dict | None] = {s: None for s in VALID_SECTIONS}


def _validate_section(section: str) -> None:
    if section not in VALID_SECTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid section '{section}'. Must be one of {VALID_SECTIONS}",
        )


# ---- GET endpoints --------------------------------------------------------

@app.get("/state")
def get_full_state():
    """Return all four sections."""
    return _store


@app.get("/{section}")
def get_section(section: str):
    _validate_section(section)
    data = _store[section]
    if data is None:
        raise HTTPException(status_code=404, detail=f"Section '{section}' has no data yet.")
    return data


# ---- POST endpoints -------------------------------------------------------

@app.post("/{section}")
def post_section(section: str, payload: dict):
    """Create or fully replace a section's data. Validates against the schema."""
    _validate_section(section)
    schema_cls = SECTION_SCHEMAS[section]
    validated = schema_cls.model_validate(payload)
    _store[section] = validated.model_dump()
    return {"status": "ok", "section": section, "data": _store[section]}


# ---- PATCH endpoints ------------------------------------------------------

@app.patch("/{section}")
def patch_section(section: str, updates: dict):
    """Merge partial updates into an existing section."""
    _validate_section(section)
    current = _store[section]
    if current is None:
        raise HTTPException(
            status_code=404,
            detail=f"Section '{section}' has no data yet. Use POST first.",
        )
    merged = {**current, **updates}
    schema_cls = SECTION_SCHEMAS[section]
    validated = schema_cls.model_validate(merged)
    _store[section] = validated.model_dump()
    return {"status": "ok", "section": section, "data": _store[section]}


# ---- DELETE endpoint ------------------------------------------------------

@app.delete("/state")
def delete_all():
    """Reset all sections to None."""
    for s in VALID_SECTIONS:
        _store[s] = None
    return {"status": "ok", "message": "All sections cleared."}


# ---------------------------------------------------------------------------
# Server lifecycle (background thread)
# ---------------------------------------------------------------------------

_server_thread: Optional[threading.Thread] = None
_server_started = threading.Event()


def start_parameter_server() -> None:
    """Launch the FastAPI server in a daemon thread (idempotent)."""
    global _server_thread
    if _server_thread is not None and _server_thread.is_alive():
        return  # already running

    def _run():
        config = uvicorn.Config(app, host=API_HOST, port=API_PORT, log_level="warning")
        server = uvicorn.Server(config)
        _server_started.set()
        server.run()

    _server_thread = threading.Thread(target=_run, daemon=True)
    _server_thread.start()
    _server_started.wait(timeout=5)
    # Give uvicorn a moment to bind the socket
    time.sleep(0.3)


# ---------------------------------------------------------------------------
# LangChain @tool wrappers
# ---------------------------------------------------------------------------


@tool
def post_parameters(
    section: Annotated[str, "Section name: 'layers', 'antenna_waveform', 'model_config', or 'advanced_params'"],
    payload: Annotated[str, "JSON string of the parameters to store for this section"],
) -> str:
    """Store (create or replace) the extracted parameters for a given section.

    The payload is validated against the section's schema before storing.
    """
    data = json.loads(payload)
    resp = httpx.post(f"{BASE_URL}/{section}", json=data, timeout=10)
    return resp.text


_SECTION_AGENT_NAMES = {
    "layers": "Layer Extraction Agent",
    "antenna_waveform": "Antenna & Waveform Agent",
    "model_config": "Model & Domain Agent",
    "advanced_params": "Advanced Parameters Agent",
}


@tool
def get_parameters(
    section: Annotated[str, "Section name: 'layers', 'antenna_waveform', 'model_config', or 'advanced_params'"],
) -> str:
    """Retrieve the currently stored parameters for a given section.

    If the section has not been populated yet, returns a message indicating
    which specialist agent is responsible for collecting those parameters.
    """
    if section not in VALID_SECTIONS:
        return json.dumps({"error": f"Invalid section '{section}'. Must be one of {VALID_SECTIONS}"})
    resp = httpx.get(f"{BASE_URL}/{section}", timeout=10)
    if resp.status_code == 404:
        agent = _SECTION_AGENT_NAMES.get(section, section)
        return json.dumps({
            "error": "section_not_populated",
            "section": section,
            "message": (
                f"Section '{section}' has not been populated yet. "
                f"The {agent} is responsible for collecting these parameters. "
                f"This section must be filled by that agent before it can be read or edited."
            ),
        })
    return resp.text


@tool
def patch_parameters(
    section: Annotated[str, "Section name: 'layers', 'antenna_waveform', 'model_config', or 'advanced_params'"],
    updates: Annotated[str, "JSON string of the fields to update (partial update, merged with existing data)"],
) -> str:
    """Partially update the stored parameters for a given section.

    Use this to edit parameters in ANY section — including sections owned by
    other agents. First call get_parameters to check the section is populated.
    Only the provided fields are overwritten; other fields remain unchanged.
    """
    if section not in VALID_SECTIONS:
        return json.dumps({"error": f"Invalid section '{section}'. Must be one of {VALID_SECTIONS}"})
    data = json.loads(updates)
    # Check that every key in updates is a valid field for the section schema
    schema_cls = SECTION_SCHEMAS[section]
    valid_fields = set(schema_cls.model_fields.keys())
    invalid_fields = set(data.keys()) - valid_fields
    if invalid_fields:
        return json.dumps({
            "error": "invalid_fields",
            "section": section,
            "invalid_fields": sorted(invalid_fields),
            "valid_fields": sorted(valid_fields),
            "message": (
                f"Fields {sorted(invalid_fields)} do not exist in section '{section}'. "
                f"Valid fields are: {sorted(valid_fields)}"
            ),
        })
    resp = httpx.patch(f"{BASE_URL}/{section}", json=data, timeout=10)
    if resp.status_code == 404:
        agent = _SECTION_AGENT_NAMES.get(section, section)
        return json.dumps({
            "error": "section_not_populated",
            "section": section,
            "message": (
                f"Section '{section}' has not been populated yet. "
                f"The {agent} is responsible for collecting these parameters. "
                f"This section must be filled by that agent before it can be edited."
            ),
        })
    return resp.text


@tool
def get_all_parameters() -> str:
    """Retrieve the full state across all four sections."""
    resp = httpx.get(f"{BASE_URL}/state", timeout=10)
    return resp.text


@tool
def delete_all_parameters() -> str:
    """Reset all sections, clearing all stored parameters."""
    resp = httpx.delete(f"{BASE_URL}/state", timeout=10)
    return resp.text


if __name__ == "__main__":
    start_parameter_server()
    print(f"Parameter state server running on {BASE_URL}")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down.")

