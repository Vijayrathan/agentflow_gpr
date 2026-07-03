# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPR Synthetic Dataset Pipeline — generates `N` labeled gprMax `.in` files for training ML on subsurface soil characterization (sand/clay %, volumetric water content, layer thickness, buried-object properties from GPR signals). Thesis project with real lab data for sim-to-real transfer.

## Development Commands

**Package manager**: `uv` (Python 3.12)

```bash
# Install dependencies
uv sync

# Start PostgreSQL (required for sessions/simulations)
docker-compose up -d

# Run database migrations
alembic upgrade head

# Start backend (FastAPI on port 8000)
uvicorn backend.api:app --reload --host 127.0.0.1 --port 8000

# Run all tests
pytest backend/tests/ -v
pytest backend/dataset_sampling/tests/ -v

# Run a single test file
pytest backend/tests/test_api_finalize.py -v

# Run a single test
pytest backend/tests/test_api_finalize.py::test_name -v
```

**Environment variables** (`.env`): `OPENAI_API_KEY`, `HF_TOKEN`, `GPR_WORKSPACE_DIR` (default `../gpr_workspace`).

## Architecture

### Hard Boundary: Agentic vs Deterministic

This is the single most important design constraint. Violating it is never acceptable.

- **Agentic layer = extraction/collection only.** LLM agents extract user config and re-elicit on human-decision failures (infeasible texture, out-of-band frequency).
- **Deterministic core = everything else.** Sampling, ε-solving, derivation, validation, emission, gprMax execution — all plain Python. Never route physics through an LLM.
- The orchestrator is **LangGraph nodes**, not an agent.

### Backend (`backend/`)

| File | Role |
|------|------|
| `api.py` | FastAPI server; WebSocket endpoint (`/ws/{sessionId}`) bridges frontend to the **single-agent** pipeline (one isolated `SingleAgentSession` per chat) |
| `agentflow_single_agent.py` | **ACTIVE** extraction pipeline: ONE deep agent on ONE thread collects all six sections; also the CLI entry (`python backend/agentflow_single_agent.py`) |
| `single_agent_prompts.py` | Slim system prompt + per-section `SECTION_KICKOFF` injection messages + remediation message builders for the single agent |
| `agentflow_langgraph.py` | LEGACY multi-agent LangGraph pipeline (6 agents + parameter server). No longer imported by `api.py`; kept as CLI reference |
| `parameters_global_state.py` | In-memory parameter server (port 8100) — LEGACY path only; the single-agent pipeline does not use it |
| `schema.py` | All Pydantic models: `DatasetConfig`, `ExtractedLayers`, `ExtractedWaveform`, `ExtractedAntenna`, `ExtractedAdvancedParams`, `GlobalDerived`, etc. |
| `rag.py` | RAG retrieval: Qdrant vector DB + BAAI/bge-m3 embeddings + Docling parsing |
| `physics_modelling.py` | Peplinski ε computation via gprMax-native routines |
| `validation_tools_new.py` | Tiered validation (Tier 0–4) for physics constraints |

### Single-Agent Extraction (ACTIVE — `backend/agentflow_single_agent.py`)

One `deepagents.create_deep_agent()` (gpt-4.1-mini + the shared RAG sub-agent) collects
all six sections on a single conversation thread. Key mechanics:

- **`SingleAgentSession`** bundles the section store, the two tools bound to it
  (closures via `_make_section_tools`), the lazily built agent, and the thread id.
  The CLI drives a module-level `_DEFAULT_SESSION`; `api.py` creates one per
  WebSocket session — never share stores across sessions.
- **Tools**: `save_section(section, payload)` (validate + FULL replace; invalid ⇒
  rejected with error, nothing stored; schema-valid but missing essentials ⇒
  `stored_incomplete`) and `get_section`. No PATCH — editing = re-saving the full section.
- **Stage completion** is store-based (`stage_done`: schema-valid + `_section_is_complete`),
  NOT tool-call scraping. The orchestrator advances the moment the store section
  is complete — this drives the prompt rules below.
- **Prompting**: slim system prompt; each stage injects `SECTION_KICKOFF[section]`
  (field groups, physics constraints, JSON schema) as an internal orchestrator
  message. Remediation errors are injected into the SAME conversation; changed
  sections are detected by store-snapshot diffing.
- **Staleness re-sampling**: the agent may edit any section at any time. `layers`,
  `dataset_config`, AND `target_ranges` are `RESAMPLE_SECTIONS` (sampling inputs);
  if they change after `layer_sampling` ran, samples are re-drawn before the derive
  chain (snapshot comparison via `_samples_stale`).
- Key-free tests: `backend/tests/test_single_agent_store.py` (agent is lazy; module
  imports and graph compiles without `OPENAI_API_KEY`).

**Prompt-authoring rules** (regressions seen in live transcripts — keep these invariants
when editing `single_agent_prompts.py`):

- Kickoff messages are internal: they must open with the `[Orchestrator instruction …]`
  marker and never instruct the agent to announce the stage — the frontend's
  `stage_change` event already displays it (avoid double announcements / banner echo).
- No "Batch N" labels in field lists — the model parrots them to the user. Grouping is
  internal pacing guidance only.
- Because the pipeline advances immediately when a save completes a section, the save
  must be the LAST act of a stage: optional fields are raised BEFORE saving; a completed
  stage's reply ends with a short summary, never a question; "keep rest at defaults" ⇒
  save immediately, don't interrogate remaining fields.
- After a remediation re-save: confirm in one line and stop (re-validation is automatic).

### Extraction Agents (`backend/extraction_agents/`) — LEGACY

Six per-section agents (dataset_config, layers, target, waveform, antenna,
advanced_params), each `create_deep_agent()` + RAG sub-agent, communicating via the
port-8100 parameter server tools (`post_parameters`, `get_parameters`,
`patch_parameters`). Only used by the legacy `agentflow_langgraph.py` / `agentflow.py`
paths; the per-section prompt guidance in `prompt_library.py` is the source the
single-agent kickoffs were adapted from.

### Deterministic Pipeline (`backend/dataset_sampling/`)

Executed after extraction, in strict order:

1. `layer_sampler.py` — Draw N concrete samples from extracted ranges; Tier 2 validation
2. `sample_validation.py` — Cross-stage compatibility (Peplinski band gate, antenna config)
3. `peplinski_derive.py` — Compute ε via gprMax's native Peplinski model; aggregate wet/dry corners
4. `global_derive.py` — ε corners → wavelength budget → Δx → domain → Δt (CFL) → time window
5. `global_validation.py` — Grid numerics: λ/10, CFL, PML, domain fit (cascade order)
6. `target_placement.py` — Per-sample target validation; redraw then drop if infeasible
7. `emit.py` — Generate `.in` files (pure string assembly, no derivation)

### Frontend (`frontend/`)

Vanilla React/JSX (no build step). Entry: `frontend/html-design.html`. Components in `frontend/app/`:
- `chatbot.jsx` — WebSocket chat interface to backend
- `viz.jsx` — 2D GPR domain visualization (layers, antenna, targets, B-scan)
- `data.jsx` — State management for model data
- `panels.jsx` — Property inspector panels

### Database (`db/`)

PostgreSQL 15 via docker-compose. ORM: SQLModel. Migrations: Alembic (`db/alembic/`).

Two tables:
- `ExtractionSession` — per-section JSONB columns for user parameter ranges
- `Simulation` — one row per sample (params JSONB + signals as float8[] arrays)

### gprMax (`gprMax/`)

Full gprMax source as a local directory (not a submodule). Used directly for:
- `PeplinskiSoil` + `calculate_debye_properties` — ε computation for grid sizing
- `Material.calculate_er(f).real` — frequency-dependent permittivity (never read raw `m.er`)

### Data Versioning

DVC + Google Drive for large dataset files. Qdrant vector DB stored at `db/qdrant_storage/`.

## Pipeline Flow

```
START → dataset_config → layers → target_ranges → layer_sampling
  → waveform → antenna → sample_validation [GATE: loops on fail]
  → advanced_params [→ layer_sampling if sampling inputs went stale]
  → peplinski_derive → global_derive
  → global_validation [GATE: loops on fail to remediation]
  → target_placement → dataset_generation → END
```

On validation failure, the errors are injected into the single agent's ongoing
conversation; it agrees the fix with the user and re-saves the offending section, then
the gate re-runs. A global-remediation edit to a `RESAMPLE_SECTIONS` member routes back
through `layer_sampling` first. All state flows through `PipelineState` (TypedDict) /
`ChatSession.state`; after every completed agent turn the WHOLE store is synced into
state, so cross-section edits land immediately.

**Frontend WebSocket protocol** (`api.py` → `chatbot.jsx`): `agent_message`,
`stage_change`, `progress`, `validation_failed`, `pipeline_busy`, `dataset_ready`,
`error`. `choice_required` is no longer emitted (the agent negotiates the fix in
conversation); the frontend handler remains for compatibility.

## Physics Constraints

The full physics spec is in **AGENT.md** at the repo root. That file is authoritative for:
- The parameter selection chain (strict derivation order from Khosravi §III)
- Peplinski-only permittivity model (gprMax-native; no manual mixers)
- Ricker peak-vs-center frequency conversion (Wang 2015 band edges)
- Coordinate convention (x=horizontal, y=vertical, z=single-cell)
- Global grid invariants (one grid for all N samples, θv always a band)
- Validation tiers (Tier 0–4) and severity levels
- Key constants table
- **REJECTED alternatives** — approaches that were tried and deliberately removed

**Read AGENT.md fully before modifying any physics, validation, or emission code.**

## Current Limitations (from README.md)

- Dielectric target objects are not yet supported (targets are PEC only)
- Target material is not sampled
- Section-tag mapping for navigation back to agents on validation failure needs work