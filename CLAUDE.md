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
| `api.py` | FastAPI server; WebSocket endpoint (`/ws/{sessionId}`) bridges frontend to LangGraph pipeline |
| `agentflow_langgraph.py` | 10-stage LangGraph pipeline definition — connects extraction agents to deterministic stages |
| `parameters_global_state.py` | In-memory parameter server (FastAPI on port 8100); agents POST/GET/PATCH sections here |
| `schema.py` | All Pydantic models: `DatasetConfig`, `ExtractedLayers`, `ExtractedWaveform`, `ExtractedAntenna`, `ExtractedAdvancedParams`, `GlobalDerived`, etc. |
| `rag.py` | RAG retrieval: Qdrant vector DB + BAAI/bge-m3 embeddings + Docling parsing |
| `physics_modelling.py` | Peplinski ε computation via gprMax-native routines |
| `validation_tools_new.py` | Tiered validation (Tier 0–4) for physics constraints |

### Extraction Agents (`backend/extraction_agents/`)

Six agents, each using `deepagents.create_deep_agent()` with GPT-4 mini + a RAG sub-agent:

- `dataset_config_extraction.py` — num_samples, naming, grid/boundary policy
- `layer_extraction.py` — soil layer ranges (sand %, clay %, thickness, densities, moisture)
- `target_extraction.py` — buried target geometry ranges (cylinder only for now)
- `waveform_extraction.py` — waveform type, amplitude, center frequency
- `antenna_extraction.py` — antenna axis, Tx-Rx offset, resistance
- `advanced_params_extraction.py` — PML, fractal, snapshots

Agents communicate through the parameter server (port 8100) via three tools: `post_parameters`, `get_parameters`, `patch_parameters`.

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

## LangGraph Pipeline Flow

```
START → dataset_config → layers → target_ranges → layer_sampling
  → waveform → antenna → sample_validation [GATE: loops on fail]
  → advanced_params → peplinski_derive → global_derive
  → global_validation [GATE: loops on fail to remediation]
  → target_placement → dataset_generation → END
```

On validation failure, the graph routes to a remediation node that re-engages the relevant agent, then loops back to validate. No direct agent-to-agent communication — all state flows through `PipelineState` (TypedDict).

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