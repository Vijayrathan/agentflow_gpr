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

**Environment variables** (`.env`): `OPENAI_API_KEY`, `HF_TOKEN`, `GPR_WORKSPACE_DIR` (default `../gpr_workspace`), `DATABASE_URL` (optional — overrides the docker-compose Postgres default in `db/db.py` and Alembic).

## Architecture

### Hard Boundary: Agentic vs Deterministic

This is the single most important design constraint. Violating it is never acceptable.

- **Agentic layer = extraction/collection only.** LLM agents extract user config and re-elicit on human-decision failures (infeasible texture, out-of-band frequency).
- **Deterministic core = everything else.** Sampling, ε-solving, derivation, validation, emission, gprMax execution — all plain Python. Never route physics through an LLM.
- The orchestrator is **LangGraph nodes**, not an agent.

### Backend (`backend/`)

| File | Role |
|------|------|
| `api.py` | FastAPI server; WebSocket endpoint (`/ws/{userId}/{sessionId}`) bridges frontend to the **single-agent** pipeline (one isolated `SingleAgentSession` per chat; chats persist per user — see Multi-chat persistence) |
| `agentflow_single_agent.py` | **ACTIVE** extraction pipeline: ONE deep agent on ONE thread collects all six sections; also the CLI entry (`python backend/agentflow_single_agent.py`) |
| `single_agent_prompts.py` | Slim system prompt + per-section `SECTION_KICKOFF` injection messages + remediation message builders for the single agent |
| `agentflow_langgraph.py` | LEGACY multi-agent LangGraph pipeline (6 agents + parameter server). No longer imported by `api.py`; kept as CLI reference |
| `parameters_global_state.py` | In-memory parameter server (port 8100) — LEGACY path only; the single-agent pipeline does not use it |
| `schema.py` | All Pydantic models: `DatasetConfig`, `ExtractedLayers`, `ExtractedWaveform`, `ExtractedAntenna`, `ExtractedAdvancedParams`, `GlobalDerived`, etc. |
| `rag.py` | RAG retrieval: Qdrant vector DB + BAAI/bge-m3 embeddings + Docling parsing |
| `physics_modelling.py` | Peplinski ε computation via gprMax-native routines |
| `validation_tools_new.py` | Tiered validation (Tier 0–4) for physics constraints |
| `viz_projection.py` | Pure projection of the section store + pipeline manifests into the `model_update` scene payload for the live frontend visualization (no FastAPI/LLM imports) |
| `simulate.py` | Batch gprMax forward-model runner (deterministic; lazy gprMax import). Backs the UI "Run forward model" button via `POST /datasets/{sid}/simulate` and doubles as a CLI |
| `deck_validation.py` | Syntax validation for user-uploaded gprMax `.in` decks: safe preprocessing (mirrors gprMax's line handling but NEVER `exec`s `#python:` blocks; `#include_file:` rejected too) + gprMax's own `check_cmd_names` rules (lazy gprMax import) |
| `sim_similarity.py` | Forward-model reuse: session-config similarity index over the `sim_sessions` Qdrant collection (raw normalized numeric feature vectors — NO embedding model — + payload hard filters + exact interval-IoU rescoring in Python). Deterministic, lazy `qdrant_client` import, failure-swallowing entry points; `python backend/sim_similarity.py backfill` indexes existing completed sessions |

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

**Per-dataset directory layout**: `output_dir` is server-fixed in `save_section`.
API sessions author `./dataset/<user_id>/<model_basename>__<sid8>` via
`_scoped_output_dir` (`sid8` = first 8 chars of sha1(session_id) — deterministic,
so re-saves within a chat never move the dataset, while identical basenames
across chats/users never collide); the CLI keeps the legacy
`./dataset/<model_basename>` path (`_default_output_dir`). Both segments are
sanitized via `_dataset_dirname` — path separators/dots never survive, so a
malicious user_id cannot traverse. Uploaded zips get the same scoping in
`_import_deck_zip`. Everything downstream (manifests, emission, simulation, viz
projection, `/datasets/{sid}` endpoints, DB rows) resolves paths from
`cfg.output_dir` — never hardcode `./dataset`. A mid-pipeline basename edit
moves `output_dir`, which is safe because `dataset_config` is a sampling input:
staleness detection re-runs `layer_sampling`, rewriting all manifests into the
new directory before the derive chain reads them. Pre-scoping
`./dataset/<basename>` dirs are orphaned legacy — safe to delete manually.

### Frontend (`frontend/`)

Vanilla React/JSX (no build step; Babel-standalone, all components exported on
`window`). Entry: `frontend/html-design.html`. Components in `frontend/app/`:
- `app.jsx` — root component; owns `model`, the live `scene`, canvas view tabs
- `chatbot.jsx` — WebSocket chat pane (drag-resizable; width persisted in localStorage)
- `viz.jsx` — 2D GPR domain visualization (layers, antenna, targets, B-scan)
- `data.jsx` — catalogs, `makeInitialModel` (BLANK scene), `makeUtilityModel` (demo
  preset), `sceneToModel`, `materialKeyForLayer`, `overviewCaveats`
- `panels.jsx` — Property inspector panels, model tree, dock

Layout sizing: chat / rail / dock use viewport-relative clamps (`clamp(...)` in
`html-design.html`), not fixed px — keep it that way. The subsurface SVG
(`viewBox 1000×660`, `meet`) fills the FULL canvas width when the bottom dock is
collapsed, so absolutely-positioned canvas overlays will cover the plot; new
overlays must be compact/collapsible (see the caveats chip) rather than permanent
panels.

### Live Subsurface Visualization (`model_update` event)

The canvas builds up live while the agent collects parameters. Data flow:

```
section store / manifests → viz_projection.build_scene() → ws `model_update {scene}`
  → ChatPane onModelUpdate (ref) → App.scene → sceneToModel() → setModel → SubsurfaceView
```

- **Emission** (`api.py`): after every agent turn (`_handle_agent_result`) and after
  each deterministic node (`_run_deterministic`), via `_send_model_update` —
  failures are swallowed (viz must never break the chat loop), identical scenes are
  deduplicated (`ChatSession.last_scene`), and the last scene is replayed on
  reconnect so a page reload repopulates the canvas.
- **Flag gating** (`ChatSession.viz_flags`: `sampled/derived/grid/placed/emitted`):
  `build_scene` only reads a manifest (`sampled_layers.json`, `derived_layers.json`,
  `global_derive.json`) when the producing node ran in THIS session — stale files
  from a previous run must never be shown. `layer_sampling` RESETS all downstream
  flags (covers staleness re-sampling and global-remediation resample).
- **Scene payload**: `ranges` (midpoint layers + thickness min/max + preview εr,
  midpoint target) always reflects the store; `samples` (capped at `SAMPLE_CAP=200`,
  with `total/included/truncated`) and `grid` coexist with it because
  `layer_sampling` runs mid-collection. `domain.provisional` is true until
  `global_derive` fixes the real grid.
- **Preview εr** is deterministic Python: `derive_layer_eps` (gprMax-native
  Peplinski, in-band `calculate_er(f).real`) at midpoint composition; evaluated at a
  0.9 GHz placeholder (`eps_provisional: true`) until the waveform frequency exists.
  σ is intentionally never previewed. Per-sample εr is joined from
  `derived_layers.json`, not recomputed.
- **Canvas tabs** (`app.jsx`): "Overview" renders range midpoints + cumulative
  thickness-uncertainty bands (`thicknessMin/Max` on layers → shaded bands in
  `SubsurfaceView`); "Samples" (disabled until samples exist) renders one concrete
  realization chosen via a dropdown keyed by `sample_id` (ids may be non-contiguous
  after target placement drops samples). Overview also shows a collapsible
  "assumptions" chip (`overviewCaveats`) listing what is still placeholder vs derived.
- **Frontend invariants**: `ws.onmessage` is bound once at mount, so callbacks
  passed into `ChatPane` must be read through a ref (stale-closure trap).
  `makeInitialModel()` is a BLANK scene — the demo model lives in
  `makeUtilityModel()` (a data.jsx fixture; the Upload menu is zip-import only —
  the scenario-presets menu was removed); don't re-populate the initial
  model. Backend sends physics only; material colors/patterns are a frontend
  display classification (`materialKeyForLayer`).
- Key-free tests: `backend/tests/test_viz_projection.py` (store→scene,
  manifests→scene, sample cap, flag gating).

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
`model_update`, `session_restore`, `simulation_progress`, `simulation_complete`,
`reuse_recommendation`, `error`. `choice_required` is no longer emitted (the
agent negotiates the fix in conversation); the frontend handler remains for
compatibility.

**Forward model** ("Run forward model" button → `POST /datasets/{sid}/simulate`):
once the dataset is emitted, the button runs `simulate.run_batch_simulation` (gprMax
Python API, lazy import) on the session's `.in` files in a worker thread, writing
`.out` files to `<output_dir>/out_files`. The batch is restricted to the emission
manifest's filenames — the per-dataset in_files dir can still hold stale decks
(re-emission after a re-sample, basename reuse across sessions), which must never
be simulated. Per-file progress streams as
`simulation_progress` (transient; drives the run button/progress bar in `app.jsx`
via `onSimulationEvent`); the recorded `simulation_complete` summary lands in the
chat, `.out` paths are written onto the session's `Simulation` rows
(`db.set_simulation_outputs`, keyed by `sample_index`; failures swallowed) and the
signal arrays are extracted into the rows' `signal_*` float8[] columns
(`signal_extraction.extract_and_prepare_batch` + `db.bulk_update_signals`, lazy
h5py import — this is what makes a dataset reusable, see Forward-model reuse). The
endpoint 409s while a run or pipeline step is in flight; `simulating` +
`simulation` ride along on `session_restore` so a refresh re-hydrates the run
state. Key-free tests: `backend/tests/test_simulate.py` (solver stubbed).

**Forward-model reuse** (`backend/sim_similarity.py`; fully deterministic — the
agent is NEVER involved): after every fully successful generated run,
`_run_forward_model` indexes the session's config envelope into the
`sim_sessions` Qdrant collection (point id = the coerced session UUID —
idempotent; same `QDRANT_URL` as RAG but a separate collection with NO embedding
model). Vectors are raw numeric features normalized by FIXED physical constants
(`SCALES`); text embeddings are useless here because configs differ only in
numbers. Retrieval = ANN candidates + payload hard filters (`num_layers`,
waveform/antenna kinds, target counts, grid policy — categorically incompatible
configs never compete); the gate is an exact Python rescoring (interval IoU over
ranges, weighted per `GROUP_WEIGHTS`) against `SIM_REUSE_THRESHOLD` (default
0.95; also `SIM_REUSE_TOPK`, `SIM_REUSE_ENABLED`). `POST /datasets/{sid}/simulate`
runs the check first (skipped for uploads and with `?force=true`): on a match it
does NOT start the run — it records a `reuse_recommendation` chat event, stores
`ChatSession.reuse_recommendation` (persisted; rides `session_restore` as
`reuse`), and returns `{"status": "reuse_recommended"}`, which `app.jsx` renders
as a compact Reuse / Simulate-anyway bar by the Run button. `POST
/datasets/{sid}/adopt` executes ONLY the pending recommendation (409 otherwise):
after verifying the source end-to-end (all rows `simulation_completed_at`, files
on disk — BEFORE any deletion), it REPLACES the current dataset with a copy of
the source's (`in_files`/`out_files`/manifests, `emitted_files.json` rewritten
with current paths + `adopted_from`; Simulation rows re-keyed to this
session/user with signals carried and paths repointed; rows lacking arrays are
healed from the copied `.out` files), then emits `dataset_ready` +
`simulation_complete`. The section store / `ExtractionSession` keep the USER's
own ranges — a later regeneration re-runs from those and overwrites the
adoption. Every similarity entry point swallows failures (Qdrant down ⇒ the run
proceeds normally). Backfill existing sessions:
`python backend/sim_similarity.py backfill`. Key-free tests:
`backend/tests/test_sim_similarity.py` + the reuse/adopt half of
`test_simulate.py`.

**Dataset upload** ("Upload → From file…" → `POST /datasets/{sid}/upload`, raw zip
as the request body — no multipart dep): every `.in` member is syntax-checked by
`deck_validation.py` BEFORE anything is written; valid decks land in
`./dataset/<sanitized zip stem>/in_files` with an emission-style
`emitted_files.json` (`"source": "upload"`), rejected files are reported per-file
in the recorded `dataset_ready` chat message (and in the HTTP response's
`rejected` list). `ChatSession.uploaded_output_dir` makes `_session_output_dir` —
and therefore ALL dataset endpoints (files/content/simulate/outputs/download) —
serve the upload exactly like a generated dataset; `_finish_dataset` clears the
override when the pipeline emits its own. Uploads create no `Simulation` DB rows:
`_record_simulation_outputs` skips `"upload"` manifests so positional sample ids
never overwrite a generated dataset's rows. Key-free tests:
`backend/tests/test_deck_validation.py`, `backend/tests/test_api_upload.py`.

**Refresh/reconnect model** (`api.py` ↔ `chatbot.jsx`): every chat-visible event
(`RECORDED_EVENT_TYPES` + user messages) is appended to `ChatSession.transcript`;
all pipeline output goes through `_send(chat, …)`, which targets `chat.ws` — the
CURRENT socket — and swallows send failures, so a page refresh mid-turn neither
kills the pipeline nor loses output (it lands in the transcript). Reconnecting to
a started session replays `session_restore` (full transcript + `dataset` result +
`busy`/`phase`/`complete`) followed by the last `model_update` scene; the frontend
rebuilds the message list via `replayToMessage` (must mirror `handleServerEvent`
rendering) and re-hydrates the dataset tab through `onDatasetReady`. `user_message`
events are record-only (never echoed back). Every top-level turn ends with a
`pipeline_busy: false` so a restored `busy: true` always converges.

**Multi-chat persistence** (user_id identity, no auth): chats belong to a typed
`user_id` (1–64 chars, ≥1 alnum — `_validate_user_id`; the raw value goes to the
DB, only the sanitized form touches the filesystem). The WS route is
`/ws/{user_id}/{session_id}`; `GET/POST /users/{user_id}/chats` list and mint
chats (POST creates the `chat_sessions` stub row with a server-minted
session_id + thread_id, so the chat is listable before its first connect).
Every ChatSession JSON field rides in one JSONB blob (`chat_sessions.session_state`,
`_PERSISTED_CHAT_FIELDS` + store + seen; promoted columns `title`/`complete`/
`has_dataset`/`updated_at` exist only for the list query). Written failure-swallowed
by `_persist_chat` at: end of every `_handle_user_text` turn, post-kickoff on
first connect (MUST be after `_start_stage`, or a restart would replay the
kickoff into a thread that already has it), after upload, after a forward-model
run, and on WS disconnect (safety net — may capture a mid-turn snapshot).
The LLM message history is NOT in the blob: it lives in LangGraph's
`langgraph-checkpoint-postgres` tables (shared sync `PostgresSaver` singleton,
`backend/checkpointer.py`, psycopg3 pool; `saver.setup()` manages its own
schema — deliberately outside Alembic), keyed by the persisted `thread_id` —
that's what makes the agent resume its exact conversation after a restart.
Hydration (`_resolve_chat`/`_resolve_chat_sync` — used by the WS route AND all
dataset endpoints, hydrate-or-404): rebuild `ChatSession` +
`SingleAgentSession(thread_id=row.thread_id, checkpointer_factory=get_checkpointer)`,
apply the blob, coerce in-flight state (`busy`/`simulating` → False, phase
`deterministic|routing` → `complete`/`agent`; `regenerating` deliberately
survives), and mutate `agent_session.store` IN PLACE — the save/get tools are
closures over that exact dict. `SingleAgentSession()` without a factory keeps
`InMemorySaver` (CLI + key-free tests never touch Postgres; the agent stays
lazy). A server crash mid-turn loses that turn's session-side output; the LLM
checkpoint may be one turn ahead — the next invoke relays the orphaned AI texts
once (benign). Single-worker assumption: the in-memory `sessions` dict + WS
affinity mean ONE uvicorn worker. `DATABASE_URL` env var overrides the
hardcoded default in `db/db.py` and `db/alembic/env.py`; docker-compose mounts
the `pgdata` volume so Postgres survives container recreation. Frontend:
localStorage keys `nl2sim_user_id` + `nl2sim_current_chat_id` (chat LIST always
from the GET endpoint, never localStorage); `UserGate`/`ChatStrip` in
`frontend/app/chats.jsx`; `getSessionId()` returns the current chat pointer so
every HTTP call re-scopes on switch; the ChatPane WS effect re-keys on
`[sessionId, userId]` and `session_restore` repaints the switched-to chat;
`resetForChatSwitch()` + stale-fetch guards in `app.jsx` keep one chat's
responses out of another's UI. Chat deletion is deferred (no endpoint/UI).
Key-free tests: `backend/tests/test_chat_persistence.py`, `test_chats_api.py`,
`test_dataset_scoping.py`.

**Post-completion chat** (edit → regenerate, restart refused): after
`dataset_ready` the session stays conversational, and each agent turn is diffed
against `ChatSession.complete_snapshot` (the store at the last successful
`_finish_dataset`) via `_check_regeneration`. No change ⇒ relay-only discussion.
A change with every changed section `stage_done` ⇒ `_start_regeneration` clears
`complete`, sets `regenerating`, syncs the store, and synchronously re-runs the
deterministic tail in the CANONICAL order — sample gate, then inside
`_run_derive_chain` a resample iff a `RESAMPLE_SECTIONS` member changed, then
peplinski/global derive → global validation (remediation loops work because
`complete` is off) → placement → emission → `_finish_dataset` (new
`dataset_ready`, snapshot refreshed, stale `out_files/` purged — filenames repeat
across regenerations — old `Simulation` rows deleted before re-insert via
`delete_simulations_for_session`). An INCOMPLETE changed section never touches
the existing dataset (diff persists to the next turn); `regenerating` 409s the
simulate/upload endpoints between remediation turns. The one-way `regenerating`
routing lives in `_run_sample_validation_gate`'s pass branch (derive chain, not
`advanced_params` collection). Prompt side: `POST_COMPLETE_BRIEFING` is injected
ONCE at the first `_finish_dataset`; the system prompt's "After the dataset is
generated" section makes the agent show a bold disclaimer + get user confirmation
before any post-complete save, and REFUSE restart/new-simulation requests (new
sims = separate chats, future work). Key-free tests:
`backend/tests/test_api_regeneration.py`.

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