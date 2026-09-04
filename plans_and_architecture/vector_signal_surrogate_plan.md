# Simulation Reuse via Session-Config Vector Search

## Context

Running the gprMax forward model on a full dataset is expensive. Users often request parameter configurations very close to ones already simulated. This feature adds a **session-level config similarity search** over previously completed simulation sessions: when the user clicks "Run forward model", the current session's parameter envelope is compared against past sessions indexed in Qdrant; if a match ≥ 95% similar (and fully simulated) exists, the user is offered the choice to **adopt that whole dataset** (samples + `.in`/`.out` files + signal arrays) into the current session instead of re-simulating.

**Critical engineering constraint** (user-stated, and correct): session configs serialize to near-identical text differing only in numbers — text embeddings are useless. So: **no embedding model**. Vectors are raw, per-dimension-normalized numeric features built deterministically from the config envelope. Qdrant does ANN candidate retrieval with **payload hard filters** for categorically incompatible configs; final similarity comes from an **exact physics-aware rescoring in Python** (interval IoU over ranges + scalar rules), which gates the ≥95% recommendation.

**Decisions made with the user:**

- Granularity: **session-level config envelope** (ranges), not per-sample.
- Store: **new Qdrant collection** (`sim_sessions`) in the already-running docker-compose Qdrant (`QDRANT_URL`, same as `rag.py`).
- Hook: **`start_forward_model`** (`POST /datasets/{sid}/simulate`), before launching gprMax.
- On agreement: **full adoption** — the matched session's samples/signals/files replace the current session's dataset entirely; no desynchronization. From then on those N samples belong to the current session.
- Scope: **global across all users** (recommendation shows source user/session).
- Sample count: **any M recommended**; the message states "M samples vs your requested N"; adoption brings the whole M.
- 100% deterministic — never routed through the LLM agent (CLAUDE.md hard boundary).

**Verified ground truth:**

- `_coerce_uuid(chat.session_id)` is the same UUID for `ExtractionSession.id` and `Simulation.session_id` (api.py:1403, 674, 1410+).
- `ExtractionSession` JSONB (`layers_ranges`, `antenna_waveform`, `model_config_data` incl. `artifacts.output_dir`, `advanced_params`, `num_samples_requested`) holds everything a backfill needs (written in `_finalize_dataset_sync`, api.py:1410-1482).
- Signals are float8[] columns on `Simulation` (db/db.py:219-239) but **api.py currently records only `.out` paths** (`_record_simulation_outputs` → `set_simulation_outputs`, api.py:650-679). Array extraction (`signal_extraction.extract_and_prepare_batch` + `bulk_update_signals`) is wired only in the legacy `parameters_global_state.py` — must be added to the api.py path so signals become reusable.
- Qdrant v1.17 runs in docker-compose; `qdrant-client` already a dependency; new collection just reuses `QDRANT_URL`.
- `emitted_files.json` contains absolute paths (must be rewritten on adoption); `sampled_layers.json` / `derived_layers.json` / `global_derive.json` contain none (copy verbatim).
- No Alembic migration needed — the new per-chat field rides in the `chat_sessions.session_state` JSONB blob.

## 1. New module `backend/sim_similarity.py`

Pure Python; `qdrant_client` imported lazily; failures swallowed by callers (recommendation must never break the simulate path); key-free tests import it without Qdrant running.

```python
COLLECTION_NAME = "sim_sessions"
MAX_LAYERS = 6; MAX_CYLINDERS = 4; MAX_BOXES = 4
VECTOR_DIM = MAX_LAYERS*12 + 6 + MAX_CYLINDERS*6 + MAX_BOXES*8   # 134
REUSE_THRESHOLD = float(os.getenv("SIM_REUSE_THRESHOLD", "0.95"))
REUSE_TOP_K   = int(os.getenv("SIM_REUSE_TOPK", "10"))
REUSE_ENABLED = os.getenv("SIM_REUSE_ENABLED", "1") not in {"0", "false"}

def build_feature_payload(*, dataset_config, layers, target_ranges, waveform, antenna) -> dict
    # canonical JSON-safe numeric dict; deterministic target sort
    # (kind, depth-mid, x_offset-mid, size-mid); ValueError if essentials missing.
    # SAME function feeds indexing, querying, and backfill.
def build_vector(payload: dict) -> list[float]          # fixed 134 dims, zero-padded
def hard_filters(payload: dict) -> dict                 # Qdrant must-conditions
def rescore(a: dict, b: dict) -> tuple[float, list[dict]]  # exact sim [0,1] + per-param breakdown

class SimilarityIndex:
    def __init__(self, url=None)                        # QdrantClient(url or $QDRANT_URL)
    def ensure_collection(self)                         # create-if-missing, EUCLID, payload indexes
    def index_session(self, session_id, payload, meta)  # upsert, point id = session UUID (idempotent)
    def find_similar(self, payload, *, exclude_session_id, threshold, top_k) -> dict | None

# failure-swallowing conveniences used by api.py:
def index_completed_session(state, *, session_id, user_id, num_samples, output_dir) -> bool
def find_similar_session(state, *, session_id) -> dict | None   # None on ANY failure

if __name__ == "__main__":   # python backend/sim_similarity.py backfill
```

### Feature vector (134 dims, fixed normalization constants — never data-derived)

Per-layer block ×6 slots, 12 dims each, zero-padded (safe: `num_layers` is hard-filtered so both sides pad identically): `thickness_m` min/max ÷3.0 · `sand_pct` min/max ÷100 · `clay_pct` min/max ÷100 · `theta_v` min/max ÷0.5 · `bulk_density` (x−1.0)/1.0 · `particle_density` (x−2.3)/0.6.

Waveform/antenna block (6 dims): `center_freq` (log10(f)−7)/3 [10 MHz–10 GHz, **log scale**] · `amplitude` log10(clamp(|A|,0.01,100))/2 · `tx_rx_offset_m` ÷1.0 · `source_height_m` ÷1.0 (0 when None) · `source_height_specified` 0/1 flag · `resistance` ÷1000 (0 when None).

Targets: cylinder slots ×4 ×6 dims (x_offset min/max (x+1)/2 over −1..1 m; depth min/max ÷2.0; radius min/max ÷0.5); box slots ×4 ×8 dims (adds width/height ÷0.5). Zero-padded; counts hard-filtered.

`num_samples` is NOT a vector dim (no physics effect) — it rides in the payload and is reported in the recommendation message.

### Hard filters (Qdrant `must`, indexed)

`num_layers`, `waveform_kind`, `antenna_kind`, `antenna_axis`, `dimensionality`, `n_cylinders`, `n_boxes`, `has_surface_roughness`, and deck-changing grid-policy ints: `cells_per_wavelength`, `pml_cells`, `buffer_cells`, `fractal_nbins`, `high_freq_factor_x100`. Plus `must_not: session_id == current`. Point payload additionally stores (unfiltered): `session_id`, `user_id`, `num_samples`, `output_dir`, `created_at`, and the full **`feature_payload`** dict so rescoring needs no Postgres round-trip.

### Distance + rescoring

- Qdrant distance: **EUCLID** (cosine's scale-invariance is wrong for absolute normalized features). ANN is candidate generation only.
- Rescore per parameter: ranges → **interval IoU** (overlap/union in raw units; both-degenerate-equal ⇒ 1.0); scalars → `max(0, 1 − |x1−x2|/scale)` with scale = the normalization constant; center freq → `max(0, 1 − |Δlog10 f|/0.30)`; one-sided None ⇒ 0, both None ⇒ 1.
- Weights (single constants table at module top): layers 0.50 (per-layer split: thickness .25, theta_v .25, sand .15, clay .15, bulk ρ .10, particle ρ .10) · waveform 0.20 (freq .80, amplitude .10, timing .10) · antenna 0.15 (offset .50, height .30, resistance .20) · targets 0.15 (equal per slot). Both-sessions-zero-targets ⇒ redistribute target weight (no vacuous inflation).
- Gate: rescored `score ≥ REUSE_THRESHOLD` (default 0.95). Candidates ordered by rescore, not ANN distance. `rescore` breakdown feeds the diff display.

### Backfill CLI

`python backend/sim_similarity.py backfill`: iterate `ExtractionSession` rows (new db helper), skip sessions whose `Simulation` rows aren't all `simulation_completed_at`-set, build payload from the JSONB columns, `index_session` each. Idempotent (UUID point ids).

## 2. `db/db.py` helper additions (no migration)

```python
def get_simulations_for_session(session_id) -> List[Simulation]
def get_extraction_session(session_id) -> Optional[ExtractionSession]
def list_extraction_sessions() -> List[ExtractionSession]        # backfill
def count_incomplete_simulations(session_id) -> tuple[int, int]  # (total, not-completed)
```

## 3. `backend/api.py` changes

### 3.1 Signal extraction on run completion (prerequisite — land first)

Extend `_record_simulation_outputs` (api.py:650): after `set_simulation_outputs`, lazily import `signal_extraction.extract_and_prepare_batch` (h5py stays off the import path — precedent at api.py:357), run it on the out_files dir + session UUID, then `bulk_update_signals`. Return `(rows_updated, signals_updated)`; include `signals_updated` in the `simulation_complete` summary in `_run_forward_model` (api.py:619-632). Upload manifests still short-circuit; all failures swallowed.

### 3.2 Index hook (end of `_run_forward_model`)

Only for fully successful generated runs (`manifest.source != "upload"`, `failed == 0`, `total > 0`): `await asyncio.to_thread(sim_similarity.index_completed_session, dict(chat.state), session_id=..., user_id=..., num_samples=result["total"], output_dir=...)` inside `contextlib.suppress(Exception)`. **Adoption does NOT index** (would duplicate the source's point).

### 3.3 Recommend gate in `start_forward_model` (api.py:530)

- Signature gains `force: bool = False` (query param `?force=true`).
- After the `regenerating` guard (api.py:559-563), before `chat.simulating = True`:
  - `if not force and manifest.get("source") != "upload":` run `_find_reuse_candidate(chat)` in a worker thread. If a candidate returns: set `chat.reuse_recommendation`, `_send` a new **`reuse_recommendation`** recorded event (markdown summary + structured `recommendation`), `_persist_chat`, return `{"status": "reuse_recommended", "recommendation": rec}` — run NOT started.
- `_find_reuse_candidate(chat)`: calls `sim_similarity.find_similar_session` (None on any failure ⇒ simulate normally — hard requirement); **verifies adoptability** (source Simulation rows all completed via `count_incomplete_simulations`; source `output_dir` + `emitted_files.json` + `out_files/` exist on disk); returns `{source_session_id, similarity_pct, num_samples, simulated_at, source_output_dir, source_user_id, params_diff}` (worst-N rescore entries). Sample-count difference is reported, never filtered ("has M samples vs your requested N").
- New `ChatSession` field `reuse_recommendation: Optional[dict] = None`; add to `_PERSISTED_CHAT_FIELDS` (api.py:776); add `"reuse": ...` to the `session_restore` payload (~api.py:718-727); add `"reuse_recommendation"` to `RECORDED_EVENT_TYPES` (api.py:189).
- Invalidate (`= None`) in `_finish_dataset` and `_import_deck_zip`.

### 3.4 New endpoint `POST /datasets/{session_id}/adopt`

Body `{"source_session_id": str}`. Guards: 404 unknown session; 409 on `simulating`/`busy`/`regenerating`; 409 if `chat.reuse_recommendation` is absent or its `source_session_id` mismatches (endpoint only executes the recommendation it issued). Sets `chat.simulating = True` around `_adopt_dataset_sync` in a worker thread (reuses the existing 409 umbrella); afterwards emits `dataset_ready` + `simulation_complete` (both already recorded + frontend-handled) + `_send_model_update`, then `_persist_chat`.

### 3.5 `_adopt_dataset_sync(chat, source_session_id)` sequence

1. Resolve `cur_dir` from the session's `DatasetConfig.output_dir`; `src_dir` from the recommendation, re-verified against `get_extraction_session(src).model_config_data["artifacts"]["output_dir"]`. Load `src_dir/emitted_files.json`.
2. Re-verify DB: `get_simulations_for_session(src)` — all rows completed, count matches manifest. **All verification BEFORE any deletion.**
3. Replace files: rmtree current `in_files/` + `out_files/`; copy manifest-listed `.in` (+ matching `.out` by stem, + sibling artifacts) from source; copy `sampled_layers.json`/`derived_layers.json`/`global_derive.json` verbatim; write rewritten `emitted_files.json` (`output_dir`/`in_dir`/per-file `path` → current dirs, add `"adopted_from"`). Filenames keep the source basename (manifest is authoritative; renaming decks is deferred).
4. Replace rows: `delete_simulations_for_session(cur)` then `batch_insert_simulations` with copies of source rows — new `id`, `session_id=cur`, `user_id=chat.user_id`, `created_at=now`, `output_dir` = current, `input/output_file_path` repointed by filename; **signals + `signal_length` + `simulation_completed_at` copy through**. Healing: if source rows lack signal arrays (pre-3.1 runs), run `extract_and_prepare_batch` on the copied `out_files/` + `bulk_update_signals`.
5. `ExtractionSession` + section store + `complete_snapshot` stay the USER's own ranges — adoption swaps dataset artifacts only; a later regeneration re-runs from the user's ranges and overwrites the adoption (documented, consistent).
6. Chat state: rebuild `chat.dataset_result` (status `"adopted"`, `adopted_from`), set `chat.simulation_result` (`succeeded=M, total=M, adopted_from`), clear `reuse_recommendation`, set all `viz_flags` True (copied manifests are now this session's products).

## 4. Frontend

`frontend/app/chatbot.jsx`:

- `handleServerEvent` (~line 264): `reuse_recommendation` → `pushBot(mdToHtml(msg.content))` (informational record only — NOT the legacy `choice_required` chips, which round-trip through the LLM agent).
- `replayToMessage` (~line 36): render `reuse_recommendation` like `agent_message`.
- `session_restore` branch: forward `msg.reuse` through `onSimulationEvent`.

`frontend/app/app.jsx`:

- State `reuseRec`; `runForward(force)` (~line 369) appends `?force=true` and, on `body.status === "reuse_recommended"`, sets `reuseRec` instead of showing progress.
- `adoptDataset()`: `POST /datasets/{sid}/adopt`; on ok clear `reuseRec` (dataset tab/outputs refresh via the existing `dataset_ready`/`simulation_complete` WS handling).
- Compact confirm bar under the run button (CLAUDE.md overlay rule: compact/collapsible, never a permanent panel): "Found a matching simulated dataset — **97% similar**, M samples (you requested N), simulated <date>, by <user>" + collapsible top param diffs + **Reuse results** / **Simulate anyway** (`runForward(true)`) / dismiss.
- `session_restore` in `onSimulationEvent` (~line 446): rehydrate `reuseRec` from `msg.reuse`; clear it in `resetForChatSwitch()`.

## 5. Tests (key-free; Qdrant + Postgres stubbed)

New `backend/tests/test_sim_similarity.py`:

1. `build_vector` determinism, length == VECTOR_DIM, padding zeros, target-sort stability under reordering.
2. `hard_filters` contents.
3. `rescore` math: identical ⇒ 1.0; half-overlap interval IoU spot value; degenerate cases; log-freq rule; one-sided None ⇒ 0; zero-target weight redistribution.
4. `find_similar` with a stubbed QdrantClient: threshold gating (0.94 ⇒ None, 0.96 ⇒ match), exclude-self, ordering by rescore not ANN distance.
5. Qdrant raising ⇒ `find_similar_session` returns None.

Extend `backend/tests/test_simulate.py` (existing `api._new_chat_session` + monkeypatch pattern): 6. Simulate gate: stubbed candidate ⇒ `reuse_recommended` response, no task started, event recorded, field persisted; `force=true` bypasses; upload manifest skips check; `_find_reuse_candidate` raising ⇒ normal run. 7. Adopt endpoint (fabricated source dir in tmp_path; DB helpers monkeypatched): file copies + manifest rewrite; rows re-keyed with signals carried; events recorded; `reuse_recommendation` cleared; viz_flags set. 8. Adopt guards: no/mismatched recommendation ⇒ 409; busy states ⇒ 409; missing source files ⇒ 409 with current dataset untouched (verification-before-rmtree ordering). 9. Index hook: success ⇒ called; partial failure ⇒ not called; raising indexer swallowed. 10. `_record_simulation_outputs` extension: signal extraction wired, upload short-circuit intact, h5py failure swallowed.

## 6. Docs + rollout

- CLAUDE.md: add `sim_similarity.py` to the backend file table + a "Forward-model reuse" paragraph (collection name, env vars `SIM_REUSE_THRESHOLD`/`SIM_REUSE_TOPK`/`SIM_REUSE_ENABLED`, deterministic/no-LLM boundary, adoption-replaces-dataset semantics, backfill CLI).
- After merge: run `python backend/sim_similarity.py backfill` against the dev DB to index existing completed sessions.

## Implementation order

1. `sim_similarity.py` pure functions + `SimilarityIndex` + CLI, with `test_sim_similarity.py`.
2. `db/db.py` helpers.
3. api.py 3.1 (signal extraction on completion) — independently valuable.
4. api.py 3.2–3.5 + `test_simulate.py` extensions.
5. Frontend (chatbot.jsx, app.jsx).
6. CLAUDE.md; manual backfill.

## Verification

- `pytest backend/tests/ -v` (all new tests are key-free; no Qdrant/Postgres needed).
- End-to-end: `docker-compose up -d` (Postgres + Qdrant), start backend, complete a session in the UI, run the forward model on a small N (signals land in `simulations.signal_*`, session indexed in Qdrant — check `curl localhost:6333/collections/sim_sessions`). Start a second chat with near-identical params; click "Run forward model" → recommendation bar appears with similarity % and diffs; **Reuse results** → dataset tab shows adopted files, outputs downloadable, `simulations` rows for the new session carry signals; **Simulate anyway** → normal run. Refresh mid-recommendation → bar rehydrates via `session_restore`.

## Known trade-offs (accepted)

- Adoption is destructive to the current session's drawn dataset (recovery = post-complete regeneration from the user's own ranges, which still lives in the store).
- Adopted filenames keep the source model basename inside the current session's directory (manifest is authoritative; all endpoints resolve through it).
- Similarity weights/normalization constants are heuristic — a single constants table is the tuning point; threshold configurable via `SIM_REUSE_THRESHOLD`.
- Sessions simulated before 3.1 lack signal arrays; adoption heals them from copied `.out` files, and the backfill only indexes fully completed sessions.



The feature is fully implemented and verified: 178/178 backend tests pass, the API imports cleanly with both new routes, and a live run against the real qdrant-client (local mode) confirmed matching, self-exclusion, hard filtering, and idempotent indexing all work.

What was built
Similarity engine — new backend/sim_similarity.py. No embedding model anywhere (as you flagged, near-identical text with different numbers would defeat semantic search). Instead:

A deterministic 134-dim feature vector built from the session's config envelope — layer thickness/sand/clay/θv/density ranges, log-scaled frequency, antenna geometry, target geometry — each dimension normalized by fixed physical constants.
Hard payload filters in Qdrant (sim_sessions collection, EUCLID distance, same QDRANT_URL as RAG) for categorically incompatible configs: layer count, waveform/antenna kind, target counts, grid policy. These never compete numerically.
The 95% gate comes from exact rescoring in Python: interval IoU over ranges + scalar rules, weighted by physics importance (layers 50%, waveform 20%, antenna 15%, targets 15%) — Qdrant only narrows candidates. Threshold/top-k tunable via SIM_REUSE_THRESHOLD / SIM_REUSE_TOPK / SIM_REUSE_ENABLED.
python backend/sim_similarity.py backfill indexes your existing completed sessions from Postgres (run it once after docker-compose up -d).
Pipeline wiring — backend/api.py:

Forward-model completion now also extracts signal arrays into the simulations float8[] columns (previously only .out paths were recorded) and indexes the session into Qdrant on fully successful runs.
POST /datasets/{sid}/simulate checks for a match first (skipped for uploads or ?force=true). On a hit it does not start gprMax — it records a reuse_recommendation chat message with the similarity %, sample counts, and closest parameter differences, and returns reuse_recommended.
New POST /datasets/{sid}/adopt executes the pending recommendation: after verifying the source end-to-end (all rows simulated, files on disk — before touching anything), it replaces this session's dataset wholesale — .in/.out files, manifests (rewritten paths + adopted_from), and Simulation rows re-keyed to this session with signals carried over. No desynchronization: those M samples now fully belong to the current session.
Frontend — a compact green bar appears by the Run button ("97% similar dataset already simulated · 50 samples · by alice") with Reuse results / Simulate anyway / dismiss, plus a collapsible diff list. It survives page refresh via session_restore. Adoption is plain HTTP — it never round-trips through the LLM agent, keeping the deterministic/agentic boundary intact.

One note: the recommendation only ever surfaces sessions whose signals are fully in the DB and whose files still exist on disk — stale Qdrant entries are silently skipped, and if Qdrant is down the simulate path proceeds exactly as before.





AUTO Regressive transformer - next token prediction of the signal given all the previous signals