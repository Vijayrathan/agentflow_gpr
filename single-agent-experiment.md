# Single-Agent Extraction Pipeline (Experiment)

## Context

The current pipeline ([backend/agentflow_langgraph.py](backend/agentflow_langgraph.py)) uses 5–6 isolated deep agents, one per parameter section, coordinating through a FastAPI parameter server (port 8100) with POST/GET/PATCH tools. This experiment tests whether **one agent with a single persistent conversation thread** can handle the entire collection process — with better parameter retention, and cross-section edits becoming trivial state updates instead of PATCH machinery.

**User decisions (confirmed):** direct in-memory state tools (no port-8100 server); single thread + full history for the whole run; slim system prompt + per-section kickoff-message injection; the same agent/thread also handles both remediation flows (killing the manual `input()` section menu in global remediation).

**New files only — no existing code is modified:**

1. `backend/single_agent_prompts.py` — slim system prompt, per-section kickoff messages, remediation message builders
2. `backend/agentflow_single_agent.py` — store + tools, the single agent, the LangGraph graph, entry point
3. `backend/tests/test_single_agent_store.py` — key-free unit tests

## File 1: `backend/single_agent_prompts.py`

Imports: the six schemas from `backend.schema`; reuse `schema_to_json` from [backend/prompt_library.py:11](backend/prompt_library.py#L11). Do **not** import the six `*_AGENT_PROMPT` constants (they bake in post/patch_parameters instructions); instead copy the reusable text out of the `_make_agent_prompt(...)` call sites — `batch_descriptions`, `skip_policy`, physics constraints, advanced_params' `extra_intro` — reworded for `save_section`.

**`SINGLE_AGENT_SYSTEM_PROMPT`** (slim, always present):

- Role: ONE agent collecting ALL six sections stage by stage (list sections in pipeline order, one line each)
- Stage protocol: orchestrator injects `=== STAGE: … ===` messages with that section's batches, constraints, and JSON schema; a stage completes only when a schema-valid section with essential fields is saved
- Tools: `save_section(section, payload)` = validate + store FULL section (create or replace; error return ⇒ fix with user and retry; `stored_incomplete` ⇒ keep collecting); `get_section(section)` = read back. To edit anything: get → modify → save the complete payload (never partial; nulls overwrite)
- Cross-section edits: allowed anytime; warn that editing `layers`/`dataset_config`/`target_ranges` after sampling triggers re-sampling
- Remediation: "VALIDATION FAILED" messages may arrive; explain in plain language, agree a fix, re-save the offending section(s)
- Knowledge questions: delegate to `knowledge-agent` subagent via the `task` tool
- Skips: `advanced_params` skip ⇒ save `"{}"`; no target ⇒ save `{"cylinder": null}`

**`SECTION_KICKOFF: dict[str, str]`** — built by a `_stage_message(section, title, batches, skip_policy, schema_class, extra="")` helper mirroring `_make_agent_prompt`'s shape: stage banner, batch instructions, "NEVER guess", skip policy, `schema_to_json(schema_class)` block, "begin by asking the first batch". Batch text copied from prompt_library per-agent call sites (layers ~173-195, waveform ~219-237, antenna ~262-284, dataset_config ~310-339, target_ranges ~367-390, advanced_params ~417-465).

**Remediation builders** `sample_remediation_message(errors, store)` / `global_remediation_message(errors, store)` — adapted from `_remediation_message` ([agentflow_langgraph.py:318](backend/agentflow_langgraph.py#L318)) and the hint block at lines 382-386: list errors, dump current values of candidate sections (sample: dataset_config/waveform/antenna; global: + layers/advanced_params), explain `[tag]`→section mapping, instruct to re-save FULL section(s), note which edits trigger re-sampling.

## File 2: `backend/agentflow_single_agent.py`

### Imports / paths

Insert repo root AND `backend/` into `sys.path` (pattern in `backend/dataset_sampling/tests/test_target_geometry.py:14-26`), then follow the existing bare-import convention: `from schema import ...`, `from dataset_sampling.layer_sampler import ...` etc. (same seven functions as [agentflow_langgraph.py:77-83](backend/agentflow_langgraph.py#L77-L83)). **Do not import agentflow_langgraph** — importing it constructs all six agents. Copy the thin wrappers instead.

### Store + tools (replaces the parameter server)

```python
_STORE: dict[str, Optional[dict]] = {s: None for s in SECTION_SCHEMA}
_store_snapshot()            # deepcopy
_changed_sections(before, after) -> set[str]
_state_sync() -> dict        # whole store -> PipelineState update
```

- `@tool save_section(section, payload)`: validate JSON against `SECTION_SCHEMA[section]`; on failure return the error text WITHOUT storing (agent self-corrects — same contract as post_parameters); on success store `model_dump()`, and if `_section_is_complete` (copied verbatim from [agentflow_langgraph.py:175-203](backend/agentflow_langgraph.py#L175-L203)) is False return `stored_incomplete` status.
- `@tool get_section(section)`: read `_STORE`, or `section_not_populated` error (mirror error style in parameters_global_state.py).
- `_stage_done(section)`: store entry present + re-validates + `_section_is_complete`. Replaces tool-call scraping (`_captured_section`) entirely.
- Constants copied: `SECTION_SCHEMA`, `OPTIONAL_SECTIONS`; `RESAMPLE_SECTIONS = {"layers", "dataset_config", "target_ranges"}` — **extended with target_ranges**, since `sample_and_write` consumes `target_ranges.cylinder` and the single agent can edit it anytime.

### The single agent (one thread)

- `rag_subagent` dict copied from `extraction_agents/layer_extraction.py:31-41` (`rag_search` from `backend.rag`, `RAG_SUBAGENT_PROMPT` from `backend.prompt_library`).
- Lazy construction: `_get_agent()` builds `create_deep_agent(model=ChatOpenAI("gpt-4.1-mini"), subagents=[rag_subagent], system_prompt=SINGLE_AGENT_SYSTEM_PROMPT, checkpointer=InMemorySaver(), tools=[save_section, get_section])` on first call — module imports & graph compile without an API key.
- ONE `_THREAD_CONFIG = {"configurable": {"thread_id": f"single-agent-{uuid4()}"}}` for the entire run.
- `_invoke_agent(text)` with a module-level `_SEEN` counter (messages accumulate across the whole run on one thread; per-stage `seen=0` reset would reprint history). `_print_response` / `_banner` copied.

### Stage driver + agent nodes

```python
_run_stage(section, display, kickoff)   # mirrors _run_agent_collect: inject kickoff,
                                        # loop input() until _stage_done(section); quit/exit halts
_make_agent_stage_node(section, display)  # returns _state_sync() — WHOLE store, so a
                                          # cross-section edit made mid-stage lands in state
```

Note in docstring: a section pre-filled during an earlier stage makes its own stage complete after one confirm turn — intended.

### PipelineState

Copy from [agentflow_langgraph.py:109-138](backend/agentflow_langgraph.py#L109-L138) + one new key: `sampling_snapshot: Optional[dict]` (the layers/dataset_config/target_ranges dicts captured when sampling ran).

### Deterministic nodes

Copy verbatim: `sample_validation_node`, `peplinski_derive_node`, `global_derive_node`, `global_validation_node`, `target_placement_node`, `dataset_generation_node`. `layer_sampling_node` gets one change — return `{"sampling_snapshot": deepcopy(_sampling_inputs(state))}`.

### Staleness re-sampling (cross-section-edit safety)

`peplinski_derive` is the first consumer of on-disk samples (`sample_validation` only takes cfg/waveform/antenna — verified). So one new conditional edge after advanced_params:

```python
_samples_stale(state)      # sampling_snapshot missing or != current sampling inputs
_route_after_advanced      # halted -> END | stale -> layer_sampling | else -> peplinski_derive
_after_sampling            # "peplinski_derive" if state.get("sample_validation_passed") else "waveform"
                           # (marker-based; on first pass gate 1 hasn't run -> waveform;
                           #  any re-entry means gate 1 passed -> derive chain)
```

### Remediation (same agent, same thread)

```python
_run_remediation(kickoff, display) -> (changed_sections, halt_reason)
```

Snapshot store → inject error message → loop until ≥1 section changed and all changed sections pass `_stage_done` (or are optional); quit/exit halts. Then:

- `sample_remediation_node`: reset gate flags, return `_state_sync()`.
- `global_remediation_node`: `resample = bool(changed & RESAMPLE_SECTIONS)`; return `_state_sync()` + reset flags + `resample_after_global`. No `input()` menu — the kickoff message carries the section hints; the agent decides with the user.

### Graph — identical topology to the original except one new branch

```
START → dataset_config → layers → target_ranges → layer_sampling
layer_sampling —_after_sampling→ waveform | peplinski_derive
waveform → antenna → sample_validation
sample_validation —_sample_gate→ advanced_params | sample_remediation | END
sample_remediation → sample_validation
advanced_params —_route_after_advanced→ peplinski_derive | layer_sampling | END   ← NEW staleness branch
peplinski_derive → global_derive → global_validation
global_validation —_global_gate→ target_placement | global_remediation | END
global_remediation —_after_global_remediation→ layer_sampling | peplinski_derive | END
target_placement → dataset_generation → END
```

Routers `_linear_route`, `_sample_gate`, `_global_gate`, `_after_global_remediation` copied unchanged (lines 651-684).

### Entry point

`run_pipeline()` — same as original minus `start_parameter_server()`; `if __name__ == "__main__": run_pipeline()`. Run: `python backend/agentflow_single_agent.py`.

## File 3: `backend/tests/test_single_agent_store.py` (no API key needed)

- `save_section` rejects bad JSON / schema-violating payloads (e.g. `theta_v_max > porosity`) and leaves `_STORE` untouched
- valid full `dataset_config` ⇒ `_stage_done` True; near-empty valid payload ⇒ `stored_incomplete` + `_stage_done` False
- `_changed_sections`, `_samples_stale`, `_after_sampling`, `_route_after_advanced` on hand-built state dicts
- resample detection: mutate `layers` between snapshots ⇒ `changed & RESAMPLE_SECTIONS` truthy
- Reset `_STORE` between tests (fixture)

## Verification

1. `pytest backend/tests/test_single_agent_store.py -v` (key-free)
2. Import/compile smoke: import the module and call `build_graph()` — must succeed without `OPENAI_API_KEY` (lazy agent)
3. Full interactive run (needs key): `python backend/agentflow_single_agent.py` with the scenario: 2 samples / 2D / one sandy layer (sand 40%, clay 10%, θv 0.05–0.20, bulk 1.5, particle 2.66 g/cm³) / skip target / ricker 900 MHz / hertzian dipole, 0.1 m offset / skip advanced. Verify the same artifacts as the original in `output_dir` (sampled layers manifest, peplinski aggregate, global derive JSON, emitted `.in` files).
4. Exercise the experiment's selling points: mid-antenna-stage say "change layer 1 thickness to 0.8–1.2 m" → confirm the staleness branch re-runs `layer_sampling`; force a gate-1 failure (200 MHz, outside Peplinski 0.3–1.3 GHz band) → confirm remediation happens in the SAME conversation and re-validates.

## Known trade-offs (document in module docstring)

- Full history + six schema-bearing kickoff messages grows context over long sessions — acceptable for the experiment; the memory-handling comparison is the point.
- Module-global store ⇒ one run per process (same property as the FastAPI `_store`).

# Single-Agent Experiment

The single-agent experiment is implemented in three new files, existing code untouched, and everything that can be verified without a live LLM session passes.

## What was built

### `backend/agentflow_single_agent.py`

The experimental pipeline. Same LangGraph topology as `agentflow_langgraph.py`, but all six collection stages drive one `create_deep_agent` instance on one `thread_id`, so the agent retains the entire conversation across every stage and both remediation flows. The port-8100 parameter server is gone:

- `save_section` / `get_section` tools read and write an in-process `_STORE`. Saving validates against the section's Pydantic schema (invalid ⇒ error returned, nothing stored) and flags `stored_incomplete` when essential fields are missing. Editing any section at any time is just re-saving it — no PATCH logic.
- Stage completion is checked directly against the store (`_stage_done`) instead of scraping `post_parameters` tool-call args.
- Every agent node syncs the whole store into `PipelineState`, so cross-section edits made mid-stage (e.g. tweaking layers while collecting the antenna) land in state immediately.
- Because the agent can edit sampling inputs after sampling already ran, `layer_sampling` snapshots its inputs and a new conditional edge after `advanced_params` re-runs sampling if `layers`/`dataset_config`/`target_ranges` changed (also added `target_ranges` to `RESAMPLE_SECTIONS` — the original omitted it only because its remediation menu never offered it).
- Both remediation nodes inject the validation errors into the same conversation; the agent decides the fix with the user, and changed sections are detected by diffing store snapshots — this replaces the manual `input("Which section to edit")` menu and drives `resample_after_global`.
- The agent is built lazily, so the module imports and the graph compiles without `OPENAI_API_KEY` or the heavy RAG imports.

### `backend/single_agent_prompts.py`

The prompt-injection layer: a slim ~3 KB always-present system prompt (role, tools, stage protocol, cross-edit and remediation rules), plus `SECTION_KICKOFF` messages injected per stage carrying that section's field batches, physics constraints and JSON schema (guidance text carried over from `prompt_library.py`, reworded for `save_section`), and the two remediation message builders.

### `backend/tests/test_single_agent_store.py`

14 key-free tests covering tool validation/rejection, completeness gating, snapshot diffing, staleness routing, and graph compilation.

## Verification

- `pytest backend/tests/ -v`: 16 passed (14 new + the 2 pre-existing)
- Graph compiles with `OPENAI_API_KEY` unset, in script-mode import — the lazy-agent path works

## What I couldn't do autonomously

The live conversational run. To try it: `python backend/agentflow_single_agent.py`, then the interesting probes are:

1. Mid-antenna-stage say "actually, change layer 1 thickness to 0.8–1.2 m" and confirm the staleness branch re-runs sampling before `peplinski_derive`.
2. Give a 200 MHz center frequency (outside the Peplinski 0.3–1.3 GHz band) and confirm remediation happens inside the same conversation.
