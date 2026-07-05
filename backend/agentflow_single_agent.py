"""
EXPERIMENTAL single-agent variant of `agentflow_langgraph.py`.

The original pipeline gives each parameter section its own deep agent (six in
total) coordinating through the FastAPI parameter server on port 8100. This
experiment keeps the SAME graph topology but replaces all six with ONE deep
agent on ONE conversation thread, testing whether a single agent with full
conversational memory can handle the whole collection process.

What changes vs agentflow_langgraph.py
---------------------------------------
- **No parameter server.** The agent's tools (`save_section` / `get_section`)
  read/write an in-process `_STORE`. Editing any section at any time is just
  re-saving it — no PATCH machinery. Each agent node syncs the WHOLE store into
  `PipelineState`, so cross-section edits made mid-stage land in state.
- **One agent, one thread.** A single `create_deep_agent` instance with a slim
  system prompt; each graph node injects that section's focused instructions
  (batches, physics constraints, JSON schema) as a stage kickoff message
  (see `single_agent_prompts.py`). The agent retains the entire conversation
  across all stages AND both remediation flows.
- **Completion detection** reads the store directly (`_stage_done`) instead of
  scraping `post_parameters` tool-call args.
- **Remediation** injects the validation errors into the same conversation; the
  agent decides WITH THE USER which section to fix (the manual
  `input("Which section to edit")` menu is gone). Changed sections are detected
  by diffing store snapshots, which drives the resample_after_global routing.
- **Staleness branch.** Because the agent can edit `layers` / `dataset_config`
  / `target_ranges` at any time (even after sampling already ran), the inputs
  used at sampling time are snapshotted; a new conditional edge after
  advanced_params re-runs layer_sampling if they changed.

Quirks (intended):
- A section pre-filled by a cross-edit makes its own later stage complete after
  a single confirm turn, without user input.
- Full history + schema-bearing kickoff messages grow the context over long
  sessions; that memory-handling trade-off is the point of the experiment.
- Store/agent/thread live together in a `SingleAgentSession`: the CLI drives a
  module-level default session (one run per process), while `backend/api.py`
  creates an isolated session per WebSocket chat.

Graph
-----
    START
      -> dataset_config        (agent stage)
      -> layers                (agent stage)
      -> target_ranges         (agent stage)
      -> layer_sampling        (derive; snapshots its inputs)
      -> waveform              (agent stage)
      -> antenna               (agent stage)
      -> sample_validation     (GATE) --fail--> sample_remediation --> sample_validation
      -> advanced_params       (agent stage)
          --samples stale?--> layer_sampling --> peplinski_derive
      -> peplinski_derive      (derive)
      -> global_derive         (derive)
      -> global_validation     (GATE) --fail--> global_remediation
                                   ( layers/dataset_config/target_ranges edit
                                       --> layer_sampling --> derive chain;
                                     other edits --> peplinski_derive )
      -> target_placement      (derive)
      -> dataset_generation    (emit)
      -> END
"""

import copy
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Annotated, Optional

import dotenv
from typing_extensions import TypedDict

# Ensure both the project root (for `backend.*`) and backend/ (for the bare
# `schema` / `dataset_sampling` imports used across the codebase) are on
# sys.path, whether this file is run as a script or imported as
# `backend.agentflow_single_agent` (e.g. from tests).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_BACKEND = Path(__file__).resolve().parent
for _p in (str(_REPO_ROOT), str(_BACKEND)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END

from dataset_sampling.layer_sampler import sample_and_write, read_samples
from dataset_sampling.sample_validation import validate_waveform_antenna
from dataset_sampling.peplinski_derive import derive_and_write, read_aggregate
from dataset_sampling.global_derive import derive_and_write_global, read_global
from dataset_sampling.global_validation import validate_global
from dataset_sampling.target_placement import run_placement
from dataset_sampling.emit import emit_dataset

from schema import (
    DatasetConfig,
    ExtractedLayers,
    ExtractedTargetRanges,
    ExtractedWaveform,
    ExtractedAntenna,
    ExtractedAdvancedParams,
)

from single_agent_prompts import (
    SINGLE_AGENT_SYSTEM_PROMPT,
    SECTION_KICKOFF,
    sample_remediation_message,
    global_remediation_message,
)

dotenv.load_dotenv()

# Section name -> schema used to validate what the agent saves.
SECTION_SCHEMA = {
    "dataset_config": DatasetConfig,
    "layers": ExtractedLayers,
    "target_ranges": ExtractedTargetRanges,
    "waveform": ExtractedWaveform,
    "antenna": ExtractedAntenna,
    "advanced_params": ExtractedAdvancedParams,
}

SECTION_DISPLAY = {
    "dataset_config": "Dataset Configuration",
    "layers": "Layer Extraction",
    "target_ranges": "Buried-Target Range Extraction",
    "waveform": "Waveform Extraction",
    "antenna": "Antenna Extraction",
    "advanced_params": "Advanced Parameters Extraction",
}

# Stages that may legitimately be SKIPPED with an empty payload.
OPTIONAL_SECTIONS = {"target_ranges", "advanced_params"}

# Sections whose values feed the per-sample draw. Unlike the 5-agent pipeline,
# target_ranges is included: sample_and_write draws every object range
# (cylinders + boxes), and the single agent can edit it at any point.
RESAMPLE_SECTIONS = {"layers", "dataset_config", "target_ranges"}

# Sampling inputs snapshotted by layer_sampling for the staleness check.
SAMPLING_INPUT_SECTIONS = ("layers", "dataset_config", "target_ranges")


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class PipelineState(TypedDict, total=False):
    """State threaded through every node.

    The six section keys mirror `_STORE` (each agent node syncs the whole
    store in). The extra keys carry gate results, the sampling-input snapshot
    for the staleness check, and a halt signal.
    """
    dataset_config: Optional[dict]
    layers: Optional[dict]
    target_ranges: Optional[dict]
    waveform: Optional[dict]
    antenna: Optional[dict]
    advanced_params: Optional[dict]

    # Gate results + the errors a failed gate produced (fed to remediation).
    sample_validation_passed: Optional[bool]
    sample_validation_errors: Optional[list]
    global_validation_passed: Optional[bool]
    global_validation_errors: Optional[list]

    # Sampling inputs as they were when layer_sampling last ran; if the agent
    # later edits any of them, the on-disk draws are stale and sampling re-runs.
    sampling_snapshot: Optional[dict]

    # When global remediation edits a sampling-affecting section, this routes
    # the re-derive back through layer_sampling first.
    resample_after_global: Optional[bool]

    # Halt signalling (ONLY a user exit halts; failed gates remediate).
    halted: bool
    halt_reason: Optional[str]


# ---------------------------------------------------------------------------
# In-memory section store + agent tools (replaces the port-8100 server)
#   The store, the two tools bound to it, the deep agent and its conversation
#   thread live together in a SingleAgentSession, so each chat session (CLI
#   run, WebSocket session) gets its own isolated conversation + parameters.
# ---------------------------------------------------------------------------

def _changed_sections(before: dict, after: dict) -> set:
    return {s for s in SECTION_SCHEMA if before.get(s) != after.get(s)}


def _section_is_complete(section: str, model) -> bool:
    """True only when `model` carries the ESSENTIAL data for `section`.

    Pydantic enforces the required scalar fields, but several fields are
    `Optional[...] = None`, so a near-empty payload can still validate. This is
    the extra gate that stops a stage completing on a blank/partial save.
    """
    if section in OPTIONAL_SECTIONS:
        return True
    if section == "dataset_config":
        # output_dir / dimensionality are server-fixed (./dataset, 2D) in
        # save_section, so completeness only hinges on the user-collected core.
        return model.num_samples > 0 and model.center_freq_is_peak is not None
    if section == "layers":
        return model.num_layers > 0 and len(model.layers) > 0
    if section == "waveform":
        return (
            model.waveform_center_freq_hz > 0
            and bool(model.waveform_name)
            and bool(model.waveform_kind)
        )
    if section == "antenna":
        return model.tx_rx_offset_m is not None and bool(model.antenna_axis)
    return True


def _make_section_tools(store: dict):
    """Build the save/get tools as closures over `store`, so every session's
    agent reads and writes its own parameters."""

    @tool
    def save_section(
        section: Annotated[str, "Section name: 'dataset_config', 'layers', 'target_ranges', 'waveform', 'antenna', or 'advanced_params'"],
        payload: Annotated[str, "JSON string of the FULL section conforming to its schema (create or fully replace)"],
    ) -> str:
        """Validate and store the complete parameter set for a section.

        This creates or FULLY REPLACES the section — editing means re-saving the
        whole payload. Invalid payloads are rejected (nothing is stored) and the
        validation error is returned so it can be fixed with the user.
        """
        if section not in SECTION_SCHEMA:
            return json.dumps({
                "error": f"Invalid section '{section}'. Must be one of {sorted(SECTION_SCHEMA)}"
            })
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as e:
            return json.dumps({"error": "invalid_json", "detail": str(e)})
        if section == "dataset_config" and isinstance(data, dict):
            # Server-fixed fields, never user-selected: output lands in the
            # server-local ./dataset, only 2D runs are supported, and OpenMP
            # threading is a deployment concern (None -> gprMax default). Any
            # value the agent passes through is overridden here.
            data["output_dir"] = "./dataset"
            data["dimensionality"] = "2D"
            data["num_threads"] = None
        try:
            model = SECTION_SCHEMA[section].model_validate(data)
        except (ValueError, TypeError) as e:
            return json.dumps({"error": "validation_failed", "detail": str(e)})

        store[section] = model.model_dump()
        if not _section_is_complete(section, model):
            return json.dumps({
                "status": "stored_incomplete",
                "section": section,
                "message": (
                    "The payload is schema-valid and was stored, but essential "
                    "fields are still missing — the stage will not complete until "
                    "they are provided. Keep collecting."
                ),
            })
        return json.dumps({"status": "ok", "section": section, "data": store[section]})

    @tool
    def get_section(
        section: Annotated[str, "Section name: 'dataset_config', 'layers', 'target_ranges', 'waveform', 'antenna', or 'advanced_params'"],
    ) -> str:
        """Retrieve the currently stored parameters for a section."""
        if section not in SECTION_SCHEMA:
            return json.dumps({
                "error": f"Invalid section '{section}'. Must be one of {sorted(SECTION_SCHEMA)}"
            })
        if store[section] is None:
            return json.dumps({
                "error": "section_not_populated",
                "section": section,
                "message": (
                    f"Section '{section}' has not been saved yet. Collect it when "
                    "its stage comes (or now, if the user asks)."
                ),
            })
        return json.dumps(store[section])

    return save_section, get_section


class SingleAgentSession:
    """One collection conversation: a section store, the two tools bound to it,
    ONE deep agent and ONE thread shared by every stage and both remediation
    flows. The CLI uses a module-level default session; the WebSocket API
    (`backend/api.py`) creates one per chat session."""

    def __init__(self):
        self.store: dict[str, Optional[dict]] = {s: None for s in SECTION_SCHEMA}
        self.thread_config = {"configurable": {"thread_id": f"single-agent-{uuid.uuid4()}"}}
        # Messages accumulate on the one thread for the whole run; consumers
        # track what they have already relayed via this counter.
        self.seen = 0
        self._agent = None
        self.save_section, self.get_section = _make_section_tools(self.store)

    def agent(self):
        """Build the agent on first use so the module imports (and the graph
        compiles) without an OPENAI_API_KEY or the heavy RAG dependencies."""
        if self._agent is None:
            from deepagents import create_deep_agent
            from langchain_openai import ChatOpenAI
            from langgraph.checkpoint.memory import InMemorySaver
            from backend.rag import rag_search
            from backend.prompt_library import RAG_SUBAGENT_PROMPT

            rag_subagent = {
                "name": "knowledge-agent",
                "description": (
                    "Geophysics knowledge expert. Answers questions about soil properties, "
                    "dielectric models (Peplinski, Topp, etc.), GPR parameters, FDTD grids, "
                    "waveforms, antennas, PML boundaries, and dataset/batch generation. "
                    "Searches the knowledge base first; falls back to domain expertise if needed."
                ),
                "system_prompt": RAG_SUBAGENT_PROMPT,
                "tools": [rag_search],
            }

            self._agent = create_deep_agent(
                model=ChatOpenAI(model="gpt-4.1-mini", api_key=os.getenv("OPENAI_API_KEY")),
                subagents=[rag_subagent],
                system_prompt=SINGLE_AGENT_SYSTEM_PROMPT,
                checkpointer=InMemorySaver(),
                tools=[self.save_section, self.get_section],
            )
        return self._agent

    def invoke(self, text: str) -> dict:
        return self.agent().invoke(
            {"messages": [HumanMessage(content=text)]}, config=self.thread_config
        )

    def new_ai_texts(self, result: dict) -> list:
        """The AI messages produced since the last call (advances `seen`)."""
        messages = result.get("messages", [])
        texts = [
            msg.content
            for msg in messages[self.seen:]
            if type(msg).__name__ == "AIMessage" and getattr(msg, "content", None)
        ]
        self.seen = len(messages)
        return texts

    def snapshot(self) -> dict:
        return copy.deepcopy(self.store)

    def state_sync(self) -> dict:
        """The whole store as a PipelineState update, so cross-section edits
        made during any stage land in state as soon as the node returns."""
        return {s: copy.deepcopy(self.store[s]) for s in SECTION_SCHEMA}

    def stage_done(self, section: str) -> bool:
        """Stage-completion signal: the section is in the store, schema-valid,
        and carries its essential content. Replaces the tool-call scraping
        (`_captured_section`) of the multi-agent pipeline."""
        data = self.store.get(section)
        if data is None:
            return False
        try:
            model = SECTION_SCHEMA[section].model_validate(data)
        except (ValueError, TypeError):
            return False
        return _section_is_complete(section, model)


# ---------------------------------------------------------------------------
# Module-level default session (CLI entry point + tests). The graph nodes
# below drive this one; backend/api.py builds its own SingleAgentSession per
# WebSocket session instead.
# ---------------------------------------------------------------------------

_DEFAULT_SESSION = SingleAgentSession()

# Aliases kept for the CLI graph nodes and the unit tests.
_STORE = _DEFAULT_SESSION.store
save_section = _DEFAULT_SESSION.save_section
get_section = _DEFAULT_SESSION.get_section


def _store_snapshot() -> dict:
    return _DEFAULT_SESSION.snapshot()


def _state_sync() -> dict:
    return _DEFAULT_SESSION.state_sync()


def _stage_done(section: str) -> bool:
    return _DEFAULT_SESSION.stage_done(section)


def _banner(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def _print_response(result: dict, seen: int = 0) -> int:
    """Print only new messages (after index `seen`). Returns updated count."""
    messages = result.get("messages", [])
    for msg in messages[seen:]:
        kind = type(msg).__name__
        if kind == "AIMessage" and msg.content:
            print(f"\n[Agent]: {msg.content}\n")
        elif kind == "ToolMessage":
            print(f"  [tool:{msg.name}] returned ({len(msg.content)} chars)")
    return len(messages)


# The thread persists across all stages, so `result["messages"]` accumulates
# for the whole run — the session's `seen` counter stops history reprinting.
def _invoke_agent(text: str) -> dict:
    result = _DEFAULT_SESSION.invoke(text)
    _DEFAULT_SESSION.seen = _print_response(result, _DEFAULT_SESSION.seen)
    return result


# ---------------------------------------------------------------------------
# Stage driver + agent nodes
# ---------------------------------------------------------------------------

def _run_stage(section: str, display_name: str, kickoff: str) -> Optional[str]:
    """Drive the agent until the section is complete in the store.

    Returns None on success, or a halt reason if the user exits. If the section
    was already filled by an earlier cross-edit, the kickoff turn lets the
    agent confirm/summarise and the stage completes without user input.
    """
    _invoke_agent(kickoff)
    while not _stage_done(section):
        user_input = input("You: ").strip()
        if not user_input or user_input.lower() in ("quit", "exit"):
            return f"user exited during {display_name}"
        _invoke_agent(user_input)
    return None


def _make_agent_stage_node(section: str):
    display_name = SECTION_DISPLAY[section]

    def node(state: PipelineState) -> dict:
        _banner(f"Starting: {display_name}")
        halt_reason = _run_stage(section, display_name, SECTION_KICKOFF[section])
        if halt_reason:
            print("Exiting pipeline.")
            return {"halted": True, "halt_reason": halt_reason}
        print(f"\n>> {display_name} complete — {section} saved to state.\n")
        return _state_sync()

    return node


# ---------------------------------------------------------------------------
# Remediation
#   Same agent, same thread: the validation errors are injected into the
#   ongoing conversation and the agent fixes whichever section is at fault.
#   Changed sections are detected by diffing store snapshots.
# ---------------------------------------------------------------------------

def _run_remediation(kickoff: str, display_name: str):
    """Inject `kickoff` and loop until the agent has changed at least one
    section and every changed section is complete. Returns
    (changed_sections, halt_reason)."""
    before = _store_snapshot()
    _invoke_agent(kickoff)
    while True:
        changed = _changed_sections(before, _store_snapshot())
        if changed and all(_stage_done(s) for s in changed):
            return changed, None
        user_input = input("You: ").strip()
        if not user_input or user_input.lower() in ("quit", "exit"):
            return set(), f"user exited during {display_name}"
        _invoke_agent(user_input)


def sample_remediation_node(state: PipelineState) -> dict:
    """Fix whatever failed the sample-validation gate (same agent + thread)."""
    errors = state.get("sample_validation_errors") or []
    _banner("Fix Required — Sample Validation")
    changed, halt_reason = _run_remediation(
        sample_remediation_message(errors, _STORE), "sample remediation"
    )
    if halt_reason:
        print("Exiting pipeline.")
        return {"halted": True, "halt_reason": halt_reason}
    print(f"\n>> Updated {sorted(changed)} — will re-validate.\n")
    return {
        **_state_sync(),
        "sample_validation_passed": None,
        "sample_validation_errors": None,
    }


def global_remediation_node(state: PipelineState) -> dict:
    """Fix whatever drove the global-grid (TIER 3) gate failure.

    No orchestrator-side section menu: the agent owns every section, so the
    kickoff message carries the section hints and the agent agrees the fix
    with the user. The snapshot diff tells us whether a sampling-affecting
    section changed (=> re-sample before re-deriving the grid).
    """
    errors = state.get("global_validation_errors") or []
    _banner("Fix Required — Global Validation (TIER 3)")
    changed, halt_reason = _run_remediation(
        global_remediation_message(errors, _STORE), "global remediation"
    )
    if halt_reason:
        print("Exiting pipeline.")
        return {"halted": True, "halt_reason": halt_reason}

    resample = bool(changed & RESAMPLE_SECTIONS)
    tail = "re-sample, re-derive the grid and re-validate" if resample else \
        "re-derive the grid and re-validate"
    print(f"\n>> Updated {sorted(changed)} — will {tail}.\n")
    return {
        **_state_sync(),
        "global_validation_passed": None,
        "global_validation_errors": None,
        "resample_after_global": resample,
    }


# ---------------------------------------------------------------------------
# Deterministic derive / validate nodes
#   Copied from agentflow_langgraph.py; layer_sampling additionally snapshots
#   its inputs for the staleness check.
# ---------------------------------------------------------------------------

def _sampling_inputs(state: PipelineState) -> dict:
    return {s: state.get(s) for s in SAMPLING_INPUT_SECTIONS}


def layer_sampling_node(state: PipelineState) -> dict:
    """Draw num_samples concrete layer sets (+ optional buried target) per sample."""
    _banner("Layer + Target Sampling")

    dataset_cfg = DatasetConfig.model_validate(state["dataset_config"])
    layers = ExtractedLayers.model_validate(state["layers"])

    target_ranges = None
    if state.get("target_ranges") is not None:
        tr = ExtractedTargetRanges.model_validate(state["target_ranges"])
        target_ranges = tr if tr.has_targets else None

    if target_ranges is not None:
        obj_desc = " + ".join(
            f"{len(lst)} {kind}(s)"
            for kind, lst in (("cylinder", target_ranges.cylinders),
                              ("box", target_ranges.boxes))
            if lst
        )
        obj_desc = f" + buried objects ({obj_desc})"
    else:
        obj_desc = ""
    print(
        f"Sampling {dataset_cfg.num_samples} parameter set(s) over "
        f"{len(layers.layers)} layer range(s){obj_desc}..."
    )
    samples, path, warnings = sample_and_write(
        extracted=layers,
        num_samples=dataset_cfg.num_samples,
        output_dir=dataset_cfg.output_dir,
        seed=42,
        target_ranges=target_ranges,
    )
    print(f"  Wrote {len(samples)} sampled parameter set(s) to:\n    {path}")
    if warnings:
        print("\n  Warnings:")
        for w in warnings:
            print(f"    - {w}")
    print()
    return {"sampling_snapshot": copy.deepcopy(_sampling_inputs(state))}


def sample_validation_node(state: PipelineState) -> dict:
    """Validate the waveform + antenna across all samples (GATE)."""
    _banner("Sample Validation")

    dataset_cfg = DatasetConfig.model_validate(state["dataset_config"])
    waveform = ExtractedWaveform.model_validate(state["waveform"])
    antenna = ExtractedAntenna.model_validate(state["antenna"])

    report = validate_waveform_antenna(dataset_cfg, waveform, antenna)

    print(f"Validated waveform and antenna for {report.num_samples} sample(s).")
    if report.warnings:
        print("\n  Warnings:")
        for w in report.warnings:
            print(f"    - {w}")
    if report.errors:
        print("\n  Errors:")
        for e in report.errors:
            print(f"    - {e}")
        print("\n>> Sample validation FAILED — routing to remediation.\n")
        return {
            "sample_validation_passed": False,
            "sample_validation_errors": list(report.errors),
        }

    print("\n>> Sample validation passed.\n")
    return {"sample_validation_passed": True, "sample_validation_errors": None}


def peplinski_derive_node(state: PipelineState) -> dict:
    """Derive in-band eps_r per sample using gprMax's own Peplinski routine."""
    _banner("Peplinski Derive (eps_r)")

    dataset_cfg = DatasetConfig.model_validate(state["dataset_config"])
    waveform = ExtractedWaveform.model_validate(state["waveform"])
    target_ranges = (
        ExtractedTargetRanges.model_validate(state["target_ranges"])
        if state.get("target_ranges") is not None
        else None
    )

    samples = read_samples(dataset_cfg.output_dir)
    _derived, aggregate, path = derive_and_write(
        samples, dataset_cfg, waveform, dataset_cfg.output_dir,
        target_ranges=target_ranges,
    )

    print(
        f"Derived in-band eps_r for {aggregate.num_samples} sample(s) at "
        f"{aggregate.frequency_hz/1e6:.1f} MHz ({aggregate.nbins} bins)."
    )
    print(
        f"  Global eps_r: min={aggregate.eps_r_min:.3f} (driest) "
        f"max={aggregate.eps_r_max:.3f} (wettest)"
    )
    print(f"  Wrote derived values to:\n    {path}\n")
    return {}


def global_derive_node(state: PipelineState) -> dict:
    """Derive the ONE global grid / domain / depth / time window for all samples."""
    _banner("Global Derive (grid / domain / depth / time window)")

    dataset_cfg = DatasetConfig.model_validate(state["dataset_config"])
    waveform = ExtractedWaveform.model_validate(state["waveform"])
    antenna = ExtractedAntenna.model_validate(state["antenna"])
    layers = ExtractedLayers.model_validate(state["layers"])

    aggregate = read_aggregate(dataset_cfg.output_dir)
    grid, path = derive_and_write_global(
        dataset_cfg, waveform, antenna, layers,
        aggregate.eps_r_max, aggregate.eps_r_min,
        dataset_cfg.output_dir,
        smallest_feature_global_m=aggregate.smallest_feature_global_m,
        largest_extent_global_m=aggregate.largest_extent_global_m,
        deepest_target_bottom_global_m=aggregate.deepest_target_bottom_global_m,
        static_x_halfwidth_global_m=aggregate.static_x_halfwidth_global_m,
    )

    print(
        f"Sized one global grid from eps_r [{grid.eps_r_min_global:.3f}, "
        f"{grid.eps_r_max_global:.3f}]:"
    )
    print(f"  dx           = {grid.dx_m*1e3:.3f} mm")
    print(f"  domain (x,y) = {grid.domain_x_m:.3f} x {grid.domain_y_m:.3f} m")
    print(f"  depth        = {grid.depth_z_m:.3f} m")
    print(f"  ground / Tx  = ground_y={grid.ground_y_m:.3f} m  Tx=({grid.tx_x_m:.3f}, {grid.tx_y_m:.3f})  Rx=({grid.rx_x_m:.3f}, {grid.rx_y_m:.3f})")
    print(f"  dt           = {grid.dt_s*1e12:.3f} ps")
    print(f"  time window  = {grid.time_window_s*1e9:.2f} ns")
    print(f"  Wrote global derive to:\n    {path}\n")
    return {}


def global_validation_node(state: PipelineState) -> dict:
    """Validate the single global grid (TIER 3) (GATE)."""
    _banner("Global Validation (TIER 3)")

    dataset_cfg = DatasetConfig.model_validate(state["dataset_config"])
    waveform = ExtractedWaveform.model_validate(state["waveform"])
    antenna = ExtractedAntenna.model_validate(state["antenna"])
    layers = ExtractedLayers.model_validate(state["layers"])
    target_ranges = (
        ExtractedTargetRanges.model_validate(state["target_ranges"])
        if state.get("target_ranges") is not None
        else None
    )
    grid = read_global(dataset_cfg.output_dir)

    report = validate_global(
        grid, dataset_cfg, waveform, antenna, layers,
        target_ranges=target_ranges,
    )

    if report.warnings:
        print("  Warnings:")
        for w in report.warnings:
            print(f"    - {w}")
    if report.skipped:
        print("  Skipped:")
        for s in report.skipped:
            print(f"    - {s}")
    if report.errors:
        print("\n  Errors:")
        for e in report.errors:
            print(f"    - {e}")
        print("\n>> Global validation FAILED — routing to remediation.\n")
        return {
            "global_validation_passed": False,
            "global_validation_errors": list(report.errors),
        }

    print("\n>> Global validation passed.\n")
    return {"global_validation_passed": True, "global_validation_errors": None}


def target_placement_node(state: PipelineState) -> dict:
    """Validate each sample's DYNAMIC buried objects against the FIXED global
    grid (static objects were validated once at the global gate)."""
    target_ranges = None
    if state.get("target_ranges") is not None:
        tr = ExtractedTargetRanges.model_validate(state["target_ranges"])
        target_ranges = tr if tr.has_targets else None
    if target_ranges is None:
        return {}  # no buried objects -> nothing to place

    _banner("Per-Sample Target Placement")

    dataset_cfg = DatasetConfig.model_validate(state["dataset_config"])
    grid = read_global(dataset_cfg.output_dir)

    result = run_placement(dataset_cfg.output_dir, dataset_cfg, grid, target_ranges, seed=1234)

    print(
        f"Placed objects: {result.n_unchanged} sample(s) kept as-is, "
        f"{result.n_redrawn} object(s) re-drawn, "
        f"{len(result.dropped)} sample(s) dropped."
    )
    if result.dropped:
        print("  Dropped samples (dataset N reduced):")
        for d in result.dropped:
            print(f"    - sample {d['sample_id']}: {d['reason']}")
    print()
    return {}


def dataset_generation_node(state: PipelineState) -> dict:
    """Emit one gprMax .in file per surviving sample onto the global grid."""
    _banner("Dataset Generation / Emission")

    dataset_cfg = DatasetConfig.model_validate(state["dataset_config"])
    waveform = ExtractedWaveform.model_validate(state["waveform"])
    antenna = ExtractedAntenna.model_validate(state["antenna"])
    advanced = (
        ExtractedAdvancedParams.model_validate(state["advanced_params"])
        if state.get("advanced_params") is not None
        else None
    )

    result = emit_dataset(
        dataset_cfg.output_dir, dataset_cfg, waveform, antenna, advanced
    )

    print(f"Emitted {result.n_written} gprMax .in file(s) to:\n    {result.in_dir}")
    if result.errors:
        print("\n  Errors (samples skipped):")
        for e in result.errors:
            print(f"    - {e}")
    print()
    return {}


# ---------------------------------------------------------------------------
# Edge routers
# ---------------------------------------------------------------------------

def _linear_route(next_node: str):
    """Router for a non-branching step: go to `next_node`, or END if halted."""
    def router(state: PipelineState) -> str:
        return END if state.get("halted") else next_node
    return router


def _sample_gate(state: PipelineState) -> str:
    """After sample validation: advance on pass, remediate on fail, END on exit."""
    if state.get("halted"):
        return END
    return "advanced_params" if state.get("sample_validation_passed") else "sample_remediation"


def _global_gate(state: PipelineState) -> str:
    """After global validation: advance on pass, remediate on fail, END on exit."""
    if state.get("halted"):
        return END
    return "target_placement" if state.get("global_validation_passed") else "global_remediation"


def _after_sampling(state: PipelineState) -> str:
    """First pass: gate 1 hasn't run yet -> continue to waveform. Any re-entry
    (staleness re-sample or global remediation) means gate 1 already passed ->
    jump straight to the derive chain."""
    return "peplinski_derive" if state.get("sample_validation_passed") else "waveform"


def _samples_stale(state: PipelineState) -> bool:
    """True when the on-disk draws no longer match the current sampling inputs
    (the agent edited layers/dataset_config/target_ranges after sampling ran)."""
    snap = state.get("sampling_snapshot")
    return snap is None or snap != _sampling_inputs(state)


def _route_after_advanced(state: PipelineState) -> str:
    """After the last agent stage: re-sample if a cross-edit made the on-disk
    draws stale, otherwise proceed to the derive chain."""
    if state.get("halted"):
        return END
    return "layer_sampling" if _samples_stale(state) else "peplinski_derive"


def _after_global_remediation(state: PipelineState) -> str:
    """Route a global fix back through layer_sampling when it changed the draws,
    otherwise re-derive directly."""
    if state.get("halted"):
        return END
    return "layer_sampling" if state.get("resample_after_global") else "peplinski_derive"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph():
    g = StateGraph(PipelineState)

    # Agent stage nodes — all six drive the SAME agent on the SAME thread.
    for section in SECTION_SCHEMA:
        g.add_node(section, _make_agent_stage_node(section))

    # Deterministic derive / validate nodes
    g.add_node("layer_sampling", layer_sampling_node)
    g.add_node("sample_validation", sample_validation_node)
    g.add_node("sample_remediation", sample_remediation_node)
    g.add_node("peplinski_derive", peplinski_derive_node)
    g.add_node("global_derive", global_derive_node)
    g.add_node("global_validation", global_validation_node)
    g.add_node("global_remediation", global_remediation_node)
    g.add_node("target_placement", target_placement_node)
    g.add_node("dataset_generation", dataset_generation_node)

    # ---- Edges (in process order) ----
    g.add_edge(START, "dataset_config")

    g.add_conditional_edges("dataset_config", _linear_route("layers"), ["layers", END])
    g.add_conditional_edges("layers", _linear_route("target_ranges"), ["target_ranges", END])
    g.add_conditional_edges("target_ranges", _linear_route("layer_sampling"), ["layer_sampling", END])

    # Sampling runs after the target-range mini-stage on the first pass. It is
    # re-entered for staleness re-samples and for global remediation edits, in
    # which case it jumps straight to the derive chain.
    g.add_conditional_edges("layer_sampling", _after_sampling, ["waveform", "peplinski_derive"])

    g.add_conditional_edges("waveform", _linear_route("antenna"), ["antenna", END])
    g.add_conditional_edges("antenna", _linear_route("sample_validation"), ["sample_validation", END])

    # Gate 1: waveform/antenna validation.
    g.add_conditional_edges(
        "sample_validation", _sample_gate,
        ["advanced_params", "sample_remediation", END],
    )
    g.add_edge("sample_remediation", "sample_validation")

    # Last agent stage. NEW vs the multi-agent pipeline: if a cross-section
    # edit made the on-disk samples stale, re-sample before deriving.
    g.add_conditional_edges(
        "advanced_params", _route_after_advanced,
        ["peplinski_derive", "layer_sampling", END],
    )
    g.add_edge("peplinski_derive", "global_derive")
    g.add_edge("global_derive", "global_validation")

    # Gate 2: TIER-3 global grid validation.
    g.add_conditional_edges(
        "global_validation", _global_gate,
        ["target_placement", "global_remediation", END],
    )
    g.add_conditional_edges(
        "global_remediation", _after_global_remediation,
        ["layer_sampling", "peplinski_derive", END],
    )

    g.add_edge("target_placement", "dataset_generation")
    g.add_edge("dataset_generation", END)

    return g.compile()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_pipeline():
    # No parameter server: the store is in-process and the single agent's
    # tools write to it directly.
    graph = build_graph()
    final_state = graph.invoke(
        {"halted": False},
        config={"recursion_limit": 100},
    )

    if final_state.get("halted"):
        print(f"\n{'='*60}")
        print("  PIPELINE HALTED")
        print(f"{'='*60}")
        print(f"  Reason: {final_state.get('halt_reason')}")
        print("  Fix the inputs above and re-run.\n")
        return

    print(f"\n{'='*60}")
    print("  Pipeline complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_pipeline()
