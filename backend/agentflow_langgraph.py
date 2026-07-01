"""
LangGraph port of the agentflow.py pipeline.

This is a TEMPORARY / experimental rewrite of `run_pipeline()` from
`agentflow.py`. The original drives the stages imperatively and shuttles state
between them by POSTing/GETting from the FastAPI parameter server. Here every
stage — each extraction agent call and each deterministic derive/validate step —
is a LangGraph **node**, and they are wired together with **edges** in exactly
the order the real process runs.

Key difference from agentflow.py
--------------------------------
We stop using the API as the cross-stage source of truth. Instead the LangGraph
`PipelineState` carries the extracted sections (and gate results) between nodes.

The deep agents still own their own tools (`post_parameters`, etc.) which talk
to the FastAPI server, so the server is started before the run — but the
*pipeline* never GETs state back from it. After an agent finishes a section we
read the section straight out of the agent's `post_parameters` tool call,
validate it against its schema, and drop it into `PipelineState`. Every
downstream node reads from state, not from httpx.

A failed validation gate does NOT crash/END the run. It routes to a remediation
node that re-engages the agent owning the bad value (it explains the problem to
the user, agrees a fix, re-posts the section) and then loops back to re-validate.
Only a user "exit" ends the run early.

Graph
-----
    START
      -> dataset_config        (agent)
      -> layers                (agent)
      -> target_ranges         (agent)
      -> layer_sampling        (derive)
      -> waveform              (agent)
      -> antenna               (agent)
      -> sample_validation     (validate, GATE) --fail--> sample_remediation --> sample_validation
      -> advanced_params       (agent)
      -> peplinski_derive      (derive)
      -> global_derive         (derive)
      -> global_validation     (validate, GATE) --fail--> global_remediation
                                   ( layers/dataset_config edit --> layer_sampling --> derive chain;
                                     other edits --> peplinski_derive )
      -> target_placement      (derive)
      -> dataset_generation    (disabled stub)
      -> END
"""

import json
import re
import sys
import uuid
from json import JSONDecodeError
from pathlib import Path
from typing import Optional

from pydantic import ValidationError
from typing_extensions import TypedDict

# Ensure the project root is on sys.path so `backend.*` and `db.*` imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

from parameters_global_state import start_parameter_server
from extraction_agents.dataset_config_extraction import agent as dataset_config_agent
from extraction_agents.layer_extraction import agent as layer_agent
from extraction_agents.target_extraction import agent as target_agent
from extraction_agents.waveform_extraction import agent as waveform_agent
from extraction_agents.antenna_extraction import agent as antenna_agent
from extraction_agents.advanced_params_extraction import agent as advanced_agent

# Add dataset_sampling to sys.path so its bare imports work
sys.path.insert(0, str(Path(__file__).parent / "dataset_sampling"))

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

# Section name -> schema used to validate what the agent posted.
SECTION_SCHEMA = {
    "dataset_config": DatasetConfig,
    "layers": ExtractedLayers,
    "target_ranges": ExtractedTargetRanges,
    "waveform": ExtractedWaveform,
    "antenna": ExtractedAntenna,
    "advanced_params": ExtractedAdvancedParams,
}


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class PipelineState(TypedDict, total=False):
    """State threaded through every node.

    The six section keys hold the `model_dump()` of each extracted section
    (the replacement for the FastAPI `_store`). The remaining keys carry
    gate results and a halt signal so conditional edges can route to END.
    """
    # Collected sections (serialised Pydantic models)
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

    # When global remediation edits a sampling-affecting section (layers /
    # dataset_config), this routes the re-derive back through layer_sampling
    # first so the on-disk draws are regenerated before the grid is re-derived.
    resample_after_global: Optional[bool]

    # Halt signalling (ONLY a user exit halts now; a failed gate routes to
    # remediation instead of ending the run).
    halted: bool
    halt_reason: Optional[str]


# ---------------------------------------------------------------------------
# Small helpers (ported from agentflow.py)
# ---------------------------------------------------------------------------

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


# Stages that may legitimately be SKIPPED with an empty payload. Everything
# else is mandatory: the pipeline must not advance past it without real,
# schema-valid values. (Matches the original pipeline, which guards these two
# with `if state.get(...) is not None`.)
OPTIONAL_SECTIONS = {"target_ranges", "advanced_params"}

# Sections whose values feed the per-sample draw. Editing one of these during
# global remediation requires regenerating the samples (via layer_sampling)
# before the grid is re-derived.
RESAMPLE_SECTIONS = {"layers", "dataset_config"}


def _section_is_complete(section: str, model) -> bool:
    """True only when `model` carries the ESSENTIAL data for `section`.

    Pydantic enforces the required scalar fields, but several fields are
    `Optional[...] = None`, so a near-empty payload can still validate. This is
    the extra gate that stops the pipeline advancing on a blank/partial post.

    Mandatory stages must have their core content; the two optional stages may
    be empty (that is the user explicitly choosing to skip them).
    """
    if section in OPTIONAL_SECTIONS:
        return True
    if section == "dataset_config":
        return (
            model.num_samples > 0
            and model.dimensionality in ("2D", "3D")
            and model.center_freq_is_peak is not None
        )
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


def _captured_section(result: dict, section: str) -> Optional[dict]:
    """Return the agent's posted `section` payload as a validated dict, else None.

    This is the stage-completion signal. We persist to LangGraph state, NOT the
    parameter server, so completion does NOT depend on the server accepting the
    post — it can't for sections the server doesn't know (e.g. `target_ranges`),
    and we don't want to GET state back from it anyway. We read the payload the
    agent passed to `post_parameters` straight from its tool-call args, then
    require it to (1) pass the section's Pydantic schema LOCALLY and (2) carry
    the section's essential content.

    Returning None means "not done yet — keep collecting", which is exactly
    what we want whenever the value is wrong, validation failed, or essential
    parameters are missing:
      * an invalid post (e.g. layer theta_v_max > porosity) fails validation;
      * a blank/partial post (optional fields all None) fails the completeness
        gate;
      * a valid, complete post for an API-unknown section is still captured,
        even though the server rejected it.

    Uses the LATEST post for the section (the agent's most recent intent).
    """
    messages = result.get("messages", [])
    payload = None
    for msg in messages:
        for tc in (getattr(msg, "tool_calls", None) or []):
            if tc.get("name") != "post_parameters":
                continue
            args = tc.get("args") or {}
            if args.get("section") == section:
                payload = args.get("payload")
    if payload is None:
        return None
    try:
        data = json.loads(payload) if isinstance(payload, str) else payload
        model = SECTION_SCHEMA[section].model_validate(data)
    except (ValueError, TypeError):
        # Bad JSON or schema/physics violation -> not done yet; keep collecting.
        return None
    if not _section_is_complete(section, model):
        # Validated but essential values missing -> do NOT proceed.
        return None
    return model.model_dump()


# section -> (agent, human-readable stage name). Used by both the collect nodes
# and the remediation nodes (to re-engage the agent that owns a bad value).
SECTION_AGENTS = {
    "dataset_config": (dataset_config_agent, "Dataset Configuration"),
    "layers": (layer_agent, "Layer Extraction"),
    "target_ranges": (target_agent, "Buried-Target Range Extraction"),
    "waveform": (waveform_agent, "Waveform Extraction"),
    "antenna": (antenna_agent, "Antenna Extraction"),
    "advanced_params": (advanced_agent, "Advanced Parameters Extraction"),
}


# ---------------------------------------------------------------------------
# Agent driver
#   Shared by the collect nodes and the remediation nodes: run an agent's
#   interactive loop until it posts a valid `section`, or the user exits.
# ---------------------------------------------------------------------------

def _run_agent_collect(agent, section: str, display_name: str, init_message: str):
    """Drive `agent` interactively until it posts a schema-valid `section`.

    Returns (captured_dict, halt_reason): on success captured_dict is the
    validated section and halt_reason is None; if the user exits, captured_dict
    is None and halt_reason explains why. Completion does NOT require the
    parameter server to accept the post — we persist to LangGraph state.
    """
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = agent.invoke({"messages": [HumanMessage(content=init_message)]}, config=config)
    seen = _print_response(result)

    captured = _captured_section(result, section)
    while captured is None:
        user_input = input("You: ").strip()
        if not user_input or user_input.lower() in ("quit", "exit"):
            return None, f"user exited during {display_name}"
        result = agent.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
        seen = _print_response(result, seen)
        captured = _captured_section(result, section)
    return captured, None


# ---------------------------------------------------------------------------
# Agent (collect) nodes
#   One node per extraction stage.
# ---------------------------------------------------------------------------

def _make_agent_node(agent, section: str, display_name: str, init_message: str):
    def node(state: PipelineState) -> dict:
        _banner(f"Starting: {display_name}")
        captured, halt_reason = _run_agent_collect(agent, section, display_name, init_message)
        if captured is None:
            print("Exiting pipeline.")
            return {"halted": True, "halt_reason": halt_reason}
        print(f"\n>> {display_name} complete — {section} saved to state.\n")
        return {section: captured}

    return node


# ---------------------------------------------------------------------------
# Remediation
#   A failed validation gate does NOT end the run. It routes here: the agent
#   that owns the offending parameters is re-engaged with the error text and the
#   current values, explains the problem to the user, helps choose a fix, and
#   re-posts the section. Control then loops back to re-run the validation.
# ---------------------------------------------------------------------------

def _remediation_message(section: str, errors: list, current: Optional[dict]) -> str:
    """Build the message handed to an agent so it can explain + fix `section`."""
    err_lines = "\n".join(f"  - {e}" for e in errors)
    current_json = json.dumps(current, indent=2) if current else "(not set)"
    return (
        f"A downstream validation check FAILED for the '{section}' parameters you "
        f"are responsible for:\n{err_lines}\n\n"
        f"The currently stored '{section}' values are:\n{current_json}\n\n"
        "Please: (1) explain the problem to the user in plain language, including "
        "what range/value would satisfy it; (2) agree a corrected value with the "
        f"user; (3) re-post the FULL '{section}' section (with the fix) using "
        "post_parameters. Do not stop until you have re-posted a corrected section."
    )


def _sections_from_tags(errors: list, allowed: set) -> list:
    """Pull the [section] tags off error strings, keeping only `allowed` ones."""
    found = []
    for e in errors:
        m = re.match(r"\s*\[([a-z_]+)\]", e)
        if m and m.group(1) in allowed and m.group(1) not in found:
            found.append(m.group(1))
    return found


def sample_remediation_node(state: PipelineState) -> dict:
    """Fix the waveform/antenna inputs that failed the sample-validation gate."""
    errors = state.get("sample_validation_errors") or []
    _banner("Fix Required — Sample Validation")

    allowed = {"dataset_config", "waveform", "antenna"}
    sections = _sections_from_tags(errors, allowed) or ["waveform"]

    # Reset the gate so the re-run re-evaluates from scratch.
    updates: dict = {"sample_validation_passed": None, "sample_validation_errors": None}
    for section in sections:
        agent, display = SECTION_AGENTS[section]
        current = {**state, **updates}.get(section)
        captured, halt_reason = _run_agent_collect(
            agent, section, display, _remediation_message(section, errors, current)
        )
        if captured is None:
            print("Exiting pipeline.")
            return {"halted": True, "halt_reason": halt_reason}
        updates[section] = captured
        print(f"\n>> {section} updated — will re-validate.\n")
    return updates


def global_remediation_node(state: PipelineState) -> dict:
    """Fix whatever drove the global-grid (TIER 3) gate failure.

    The TIER-3 errors are tagged with grid-internal names (global_grid,
    antenna_placement, ...), not agent names, so we let the user pick which
    section to adjust — the grid is driven by dataset_config (resolution / PML),
    the antenna (source height / offset), the waveform (frequency) and layers.
    """
    errors = state.get("global_validation_errors") or []
    _banner("Fix Required — Global Validation (TIER 3)")
    print("The global grid validation failed with:")
    for e in errors:
        print(f"  - {e}")
    print(
        "\nThese are usually resolved by adjusting one of:\n"
        "  dataset_config  (cells_per_wavelength, pml_cells, buffer_cells)\n"
        "  antenna         (source_height_m, tx_rx_offset_m)\n"
        "  waveform        (center frequency)\n"
        "  layers          (thicknesses)\n"
    )
    allowed = ["dataset_config", "antenna", "waveform", "layers", "advanced_params"]
    while True:
        choice = input(f"Which section to edit ({'/'.join(allowed)}) or 'exit': ").strip().lower()
        if choice in ("", "quit", "exit"):
            print("Exiting pipeline.")
            return {"halted": True, "halt_reason": "user exited during global remediation"}
        if choice in allowed:
            break
        print("  Please choose one of: " + ", ".join(allowed))

    agent, display = SECTION_AGENTS[choice]
    captured, halt_reason = _run_agent_collect(
        agent, choice, display, _remediation_message(choice, errors, state.get(choice))
    )
    if captured is None:
        print("Exiting pipeline.")
        return {"halted": True, "halt_reason": halt_reason}

    # Editing layers (or dataset_config, which carries num_samples) changes the
    # per-sample draws, so the grid must be re-derived from FRESH samples — route
    # the re-derive back through layer_sampling. Other edits (antenna / waveform /
    # advanced) leave the draws valid, so re-derive directly.
    resample = choice in RESAMPLE_SECTIONS
    tail = "re-sample, re-derive the grid and re-validate" if resample else \
        "re-derive the grid and re-validate"
    print(f"\n>> {choice} updated — will {tail}.\n")
    return {
        choice: captured,
        "global_validation_passed": None,
        "global_validation_errors": None,
        "resample_after_global": resample,
    }


# ---------------------------------------------------------------------------
# Deterministic derive / validate nodes
#   All read sections from PipelineState (no httpx). Disk artifacts written by
#   the derive stages are still read back via output_dir by later stages.
# ---------------------------------------------------------------------------

def layer_sampling_node(state: PipelineState) -> dict:
    """Draw num_samples concrete layer sets (+ optional buried target) per sample."""
    _banner("Layer + Target Sampling")

    dataset_cfg = DatasetConfig.model_validate(state["dataset_config"])
    layers = ExtractedLayers.model_validate(state["layers"])

    target_range = None
    if state.get("target_ranges") is not None:
        target_range = ExtractedTargetRanges.model_validate(state["target_ranges"]).cylinder

    print(
        f"Sampling {dataset_cfg.num_samples} parameter set(s) over "
        f"{len(layers.layers)} layer range(s)"
        f"{' + a buried cylinder target' if target_range is not None else ''}..."
    )
    samples, path, warnings = sample_and_write(
        extracted=layers,
        num_samples=dataset_cfg.num_samples,
        output_dir=dataset_cfg.output_dir,
        seed=42,
        target_range=target_range,
    )
    print(f"  Wrote {len(samples)} sampled parameter set(s) to:\n    {path}")
    if warnings:
        print("\n  Warnings:")
        for w in warnings:
            print(f"    - {w}")
    print()
    return {}


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

    samples = read_samples(dataset_cfg.output_dir)
    _derived, aggregate, path = derive_and_write(
        samples, dataset_cfg, waveform, dataset_cfg.output_dir
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
    advanced = (
        ExtractedAdvancedParams.model_validate(state["advanced_params"])
        if state.get("advanced_params") is not None
        else None
    )

    aggregate = read_aggregate(dataset_cfg.output_dir)
    grid, path = derive_and_write_global(
        dataset_cfg, waveform, antenna, layers, advanced,
        aggregate.eps_r_max, aggregate.eps_r_min,
        dataset_cfg.output_dir,
        smallest_feature_global_m=aggregate.smallest_feature_global_m,
        largest_extent_global_m=aggregate.largest_extent_global_m,
        deepest_target_bottom_global_m=aggregate.deepest_target_bottom_global_m,
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
    advanced = (
        ExtractedAdvancedParams.model_validate(state["advanced_params"])
        if state.get("advanced_params") is not None
        else None
    )
    grid = read_global(dataset_cfg.output_dir)

    report = validate_global(grid, dataset_cfg, waveform, antenna, layers, advanced)

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
    """Validate each sample's buried target against the FIXED global grid."""
    target_range = None
    if state.get("target_ranges") is not None:
        target_range = ExtractedTargetRanges.model_validate(state["target_ranges"]).cylinder
    if target_range is None:
        return {}  # no buried target -> nothing to place

    _banner("Per-Sample Target Placement")

    dataset_cfg = DatasetConfig.model_validate(state["dataset_config"])
    grid = read_global(dataset_cfg.output_dir)

    result = run_placement(dataset_cfg.output_dir, dataset_cfg, grid, target_range, seed=1234)

    print(
        f"Placed targets: {result.n_unchanged} kept as-is, {result.n_redrawn} re-drawn, "
        f"{len(result.dropped)} dropped."
    )
    if result.dropped:
        print("  Dropped samples (dataset N reduced):")
        for d in result.dropped:
            print(f"    - sample {d['sample_id']}: {d['reason']}")
    print()
    return {}


def dataset_generation_node(state: PipelineState) -> dict:
    """Dataset generation / emission (STAGE 8).

    Emits one gprMax .in file per surviving sample onto the single global grid,
    reading the staged manifests (sampled_layers / global_derive) plus the
    collected waveform / antenna / advanced sections carried in PipelineState.
    """
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
    """layer_sampling routes to the waveform agent on the first pass, but jumps
    straight to the derive chain when it was re-entered to re-sample for global
    remediation (waveform/antenna are already collected and valid)."""
    return "peplinski_derive" if state.get("resample_after_global") else "waveform"


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

    # Agent nodes
    g.add_node("dataset_config",
               _make_agent_node(
                   dataset_config_agent, "dataset_config", "Dataset Configuration",
                   "I need to configure the dataset/run parameters for a gprMax "
                   "simulation batch. Please begin the dataset configuration process."))
    g.add_node("layers",
               _make_agent_node(
                   layer_agent, "layers", "Layer Extraction",
                   "I need to set up the soil layers for a gprMax simulation. "
                   "Please begin the layer parameter extraction process."))
    g.add_node("target_ranges",
               _make_agent_node(
                   target_agent, "target_ranges", "Buried-Target Range Extraction",
                   "I need to configure the buried-target geometry ranges for a "
                   "gprMax simulation batch. Please begin the target range extraction process."))
    g.add_node("waveform",
               _make_agent_node(
                   waveform_agent, "waveform", "Waveform Extraction",
                   "I need to configure the waveform for a gprMax simulation. "
                   "Please begin the waveform parameter extraction process."))
    g.add_node("antenna",
               _make_agent_node(
                   antenna_agent, "antenna", "Antenna Extraction",
                   "I need to configure the antenna for a gprMax simulation. "
                   "Please begin the antenna parameter extraction process."))
    g.add_node("advanced_params",
               _make_agent_node(
                   advanced_agent, "advanced_params", "Advanced Parameters Extraction",
                   "I need to configure the advanced/optional parameters for a gprMax "
                   "simulation. Please begin the advanced parameters extraction process."))

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

    # Linear agent steps route through a halt-aware router so a user "exit"
    # short-circuits the whole graph to END.
    g.add_conditional_edges("dataset_config", _linear_route("layers"), ["layers", END])
    g.add_conditional_edges("layers", _linear_route("target_ranges"), ["target_ranges", END])
    g.add_conditional_edges("target_ranges", _linear_route("layer_sampling"), ["layer_sampling", END])

    # Sampling runs after the target-range mini-stage, before the rest. It also
    # serves as the re-sample step for global remediation (layers/dataset_config
    # edits), in which case it jumps straight to the derive chain.
    g.add_conditional_edges("layer_sampling", _after_sampling, ["waveform", "peplinski_derive"])

    g.add_conditional_edges("waveform", _linear_route("antenna"), ["antenna", END])
    g.add_conditional_edges("antenna", _linear_route("sample_validation"), ["sample_validation", END])

    # Gate 1: waveform/antenna validation. On PASS -> advanced_params; on FAIL ->
    # remediation, which re-engages the offending agent and loops back to
    # re-validate. Only a user exit ends the run.
    g.add_conditional_edges(
        "sample_validation", _sample_gate,
        ["advanced_params", "sample_remediation", END],
    )
    g.add_edge("sample_remediation", "sample_validation")

    # Last agent stage, then the AGENT-FREE derive chain. Keeping advanced_params
    # before the derives lets global remediation re-run peplinski/global derive
    # without re-triggering any agent.
    g.add_conditional_edges("advanced_params", _linear_route("peplinski_derive"), ["peplinski_derive", END])
    g.add_edge("peplinski_derive", "global_derive")
    g.add_edge("global_derive", "global_validation")

    # Gate 2: TIER-3 global grid validation. On PASS -> placement; on FAIL ->
    # remediation, which re-derives the grid and loops back to re-validate.
    g.add_conditional_edges(
        "global_validation", _global_gate,
        ["target_placement", "global_remediation", END],
    )
    # A layers/dataset_config fix re-samples first; other fixes re-derive directly.
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
    # The deep agents' own tools (post_parameters/get_parameters) still talk to
    # the FastAPI server, so it must be up. The PIPELINE state itself, however,
    # is carried in LangGraph — not fetched back from the API.
    start_parameter_server()
    print("Parameter state server started.\n")

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
