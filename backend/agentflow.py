import json
import sys
import time
import uuid
from pathlib import Path

# Ensure the project root is on sys.path so `backend.*` and `db.*` imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from langchain_core.messages import HumanMessage

from parameters_global_state import start_parameter_server, BASE_URL
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

from schema import (
    DatasetConfig,
    ExtractedLayers,
    ExtractedTargetRanges,
    ExtractedWaveform,
    ExtractedAntenna,
    ExtractedAdvancedParams,
)

# NOTE: emission of the N gprMax .in files (STAGE 9) is not yet wired — the old
# combined-schema generator was removed in the cleanup. _run_dataset_generation()
# is a disabled stub until the new writer (built against the staged manifests)
# lands. Everything up to and including global validation + target placement runs.

# Ordered pipeline: (agent, section_name, display_name, init_message)
PIPELINE = [
    (
        dataset_config_agent,
        "dataset_config",
        "Dataset Configuration",
        "I need to configure the dataset/run parameters for a gprMax simulation "
        "batch. Please begin the dataset configuration process.",
    ),
    (
        layer_agent,
        "layers",
        "Layer Extraction",
        "I need to set up the soil layers for a gprMax simulation. "
        "Please begin the layer parameter extraction process.",
    ),
    (
        target_agent,
        "target_ranges",
        "Buried-Target Range Extraction",
        "I need to configure the buried-target geometry ranges for a gprMax "
        "simulation batch. Please begin the target range extraction process.",
    ),
    (
        waveform_agent,
        "waveform",
        "Waveform Extraction",
        "I need to configure the waveform for a gprMax simulation. "
        "Please begin the waveform parameter extraction process.",
    ),
    (
        antenna_agent,
        "antenna",
        "Antenna Extraction",
        "I need to configure the antenna for a gprMax simulation. "
        "Please begin the antenna parameter extraction process.",
    ),
    (
        advanced_agent,
        "advanced_params",
        "Advanced Parameters Extraction",
        "I need to configure the advanced/optional parameters for a gprMax "
        "simulation. Please begin the advanced parameters extraction process.",
    ),
]


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


def _posted(result: dict) -> bool:
    """Return True if any message is a post_parameters or post_dataset_to_db tool call."""
    return any(
        type(msg).__name__ == "ToolMessage"
        and msg.name in ("post_parameters", "post_dataset_to_db")
        for msg in result.get("messages", [])
    )




def _run_layer_sampling():
    """Draw num_samples concrete layer sets (and optional buried target) per sample.

    Runs right after the target-range mini-stage. num_samples comes from the
    dataset_config section; the layer ranges come from the layers section. Sand,
    clay, thickness and densities are drawn per sample; silt is derived and
    theta_v is passed through as its (min, max) band. If a `target_ranges`
    cylinder was collected, a buried target is drawn per sample too (grid-
    independent; placement validated downstream). The draws are written to a JSON
    manifest in the dataset output directory for the downstream stages.
    """
    print(f"\n{'='*60}")
    print("  Layer + Target Sampling")
    print(f"{'='*60}\n")

    resp = httpx.get(f"{BASE_URL}/state", timeout=10)
    resp.raise_for_status()
    state = resp.json()

    if state.get("dataset_config") is None:
        raise RuntimeError("dataset_config not found — run the dataset configuration stage first.")
    if state.get("layers") is None:
        raise RuntimeError("layers not found — run the layer extraction stage first.")

    dataset_cfg = DatasetConfig.model_validate(state["dataset_config"])
    layers = ExtractedLayers.model_validate(state["layers"])

    # LEGACY path: mechanically updated for the unified multi-object
    # target_ranges schema (cylinders + boxes); predates that refactor.
    target_ranges = None
    if state.get("target_ranges") is not None:
        tr = ExtractedTargetRanges.model_validate(state["target_ranges"])
        target_ranges = tr if tr.has_targets else None

    print(
        f"Sampling {dataset_cfg.num_samples} parameter set(s) over "
        f"{len(layers.layers)} layer range(s)"
        f"{' + buried object(s)' if target_ranges is not None else ''}..."
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


def _run_sample_validation() -> bool:
    """Validate the waveform and antenna before proceeding past the antenna stage.

    Runs right after the antenna stage. The per-sample layer checks already ran on
    every draw inside the sampler, so this gate applies only the validations that
    need the waveform/antenna: the Peplinski frequency gate and antenna config.
    Returns True when there are no errors (safe to proceed), False otherwise.
    """
    print(f"\n{'='*60}")
    print("  Sample Validation")
    print(f"{'='*60}\n")

    resp = httpx.get(f"{BASE_URL}/state", timeout=10)
    resp.raise_for_status()
    state = resp.json()

    for section in ("dataset_config", "waveform", "antenna"):
        if state.get(section) is None:
            raise RuntimeError(f"{section} not found — run the {section} stage first.")

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
        print("\n>> Sample validation FAILED.\n")
        return False

    print("\n>> Sample validation passed.\n")
    return True


def _run_peplinski_derive():
    """Derive in-band eps_r per sample using gprMax's own Peplinski routine.

    Runs after the post-antenna validation gate, once the waveform frequency is
    known. For each drawn sample/layer it builds gprMax's PeplinskiSoil over the
    moisture band, evaluates the real in-band permittivity at the operating
    frequency, and aggregates the driest/wettest eps_r across all samples for the
    downstream global grid derive. sigma is NOT derived — gprMax writes the actual
    eps/sigma materials at model-build time.
    """
    print(f"\n{'='*60}")
    print("  Peplinski Derive (eps_r)")
    print(f"{'='*60}\n")

    resp = httpx.get(f"{BASE_URL}/state", timeout=10)
    resp.raise_for_status()
    state = resp.json()

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


def _run_global_derive():
    """Derive the ONE global grid / domain / depth / time window for all samples.

    Runs after all collect stages and the per-sample eps derive. Uses the
    aggregated eps_r corners (from derived_layers.json) together with the
    waveform, antenna, layer thicknesses and advanced/target geometry to size a
    single Yee grid shared by every generated input file.
    """
    print(f"\n{'='*60}")
    print("  Global Derive (grid / domain / depth / time window)")
    print(f"{'='*60}\n")

    resp = httpx.get(f"{BASE_URL}/state", timeout=10)
    resp.raise_for_status()
    state = resp.json()

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


def _run_global_validation() -> bool:
    """Validate the single global grid (TIER 3) right after the global derive.

    Runs the cascade-gated TIER 3 battery (grid fundamentals / placement /
    feasibility) plus the static Tx/Rx + source-height checks. Returns True when
    there are no errors (safe to proceed), False otherwise.
    """
    print(f"\n{'='*60}")
    print("  Global Validation (TIER 3)")
    print(f"{'='*60}\n")

    resp = httpx.get(f"{BASE_URL}/state", timeout=10)
    resp.raise_for_status()
    state = resp.json()

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
        print("\n>> Global validation FAILED.\n")
        return False

    print("\n>> Global validation passed.\n")
    return True


def _run_target_placement():
    """Validate each sample's buried target against the FIXED global grid.

    Re-draws a target's geometry (radius shrink + reposition) when it lands in the
    PML/clearance zone or is not fully buried; drops the sample (reducing N) when
    it cannot be placed. No-op when no buried target was collected.
    """
    resp = httpx.get(f"{BASE_URL}/state", timeout=10)
    resp.raise_for_status()
    state = resp.json()

    target_ranges = None
    if state.get("target_ranges") is not None:
        tr = ExtractedTargetRanges.model_validate(state["target_ranges"])
        target_ranges = tr if tr.has_targets else None
    if target_ranges is None:
        return  # no buried objects -> nothing to place

    print(f"\n{'='*60}")
    print("  Per-Sample Target Placement")
    print(f"{'='*60}\n")

    dataset_cfg = DatasetConfig.model_validate(state["dataset_config"])
    grid = read_global(dataset_cfg.output_dir)

    result = run_placement(dataset_cfg.output_dir, dataset_cfg, grid, target_ranges, seed=1234)

    print(
        f"Placed targets: {result.n_unchanged} kept as-is, {result.n_redrawn} re-drawn, "
        f"{len(result.dropped)} dropped."
    )
    if result.dropped:
        print("  Dropped samples (dataset N reduced):")
        for d in result.dropped:
            print(f"    - sample {d['sample_id']}: {d['reason']}")
    print()


def _run_dataset_generation():
    """Dataset generation / emission (DISABLED — pending the new emitter).

    The staged pipeline now persists per-sample manifests (sampled_layers.json,
    derived_layers.json, global_derive.json). Emitting the N gprMax `.in` files on
    the single global grid (STAGE 9) against those manifests is a separate task;
    the old combined-schema generator was removed in the cleanup. This stage is a
    no-op so the extraction + derive + validation loop stays runnable.
    """
    print(f"\n{'='*60}")
    print("  Dataset Generation / Emission")
    print(f"{'='*60}\n")
    print("Emission is disabled pending the new .in writer built against the "
          "staged manifests (sampled_layers / derived_layers / global_derive).\n")



def _run_simulation_stage(files_dir: str):
    """Run gprMax batch simulation on generated .in files."""
    print(f"\n{'='*60}")
    print("  Starting: Simulation")
    print(f"{'='*60}\n")

    print(f"  Input directory:  {files_dir}")
    output_dir = str(Path(files_dir).parent / "out_files")
    print(f"  Output directory: {output_dir}\n")

    try:
        from simulate import run_batch_simulation  # lazy: targets old schema
        sim_result = run_batch_simulation(
            input_dir=files_dir,
            skip_existing=True,
            verbose=True,
        )
    except Exception as e:
        print(f"\n[ERROR] Simulation failed: {e}")
        return

    print(f"\n{'='*60}")
    print("  Simulation Complete")
    print(f"{'='*60}")
    print(f"  Succeeded: {sim_result['succeeded']}")
    print(f"  Failed:    {sim_result['failed']}")
    print(f"  Skipped:   {sim_result['skipped']}")
    print(f"  Total:     {sim_result['total']}")
    print(f"  Output:    {sim_result['output_dir']}")
    print()


def run_pipeline():
    start_parameter_server()
    print("Parameter state server started.\n")

    for agent, section, display_name, init_message in PIPELINE:
        print(f"\n{'='*60}")
        print(f"  Starting: {display_name}")
        print(f"{'='*60}\n")

        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        result = agent.invoke(
            {"messages": [HumanMessage(content=init_message)]},
            config=config,
        )
        seen = _print_response(result)

        if not _posted(result):
            while True:
                user_input = input("You: ").strip()
                if not user_input or user_input.lower() in ("quit", "exit"):
                    print("Exiting pipeline.")
                    return

                result = agent.invoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config,
                )
                seen = _print_response(result, seen)

                if _posted(result):
                    break

        print(f"\n>> {display_name} complete — {section} saved.\n")

        # Sampling runs after the target-range mini-stage (which follows layers):
        # by now both the layer ranges and the optional buried-target ranges are
        # collected, so draw num_samples concrete soil+target sets before the
        # remaining extraction stages and the downstream derive/emit process.
        if section == "target_ranges":
            _run_layer_sampling()
        elif section == "antenna":
            # Waveform + antenna are now collected and the N samples are drawn:
            # validate across all samples before proceeding to the rest.
            if not _run_sample_validation():
                print("Halting pipeline: sample validation failed. "
                      "Fix the inputs and re-run.")
                return
            # Validation passed: derive in-band eps_r per sample for the grid.
            _run_peplinski_derive()

        time.sleep(30)

    print(f"\n{'='*60}")
    print("  All extraction agents complete!")
    print(f"{'='*60}\n")

    # ── Global Derive Stage ──
    # All params are collected and per-sample eps_r is derived: size the single
    # global grid / domain / depth / time window shared by every sample.
    _run_global_derive()

    # ── Global Validation Stage (TIER 3) ──
    if not _run_global_validation():
        print("Halting pipeline: global validation failed. "
              "Fix the inputs and re-run.")
        return

    # ── Per-Sample Target Placement Stage ──
    # Grid-dependent, per-sample: validate each drawn target against the fixed
    # domain; re-draw into the valid envelope or drop (reducing N).
    _run_target_placement()

    # ── Dataset Generation Stage ──
    # Disabled pending downstream sampler/generator migration to the new
    # 5-section extract schema. See _run_dataset_generation().
    _run_dataset_generation()

if __name__ == "__main__":
    run_pipeline()
