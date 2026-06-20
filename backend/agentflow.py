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
from extraction_agents.waveform_extraction import agent as waveform_agent
from extraction_agents.antenna_extraction import agent as antenna_agent
from extraction_agents.advanced_params_extraction import agent as advanced_agent

# Add dataset_sampling to sys.path so its bare imports work
sys.path.insert(0, str(Path(__file__).parent / "dataset_sampling"))

from dataset_sampling.layer_sampler import sample_and_write, read_samples
from dataset_sampling.sample_validation import validate_waveform_antenna
from dataset_sampling.peplinski_derive import derive_and_write, read_aggregate
from dataset_sampling.global_derive import derive_and_write_global

from schema import (
    DatasetConfig,
    ExtractedLayers,
    ExtractedWaveform,
    ExtractedAntenna,
    ExtractedAdvancedParams,
)

# NOTE: the downstream sampler/generator/simulation modules
# (dataset_sampling.dataset_generation_agent, resolvers.merge_extractions,
# dataset_generator.generate_dataset, simulate.run_batch_simulation) still
# target the OLD extract schema and would fail to import. They are imported
# lazily inside the (currently disabled) generation/simulation stages so the
# extraction loop above keeps working after the schema migration.

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




def _fetch_all_sections():
    """Fetch all 5 sections from the parameter server and parse into schemas."""
    resp = httpx.get(f"{BASE_URL}/state", timeout=10)
    resp.raise_for_status()
    state = resp.json()

    section_map = {
        "dataset_config": DatasetConfig,
        "layers": ExtractedLayers,
        "waveform": ExtractedWaveform,
        "antenna": ExtractedAntenna,
        "advanced_params": ExtractedAdvancedParams,
    }
    parsed = {}
    for section, cls in section_map.items():
        data = state.get(section)
        if data is None:
            raise RuntimeError(f"Section '{section}' has no data — extraction incomplete.")
        parsed[section] = cls.model_validate(data)

    return parsed


def _run_layer_sampling():
    """Draw num_samples concrete layer sets from the extracted ranges.

    Runs right after the layer-extraction stage. num_samples comes from the
    dataset_config section; the layer ranges come from the layers section. Sand,
    clay, thickness and densities are drawn per sample; silt is derived and
    theta_v is passed through as its (min, max) band. The draws are written to a
    JSON manifest in the dataset output directory for the downstream stages.
    """
    print(f"\n{'='*60}")
    print("  Layer Sampling")
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

    print(
        f"Sampling {dataset_cfg.num_samples} parameter set(s) over "
        f"{len(layers.layers)} layer range(s)..."
    )
    samples, path, warnings = sample_and_write(
        extracted=layers,
        num_samples=dataset_cfg.num_samples,
        output_dir=dataset_cfg.output_dir,
        seed=42,
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
        dataset_cfg, waveform, antenna, layers, advanced,
        aggregate.eps_r_max, aggregate.eps_r_min,
        dataset_cfg.output_dir,
    )

    print(
        f"Sized one global grid from eps_r [{grid.eps_r_min_global:.3f}, "
        f"{grid.eps_r_max_global:.3f}]:"
    )
    print(f"  dx           = {grid.dx_m*1e3:.3f} mm")
    print(f"  domain (x,y) = {grid.domain_x_m:.3f} x {grid.domain_y_m:.3f} m")
    print(f"  depth        = {grid.depth_z_m:.3f} m")
    print(f"  dt           = {grid.dt_s*1e12:.3f} ps")
    print(f"  time window  = {grid.time_window_s*1e9:.2f} ns")
    print(f"  Wrote global derive to:\n    {path}\n")


def _run_dataset_generation():
    """Orchestrate dataset generation after all extractions are complete.

    NOTE (schema migration): the extraction layer now uses the 5-section
    collect schema (dataset_config / layers / waveform / antenna /
    advanced_params). The downstream sampler/generator below
    (`resolvers.merge_extractions`, `dataset_generator.generate_dataset`,
    `dataset_generation_agent`) still expects the OLD combined
    antenna_waveform + model_config schema and old layer fields. Wiring the
    sampler/derive stages to the new schema is a separate task — until then the
    generation stage is disabled so the extraction loop remains usable.
    """
    print(f"\n{'='*60}")
    print("  Dataset Generation Setup")
    print(f"{'='*60}\n")

    

    
    print("\nFetching extracted parameters from server...")
    sections = _fetch_all_sections()

    layers_result = sections["layers"]
    antenna_wf_result = sections["antenna_waveform"]
    model_result = sections["model_config"]
    advanced_result = sections["advanced_params"]



    # Merge and validate
    gpr_template, resolved_ranges, missing = merge_extractions(
        layers_result, antenna_wf_result, model_result, advanced_result,
    )

    if missing:
        print("\nCannot generate dataset — missing fields:")
        for m in missing:
            print(f"  {m}")
        return

    print(f"\nGenerating { model_result.num_samples} samples as '{dataset_name}'...")
    result = generate_dataset(
        resolved_layer_ranges=resolved_ranges,
        gpr_schema_template=gpr_template,
        num_samples= model_result.num_samples,
        dataset_name=dataset_name,
        seed=42,
    )

    print(f"\n{'='*60}")
    print(f"  Dataset Generation: {result.status.upper()}")
    print(f"{'='*60}")
    print(f"  Generated: {result.num_generated}/{result.num_requested}")
    print(f"  Failed:    {result.num_failed}")
    print(f"  Output:    {result.output_dir}")
    print(f"  Manifest:  {result.manifest_csv_path}")
    if result.errors:
        print(f"\n  Errors:")
        for err in result.errors:
            print(f"    - {err}")
    print()



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

        # Sampling runs immediately after the layer section: draw num_samples
        # concrete parameter sets over the layer ranges before the remaining
        # extraction stages (waveform/antenna/advanced) and the downstream
        # derive/emit process run.
        if section == "layers":
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

    # ── Dataset Generation Stage ──
    # Disabled pending downstream sampler/generator migration to the new
    # 5-section extract schema. See _run_dataset_generation().
    _run_dataset_generation()

if __name__ == "__main__":
    run_pipeline()
