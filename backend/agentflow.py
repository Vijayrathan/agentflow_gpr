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
from extraction_agents.layer_extraction import agent as layer_agent
from extraction_agents.antenna_extraction import agent as antenna_agent
from extraction_agents.model_specifics_extraction import agent as model_agent
from extraction_agents.advanced_params_extraction import agent as advanced_agent
from dataset_sampling.dataset_generation_agent import agent as dataset_agent

# Add dataset_sampling to sys.path so its bare imports work
sys.path.insert(0, str(Path(__file__).parent / "dataset_sampling"))

from schema import (
    ExtractedLayers,
    ExtractedAntennaWaveform,
    ExtractedModelConfig,
    ExtractedAdvancedParams,
)
from resolvers import merge_extractions
from dataset_generator import generate_dataset
from simulate import run_batch_simulation

# Ordered pipeline: (agent, section_name, display_name, init_message)
PIPELINE = [
    (
        layer_agent,
        "layers",
        "Layer Extraction",
        "I need to set up the soil layers for a gprMax simulation. "
        "Please begin the layer parameter extraction process.",
    ),
    (
        antenna_agent,
        "antenna_waveform",
        "Antenna & Waveform Extraction",
        "I need to configure the antenna and waveform for a gprMax simulation. "
        "Please begin the antenna/waveform parameter extraction process.",
    ),
    (
        model_agent,
        "model_config",
        "Model & Domain Extraction",
        "I need to configure the simulation model and domain parameters for a "
        "gprMax simulation. Please begin the model/domain parameter extraction process.",
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


def _extract_files_dir(result: dict) -> str | None:
    """Extract files_dir from post_dataset_to_db tool result."""
    for msg in result.get("messages", []):
        if type(msg).__name__ == "ToolMessage" and msg.name == "post_dataset_to_db":
            try:
                data = json.loads(msg.content)
                if data.get("status") == "ok":
                    return data.get("files_dir")
            except (json.JSONDecodeError, TypeError):
                pass
    return None




def _ask_dataset_config() -> dict:
    """Prompt the user for dataset-level configuration."""
    # dataset_name
    while True:
        dataset_name = input("Dataset name (used for the output directory): ").strip()
        if dataset_name:
            break
        print("Dataset name cannot be empty.")

    return {"dataset_name": dataset_name}


def _fetch_all_sections():
    """Fetch all 4 sections from the parameter server and parse into schemas."""
    resp = httpx.get(f"{BASE_URL}/state", timeout=10)
    resp.raise_for_status()
    state = resp.json()

    section_map = {
        "layers": ExtractedLayers,
        "antenna_waveform": ExtractedAntennaWaveform,
        "model_config": ExtractedModelConfig,
        "advanced_params": ExtractedAdvancedParams,
    }
    parsed = {}
    for section, cls in section_map.items():
        data = state.get(section)
        if data is None:
            raise RuntimeError(f"Section '{section}' has no data — extraction incomplete.")
        parsed[section] = cls.model_validate(data)

    return parsed


def _run_dataset_generation():
    """Orchestrate dataset generation after all extractions are complete."""
    print(f"\n{'='*60}")
    print("  Dataset Generation Setup")
    print(f"{'='*60}\n")

    config = _ask_dataset_config()
    dataset_name = config["dataset_name"]

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

        if _posted(result):
            print(f"\n>> {display_name} complete — {section} saved.\n")
            time.sleep(30)
            continue
    
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
                print(f"\n>> {display_name} complete — {section} saved.\n")
                time.sleep(30)
                break

    print(f"\n{'='*60}")
    print("  All extraction agents complete!")
    print(f"{'='*60}\n")

    # ── Dataset Generation Agent ──
    print(f"\n{'='*60}")
    print("  Starting: Dataset Generation")
    print(f"{'='*60}\n")

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = dataset_agent.invoke(
        {"messages": [HumanMessage(
            content="All parameter extractions are complete. Begin dataset generation."
        )]},
        config=config,
    )
    seen = _print_response(result)

    while not _posted(result):
        user_input = input("You: ").strip()
        if not user_input or user_input.lower() in ("quit", "exit"):
            print("Exiting.")
            break
        result = dataset_agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )
        seen = _print_response(result, seen)

    # ── Simulation Stage ──
    files_dir = _extract_files_dir(result)
    if files_dir:
        _run_simulation_stage(files_dir)
    else:
        print("\n[WARNING] Dataset not posted or no files_dir — skipping simulation.")

if __name__ == "__main__":
    run_pipeline()
