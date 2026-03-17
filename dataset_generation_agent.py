import json
import os
import sys
import uuid
from pathlib import Path
from typing import Annotated, Optional

import dotenv
import httpx
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI

from deepagents import create_deep_agent
from parameters_global_state import get_parameters, patch_parameters, BASE_URL
from prompt_library import DATASET_GENERATION_PROMPT

# Add dataset_sampling to sys.path so its bare imports work
sys.path.insert(0, str(Path(__file__).parent / "dataset_sampling"))

from schema import (
    ExtractedLayers,
    ExtractedLayerParams,
    ExtractedAntennaWaveform,
    ExtractedModelConfig,
    ExtractedAdvancedParams,
    SurfaceRoughnessConfigSchema,
    RxArrayConfigSchema,
)
from resolvers import merge_extractions
from dataset_generator import generate_dataset

dotenv.load_dotenv()


# ── Helper: fetch & parse all sections from parameter server ──────────

def _fetch_and_parse():
    """Fetch all 4 sections from the parameter server and parse into schema objects.

    Returns (sections_dict, errors_list) where sections_dict maps section name
    to its parsed Pydantic object, and errors_list contains any parse failures.
    """
    # TODO: restore actual server fetch when done testing
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
    errors = []
    for section, cls in section_map.items():
        data = state.get(section)
        if data is None:
            errors.append(f"Section '{section}' has no data — extraction not yet complete.")
            continue
        try:
            parsed[section] = cls.model_validate(data)
        except Exception as e:
            errors.append(f"Section '{section}' failed validation: {e}")
    
    return parsed, errors
    return _dummy_sections(), []


# def _dummy_sections():
#     """Hardcoded extraction results for testing dataset generation directly."""
#     _layer = ExtractedLayerParams(
#         name="Layer 1",
#         thickness_m_min=0.1, thickness_m_max=0.3,
#         sand_pct_min=5, sand_pct_max=65,
#         silt_pct_min=20, silt_pct_max=65,
#         clay_pct_min=10, clay_pct_max=50,
#         theta_v_min=0.14, theta_v_max=0.18,
#         bulk_density_gcm3_min=1.3, bulk_density_gcm3_max=1.7,
#         particle_density_gcm3_min=2.6, particle_density_gcm3_max=2.7,
#         organic_fraction=0.03,
#         salinity_classes=["fresh"],
#         porewater_sigma_Sm=0.01,
#     )
#     _layer2 = ExtractedLayerParams(
#         name="Layer 2",
#         thickness_m_min=0.1, thickness_m_max=0.3,
#         sand_pct_min=5, sand_pct_max=65,
#         silt_pct_min=20, silt_pct_max=65,
#         clay_pct_min=10, clay_pct_max=50,
#         theta_v_min=0.14, theta_v_max=0.18,
#         bulk_density_gcm3_min=1.3, bulk_density_gcm3_max=1.7,
#         particle_density_gcm3_min=2.6, particle_density_gcm3_max=2.7,
#         organic_fraction=0.03,
#         salinity_classes=["fresh"],
#         porewater_sigma_Sm=0.01,
#     )

#     return {
#         "layers": ExtractedLayers(num_layers=2, layers=[_layer, _layer2]),
#         "antenna_waveform": ExtractedAntennaWaveform(
#             antenna_kind="voltage_source",
#             antenna_axis="z",
#             tx_rx_offset_m=0.05,
#             resistance=75.0,
#             source_start_time=0.0,
#             source_end_time=1e-9,
#             waveform_kind="gaussian",
#             waveform_amplitude=5.0,
#             waveform_center_freq_hz=1.2e9,
#             waveform_name="my_gauss_pulse",
#         ),
#         "model_config": ExtractedModelConfig(
#             model="crim",
#             title="test2",
#             domain_x=0.3,
#             domain_y=0.8,
#             top_air_extra_m=0.1,
#             cells_per_wavelength=15,
#             max_cell_m=0.01,
#             source_height_m=0.03,
#             rx_same_height=True,
#             temperature_c=20.0,
#             enforce_validity=True,
#             salinity_defaults_Sm=[0.01, 0.1, 1.0, 3.5],
#             num_samples=512,
#         ),
#         "advanced_params": ExtractedAdvancedParams(
#             surface_roughness=SurfaceRoughnessConfigSchema(
#                 fractal_dim=1.5,
#                 weight_x=1.0,
#                 weight_y=1.0,
#                 amplitude_m=0.01,
#                 add_water=False,
#                 water_depth_m=0.005,
#             ),
#             rx_array=RxArrayConfigSchema(
#                 x1=0.0, y1=0.0, z1=0.0,
#                 x2=1.0, y2=0.0, z2=0.0,
#                 dx=0.05, dy=0.05, dz=0.05,
#             ),
#         ),
#     }

# ── Tools ─────────────────────────────────────────────────────────────

@tool
def fetch_all_extractions() -> str:
    """Fetch and summarise all extracted parameter sections from the server.

    Returns a JSON summary of what is currently stored for each section
    (layers, antenna_waveform, model_config, advanced_params), or which
    sections are missing / invalid.
    """
    parsed, errors = _fetch_and_parse()

    summary = {}
    if "layers" in parsed:
        layers = parsed["layers"]
        summary["layers"] = {
            "num_layers": layers.num_layers,
            "layer_names": [l.name for l in layers.layers],
        }
    if "antenna_waveform" in parsed:
        aw = parsed["antenna_waveform"]
        summary["antenna_waveform"] = {
            "antenna_kind": aw.antenna_kind,
            "waveform_kind": aw.waveform_kind,
            "center_freq_hz": aw.waveform_center_freq_hz,
        }
    if "model_config" in parsed:
        mc = parsed["model_config"]
        summary["model_config"] = {
            "model": mc.model,
            "title": mc.title,
            "num_samples": mc.num_samples,
            "domain_x": mc.domain_x,
            "domain_y": mc.domain_y,
        }
    if "advanced_params" in parsed:
        summary["advanced_params"] = "populated"

    result = {"sections": summary}
    if errors:
        result["errors"] = errors

    return json.dumps(result, indent=2)


@tool
def resolve_and_validate() -> str:
    """Resolve all extracted parameters and validate readiness for dataset generation.

    Fetches sections from the server, parses them, and calls merge_extractions
    to check completeness. Returns JSON with status 'ready' or 'incomplete'
    with a list of missing fields.
    """
    parsed, parse_errors = _fetch_and_parse()

    if parse_errors:
        return json.dumps({
            "status": "incomplete",
            "missing": parse_errors,
        }, indent=2)

    required = ["layers", "antenna_waveform", "model_config", "advanced_params"]
    missing_sections = [s for s in required if s not in parsed]
    if missing_sections:
        return json.dumps({
            "status": "incomplete",
            "missing": [f"Section '{s}' not populated" for s in missing_sections],
        }, indent=2)

    _schema, _ranges, missing = merge_extractions(
        parsed["layers"],
        parsed["antenna_waveform"],
        parsed["model_config"],
        parsed["advanced_params"],
    )

    if missing:
        return json.dumps({
            "status": "incomplete",
            "missing": missing,
        }, indent=2)

    return json.dumps({
        "status": "ready",
        "num_layers": len(parsed["layers"].layers),
        "model": parsed["model_config"].model,
        "num_samples": parsed["model_config"].num_samples,
    }, indent=2)


@tool
def run_dataset_generation(
    dataset_name: Annotated[str, "Name for the output dataset directory"],
    seed: Annotated[Optional[int], "Random seed for reproducibility (optional)"] = None,
) -> str:
    """Generate the dataset from extracted parameters.

    Fetches all sections, resolves them, and generates the requested number
    of gprMax .in files. Returns a JSON summary with generation status,
    counts, and any errors.
    """
    # Fetch and parse
    parsed, parse_errors = _fetch_and_parse()
    if parse_errors:
        return json.dumps({"status": "error", "errors": parse_errors}, indent=2)

    required = ["layers", "antenna_waveform", "model_config", "advanced_params"]
    missing_sections = [s for s in required if s not in parsed]
    if missing_sections:
        return json.dumps({
            "status": "error",
            "errors": [f"Section '{s}' not populated" for s in missing_sections],
        }, indent=2)

    # Resolve
    gpr_template, resolved_ranges, missing = merge_extractions(
        parsed["layers"],
        parsed["antenna_waveform"],
        parsed["model_config"],
        parsed["advanced_params"],
    )

    if missing:
        return json.dumps({"status": "error", "errors": missing}, indent=2)

    num_samples = parsed["model_config"].num_samples

    # Generate
    result = generate_dataset(
        resolved_layer_ranges=resolved_ranges,
        gpr_schema_template=gpr_template,
        num_samples=num_samples,
        dataset_name=dataset_name,
        seed=seed,
    )

    # Determine effective status (partial with >=90% is acceptable)
    effective_status = result.status
    if result.status == "partial" and result.num_generated >= 0.9 * result.num_requested:
        effective_status = "complete"

    output = {
        "status": effective_status,
        "num_requested": result.num_requested,
        "num_generated": result.num_generated,
        "num_failed": result.num_failed,
    }
    if result.errors:
        output["errors"] = result.errors

    return json.dumps(output, indent=2)


# ── Agent ─────────────────────────────────────────────────────────────

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

agent = create_deep_agent(
    model=llm,
    subagents=[],
    system_prompt=DATASET_GENERATION_PROMPT,
    checkpointer=InMemorySaver(),
    tools=[
        fetch_all_extractions,
        resolve_and_validate,
        run_dataset_generation,
        get_parameters,
        patch_parameters,
    ],
)


# ── Standalone runner ─────────────────────────────────────────────────

def _print_response(result: dict, seen: int = 0) -> int:
    messages = result.get("messages", [])
    for msg in messages[seen:]:
        kind = type(msg).__name__
        if kind == "AIMessage" and msg.content:
            print(f"\n[Agent]: {msg.content}\n")
        elif kind == "ToolMessage":
            print(f"  [tool:{msg.name}] returned ({len(msg.content)} chars)")
    return len(messages)


if __name__ == "__main__":
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    result = agent.invoke(
        {"messages": [HumanMessage(
            content="All parameter extractions are complete. Begin dataset generation."
        )]},
        config=config,
    )
    seen = _print_response(result)

    while True:
        user_input = input("You: ").strip()
        if not user_input or user_input.lower() in ("quit", "exit"):
            break
        result = agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )
        seen = _print_response(result, seen)
