import time
import uuid

from langchain_core.messages import HumanMessage

from parameters_global_state import start_parameter_server
from extraction_agents.layer_extraction import agent as layer_agent
from extraction_agents.antenna_extraction import agent as antenna_agent
from extraction_agents.model_specifics_extraction import agent as model_agent
from extraction_agents.advanced_params_extraction import agent as advanced_agent

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


def _print_response(result: dict) -> None:
    for msg in result.get("messages", []):
        kind = type(msg).__name__
        if kind == "AIMessage" and msg.content:
            print(f"\n[Agent]: {msg.content}\n")
        elif kind == "ToolMessage":
            print(f"  [tool:{msg.name}] returned ({len(msg.content)} chars)")


def _posted(result: dict) -> bool:
    """Return True if any message in the result is a post_parameters tool call."""
    return any(
        type(msg).__name__ == "ToolMessage" and msg.name == "post_parameters"
        for msg in result.get("messages", [])
    )


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
        _print_response(result)

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
            _print_response(result)

            if _posted(result):
                print(f"\n>> {display_name} complete — {section} saved.\n")
                time.sleep(30)
                break

    print(f"\n{'='*60}")
    print("  All extraction agents complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_pipeline()
