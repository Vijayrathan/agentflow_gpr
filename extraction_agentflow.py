from langchain_core.messages import HumanMessage
import uuid



def _print_response(result: dict) -> None:
    """Print all new assistant messages and log tool calls for debugging."""
    for msg in result.get("messages", []):
        kind = type(msg).__name__
        if kind == "AIMessage" and msg.content:
            print(f"\n[Master Agent]: {msg.content}\n")
        elif kind == "ToolMessage":
            # Tool results — show which tool returned
            print(f"  [tool:{msg.name}] returned ({len(msg.content)} chars)")


def agent_react_loop(agent,init_message):
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # Kick off with the initial request
    print("Starting model & domain extraction agent...\n")
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(init_message)
                )
            ]
        },
        config=config,
    )

    _print_response(result)

    # Interactive loop: feed user replies back until the agent is done
    while True:
        user_input = input("You: ").strip()
        if not user_input or user_input.lower() in ("quit", "exit"):
            print("Exiting.")
            break

        result = agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )

        _print_response(result)


