from flask import Flask, request, jsonify
from flask_cors import CORS
import asyncio
import json
import logging
from generator_agent import (
    simulate_workflow,
    qa_workflow,
    get_workspace_directory,
    run_gprmax_simulation_tool,
)
from schema import AggregatedExtraction
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

conversations = {}


def _get_event_loop():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def _build_conversation_context(session_id: str, message: str) -> str:
    """Build a single string containing the full conversation history."""
    msgs = conversations.get(session_id, {}).get("messages", [])
    if len(msgs) <= 1:
        return message

    parts = []
    for msg in msgs[:-1]:
        parts.append(f"{msg['role'].capitalize()}: {msg['content']}")
    parts.append(f"User: {message}")
    return "\n\n".join(parts)


@app.route("/api/chat", methods=["POST"])
def chat():
    """Handle chat messages, routing to simulate or Q&A workflow based on mode."""
    try:
        data = request.json
        message = data.get("message", "")
        session_id = data.get("session_id", "default")
        mode = data.get("mode")

        if not message:
            return jsonify({"error": "Message is required"}), 400

        # Initialise session
        if session_id not in conversations:
            conversations[session_id] = {"messages": [], "mode": None, "params": None}

        # Use provided mode, or fall back to session's stored mode
        if not mode:
            mode = conversations[session_id].get("mode")
        if not mode:
            return jsonify({"error": "mode is required ('simulate' or 'qa')"}), 400

        conversations[session_id]["mode"] = mode

        # Store user message
        conversations[session_id]["messages"].append(
            {"role": "user", "content": message}
        )

        loop = _get_event_loop()

        # ---- Route by mode ----
        if mode == "simulate":
            # Restore persisted parameter state
            stored_params = conversations[session_id].get("params")
            current_state = None
            if stored_params:
                try:
                    current_state = AggregatedExtraction(**stored_params)
                except Exception:
                    logger.warning("Failed to deserialise stored params; starting fresh")
                    current_state = None

            result = loop.run_until_complete(
                simulate_workflow(message, user_id=session_id, current_state=current_state)
            )

            # Persist updated parameter state
            if result.get("params"):
                conversations[session_id]["params"] = result["params"]
        elif mode == "qa":
            conversation_context = _build_conversation_context(session_id, message)
            result = loop.run_until_complete(qa_workflow(conversation_context))
        else:
            return jsonify({"error": f"Unknown mode '{mode}'. Use 'simulate' or 'qa'."}), 400

        response_content = result.get("message", "")

        # Store assistant response
        conversations[session_id]["messages"].append(
            {"role": "assistant", "content": response_content}
        )

        response_data = {
            "message": response_content,
            "status": result.get("status", "incomplete"),
        }

        if result.get("file_path"):
            response_data["generated_file_path"] = result["file_path"]
        if result.get("file_content"):
            response_data["file_content"] = result["file_content"]
        if result.get("missing_params"):
            response_data["missing_params"] = result["missing_params"]
        if result.get("validation_errors"):
            response_data["validation_errors"] = result["validation_errors"]
        if result.get("sources"):
            response_data["sources"] = result["sources"]
        if result.get("dataset_result"):
            dr = result["dataset_result"]
            response_data["num_generated"] = dr.get("num_generated")
            response_data["num_failed"] = dr.get("num_failed")
            response_data["num_requested"] = dr.get("num_requested")
            response_data["dataset_output_dir"] = dr.get("output_dir")
            response_data["manifest_csv_path"] = dr.get("manifest_csv_path")
            response_data["manifest_json_path"] = dr.get("manifest_json_path")
            if dr.get("errors"):
                response_data["dataset_errors"] = dr.get("errors")

        return jsonify(response_data)

    except Exception as e:
        import traceback

        traceback.print_exc()
        return (
            jsonify({"error": f"Server error: {str(e)}", "traceback": traceback.format_exc()}),
            500,
        )


@app.route("/api/reset", methods=["POST"])
def reset():
    """Reset conversation for a session"""
    try:
        data = request.json
        session_id = data.get("session_id", "default")

        if session_id in conversations:
            del conversations[session_id]

        return jsonify({"status": "success", "message": "Conversation reset"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def root():
    return jsonify(
        {
            "message": "gprMax Chatbot API",
            "status": "running",
            "frontend": "http://localhost:3000",
            "endpoints": {
                "health": "/api/health",
                "chat": "/api/chat (POST: message, session_id, mode='simulate'|'qa')",
                "params": "/api/params?session_id=...",
                "file": "/api/file",
                "simulate": "/api/simulate",
                "reset": "/api/reset",
            },
        }
    )


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


@app.route("/api/file", methods=["GET"])
def get_file():
    """Get the generated .in file content"""
    try:
        workspace_dir = get_workspace_directory()
        generated_files_dir = workspace_dir / "generated_files"

        file_path = None
        if generated_files_dir.exists():
            for file in generated_files_dir.glob("generated*.in"):
                file_path = str(file)
                break

        if not file_path or not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        return jsonify({"content": content, "filename": os.path.basename(file_path)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/params", methods=["GET"])
def get_params():
    """Return the current session's collected parameter state."""
    try:
        session_id = request.args.get("session_id", "default")
        session = conversations.get(session_id)
        if not session or not session.get("params"):
            return jsonify({"params": None, "message": "No parameters collected yet."})

        return jsonify({"params": session["params"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/simulate", methods=["POST"])
def simulate():
    """Run gprMax simulation with the generated input file"""
    try:
        data = request.json
        file_path = data.get("file_path")

        if not file_path:
            return jsonify({"error": "file_path is required"}), 400
        if not os.path.exists(file_path):
            return jsonify({"error": f"File not found: {file_path}"}), 404

        logger.info(f"Running simulation for file: {file_path}")
        simulation_result = run_gprmax_simulation_tool(file_path)

        is_failure = "failed" in simulation_result.lower() or "exit code" in simulation_result.lower()
        if is_failure:
            logger.error(f"Simulation failed: {simulation_result[:200]}...")
            return (
                jsonify(
                    {
                        "status": "error",
                        "error": simulation_result,
                        "message": "Simulation failed",
                        "result": simulation_result,
                    }
                ),
                500,
            )

        logger.info(f"Simulation completed successfully. Logs length: {len(simulation_result)} chars")
        return jsonify(
            {
                "status": "success",
                "result": simulation_result,
                "message": "Simulation completed successfully",
            }
        )

    except Exception as e:
        import traceback

        logger.error(f"Error in simulate endpoint: {str(e)}", exc_info=True)
        return (
            jsonify({"error": f"Server error: {str(e)}", "traceback": traceback.format_exc()}),
            500,
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    app.run(host="0.0.0.0", port=port, debug=True)
