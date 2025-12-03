from flask import Flask, request, jsonify
from flask_cors import CORS
import asyncio
import json
import logging
from generator_agent import central_agent, runner_agent, get_workspace_directory, run_gprmax_simulation_tool
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Store conversation history per session
conversations = {}

def agent_result_to_dict(result):
    """Convert agent result to JSON-serializable dict"""
    try:
        # If result has model_dump method (Pydantic model), use it
        if hasattr(result, 'model_dump'):
            return result.model_dump()
        # If result is a dict, return as is
        elif isinstance(result, dict):
            return result
        # If result has output attribute, extract it
        elif hasattr(result, 'output'):
            return {
                'output': str(result.output) if result.output is not None else '',
                'messages': [str(msg) for msg in result.messages] if hasattr(result, 'messages') else [],
                'data': result.data if hasattr(result, 'data') else None
            }
        # Otherwise, convert to string representation
        else:
            return {'output': str(result)}
    except Exception as e:
        return {'output': str(result), 'error': f'Error serializing result: {str(e)}'}

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages and pass through agent responses"""
    try:
        data = request.json
        message = data.get('message', '')
        session_id = data.get('session_id', 'default')
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Initialize conversation if new session
        if session_id not in conversations:
            conversations[session_id] = {
                'messages': []
            }
        
        # Store user message in conversation history
        conversations[session_id]['messages'].append({
            'role': 'user',
            'content': message
        })
        
        # Build conversation context from history
        # Include previous messages so agent has full context
        if len(conversations[session_id]['messages']) > 1:
            # Build conversation string from history
            conversation_parts = []
            for msg in conversations[session_id]['messages'][:-1]:  # Exclude current message
                role = msg['role']
                content = msg['content']
                conversation_parts.append(f"{role.capitalize()}: {content}")
            conversation_parts.append(f"User: {message}")
            conversation_context = "\n\n".join(conversation_parts)
        else:
            # First message in conversation
            conversation_context = message
        
        # Run async agent
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Call the agent with the current message
        # The agent orchestrates everything internally
        agent_result, thought_process, generated_file_path = loop.run_until_complete(
            central_agent(conversation_context, user_id=session_id)
        )
        
        # Log the agent result structure for debugging
        logger.info(f"Agent result type: {type(agent_result)}")
        logger.info(f"Agent result attributes: {dir(agent_result)}")
        if hasattr(agent_result, 'output'):
            logger.info(f"Agent result.output: {agent_result.output}")
            logger.info(f"Agent result.output type: {type(agent_result.output)}")
        if hasattr(agent_result, 'messages'):
            logger.info(f"Agent result.messages: {agent_result.messages}")
        
        # Convert agent result to dict
        result_dict = agent_result_to_dict(agent_result)
        logger.info(f"Result dict: {json.dumps(result_dict, indent=2, default=str)}")
        
        # Extract response message from agent output
        # Try multiple ways to get the response content
        response_content = ''
        
        # First, try to get from result_dict
        if isinstance(result_dict, dict):
            response_content = result_dict.get('output', '')
            # If output is empty, try getting from messages
            if not response_content and result_dict.get('messages'):
                # Get the last message if available
                messages = result_dict.get('messages', [])
                if messages:
                    response_content = str(messages[-1]) if messages else ''
        
        # If still empty, try directly from agent_result
        if not response_content:
            if hasattr(agent_result, 'output') and agent_result.output:
                response_content = str(agent_result.output)
            elif hasattr(agent_result, 'messages') and agent_result.messages:
                # Get the last message from the agent
                response_content = str(agent_result.messages[-1]) if agent_result.messages else ''
        
        # Fallback if still empty
        if not response_content:
            response_content = 'Processing your request...'
        
        logger.info(f"Final response_content: {response_content[:200]}...")
        
        # Store assistant response in conversation history
        conversations[session_id]['messages'].append({
            'role': 'assistant',
            'content': response_content
        })
        
        # Determine status based on response content
        status = 'incomplete'
        file_content = None
        
        # If file was generated, read it and ensure it's included in the response
        if generated_file_path and os.path.exists(generated_file_path):
            logger.info(f"File generated at {generated_file_path}")
            try:
                with open(generated_file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                
                # Check if the response already contains the file in the expected format
                if 'Input parameters file:' not in response_content or '```' not in response_content:
                    # Format the response to include the file content in the expected format
                    response_content = f"{response_content}\n\nInput parameters file:\n```\n{file_content}\n```"
                
                status = 'complete'
            except Exception as e:
                logger.error(f"Error reading generated file: {str(e)}", exc_info=True)
        
        # Also check if response indicates completion
        if 'successfully generated' in response_content.lower() or 'generated' in response_content.lower():
            status = 'complete'
        
        # Serialize thought process for JSON response
        serialized_thought_process = []
        for step in thought_process:
            serialized_step = {}
            for key, value in step.items():
                try:
                    # Try to serialize the value
                    json.dumps(value)
                    serialized_step[key] = value
                except (TypeError, ValueError):
                    # If it can't be serialized, convert to string
                    serialized_step[key] = str(value)
            serialized_thought_process.append(serialized_step)
        
        # Return response in format expected by frontend
        response_data = {
            'message': response_content,  # Frontend expects 'message' field
            'status': status,
            'output': response_content,  # Keep for backward compatibility
            'data': result_dict.get('data') if isinstance(result_dict, dict) else None,
            'thought_process': serialized_thought_process  # Add thought process
        }
        
        # Include file path and content if available
        if generated_file_path:
            response_data['generated_file_path'] = generated_file_path
        if file_content:
            response_data['file_content'] = file_content
        
        return jsonify(response_data)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Server error: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset conversation for a session"""
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        
        if session_id in conversations:
            del conversations[session_id]
        
        return jsonify({'status': 'success', 'message': 'Conversation reset'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def root():
    """Root endpoint - redirect to frontend"""
    return jsonify({
        'message': 'gprMax Chatbot API',
        'status': 'running',
        'frontend': 'http://localhost:3000',
        'endpoints': {
            'health': '/api/health',
            'chat': '/api/chat',
            'file': '/api/file',
            'simulate': '/api/simulate',
            'reset': '/api/reset'
        }
    })

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

@app.route('/api/file', methods=['GET'])
def get_file():
    """Get the generated.in file content"""
    try:
        # Check workspace directory for generated files
        workspace_dir = get_workspace_directory()
        generated_files_dir = workspace_dir / "generated_files"
        
        # Try to find any generated file
        file_path = None
        if generated_files_dir.exists():
            # Look for generated.in or any generated_*.in file
            for file in generated_files_dir.glob("generated*.in"):
                file_path = str(file)
                break
        
        if not file_path or not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return jsonify({
            'content': content,
            'filename': os.path.basename(file_path)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/simulate', methods=['POST'])
def simulate():
    """Run gprMax simulation with the generated input file"""
    try:
        data = request.json
        file_path = data.get('file_path')
        session_id = data.get('session_id', 'default')
        
        if not file_path:
            return jsonify({'error': 'file_path is required'}), 400
        
        if not os.path.exists(file_path):
            return jsonify({'error': f'File not found: {file_path}'}), 404
        
        logger.info(f"Running simulation for file: {file_path}")
        
        # Run async runner agent
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            # Call the tool directly to get raw simulation logs without agent processing
            simulation_result = run_gprmax_simulation_tool(file_path)
            
            # Check if the result indicates an error (contains "failed" or "exit code")
            if "failed" in simulation_result.lower() or "exit code" in simulation_result.lower():
                # It's an error - return the full error message
                logger.error(f"Simulation failed: {simulation_result[:200]}...")
                return jsonify({
                    'status': 'error',
                    'error': simulation_result,
                    'message': 'Simulation failed',
                    'result': simulation_result
                }), 500
            else:
                # Success - the result contains the actual simulation logs
                logger.info(f"Simulation completed successfully. Logs length: {len(simulation_result)} chars")
                
                return jsonify({
                    'status': 'success',
                    'result': simulation_result,  # This is the actual gprMax output
                    'message': 'Simulation completed successfully'
                })
                
        except Exception as e:
            logger.error(f"Error running simulation: {str(e)}", exc_info=True)
            return jsonify({
                'status': 'error',
                'error': str(e),
                'message': f'Error running simulation: {str(e)}'
            }), 500
            
    except Exception as e:
        import traceback
        logger.error(f"Error in simulate endpoint: {str(e)}", exc_info=True)
        return jsonify({
            'error': f'Server error: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    app.run(host='0.0.0.0', port=port, debug=True)

