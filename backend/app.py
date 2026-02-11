from flask import Flask, request, jsonify
from flask_cors import CORS
import asyncio
import json
import logging
import os
from langchain_core.messages import HumanMessage

# Import LangGraph workflow
from langgraph_workflow import langgraph_app
from simulation_agent import run_gprmax_simulation_tool

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# No longer needed - LangGraph state is already a dict

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages using LangGraph workflow"""
    try:
        data = request.json
        message = data.get('message', '')
        session_id = data.get('session_id', 'default')
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        logger.info(f"[CHAT] Session: {session_id}, Message: {message[:100]}...")
        
        # Configure LangGraph with thread_id for persistence
        config = {"configurable": {"thread_id": session_id}}
        
        # Run async LangGraph workflow
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Invoke LangGraph workflow with new user message
        result = loop.run_until_complete(
            langgraph_app.ainvoke(
                {"messages": [HumanMessage(content=message)]},
                config=config
            )
        )
        
        logger.info(f"[CHAT] LangGraph result keys: {result.keys()}")
        
        # Extract response from LangGraph state
        messages = result.get("messages", [])
        response_content = ""
        
        if messages:
            # Get the last AI message
            for msg in reversed(messages):
                if hasattr(msg, 'content') and msg.type != "human":
                    response_content = msg.content
                    break
        
        if not response_content:
            response_content = "Processing your request..."
        
        logger.info(f"[CHAT] Response: {response_content[:200]}...")
        
        # Extract state information
        file_path = result.get("file_path")
        file_content = result.get("file_content")
        file_generated = result.get("file_generated", False)
        validation_errors = result.get("validation_errors", {})
        parameters_complete = result.get("parameters_complete", False)
        
        # Determine status
        status = 'complete' if file_generated else 'incomplete'
        
        logger.info(f"[CHAT] Status: {status}, File generated: {file_generated}")
        
        # Helper function to check if response contains validation errors
        def has_validation_errors(response_text):
            """Check if the response indicates validation errors that need to be fixed"""
            response_lower = response_text.lower()
            error_indicators = [
                'must be between',
                'must be',
                'required',
                'validation error',
                'validation issues',
                'outside these required ranges',
                'does not resolve',
                'still apply',
                'please:',
                'please ',
                'adjust the',
                'select a',
                'update these values',
                'errors still',
                'the following errors',
                'error:',
                'invalid',
                'incorrect',
                'missing',
                'not valid',
                'not complete'
            ]
            # Check if response contains error indicators but doesn't contain success indicators
            has_errors = any(indicator in response_lower for indicator in error_indicators)
            has_success = any(phrase in response_lower for phrase in [
                'successfully generated',
                'generated successfully',
                'input file has been generated',
                'file has been generated'
            ])
            # If it has errors and no success, it's a validation error response
            return has_errors and not has_success
        
        # Helper function to check if model is presenting the file
        def is_presenting_file(response_text):
            """Check if the model's response indicates it is presenting the input file"""
            response_lower = response_text.lower()
            # Check if response already contains the file format (model is presenting it)
            if 'input parameters file:' in response_text and '```' in response_text:
                return True
            # Check if response indicates successful generation without errors
            success_phrases = [
                'successfully generated',
                'generated successfully',
                'input file has been generated',
                'file has been generated',
                'here is the generated',
                'the generated input file',
                'gprmax input file:'
            ]
            has_success = any(phrase in response_lower for phrase in success_phrases)
            has_errors = has_validation_errors(response_text)
            return has_success and not has_errors
        
        # Return response in format expected by frontend
        response_data = {
            'message': response_content,
            'status': status,
            'output': response_content,
            'parameters_complete': parameters_complete,
            'validation_errors': validation_errors,
            'file_generated': file_generated,
        }
        
        # Include file path and content if available
        if file_path:
            response_data['generated_file_path'] = file_path
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
        
        # LangGraph handles persistence via checkpoints
        # User can just start a new session with a different thread_id
        # No manual cleanup needed
        
        return jsonify({'status': 'success', 'message': 'Conversation reset. Use a new session_id for a fresh start.'})
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
        from sim_setup_agent import get_workspace_directory
        
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
            # Read the file content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
            except Exception as e:
                error_msg = f"Failed to read input file: {str(e)}"
                logger.error(error_msg)
                return jsonify({'error': error_msg}), 500
            
            # Call the tool using invoke() method since it's a StructuredTool object
            simulation_result = run_gprmax_simulation_tool.invoke({"input_file_content": file_content})
            
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

