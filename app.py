from flask import Flask, request, jsonify
from flask_cors import CORS
import asyncio
import json
from generator_agent import run_workflow
import os

app = Flask(__name__)
CORS(app)

# Store conversation history per session
conversations = {}

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages and process workflow"""
    try:
        data = request.json
        message = data.get('message', '')
        session_id = data.get('session_id', 'default')
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Initialize conversation if new session
        if session_id not in conversations:
            conversations[session_id] = {
                'messages': [],
                'initial_input': message,
                'user_responses': []
            }
        else:
            # Add to user responses for iterative workflow
            conversations[session_id]['user_responses'].append(message)
        
        # Run the workflow
        initial_input = conversations[session_id]['initial_input']
        user_responses = conversations[session_id]['user_responses']
        
        # Run async workflow
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(
            run_workflow(initial_input, user_responses if user_responses else None)
        )
        
        # Store message in conversation history
        conversations[session_id]['messages'].append({
            'role': 'user',
            'content': message
        })
        
        # Process result
        if result.get("status") == "complete":
            # Save output to file
            output_data = result.get("data", {})
            with open("output.json", "w") as f:
                json.dump(output_data, f, indent=2)
            
            response_message = f"✅ Successfully generated GPRMax input file!\n\n{result.get('output', 'File generated successfully.')}\n\nThe configuration has been saved to output.json and the input file has been generated."
            
            conversations[session_id]['messages'].append({
                'role': 'assistant',
                'content': response_message
            })
            
            return jsonify({
                'status': 'complete',
                'message': response_message,
                'data': output_data
            })
        
        elif result.get("status") == "incomplete":
            response_message = result.get("user_message", "Please provide the missing parameters.")
            
            conversations[session_id]['messages'].append({
                'role': 'assistant',
                'content': response_message
            })
            
            return jsonify({
                'status': 'incomplete',
                'message': response_message,
                'missing_params': result.get('missing_params', '')
            })
        
        elif result.get("status") == "validation_error":
            response_message = result.get("user_message", "Please correct the validation errors.")
            
            conversations[session_id]['messages'].append({
                'role': 'assistant',
                'content': response_message
            })
            
            return jsonify({
                'status': 'validation_error',
                'message': response_message,
                'error': result.get('error', '')
            })
        
        elif result.get("status") == "error":
            error_message = f"An error occurred: {result.get('error', 'Unknown error')}"
            
            conversations[session_id]['messages'].append({
                'role': 'assistant',
                'content': error_message
            })
            
            return jsonify({
                'status': 'error',
                'message': error_message
            }), 500
        
        else:
            return jsonify({
                'status': 'unknown',
                'message': 'Unknown status returned from workflow'
            }), 500
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'Server error: {str(e)}'
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

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    app.run(host='0.0.0.0', port=port, debug=True)

