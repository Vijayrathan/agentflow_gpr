# GPRMax Input File Generator - Chatbot UI

An intelligent chatbot interface for generating GPRMax input files through natural conversation. The system guides users through providing all necessary parameters until a complete input file is generated.

## Features

- 🤖 **Intelligent Conversation**: Natural language interface for describing GPRMax simulations
- 🎨 **Modern Dark Theme UI**: Beautiful, intuitive React-based chatbot interface
- 🔄 **Iterative Workflow**: System asks for missing parameters until complete
- ✅ **Validation**: Automatic parameter validation before file generation
- 📁 **File Generation**: Automatically generates `generated.in` and `output.json`

## Architecture

- **Backend**: Flask API server handling chat requests and workflow processing
- **Frontend**: React application with modern dark theme UI
- **Agent**: Pydantic AI agent for parameter extraction and validation

## Setup Instructions

### Prerequisites

- Python 3.8+
- Node.js 16+ and npm
- HuggingFace API token (set in `.env` file)

### Backend Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the root directory:
```
HF_TOKEN=your_huggingface_token_here
```

3. Start the Flask backend server:
```bash
python app.py
```

The backend will run on `http://localhost:5002`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the React development server:
```bash
npm start
```

The frontend will run on `http://localhost:8004` and automatically proxy API requests to the backend.

## Usage

1. Start both the backend and frontend servers
2. Open `http://localhost:8004` in your browser
3. Start a conversation by describing your GPRMax simulation
4. The chatbot will guide you through providing all necessary parameters
5. Once complete, the input file will be generated as `generated.in`

### Example Conversation

**User**: "I want to simulate a model with 2 layers."

**Bot**: "The following parameters are missing or incomplete:
- model (dielectric model: 'crim', 'peplinski', 'dobson', or 'mironov')
- title (simulation title)
- source_height_m (source height in meters)
..."

**User**: "Use mironov model, title is '2-layer test', source height 0.07m, domain 0.8m x 0.4m..."

**Bot**: Continues asking for remaining parameters until complete.

## API Endpoints

- `POST /api/chat` - Send a chat message and process workflow
- `POST /api/reset` - Reset conversation for a session
- `GET /api/health` - Health check endpoint

**Note:** The backend runs on port 5002 by default. Make sure the frontend proxy in `frontend/package.json` matches this port.

## Project Structure

```
intelligent_gpr/
├── app.py                 # Flask backend server
├── generator_agent.py     # Main workflow and agent logic
├── physics_modelling.py   # GPRMax file generation
├── requirements.txt       # Python dependencies
├── frontend/              # React frontend
│   ├── src/
│   │   ├── App.js        # Main React component
│   │   ├── App.css       # Styles
│   │   └── index.js      # Entry point
│   └── package.json      # Node dependencies
└── README.md             # This file
```

## Notes

- The chatbot maintains conversation state per session
- Each session tracks the initial input and subsequent user responses
- The workflow continues until all parameters are provided and validated
- Generated files are saved in the project root directory

