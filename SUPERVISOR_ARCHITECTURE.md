# Supervisor Architecture Documentation

## Overview

This project now implements a **supervisor pattern** multi-agent architecture, following the LangChain subagents pattern. The central supervisor agent coordinates specialized sub-agents, each with their own focused responsibilities and toolset.

## Architecture

```
                    ┌─────────────────────┐
                    │  Supervisor Agent    │
                    │  (Central Router)    │
                    └──────────┬───────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
                ▼              ▼              ▼
    ┌─────────────────┐  ┌──────────────┐  ┌──────────────┐
    │ Extraction Agent│  │  RAG Agent   │  │ Setup Agent │
    │                 │  │              │  │             │
    │ - extract_      │  │ - search_    │  │ - validate_ │
    │   parameters    │  │   knowledge  │  │   gpr_      │
    │                 │  │              │  │   params    │
    │                 │  │              │  │ - check_    │
    │                 │  │              │  │   complete  │
    │                 │  │              │  │ - generate_ │
    │                 │  │              │  │   file      │
    └─────────────────┘  └──────────────┘  └──────┬──────┘
                                                   │
                                                   ▼
                                         ┌─────────────────┐
                                         │ Simulation Agent│
                                         │                 │
                                         │ - run_gprmax_   │
                                         │   simulation     │
                                         └─────────────────┘
```

## Components

### 1. Supervisor Agent (`supervisor_agent.py`)

The central coordinator that:

- Understands user requests
- Routes requests to appropriate sub-agents
- Can orchestrate multiple sub-agents in sequence
- Provides unified responses to users

**Tools Available to Supervisor:**

- `setup_simulation`: Routes to sim_setup_agent
- `run_simulation`: Routes to simulation_agent
- `search_knowledge`: Routes to rag_agent
- `extract_parameters`: Routes to extraction_agent

### 2. Sub-Agents

#### Extraction Agent

**Purpose**: Extract GPR simulation parameters from natural language queries

**Tools**:

- `extraction_agent`: Extracts structured parameters from user input

**When to use**: User provides initial requirements or parameter specifications

#### Sim Setup Agent

**Purpose**: Validate parameters and generate gprMax input files

**Tools**:

- `generate_gprmax_input_file_tool`: Generate input files
- `check_input_completeness`: Check if all required parameters are present
- `validate_gpr_parameters`: Validate parameters against physics constraints

**When to use**: Need to validate parameters or generate input files

#### Simulation Agent

**Purpose**: Execute gprMax simulations

**Tools**:

- `run_gprmax_simulation_tool`: Run simulations with input files

**When to use**: User wants to execute a simulation

#### RAG Agent

**Purpose**: Retrieve geophysics knowledge from research documents

**Tools**:

- `search_geophysics_knowledge`: Search the knowledge base

**When to use**: User asks questions about geophysics concepts, models, or best practices

## Workflow Examples

### Example 1: Complete Simulation Workflow

1. User: "Create a 3-layer simulation with..."
2. Supervisor → Extraction Agent: Extract parameters
3. Supervisor → Setup Agent: Validate and generate input file
4. Supervisor → Simulation Agent: Run simulation
5. Supervisor: Return results to user

### Example 2: Knowledge Query

1. User: "What is the frequency range for Peplinski model?"
2. Supervisor → RAG Agent: Search knowledge base
3. Supervisor: Return information to user

### Example 3: Parameter Validation

1. User: "Generate a file with these parameters..."
2. Supervisor → Setup Agent: Validate parameters
3. If invalid → Supervisor: Ask user for corrections
4. If valid → Supervisor → Setup Agent: Generate file

## File Structure

```
backend/
├── supervisor_agent.py      # Main supervisor and sub-agents
├── sim_setup_agent.py        # Tools for simulation setup
├── simulation_agent.py      # Tools for running simulations
├── extraction_agent.py       # Parameter extraction tool
├── rag.py                    # RAG system implementation
├── app.py                    # Flask API (uses supervisor_agent)
└── generator_agent.py         # Legacy (kept for runner_agent)
```

## Key Benefits

1. **Separation of Concerns**: Each agent has a focused responsibility
2. **Modularity**: Easy to update or replace individual agents
3. **Scalability**: Easy to add new sub-agents
4. **Maintainability**: Clear boundaries between different domains
5. **Flexibility**: Supervisor can orchestrate complex multi-step workflows

## Usage

The supervisor agent is used through the Flask API in `app.py`:

```python
from supervisor_agent import supervisor_agent

# In the chat endpoint:
agent_result, thought_process, generated_file_path = await supervisor_agent(
    conversation_context,
    user_id=session_id
)
```

## Implementation Details

- All sub-agents are created using LangChain's `create_agent`
- Sub-agents are wrapped as tools for the supervisor using the `@tool` decorator
- The supervisor uses LangChain's `create_agent` with the wrapped sub-agent tools
- All tools are async to support concurrent operations
- Context is passed from supervisor to sub-agents via `ToolRuntime`

## Future Enhancements

- Add human-in-the-loop review for critical operations
- Implement agent-to-agent handoffs for complex workflows
- Add monitoring and logging for agent decisions
- Support for parallel agent execution where appropriate
