# Accelerator Compatibility Agent

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet)](https://github.com/astral-sh/uv)
[![LangGraph](https://img.shields.io/badge/Inspired%20by-LangGraph-orange)](https://langchain-ai.github.io/langgraph/)
[![Red Hat](https://img.shields.io/badge/Red%20Hat-OpenShift%20AI-red)](https://www.redhat.com/en/technologies/cloud-computing/openshift/openshift-ai)

<div align="center">
<pre>
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║             Accelerator Compatibility Agent                   ║
║                                                              ║
║      Validating Model-Accelerator Compatibility in RHOAI     ║
║                                                              ║
╚═══════════════════════════════════════════════════════════════╝
</pre>
</div>

An intelligent agent for validating and managing model-accelerator compatibility in Red Hat OpenShift AI deployments. The agent ensures optimal matching between AI/ML models and available hardware accelerators.

## Multi-Agent Architecture

This project implements a Supervisor-based multi-agent architecture inspired by [LangGraph's multi-agent approach](https://langchain-ai.github.io/langgraph/concepts/multi_agent/). The system consists of:

- **Supervisor Agent**: A central LLM-powered agent that coordinates workflows and makes high-level decisions
- **Specialized Agent Nodes**: Task-specific agents focused on configuration parsing, compatibility validation, and reporting
- **Command Routing**: Uses the Command pattern to route execution between agent nodes based on the supervisor's decisions

### Supervisor Architecture

```mermaid
graph TD
    subgraph "Supervisor LLM"
        S[LLM Agent]
        DP[Decision Process]
        TR[Tool Router]
    end
    
    subgraph "Agent Nodes"
        CP[Config Parser]
        VA[Validator]
        RG[Report Generator]
    end
    
    U[User Query] -->|Input| S
    S -->|Analyze task| DP
    DP -->|Select tool| TR
    
    TR -->|Parse config| CP
    TR -->|Check compatibility| VA
    TR -->|Generate report| RG
    
    CP -->|Return results| S
    VA -->|Return results| S
    RG -->|Return results| S
    
    S -->|Final response| R[Response]
    
    classDef supervisor fill:#ffcfcf,stroke:#ff0000
    classDef agents fill:#d4f1f9,stroke:#0080ff
    classDef flow fill:#fff,stroke:#333
    
    class S,DP,TR supervisor
    class CP,VA,RG agents
    class U,R flow
```

This architecture provides several advantages:
- **Modularity**: Separate agents make the system easier to develop and maintain
- **Specialization**: Each agent focuses on a specific domain
- **Control**: The supervisor explicitly controls agent communication
- **Flexibility**: Supports both sequential and parallel agent execution

### Execution Flow

```mermaid
sequenceDiagram
    participant User
    participant Supervisor as Supervisor LLM
    participant Config as Config Parser Agent
    participant Validator as Validator Agent
    participant Reporter as Report Generator Agent
    
    User->>Supervisor: Query about model compatibility
    
    Supervisor->>Supervisor: Analyze query
    Supervisor->>Config: Request model configuration parsing
    Config->>Config: Parse model-car.yaml
    Config->>Supervisor: Return model specifications
    
    Supervisor->>Validator: Request compatibility check
    Validator->>Validator: Validate GPU requirements
    Validator->>Validator: Check accelerator types
    Validator->>Supervisor: Return compatibility results
    
    Supervisor->>Reporter: Generate compatibility report
    Reporter->>Reporter: Format validation results
    Reporter->>Supervisor: Return formatted report
    
    Supervisor->>User: Deliver final response
```

## Agent Workflow

```mermaid
graph TD
    A[model-car.yaml] --> B[Configuration Parser Agent]
    B --> C[Compatibility Validator Agent]
    C --> D[GPU Memory Analysis]
    C --> E[Accelerator Type Check]
    D --> F[Report Generator Agent]
    E --> F
    F --> G[Compatibility Matrix]
    
    S[Supervisor LLM] --> B
    S --> C
    S --> F
    B --> S
    C --> S
    F --> S
    S --> R[Final Response]
```

## Core Functions

- Parse and validate model-car configurations
- Analyze accelerator requirements (CUDA, ROCm, vLLM-Spyre-x86 etc.)
- Verify GPU capacity and memory compatibility
- Generate compatibility reports
- Skip unsupported model-accelerator combinations

## Implementation Details

The agent system is built using the Supervisor pattern where:

- The LLM acts as the supervisor that decides which specialized agent to call
- Agent nodes are implemented as tools that the supervisor can invoke
- Communication happens through a shared state that passes between agents
- The Command pattern routes execution to the appropriate agent based on the supervisor's decision

### Command Routing Mechanism

```mermaid
classDiagram
    class LLMAgent {
        +llm: ChatGoogleGenerativeAI
        +agent_nodes: Dict~str, Callable~
        +tools: List~Any~
        +register_agent_node(name, agent_func)
        +register_tool(tool_func)
        +route_command(agent_name, *args, **kwargs)
        +execute_parallel(agent_calls)
        +create_supervisor()
        +run_supervisor(user_input)
    }
    
    class Command {
        +agent_name: str
        +args: List
        +kwargs: Dict
        +execute(agent_nodes)
    }
    
    class AgentNode {
        +process(input_data)
        +execute()
    }
    
    LLMAgent --> Command : creates
    LLMAgent --> AgentNode : registers
    Command --> AgentNode : routes to
```

This architecture supports both sequential agent execution and parallel processing, making it suitable for complex workflows.

## Project Structure

```
.
├── src/
│   ├── agent/
│   │   └── llm_agent.py         # Supervisor LLM agent orchestrator
│   ├── config/
│   │   ├── model_config.py      # Configuration parser agent
│   │   └── model-car.yaml       # Master model configuration
│   ├── validators/
│   │   └── accelerator_validator.py  # Compatibility validator agent
│   └── reports/
│       └── validation_report.py  # Report generator agent
├── execute_agent.py             # Main entry point for running the agent
└── tests/
    └── test_llm_agent.py        # Test suite
