# Accelerator Compatibility Agent

An intelligent agent for validating and managing accelerator compatibility in RHOAI model validation automation.

## Core Functions

- Parse and validate model-car configurations
- Analyze accelerator requirements (CUDA, ROCm, vLLM-Spyre-x86 etc.)
- Verify GPU capacity and memory compatibility
- Generate compatibility reports
- Skip unsupported model-accelerator combinations

## Project Structure

```
.
├── src/
│   ├── agent/
│   │   └── llm_agent.py         # Main LLM agent orchestrator
│   ├── config/
│   │   ├── model_config.py      # Configuration parser
│   │   └── model-car.yaml       # Master model configuration
│   ├── validators/
│   │   └── accelerator_validator.py  # Compatibility validators
│   └── reports/
│       └── validation_report.py  # Validation report generator
└── tests/
    └── test_llm_agent.py        # Test suite
