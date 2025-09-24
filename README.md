# Accelerator Compatibility Agent

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet)](https://github.com/astral-sh/uv)
[![Red Hat](https://img.shields.io/badge/Red%20Hat-OpenShift%20AI-red)](https://www.redhat.com/en/technologies/cloud-computing/openshift/openshift-ai)

<div align="center">
  <img src="assets/banner.png" alt="Accelerator Compatibility Agent" width="600"/>
</div>

An intelligent agent for validating and managing model-accelerator compatibility in Red Hat OpenShift AI deployments. The agent ensures optimal matching between AI/ML models and available hardware accelerators.

## Overview

```mermaid
graph TD
    A[model-car.yaml] --> B[Configuration Parser]
    B --> C[Compatibility Validator]
    C --> D[GPU Memory Analysis]
    C --> E[Accelerator Type Check]
    D --> F[Validation Report]
    E --> F
    F --> G[Compatibility Matrix]
```

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
