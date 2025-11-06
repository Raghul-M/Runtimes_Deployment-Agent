# Runtimes Deployment Agent

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet)](https://github.com/astral-sh/uv)
[![LangChain](https://img.shields.io/badge/Built%20with-LangChain%201.x-orange)](https://python.langchain.com)

Supervisor-driven orchestration for analysing model-car configurations. The tool follows LangChain’s [supervisor pattern](https://docs.langchain.com/oss/python/langchain/supervisor), combining a primary LLM with a configuration specialist that understands Red Hat “model-car” manifests and the container images they reference.

## Features

- **CLI-first workflow** – install the package and run `agent configuration --config …` to query any model-car file.
- **LangChain supervisor** – the `LLMAgent` composes a specialist registry and exposes a single `run_supervisor` entry point.
- **Configuration specialist** – parses YAML, surfaces serving arguments, GPU counts, and other runtime hints as JSON.
- **Container metadata enrichment** – shells out to `skopeo inspect` to capture aggregate image size (GB) and supported CPU architecture per model.
- **Config bootstrap** – pass a bootstrap file at agent creation time so repeated prompts reuse cached requirements.

## Requirements

- Python **3.12+**
- A Google Gemini API key (`GEMINI_API_KEY`) for `langchain-google-genai`
- `skopeo` available on `PATH` (for container metadata; falls back gracefully if missing)
- Dependencies listed in `pyproject.toml`

## Installation

```bash
# Editable install while iterating
uv pip install -e .

# or with pip
pip install -e .
```

## CLI Usage

```bash
export GEMINI_API_KEY="your-key-here"

# Inspect the bundled sample configuration
agent configuration --config config-yaml/sample_modelcar_config.yaml
```

The command prints a **Configuration** section containing the parsed model requirements:

```json
{
  "granite-3.1-8b-instruct": {
    "model_name": "granite-3.1-8b-instruct",
    "image": "oci://registry.redhat.io/rhelai1/modelcar-granite-3-1-8b-instruct:1.5",
    "gpu_count": 1,
    "arguments": [
      "--uvicorn-log-level=info",
      "--max-model-len=2048",
      "--trust-remote-code",
      "--distributed-executor-backend=mp"
    ],
    "model_size_gb": 15.23,
    "supported_arch": "amd64"
  }
}
```

- Use `--config` to point at any other YAML file.
- `LLMAgent` also accepts a `bootstrap_config` parameter if you embed it in your own Python application.

## Configuration Files

- `model-car`: list (or single mapping) describing each model. Keys:
  - `name`: identifier reported in CLI output.
  - `image`: OCI reference (transport prefixes such as `oci://` are supported).
  - `serving_arguments.gpu_count`: advertised GPU count; surfaced alongside metadata.
  - `serving_arguments.args`: extra runtime flags (e.g., `--max-model-len`).
- `default`: optional fallback block; currently used only as documentation.

## Architecture

```
src/runtimes_dep_agent/
├── agent/
│   ├── llm_agent.py          # Supervisor wiring + specialist registry
│   └── specialists/
│       └── config_specialist.py
├── config/
│   └── model_config.py       # YAML + skopeo helpers
├── execute_agent.py          # CLI entry point
├── reports/
│   └── validation_report.py  # Future reporting hooks
└── validators/
    └── accelerator_validator.py
```

- Packages live under `src/runtimes_dep_agent`; installing the project exposes the console script `agent`.
- The specialist tooling is deliberately modular—drop another specialist builder into `agent/specialists/` and register it inside `LLMAgent._initialise_specialists`.

## Development Notes

- Run `python -m compileall src` for a quick syntax check; add pytest suites under `tests/` as functionality grows.
- Regenerate the CLI entry point after edits with `pip install -e .` (the `agent` command resolves to `runtimes_dep_agent.execute_agent:main`).
- When `skopeo` cannot inspect an image (permissions, offline, etc.), the tool falls back to `0` size and `unknown` architecture while still printing other requirements.

## License

Licensed under the [Apache License 2.0](LICENSE).
