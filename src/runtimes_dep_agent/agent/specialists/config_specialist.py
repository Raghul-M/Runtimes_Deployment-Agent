"""Configuration specialist agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool
import yaml
from ...config.model_config import calculate_gpu_requirements

from . import SpecialistSpec


def build_config_specialist(
    llm: BaseChatModel,
    extract_text: Callable[[dict], str],
    precomputed_requirements: dict | None = None,
) -> SpecialistSpec:
    """Return the configuration specialist agent and the supervisor-facing tool."""

    @tool
    def describe_preloaded_requirements() -> str:
        """Return the preloaded model requirements as JSON."""
        return json.dumps(precomputed_requirements, indent=2)
    
    @tool
    def infer_gpu_needs() -> str:
        """Infer VRAM needs from preloaded requirements."""
        if not precomputed_requirements:
            return "No preloaded requirements available to infer VRAM needs."
        total_vram = calculate_gpu_requirements(precomputed_requirements)
        per_model = [
            f"{name}: {info.get('required_vram_gb', 'unknown')} GB"
            for name, info in precomputed_requirements.items()
        ]
        per_model_str = "; ".join(per_model)
        return f"Total VRAM requirements inferred: {total_vram} GB; per-model: {per_model_str}"
    
    @tool
    def generate_optimal_serving_arguments(optimized_args_json: str) -> str:
        """
        Update the modelcar.yaml on disk with optimized serving arguments.

        Expected JSON structure:
        {
        "model_name": "granite-3.1-8b-instruct",
        "serving_arguments": {
            "args": [
            "--uvicorn-log-level=info",
            "--max-model-len=2048",
            "--trust-remote-code",
            "--tensor-parallel-size=1"
            ]
        }
        }

        - model_name: name of the target model in the `model-car` list.
        - serving_arguments: dict to merge into that model's serving_arguments.
        """
        modelcar_path = Path("config-yaml/sample_modelcar_config.yaml")
        if not modelcar_path.exists():
            return f"Error: model-car configuration file not found at {modelcar_path}."

        try:
            overrides = json.loads(optimized_args_json)
        except json.JSONDecodeError:
            return "Error: Provided serving arguments are not valid JSON."

        target_name = overrides.get("model_name")
        sa_overrides = overrides.get("serving_arguments") or {}

        if not target_name:
            return "Error: optimized JSON is missing 'model_name'."
        if not sa_overrides:
            return "No serving arguments provided to update."

        # Load existing YAML
        with open(modelcar_path, "r") as f:
            cfg = yaml.safe_load(f) or {}

        # Normalise model-car to a list
        model_car_block = cfg.get("model-car", [])
        if isinstance(model_car_block, dict):
            model_list = [model_car_block]
        elif isinstance(model_car_block, list):
            model_list = model_car_block
        else:
            return "Error: 'model-car' section is not a dict or list; cannot apply overrides."

        updated_any = False
        for entry in model_list:
            if not isinstance(entry, dict):
                continue

            if entry.get("name") != target_name:
                continue

            if "serving_arguments" not in entry or not isinstance(entry["serving_arguments"], dict):
                entry["serving_arguments"] = {}

            # Merge new serving_arguments into existing block
            entry["serving_arguments"].update(sa_overrides)
            updated_any = True

        if not updated_any:
            return f"No model-car entry matched model_name '{target_name}'."

        # Write back normalised structure
        if isinstance(model_car_block, dict):
            cfg["model-car"] = model_list[0]
        else:
            cfg["model-car"] = model_list

        # Cleanup: remove accidental top-level serving_arguments created by older versions
        if "serving_arguments" in cfg:
            cfg.pop("serving_arguments")

        with open(modelcar_path, "w") as f:
            yaml.safe_dump(cfg, f)

        return f"Updated serving arguments for model '{target_name}' in {modelcar_path}."

        
            
    prompt = """
        You are the Configuration Specialist.

        Your responsibilities:
        1. Load and analyze the preloaded model-car requirements.
        2. Infer VRAM requirements using the cached model information.
        3. Provide per-model deployment summaries.
        4. When the supervisor or Decision Specialist provides optimized serving arguments JSON,
        apply them to the model-car YAML by calling the tool:
        generate_optimal_serving_arguments(optimized_args_json).

        Rules:
        - Always begin your reasoning with a short checklist using [ ] and [x].
        - The checklist must always include:
            [ ] Load preloaded requirements
            [ ] Infer VRAM needs
            [ ] Prepare configuration summary
        (Only include YAML-update tasks if explicitly requested.)
        - For normal config analysis, you must call describe_preloaded_requirements() first,
        then infer_gpu_needs(), before writing any report.
        - Never ask the user for file paths or external input.

        Handling optimized serving arguments:
        - If the request you receive contains an 'OPTIMIZED_SERVING_ARGUMENTS_JSON' block with a JSON
        code fence, you MUST:
        1) Extract the JSON content inside the ```json ... ``` fence exactly.
        2) Call generate_optimal_serving_arguments(optimized_args_json=<that JSON string>).
        3) Return ONLY the message returned by generate_optimal_serving_arguments, without additional prose.

        Output Requirements:
        - For configuration reports, provide a clean and concise summary:
            - model names
            - image size (GB)
            - parameter counts
            - quantization bits
            - estimated VRAM needs
            - supported architectures
        - When updating YAML, respond only with the return value of the tool you call.

        Tools available:
        - describe_preloaded_requirements(): returns cached model-car fields as JSON
        - infer_gpu_needs(): computes estimated VRAM from cached requirements
        - generate_optimal_serving_arguments(optimized_args_json): updates modelcar.yaml on disk

        Use the tools appropriately based on the user's request.
        """


    agent = create_agent(
        llm,
        tools=[describe_preloaded_requirements, infer_gpu_needs, generate_optimal_serving_arguments],
        system_prompt=prompt,
    )

    @tool
    def analyze_model_config(request: str) -> str:
        """Delegate configuration requests to the configuration specialist."""
        result = agent.invoke({"messages": [{"role": "user", "content": request}]})
        return extract_text(result)

    analyze_model_config.name = "analyze_model_config"

    return SpecialistSpec(
        name="config_specialist",
        agent=agent,
        tool=analyze_model_config,
    )


__all__ = ["build_config_specialist"]
