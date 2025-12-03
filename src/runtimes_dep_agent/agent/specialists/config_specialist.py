"""Configuration specialist agent."""

from __future__ import annotations

import json
from pathlib import Path
from time import time
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
        Create a *new* modelcar YAML from a base version with optimized serving arguments.

        Accepts EITHER:
        {
            "model_name": "granite-3.1-8b-instruct",
            "serving_arguments": { ... }
        }
        OR:
        [
            {
            "model_name": "...",
            "serving_arguments": { ... }
            },
            ...
        ]

        Behavior:
        - Uses `config-yaml/sample_modelcar_config.base.yaml` as base if it exists,
        otherwise falls back to `config-yaml/sample_modelcar_config.yaml`.
        - Writes the updated config to:
            `config-yaml/sample_modelcar_config.generated.yaml`
        without mutating the base file.
        - For each object, only updates the matching `model-car` entry.
        """
        base_path_primary = Path("config-yaml/sample_modelcar_config.base.yaml")
        base_path_fallback = Path("config-yaml/sample_modelcar_config.yaml")
        output_path = Path("config-yaml/sample_modelcar_config.generated.yaml")

        if base_path_primary.exists():
            modelcar_path = base_path_primary
        elif base_path_fallback.exists():
            modelcar_path = base_path_fallback
        else:
            return (
                "Error: No base model-car configuration file found. "
                f"Expected either {base_path_primary} or {base_path_fallback}."
            )

        try:
            overrides_obj = json.loads(optimized_args_json)
        except json.JSONDecodeError:
            return "Error: Provided serving arguments are not valid JSON."

        if isinstance(overrides_obj, dict):
            overrides_list = [overrides_obj]
        elif isinstance(overrides_obj, list):
            overrides_list = overrides_obj
        else:
            return (
                "Error: Expected a JSON object or a list of objects with "
                "'model_name' and 'serving_arguments'."
            )
        with open(modelcar_path, "r") as f:
            cfg = yaml.safe_load(f) or {}

        model_car_block = cfg.get("model-car", [])
        if isinstance(model_car_block, dict):
            model_list = [model_car_block]
        elif isinstance(model_car_block, list):
            model_list = model_car_block
        else:
            return (
                "Error: 'model-car' section is not a dict or list; "
                "cannot apply overrides."
            )

        updated_models: list[str] = []

        for overrides in overrides_list:
            if not isinstance(overrides, dict):
                continue

            target_name = overrides.get("model_name")
            sa_overrides = overrides.get("serving_arguments") or {}

            if not target_name or not sa_overrides:
                continue

            for entry in model_list:
                if not isinstance(entry, dict):
                    continue

                if entry.get("name") != target_name:
                    continue

                if "serving_arguments" not in entry or not isinstance(entry["serving_arguments"], dict):
                    entry["serving_arguments"] = {}

                entry["serving_arguments"].update(sa_overrides)
                updated_models.append(target_name)
                break

        if not updated_models:
            return (
                "No model-car entries were updated. Check that 'model_name' values "
                "match the names in the model-car config."
            )

        if isinstance(model_car_block, dict):
            cfg["model-car"] = model_list[0]
        else:
            cfg["model-car"] = model_list

        cfg.pop("serving_arguments", None)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            yaml.safe_dump(
                cfg,
                f,
                sort_keys=False,  # keep order sane
                default_flow_style=False,
            )

        updated_str = ", ".join(sorted(set(updated_models)))
        
        return (
            f"Updated serving arguments for model(s): {updated_str}. "
            f"Generated config: {output_path}"
        )

            
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
            - model names of each preloaded model
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
