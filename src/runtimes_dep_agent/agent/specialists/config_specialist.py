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
        :param optimized_args_json: JSON string of optimized serving arguments.
        :return: Confirmation message.
        """
        modelcar_path = Path("config-yaml/sample_modelcar_config.yaml")
        if not modelcar_path.exists():
            return f"Error: model-car configuration file not found at {modelcar_path}."
        try:
            overrides = json.loads(optimized_args_json)
        except json.JSONDecodeError:
            return "Error: Provided serving arguments are not valid JSON."
        
        config_overrides = overrides.get("serving_arguments", {})
        if not config_overrides:
            return "No serving arguments provided to update."
        with open(modelcar_path, "r") as f:
            cfg = yaml.safe_load(f)
        if "serving_arguments" not in cfg:
            cfg["serving_arguments"] = {}
        cfg["serving_arguments"].update(config_overrides)
        with open(modelcar_path, "w") as f:
            yaml.safe_dump(cfg, f)
        return f"Updated serving arguments in {modelcar_path}."
        
        
    prompt = """
        You are the Configuration Specialist.

        Your responsibilities:
        1. Load and analyze the preloaded model-car requirements.
        2. Infer VRAM requirements using the cached model information.
        3. Provide per-model deployment summaries.
        4. When the supervisor or Decision Specialist provides optimized serving arguments,
        apply them to the model-car YAML by calling the tool: `generate_optimal_serving_arguments`.

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
        - When updating YAML, only use the JSON strings given to you. Do not invent keys or values.

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
