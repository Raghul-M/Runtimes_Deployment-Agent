"""Configuration specialist agent."""

from __future__ import annotations

import json
from typing import Callable

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool
from ...config.model_config import calculate_gpu_requirements, optimal_serving_arguments

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
    def generate_optimal_serving_arguments() -> str:
        """Generate optimal serving arguments based on model requirements."""
        if not precomputed_requirements:
            return "No preloaded requirements available to generate serving arguments."
        return optimal_serving_arguments(precomputed_requirements)
        

    prompt = (
        "You are a model-car configuration specialist. "
        "First, print a short checklist showing the steps you are taking with [ ] / [x] (e.g., load cached "
        "requirements, estimate VRAM, compose report) and mark items complete as you go. "
        "Always call describe_preloaded_requirements first, then infer_gpu_needs, before writing the report. "
        "Base your response entirely on the cached JSON returned by describe_preloaded_requirements; do not ask "
        "the user for file paths or additional input. After the checklist, craft a concise, human-readable "
        "deployment report that summarises model count, model size (container image disk footprint), parameter "
        "counts, quantization bits, estimated VRAM needs, and supported architectures. Always include per-model "
        "bullet points, and cite the JSON facts accurately."
    )

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
