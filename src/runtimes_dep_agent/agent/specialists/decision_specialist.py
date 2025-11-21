"""Decision specialist that compares model requirements with cluster capacity."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Callable

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool

from . import SpecialistSpec


GPU_INFO_DEFAULT = Path("gpu_info") / "gpu_info.txt"


def _parse_gpu_summary(text: str) -> tuple[int, float | None]:
    """Extract (total_gpus, per_gpu_mem_gb) from the GPU info text."""
    total = 0
    per_gpu = None
    for line in text.splitlines():
        normalized = line.lower()
        if "allocatable gpus" in normalized:
            match = re.search(r"(\d+)", line)
            if match:
                total += int(match.group(1))
        if ("per-gpu memory" in normalized or "per gpu memory" in normalized) and per_gpu is None:
            match = re.search(r"(\d+(?:\.\d+)?)", line)
            if match:
                try:
                    per_gpu = float(match.group(1))
                except ValueError:
                    continue
        if per_gpu is None and "gpu product" in normalized:
            match = re.search(r"(\d+(?:\.\d+)?)\s*gb", line, re.IGNORECASE)
            if match:
                try:
                    per_gpu = float(match.group(1))
                except ValueError:
                    continue
    return total, per_gpu


def build_decision_specialist(
    llm: BaseChatModel,
    extract_text: Callable[[dict], str],
    precomputed_requirements: dict | None = None,
) -> SpecialistSpec:
    """Create the decision specialist that determines deployment feasibility."""

    @tool
    def assess_deployment_fit(file_path: str | None = None) -> str:
        """Evaluate whether cached models fit on the cluster GPUs (optionally override GPU info path)."""
        if not precomputed_requirements:
            return "Deployment Fit Analysis:\n- No preloaded model requirements available."

        gpu_file = Path(file_path) if file_path else GPU_INFO_DEFAULT
        if not gpu_file.exists():
            return f"Deployment Fit Analysis:\n- GPU info file not found at {gpu_file}."

        try:
            gpu_text = gpu_file.read_text(encoding="utf-8")
        except OSError as exc:
            return f"Deployment Fit Analysis:\n- Error reading GPU info file ({gpu_file}): {exc}"

        total_gpus, per_gpu_mem = _parse_gpu_summary(gpu_text)

        per_model_lines = []
        total_required = 0
        for name, info in precomputed_requirements.items():
            required_vram = info.get("required_vram_gb")
            if required_vram and per_gpu_mem:
                needed = max(1, math.ceil(required_vram / per_gpu_mem))
                total_required += needed
                per_model_lines.append(
                    f"- {name}: needs ~{needed} GPU(s) (requires {required_vram} GB; ~{per_gpu_mem} GB per GPU)"
                )
            elif required_vram:
                per_model_lines.append(
                    f"- {name}: requires {required_vram} GB VRAM but per-GPU memory is unknown."
                )
            else:
                per_model_lines.append(f"- {name}: VRAM requirement could not be inferred.")

        comparison = "Insufficient data to compare cluster capacity with model needs."
        if per_gpu_mem and total_required:
            if total_gpus >= total_required:
                comparison = (
                    f"Cluster GPUs available ({total_gpus}) meet or exceed the inferred need ({total_required})."
                )
            else:
                comparison = (
                    f"Cluster GPUs available ({total_gpus}) are below the inferred need ({total_required})."
                )

        per_model_report = "\n".join(per_model_lines) if per_model_lines else "- No models found."
        return (
            "Deployment Fit Analysis:\n"
            f"- Source GPU file: {gpu_file}\n"
            f"- Total GPUs available: {total_gpus}\n"
            f"- Per-GPU memory (parsed): {per_gpu_mem or 'unknown'} GB\n"
            "- Per-model breakdown:\n"
            f"{per_model_report}\n"
            f"- Comparison: {comparison}"
        )

    prompt = (
        "You are a deployment decision specialist. Use the cached requirements and the GPU info file to decide "
        "whether the models fit on the available hardware. Always call assess_deployment_fit to produce the "
        "final analysis. Provide a clear comparison (e.g., model VRAM vs per-GPU memory) and state whether the "
        "deployment is feasible."
    )

    agent = create_agent(
        llm,
        tools=[assess_deployment_fit],
        system_prompt=prompt,
    )

    @tool
    def analyze_deployment_decision(request: str) -> str:
        """Delegate deployment fit decisions to the decision specialist."""
        result = agent.invoke({"messages": [{"role": "user", "content": request}]})
        return extract_text(result)

    analyze_deployment_decision.name = "analyze_deployment_decision"

    return SpecialistSpec(
        name="decision_specialist",
        agent=agent,
        tool=analyze_deployment_decision,
    )


__all__ = ["build_decision_specialist"]
