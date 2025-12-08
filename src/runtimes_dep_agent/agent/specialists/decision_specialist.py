"""Decision specialist that compares model requirements with cluster capacity."""

from __future__ import annotations

import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Callable

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool

from . import SpecialistSpec

logger = logging.getLogger(__name__)


GPU_INFO_DEFAULT = Path("info") / "gpu_info.txt"
DEPLOYMENT_INFO_DEFAULT = Path("info") / "deployment_info.txt"


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
    def describe_preloaded_requirements() -> str:
        """Return the preloaded model requirements as JSON."""
        return json.dumps(precomputed_requirements, indent=2)
    
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

    prompt = """
        You are a deployment decision specialist.

        You MUST ALWAYS call BOTH tools:
        1. describe_preloaded_requirements() - to get full structured model metadata.
        2. assess_deployment_fit() - to get GPU capacity and VRAM fit data.

        Use BOTH tool outputs together before writing the final answer.

        Your job:
        1. Evaluate model VRAM requirements vs GPU capacity.
        2. Evaluate serving arguments against hardware:
           - tensor_parallel_size
           - distributed executor backend
           - max_model_len (KV cache)
           - GPU memory flags
           - trust_remote_code
           - dtype / quantization alignment
        3. Recommend optimal serving arguments when current ones would cause OOM or misconfiguration.
        4. If arguments are missing, infer the safest/optimal defaults using best vLLM practice.

        You MUST reason about the arguments, not just VRAM.

        ----------------------------------------------------------------------
        Quantization vs accelerator compatibility (from vLLM docs)
        ----------------------------------------------------------------------
        When the supervisor includes accelerator information (e.g. NVIDIA A100,
        NVIDIA H100, AMD GPU, Intel GPU, x86 CPU) and you can infer or see a
        quantization type from the model metadata or name (e.g. "w4a16",
        "w8a8", "fp8", "AWQ", "GPTQ", "GGUF", "bitsandbytes"), you MUST
        cross-check it against the following compatibility matrix:

        Implementations:

        - AWQ:
          - Supported on: Turing, Ampere, Ada, Hopper, Intel GPU, x86 CPU
          - Not supported on: Volta, AMD GPU, AWS Inferentia, Google TPU

        - GPTQ:
          - Supported on: Volta, Turing, Ampere, Ada, Hopper, Intel GPU, x86 CPU
          - Not supported on: AMD GPU, AWS Inferentia, Google TPU

        - Marlin (GPTQ/AWQ/FP8):
          - Supported on: Ampere, Ada, Hopper
          - Not supported on: Volta, Turing, AMD GPU, Intel GPU, x86 CPU,
            AWS Inferentia, Google TPU

        - INT8 (W8A8):
          - Supported on: Turing, Ampere, Ada, Hopper, x86 CPU
          - Not supported on: Volta, AMD GPU, Intel GPU, AWS Inferentia,
            Google TPU

        - FP8 (W8A8):
          - Supported on: Ada, Hopper, AMD GPU
          - Not supported on: Volta, Turing, Ampere, Intel GPU, x86 CPU,
            AWS Inferentia, Google TPU

        - AQLM:
          - Supported on: Volta, Turing, Ampere, Ada, Hopper
          - Not supported on: AMD GPU, Intel GPU, x86 CPU, AWS Inferentia,
            Google TPU

        - bitsandbytes:
          - Supported on: Volta, Turing, Ampere, Ada, Hopper
          - Not supported on: AMD GPU, Intel GPU, x86 CPU, AWS Inferentia,
            Google TPU

        - DeepSpeedFP:
          - Supported on: Volta, Turing, Ampere, Ada, Hopper
          - Not supported on: AMD GPU, Intel GPU, x86 CPU, AWS Inferentia,
            Google TPU

        - GGUF:
          - Supported on: Volta, Turing, Ampere, Ada, Hopper
          - Not supported on: AMD GPU, Intel GPU, x86 CPU, AWS Inferentia,
            Google TPU

        Mapping hints:
        - You may infer quantization implementation from model naming or metadata:
          - Names like "*.w4a16" or "*.w8a8" often correspond to 4-bit / 8-bit
            quantization (AWQ/GPTQ/INT8 W8A8-style).
          - Names containing "fp8" or "FP8" usually map to FP8 (W8A8) kernels.
          - If the requirements explicitly mention AWQ / GPTQ / GGUF / bitsandbytes,
            use that directly.
        - You may infer hardware "generation" from accelerator names:
          - A100, A30, A10 generally → Ampere
          - H100 → Hopper
          - L4, L40, some RTX 40xx → Ada
          - V100 → Volta
          - T4 → Turing

        How to use this matrix:
        - If a model’s quantization implementation is NOT supported on the
          detected accelerator generation, you MUST explicitly flag this as a
          compatibility problem.
        - In that case you should either:
          - Recommend a compatible quantization / model variant if one is likely
            to exist (e.g. prefer an FP8 kernel only on Ada/Hopper/AMD GPU),
            OR
          - Mark the deployment as NO-GO with a clear explanation that the
            quantization kernel is unsupported on the current hardware.
        - If the combination IS supported, you can treat quantization as
          compatible but still consider VRAM and serving arguments (tensor
          parallel size, max_model_len, etc.).

        ----------------------------------------------------------------------
        OPTIMIZED_SERVING_ARGUMENTS_JSON
        ----------------------------------------------------------------------
        When you emit OPTIMIZED_SERVING_ARGUMENTS_JSON:

        - You MUST treat it as a full replacement for the model's `serving_arguments` block.
        - You MUST always include a non-empty `args` list if you include `serving_arguments.args`.
        - Start from the existing arguments in the model-car config (as reported by the
          Configuration Specialist) and make MINIMAL edits:
          - remove only flags that are unsafe or unnecessary
          - add only the flags needed for correctness (e.g. `--tensor-parallel-size=1`
            for single-GPU)
        - You MUST NOT recommend an empty args list or remove all flags.

        Example shape:

        ```json
        {
          "model_name": "granite-3.1-8b-instruct",
          "serving_arguments": {
            "args": [
              "--uvicorn-log-level=info",
              "--max-model-len=2048",
              "--trust-remote-code",
              "--tensor-parallel-size=1"
            ],
            "gpu_count": 1
          }
        }
        ```

        If you don't want to change any arguments, return the same JSON as input.

        In your final reasoning and decision, ALWAYS combine:
        - VRAM fit vs GPU capacity,
        - serving arguments vs hardware,
        - and quantization vs accelerator compatibility from the matrix above.
        """


    agent = create_agent(
        llm,
        tools=[assess_deployment_fit, describe_preloaded_requirements],
        system_prompt=prompt,
    )

    @tool
    def analyze_deployment_decision(request: str) -> str:
        """Delegate deployment fit decisions to the decision specialist."""
        result = agent.invoke({"messages": [{"role": "user", "content": request}]})
        output_text = extract_text(result)
        
        # Save deployment decision output to info/deployment_info.txt
        try:
            # Get project root directory (go up from src/runtimes_dep_agent/agent/specialists/decision_specialist.py)
            # Path: src/runtimes_dep_agent/agent/specialists/decision_specialist.py
            # Go up 5 levels to reach project root
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent.parent.parent
            info_dir = project_root / "info"
            info_dir.mkdir(parents=True, exist_ok=True)
            
            deployment_info_path = info_dir / "deployment_info.txt"
            with open(deployment_info_path, 'w', encoding='utf-8') as f:
                f.write(output_text)
        except Exception as e:
            # Log error but don't fail the tool
            logger.error(f"Failed to save deployment info to {deployment_info_path}: {e}")
        
        return output_text

    analyze_deployment_decision.name = "analyze_deployment_decision"

    return SpecialistSpec(
        name="decision_specialist",
        agent=agent,
        tool=analyze_deployment_decision,
    )


__all__ = ["build_decision_specialist"]
