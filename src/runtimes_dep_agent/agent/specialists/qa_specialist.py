""" QA Specialist for running """

from __future__ import annotations
from typing import Callable
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool
from . import SpecialistSpec
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_qa_specialist(
        llm: BaseChatModel,
        gh_repo_path: str,
        supported_accelerator_type: str,
        vllm_runtime_image: str,
        modelcar_image_name: str
) -> SpecialistSpec:
    """Return the QA specialist agent and the supervisor-facing tool."""
    
    @tool
    def run_model_validation() -> str:
        """Run model validation and return results."""
        command = ["uv", "run", "pytest", "-vv", gh_repo_path,
                     f"--supported-accelerator-type={supported_accelerator_type}",
                     f"--vllm-runtime-image={vllm_runtime_image}",
                     f"--modelcar_image_name={modelcar_image_name}",
                     "--registry-host=registry.redhat.io",
                     "--snapshot-update"
                     ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info("Model validation completed successfully.")
            return result.stdout
        except subprocess.CalledProcessError as exc:
            logger.error("Model validation failed: %s", exc)
            return exc.stdout + "\n" + exc.stderr
        
    prompt = (
        "You are a QA specialist for validating model-car deployments. "
        "Use the provided tools to run model validation tests and return the results."
    )

    agent = create_agent(
        llm,
        tools=[run_model_validation],
        system_prompt=prompt,
    )

    return agent

__all__ = ["build_qa_specialist"]