""" QA Specialist for running """

from __future__ import annotations
import os
import tempfile
from typing import Callable
from anyio import Path
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
        extract_text: Callable[[dict], str],
        precomputed_requirements: dict | None = None,
) -> SpecialistSpec:
    """Return the QA specialist agent and the supervisor-facing tool."""
    
    @tool
    def run_odh_tests(image_tag: str = "latest") -> str:
        """Run the ODH model validation test suite using a fixed kubeconfig mount."""

        image = f"quay.io/opendatahub/opendatahub-tests:{image_tag}"

        user_kubeconfig = os.path.expanduser("~/.kube/config")
        results_dir = "/tmp/opendatahub-tests-results"

        # Ensure results directory exists
        os.makedirs(results_dir, exist_ok=True)

        try:
            result = subprocess.run(
                [
                    "podman", "run", "--rm",
                    "-e", "KUBECONFIG=/home/odh/.kube/config",
                    "-v", f"{user_kubeconfig}:/home/odh/.kube/config:Z",
                    "-v", f"{results_dir}:/home/odh/opendatahub-tests/results:Z",
                    image,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout

        except subprocess.CalledProcessError as e:
            logger.error(f"Error running ODH tests: {e.stderr}")
            return f"QA_ERROR: {e.stderr}"

    prompt = (
        "You are a QA Specialist responsible for validating machine learning model deployments "
        "and configurations on OpenShift / Kubernetes. "
        "Use the provided tool to run Opendatahub model validation tests inside a container. "
        "When you call the tool, inspect the test output and summarize:\n"
        "- Whether tests passed or failed.\n"
        "- Any configuration issues detected.\n"
        "- Recommendations for fixing failures or misconfigurations.\n"
        "Provide clear, structured responses with validation results and recommendations."
    )
    
    agent = create_agent(
        llm,
        tools=[run_odh_tests],
        system_prompt=prompt,
    )

    @tool
    def analyze_qa_results(
        request: str,
    ) -> str:
        """Analyze the QA test output and provide a summary."""
        result = agent.invoke({"messages": [{"role": "user", "content": request}]})
        return extract_text(result)
    
    analyze_qa_results.name = "analyze_qa_results"


    return SpecialistSpec(
        name="qa_specialist",
        agent=agent,
        tool=analyze_qa_results,
    )

__all__ = ["build_qa_specialist"]