"""Main entry point for exercising the supervisor agent with fallbacks."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Callable, Iterable
import argparse
from langgraph.errors import GraphRecursionError

from runtimes_dep_agent.agent.llm_agent import LLMAgent


DEFAULT_CONFIG_PATH = "config-yaml/sample_modelcar_config.yaml"

SCENARIO_REQUESTS: dict[str, list[tuple[str, str]]] = {
    "default": [
        (
            "Configuration",
            "Summarize the model-car configuration and extract the model requirements."
        ),
        (
            "Accelerator Compatibility",
            "Validate accelerator availability and hardware compatibility for this model-car configuration."
        ),
        (
            "Deployment Decision",
            "Provide a GO/NO-GO deployment recommendation based on the configuration and the cluster accelerators. "
            "If GO, run full QA validation before concluding."
        ),
    ],

    "configuration": [
        (
            "Configuration",
            "Summarize the model-car configuration and extract model requirements."
        ),
    ],

    "accelerator": [
        (
            "Accelerator Authentication",
            "Check whether I am authenticated to the OpenShift cluster."
        ),
        (
            "Accelerator Validation",
            "Describe GPU availability, provider details, and accelerator compatibility."
        ),
    ],
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supervisor agent for model-car configuration analysis."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to the model-car YAML configuration file.",
    )
    parser.add_argument(
        "--handler",
        default="default",
        choices=sorted(SCENARIO_REQUESTS.keys()),
        help="Select which scenario handler to execute.",
    )
    return parser.parse_args()


def run_requests(
    agent: LLMAgent,
    requests: Iterable[tuple],
) -> None: 
    """Run a series of requests through the agent with fallback handling.
    param: agent: The LLMAgent instance to use.
    param: requests: An iterable of tuples containing (label, prompt).
    """
    for label, prompt in requests:
        print(f"\n{label}")
        print("-" * len(label))
        try:
            result = agent.run_supervisor(prompt)
            output_text = agent.extract_final_text(result)
        except GraphRecursionError:
            output_text = ""

        print(output_text)

def main() -> None:
    args = _parse_args()
    
    # Check API key 
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable must be set")
    
    # Initialize agent and proceed
    agent = LLMAgent(
        api_key=api_key,
        bootstrap_config=args.config
    )
    handler = getattr(args, "handler", "default")
    scenario_requests = SCENARIO_REQUESTS.get(handler)
    if scenario_requests is None:
        available = ", ".join(sorted(SCENARIO_REQUESTS))
        raise ValueError(
            f"Unknown handler specified: {handler}. Available handlers: {available}"
        )

    run_requests(agent, scenario_requests)

if __name__ == "__main__":
    main()
