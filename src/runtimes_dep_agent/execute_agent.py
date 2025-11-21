"""Main entry point for exercising the supervisor agent with fallbacks."""

from __future__ import annotations

import os
from typing import Iterable
import argparse
from langgraph.errors import GraphRecursionError

from runtimes_dep_agent.agent.llm_agent import LLMAgent


DEFAULT_CONFIG_PATH = "config-yaml/sample_modelcar_config.yaml"

SCENARIO_REQUESTS: dict[str, list[tuple[str, str]]] = {
    "default": [
        (
            "Configuration",
            "Load the provided model-car configuration and summarise the model requirements as structured text.",
        ),
        (
            "Accelerator Compatibility",
            "Using the configuration context you just produced as guidance, validate accelerator availability and compatibility on the cluster. "
            "Call the accelerator specialist to confirm login status, GPU provider, and any gaps versus the model requirements.",
        ),
    ],
    "configuration": [
        (
            "Configuration",
            "Report the model-car requirements for the given configuration file.",
        ),
    ],
    "accelerator": [
        (
            "Accelerator Authentication",
            "Confirm whether I am authenticated to the OpenShift cluster.",
        ),
        (
            "Accelerator Validation",
            "Check cluster GPU availability, provider details, and accelerator compatibility recommendations.",
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

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable must be set")
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
