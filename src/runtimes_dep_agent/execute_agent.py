"""Main entry point for exercising the supervisor agent with fallbacks."""

from __future__ import annotations

import json
import os
import re
from typing import Callable, Iterable
import argparse
from langgraph.errors import GraphRecursionError

from runtimes_dep_agent.agent.llm_agent import LLMAgent


DEFAULT_CONFIG_PATH = "config-yaml/sample_modelcar_config.yaml"


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
        "--supported-accelerator-type",
        default="NVIDIA",
        help="Type of accelerator supported (e.g., NVIDIA, AMD).",
    )
    parser.add_argument(
        "--vllm-runtime-image",
        default="registry.redhat.io/rh-ai/llm-runtime-rhel9:vllm-0.5.3",
        help="Container image for the vLLM runtime.",
    )
    parser.add_argument(
        "--modelcar-image-name",
        default="quay.io/rh-ai/model-car-rhel9:vllm-0.5.3",
        help="Container image name for the model-car.",
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
    if handler == "default":
        scenario_requests = [
            (
                "Configuration",
                "Report the model-car requirements for the given configuration file "
            )
        ]
        run_requests(agent, scenario_requests)
    else:
        raise ValueError(f"Unknown handler specified: {handler}")

if __name__ == "__main__":
    main()
