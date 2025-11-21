"""Main entry point for exercising the supervisor agent with fallbacks."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Callable, Iterable
import argparse
from langgraph.errors import GraphRecursionError

from runtimes_dep_agent.agent.llm_agent import LLMAgent

# Add src to path for utils
src_path = Path(__file__).parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
from utils import check_cluster_login


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

    # Check cluster login 
    print("Checking cluster authentication...")
    login_status = check_cluster_login()
    
    if "failed" in login_status.lower() or "login" in login_status.lower():
        print(f" Error: {login_status}")
        print("Please login to your cluster first using: oc login")
        sys.exit(1)
    
    print(f" Successfully Logged : {login_status}")
    
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
