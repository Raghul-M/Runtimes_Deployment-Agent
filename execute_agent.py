"""Main entry point for exercising the supervisor agent with fallbacks."""

from __future__ import annotations

import json
import os
import re
from typing import Callable, Iterable

from langgraph.errors import GraphRecursionError

from src.agent.llm_agent import LLMAgent
from src.config.model_config import (
    estimate_model_size,
    get_model_requirements,
    load_llm_model_config,
)


def _extract_config_path(prompt: str) -> str:
    match = re.search(r"'([^']+\.ya?ml)'", prompt)
    if match:
        return match.group(1)
    return "config-yaml/sample_modelcar_config.yaml"


def fallback_config(prompt: str) -> str:
    config_path = _extract_config_path(prompt)
    config = load_llm_model_config(config_path)
    requirements = get_model_requirements(config)
    return json.dumps(requirements, indent=2)


def run_requests(
    agent: LLMAgent,
    requests: Iterable[tuple[str, str, Callable[[str], str]]],
) -> None:
    for label, prompt, fallback in requests:
        print(f"\n{label}")
        print("-" * len(label))
        try:
            result = agent.run_supervisor(prompt)
            output_text = agent.extract_final_text(result)
        except GraphRecursionError:
            output_text = ""

        if not output_text.strip():
            output_text = fallback(prompt)

        print(output_text)


if __name__ == "__main__":
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable must be set")

    agent = LLMAgent(api_key=api_key)

    scenario_requests = [
        (
            "Configuration",
            "Load the model configuration from "
            "'config-yaml/sample_modelcar_config.yaml' and extract the model requirements.",
            fallback_config,
        )
    ]

    run_requests(agent, scenario_requests)
