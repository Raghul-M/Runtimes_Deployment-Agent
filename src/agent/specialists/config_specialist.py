"""Configuration specialist agent."""

from __future__ import annotations

import json
from typing import Callable

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool

from src.agent.specialists import SpecialistSpec
from src.config.model_config import get_model_requirements, load_llm_model_config


def build_config_specialist(
    llm: BaseChatModel, extract_text: Callable[[dict], str]
) -> SpecialistSpec:
    """Return the configuration specialist agent and the supervisor-facing tool."""

    @tool
    def load_model_config(config_path: str) -> str:
        """Return the raw YAML contents as formatted JSON."""
        config = load_llm_model_config(config_path)
        return json.dumps(config, indent=2)

    @tool
    def extract_model_requirements(config_path: str) -> str:
        """Return derived model-car requirements as JSON."""
        config = load_llm_model_config(config_path)
        requirements = get_model_requirements(config)
        return json.dumps(requirements, indent=2)

    prompt = (
        "You are a model-car configuration specialist. "
        "Use the provided tools to inspect YAML files, extract key fields, "
        "and report concise JSON summaries. Prefer `extract_model_requirements` "
        "when the user asks for requirements."
    )

    agent = create_agent(
        llm,
        tools=[load_model_config, extract_model_requirements],
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
