"""Configuration specialist agent."""

from __future__ import annotations

import json
from typing import Callable

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool

from . import SpecialistSpec
from ...config.model_config import get_model_requirements, load_llm_model_config



def build_config_specialist(
    llm: BaseChatModel,
    extract_text: Callable[[dict], str],
    precomputed_requirements: dict | None = None,
) -> SpecialistSpec:
    """Return the configuration specialist agent and the supervisor-facing tool."""

    @tool
    def describe_preloaded_requirements() -> str:
        """Return the preloaded model requirements as JSON."""
        return json.dumps(precomputed_requirements, indent=2)

    prompt = (
        "You are a model-car configuration specialist. "
        "Use the provided tools to print out model requirements provided to you. "
        "Always respond with the tool output (JSON)."
    )

    agent = create_agent(
        llm,
        tools=[describe_preloaded_requirements],
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
