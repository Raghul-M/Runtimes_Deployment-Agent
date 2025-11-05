"""Model size specialist agent."""

from __future__ import annotations

import json
from typing import Callable

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool

from src.agent.specialists import SpecialistSpec
from src.config.model_config import (
    estimate_model_size,
    get_model_requirements,
    load_llm_model_config,
)


def build_model_size_specialist(
    llm: BaseChatModel, extract_text: Callable[[dict], str]
) -> SpecialistSpec:
    """Return the model size specialist agent and supervisor-facing tool."""

    @tool
    def lookup_model_sizes(config_path: str) -> str:
        """Return model names with their image sizes in bytes."""
        config = load_llm_model_config(config_path)
        requirements = get_model_requirements(config)
        sizes = {}
        for name, info in requirements.items():
            image = info.get("image", "")
            sizes[name] = {
                "image": image,
                "model_size_bytes": estimate_model_size(image),
            }
        return json.dumps(sizes, indent=2)

    prompt = (
        "You are a model size specialist. Use the provided tool to calculate container "
        "image sizes for models described in a configuration file. Always respond with "
        "the tool output (JSON mapping model names to model_size_bytes)."
    )

    agent = create_agent(
        llm,
        tools=[lookup_model_sizes],
        system_prompt=prompt,
    )

    @tool
    def lookup_sizes(request: str) -> str:
        """Delegate model size lookups to the specialist."""
        result = agent.invoke(
            {"messages": [{"role": "user", "content": request}]},
            config={"recursion_limit": 4},
        )
        return extract_text(result)

    lookup_sizes.name = "lookup_model_sizes"

    return SpecialistSpec(
        name="model_size_specialist",
        agent=agent,
        tool=lookup_sizes,
    )


__all__ = ["build_model_size_specialist"]
