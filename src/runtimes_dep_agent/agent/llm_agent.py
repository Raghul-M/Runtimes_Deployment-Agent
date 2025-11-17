"""Supervisor orchestration that wires specialist agents together."""

from __future__ import annotations

from typing import Any, Dict, List

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from .specialists import SpecialistSpec
from .specialists.config_specialist import build_config_specialist
from .specialists.accelerator_specialist import build_accelerator_specialist
from ..config.model_config import load_llm_model_config, get_model_requirements






class LLMAgent:
    """Builds a collection of specialists and exposes a supervisor entry point."""

    def __init__(self, 
                 api_key: str, 
                 model: str = "gemini-2.5-pro",
                 bootstrap_config: str | None = None
                 ) -> None:
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            api_key=api_key,
            temperature=0,
        )

        self.precomputed_requirements = None
        if bootstrap_config:
            config = load_llm_model_config(bootstrap_config)
            self.precomputed_requirements = get_model_requirements(config)

        self.specialists: List[SpecialistSpec] = self._initialise_specialists()
        self._supervisor = self._create_supervisor()

    # ------------------------------------------------------------------ #
    # Supervisor operations
    # ------------------------------------------------------------------ #
    def run_supervisor(
        self, user_input: str, recursion_limit: int = 15
    ) -> Dict[str, Any]:
        """Invoke the top-level supervisor on a natural language request."""
        return self._supervisor.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config={"recursion_limit": recursion_limit},
        )

    def extract_final_text(self, result: Dict[str, Any]) -> str:
        """Extract the supervisor's final textual response."""
        return self._extract_final_text(result)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _initialise_specialists(self) -> List[SpecialistSpec]:
        builders = [
            build_config_specialist,
            build_accelerator_specialist,
        ]
        return [
            builder(
                self.llm,
                self._extract_final_text,
                self.precomputed_requirements
            )
            for builder in builders
        ]

    def _create_supervisor(self):
        tools = [spec.tool for spec in self.specialists]
        prompt = (
            "You are an orchestration supervisor. Decide which specialist tool to call "
            "for each user request, execute it, and synthesise the response. "
            "Use the configuration specialist for YAML/model questions and image size information. "
            "Use the accelerator specialist for GPU availability, accelerator compatibility, "
            "and cluster validation questions. Always return the tool output verbatim as the final "
            "assistant message; do not rewrite, summarise, or add commentary."
        )
        return create_agent(
            self.llm,
            tools=tools,
            system_prompt=prompt,
        )

    @staticmethod
    def _extract_final_text(result: Dict[str, Any]) -> str:
        messages: List[Any] = result.get("messages", [])
        if not messages:
            return ""

        final_message = messages[-1]
        content = getattr(final_message, "content", final_message)

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "\n".join(part for part in parts if part).strip()

        return str(content)
