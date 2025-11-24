"""Supervisor orchestration that wires specialist agents together."""

from __future__ import annotations

from typing import Any, Dict, List
import logging

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from .specialists import SpecialistSpec
from .specialists.config_specialist import build_config_specialist
from .specialists.qa_specialist import build_qa_specialist
from .specialists.accelerator_specialist import build_accelerator_specialist
from .specialists.decision_specialist import build_decision_specialist
from ..config.model_config import load_llm_model_config, get_model_requirements



logger = logging.getLogger(__name__)






class LLMAgent:
    """Builds a collection of specialists and exposes a supervisor entry point."""

    def __init__(self, 
                 api_key: str, 
                 model: str = "gemini-2.5-pro",
                 bootstrap_config: str | None = None,
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
            build_decision_specialist,
            build_qa_specialist,
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
            "You are a supervisor agent that coordinates several specialist tools:\n"
            "- Configuration Specialist: preloaded model-car requirements, YAML-derived details, VRAM estimates.\n"
            "- Accelerator Specialist: cluster accelerators, GPU/Spyre profiles, hardware details.\n"
            "- Decision Specialist: GO/NO-GO deployment decisions based on config + accelerators.\n"
            "- QA Specialist: runs the Opendatahub model validation test suite and reports results.\n\n"

            "A model-car configuration has already been processed by the host program. "
            "You can access its details only via your tools; never ask the user for YAML or file paths.\n\n"

            "Dynamic reasoning:\n"
            "- Read the user's request and decide which specialist tool(s) to call.\n"
            "- For generic triggers such as 'Start supervisor agent', you MUST perform a full deployment assessment:\n"
            "  1) Use the Configuration Specialist to summarise preloaded model requirements.\n"
            "  2) Use the Accelerator Specialist to inspect accelerators.\n"
            "  3) Use the Decision Specialist to decide GO or NO-GO.\n"
            "  4) If the user expects QA or you are issuing a deployment verdict, you MAY call the QA Specialist to\n"
            "     run validation tests and include the results.\n\n"

            "Output format:\n"
            "- Always respond with a single structured report with the following sections:\n"
            "  ### Configuration Summary\n"
            "  ### Accelerator Summary\n"
            "  ### Deployment Decision\n"
            "  ### QA Validation (even if you only say it was not run)\n"
            "- In each section, clearly state which facts came from which type of specialist.\n"
            "- Do not introduce yourself or explain that you are a supervisor.\n"
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
            for key in ("output", "output_text", "output_str"):
                text = result.get(key)
                if isinstance(text, str) and text.strip():
                    return text.strip()
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
