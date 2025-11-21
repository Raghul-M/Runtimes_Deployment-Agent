"""Supervisor orchestration that wires specialist agents together."""

from __future__ import annotations

from typing import Any, Dict, List
import logging
import subprocess

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
            "You are an orchestration supervisor coordinating multiple specialist agents:\n\n"
            "- Configuration Specialist: Parses and reasons about YAML/model-car configs, model sizes, "
            "  quantization, and VRAM requirements.\n"
            "- Accelerator Specialist: Retrieves cluster/accelerator information, GPU/Spyre profiles, "
            "  hardware specs, and runtime compatibility.\n"
            "- QA Specialist: Runs the Opendatahub model validation test suite inside a container "
            "  (via podman) and returns the logs and results.\n"
            "- Decision Specialist: Compares the config and accelerator outputs and issues a final "
            "  GO/NO-GO decision.\n\n"

            "### Your Orchestration Responsibilities\n"
            "1. Decide which specialist tool to call for each request.\n"
            "2. When a user asks about deployment feasibility:\n"
            "   - Call the **Configuration Specialist** first to extract model requirements.\n"
            "   - Then call the **Accelerator Specialist** to gather GPU hardware specs.\n"
            "   - After gathering these inputs, call the **Decision Specialist**.\n\n"

            "### IMPORTANT RULE: Automatic QA on GO Decisions\n"
            "• If the **Decision Specialist returns a GO verdict**, you MUST:\n"
            "  → Automatically call the **QA Specialist** to run the full ODH validation tests.\n"
            "  → Summarize the QA test results in the final answer.\n"
            "  → Only after QA completes should you deliver the final conclusion.\n\n"
            "• If the Decision Specialist returns **NO-GO**, you MUST NOT call QA.\n"
            "  Instead, explain why the deployment is not feasible and provide recommendations.\n\n"

            "### Formatting Requirements\n"
            "- Explicitly compare model VRAM requirements to GPU VRAM "
            "  (e.g. `Model needs 18 GB < GPU provides 80 GB`).\n"
            "- State how many GPUs are needed per model.\n"
            "- Reference which specialists you consulted in your reasoning.\n"
            "- In GO cases:\n"
            "    *Include a summary of the QA test output.*\n"
            "- In NO-GO cases:\n"
            "    *Give clear remediation steps, but do not run QA.*\n\n"

            "You must always reason step-by-step, selecting the correct specialist tool calls "
            "in the required order."
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
