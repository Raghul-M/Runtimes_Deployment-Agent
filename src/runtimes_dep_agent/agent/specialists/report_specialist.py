"""Report Specialist for generating deployment reports."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Callable
import logging

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool
from bs4 import BeautifulSoup

from . import SpecialistSpec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_report_specialist(
    llm: BaseChatModel,
    extract_text: Callable[[dict], str],
    precomputed_requirements: dict | None = None,
) -> SpecialistSpec:
    """Create the report specialist that generates deployment reports."""

    @tool
    def convert_text_to_html_report(summary: str) -> str:
        """Convert a text summary into a simple HTML report format."""
        html_content = f"""
        <html>
        <head><title>Deployment Validation Report</title></head>
        <body>
        <h1>Deployment Validation Report</h1>
        <pre>{summary}</pre>
        </body>
        </html>
        """
        report_path = Path(__file__).parent / "../../../../reports/deployment_validation_report.html"
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return str(report_path)

    
    prompt = (
        "You are a Report Specialist responsible for generating deployment validation reports.\n\n"
        "You have access to a tool called `convert_text_to_html_report` that takes a text summary "
        "and converts it into a simple HTML report format, saving it to a file and returning the file path.\n\n"
        "When given a request to generate a deployment report, you must produce a detailed summary "
        "of the deployment validation findings, including model requirements, GPU availability, "
        "QA test results, and any decisions made by the Decision Specialist.\n\n"
        "You MUST use the `convert_text_to_html_report` tool to create the final report in HTML format.\n\n"
        "Always respond with the path to the generated HTML report file."
    )

    agent = create_agent(
        llm,
        tools=[convert_text_to_html_report],
        system_prompt=prompt,
    )

    @tool
    def generate_deployment_report(request: str, summary: str) -> str:
        """
        Supervisor-facing entrypoint. The supervisor must pass:
            - request: what to do (e.g. "generate deployment report")
        """
        report_input = (
            f"{request}\n"
            f"Here is the deployment validation summary:\n{summary}\n\n"
            "You MUST call convert_text_to_html_report to generate the HTML report and return the file path."
        )

        result = agent.invoke({"messages": [{"role": "user", "content": report_input}]})
        return extract_text(result)
    
    generate_deployment_report.name = "generate_deployment_report"

    return SpecialistSpec(
        name="report_specialist",
        agent=agent,
        tool=generate_deployment_report,
    )
__all__ = ["build_report_specialist"]
