"""Core LLM Agent class."""


from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Dict, Callable, List, Any


class LLMAgent:
    """
    LLM Agent that connects Gemini Pro with all the tools.
    In this architecture, we define agents as nodes and add a
    supervisor node (LLM) that decides which agent nodes should be called next.
    """

    def __init__(self, api_key: str, model: str = "gemini-pro"):
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            api_key=api_key,
            temperature=0,
        )
        # A dictionary of agent nodes
        self.agent_nodes: Dict[str, Callable] = {}

        # Tools available to the supervisor LLM
        self.tools: List[Any] = []