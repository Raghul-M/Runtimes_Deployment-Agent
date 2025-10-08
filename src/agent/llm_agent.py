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
    
    def register_agent_node(self, name: str, agent_func: Callable):
        """
        Register an agent node with a specifi name.

        Args:
            name (str): The name of the agent node.
            agent_func (Callable): The function that implements the agent node.
        """
        self.agent_nodes[name] = agent_func

    def register_tool(self, tool_func: Any):
        """
        Register a tool for the LLM to use.

        Args:
            tool_func (Any): The tool function to register.
        """
        self.tools.append(tool_func)

    def route_command(self, agent_name: str, *args, **kwargs):
        """
        Route a command to the specified agent node.

        Args:
            agent_name (str): The name of the agent node to route the command to.
            *args: Positional arguments to pass to the agent function.
            **kwargs: Keyword arguments to pass to the agent function.

        Returns:
            The result of the agent function call.
        """
        if agent_name not in self.agent_nodes:
            raise ValueError(f"Agent '{agent_name}' not registered.")
        
        agent_func = self.agent_nodes[agent_name]
        return agent_func(*args, **kwargs)