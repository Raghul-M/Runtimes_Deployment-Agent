"""Core LLM Agent class."""


from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Callable, List, Any


class LLMAgent:
    """
    LLM Agent that connects Gemini Pro with all the tools.
    In this architecture, we define agents as nodes and add a
    supervisor node (LLM) that decides which agent nodes should be called next.
    """

    def __init__(self, api_key: str, model: str = "gemini-2.5-pro"):
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
    
    def execute_parallel(self, agent_calls: List[Dict[str, Any]]):
        """
        Execute multiple agent calls in parallel.

        Args:
            agent_calls (List[Dict[str, Any]]): A list of dictionaries, each containing:
                - 'agent_name': The name of the agent node to call.
                - 'args': Positional arguments for the agent function.
                - 'kwargs': Keyword arguments for the agent function.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = []
        with ThreadPoolExecutor() as executor:
            future_to_call = {
                executor.submit(
                    self.route_command,
                    call['agent_name'],
                    *(call.get('args', [])),
                    **(call.get('kwargs', {}))
                ): call for call in agent_calls
            }
            for future in as_completed(future_to_call):
                call = future_to_call[future]
                try:
                    result = future.result()
                    results.append((call['agent_name'], result))
                except Exception as exc:
                    results.append((call['agent_name'], f"Generated an exception: {exc}"))
        return results
    
    def create_supervisor(self):
        """
        Create and return the supervisor agent with registered tools.

        Returns:
            AgentExecutor: The configured supervisor agent.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a supervisor agent that decides which tool to use based on user input.
            You have access to the following tools:
            {tools}
            """),
            ("user", "{user_input}"),
            ("ai", "{agent_scratchpad}")
        ])
        
        tools = [tool.name for tool in self.tools]
        agent = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt.partial(tools=", ".join(tools))
        )
        return AgentExecutor(agent=agent, tools=self.tools)
    
    def run_supervisor(self, user_input: str):
        """
        Run the supervisor agent with the given user input.

        Args:
            user_input (str): The input string from the user.

        Returns:
            The result of the supervisor agent execution.
        """
        supervisor = self.create_supervisor()
        return supervisor.invoke({"user_input": user_input})