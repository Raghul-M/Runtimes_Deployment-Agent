"""Main entry point for the Accelerator Compatibility Agent."""

from src.agent.llm_agent import LLMAgent
from langchain_core.tools import tool
from typing import Any, Dict, List
import os


if __name__ == "__main__":
    # Get API key from environment variable
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable must be set")

    # Initialize the agent
    agent = LLMAgent(api_key=api_key)

    # Register agent nodes
    agent.register_agent_node("calculate_sum", lambda x, y: x + y)
    agent.register_agent_node("multiply", lambda x, y: x * y)

    # Register tools
    @tool
    def add_numbers(x: int, y: int) -> int:
        """Add two numbers together by routing to the calculate_sum agent."""
        return agent.route_command("calculate_sum", x, y)
    
    @tool
    def multiply_numbers(x: int, y: int) -> int:
        """Multiply two numbers together by routing to the multiply agent."""
        return agent.route_command("multiply", x, y)

    agent.register_tool(add_numbers)
    agent.register_tool(multiply_numbers)

    # Direct routing (bypassing supervisor)
    sum_result = agent.route_command("calculate_sum", 5, 10)
    multiply_result = agent.route_command("multiply", 5, 10)
    print(f"Direct routing results: Sum = {sum_result}, Product = {multiply_result}")

    # Using the supervisor
    result = agent.run_supervisor("I need to add 7 and 3, and also multiply 4 and 5")
    print("\nSupervisor result:", result["output"])

    # Execute parallel calls
    parallel_results = agent.execute_parallel([
        {"agent_name": "calculate_sum", "args": [10, 20]},
        {"agent_name": "multiply", "args": [10, 20]}
    ])
    print("\nParallel execution results:")
    for agent_name, result in parallel_results:
        print(f"{agent_name}: {result}")