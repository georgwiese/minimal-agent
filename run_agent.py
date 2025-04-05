import os

from dotenv import load_dotenv

from minimal_agent.agent import Agent
from minimal_agent.tools import DuckDuckGoSearchTool, VisitWebpageTool

load_dotenv()


if __name__ == "__main__":
    agent = Agent(
        model=os.environ.get("MODEL"),
        tools=[
            DuckDuckGoSearchTool(max_results=10),
            VisitWebpageTool(max_output_length=1000),
        ],
    )

    res = agent.run(
        "What was the hottest day in 2024 and how much was the Dow Jones on that day?"
    )

    print(20 * "-")
    print(f"The final answer is:\n\n{res}")
