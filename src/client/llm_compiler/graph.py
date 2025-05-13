import asyncio
import getpass
import os
import logging
import itertools
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langchain_core.runnables import chain as as_runnable
from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage

from client.llm_compiler.llm_compiler import Planner
from client.configuration.configuration import Configuration
from client.llm_compiler.executor import TaskScheduler, schedule_tasks
from client.llm_compiler.joiner import Joiner


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
load_dotenv()  # Load environment variables from .env file

LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
if LANGSMITH_API_KEY is None:
    LANGSMITH_API_KEY = getpass.getpass("Enter your LangSmith API key: ")
    os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY

os.environ["LANGSMITH_TRACING"] = "true"
class State(TypedDict):
    messages: Annotated[list, add_messages]


class LangGraphWorkflow:
    def __init__(self, config: Configuration):
        self.config = config
        self.task_scheduler = None
        self.joiner = Joiner(config)
        self.graph_builder = None
        self.planner = None
    async def __init_planner__(self):
        self.planner = Planner(self.config)
        planner = await self.planner.create_planner()
        self.task_scheduler = TaskScheduler(planner)
    def _should_continue(self, state: dict) -> str:
        messages = state["messages"]
        if isinstance(messages[-1], AIMessage):
            return END
        return "plan_and_schedule"

    async def build_graph(self):
        await self.__init_planner__()
        joiner = self.joiner.create_joiner()
        @as_runnable
        def plan_and_schedule(state):
            messages = state["messages"]
            tasks = self.planner.model.stream(messages)
            # Begin executing the planner immediately
            try:
                tasks = itertools.chain([next(tasks)], tasks)
            except StopIteration:
                # Handle the case where tasks is empty.
                tasks = iter([])
            scheduled_tasks = schedule_tasks.invoke(
                {
                    "messages": messages,
                    "tasks": tasks,
                }
            )
            return {"messages": scheduled_tasks}

        # Add nodes
        self.graph_builder = StateGraph(State)

        self.graph_builder.add_node("plan_and_schedule", plan_and_schedule)
        self.graph_builder.add_node("join", joiner)

        # Add edges
        self.graph_builder.add_edge("plan_and_schedule", "join")
        self.graph_builder.add_conditional_edges("join", self._should_continue)
        self.graph_builder.add_edge(START, "plan_and_schedule")

        # Compile graph
        return self.graph_builder.compile()
        # return self.graph_builder.compile(checkpointer=self.checkpointer, store=self.store)


async def main():
    mcp_config_path = "src/client/configuration/servers_config.json"
    config = Configuration(mcp_config_path)
    workflow = LangGraphWorkflow(config)
    
    graph = await workflow.build_graph()
    while True:
        user_input = input("You: ").strip().lower()
        if user_input in ["exit", "quit"]:
            break
        for step in graph.stream(
        {"messages": [HumanMessage(content=user_input)]}
        ):
            print(step)
            print("---")
    
    # Final answer
    print(step["join"]["messages"][-1].content)




if __name__ == "__main__":
    logging.info("Starting LangGraph workflow...")
    logging.info("Workflow initialized.")
    asyncio.run(main())