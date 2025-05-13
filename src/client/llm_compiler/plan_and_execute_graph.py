import itertools
import os
import getpass
import logging

from dotenv import load_dotenv
from langchain_core.runnables import chain as as_runnable
from langchain_core.messages import AIMessage
from langgraph.graph import START, END, StateGraph

from client.llm_compiler.planner import Planner
from client.llm_compiler.executor import schedule_tasks
from client.llm_compiler.joiner import Joiner
from client.llm_compiler.state import State



logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

load_dotenv()
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
if LANGSMITH_API_KEY is None:
    LANGSMITH_API_KEY = getpass.getpass("Enter your LangSmith API key: ")
    os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY

os.environ["LANGSMITH_TRACING"] = "true"

class PlanAndExecuteGraph:
    def __init__(self, llm, joiner: Joiner, planner: Planner):
        self.llm = llm
        self.joiner = joiner
        self.planner = planner
        self.task_scheduler = None

    async def build_planning_subgraph(self):
        logging.info("Building planning subgraph...")

        @as_runnable
        async def plan_and_schedule(state):
            logging.info("Planning and scheduling...")
            messages = state["messages"]
            tools = await self.planner.model.ainvoke(messages)
            tasks = iter(tools)
            # Begin executing the planner immediately
            try:
                tasks = itertools.chain([next(tasks)], tasks)
            except StopIteration:
                # Handle the case where tasks is empty.
                tasks = iter([])
            scheduled_tasks = await schedule_tasks.ainvoke(
                {
                    "messages": messages,
                    "tasks": tasks,
                }
            )
            logging.info(f"Scheduled tasks: {scheduled_tasks}")
            return {"messages": scheduled_tasks}


        def should_continue(state):
            logging.info("Checking if we should continue...")
            messages = state["messages"]
            logging.info(f"Messages: {messages}")
            if isinstance(messages[-1], AIMessage):
                return END
            return "plan_and_schedule"

        @as_runnable
        async def join(state):
            logging.info("Joining and summarizing...")
            joined = await self.joiner.joiner.ainvoke(state)
            logging.info(f"Joined: {joined}")
            return {
                "messages": joined["messages"]
            }
            
        logging.info("Creating planning graph...")
        planning_graph = StateGraph(State)

        planning_graph.add_node("plan_and_schedule", plan_and_schedule)
        planning_graph.add_node("join", join)

        planning_graph.add_edge("plan_and_schedule", "join")
        planning_graph.add_conditional_edges("join", should_continue)
        planning_graph.add_edge(START, "plan_and_schedule")

        return planning_graph.compile({
        "recursion_limit": 100,
    })
