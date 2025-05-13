import asyncio
import getpass
import os
import logging

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph, START
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from client.llm_compiler.planner import Planner
from client.configuration.configuration import Configuration
from client.llm_compiler.joiner import Joiner
from client.llm_compiler.classifier_graph import QueryClassificationGraph
from client.llm_compiler.state import State
from client.llm_compiler.plan_and_execute_graph import PlanAndExecuteGraph
from client.llm_compiler.postprocess_graph import PostprocessingGraph
from client.persistence.models.thread import UserThread



logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
if LANGSMITH_API_KEY is None:
    LANGSMITH_API_KEY = getpass.getpass("Enter your LangSmith API key: ")
    os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY

os.environ["LANGSMITH_TRACING"] = "true"

class LangGraphWorkflow:
    def __init__(self, config: Configuration, user_id: str = None, thread_id: str = None):
        self.user_id = user_id
        self.thread_id = thread_id
        self.config = config
        self.task_scheduler = None
        self.joiner = Joiner(config)
        self.graph_builder = None
        self.planner = None
        self.llm = ChatGoogleGenerativeAI(
            model=self.config.llm_model,
            google_api_key=self.config.api_key,
        )
        self.checkpointer = InMemorySaver()
        self.store = InMemoryStore()

    async def persist_session(self):
        await UserThread.save_or_update(
            user_id=self.user_id,
            thread_id=self.thread_id,
            store=self.store.to_json(),
            checkpoint=self.checkpointer.to_json()
        )
    async def __initialization__(self):
        logging.info("Initializing planner...")
        self.planner = Planner(self.llm, self.config)
        self.joiner.create_joiner()
        await self.planner.create_planner()

    @classmethod
    async def load_session(cls, config: Configuration, user_id: str, thread_id: str):
        session_row = await UserThread.load(user_id, thread_id)
        if session_row is None:
            raise ValueError(f"No session found for user {user_id} and thread {thread_id}")

        graph = cls(config, user_id, thread_id)
        graph.store.load_json(session_row.store)
        graph.checkpointer.load_json(session_row.checkpoint)
        await graph.__initialization__()
        return graph

    def build_classifier_subgraph(self):
        logging.info("Building classifier subgraph...")
        classifier = QueryClassificationGraph(self.llm)
        return classifier.build_subgraph()

    def build_plan_and_execute_subgraph(self):
        logging.info("Building plan and execute subgraph...")
        plan_and_execute = PlanAndExecuteGraph(self.llm, self.joiner, self.planner)
        return plan_and_execute.build_planning_subgraph()

    def build_postprocessing_subgraph(self):
        logging.info("Building postprocessing subgraph...")
        postprocessing = PostprocessingGraph(self.llm)
        return postprocessing.build_subgraph()

    
    async def build_graph(self):
        logging.info("Building LangGraph graph...")
        
        classifier_subgraph = await self.build_classifier_subgraph()
        plan_and_execute_subgraph = await self.build_plan_and_execute_subgraph()
        postprocessing_subgraph = await self.build_postprocessing_subgraph()
        self.graph_builder = StateGraph(State)

        self.graph_builder.add_node("simple_or_complex", classifier_subgraph)
        self.graph_builder.add_node("planning", plan_and_execute_subgraph)
        self.graph_builder.add_node("postprocess", postprocessing_subgraph)
        self.graph_builder.add_edge(START, "simple_or_complex")


        self.graph_builder.add_conditional_edges(
            "simple_or_complex",
            lambda state: END if isinstance(state["messages"][-1], AIMessage) else "planning"
        )
        self.graph_builder.add_conditional_edges("planning", lambda state: "postprocess")
        self.graph_builder.add_edge("postprocess", END)

        return self.graph_builder.compile(checkpointer=self.checkpointer, store=self.store)

    
async def main():
    mcp_config_path = "src/client/configuration/servers_config.json"
    config = Configuration(mcp_config_path)
    workflow = LangGraphWorkflow(config)
    graph = await workflow.build_graph()

    try:
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                break

            state = {"messages": [HumanMessage(content=user_input)]}
            result = await graph.ainvoke(state)
            print("Result:", result)


    finally:
        if workflow.planner and workflow.planner.client:
            await workflow.planner.client.__aexit__(None, None, None)

if __name__ == "__main__":
    import logging
    logging.info("Starting LangGraph workflow...")
    asyncio.run(main())
