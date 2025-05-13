from __future__ import annotations
import getpass
import logging
import os
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import List, Optional
from langchain_core.messages import (
    FunctionMessage,
    SystemMessage,
)
from langchain import hub
from langchain_core.runnables import RunnableBranch
from langchain_mcp_adapters.client import MultiServerMCPClient

from client.llm_compiler.output_parser import LLMCompilerPlanParser
from client.configuration.configuration import Configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()  # Load environment variables from .env file

LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
if LANGSMITH_API_KEY is None:
    LANGSMITH_API_KEY = getpass.getpass("Enter your LangSmith API key: ")
    os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY

os.environ["LANGSMITH_TRACING"] = "true"

base_prompt = hub.pull("wfh/llm-compiler")

@dataclass
class CompilerState:
    """State object that flows through the LangGraph nodes."""

    question: str
    plan: Optional[str] = None
    stack: List[str] = None 


class Planner:
    """LLM that produces a DAG‑style plan from the user question."""
    def __init__(self, llm, config: Configuration) -> None:
        self.config = config
        self.llm = llm
        self.planner = None
        self.replanner = None
        self.base_prompt = base_prompt
        self.tools = None
        self.model = None
        self.client = None
        

    async def init_client_and_tools(self):
        """Start client and fetch tools (keeping session open)."""
        if self.client is None:
            connections = self.config.load_config()["mcpServers"]
            self.client = MultiServerMCPClient(connections)
            await self.client.__aenter__()

        self.tools = self.client.get_tools()
    async def create_planner(self):
        logging.info("Getting tools from the config...")
        """Get the tools from the config."""
        await self.init_client_and_tools()
        logging.info(f"tools: {self.tools}")
        """Create the planner and replanner."""
        tool_descriptions = "\n".join(
            f"{i+1}. {tool.name}: {tool.description}"
            for i, tool in enumerate(self.tools)
        )
        planner_prompt = self.base_prompt.partial(
            replan="",
            num_tools=len(self.tools)
            + 1,  # Add one because we're adding the join() tool at the end.
            tool_descriptions=tool_descriptions,
        )
        replanner_prompt = self.base_prompt.partial(
            replan=' - You are given "Previous Plan" which is the plan that the previous agent created along with the execution results '
            "(given as Observation) of each plan and a general thought (given as Thought) about the executed results."
            'You MUST use these information to create the next plan under "Current Plan".\n'
            ' - When starting the Current Plan, you should start with "Thought" that outlines the strategy for the next plan.\n'
            " - In the Current Plan, you should NEVER repeat the actions that are already executed in the Previous Plan.\n"
            " - You must continue the task index from the end of the previous one. Do not repeat task indices.",
            num_tools=len(self.tools) + 1,
            tool_descriptions=tool_descriptions,
        )
        def should_replan(state: list):
            # Context is passed as a system message
            return isinstance(state[-1], SystemMessage)

        def wrap_messages(state: list):
            return {"messages": state}

        def wrap_and_get_last_index(state: list):
            next_task = 0
            for message in state[::-1]:
                if isinstance(message, FunctionMessage):
                    next_task = message.additional_kwargs["idx"] + 1
                    break
            state[-1].content = state[-1].content + f" - Begin counting at : {next_task}"
            return {"messages": state}
        self.model =  (
            RunnableBranch(
                (should_replan, wrap_and_get_last_index | replanner_prompt),
                wrap_messages | planner_prompt,
            )
            | self.llm
            | LLMCompilerPlanParser(tools=self.tools)
        )
        return self.model
