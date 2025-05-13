from __future__ import annotations
import getpass
import os
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import List, Optional, Sequence
from langchain_core.messages import (
    FunctionMessage,
    SystemMessage,
)
from langchain import hub
from langchain_core.runnables import RunnableBranch
from langchain_core.tools import BaseTool
from output_parser import LLMCompilerPlanParser
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_google_genai import ChatGoogleGenerativeAI

from client.configuration.configuration import Configuration


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
    def __init__(self, config: Configuration) -> None:
        self.config = config
        self.llm = ChatGoogleGenerativeAI(
            model=self.config.llm_model,
            google_api_key=self.config.api_key,
        )
        self.planner = None
        self.replanner = None
        self.base_prompt = base_prompt
        self.tools: Sequence[BaseTool]= None
        self.model = None


    async def __get_tools__(self):
        """Get the tools from the config."""
        connections = self.config.load_config()["mcpServers"]
        async with MultiServerMCPClient(connections) as client:
            tools = tuple(client.get_tools())
            self.tools = tools
    async def create_planner(self):
        await self.__get_tools__()
        """Create the planner and replanner."""
        tool_descriptions = "\n".join(
            f"{i+1}. {tool.description}\n"
            for i, tool in enumerate(
                self.tools
            )  # +1 to offset the 0 starting index, we want it count normally from 1.
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
