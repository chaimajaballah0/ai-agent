import logging
from typing import List, Union
import getpass
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain import hub
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
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

class FinalResponse(BaseModel):
    """The final response/answer."""

    response: str


class Replan(BaseModel):
    feedback: str = Field(
        description="Analysis of the previous attempts and recommendations on what needs to be fixed."
    )


class JoinOutputs(BaseModel):
    """Decide whether to replan or whether you can return the final response."""

    thought: str = Field(
        description="The chain of thought reasoning for the selected action"
    )
    action: Union[FinalResponse, Replan]
    
def _parse_joiner_output(decision: JoinOutputs) -> List[BaseMessage]:
    response = [AIMessage(content=f"Thought: {decision.thought}")]
    if isinstance(decision.action, Replan):
        return {
            "messages": response
            + [
                SystemMessage(
                    content=f"Context from last attempt: {decision.action.feedback}"
                )
            ]
        }
    else:
        return {"messages": response + [AIMessage(content=decision.action.response)]}


def select_recent_messages(state) -> dict:
    messages = state["messages"]
    selected = []
    for msg in messages[::-1]:
        selected.append(msg)
        if isinstance(msg, HumanMessage):
            break
    return {"messages": selected[::-1]}




class Joiner():
    def __init__(self, config: Configuration) -> None:
        self.config = config
        self.llm = ChatGoogleGenerativeAI(
            model=self.config.llm_model,
            google_api_key=self.config.api_key,
        )
        self.prompt = hub.pull("wfh/llm-compiler-joiner").partial(
    examples="")
        self.joiner = None
    def create_joiner(self):
        runnable = self.prompt | self.llm.with_structured_output(
            JoinOutputs, method="function_calling"
        )
        self.joiner = select_recent_messages | runnable | _parse_joiner_output
        return self.joiner


    def invoke(self, input, tool_messages):
        input_messages = [HumanMessage(content=input)] + tool_messages
        return self.joiner.invoke({"messages": input_messages})