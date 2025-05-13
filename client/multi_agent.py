from typing import Literal, TypedDict
import asyncio
import logging

from configuration.configuration import Configuration
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

from supervisor import Supervisor
from tool_knowledge import LLMAgentClient
from tool_agent import ToolsAgentClient

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

checkpointer = InMemorySaver()
store = InMemoryStore()

class AgentState(TypedDict):
    input: str
    next: Literal["llm_agent", "tools_agent"]



class MultiAgentSystem:
    def __init__(self, llm_model: str, gemini_key: str, project: str, mcp_config: dict):
        self.llm_model = llm_model
        self.gemini_key = gemini_key
        self.mcp_config = mcp_config
        self.supervisor = None
        self.tool_agent = ToolsAgentClient(self.llm_model, self.mcp_config)
        self.llm_agent = LLMAgentClient(api_key=self.gemini_key)
        self.project = project

        self.workflow = None
        self.checkpointer = InMemorySaver()
        self.store = InMemoryStore()

    async def __initialize_tool_agent__(self):
        await self.tool_agent.setup(self.project)

    def __initialize_supervisor__(self):
        self.supervisor = Supervisor(self.gemini_key, self.llm_model)
        self.supervisor.setup([self.tool_agent, self.llm_agent])

    async def initialize_and_start(self):
        await self.__initialize_tool_agent__()
        self.__initialize_supervisor__()

    def research_node(state: State) -> Command[Literal["supervisor"]]:
        result = research_agent.invoke(state)
        return Command(
            update={
                "messages": [
                    HumanMessage(content=result["messages"][-1].content, name="researcher")
                ]
            },
            goto="supervisor",
        )


    # NOTE: THIS PERFORMS ARBITRARY CODE EXECUTION, WHICH CAN BE UNSAFE WHEN NOT SANDBOXED
    code_agent = create_react_agent(llm, tools=[python_repl_tool])


    def code_node(state: State) -> Command[Literal["supervisor"]]:
        result = code_agent.invoke(state)
        return Command(
            update={
                "messages": [
                    HumanMessage(content=result["messages"][-1].content, name="coder")
                ]
            },
            goto="supervisor",
        )


    def setup_workflow(self):
        graph = StateGraph(AgentState)
        graph.add_node("supervisor", self.supervisor)
        graph.add_node("tools_agent", self.tool_agent)
        graph.add_node("llm_agent", self.llm_agent)
        graph.set_entry_point("supervisor")
        graph.add_edge("tools_agent", END)
        graph.add_edge("llm_agent", END)

        self.workflow = graph.compile(
            checkpointer=self.checkpointer,
            store=self.store,
        )

    async def run(self, input: str):
        return await self.workflow.ainvoke({"input": input})
    
    async def cleanup(self):
        await self.tool_agent.close()
        await self.supervisor.close()


    async def start(self):
        try:
            await self.setup_workflow()
            print("Multi-Agent System initialized successfully.")
            print("You can start chatting with the system. Type 'exit' or 'quit' to end the session.")
            print("Welcome to the Multi-Agent System! Type 'exit' or 'quit' to end the session.")
            while True:
                user_input = input("You: ").strip().lower()
                if user_input in ["exit", "quit"]:
                    break
                initial_messages = [{"role": "user", "content": user_input}]
                output = await self.workflow.ainvoke({"messages": initial_messages, "agent": self.supervisor.agent})
                messages = output["messages"]
                print(f"Assistant: {messages[-1]['content']}")

        
        except asyncio.CancelledError:
            logging.warning("Session cancelled. Skipping async cleanup.")
        except Exception as e:
            logging.error(f"Unhandled error in session: {e}")
        finally:
            try:
                await self.cleanup()
            except Exception as e:
                logging.warning(f"Cleanup skipped due to async cancel conflict: {e}")


async def main() -> None:
    """Initialize and run the chat session."""
    config = Configuration()
    server_config = config.load_config("servers_config.json")["mcpServers"]
    multi_agent = MultiAgentSystem(config.llm_model, config.api_key, config.project, server_config)
    await multi_agent.initialize_and_start()


if __name__ == "__main__":
    asyncio.run(main())