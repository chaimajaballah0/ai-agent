import asyncio
from contextlib import AsyncExitStack
from langgraph_supervisor import create_supervisor
from langchain_google_genai import ChatGoogleGenerativeAI


class Supervisor():
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self.agent = None
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
        )
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()
    
    def setup(self, agents: list):
        self.agent = create_supervisor(
            agents=agents,
            model=self.llm,
            prompt=(
                "You are a team supervisor managing a tools agent and a knowledge agent. "
                "For any information and common questions, use knowledge_agent. "
                "For any tools like managing emails, searching the web, use tool_agent."
            )
        )
    
    async def close(self):
        await self.agent.aclose()

    async def __aenter__(self):
        return self
