import asyncio
from client.assistant.agent import ChatSession, LLMClient
from configuration.configuration import Configuration
from client.agent.server import Server

async def main() -> None:
    """Initialize and run the chat session."""
    config = Configuration()
    server_config = config.load_config("servers_config.json")
    servers = [
        Server(name, srv_config)
        for name, srv_config in server_config["mcpServers"].items()
    ] 
    llm_client = LLMClient(api_key=config.api_key)
    chat_session = ChatSession(servers, llm_client)
    await chat_session.start()


if __name__ == "__main__":
    asyncio.run(main())