import logging
import asyncio
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_google_genai import ChatGoogleGenerativeAI


from client.configuration.configuration import Configuration


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

class LLMClient:
    """Manages communication with Google Gemini"""

    def __init__(self, config: Configuration) -> None:
        """
        Initialize the LLMClient with the given api_key, model, and servers.

        Args:
        api_key (str): The apAi key for the Google Gemini model.
        model (str): The name of the Google Gemini model.
        servers (list[Server]): The list of servers to use for the agent.
        """
        self.name = "AIAssistant"
        self.config = config
        self.llm = ChatGoogleGenerativeAI(
            model=self.config.llm_model,
            google_api_key=self.config.api_key,
        )
        self.agent = None   
    async def mcp_call(self, input: str) -> None:
        connections = self.config.load_config()["mcpServers"]
        logging.info(f"Connections: {connections}")
        async with MultiServerMCPClient(connections) as client:
            tools = client.get_tools()
            logging.info(f"Tools: {tools}")
            prompt =  """You are a helpful assistant who answers questions with access to these tools:\n
            {tools}\n
            Please use only the tools that are explicitly above. \n
            Here is more details on the tools you can use:\n
            1. Email Tools:\n
            - Test tool (test_tool) - No arguments needed\n
            - Send emails (send_email) - Arguments: recipient_id, subject, message\n
            - Read unread emails (get_unread_emails)\n
            - Read specific email content (read_email)\n
            - Trash emails (trash_email)\n
            - Open emails in browser (open_email)\n
            2. Web Search Tools:\n
            - Search the web (search_web)\n
            - Browse website content (website_browsing)\n
            Guidelines:\n
            - For email operations:\n
            * Never send an email or trash an email without user confirmation\n
            * Always ask for approval if not already given\n
            * Be careful with sensitive information\n
            * For send_email, only use recipient_id, subject, and message arguments\n
            - For web operations:\n
            * Use search_web to find information\n
            * Use website_browsing to read specific web pages\n
            * Be mindful of website content and respect privacy\n
            """.format(tools=tools)
        
            self.agent = create_react_agent(model=self.llm, tools=tools, prompt=prompt, name=self.name)
            return await self.agent.ainvoke({"messages": [("user", input)]})
            
async def call_mcp_test():
    """Run with multiple MCP servers using MultiServerMCPClient."""
    try:
        config = Configuration("./src/client/configuration/servers_config.json")
        llm = LLMClient(config)
        connections = config.load_config()["mcpServers"]
        logging.info(f"Connections: {connections}")
        async with MultiServerMCPClient(connections) as client:
            tools = client.get_tools()
            logging.info(f"Tools: {tools}")
            prompt =  """You are a helpful assistant who answers questions with access to these tools:\n
            {tools}\n
            Please use only the tools that are explicitly above. \n
            Here is more details on the tools you can use:\n
            1. Email Tools:\n
            - Test tool (test_tool) - No arguments needed\n
            - Send emails (send_email) - Arguments: recipient_id, subject, message\n
            - Read unread emails (get_unread_emails)\n
            - Read specific email content (read_email)\n
            - Trash emails (trash_email)\n
            - Open emails in browser (open_email)\n
            2. Web Search Tools:\n
            - Search the web (search_web)\n
            - Browse website content (website_browsing)\n
            Guidelines:\n
            - For email operations:\n
            * Never send an email or trash an email without user confirmation\n
            * Always ask for approval if not already given\n
            * Be careful with sensitive information\n
            * For send_email, only use recipient_id, subject, and message arguments\n
            - For web operations:\n
            * Use search_web to find information\n
            * Use website_browsing to read specific web pages\n
            * Be mindful of website content and respect privacy\n
            """.format(tools=tools)
            agent = create_react_agent(model=llm.llm, tools=tools, prompt=prompt, name=llm.name)
            while True:
                user_input = input("You: ").strip().lower()
                if user_input in ["exit", "quit"]:
                    break
                prompt_input = {"messages": user_input}
                response = await agent.ainvoke(prompt_input)
                logging.info(f"Response: {response}")
                print(f"Assistant: {response['messages'][-1].content}")
    except (KeyboardInterrupt, EOFError, SystemExit, asyncio.CancelledError):
        print("\nExiting...")
    
    except Exception as e:
        logging.error(f"Unhandled exception: {e}")
        raise
if __name__ == "__main__":
    asyncio.run(call_mcp_test())

