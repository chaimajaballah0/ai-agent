import logging
from typing import Any
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent




logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

class ToolsAgentClient:
    def __init__(self, llm_model: str, config: dict[str, Any]):
        self.client = None
        self.agent = None
        self.agent_executor = None
        self.config: dict[str, Any] = config
        self.llm_model = llm_model
        self.name = "ToolsAgent"  

    async def setup(self, project: str):
        from google.cloud import aiplatform
        aiplatform.init(project=project)
        self.client = MultiServerMCPClient(self.config)
        await self.client.__aenter__()
        tools = self.client.get_tools()
        prompt = (
            "You are a helpful assistant with access to these tools:\n\n{tools}\n\nPlease use only the tools that are explicitly defined above."
            "1. Email Tools:\n"
        "   - Test tool (test_tool) - No arguments needed\n"
        "   - Send emails (send_email) - Arguments: recipient_id, subject, message\n"
        "   - Read unread emails (get_unread_emails)\n"
        "   - Read specific email content (read_email)\n"
        "   - Trash emails (trash_email)\n"
        "   - Open emails in browser (open_email)\n"
        "2. Web Search Tools:\n"
        "   - Search the web (search_web)\n"
        "   - Browse website content (website_browsing)\n"
        "Guidelines:\n"
        "- For email operations:\n"
        "  * Never send an email or trash an email without user confirmation\n"
        "  * Always ask for approval if not already given\n"
        "  * Be careful with sensitive information\n"
        "  * For send_email, only use recipient_id, subject, and message arguments\n"
        "- For web operations:\n"
        "  * Use search_web to find information\n"
        "  * Use website_browsing to read specific web pages\n"
        "  * Be mindful of website content and respect privacy\n"
        ).format(tools=tools)

        self.agent = create_react_agent(model=self.llm_model, tools=tools, prompt=prompt, name=self.name)
    async def close(self):
        if self.client:
            await self.client.__aexit__(None, None, None)

    async def invoke(self, prompt):
        return await self.agent.invoke({"input": {prompt}})
    
