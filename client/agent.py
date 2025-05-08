import asyncio
import httpx
import logging
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from server import Server
# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class LLMClient:
    """Manages communication with Google Gemini"""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
    ) -> None:
        # ChatGoogleGenerativeAI will read GOOGLE_API_KEY env var or use google_api_key kwarg :contentReference[oaicite:0]{index=0}
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
        )

    def get_response(self, messages: list[dict[str, str]]) -> str:
        """Get a chat completion from Gemini via LangChain.

        Args:
            messages: A list of {"role": "system"|"user"|"assistant", "content": "..."} dicts.

        Returns:
            The assistant's reply as a string.
        """
        # Convert your dicts into LangChain message objects
        lc_msgs = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                lc_msgs.append(SystemMessage(content=content))
            elif role == "user":
                lc_msgs.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_msgs.append(AIMessage(content=content))
            else:
                raise ValueError(f"Unknown role: {role!r}")

        try:
            # invoke() returns an AIMessage
            ai_msg = self.llm.invoke(lc_msgs)
            return ai_msg.content
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logging.error(f"Error getting LLM response: {e}")
            if isinstance(e, httpx.HTTPStatusError):
                logging.error(f"Status code: {e.response.status_code}")
                logging.error(f"Response body: {e.response.text}")
            return "I encountered an error communicating with the model. Please try again."

class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    def __init__(self, servers: list[Server], llm_client: LLMClient) -> None:
        self.servers: list[Server] = servers
        self.llm_client: LLMClient = llm_client

    async def cleanup_servers(self) -> None:
        """Clean up all servers properly."""
        cleanup_tasks = [
            asyncio.create_task(server.cleanup()) for server in self.servers
        ]
        if cleanup_tasks:
            try:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            except Exception as e:
                logging.warning(f"Warning during final cleanup: {e}")

    async def process_llm_response(self, llm_response: str) -> str:
        """Process the LLM response and execute tools if needed.

        Args:
            llm_response: The response from the LLM.

        Returns:
            The result of tool execution or the original response.
        """
        import json

        try:
            tool_call = json.loads(llm_response)
            if "tool" in tool_call and "arguments" in tool_call:
                logging.info(f"Executing tool: {tool_call['tool']}")
                logging.info(f"With arguments: {tool_call['arguments']}")

                for server in self.servers:
                    tools = await server.list_tools()
                    if any(tool.name == tool_call["tool"] for tool in tools):
                        try:
                            result = await server.execute_tool(
                                tool_call["tool"], tool_call["arguments"]
                            )

                            if isinstance(result, dict) and "progress" in result:
                                progress = result["progress"]
                                total = result["total"]
                                percentage = (progress / total) * 100
                                logging.info(
                                    f"Progress: {progress}/{total} ({percentage:.1f}%)"
                                )

                            return f"Tool execution result: {result}"
                        except Exception as e:
                            error_msg = f"Error executing tool: {str(e)}"
                            logging.error(error_msg)
                            return error_msg

                return f"No server found with tool: {tool_call['tool']}"
            return llm_response
        except json.JSONDecodeError:
            return llm_response

    async def start(self) -> None:
        """Main chat session handler."""
        try:
            for server in self.servers:
                try:
                    await server.initialize()
                except Exception as e:
                    logging.error(f"Failed to initialize server: {e}")
                    await self.cleanup_servers()
                    return

            all_tools = []
            for server in self.servers:
                tools = await server.list_tools()
                all_tools.extend(tools)

            tools_description = "\n".join([tool.format_for_llm() for tool in all_tools])

            system_message = (
                "You are a helpful assistant with access to these tools:\n\n"
                f"{tools_description}\n"
                "Further details about the tools:\n"
                "1. Email Tools:\n"
                "   - Send emails (send-email)\n"
                "   - Read unread emails (get-unread-emails)\n"
                "   - Read specific email content (read-email)\n"
                "   - Trash emails (trash-email)\n"
                "   - Open emails in browser (open-email)\n"
                "2. Web Search Tools:\n"
                "   - Search the web (search-web)\n"
                "   - Browse website content (website-browsing)\n"
                "Guidelines:\n"
                "- For email operations:\n"
                "  * Never send an email or trash an email without user confirmation\n"
                "  * Always ask for approval if not already given\n"
                "  * Be careful with sensitive information\n"
                "- For web operations:\n"
                "  * Use search-web to find information\n"
                "  * Use website-browsing to read specific web pages\n"
                "  * Be mindful of website content and respect privacy\n"
                "Choose the appropriate tool based on the user's question. "
                "If no tool is needed, reply directly.\n\n"
                "IMPORTANT: When you need to use a tool, you must ONLY respond with "
                "the exact JSON object format below, nothing else:\n"
                "{\n"
                '    "tool": "tool-name",\n'
                '    "arguments": {\n'
                '        "argument-name": "value"\n'
                "    }\n"
                "}\n\n"
                "After receiving a tool's response:\n"
                "1. Transform the raw data into a natural, conversational response\n"
                "2. Keep responses concise but informative\n"
                "3. Focus on the most relevant information\n"
                "4. Use appropriate context from the user's question\n"
                "5. Avoid simply repeating the raw data\n\n"
                "Please use only the tools that are explicitly defined above."
            )

            messages = [{"role": "system", "content": system_message}]

            while True:
                try:
                    user_input = input("You: ").strip().lower()
                    if user_input in ["quit", "exit"]:
                        logging.info("\nExiting...")
                        break

                    messages.append({"role": "user", "content": user_input})

                    llm_response = self.llm_client.get_response(messages)
                    logging.info("\nAssistant: %s", llm_response)

                    result = await self.process_llm_response(llm_response)

                    if result != llm_response:
                        messages.append({"role": "assistant", "content": llm_response})
                        messages.append({"role": "system", "content": result})

                        final_response = self.llm_client.get_response(messages)
                        logging.info("\nFinal response: %s", final_response)
                        messages.append(
                            {"role": "assistant", "content": final_response}
                        )
                    else:
                        messages.append({"role": "assistant", "content": llm_response})

                except KeyboardInterrupt:
                    logging.info("\nExiting...")
                    break

        finally:
            await self.cleanup_servers()