import asyncio
from helper import extract_json
import httpx
import logging
import json
from typing import Any, TypedDict, Literal, List, Union

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field

from langgraph.graph import StateGraph, END

from client.agent.server import Server

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    messages: list[dict[str, str]]
    llm_response: str | None
    next: str | None

class ToolCall(BaseModel):
    tool: str
    arguments: dict

class ToolChain(BaseModel):
    tool_chain: Literal["sequential", "parallel"]
    tools: List[Union["ToolCall", "ToolChain"]]

ToolChain.update_forward_refs()
ToolCallOrChain = Union[ToolCall, ToolChain]

class LLMClient:
    """Manages communication with Google Gemini"""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash") -> None:
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
        )

    def get_response(self, messages: list[dict[str, str]]) -> str:
        lc_msgs = []
        for msg in messages:
            role, content = msg["role"], msg["content"]
            if role == "system":
                lc_msgs.append(SystemMessage(content=content))
            elif role == "user":
                if not content.strip():
                    logging.warning("Skipping empty user input")
                    return "Please say something so I can help you."
                lc_msgs.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_msgs.append(AIMessage(content=content))
            else:
                raise ValueError(f"Unknown role: {role!r}")

        try:
            ai_msg = self.llm.invoke(lc_msgs)
            return ai_msg.content
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logging.error(f"Error getting LLM response: {e}")
            return "I encountered an error communicating with the model. Please try again."

        
def flatten(nested):
    """Flattens nested lists from mixed parallel execution."""
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

class Planner:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def plan(self, messages: list[dict[str, str]]) -> list[str]:
        plan_prompt = (
            "You are a task planner. Based on the user query, outline a list of concrete steps "
            "to achieve the user's goal. Respond with one step per line."
        )
        planning_messages = [{"role": "system", "content": plan_prompt}] + messages
        plan_text = self.llm_client.get_response(planning_messages)
        steps = [step.strip("- ").strip() for step in plan_text.strip().split("\n") if step.strip()]
        return steps


class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    def __init__(self, servers: list[Server], llm_client: LLMClient) -> None:
        self.servers = servers
        self.llm_client = llm_client

    async def cleanup_servers(self) -> None:
        for server in self.servers:
            try:
                await server.cleanup()
            except asyncio.CancelledError:
                logging.warning(f"Cleanup cancelled for {server.name}")
            except Exception as e:
                logging.warning(f"Cleanup error for {server.name}: {e}")

    async def process_llm_response(self, llm_response: str) -> str:
        cleaned = extract_json(llm_response)
        try:
            tool_data = json.loads(cleaned)

            def is_tool_instruction(data: Any) -> bool:
                if isinstance(data, dict):
                    return "tool" in data or "tool_chain" in data
                if isinstance(data, list):
                    return any(is_tool_instruction(item) for item in data)
                return False

            if not is_tool_instruction(tool_data):
                return llm_response

            async def run_tool(tool_call):
                    name = tool_call.get("tool")
                    args = tool_call.get("arguments", {})
                    for server in self.servers:
                        tools = await server.list_tools()
                        if any(tool.name == name for tool in tools):
                            try:
                                result = await server.execute_tool(name, args)
                                return f"{name} → {result}"
                            except Exception as e:
                                return f"{name} → Error: {str(e)}"
                    return f"{name} → Tool not found"

            async def execute_node(node) -> list[str]:
                    # If it's a single tool
                    if "tool" in node:
                        return [await run_tool(node)]

                    # If it's a nested tool_chain group
                    if "tool_chain" in node and "tools" in node:
                        chain_type = node["tool_chain"]
                        tools = node["tools"]

                        if chain_type == "parallel":
                            return await asyncio.gather(*[execute_node(t) for t in tools])
                        else:  # sequential
                            results = []
                            for t in tools:
                                res = await execute_node(t)
                                results.extend(res)
                            return results

                    return [f"Invalid tool structure: {node}"]

                # Handle root object or list
            results = []
            if isinstance(tool_data, list):
                for node in tool_data:
                    results.extend(await execute_node(node))
            else:
                results.extend(await execute_node(tool_data))

            return "\n".join(flatten(results))

        except json.JSONDecodeError:
            return llm_response
        except Exception as e:
            return f"Tool execution failed: {str(e)}"




    async def start(self) -> None:
        try:
            for server in self.servers:
                await server.initialize()

            all_tools = []
            for server in self.servers:
                tools = await server.list_tools()
                all_tools.extend(tools)

            tools_description = "\n".join([tool.format_for_llm() for tool in all_tools])

            system_message = (
                "You are a helpful assistant with access to these tools:\n\n"
                f"{tools_description}\n"
                # "Further details about the tools:\n"
                # "1. Email Tools:\n"
                # "   - Test tool (test_tool) - No arguments needed\n"
                # "   - Send emails (send_email) - Arguments: recipient_id, subject, message\n"
                # "   - Read unread emails (get_unread_emails)\n"
                # "   - Read specific email content (read_email)\n"
                # "   - Trash emails (trash_email)\n"
                # "   - Open emails in browser (open_email)\n"
                # "2. Web Search Tools:\n"
                # "   - Search the web (search_web)\n"
                # "   - Browse website content (website_browsing)\n"
                # "Guidelines:\n"
                # "- For email operations:\n"
                # "  * Never send an email or trash an email without user confirmation\n"
                # "  * Always ask for approval if not already given\n"
                # "  * Be careful with sensitive information\n"
                # "  * For send_email, only use recipient_id, subject, and message arguments\n"
                # "- For web operations:\n"
                # "  * Use search_web to find information\n"
                # "  * Use website_browsing to read specific web pages\n"
                # "  * Be mindful of website content and respect privacy\n"
                "If prompt is about actions, use tools.\n\n"
                "If prompt is about information available in your own knowledge, reply directly.\n\n"
                "If prompt composed of questions about information available in your own knowledge, and actions are needed, use your knowledge when needed and use tools when needed.\n\n"
                "If no tool is needed, reply directly.\n\n"
                "You may use tools sequentially or in parallel.\n"
                "After using a tool, wait for its result and use it to decide next steps.\n"
                "IMPORTANT: When you need to use a tool, you must ONLY respond with "
                "the exact JSON object format below, nothing else:\n"

                "{\n"
                '    "tool_chain": "sequential" or "parallel",\n'
                '    "tools": [\n'
                '        {\n'
                '            "tool_chain": "sequential" or "parallel",       // optional nested group\n'
                '            "tools": [ ... ]                                // nested tool calls\n'
                '        },\n'
                '        {\n'
                '            "tool": "tool-name",\n'
                '            "arguments": {\n'
                '                "arg1": "value1",\n'
                '                "arg2": "value2"\n'
                '            }\n'
                '        },\n'
                '        ...\n'
                "    ]\n"
                "}\n"

                "Alternatively, if only one tool is needed:\n"
                "{\n"
                '    "tool": "tool-name",\n'
                '    "arguments": {\n'
                '        "arg1": "value1",\n'
                '        "arg2": "value2"\n'
                '    }\n'
                "}\n"
                "Return only JSON. Do not include markdown fences, explanations, or extra text."
                "After receiving a tool's response:\n"
                "1. Transform the raw data into a natural, conversational response\n"
                "2. Keep responses concise but informative\n"
                "3. Focus on the most relevant information\n"
                "4. Use appropriate context from the user's question\n"
                "5. Avoid simply repeating the raw data\n\n"
                "Please use only the tools that are explicitly defined above."
            )


            initial_messages = [{"role": "system", "content": system_message}]

            graph = self.build_langgraph_agent()

            while True:
                user_input = input("You: ").strip().lower()
                if user_input in ["exit", "quit"]:
                    break

                initial_messages.append({"role": "user", "content": user_input})
                output = await graph.ainvoke({"messages": initial_messages})
                messages = output["messages"]
                print(f"Assistant: {messages[-1]['content']}")

        
        except asyncio.CancelledError:
            logging.warning("Session cancelled. Skipping async cleanup.")
        except Exception as e:
            logging.error(f"Unhandled error in session: {e}")
        finally:
            try:
                await self.cleanup_servers()
            except Exception as e:
                logging.warning(f"Cleanup skipped due to async cancel conflict: {e}")

    def build_langgraph_agent(self):
        planner = Planner(self.llm_client)

        """LangGraph state machine that manages LLM + tool steps."""
        def llm_step(state: AgentState) -> AgentState:
            messages = state["messages"]
            response = self.llm_client.get_response(messages)
            logger.info(f"LLM response: {response}")
            messages.append({"role": "assistant", "content": response})
            return AgentState(messages=messages, llm_response=response, next="tool_or_continue")

        async def tool_or_continue(state: AgentState) -> AgentState:
            llm_response = state["llm_response"]
            if not llm_response or not llm_response.strip():
                logger.warning("LLM response was empty.")
                return AgentState(messages=state["messages"], llm_response=llm_response, next=END)

            cleaned = extract_json(llm_response)
            if not cleaned:
                logger.info("LLM response did not contain JSON. Assuming no tool use.")
                return AgentState(messages=state["messages"], llm_response=llm_response, next=END)

            try:
                tool_data = json.loads(cleaned)

                def is_tool_instruction(data: Any) -> bool:
                    if isinstance(data, dict):
                        return "tool" in data or "tool_chain" in data
                    if isinstance(data, list):
                        return any(is_tool_instruction(item) for item in data)
                    return False

                if not is_tool_instruction(tool_data):
                    return AgentState(messages=state["messages"], llm_response=llm_response, next=END)

                # Tool use detected → run planner
                steps = planner.plan(state["messages"])
                plan_text = "\n".join([f"- {step}" for step in steps])
                logger.info(f"Generated plan:\n{plan_text}")

                messages = state["messages"] + [{"role": "assistant", "content": f"Here's the plan:\n{plan_text}"}]
                return AgentState(messages=messages, llm_response=None, next="llm_step")

            except json.JSONDecodeError:
                logger.info("No valid tool JSON detected.")
                return AgentState(messages=state["messages"], llm_response=llm_response, next=END)

            except Exception as e:
                logger.error(f"Tool detection failed: {e}")
                return AgentState(messages=state["messages"], llm_response=str(e), next=END)

        # Define the LangGraph workflow
        workflow = StateGraph(AgentState)
        workflow.add_node("llm_step", llm_step)
        workflow.add_node("tool_or_continue", tool_or_continue)

        workflow.set_entry_point("llm_step")
        workflow.add_edge("llm_step", "tool_or_continue")
        workflow.add_conditional_edges("tool_or_continue", lambda state: state["next"])



        return workflow.compile()
