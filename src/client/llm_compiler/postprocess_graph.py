import logging
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    chain as as_runnable,
)
from client.llm_compiler.state import State

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

class PostprocessingGraph:
    def __init__(self, llm):
        self.llm = llm
        self.subgraph = None

    async def build_subgraph(self):
        logging.info("Building postprocessing subgraph...")

        @as_runnable
        async def summarize_if_needed(state):
            last_msg = state["messages"][-1]
            if isinstance(last_msg, AIMessage) and "All tasks completed" in last_msg.content:
                summarization_prompt = ChatPromptTemplate.from_template("""
                You are a helpful assistant. Based on the user's question and the content retrieved using tools, provide a clear answer.

                User question:
                {question}

                Retrieved content:
                {tool_output}
                """)
                summarizer = summarization_prompt | self.llm
                context = "\n".join(
                    [msg.content for msg in state["messages"] if hasattr(msg, "content")]
                )
                summary = await summarizer.ainvoke({"context": context})
                return {"messages": state["messages"] + [AIMessage(content=summary.content)]}
            return state

        postprocessing_graph = StateGraph(State)
        postprocessing_graph.add_node("summarize_if_needed", summarize_if_needed)
        postprocessing_graph.set_entry_point("summarize_if_needed")

        self.subgraph = postprocessing_graph.compile()
        return self.subgraph
