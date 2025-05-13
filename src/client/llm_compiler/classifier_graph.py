import logging

from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain as as_runnable

from client.llm_compiler.state import State

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

class QueryClassificationGraph:
    def __init__(self, llm):
        self.llm = llm
        self.subgraph = None


    async def build_subgraph(self):
        @as_runnable
        async def classify_query(state):
            logging.info("Building simple or complex subgraph...")
            classifier_prompt = ChatPromptTemplate.from_messages([
                ("system", "Classify if the user query is a simple query that you have knowledge about, answer directly. If user query requires tools calling, plan. Respond with one word: 'simple' or 'complex'."),
                ("human", "{question}")
            ])
            classifier_chain = classifier_prompt | self.llm
            logging.info("Classifying query...")
            question = state["messages"][-1].content
            result = await classifier_chain.ainvoke({"question": question})
            classification = result.content.strip().lower()
            logging.info(f"Classification result: {classification}")
            return {"messages": state["messages"], "classification": classification}

        @as_runnable
        async def simple_answer(state):
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", "Answer the user's query clearly and concisely."),
                ("human", "{question}")
            ])
            qa_chain = qa_prompt | self.llm

            logging.info("Generating simple answer...")
            question = state["messages"][-1].content
            result = await qa_chain.ainvoke({"question": question})
            logging.info(f"Simple answer result: {result.content}")
            return {"messages": state["messages"] + [AIMessage(content=result.content)]}

        subgraph = StateGraph(State)

        subgraph.add_node("classify_query", classify_query)
        subgraph.add_node("simple_answer", simple_answer)

        subgraph.add_conditional_edges(
            "classify_query",
            lambda state: state.get("classification", "complex"),
            {"simple": "simple_answer", "complex": END}
        )

        subgraph.set_entry_point("classify_query")
        self.subgraph = subgraph.compile()
        return self.subgraph