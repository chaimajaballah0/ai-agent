import logging
import operator
from typing import List, Union, Tuple, TypedDict, Annotated

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from client.assistant.agent import LLMClient
from client.configuration.configuration import Configuration
from client.persistence.models.thread import UserThread

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)



class Plan(BaseModel):
    """Plan to follow in future"""
    
    # A list of steps to follow (in order)
    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )
# Enforce the response as a string (unlikely it would be anything else, but good to be safe!)
class Response(BaseModel):
    """Response to user."""

    response: str

# Create a structure for a plan - combine the current information with the next step of the plan.
class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If no tools needed, respond to user directly, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )

class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str
     
class Planner:
    def __init__(self, config: Configuration, user_id: str, thread_id: str ) -> None:
        self.config = config
        self.user_id = user_id
        self.thread_id = thread_id
        self.agent_executor = None
        self.planner = None
        self.replanner = None
        self.llm = ChatGoogleGenerativeAI(
            model=self.config.llm_model,
            google_api_key=self.config.api_key,
        )
        self.checkpointer = InMemorySaver()
        self.store = InMemoryStore()

# Save session state to DB
    async def persist_session(self):
        await UserThread.save_or_update(
            user_id=self.user_id,
            session_id=self.session_id,
            store=self.store.to_json(),
            checkpoint=self.checkpointer.to_json()
        )

    # Load a planner with session state from DB
    @classmethod
    async def load_session(cls, config: Configuration, user_id: str, thread_id: str):
        session_row = await UserThread.load(user_id, thread_id)
        if session_row is None:
            raise ValueError(f"No session found for user {user_id} and thread {thread_id}")

        planner = cls(config, user_id, thread_id)
        planner.store.load_json(session_row.store)
        planner.checkpointer.load_json(session_row.checkpoint)
        await planner.setup()
        return planner
    def set_planner(self):
        planner_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    For the given objective, come up with a simple step by step plan. 
                    This plan should involve individual tasks, that if executed correctly will yield the correct answer. 
                    Do not add any superfluous steps. 
                    The result of the final step should be the final answer. Make sure that each step has all the information needed.
                    Do not skip steps.
                    """,
                ),
                (
                    "placeholder", 
                    "{messages}"
                ),
            ]
        )
        # Use a structured output to ensure consistency
        # This basically forces the LLM to output its response in the same Pydantic structure every time
        planner = planner_prompt | self.llm.with_structured_output(Plan) 

        self.planner = planner

    def set_replanner(self):
        replanner_prompt = ChatPromptTemplate.from_template(
            """
            For the given objective, come up with a simple step by step plan. 
            This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. 
            The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

            Your objective was this:
            {input}

            Your original plan was this:
            {plan}

            You have currently done the follow steps:
            {past_steps}

            Update your plan accordingly. If no more steps are needed then respond with to the user. Otherwise, fill out the plan. 
            Only add steps to the plan that still NEED to be done. 
            Encourage the use of tools as part of the plan.
            """
        )
        replanner = replanner_prompt | self.llm.with_structured_output(Act)
        self.replanner = replanner

        
    async def setup(self):
        self.agent_executor = LLMClient(self.config)
        self.set_planner()
        self.set_replanner()    
    async def execute_step(self, state: PlanExecute):
        # Get our current plan
        plan = state["plan"]
        logging.info(f"Plan: {plan}")
        plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
        task = plan[0]

        # Tell the llm what the plan is, and what the next step is
        task_formatted = f"""
        For the following plan:
        {plan_str}\n
        You are tasked with executing step {1}, {task}.
        """
        
        # Get the llm to execute this stage of the plan
        agent_response = await self.agent_executor.mcp_call(task_formatted)
        # Return the past steps of the plan
        return {
            "past_steps": [(task, agent_response["messages"][-1].content)],
        }

# Ask the planner LLM to create a plan given the input - only called once
    async def plan_step(self, state: PlanExecute):
        plan = await self.planner.ainvoke({"messages": [("user", state["input"])]})
        
        return {"plan": plan.steps}

    # Ask the replanner to take the current plan and outputs of the plan to refine next steps
    async def replan_step(self, state: PlanExecute):
        logging.info(f"Replan state: {state}")

        output = await self.replanner.ainvoke({
            "input": state["input"],
            "plan": state["plan"],
            "past_steps": state["past_steps"],
        })
        logging.info(f"Replan output: {output}")
        if isinstance(output.action, Response):
            return {"response": output.action.response}
        elif not output.action.steps:
            return {"response": "I have completed the task. Thank you!"}
        else:
            return {"plan": output.action.steps}

    # Check if we have reached the end
    def should_end(self, state: PlanExecute):
        if "response" in state and state["response"]:
            return END
        else:
            return "execute_step"
     

    def build_graph(self):
        # Create the workflow
        # This is a simple workflow with 3 steps - plan, execute, replan
        workflow = StateGraph(PlanExecute)

        # Add the plan node
        workflow.add_node("plan_step", self.plan_step)

        # Add the execution step
        workflow.add_node("execute_step", self.execute_step)

        # Add a replan node
        workflow.add_node("replan_step", self.replan_step)

        workflow.add_edge(START, "plan_step")

        # From plan we go to agent
        workflow.add_edge("plan_step", "execute_step")

        # From agent, we replan
        workflow.add_edge("execute_step", "replan_step")

        workflow.add_conditional_edges(
            "replan_step",
            self.should_end,
            ["execute_step", END],
        )

        # Finally, we compile it!
        # This compiles it into a LangChain Runnable,
        # meaning you can use it as you would any other runnable
        return workflow.compile(checkpointer=self.checkpointer, store=self.store)

    async def start(self):
        workflow = self.build_graph()
        while True:
                user_input = input("You: ").strip().lower()
                if user_input in ["exit", "quit"]:
                    break
                config = {"recursion_limit": 50, 
                          "configurable": {
                              "thread_id": self.thread_id,
                              }}
                prompt_input = {"input": user_input}
                result = await workflow.ainvoke(prompt_input, config=config)
                logging.info(f"Result: {result.get('response')}")
     
