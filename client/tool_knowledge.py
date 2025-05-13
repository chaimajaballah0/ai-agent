import httpx
import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import PromptTemplate

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)



class LLMAgentClient:
    """Manages communication with Google Gemini"""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,

        )
        self.name = "KnowledgeAgent"
        self.__initialize_chain()

    def __initialize_chain(self):
        template = """
            You are a knowledge agent. Answer general knowledge questions using your internal knowledge.\n
            User's question: {question}\n
            Do not perform any external searches or calculations.\n
            Provide concise and accurate answers.
        """
        prompt = PromptTemplate(template=template, input_variables=["question"])
        self.chain = prompt | self.llm
        

    def get_response(self, question: str) -> str:
        return self.chain.invoke({"question": question})



            
