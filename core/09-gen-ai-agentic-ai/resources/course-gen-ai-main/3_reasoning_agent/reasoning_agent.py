from dotenv import load_dotenv
import os

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model="qwen/qwen3-32b", temperature=0)


agent = create_agent(
    tools=[],
    model=llm,
    system_prompt="""
    You are an advanced reasoning assistant.
    List out all the steps you carry to reason with numbers
    If you're using any formula, it should not be in Latex but plain formulas
    """
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "How much time will it take a cheetah to travel from New Delhi to Mumbai?"
            }
        ]
    }
)

print(result["messages"][-1].content)

