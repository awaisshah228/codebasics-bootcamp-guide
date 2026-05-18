from dotenv import load_dotenv
import os

from langchain.agents import create_agent
from langchain_community.utilities import SerpAPIWrapper
from langchain.tools import tool
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model="qwen/qwen3-32b", temperature=0)

# Fetch SerpAPI key
serpapi_key = os.getenv("SERPAPI_API_KEY")

serp = SerpAPIWrapper(
    serpapi_api_key=serpapi_key,
    params={
        "tbm": "nws",     # Search in Google News
        "tbs": "qdr:d",   # Past day (24 hours)
    },
)

@tool
def search_news(query: str) -> str:
    """
    Search last-24h Google News via SerpAPI.
    Returns news results with URLs.
    """
    return serp.run(query)

agent = create_agent(
    tools=[search_news],
    model=llm,
    system_prompt=(
        "You are a news reporter who reports news using a simple language "
        "in 3 lines along with date of news"
    )
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Tell me the most latest breaking political news from the USA "
                    "along with the web url of news"
                ),
            }
        ]
    }
)



print(result["messages"][-1].content)

