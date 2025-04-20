from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain_community.agent_toolkits.load_tools import load_tools,get_all_tool_names
from langchain.tools import tool,Tool,StructuredTool
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, List
from tools import calculate_age, compare_age
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# Set up the Google GenAI model
llm = ChatGoogleGenerativeAI(
    api_key=GEMINI_API_KEY,
    model="gemini-2.0-flash-001",
    temperature=0.5,
    max_tokens=1000,
)

# load additional tools
math_tool = load_tools(['llm-math'], llm=llm)
# Set up the Google Search tool
google_search_tool = load_tools(
    ["google-search"], 
    llm=llm, 
    google_api_key=GOOGLE_API_KEY, 
    google_cse_id=GOOGLE_CSE_ID
)

# combine all tools
calculate_age_tool = calculate_age
compare_age_tool = compare_age
tools = google_search_tool + math_tool + [calculate_age_tool, compare_age_tool]

# Initialize the agent with the tools
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    handle_parsing_errors=True,
)

def run_agent(input_text: str) -> str:
    """
    Run the agent with the given input text.
    """
    response = agent.invoke(input_text)
    return response

if __name__ == "__main__":
    while True:
        user_input = input("Enter a question: ")
        if user_input.lower() == "exit":
            break
        response = run_agent(user_input)
        print(response)