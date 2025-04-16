from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain_community.agent_toolkits.load_tools import load_tools,get_all_tool_names
from langchain.tools import tool,Tool,StructuredTool
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, List
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

# Define a custom tool to calculate age
class CalculateAgeInput(BaseModel):
    birth_year : int = Field(..., description="The year of birth")
    current_year : int = Field(..., description="The current year")

class CompareAgeInput(BaseModel):
    age : int = Field(..., description="The age to compare with")
    threshold : int = Field(..., description="The threshold to compare against")
    

@tool(
    name_or_callable="calculate_age",
    description="Calculates a person's age using their birth year. Needs the current year from another tool like 'google-search'.",
    args_schema=CalculateAgeInput,
    return_direct=True)
def calculate_age(birth_year: int, current_year: int) -> int:
    """
    Calculates a person's age using their birth year. Needs the current year from another tool like 'google-search'.
    """
    age = current_year - birth_year
    return age

@tool(
    name_or_callable="compare_age",
    description="Compare a person's age to a target threshold (e.g., to check if someone is older than 45).",
    args_schema=CompareAgeInput,
    return_direct=True)
def compare_age(age: int, threshold: int) -> str:
    """
    Compare a person's age to a target threshold (e.g., to check if someone is older than 45).",
    """
    if age > threshold:
        return f"Yes the Person is older than the {threshold}."
    elif age < threshold:
        return f"Yes the Person is younger than the {threshold}."
    else:
        return f"The person is exactly {threshold} years old"
    
# load_tools
math_tool = load_tools(["llm-math"], llm=llm)
search_tool = load_tools(
    ["google-search"], 
    llm=llm,
    google_api_key = GOOGLE_API_KEY,
    google_cse_id = GOOGLE_CSE_ID
)

calculate_age_tool = calculate_age
compare_age_tool = compare_age
tools = math_tool + search_tool + [calculate_age_tool,compare_age_tool]


# lets initialize the agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
)
# Define a simple function to run the agent
def run_agent(input_text):
    # Run the agent with the input text
    response = agent.run(input_text)
    return response
if __name__ == "__main__":
    # Run the agent with user input
    while True:
        user_input = input("Enter a question: ")
        if user_input.lower() == "exit":
            break
        response = run_agent(user_input)
        print(response)
