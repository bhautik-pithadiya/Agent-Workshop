from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain_community.agent_toolkits.load_tools import load_tools,get_all_tool_names
from langchain.tools import tool,Tool,StructuredTool
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, List
import os

load_dotenv()





# Define a pydantic model for the input to the custom tool
class CalculateAgeInput(BaseModel):
    birth_year: int = Field(..., description="The year of birth")
    current_year: int = Field(..., description="The current year")

class CompareAgeInput(BaseModel):
    age: int = Field(..., description="The age to compare with")
    threshold: int = Field(..., description="The threshold to compare against")

@tool(
    name_or_callable="calculate_age",
    description="Calculates a person's age using their birth year. Needs the current year from another tool like 'google-search'.",
    args_schema=CalculateAgeInput,
    return_direct=True
)
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
    return_direct=True
)
def compare_age(age: int, threshold: int) -> str:
    """
    Compare a person's age to a target threshold (e.g., to check if someone is older than 45).
    """
    if age > threshold:
        return f"Yes the Person is older than the {threshold}."
    elif age < threshold:
        return f"Yes the Person is younger than the {threshold}."
    else:
        return f"The person is exactly {threshold} years old"