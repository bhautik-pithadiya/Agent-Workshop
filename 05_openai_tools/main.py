import os 
import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, WebSearchTool

# load the environment variables from .env file
load_dotenv()

# Set up the OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

# Specilalist agent
age_calculator_agent = Agent(
    name = 'Age Calculator',
    instructions = """
    You are an age calculator that helps users calculate their age based on their birth year and current year.
    """,
)

async def main(query: str = "What is my age if I was born in 1990?"):
    agent_assistant = Agent(
        name = 'Agent Assistant',
        instructions = """
        You are a assistant that helps users calculate his/her age.
        When asked about a topic:
        1. Search the web for relevant, up-to-date information
        2. If needed do check the current year using the web search tool
        3. after generating the answer, do check is this what the user is looking for
        """,
        tools=[WebSearchTool(), 
               age_calculator_agent.as_tool(
                   tool_name="age_calculator",
                   tool_description="Calculate the age based on birth year and current year.",  
                   )],
        )
        
    # Initialize the agent with the OpenAI API key
    runner = Runner()
    
    result = await runner.run(agent_assistant, query,)
    print(result.final_output)

if __name__ == "__main__":
    # Run the main function with a sample query
    query = input("Enter a query (or 'exit' to quit): ")
    if query.lower() == "exit":
        print("Exiting...")
    else:
        # Run the main function with the user input
        print(f"Running agent with query: {query}")
        asyncio.run(main(query))