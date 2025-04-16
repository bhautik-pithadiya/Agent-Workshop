from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain_community.agent_toolkits.load_tools import load_tools,get_all_tool_names
from langchain.tools import Tool
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Set up the Google GenAI model
llm = ChatGoogleGenerativeAI(
    api_key=GEMINI_API_KEY,
    model="gemini-2.0-flash-001",
    temperature=0.5,
    max_tokens=1000,
)

# Load tools
tools = load_tools(["llm-math"], llm=llm)

# lets initialize the agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
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