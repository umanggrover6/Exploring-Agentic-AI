import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

# Set up environmemt and LLM
load_dotenv()

llm = ChatGroq(model_name="Llama3-8b-8192")


# Tools
def book_hotel(hotel_name: str):
    """Book a hotel"""
    return f"Sucessfully booked a stay at {hotel_name}."

def book_flight(from_airport: str, to_airport: str):
    """Book a flight"""
    return f"Succefully booked a flight from {from_airport} to {to_airport}."


# Agents
hotel_assistant = create_react_agent(
    model = llm,
    tools = [book_hotel],
    prompt = "You a hotel booking assistant",
    name = "hotel_assistant"
)

fligh_assistant = create_react_agent(
    model = llm,
    tools = [book_flight],
    prompt = "You a flight booking assistant",
    name = "flight_assistant"
)


# Set Up Supervisor

supervisor = create_supervisor(
    agents = [hotel_assistant, fligh_assistant],
    model=llm,
    prompt=("You manage a hotel booking assistant and a flight booking assistant."
            "Assign work to them as per user request")
).compile()


for chunk in supervisor.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "book a flight from BOS to JFK and a stay at McKittrick Hotel"
            }
        ]
    }
):
    print(chunk)
    print("\n")
