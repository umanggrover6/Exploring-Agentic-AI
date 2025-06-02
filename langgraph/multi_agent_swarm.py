import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_swarm, create_handoff_tool

# Set up environmemt and LLM
load_dotenv()
groq_api_key = os.getenv("groq")
llm = ChatGroq(groq_api_key=groq_api_key, 
               model_name="Llama3-8b-8192")

# Tools
def book_hotel(hotel_name: str):
    """Book a hotel"""
    return f"Sucessfully booked a stay at {hotel_name}."

def book_flight(from_airport: str, to_airport: str):
    """Book a flight"""
    return f"Succefully booked a flight from {from_airport} to {to_airport}."


# Hand-off tools
transfer_to_hotel_assistant = create_handoff_tool(
    agent_name="hotel_assistant",
    description="Transfer user to hotel-booking assistant"
)

transfer_to_flight_assistant = create_handoff_tool(
    agent_name="flight_assistant",
    description="Transfer user to flight-booking assistant"
)

# Agents
hotel_assistant = create_react_agent(
    model = llm,
    tools = [book_hotel, transfer_to_flight_assistant],
    prompt = "You a hotel booking assistant",
    name = "hotel_assistant"
)

fligh_assistant = create_react_agent(
    model = llm,
    tools = [book_flight, transfer_to_hotel_assistant],
    prompt = "You a flight booking assistant",
    name = "flight_assistant"
)


# Swarm
swarm = create_swarm(
    agents=[fligh_assistant, hotel_assistant],
    default_active_agent="flight_assistant"
).compile()

for chunk in swarm.stream(
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



