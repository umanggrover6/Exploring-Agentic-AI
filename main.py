import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from IPython.display import Image, display


# Set up the environment variables
load_dotenv()
groq_api_key = os.getenv("groq")
langsmith = os.getenv("langsmith")

#Set up the LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")


class State(TypedDict):
    messages: Annotated[list,add_messages]


graph_builder = StateGraph(State)


def chatbot(state: State) -> str:
   return {"messages": llm.invoke(state["messages"])}

# Add the nodes and edges to the graph
graph_builder.add_node("chatbot",chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()


# Save the Graph to a file
try:
    image_data = graph.get_graph().draw_mermaid_png()  # Get image bytes
    with open("output.png", "wb") as f:  # Save as PNG file
        f.write(image_data)

    display(Image("output.png"))  # Display the saved image
except Exception as e:
    print(f"Error: {e}")


while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit","quit", "stop"]:
        print("Goodbye!")
        break

    for event in graph.stream({"messages":("user",user_input)}):
        # print(event.values())
        for value in event.values():
            print("Assistant:",value["messages"].content)