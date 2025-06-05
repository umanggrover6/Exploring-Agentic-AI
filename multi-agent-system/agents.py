import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from tools import add, multiply, web_search

load_dotenv()

#Set up the LLM
llm = ChatGroq(model_name="Llama3-8b-8192")


#Agents
math_agent = create_react_agent(model=llm,
                                tools=[add, multiply],
                                name="math_expert",
                                prompt="You are a math expert. You can add and multiply numbers. Use one tool at a time.")

research_agent = create_react_agent(model=llm,
                                tools=[web_search],
                                name="research_expert",
                                prompt="You are a research expert. You can search the web for information.")


#Supervisor
prompt = ("You are a team supervisor managing two experts: a math expert and a research expert."
          "For current event, use research agent"
          "For math problems, use math agent.")


math_research_supervisor = create_supervisor(model=llm,
                                             agents=[math_agent, research_agent],
                                             output_mode="last_message",
                                             prompt=prompt)



app = math_research_supervisor.compile()

graph = app.get_graph().draw_ascii()  # Get image bytes
with open("output.txt", "w",encoding="utf-8") as f:  # Save as PNG file
    f.write(graph)
