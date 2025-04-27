from fastapi import FastAPI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_community.utilities import SerpAPIWrapper
import os
import sys
import streamlit as st
from dotenv import load_dotenv
from google import genai
from langchain_groq import ChatGroq

load_dotenv()

os.environ["SERPAPI_API_KEY"]=os.getenv("SERPAPI_API_KEY")
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

app=FastAPI(
    title="Langchain Server",
    version="1.0",
    decsription="A simple API Server"

)

@tool
def search(query:str):
    """Use the SERP API to run a Google Search."""
    search = SerpAPIWrapper()
    return search.run(query)


sys.stdout.reconfigure(encoding='utf-8')

model = ChatGroq(model = "llama-3.3-70b-versatile")
tools = [search]

st.title("UPSC News Summarizer")
st.write("Provide the newspaper name or topic you'd like summarized:")

input_text = st.text_input("Enter the newspaper name or topic:")


if input_text:
    # Construct the query dynamically
    query = f"I am a UPSC aspirant. Summarize today's {input_text} newspaper in detail, focusing on today's news articles which are useful to me."

    # Create the agent
    agent = create_react_agent(model, tools)
    input_data = {"messages": [("human", query)]}

    # Display the results
    st.write("### Summary:")
    final_output = None  # Variable to store the final output
    for s in agent.stream(input_data, stream_mode="values"):
        message = s["messages"][-1]
        if isinstance(message, tuple):
            final_output = message[1]  # Store the final message content
        else:
            final_output = message.pretty_repr()

    # Display only the final analysis
    if final_output:
        st.write(final_output)

# query = """I am a UPSC aspirant . Summarize today's {input_text} newspaper whatever news article will be useful to me."""

# agent = create_react_agent(model, tools)
# input = {"messages" : [("human", query)]}



# for s in agent.stream(input , stream_mode = "values"):
#     message = s["messages"][-1]
#     if isinstance(message, tuple):
#         print(message)
#         #message.pretty_print()
#     else:
#         #print(message)
#         message.pretty_print()

