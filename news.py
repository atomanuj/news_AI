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
os.environ["GEMINI_API_KEY"]=os.getenv("GEMINI_API_KEY")


@tool
def search(query:str):
    """Use the SERP API to run a Google Search."""
    search = SerpAPIWrapper()
    return search.run(query)


sys.stdout.reconfigure(encoding='utf-8')

model = ChatGroq(model="llama-3.3-70b-versatile")
tools = [search]

st.title("UPSC News Summarizer")
st.write("Provide the newspaper name or topic you'd like summarized:")

# Input for newspaper name or topic
input_text = st.text_input("Enter the newspaper name or topic:")

# Input for selecting the type of output
output_type = st.radio(
    "Select the type of output:",
    ("Summary Only", "Detailed Analysis")
)

if input_text:
    # Construct the query dynamically based on the selected output type
    if output_type == "Summary Only":
        query = f"Summarize the key articles from today's {input_text} newspaper, focusing on the main points and headlines."
    else:  # Detailed Analysis
        query = f"Provide a detailed analysis of the key articles from today's {input_text} newspaper, including their main arguments and insights."

    # Create the agent
    agent = create_react_agent(model, tools)
    input_data = {"messages": [("human", query)]}

    # Display the results
    st.write("### Output:")
    final_output = None  # Variable to store the final output
    for s in agent.stream(input_data, stream_mode="values"):
        message = s["messages"][-1]
        if isinstance(message, tuple):
            # st.write(message)
            final_output = message  # Store the final message content
        else:
            #st.write(message.pretty_repr())
            final_output = message.pretty_repr()

    # Display only the final analysis
    if final_output:
        st.write(final_output)