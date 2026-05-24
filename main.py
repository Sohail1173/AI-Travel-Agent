import os
from typing import TypedDict, Annotated
import operator

import psycopg
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from langchain_groq import ChatGroq

from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flight
from dotenv import load_dotenv
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)


class TravelState(TypedDict):
    messages=Annotated[list[AnyMessage],operator.add]
    user_query:str
    flight_results:str
    hotel_results:str
    itinerary:str
    llm_calls:int
    
def flght_agent(state:TravelState):
    query=state["user_query"]
    flight_data=search_flight(query)
    return {
        "flight_results":flight_data,
        "messages":[
            AIMessage(content=f"Flight results fetched")
        ],
        "llm_calls":state.get("llm_calls",0)+1
    }
    
def hotel_agent(state:TravelState):
    query=f"Best hotels for {state["user_query"]}"
    hotel_restuls=tavily_search(query)
    return {
        "hotel_restuls":hotel_restuls,
        "messages":[
            AIMessage(content=f"Hotel results fetched")
        ],
        "llm_calls":state.get("llm_calls",0)+1
    }
    
    
    
def itinerary_agent(state:TravelState):
    prompt=f"""
    Create a travel itinerary.
    
    User Query:
    {state["user_query"]}
    
    Flight Results:
    {state["flight_results"]}
    
    Hotel Results:
    {state["hotel_results"]}
    
    """
    
    response=llm.invoke([
        SystemMessage(content="You are an expert travel planner")
    ,
    HumanMessage(content=prompt)])
    
    return {
        "itinerary":response.content,
        "messages": [response],
        "llm_calls":state.get("llm_calls",0)+1
    }
    
def final_agent(state:TravelState):
        final_prompt=f"""
        
        Generate final travel response.
        
        Flights:
        {state["flight_results"]}
        
        Hotels:
        {state["hotel_results"]}
        
        Itinerary
        {state["itinerary"]}
        
        """
        response=llm.invoke([
            HumanMessage(content=final_prompt)
        ])
        return {
            "messages":[response],
            "llm_calls":state.get("llm_calls",0)+1
        }