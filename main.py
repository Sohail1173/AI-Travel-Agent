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
import logging

from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flight
from dotenv import load_dotenv
load_dotenv()
print("USER:", os.getenv("DB_USER"))
print("HOST:", os.getenv("DB_HOST"))
print("PORT:", os.getenv("DB_PORT"))
print("DB:  ", os.getenv("DB_NAME"))

DATABASE_URL = os.getenv("DATABASE_URL")



# LLM
llm = ChatGroq(
    model=os.getenv("model")
)

LOG_DIR="logs"
os.makedirs(LOG_DIR,exist_ok=True)
logging.basicConfig(filename=os.path.join(LOG_DIR,"execution.log"),
                    level=logging.INFO,
                    format="%(asctime)s-%(levelname)s-%(message)s",
                    filemode="w")

logger=logging.getLogger(__name__)
def write_text_file(filename:str,content:str):
    filepath=os.path.join(LOG_DIR,filename)
    with open(filepath,"w",encoding="utf-8") as f:
        f.write(content)



class TravelState(TypedDict):
    messages:Annotated[list[AnyMessage],operator.add]
    user_query:str
    flight_results:str
    hotel_results:str
    itinerary:str
    llm_calls:int
    
def flight_agent(state:TravelState):
    logger.info("Flight agent started")
    logger.info(f"User query:{state["user_query"]}")
    query=state["user_query"]
    flight_data=search_flight(query)
    write_text_file("flight_agent.txt",flight_data)
    logger.info(f"Flight Agent output:{flight_data}")
    return {
        "flight_results":flight_data,
        "messages":[
            AIMessage(content=f"Flight results fetched")
        ],
        "llm_calls":state.get("llm_calls",0)+1
    }
    
    
def hotel_agent(state:TravelState):
    logger.info("Hotel agent started")
    logger.info(f"User query:{state["user_query"]}")
    query=f"Best hotels for {state["user_query"]}"
    hotel_restuls=tavily_search(query)
    write_text_file("hotel_agent.txt",hotel_restuls)
    logger.info(f"Hotel Agent output:{hotel_restuls}")
    return {
        "hotel_results":hotel_restuls,
        "messages":[
            AIMessage(content=f"Hotel results fetched")
        ],
        "llm_calls":state.get("llm_calls",0)+1
    }

    
def itinerary_agent(state:TravelState):
    logger.info("itinerary_agent started")
    logger.info(f"User query:{state["user_query"]}")
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
    write_text_file("itinerary_agent.txt",response.content)
    logger.info(f"itinerary_agent output:{response.content}")
    
    return {
        "itinerary":response.content,
        "messages": [response],
        "llm_calls":state.get("llm_calls",0)+1
    }
    
def final_agent(state:TravelState):
        logger.info("final_agent started")
        final_prompt=f"""
        
        Generate final travel response.
        
        Flights:
        {state["flight_results"]}
        
        Hotels:
        {state["hotel_results"]} 
        
        itinerary
        {state["itinerary"]}
        
        """
        response=llm.invoke([
            HumanMessage(content=final_prompt)
        ])
        write_text_file("final_agent.txt",response.content)
        logger.info(f"final_agent output:{response.content}")
        return {
            "messages":[response],
            "llm_calls":state.get("llm_calls",0)+1
        }


        
graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)

import psycopg
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver

# ── 1. Run setup() on a raw autocommit connection ─────────────────────────
with psycopg.connect(
    host=os.getenv("DB_HOST", "localhost"),
    port=os.getenv("DB_PORT", "5432"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    dbname=os.getenv("DB_NAME"),
    autocommit=True          # ← required for CREATE INDEX CONCURRENTLY
) as conn:
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()
    print("✅ Checkpointer tables ready")

# ── 2. Now create the pool for actual use ─────────────────────────────────
pool = ConnectionPool(
    kwargs={
        "host":     os.getenv("DB_HOST", "localhost"),
        "port":     os.getenv("DB_PORT", "5432"),
        "user":     os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "dbname":   os.getenv("DB_NAME"),
    },
    open=True
)

checkpointer = PostgresSaver(pool)

# ── 3. Compile and run ─────────────────────────────────────────────────────
app = graph.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "user_sohail"}}

    user_input = input("Enter travel request: ")

    result = app.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "llm_calls": 0,
        },
        config=config,
    )

    for msg in result["messages"]:
        print(msg.content)