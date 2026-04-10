import os
import pandas as pd
from typing import TypedDict
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph

# -----------------------
# Load environment
# -----------------------

load_dotenv()

# -----------------------
# Load dataset
# -----------------------

df = pd.read_csv("data/logistics_disruption_dataset.csv")


# -----------------------
# LLM setup
# -----------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# -----------------------
# Prompt
# -----------------------

prompt = PromptTemplate.from_template(
"""
You are an expert supply chain analyst.

Shipment route: {origin} → {destination}

Port risk data:
{port_risk}

Route history data:
{route_history}

Explain briefly why this shipment may be risky and suggest ONE rerouting recommendation.

Respond in this format:

RISK:
RECOMMENDATION:
CONFIDENCE:
"""
)

# -----------------------
# Graph State
# -----------------------

class State(TypedDict, total=False):

    origin: str
    destination: str
    port_risk: str
    route_history: str
    analysis: str


# -----------------------
# Node 1: Port Risk
# -----------------------

def port_risk_node(state: State):

    origin = state.get("origin")

    sub = df[df["origin_port"] == origin]

    if sub.empty:
        state["port_risk"] = "No port data available."

    else:

        delay_rate = sub["is_delayed"].mean() if "is_delayed" in df.columns else 0
        weather = sub["weather_severity_score"].mean() if "weather_severity_score" in df.columns else 0

        state["port_risk"] = f"""
Delay Rate: {delay_rate:.2%}
Weather Severity: {weather:.2f}
Records: {len(sub)}
"""

    return state


# -----------------------
# Node 2: Route History
# -----------------------

def route_history_node(state: State):

    origin = state.get("origin")
    dest = state.get("destination")

    sub = df[
        (df["origin_port"] == origin) &
        (df["destination_port"] == dest)
    ]

    if sub.empty:
        state["route_history"] = "No historical route data."

    else:

        delay_rate = sub["is_delayed"].mean() if "is_delayed" in df.columns else 0
        avg_delay = sub["delay_minutes"].mean() if "delay_minutes" in df.columns else 0

        state["route_history"] = f"""
Historical Delay Rate: {delay_rate:.2%}
Average Delay: {avg_delay:.1f} minutes
Shipments: {len(sub)}
"""

    return state


# -----------------------
# Node 3: LLM Analysis
# -----------------------

def reroute_node(state: State):

    chain = prompt | llm

    result = chain.invoke(state)

    state["analysis"] = result.content

    return state


# -----------------------
# Build LangGraph
# -----------------------

graph = StateGraph(State)

graph.add_node("port_risk", port_risk_node)
graph.add_node("route_history", route_history_node)
graph.add_node("reroute", reroute_node)

graph.set_entry_point("port_risk")

graph.add_edge("port_risk", "route_history")
graph.add_edge("route_history", "reroute")

app = graph.compile()

print("LangGraph logistics agent ready")

# -----------------------
# Test run
# -----------------------

result = app.invoke(
    {
        "origin": "Dubai",
        "destination": "Los Angeles"
    }
)

print("\n===== AI ANALYSIS =====\n")
print(result["analysis"])