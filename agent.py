import json
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    """State schema for the Personal Finance Tracker Agent"""
    transactions: str
    category_budget: str
    categorized_str: str
    summary_str: str


# Initialize the StateGraph with the State class
graph_builder = StateGraph(State)




