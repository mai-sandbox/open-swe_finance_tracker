import json
from dataclasses import dataclass
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


@dataclass
class State:
    """State class with default values for the Personal Finance Tracker Agent"""
    transactions: str = "[]"
    category_budget: str = '{"Groceries":200, "Rent":1000, "Utilities":150, "Entertainment":100}'
    categorized_str: str = ""
    summary_str: str = ""


# Initialize the StateGraph with the State class
graph_builder = StateGraph(State)


