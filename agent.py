import json
from typing_extensions import TypedDict
from dataclasses import dataclass
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    """State schema for the Personal Finance Tracker Agent"""
    transactions: str
    category_budget: str
    categorized_str: str
    summary_str: str


@dataclass
class DefaultState:
    """Default state values for the Personal Finance Tracker Agent"""
    transactions: str = "[]"
    category_budget: str = '{"Groceries":200, "Rent":1000, "Utilities":150, "Entertainment":100}'
    categorized_str: str = ""
    summary_str: str = ""


# Initialize the StateGraph with the State class
graph_builder = StateGraph(State)


# Default state instance for initialization
default_state = DefaultState()


