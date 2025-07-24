import json
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END

# Define the state schema
class State(TypedDict):
    transactions: str  # JSON string of list with {date, description, amount}
    category_budget: str  # JSON string of {category: budget_amount}
    categorized_str: str  # JSON string of enriched transactions with category field
    summary_str: str  # JSON string of {category: total} amounts

# Initialize LLM for categorizer and advisor nodes
llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")

# Create the StateGraph
graph_builder = StateGraph(State)

# Node functions will be implemented in subsequent tasks
def categorizer(state: State) -> dict:
    """Categorizer node - to be implemented"""
    return {"categorized_str": ""}

def summarizer(state: State) -> dict:
    """Summarizer node - to be implemented"""
    return {"summary_str": ""}

def advisor(state: State) -> dict:
    """Advisor node - to be implemented"""
    return {"summary_str": ""}

# Add nodes to the graph
graph_builder.add_node("categorizer", categorizer)
graph_builder.add_node("summarizer", summarizer)
graph_builder.add_node("advisor", advisor)
