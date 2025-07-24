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
# Define default state values so the graph can run with empty input
DEFAULT_STATE = {
    "transactions": "[]",
    "category_budget": '{"Groceries":200,"Rent":1000,"Utilities":150,"Entertainment":100}',
    "categorized_str": "",
    "summary_str": ""
}

# Create the StateGraph
graph_builder = StateGraph(State)

# Node functions will be implemented in subsequent tasks
def categorizer(state: State) -> dict:
    """Categorizer node - uses LLM to assign category field to each transaction"""
    # Ensure default state values are set if not provided
    transactions = state.get("transactions", "[]")
    
    # Apply default values for any missing state fields
    # Placeholder implementation - will be completed in next task
    if transactions == "[]":
        return {"categorized_str": "[]"}
    return {"categorized_str": transactions}  # Temporary passthrough

def summarizer(state: State) -> dict:
    """Summarizer node - parses categorized transactions and sums amounts per category"""
    # Placeholder implementation - will be completed in next task
    return {"summary_str": "{}"}

def advisor(state: State) -> dict:
    """Advisor node - compares spending vs budget and generates advice"""
    # Ensure default state values are set if not provided
    
    # Placeholder implementation - will be completed in next task
    category_summary = json.loads(state.get("summary_str", "{}"))
    final_report = {
        "category_summary": category_summary,
        "advice": {}
    }
    return {"summary_str": json.dumps(final_report)}

# Add nodes to the graph
graph_builder.add_node("categorizer", categorizer)
graph_builder.add_node("summarizer", summarizer)
graph_builder.add_node("advisor", advisor)

# Add edges to connect the nodes: START -> categorizer -> summarizer -> advisor -> END
graph_builder.add_edge(START, "categorizer")
graph_builder.add_edge("categorizer", "summarizer")
graph_builder.add_edge("summarizer", "advisor")
graph_builder.add_edge("advisor", END)

# Compile the graph and export as compiled_graph for evaluation script
compiled_graph = graph_builder.compile()


