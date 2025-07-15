"""Personal Finance Tracker Agent using LangGraph."""

import json
import os
from typing import Dict, Any
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END

# Load environment variables
load_dotenv()


class State(TypedDict):
    """State schema for the Personal Finance Tracker."""
    transactions: str
    category_budget: str
    categorized_str: str
    summary_str: str


# Initialize the LLM
llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)


def categorizer(state: State) -> Dict[str, Any]:
    """
    Categorizer node that uses LLM to assign categories to transactions.
    
    Args:
        state: Current state containing transactions
        
    Returns:
        Dict with categorized_str field containing JSON string of categorized transactions
    """
    transactions_str = state["transactions"]
    
    # If no transactions, return empty list
    if not transactions_str or transactions_str == "[]":
        return {"categorized_str": "[]"}
    
    prompt = f"""
    You are a financial categorization expert. Given a list of transactions, assign each transaction an appropriate category.
    
    Available categories: Groceries, Rent, Utilities, Entertainment, Transportation, Healthcare, Shopping, Dining, Other
    
    Transactions: {transactions_str}
    
    For each transaction, add a "category" field with the most appropriate category.
    Return the result as a valid JSON array with the same structure but with added category fields.
    
    Example format:
    [
        {{"date": "2024-01-01", "description": "Grocery Store", "amount": 50.0, "category": "Groceries"}},
        {{"date": "2024-01-02", "description": "Electric Bill", "amount": 120.0, "category": "Utilities"}}
    ]
    
    Return only the JSON array, no additional text.
    """
    
    response = llm.invoke(prompt)
    categorized_str = response.content.strip()
    
    return {"categorized_str": categorized_str}


def summarizer(state: State) -> Dict[str, Any]:
    """
    Summarizer node that parses categorized transactions and calculates spending totals per category.
    
    Args:
        state: Current state containing categorized_str
        
    Returns:
        Dict with summary_str field containing JSON string of category totals
    """
    categorized_str = state["categorized_str"]
    
    # Parse the categorized transactions
    try:
        transactions = json.loads(categorized_str)
    except (json.JSONDecodeError, TypeError):
        transactions = []
    
    # Calculate totals per category
    category_totals = {}
    for transaction in transactions:
        category = transaction.get("category", "Other")
        amount = float(transaction.get("amount", 0))
        category_totals[category] = category_totals.get(category, 0) + amount
    
    summary_str = json.dumps(category_totals)
    return {"summary_str": summary_str}


def advisor(state: State) -> str:
    """
    Advisor node that compares spending with budget and generates advice for over-budget categories.
    
    Args:
        state: Current state containing summary_str and category_budget
        
    Returns:
        Final report JSON string with category_summary and advice fields
    """
    summary_str = state["summary_str"]
    category_budget_str = state["category_budget"]
    
    # Parse the data
    try:
        category_summary = json.loads(summary_str)
        category_budget = json.loads(category_budget_str)
    except (json.JSONDecodeError, TypeError):
        category_summary = {}
        category_budget = {}
    
    # Find over-budget categories
    over_budget_categories = []
    for category, spent in category_summary.items():
        budget = category_budget.get(category, 0)
        if spent > budget:
            over_budget_categories.append({
                "category": category,
                "spent": spent,
                "budget": budget,
                "overage": spent - budget
            })
    
    # Generate advice for over-budget categories using LLM
    advice = {}
    for over_budget in over_budget_categories:
        category = over_budget["category"]
        spent = over_budget["spent"]
        budget = over_budget["budget"]
        overage = over_budget["overage"]
        
        prompt = f"""
        You are a financial advisor. A person has overspent in the {category} category.
        
        Budget: ${budget}
        Actual spending: ${spent}
        Overage: ${overage}
        
        Provide a brief, practical tip (1-2 sentences) to help them reduce spending in this category next month.
        Be specific and actionable.
        """
        
        response = llm.invoke(prompt)
        advice[category] = response.content.strip()
    
    # Create final report
    final_report = {
        "category_summary": category_summary,
        "advice": advice
    }
    
    return json.dumps(final_report)


# Create the StateGraph
graph_builder = StateGraph(State)

# Add nodes
graph_builder.add_node("categorizer", categorizer)
graph_builder.add_node("summarizer", summarizer)
graph_builder.add_node("advisor", advisor)

# Add edges to create the flow: START -> categorizer -> summarizer -> advisor -> END
graph_builder.add_edge(START, "categorizer")
graph_builder.add_edge("categorizer", "summarizer")
graph_builder.add_edge("summarizer", "advisor")
graph_builder.add_edge("advisor", END)

# Compile the graph
graph = graph_builder.compile()

# Export the compiled graph as required by evaluation script
compiled_graph = graph

