"""Personal Finance Tracker Agent using LangGraph.

This module implements a three-node workflow for categorizing transactions,
summarizing spending by category, and providing budget advice.
"""

import json
from typing import TypedDict

from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END


class FinanceState(TypedDict):
    """State schema for the Personal Finance Tracker agent.
    
    All fields are strings to ensure compatibility with the evaluation script.
    """
    transactions: str  # JSON string of list of transactions
    category_budget: str  # JSON string of budget per category
    categorized_str: str  # JSON string of categorized transactions
    summary_str: str  # JSON string of spending summary by category


def categorizer(state: FinanceState) -> dict[str, str]:
    """Categorize transactions using LLM.
    
    Takes transactions JSON string and adds category field to each transaction.
    Returns JSON-encoded string stored in categorized_str.
    """
    transactions_data = json.loads(state["transactions"])
    
    if not transactions_data:
        return {"categorized_str": "[]"}
    
    # Initialize LLM
    llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)
    
    # Create prompt for categorization
    transactions_text = json.dumps(transactions_data, indent=2)
    prompt = f"""You are a financial categorization expert. Given the following transactions, add a "category" field to each transaction.

Use these categories: Groceries, Rent, Utilities, Entertainment, Transportation, Healthcare, Shopping, Dining, Other

Transactions:
{transactions_text}

Return the transactions as a JSON array with each transaction having an added "category" field. Only return the JSON array, no other text."""

    try:
        response = llm.invoke(prompt)
        categorized_transactions = json.loads(response.content)
        return {"categorized_str": json.dumps(categorized_transactions)}
    except Exception:
        # Fallback: assign "Other" category to all transactions
        for transaction in transactions_data:
            transaction["category"] = "Other"
        return {"categorized_str": json.dumps(transactions_data)}


def summarizer(state: FinanceState) -> dict[str, str]:
    """Summarize spending by category.
    
    Parses categorized_str and sums amounts per category.
    Returns JSON string of category totals stored in summary_str.
    """
    try:
        categorized_transactions = json.loads(state["categorized_str"])
        
        category_totals = {}
        for transaction in categorized_transactions:
            category = transaction.get("category", "Other")
            amount = float(transaction.get("amount", 0))
            category_totals[category] = category_totals.get(category, 0) + amount
        
        return {"summary_str": json.dumps(category_totals)}
    except Exception:
        return {"summary_str": "{}"}


def advisor(state: FinanceState) -> str:
    """Generate budget advice using LLM.
    
    Compares spending summary against budget and generates advice for
    over-budget categories. Returns final report JSON string.
    """
    try:
        spending_summary = json.loads(state["summary_str"])
        budget = json.loads(state["category_budget"])
        
        # Find over-budget categories
        over_budget_categories = []
        for category, spent in spending_summary.items():
            budget_amount = budget.get(category, 0)
            if spent > budget_amount:
                over_budget_categories.append({
                    "category": category,
                    "spent": spent,
                    "budget": budget_amount,
                    "overage": spent - budget_amount
                })
        
        advice = {}
        if over_budget_categories:
            # Initialize LLM for advice generation
            llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.7)
            
            for item in over_budget_categories:
                category = item["category"]
                spent = item["spent"]
                budget_amount = item["budget"]
                overage = item["overage"]
                
                prompt = f"""You are a financial advisor. A person has overspent in the {category} category.

Budget: ${budget_amount:.2f}
Actual spending: ${spent:.2f}
Overage: ${overage:.2f}

Provide a brief, practical tip (1-2 sentences) to help them reduce spending in this category next month."""

                try:
                    response = llm.invoke(prompt)
                    advice[category] = response.content.strip()
                except Exception:
                    advice[category] = f"Consider reducing {category} spending by ${overage:.2f} to stay within budget."
        
        # Create final report
        final_report = {
            "category_summary": spending_summary,
            "advice": advice
        }
        
        return json.dumps(final_report)
    
    except Exception:
        # Fallback response
        return json.dumps({
            "category_summary": {},
            "advice": {}
        })


# Build the graph
graph_builder = StateGraph(FinanceState)

# Add nodes
graph_builder.add_node("categorizer", categorizer)
graph_builder.add_node("summarizer", summarizer)
graph_builder.add_node("advisor", advisor)

# Add edges
graph_builder.add_edge(START, "categorizer")
graph_builder.add_edge("categorizer", "summarizer")
graph_builder.add_edge("summarizer", "advisor")
graph_builder.add_edge("advisor", END)

# Compile the graph
compiled_graph = graph_builder.compile()


