"""
Personal Finance Tracker Agent using LangGraph

This module implements a LangGraph-based agent for personal finance tracking
with three main nodes: categorizer, summarizer, and advisor.
"""

import json
import os
from typing import Dict, Any, TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic

# Load environment variables
load_dotenv()


class FinanceState(TypedDict):
    """State schema for the Personal Finance Tracker."""
    transactions: str
    category_budget: str
    categorized_str: str
    summary_str: str


def categorizer(state: FinanceState) -> Dict[str, Any]:
    """
    Categorize transactions using LLM.
    
    Args:
        state: Current state containing transactions JSON string
        
    Returns:
        Dictionary with categorized_str field containing JSON string of categorized transactions
    """
    try:
        transactions_str = state.get("transactions", "[]")
        
        # Parse transactions
        transactions = json.loads(transactions_str)
        
        if not transactions:
            return {"categorized_str": "[]"}
        
        # Initialize LLM
        llm = ChatAnthropic(
            model="claude-3-haiku-20240307",
            api_key=os.getenv("ANTHROPIC_API_KEY", "")
        )
        
        # Create prompt for categorization
        prompt = f"""
        You are a financial categorization expert. Given a list of transactions, assign each transaction an appropriate category.
        
        Available categories: Groceries, Rent, Utilities, Entertainment, Transportation, Healthcare, Shopping, Dining, Other
        
        Transactions to categorize:
        {json.dumps(transactions, indent=2)}
        
        Return ONLY a JSON array where each transaction has an added "category" field. Do not include any other text or explanation.
        
        Example format:
        [
            {{"date": "2024-01-01", "description": "Grocery Store", "amount": 50.0, "category": "Groceries"}},
            {{"date": "2024-01-02", "description": "Electric Bill", "amount": 120.0, "category": "Utilities"}}
        ]
        """
        
        response = llm.invoke(prompt)
        categorized_transactions = response.content.strip()
        
        # Validate JSON format
        try:
            json.loads(categorized_transactions)
        except json.JSONDecodeError:
            # Fallback: assign "Other" category to all transactions
            for transaction in transactions:
                transaction["category"] = "Other"
            categorized_transactions = json.dumps(transactions)
        
        return {"categorized_str": categorized_transactions}
        
    except Exception as e:
        # Error handling: return empty categorized transactions
        return {"categorized_str": "[]"}


def summarizer(state: FinanceState) -> Dict[str, Any]:
    """
    Summarize categorized transactions by calculating totals per category.
    
    Args:
        state: Current state containing categorized_str JSON string
        
    Returns:
        Dictionary with summary_str field containing JSON string of category totals
    """
    try:
        categorized_str = state.get("categorized_str", "[]")
        
        # Parse categorized transactions
        categorized_transactions = json.loads(categorized_str)
        
        # Calculate totals per category
        category_totals: Dict[str, float] = {}
        
        for transaction in categorized_transactions:
            category = transaction.get("category", "Other")
            amount = float(transaction.get("amount", 0))
            
            if category in category_totals:
                category_totals[category] += amount
            else:
                category_totals[category] = amount
        
        # Convert to JSON string
        summary_json = json.dumps(category_totals)
        
        return {"summary_str": summary_json}
        
    except Exception as e:
        # Error handling: return empty summary
        return {"summary_str": "{}"}


def advisor(state: FinanceState) -> str:
    """
    Generate financial advice by comparing spending vs budget.
    
    Args:
        state: Current state containing summary_str and category_budget
        
    Returns:
        JSON string containing category summary and advice for over-budget categories
    """
    try:
        summary_str = state.get("summary_str", "{}")
        budget_str = state.get("category_budget", "{}")
        
        # Parse summary and budget
        category_summary = json.loads(summary_str)
        category_budget = json.loads(budget_str)
        
        # Find over-budget categories
        over_budget_categories = []
        advice = {}
        
        for category, spent in category_summary.items():
            budget_amount = category_budget.get(category, 0)
            if spent > budget_amount:
                over_budget_categories.append(category)
        
        # Generate advice for over-budget categories using LLM
        if over_budget_categories:
            try:
                llm = ChatAnthropic(
                    model="claude-3-haiku-20240307",
                    api_key=os.getenv("ANTHROPIC_API_KEY", "")
                )
                
                for category in over_budget_categories:
                    spent = category_summary[category]
                    budget = category_budget.get(category, 0)
                    overage = spent - budget
                    
                    prompt = f"""
                    You are a financial advisor. A person has overspent in the {category} category.
                    
                    Budget: ${budget:.2f}
                    Actual spending: ${spent:.2f}
                    Overage: ${overage:.2f}
                    
                    Provide a brief, practical tip (1-2 sentences) to help them reduce spending in this category next month.
                    Return only the advice text, no additional formatting or explanation.
                    """
                    
                    response = llm.invoke(prompt)
                    advice[category] = response.content.strip()
                    
            except Exception as e:
                # Fallback advice for over-budget categories
                for category in over_budget_categories:
                    advice[category] = f"Consider reducing {category} expenses to stay within budget."
        
        # Create final report
        final_report = {
            "category_summary": category_summary,
            "advice": advice
        }
        
        return json.dumps(final_report)
        
    except Exception as e:
        # Error handling: return basic report structure
        return json.dumps({"category_summary": {}, "advice": {}})


# Create the state graph
graph_builder = StateGraph(FinanceState)

# Add nodes
graph_builder.add_node("categorizer", categorizer)
graph_builder.add_node("summarizer", summarizer)
graph_builder.add_node("advisor", advisor)

# Add edges to define the workflow
graph_builder.add_edge(START, "categorizer")
graph_builder.add_edge("categorizer", "summarizer")
graph_builder.add_edge("summarizer", "advisor")
graph_builder.add_edge("advisor", END)

# Compile the graph
compiled_graph = graph_builder.compile()

