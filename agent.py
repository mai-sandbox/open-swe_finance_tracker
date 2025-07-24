import json
from typing_extensions import TypedDict
from dataclasses import dataclass
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model
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

# Initialize LLM for categorizer and advisor nodes
try:
    llm = init_chat_model("gpt-3.5-turbo", model_provider="openai")
except Exception:
    # Fallback to a mock LLM for testing if OpenAI is not available
    class MockLLM:
        def invoke(self, messages):
            class MockResponse:
                def __init__(self, content):
                    self.content = content
            return MockResponse("Mock LLM response")
    llm = MockLLM()


def categorizer(state: State) -> dict:
    """
    Categorizer node that takes transactions and assigns categories using LLM.
    
    Input: state['transactions'] - JSON string of list of {date, description, amount}
    Output: JSON string of enriched list with category field -> stored in state['categorized_str']
    """
    transactions_str = state.get('transactions', '[]')
    
    # Parse the transactions JSON string
    try:
        transactions = json.loads(transactions_str)
    except json.JSONDecodeError:
        transactions = []
    
    # If no transactions, return empty categorized list
    if not transactions:
        return {"categorized_str": "[]"}
    
    # Create prompt for LLM to categorize transactions
    prompt = f"""Please categorize the following financial transactions. For each transaction, add a "category" field with an appropriate category like "Groceries", "Rent", "Utilities", "Entertainment", "Transportation", "Healthcare", etc.

Transactions: {json.dumps(transactions, indent=2)}

Return the transactions as a JSON array with the added "category" field for each transaction. Only return the JSON array, no other text."""
    
    try:
        # Use LLM to categorize transactions
        response = llm.invoke([HumanMessage(content=prompt)])
        categorized_response = response.content.strip()
        
        # Try to extract JSON from the response
        # Remove any markdown code blocks if present
        if "```json" in categorized_response:
            categorized_response = categorized_response.split("```json")[1].split("```")[0].strip()
        elif "```" in categorized_response:
            categorized_response = categorized_response.split("```")[1].split("```")[0].strip()
        
        # Validate that the response is valid JSON
        try:
            categorized_transactions = json.loads(categorized_response)
            # Ensure it's a list
            if isinstance(categorized_transactions, list):
                return {"categorized_str": json.dumps(categorized_transactions)}
            else:
                # If not a list, wrap it
                return {"categorized_str": json.dumps([categorized_transactions])}
        except json.JSONDecodeError:
            # If LLM response is not valid JSON, add default categories
            for transaction in transactions:
                if "category" not in transaction:
                    transaction["category"] = "Other"
            return {"categorized_str": json.dumps(transactions)}
    
    except Exception as e:
        # Fallback: add default categories if LLM fails
        for transaction in transactions:
            if "category" not in transaction:
                transaction["category"] = "Other"
        return {"categorized_str": json.dumps(transactions)}


def summarizer(state: State) -> dict:
    """
    Summarizer node that takes categorized transactions and sums amounts per category.
    
    Input: state['categorized_str'] - JSON string of categorized transactions
    Output: JSON string of {category: total} -> stored in state['summary_str']
    """
    categorized_str = state.get('categorized_str', '[]')
    
    # Parse the categorized transactions JSON string
    try:
        categorized_transactions = json.loads(categorized_str)
    except json.JSONDecodeError:
        categorized_transactions = []
    
    # Initialize category totals dictionary
    category_totals = {}
    
    # Sum amounts per category
    for transaction in categorized_transactions:
        if isinstance(transaction, dict):
            category = transaction.get('category', 'Other')
            amount = transaction.get('amount', 0)
            # Convert amount to float if it's a string
            try:
                amount = float(amount)
            except (ValueError, TypeError):
                amount = 0.0
            
            category_totals[category] = category_totals.get(category, 0.0) + amount
    
    # Return the category totals as JSON string
    return {"summary_str": json.dumps(category_totals)}


# Add the categorizer node to the graph
graph_builder.add_node("categorizer", categorizer)

# Add the summarizer node to the graph
graph_builder.add_node("summarizer", summarizer)


