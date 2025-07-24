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


def advisor(state: State) -> dict:
    """
    Advisor node that compares spending vs budget and generates advice for over-budget categories.
    
    Input: state['summary_str'] and state['category_budget']
    Output: Final report JSON string with category_summary and advice fields
    """
    summary_str = state.get('summary_str', '{}')
    budget_str = state.get('category_budget', '{}')
    
    # Parse the summary and budget JSON strings
    try:
        category_summary = json.loads(summary_str)
    except json.JSONDecodeError:
        category_summary = {}
    
    try:
        category_budget = json.loads(budget_str)
    except json.JSONDecodeError:
        category_budget = {}
    
    # Initialize advice dictionary
    advice = {}
    
    # Compare spending vs budget and generate advice for over-budget categories
    for category, budget_amount in category_budget.items():
        spent_amount = category_summary.get(category, 0.0)
        
        if spent_amount > budget_amount:
            # Category is over budget, generate advice using LLM
            over_amount = spent_amount - budget_amount
            prompt = f"""You are a financial advisor. A person has overspent in the "{category}" category.

Budget: ${budget_amount:.2f}
Actual spending: ${spent_amount:.2f}
Over budget by: ${over_amount:.2f}

Please provide a brief, practical tip (1-2 sentences) to help them reduce spending in the {category} category. Be specific and actionable."""

            try:
                response = llm.invoke([HumanMessage(content=prompt)])
                advice[category] = response.content.strip()
            except Exception:
                # Fallback advice if LLM fails
                advice[category] = f"You're over budget by ${over_amount:.2f}. Consider reducing {category.lower()} expenses."
        else:
            # Category is within budget, provide positive reinforcement
            remaining = budget_amount - spent_amount
            if remaining > 0:
                advice[category] = f"Great job! You're ${remaining:.2f} under budget in {category}."
            else:
                advice[category] = f"Perfect! You've stayed exactly on budget for {category}."
    
    # Create final report
    final_report = {
        "category_summary": category_summary,
        "advice": advice
    }
    
    # Return the final report as JSON string stored in a state field
    return {"final_report": json.dumps(final_report)}


# Add the categorizer node to the graph
graph_builder.add_node("categorizer", categorizer)

# Add the summarizer node to the graph
graph_builder.add_node("summarizer", summarizer)


# Add the advisor node to the graph
graph_builder.add_node("advisor", advisor)

# Add edges to connect the nodes in the proper sequence
# START -> categorizer -> summarizer -> advisor -> END
graph_builder.add_edge(START, "categorizer")
graph_builder.add_edge("categorizer", "summarizer")
graph_builder.add_edge("summarizer", "advisor")
graph_builder.add_edge("advisor", END)

# Compile the graph
graph = graph_builder.compile()

# Export the compiled graph as required
compiled_graph = graph

