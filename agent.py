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
    
    # If no transactions, return empty categorized list
    if transactions == "[]":
        return {"categorized_str": "[]"}
    
    try:
        # Parse the transactions JSON string
        transactions_list = json.loads(transactions)
        
        # If empty list, return empty categorized list
        if not transactions_list:
            return {"categorized_str": "[]"}
        
        # Create prompt for LLM to categorize transactions
        prompt = f"""You are a financial categorization assistant. Given a list of transactions, assign each transaction to one of these categories: Groceries, Rent, Utilities, Entertainment, or Other.

Transactions to categorize:
{json.dumps(transactions_list, indent=2)}

For each transaction, add a "category" field with the most appropriate category. Return the complete list as valid JSON with all original fields plus the new "category" field.

Example format:
[{{"date": "2024-01-01", "description": "Grocery Store", "amount": 50.0, "category": "Groceries"}}]

Return only the JSON array, no additional text."""

        # Use LLM to categorize transactions
        response = llm.invoke(prompt)
        categorized_transactions = response.content.strip()
        
        # Validate that the response is valid JSON
        json.loads(categorized_transactions)  # This will raise an exception if invalid
        
        return {"categorized_str": categorized_transactions}
        
    except (json.JSONDecodeError, Exception) as e:
        # If there's an error, return the original transactions without categories
        return {"categorized_str": transactions}

def summarizer(state: State) -> dict:
    """Summarizer node - parses categorized transactions and sums amounts per category"""
    # Get categorized transactions from state
    categorized_str = state.get("categorized_str", "[]")
    
    try:
        # Parse the categorized transactions JSON string
        categorized_transactions = json.loads(categorized_str)
        
        # Initialize category totals dictionary
        category_totals = {}
        
        # Sum amounts per category
        for transaction in categorized_transactions:
            # Get category and amount from transaction
            category = transaction.get("category", "Other")
            amount = transaction.get("amount", 0)
            
            # Convert amount to float if it's a string
            if isinstance(amount, str):
                try:
                    amount = float(amount)
                except ValueError:
                    amount = 0
            
            # Add to category total
            category_totals[category] = category_totals.get(category, 0) + amount
        
        # Return category totals as JSON string
        return {"summary_str": json.dumps(category_totals)}
        
    except (json.JSONDecodeError, Exception) as e:
        # If there's an error, return empty summary
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




