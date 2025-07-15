import json
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    transactions: str
    category_budget: str
    categorized_str: str
    summary_str: str


# Initialize LLM
llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")


def categorizer(state: State) -> dict:
    """
    Categorize transactions using LLM.
    Takes transactions JSON string and returns categorized transactions.
    """
    transactions_str = state["transactions"]
    
    # If no transactions, return empty categorized list
    if not transactions_str or transactions_str == "[]":
        return {"categorized_str": "[]"}
    
    # Parse transactions
    try:
        transactions = json.loads(transactions_str)
    except json.JSONDecodeError:
        return {"categorized_str": "[]"}
    
    # Create prompt for LLM to categorize transactions
    prompt = f"""You are a financial categorization assistant. Given a list of transactions, assign each transaction a "category" field.

Available categories: Groceries, Rent, Utilities, Entertainment, Other

Transactions to categorize:
{json.dumps(transactions, indent=2)}

Return the same list of transactions but with a "category" field added to each transaction. Return only valid JSON, no additional text.

Example format:
[
  {{"date": "2024-01-01", "description": "Grocery Store", "amount": 50.0, "category": "Groceries"}},
  {{"date": "2024-01-02", "description": "Rent Payment", "amount": 1000.0, "category": "Rent"}}
]"""
    
    # Get LLM response
    response = llm.invoke(prompt)
    categorized_transactions = response.content.strip()
    
    # Validate JSON response
    try:
        json.loads(categorized_transactions)
    except json.JSONDecodeError:
        # Fallback: assign "Other" category to all transactions
        for transaction in transactions:
            transaction["category"] = "Other"
        categorized_transactions = json.dumps(transactions)
    
    return {"categorized_str": categorized_transactions}


def summarizer(state: State) -> dict:
    """
    Summarize categorized transactions by summing amounts per category.
    """
    categorized_str = state["categorized_str"]
    
    # Parse categorized transactions
    try:
        transactions = json.loads(categorized_str)
    except json.JSONDecodeError:
        return {"summary_str": "{}"}
    
    # Sum amounts by category
    category_totals = {}
    for transaction in transactions:
        category = transaction.get("category", "Other")
        amount = transaction.get("amount", 0)
        category_totals[category] = category_totals.get(category, 0) + amount
    
    return {"summary_str": json.dumps(category_totals)}


def advisor(state: State) -> dict:
    """
    Compare spending vs budget and provide advice for over-budget categories.
    Returns final report as JSON string.
    """
    summary_str = state["summary_str"]
    budget_str = state["category_budget"]
    
    # Parse data
    try:
        category_summary = json.loads(summary_str)
        budget = json.loads(budget_str)
    except json.JSONDecodeError:
        return {"summary_str": json.dumps({"category_summary": {}, "advice": {}})}
    
    # Find over-budget categories and generate advice
    advice = {}
    for category, spent in category_summary.items():
        budget_amount = budget.get(category, 0)
        if spent > budget_amount:
            prompt = f"""You are a financial advisor. A person has spent ${spent:.2f} on {category} but their budget was only ${budget_amount:.2f}. 
They are ${spent - budget_amount:.2f} over budget in this category.

Provide a brief, helpful tip (1-2 sentences) on how they can reduce spending in the {category} category. Be practical and specific."""
            
            response = llm.invoke(prompt)
            advice[category] = response.content.strip()
    
    # Create final report
    final_report = {
        "category_summary": category_summary,
        "advice": advice
    }
    
    return {"summary_str": json.dumps(final_report)}


# Create StateGraph
graph_builder = StateGraph(State)

# Add nodes
graph_builder.add_node("categorizer", categorizer)
graph_builder.add_node("summarizer", summarizer)
graph_builder.add_node("advisor", advisor)

# Add edges to create linear flow: START -> categorizer -> summarizer -> advisor -> END
graph_builder.add_edge(START, "categorizer")
graph_builder.add_edge("categorizer", "summarizer")
graph_builder.add_edge("summarizer", "advisor")
graph_builder.add_edge("advisor", END)

# Compile the graph
graph = graph_builder.compile()

# Export compiled graph for evaluation
compiled_graph = graph

