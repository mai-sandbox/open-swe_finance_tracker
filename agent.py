
import json
from typing import Annotated
from langchain_anthropic import ChatAnthropic

from typing_extensions import TypedDict

from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]
    transactions: str
    category_budget: str
    categorized_str: str
    summary_str: str


default_transactions = '[{"date": "2024-01-05", "description": "Whole Foods Market", "amount": 125.50}, {"date": "2024-01-12", "description": "Safeway Grocery", "amount": 99.50}, {"date": "2024-01-01", "description": "Monthly Rent Payment", "amount": 1005.00}, {"date": "2024-01-15", "description": "PG&E Electric Bill", "amount": 85.00}, {"date": "2024-01-20", "description": "Water Utility", "amount": 65.00}, {"date": "2024-01-08", "description": "Netflix Subscription", "amount": 15.99}, {"date": "2024-01-14", "description": "Movie Theater Tickets", "amount": 45.00}, {"date": "2024-01-22", "description": "Concert Tickets", "amount": 89.01}]'
default_category_budget = '{"Groceries":200,"Rent":1000,"Utilities":150,"Entertainment":100}'


def categorizer(state: State):
    transactions = state.get("transactions", default_transactions)
    prompt = f"""Categorize these transactions into: Groceries, Rent, Utilities, Entertainment, or Other.
Return valid JSON with original fields plus 'category' field.
Transactions: {transactions}"""
    llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
    response = llm.invoke(prompt)
    try:
        categorized_json = json.loads(response.content)
        return {"categorized_str": json.dumps(categorized_json)}
    except json.JSONDecodeError:
        return {"categorized_str": transactions}


def summarizer(state: State):
    categorized_str = state.get("categorized_str")
    try:
        transactions = json.loads(categorized_str)
        summary = {}
        for t in transactions:
            category = t.get("category")
            amount = t.get("amount")
            summary[category] = summary.get(category, 0) + amount
        return {"summary_str": json.dumps(summary)}
    except (json.JSONDecodeError, TypeError):
        return {"summary_str": "{}"}




