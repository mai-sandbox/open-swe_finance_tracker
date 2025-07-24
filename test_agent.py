#!/usr/bin/env python3
"""Simple test to verify agent.py implementation"""

try:
    from agent import compiled_graph, State
    print("✅ Successfully imported compiled_graph and State")
    
    # Test with empty input to verify default state handling
    result = compiled_graph.invoke({})
    print("✅ Graph executed successfully with empty input")
    print(f"Result type: {type(result)}")
    print(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
    
except Exception as e:
    print(f"❌ Error: {e}")
