#!/usr/bin/env python3
"""
Test script to verify the agent.py implementation works correctly.
This will be deleted after testing.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from langchain_core.messages import HumanMessage
    from agent import app
    
    # Test the agent with evaluator format input
    test_input = {"messages": [HumanMessage(content="Please analyze my finances")]}
    
    print("Testing agent with evaluator input format...")
    result = app.invoke(test_input)
    
    print("✓ Agent executed successfully")
    print(f"✓ Result type: {type(result)}")
    print(f"✓ Result keys: {list(result.keys())}")
    
    # Check if summary_str exists and is valid JSON
    if 'summary_str' in result:
        import json
        try:
            summary_data = json.loads(result['summary_str'])
            print(f"✓ summary_str is valid JSON: {type(summary_data)}")
            print(f"✓ summary_str keys: {list(summary_data.keys())}")
            
            # Check expected structure
            if 'category_summary' in summary_data and 'advice' in summary_data:
                print("✓ Final output has correct structure (category_summary + advice)")
            else:
                print("⚠ Final output missing expected keys")
                
        except json.JSONDecodeError as e:
            print(f"✗ summary_str is not valid JSON: {e}")
    else:
        print("✗ summary_str not found in result")
    
    print("\nTest completed successfully!")
    
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
