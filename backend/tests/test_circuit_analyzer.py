#!/usr/bin/env python3
"""
Individual Test Script for CircuitAnalyzer

This script tests the CircuitAnalyzer feature in isolation,
providing focused validation of its core functionality.
"""

import sys
import os
import json
from datetime import datetime

# Add parent directory to path to import core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_circuit_analyzer():
    """Test CircuitAnalyzer functionality in isolation"""
    print("Testing CircuitAnalyzer - Individual Feature Test")
    print("=" * 60)
    
    try:
        # Import required modules
        print("Importing CircuitAnalyzer...")
        from core.circuit_analyzer import CircuitAnalyzer
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        print("CircuitAnalyzer imported successfully")
        
        # Load model and tokenizer
        print("\nLoading GPT-2 Large model...")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
        model = GPT2LMHeadModel.from_pretrained('gpt2-large')
        model.eval()
        print("GPT-2 Large loaded successfully")
        
        # Initialize CircuitAnalyzer
        print("\nInitializing CircuitAnalyzer...")
        analyzer = CircuitAnalyzer(model, tokenizer)
        print("CircuitAnalyzer initialized successfully")
        
        # Test 1: Trace information flow
        print("\nTest 1: Tracing information flow")
        print("-" * 50)
        flow_result = analyzer.trace_information_flow(
            input_tokens=["The", "cat", "is"],
            target_token="happy",
            max_depth=3
        )
        
        if 'error' in flow_result:
            print(f"Test 1 failed: {flow_result['error']}")
            return False
        else:
            print(f"Test 1 passed: Traced flow through {len(flow_result.get('layers_analyzed', []))} layers")
            print(f"   Critical paths found: {len(flow_result.get('critical_paths', []))}")
            print(f"   Information flow score: {flow_result.get('flow_score', 'N/A')}")
        
        # Test 2: Find circuit components
        print("\nTest 2: Finding circuit components")
        print("-" * 50)
        components_result = analyzer.find_circuit_components(
            concept_tokens=["The", "weather", "is", "sunny"],
            layer=6
        )
        
        if 'error' in components_result:
            print(f"Test 2 failed: {components_result['error']}")
            return False
        else:
            print(f"Test 2 passed: Found {len(components_result.get('circuit_components', []))} components")
            print(f"   MLP neurons: {len(components_result.get('mlp_neurons', []))}")
            print(f"   Attention heads: {len(components_result.get('attention_heads', []))}")
        
        # Test 3: Test export functionality
        print("\nTest 3: Testing export functionality")
        print("-" * 50)
        export_result = analyzer.export_circuit_analysis(
            analysis={
                'information_flow': flow_result,
                'circuit_components': components_result
            },
            filename=f"circuit_analyzer_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        if isinstance(export_result, str):
            print(f"Test 3 passed: Analysis exported to {export_result}")
        else:
            print(f"Test 3 failed: Unexpected export result type: {type(export_result)}")
            return False
        
        print("\n" + "=" * 60)
        print("CircuitAnalyzer Individual Test: ALL TESTS PASSED")
        print("=" * 60)
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_circuit_analyzer()
    if success:
        print("\nCircuitAnalyzer test completed successfully!")
    else:
        print("\nCircuitAnalyzer test failed!")
        sys.exit(1)
