#!/usr/bin/env python3
"""
Individual Test Script for FeatureAnalyzer

This script tests the FeatureAnalyzer feature in isolation,
providing focused validation of its core functionality.
"""

import sys
import os
import json
from datetime import datetime

# Add parent directory to path to import core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_feature_analyzer():
    """Test FeatureAnalyzer functionality in isolation"""
    print("Testing FeatureAnalyzer - Individual Feature Test")
    print("=" * 60)
    
    try:
        # Import required modules
        print("Importing FeatureAnalyzer...")
        from core.feature_analyzer import FeatureAnalyzer
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        print("FeatureAnalyzer imported successfully")
        
        # Load model and tokenizer
        print("\nLoading GPT-2 Large model...")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
        model = GPT2LMHeadModel.from_pretrained('gpt2-large')
        model.eval()
        print("GPT-2 Large loaded successfully")
        
        # Initialize FeatureAnalyzer
        print("\nInitializing FeatureAnalyzer...")
        analyzer = FeatureAnalyzer(model, tokenizer)
        print("FeatureAnalyzer initialized successfully")
        
        # Test 1: Find feature neurons
        print("\nTest 1: Finding feature neurons for 'happy' concept")
        print("-" * 50)
        happy_neurons = analyzer.find_feature_neurons(
            concept_tokens=["happy", "joy", "smile"], 
            layer=6, 
            top_k=5
        )
        
        if 'error' in happy_neurons:
            print(f"Test 1 failed: {happy_neurons['error']}")
            return False
        else:
            print(f"Test 1 passed: Found {len(happy_neurons.get('top_neurons', []))} neurons")
            print(f"   Top neuron: {happy_neurons.get('top_neurons', [])[0] if happy_neurons.get('top_neurons') else 'None'}")
        
        # Test 2: Find concept-distinguishing neurons
        print("\nTest 2: Finding concept-distinguishing neurons")
        print("-" * 50)
        concept_pairs = [("happy", "sad"), ("good", "bad"), ("love", "hate")]
        concept_neurons = analyzer.find_concept_neurons(
            concept_pairs=concept_pairs, 
            layer=6, 
            top_k=5
        )
        
        if 'error' in concept_neurons:
            print(f"Test 2 failed: {concept_neurons['error']}")
            return False
        else:
            print(f"Test 2 passed: Found {len(concept_neurons.get('concept_distinguishing_neurons', []))} neurons")
            print(f"   Strongest distinction: {concept_neurons.get('analysis_summary', {}).get('strongest_distinction', 'N/A')}")
        
        # Test 3: Analyze single neuron
        print("\nTest 3: Analyzing single neuron")
        print("-" * 50)
        if happy_neurons.get('top_neurons'):
            neuron_idx = happy_neurons['top_neurons'][0]
            neuron_analysis = analyzer.analyze_single_neuron(
                neuron_idx=neuron_idx,
                layer=6,
                test_tokens=["happy", "joy", "smile", "sad", "angry"]
            )
            
            if 'error' in neuron_analysis:
                print(f"Test 3 failed: {neuron_analysis['error']}")
                return False
            else:
                print(f"Test 3 passed: Neuron {neuron_analysis['neuron_index']} analyzed")
                print(f"   Neuron type: {neuron_analysis.get('neuron_type', 'unknown')}")
                print(f"   Top activating tokens: {neuron_analysis.get('top_activating_tokens', [])}")
        else:
            print("Skipping Test 3: No neurons found in Test 1")
        
        # Test 4: Get layer representations
        print("\nTest 4: Getting layer representations")
        print("-" * 50)
        layer_reps = analyzer.get_layer_representations(
            tokens=["The", "weather", "is", "sunny", "and", "I", "feel", "happy"],
            layer=6
        )
        
        if 'error' in layer_reps:
            print(f"Test 4 failed: {layer_reps['error']}")
            return False
        else:
            print(f"Test 4 passed: Retrieved representations for {len(layer_reps.get('layer_representations', {}))} layers")
            for layer, data in layer_reps.get('layer_representations', {}).items():
                print(f"   Layer {layer}: {len(data.get('mlp_activations', []))} MLP activations")
        
        # Test 5: Export analysis
        print("\nTest 5: Exporting analysis results")
        print("-" * 50)
        export_result = analyzer.export_analysis(
            analysis={
                'feature_neurons': happy_neurons,
                'concept_neurons': concept_neurons,
                'layer_representations': layer_reps
            },
            filename=f"feature_analyzer_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        if isinstance(export_result, str):
            print(f"Test 5 passed: Analysis exported to {export_result}")
        else:
            print(f"Test 5 failed: Unexpected export result type: {type(export_result)}")
            return False
        
        print("\n" + "=" * 60)
        print("FeatureAnalyzer Individual Test: ALL TESTS PASSED")
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
    success = test_feature_analyzer()
    if success:
        print("\nFeatureAnalyzer test completed successfully!")
    else:
        print("\nFeatureAnalyzer test failed!")
        sys.exit(1)
