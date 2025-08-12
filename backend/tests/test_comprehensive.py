#!/usr/bin/env python3
"""
Comprehensive Test Script for All Glassbox Features

This script tests the complete integration of all four mechanistic interpretability features
working together in a coordinated workflow, simulating real-world research scenarios.
"""

import sys
import os
import json
from datetime import datetime

# Add parent directory to path to import core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_comprehensive_workflow():
    """Test all features working together in a comprehensive workflow"""
    print("Testing Glassbox - Comprehensive Integration Test")
    print("=" * 70)
    
    try:
        # Import all required modules
        print("Importing all core modules...")
        from core.feature_analyzer import FeatureAnalyzer
        from core.circuit_analyzer import CircuitAnalyzer
        from core.intervention_tester import InterventionTester
        from core.safety_analyzer import SafetyAnalyzer
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        print("All modules imported successfully")
        
        # Load model and tokenizer
        print("\nLoading GPT-2 Large model...")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
        model = GPT2LMHeadModel.from_pretrained('gpt2-large')
        model.eval()
        print("GPT-2 Large loaded successfully")
        
        # Initialize all analyzers
        print("\nInitializing all analyzers...")
        feature_analyzer = FeatureAnalyzer(model, tokenizer)
        circuit_analyzer = CircuitAnalyzer(model, tokenizer)
        intervention_tester = InterventionTester(model, tokenizer)
        safety_analyzer = SafetyAnalyzer(model, tokenizer)
        print("All analyzers initialized successfully")
        
        # Test prompts for comprehensive analysis
        test_prompts = [
            "The cat is",
            "I feel happy because",
            "The weather is sunny and"
        ]
        
        # Comprehensive Test 1: Feature Analysis + Circuit Analysis
        print("\nComprehensive Test 1: Feature + Circuit Analysis")
        print("-" * 60)
        try:
            # Find feature neurons for emotional concepts
            feature_result = feature_analyzer.find_feature_neurons(
                concept_tokens=["happy", "joy", "smile"],
                layer=6,
                top_k=5
            )
            
            if 'error' in feature_result:
                print(f"Feature analysis failed: {feature_result['error']}")
                return False
            
            print(f"Feature analysis successful: Found {len(feature_result.get('top_neurons', []))} neurons")
            
            # Use found neurons to trace information flow
            if feature_result.get('top_neurons'):
                top_neuron = feature_result['top_neurons'][0]
                circuit_result = circuit_analyzer.trace_information_flow(
                    input_tokens=["The", "cat", "is"],
                    target_token="happy",
                    max_depth=3
                )
                
                if 'error' not in circuit_result:
                    print(f"Circuit analysis successful: Traced through {len(circuit_result.get('layers_analyzed', []))} layers")
                else:
                    print(f"Circuit analysis failed: {circuit_result['error']}")
                    return False
            else:
                print("No feature neurons found for circuit analysis")
                return False
                
        except Exception as e:
            print(f"Comprehensive Test 1 failed: {e}")
            return False
        
        # Comprehensive Test 2: Intervention Testing + Safety Analysis
        print("\nComprehensive Test 2: Intervention + Safety Analysis")
        print("-" * 60)
        try:
            # Test intervention on attention heads
            intervention_result = intervention_tester.test_attention_head_importance(
                layer_idx=6,
                test_prompts=["The cat is happy and playful"]
            )
            
            if 'error' in intervention_result:
                print(f"Intervention testing failed: {intervention_result['error']}")
                return False
            
            print(f"Intervention testing successful: Tested {len(intervention_result.get('head_importance', []))} attention heads")
            
            # Use intervention results for safety analysis
            safety_result = safety_analyzer.run_safety_audit(
                test_prompts=test_prompts,
                max_length=10
            )
            
            if 'error' not in safety_result:
                print(f"Safety analysis successful: Completed audit of {len(test_prompts)} prompts")
                print(f"   Uncertainty score: {safety_result.get('uncertainty_score', 'N/A')}")
                print(f"   Consistency score: {safety_result.get('consistency_score', 'N/A')}")
                print(f"   Deception indicators: {len(safety_result.get('deception_indicators', []))}")
            else:
                print(f"Safety analysis failed: {safety_result['error']}")
                return False
                
        except Exception as e:
            print(f"Comprehensive Test 2 failed: {e}")
            return False
        
        # Comprehensive Test 3: Cross-Feature Analysis
        print("\nComprehensive Test 3: Cross-Feature Analysis")
        print("-" * 60)
        try:
            # Analyze concept-distinguishing neurons
            concept_result = feature_analyzer.find_concept_neurons(
                concept_pairs=[("happy", "sad"), ("good", "bad")],
                layer=6,
                top_k=3
            )
            
            if 'error' in concept_result:
                print(f"Concept analysis failed: {concept_result['error']}")
                return False
            
            print(f"Concept analysis successful: Found {len(concept_result.get('concept_distinguishing_neurons', []))} neurons")
            
            # Use concept neurons for targeted intervention
            if concept_result.get('concept_distinguishing_neurons'):
                target_neuron = concept_result['concept_distinguishing_neurons'][0]
                neuron_importance = intervention_tester.test_neuron_importance(
                    layer_idx=6,
                    neuron_indices=[target_neuron],
                    test_prompts=["The weather affects my mood"]
                )
                
                if 'error' not in neuron_importance:
                    print(f"Neuron importance testing successful: Tested {len(neuron_importance.get('neuron_importance', []))} neurons")
                else:
                    print(f"Neuron importance testing failed: {neuron_importance['error']}")
                    return False
            else:
                print("No concept-distinguishing neurons found for intervention testing")
                return False
                
        except Exception as e:
            print(f"Comprehensive Test 3 failed: {e}")
            return False
        
        # Comprehensive Test 4: Export Combined Results
        print("\nComprehensive Test 4: Export Combined Results")
        print("-" * 60)
        try:
            # Collect all analysis results
            combined_results = {
                'feature_analysis': feature_result,
                'circuit_analysis': circuit_result if 'circuit_result' in locals() else {},
                'intervention_analysis': intervention_result,
                'safety_analysis': safety_result,
                'concept_analysis': concept_result,
                'neuron_importance': neuron_importance if 'neuron_importance' in locals() else {},
                'timestamp': datetime.now().isoformat(),
                'test_type': 'comprehensive_integration'
            }
            
            # Export combined results
            export_filename = f"comprehensive_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(os.path.join('..', 'exports', export_filename), 'w') as f:
                json.dump(combined_results, f, indent=2, default=str)
            
            print(f"Comprehensive results exported to: {export_filename}")
            
        except Exception as e:
            print(f"Export failed: {e}")
            return False
        
        print("\n" + "=" * 70)
        print("Comprehensive Test: ALL TESTS PASSED")
        print("Glassbox is working as a complete mechanistic interpretability platform!")
        print("=" * 70)
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
    success = test_comprehensive_workflow()
    if success:
        print("\nComprehensive test completed successfully!")
        print("All features are working together correctly!")
    else:
        print("\nComprehensive test failed!")
        sys.exit(1)
