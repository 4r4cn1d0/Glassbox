#!/usr/bin/env python3
"""
Test script for InterventionTester - Feature #3

This script tests the intervention testing capabilities for mechanistic interpretability,
validating the implementation of attention head ablation, MLP neuron ablation,
and importance testing functionalities.
"""

import sys
import os
import json
from datetime import datetime

# Add parent directory to path to import core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_intervention_tester():
    """Test the InterventionTester functionality"""
    print("Testing InterventionTester - Feature #3")
    print("=" * 60)

    try:
        # Import required modules
        print("Importing modules...")
        from core.intervention_tester import InterventionTester
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        print("Modules imported successfully")

        # Load model and tokenizer
        print("\nLoading GPT-2 Large model...")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
        model = GPT2LMHeadModel.from_pretrained('gpt2-large')
        model.eval()
        print("GPT-2 Large loaded successfully")

        # Initialize InterventionTester
        print("\nInitializing InterventionTester...")
        intervention_tester = InterventionTester(model, tokenizer)
        print("InterventionTester initialized successfully")

        # Test prompts
        test_prompts = [
            "The cat is",
            "I feel happy because",
            "The weather is sunny and"
        ]

        # Test 1: Attention Head Ablation
        print("\nTest 1: Attention Head Ablation")
        print("-" * 40)
        try:
            result = intervention_tester.ablate_attention_head(
                layer_idx=6,
                head_idx=2,
                test_prompts=test_prompts,
                max_length=30
            )
            
            if 'error' not in result:
                impact = result['impact_metrics']
                print(f"Attention head ablation successful!")
                print(f"   Outputs changed: {impact['outputs_changed']}/{impact['total_prompts']}")
                print(f"   Change rate: {impact['change_rate']:.2%}")
                print(f"   Avg token change: {impact.get('avg_token_change', 0):.1f}")
            else:
                print(f"Attention head ablation failed: {result['error']}")
                return False
                
        except Exception as e:
            print(f"Attention head ablation test failed: {e}")
            return False

        # Test 2: MLP Neuron Ablation
        print("\nTest 2: MLP Neuron Ablation")
        print("-" * 40)
        try:
            result = intervention_tester.ablate_mlp_neuron(
                layer_idx=6,
                neuron_idx=100,
                test_prompts=test_prompts,
                max_length=30
            )
            
            if 'error' not in result:
                impact = result['impact_metrics']
                print(f"MLP neuron ablation successful!")
                print(f"   Outputs changed: {impact['outputs_changed']}/{impact['total_prompts']}")
                print(f"   Change rate: {impact['change_rate']:.2%}")
                print(f"   Avg semantic change: {impact.get('avg_semantic_change', 0):.1f}")
            else:
                print(f"MLP neuron ablation failed: {result['error']}")
                return False
                
        except Exception as e:
            print(f"MLP neuron ablation test failed: {e}")
            return False

        # Test 3: Attention Head Importance Testing
        print("\nTest 3: Attention Head Importance Testing")
        print("-" * 40)
        try:
            # Test only first 3 heads to save time
            result = intervention_tester.test_attention_head_importance(
                layer_idx=6,
                test_prompts=test_prompts[:2],  # Use fewer prompts for speed
                max_length=25
            )
            
            if 'error' not in result:
                print(f"Attention head importance testing successful!")
                print(f"   Total heads tested: {result['total_heads']}")
                print(f"   Most important: {result['most_important_head']}")
                print(f"   Least important: {result['least_important_head']}")
                
                # Show importance scores for first few heads
                for i in range(min(3, result['total_heads'])):
                    head_key = f"head_{i}"
                    if head_key in result['head_importance']:
                        head_data = result['head_importance'][head_key]
                        print(f"   {head_key}: {head_data.get('importance_score', 0):.1f}% importance")
            else:
                print(f"Attention head importance testing failed: {result['error']}")
                return False
                
        except Exception as e:
            print(f"Attention head importance testing failed: {e}")
            return False

        # Test 4: Neuron Importance Testing
        print("\nTest 4: Neuron Importance Testing")
        print("-" * 40)
        try:
            # Test only first 5 neurons to save time
            result = intervention_tester.test_neuron_importance(
                layer_idx=6,
                neuron_indices=[50, 100, 150, 200, 250],
                test_prompts=test_prompts[:2],  # Use fewer prompts for speed
                max_length=25
            )
            
            if 'error' not in result:
                print(f"Neuron importance testing successful!")
                print(f"   Total neurons tested: {result['total_neurons_tested']}")
                print(f"   Most important: {result['most_important_neuron']}")
                print(f"   Least important: {result['least_important_neuron']}")
                
                # Show importance scores for first few neurons
                for i, neuron_idx in enumerate([50, 100, 150]):
                    neuron_key = f"neuron_{neuron_idx}"
                    if neuron_key in result['neuron_importance']:
                        neuron_data = result['neuron_importance'][neuron_key]
                        print(f"   {neuron_key}: {neuron_data.get('importance_score', 0):.1f}% importance")
            else:
                print(f"Neuron importance testing failed: {result['error']}")
                return False
                
        except Exception as e:
            print(f"Neuron importance testing failed: {e}")
            return False

        # Test 5: Export Functionality
        print("\nTest 5: Export Functionality")
        print("-" * 40)
        try:
            # Test export with a simple result
            test_result = {
                'intervention_type': 'test_intervention',
                'test_data': 'sample_data',
                'timestamp': datetime.now().isoformat()
            }
            
            export_path = intervention_tester.export_intervention_results(test_result, "test_intervention_export.json")
            
            if export_path and os.path.exists(export_path):
                print(f"Export functionality working!")
                print(f"   Exported to: {export_path}")
                
                # Clean up test file
                os.remove(export_path)
                print("   Test file cleaned up")
            else:
                print("Export functionality failed")
                return False
                
        except Exception as e:
            print(f"Export functionality test failed: {e}")
            return False

        print("\nAll InterventionTester tests passed!")
        print("Feature #3 (Intervention Testing) is working correctly!")
        
        # Print comprehensive summary
        print(f"\nInterventionTester Summary:")
        print(f"   Model: GPT-2 Large")
        print(f"   Layers: {intervention_tester.layer_count}")
        print(f"   Hidden Size: {intervention_tester.hidden_size}")
        print(f"   Attention Heads: {intervention_tester.num_heads}")
        print(f"   Attention Head Ablation: Working")
        print(f"   MLP Neuron Ablation: Working")
        print(f"   Head Importance Testing: Working")
        print(f"   Neuron Importance Testing: Working")
        print(f"   Export Functionality: Working")
        
        return True

    except Exception as e:
        print(f"InterventionTester test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the InterventionTester test suite"""
    print("InterventionTester Test Suite")
    print("=" * 60)
    
    success = test_intervention_tester()
    
    if success:
        print("\nINTERVENTION TESTER IS READY!")
        print("This is a GAME CHANGER for mechanistic interpretability!")
        print("\nNext steps:")
        print("   1. Integrate into the main API")
        print("   2. Move to Safety-Specific Metrics (Feature #4)")
        print("   3. Test with real research questions")
        print("\nCurrent Features:")
        print("   Feature Visualization (FeatureAnalyzer)")
        print("   Circuit Analysis (CircuitAnalyzer)")
        print("   Intervention Testing (InterventionTester)")
        print("   Safety-Specific Metrics (Next)")
    else:
        print("\nSome tests failed. Check the errors above.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
