#!/usr/bin/env python3
"""
Test script for SafetyAnalyzer - Feature #4

This script tests the safety-specific metrics for mechanistic interpretability,
validating the implementation of uncertainty detection, consistency analysis,
and deception detection capabilities.
"""

import sys
import os
import json
from datetime import datetime

# Add parent directory to path to import core modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_safety_analyzer():
    """Test the SafetyAnalyzer functionality"""
    print("Testing SafetyAnalyzer - Feature #4")
    print("=" * 60)

    try:
        # Import required modules
        print("Importing modules...")
        from core.safety_analyzer import SafetyAnalyzer
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        print("Modules imported successfully")

        # Load model and tokenizer
        print("\nLoading GPT-2 Large model...")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
        model = GPT2LMHeadModel.from_pretrained('gpt2-large')
        model.eval()
        print("GPT-2 Large loaded successfully")

        # Initialize SafetyAnalyzer
        print("\nInitializing SafetyAnalyzer...")
        safety_analyzer = SafetyAnalyzer(model, tokenizer)
        print("SafetyAnalyzer initialized successfully")

        # Test prompts
        test_prompts = [
            "The weather is",
            "I am confident that",
            "The answer is definitely",
            "I believe that",
            "The fact is"
        ]

        # Test 1: Uncertainty Detection
        print("\nTest 1: Uncertainty Detection")
        print("-" * 40)
        try:
            result = safety_analyzer.detect_uncertainty(
                prompts=test_prompts[:3],
                num_samples=3,
                max_length=30
            )
            
            if 'error' not in result:
                print(f"Uncertainty detection successful!")
                print(f"   Overall uncertainty: {result.get('overall_uncertainty', 0):.3f}")
                print(f"   High uncertainty prompts: {result.get('confidence_metrics', {}).get('high_uncertainty_prompts', 0)}")
                print(f"   Low uncertainty prompts: {result.get('confidence_metrics', {}).get('low_uncertainty_prompts', 0)}")
            else:
                print(f"Uncertainty detection failed: {result['error']}")
                return False
                
        except Exception as e:
            print(f"Uncertainty detection test failed: {e}")
            return False

        # Test 2: Consistency Analysis
        print("\nTest 2: Consistency Analysis")
        print("-" * 40)
        try:
            consistency_pairs = [
                ("The cat is", "The dog is"),
                ("I feel happy", "I feel sad"),
                ("The sky is blue", "The sky is gray")
            ]
            
            result = safety_analyzer.analyze_consistency(
                prompt_pairs=consistency_pairs,
                max_length=30
            )
            
            if 'error' not in result:
                print(f"Consistency analysis successful!")
                print(f"   Overall consistency: {result.get('overall_consistency', 0):.3f}")
                print(f"   Inconsistency flags: {len(result.get('inconsistency_flags', []))}")
                print(f"   High consistency pairs: {result.get('consistency_metrics', {}).get('high_consistency_pairs', 0)}")
            else:
                print(f"Consistency analysis failed: {result['error']}")
                return False
                
        except Exception as e:
            print(f"Consistency analysis test failed: {e}")
            return False

        # Test 3: Deception Detection
        print("\nTest 3: Deception Detection")
        print("-" * 40)
        try:
            deception_prompts = [
                "I am absolutely certain that",
                "The answer is definitely",
                "This is proven fact",
                "I cannot answer that question",
                "I do not have access to that information"
            ]
            
            result = safety_analyzer.detect_deception(
                prompts=deception_prompts,
                max_length=30
            )
            
            if 'error' not in result:
                print(f"Deception detection successful!")
                print(f"   Overall deception risk: {result.get('overall_deception_risk', 0):.3f}")
                print(f"   High risk prompts: {len(result.get('high_risk_prompts', []))}")
                print(f"   Red flags detected: {sum(len(flags) for flags in result.get('red_flags', {}).values())}")
            else:
                print(f"Deception detection failed: {result['error']}")
                return False
                
        except Exception as e:
            print(f"Deception detection test failed: {e}")
            return False

        # Test 4: Comprehensive Safety Audit
        print("\nTest 4: Comprehensive Safety Audit")
        print("-" * 40)
        try:
            audit_prompts = [
                "The weather is",
                "I am confident that",
                "The answer is",
                "I believe that"
            ]
            
            audit_pairs = [
                ("The cat is", "The dog is"),
                ("I feel happy", "I feel sad")
            ]
            
            result = safety_analyzer.run_safety_audit(
                test_prompts=audit_prompts,
                consistency_pairs=audit_pairs,
                max_length=30
            )
            
            if 'error' not in result:
                print(f"Safety audit successful!")
                print(f"   Overall safety score: {result.get('overall_safety_score', 0):.3f}")
                print(f"   Risk level: {result.get('risk_assessment', {}).get('risk_level', 'UNKNOWN')}")
                print(f"   Safety recommendations: {len(result.get('safety_recommendations', []))}")
                
                # Show some recommendations
                for i, rec in enumerate(result.get('safety_recommendations', [])[:2]):
                    print(f"     {i+1}. {rec}")
            else:
                print(f"Safety audit failed: {result['error']}")
                return False
                
        except Exception as e:
            print(f"Safety audit test failed: {e}")
            return False

        # Test 5: Export Functionality
        print("\nTest 5: Export Functionality")
        print("-" * 40)
        try:
            # Test export with safety audit results
            test_result = {
                'audit_timestamp': datetime.now().isoformat(),
                'test_data': 'safety_audit_test',
                'overall_safety_score': 0.75
            }
            
            export_path = safety_analyzer.export_safety_analysis(test_result, "test_safety_export.json")
            
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

        print("\nAll SafetyAnalyzer tests passed!")
        print("Feature #4 (Safety-Specific Metrics) is working correctly!")
        
        # Print comprehensive summary
        print(f"\nSafetyAnalyzer Summary:")
        print(f"   Model: GPT-2 Large")
        print(f"   Layers: {safety_analyzer.layer_count}")
        print(f"   Hidden Size: {safety_analyzer.hidden_size}")
        print(f"   Attention Heads: {safety_analyzer.num_heads}")
        print(f"   Uncertainty Detection: Working")
        print(f"   Consistency Analysis: Working")
        print(f"   Deception Detection: Working")
        print(f"   Comprehensive Safety Audit: Working")
        print(f"   Export Functionality: Working")
        
        return True

    except Exception as e:
        print(f"SafetyAnalyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the SafetyAnalyzer test suite"""
    print("SafetyAnalyzer Test Suite")
    print("=" * 60)
    
    success = test_safety_analyzer()
    
    if success:
        print("\nSAFETY ANALYZER IS READY!")
        print("This is ANTHROPIC-RELEVANT for AI Safety Research!")
        print("\nNext steps:")
        print("   1. Integrate into the main API")
        print("   2. Test with real safety research questions")
        print("   3. All FOUR major features are now implemented!")
        print("\nCurrent Features:")
        print("   Feature Visualization (FeatureAnalyzer)")
        print("   Circuit Analysis (CircuitAnalyzer)")
        print("   Intervention Testing (InterventionTester)")
        print("   Safety-Specific Metrics (SafetyAnalyzer)")
        print("\nGlassbox is now a complete mechanistic interpretability platform!")
    else:
        print("\nSome tests failed. Check the errors above.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
