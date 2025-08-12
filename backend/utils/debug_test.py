#!/usr/bin/env python3
"""
Debug script for FeatureAnalyzer

This script provides a minimal test to identify issues
with the FeatureAnalyzer implementation.
"""

import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def test_minimal():
    """Minimal test to debug the issue"""
    print("Minimal FeatureAnalyzer Debug Test")
    print("=" * 50)
    
    try:
        # Load model and tokenizer
        print("Loading model...")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        model.eval()
        
        # Set padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("Model loaded successfully")
        print(f"   Model type: {type(model)}")
        print(f"   Has transformer: {hasattr(model, 'transformer')}")
        print(f"   Has h: {hasattr(model.transformer, 'h') if hasattr(model, 'transformer') else False}")
        print(f"   Layers: {len(model.transformer.h) if hasattr(model, 'transformer') else 'N/A'}")
        
        # Test simple tokenization
        print("\nTesting tokenization...")
        test_tokens = ["happy", "joy"]
        
        for token in test_tokens:
            token_ids = tokenizer.encode(token, return_tensors="pt")
            print(f"   '{token}' -> {token_ids.shape}: {token_ids}")
        
        # Test simple forward pass
        print("\nTesting forward pass...")
        test_input = tokenizer.encode("happy", return_tensors="pt")
        print(f"   Input shape: {test_input.shape}")
        
        with torch.no_grad():
            outputs = model(test_input)
            print(f"   Output logits shape: {outputs.logits.shape}")
        
        # Test hook on layer 0
        print("\nTesting hook on layer 0...")
        activations = []
        
        def hook_fn(module, input, output):
            print(f"   Hook called! Output shape: {output.shape}")
            activations.append(output.detach().cpu().numpy())
        
        hook = model.transformer.h[0].mlp.register_forward_hook(hook_fn)
        
        with torch.no_grad():
            model(test_input)
        
        hook.remove()
        
        if activations:
            print(f"   Captured activation shape: {activations[0].shape}")
            print(f"   Activation type: {type(activations[0])}")
            print(f"   Activation dtype: {activations[0].dtype}")
        else:
            print("   No activations captured")
        
        print("\nMinimal test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Minimal test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_minimal()
