import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Dict, List, Any
import os

# Disable TensorFlow to avoid Keras conflicts
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

class GPT2LargeTracer:
    def __init__(self):
        print("Loading GPT-2 Large (774M parameters)...")
        
        # Load model and tokenizer - EXACTLY like the working test
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2-large')
        self.model.eval()
        
        print(f"✅ GPT-2 Large loaded successfully!")
        print(f"📊 Model size: {sum(p.numel() for p in self.model.parameters())/1e6:.1f}M parameters")
        print(f"🧠 Architecture: {self.model.config.n_layer} layers, {self.model.config.n_head} heads")
        
    def trace_generation(self, prompt: str, max_new_tokens: int = 10) -> List[Dict[str, Any]]:
        """
        Generate tokens step by step - EXACTLY like the working standalone test
        """
        print(f"🔍 Input prompt: '{prompt}'")
        
        # Tokenize prompt - EXACTLY like working test
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        print(f"🔍 Input tokens: {input_ids.tolist()[0]}")
        
        trace_data = []
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                print(f"\n--- STEP {step+1} ---")
                
                # Forward pass - EXACTLY like working test
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1, :]  # Last position logits
                
                # Get top 5 most likely tokens for debugging
                probs = torch.softmax(logits, dim=-1)
                top_probs, top_indices = torch.topk(probs, 5)
                
                print(f"🏆 TOP 5 TOKENS:")
                for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                    token = self.tokenizer.decode([idx.item()])
                    print(f"  {i+1}. '{token}' (ID: {idx.item()}) - {prob.item():.4f}")
                
                # Greedy decoding - EXACTLY like working test
                next_token_id = torch.argmax(logits).item()
                next_token = self.tokenizer.decode([next_token_id])
                
                print(f"🎯 CHOSEN: '{next_token}' (ID: {next_token_id})")
                
                # Get attention weights (simplified for now)
                attention_data = []
                if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                    for layer_idx, layer_attn in enumerate(outputs.attentions):
                        layer_data = []
                        for head_idx in range(layer_attn.shape[1]):
                            head_attn = layer_attn[0, head_idx, -1, :].cpu().numpy().tolist()
                            layer_data.append(head_attn)
                        attention_data.append(layer_data)
                else:
                    # Create dummy attention if not available
                    for layer_idx in range(36):  # GPT-2 Large has 36 layers
                        layer_data = []
                        for head_idx in range(20):  # GPT-2 Large has 20 heads
                            head_attn = [0.0] * input_ids.shape[1]
                            layer_data.append(head_attn)
                        attention_data.append(layer_data)
                
                # Get top tokens for display
                top_tokens = [
                    {
                        "token": self.tokenizer.decode([idx.item()]),
                        "token_id": idx.item(),
                        "probability": prob.item()
                    }
                    for prob, idx in zip(top_probs, top_indices)
                ]
                
                # Store trace data
                trace_data.append({
                    "token": next_token,
                    "token_id": next_token_id,
                    "position": input_ids.shape[1],
                    "logits": logits.cpu().numpy().tolist(),
                    "attention": attention_data,
                    "top_tokens": top_tokens,
                    "is_generated": True,
                    "prompt_tokens": [
                        self.tokenizer.decode([t.item()]) 
                        for t in input_ids[0]
                    ]
                })
                
                # Add token to sequence - EXACTLY like working test
                input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]])], dim=1)
                
                # Stop conditions
                if next_token_id == self.tokenizer.eos_token_id:
                    break
        
        # Print final result
        final_text = self.tokenizer.decode(input_ids[0].tolist())
        generated_part = final_text[len(prompt):]
        print(f"\n✅ FINAL RESULT: '{prompt}' → '{generated_part}'")
        
        return trace_data

# Global instance
gpt2_tracer = GPT2LargeTracer() 