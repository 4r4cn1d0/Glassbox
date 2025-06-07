from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from typing import Dict, List, Any

class ModelTracer:
    def __init__(self, model_name: str = "gpt2"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_attentions=True,
            output_hidden_states=False
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def trace_generation(self, prompt: str, max_new_tokens: int = 50) -> List[Dict[str, Any]]:
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        past_key_values = None
        trace_data = []
        
        for _ in range(max_new_tokens):
            outputs = self.model(input_ids=input_ids, past_key_values=past_key_values)
            logits = outputs.logits
            attention = outputs.attentions
            
            # Get next token
            next_token_logits = logits[0, -1]
            next_token = torch.argmax(next_token_logits)
            
            # Convert attention to list for JSON serialization
            attention_list = [
                layer_attention[0].tolist()  # Remove batch dimension
                for layer_attention in attention
            ]
            
            # Record step data
            step_data = {
                "token": self.tokenizer.decode(next_token),
                "token_id": next_token.item(),
                "logits": next_token_logits.tolist(),
                "attention": attention_list
            }
            trace_data.append(step_data)
            
            # Update for next iteration
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=-1)
            past_key_values = outputs.past_key_values
            
        return trace_data

if __name__ == "__main__":
    # Test the tracer
    tracer = ModelTracer()
    prompt = "Once upon a time"
    trace = tracer.trace_generation(prompt)
    print(json.dumps(trace[0], indent=2))  # Print first step as example 