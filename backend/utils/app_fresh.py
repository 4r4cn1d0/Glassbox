from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import List, Dict, Any
import os

# Disable TensorFlow to avoid Keras conflicts
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

app = FastAPI()

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model directly in this file
print("Loading GPT-2 Large directly in server...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
model = GPT2LMHeadModel.from_pretrained('gpt2-large')
model.eval()
print("GPT-2 Large loaded in server!")

class TraceRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 20

def generate_tokens(prompt: str, max_new_tokens: int = 10) -> List[Dict[str, Any]]:
    """Generate tokens directly in server"""
    print(f"Server generating: '{prompt}'")
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    print(f"Server tokens: {input_ids.tolist()[0]}")
    
    trace_data = []
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            print(f"\n--- SERVER STEP {step+1} ---")
            
            # Forward pass
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :]
            
            # Get top 5 tokens
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, 5)
            
            print(f"SERVER TOP 5:")
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                token = tokenizer.decode([idx.item()])
                print(f"  {i+1}. '{token}' (ID: {idx.item()}) - {prob.item():.4f}")
            
            # Greedy decoding
            next_token_id = torch.argmax(logits).item()
            next_token = tokenizer.decode([next_token_id])
            
            print(f"SERVER CHOSEN: '{next_token}' (ID: {next_token_id})")
            
            # Store trace data
            trace_data.append({
                "token": next_token,
                "token_id": next_token_id,
                "position": input_ids.shape[1],
                "logits": logits.cpu().numpy().tolist(),
                "attention": [],  # Simplified for now
                "top_tokens": [
                    {
                        "token": tokenizer.decode([idx.item()]),
                        "token_id": idx.item(),
                        "probability": prob.item()
                    }
                    for prob, idx in zip(top_probs, top_indices)
                ],
                "is_generated": True,
                "prompt_tokens": [
                    tokenizer.decode([t.item()]) 
                    for t in input_ids[0]
                ]
            })
            
            # Add token to sequence
            input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]])], dim=1)
            
            # Stop conditions
            if next_token_id == tokenizer.eos_token_id:
                break
    
    # Print final result
    final_text = tokenizer.decode(input_ids[0].tolist())
    generated_part = final_text[len(prompt):]
    print(f"\nSERVER FINAL: '{prompt}' → '{generated_part}'")
    
    return trace_data

@app.post("/api/trace")
async def trace_endpoint(request: TraceRequest) -> List[Dict[str, Any]]:
    try:
        print(f"API CALL: prompt='{request.prompt}', max_new_tokens={request.max_new_tokens}")
        trace_data = generate_tokens(request.prompt, request.max_new_tokens)
        print(f"API RESPONSE: {len(trace_data)} tokens generated")
        return trace_data
    except Exception as e:
        print(f"API ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Glassbox LLM Debugger - Fresh Server"}

@app.get("/api/health")
async def health():
    return {"status": "healthy", "mode": "fresh_server"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)  # Different port 