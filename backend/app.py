from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import List, Dict, Any, Optional
import os
import numpy as np

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

# Load model directly in this file - WORKING IMPLEMENTATION
print("🔄 Loading GPT-2 Large directly in server...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
model = GPT2LMHeadModel.from_pretrained('gpt2-large')
model.eval()
print("✅ GPT-2 Large loaded in server!")

# Global cache for past_key_values
generation_cache = {}

class TraceRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 20
    top_k_attention: int = 50  # Only return top-K attention weights
    top_k_tokens: int = 20     # Number of top tokens to return

class ScrubRequest(BaseModel):
    session_id: str
    token_index: int

class LogitLensRequest(BaseModel):
    session_id: str
    token_index: int
    layers: List[int]
    mode: str = "top5"  # "top5", "diff", "entropy"

class ComponentHooksRequest(BaseModel):
    session_id: str
    token_index: int
    layer: int
    components: List[str]  # ["embed", "pos_embed", "attn", "mlp", "ln", "residual"]

def softmax(logits: torch.Tensor) -> torch.Tensor:
    """Compute softmax probabilities"""
    return torch.softmax(logits, dim=-1)

def get_top_k_attention(attention_matrix: torch.Tensor, k: int) -> Dict[str, Any]:
    """Return only top-K attention weights to reduce bandwidth"""
    # Flatten the attention matrix for the last token
    flat_attention = attention_matrix.flatten()
    
    # Get top-K indices and values
    top_values, top_indices = torch.topk(flat_attention, min(k, len(flat_attention)))
    
    # Convert back to 2D coordinates
    original_shape = attention_matrix.shape
    top_positions = []
    for idx in top_indices:
        pos = torch.unravel_index(idx, original_shape)
        top_positions.append({
            "from_token": pos[0].item(),
            "to_token": pos[1].item(),
            "weight": top_values[top_positions.__len__()].item()
        })
    
    return {
        "top_k_weights": top_positions,
        "shape": list(original_shape),
        "total_weights": len(flat_attention)
    }

def get_token_embeddings(input_ids: torch.Tensor) -> torch.Tensor:
    """Get token embeddings for UMAP/PCA visualization"""
    with torch.no_grad():
        embeddings = model.transformer.wte(input_ids)  # Word token embeddings
        return embeddings.cpu().numpy()

def generate_tokens_with_cache(prompt: str, max_new_tokens: int = 10, top_k_attention: int = 50, top_k_tokens: int = 20) -> Dict[str, Any]:
    """Generate tokens with caching for fast timeline scrubbing"""
    session_id = f"session_{hash(prompt)}_{max_new_tokens}"
    
    print(f"🔍 Server generating: '{prompt}' (session: {session_id})")
    
    # Check if we have cached data
    if session_id in generation_cache:
        print(f"♻️ Using cached generation for session {session_id}")
        return generation_cache[session_id]
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    print(f"🔍 Server tokens: {input_ids.tolist()[0]}")
    
    trace_data = []
    past_key_values = None
    all_embeddings = []
    
    # Get initial embeddings for prompt tokens
    prompt_embeddings = get_token_embeddings(input_ids)
    all_embeddings.extend(prompt_embeddings[0].tolist())
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            print(f"\n--- SERVER STEP {step+1} ---")
            
            # Forward pass with cached past_key_values for efficiency
            if past_key_values is not None:
                # Only process the last token when we have cache
                current_input = input_ids[:, -1:]
                outputs = model(current_input, past_key_values=past_key_values, output_attentions=True, use_cache=True)
            else:
                # First pass - process all tokens
                outputs = model(input_ids, output_attentions=True, use_cache=True)
            
            # Update cache
            past_key_values = outputs.past_key_values
            
            logits = outputs.logits[0, -1, :]
            
            # Get softmax probabilities
            probabilities = softmax(logits)
            
            # Get top tokens with real vocabulary
            top_probs, top_indices = torch.topk(probabilities, top_k_tokens)
            top_tokens_data = []
            
            for prob, idx in zip(top_probs, top_indices):
                token_text = tokenizer.decode([idx.item()])
                top_tokens_data.append({
                    "token": token_text,
                    "token_id": idx.item(),
                    "probability": prob.item(),
                    "logit": logits[idx].item()
                })
            
            print(f"🏆 SERVER TOP {top_k_tokens}:")
            for i, token_data in enumerate(top_tokens_data[:5]):  # Show top 5 in logs
                print(f"  {i+1}. '{token_data['token']}' - {token_data['probability']:.4f}")
            
            # Greedy decoding
            next_token_id = torch.argmax(logits).item()
            next_token = tokenizer.decode([next_token_id])
            
            print(f"🎯 SERVER CHOSEN: '{next_token}' (ID: {next_token_id})")
            
            # Extract optimized attention weights (top-K only)
            attention_data = []
            if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                for layer_idx, layer_attn in enumerate(outputs.attentions):
                    layer_data = []
                    for head_idx in range(layer_attn.shape[1]):
                        # Get attention for the last token only
                        head_attn = layer_attn[0, head_idx, -1, :]
                        
                        # Use top-K optimization to reduce data size
                        top_k_attn = get_top_k_attention(head_attn.unsqueeze(0), top_k_attention)
                        layer_data.append(top_k_attn)
                    attention_data.append(layer_data)
            
            # Get embeddings for the new token
            new_token_embedding = get_token_embeddings(torch.tensor([[next_token_id]]))
            all_embeddings.extend(new_token_embedding[0].tolist())
            
            # Store comprehensive trace data
            trace_data.append({
                "token": next_token,
                "token_id": next_token_id,
                "position": input_ids.shape[1],
                "logits": logits.cpu().numpy().tolist(),
                "probabilities": probabilities.cpu().numpy().tolist(),  # Full softmax probabilities
                "attention": attention_data,  # Optimized top-K attention
                "top_tokens": top_tokens_data,  # Real vocabulary with probabilities
                "embedding": new_token_embedding[0].tolist()[0],  # Token embedding for visualization
                "is_generated": True
            })
            
            # Add token to sequence
            input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]])], dim=1)
            
            # Stop conditions
            if next_token_id == tokenizer.eos_token_id:
                break
    
    # Prepare final result with caching
    result = {
        "session_id": session_id,
        "prompt": prompt,
        "trace_data": trace_data,
        "prompt_tokens": [
            {
                "token": tokenizer.decode([t.item()]),
                "token_id": t.item(),
                "embedding": emb
            }
            for t, emb in zip(input_ids[0][:len(tokenizer.encode(prompt))], all_embeddings[:len(tokenizer.encode(prompt))])
        ],
        "all_embeddings": all_embeddings,
        "vocabulary_size": tokenizer.vocab_size
    }
    
    # Cache the result for fast timeline scrubbing
    generation_cache[session_id] = result
    
    # Print final result
    final_text = tokenizer.decode(input_ids[0].tolist())
    generated_part = final_text[len(prompt):]
    print(f"\n✅ SERVER FINAL: '{prompt}' → '{generated_part}'")
    print(f"💾 Cached session {session_id} for timeline scrubbing")
    
    return result

@app.post("/api/trace")
async def trace_generation(request: TraceRequest):
    """Generate tokens with attention tracing and caching"""
    try:
        print(f"🌐 API CALL: prompt='{request.prompt}', max_new_tokens={request.max_new_tokens}")
        
        # Generate tokens with caching
        result = generate_tokens_with_cache(
            request.prompt, 
            request.max_new_tokens, 
            request.top_k_attention, 
            request.top_k_tokens
        )
        
        print(f"🌐 API RESPONSE: {len(result['trace_data'])} tokens generated")
        return result
    except Exception as e:
        print(f"🌐 API ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/scrub")
async def scrub_timeline(request: ScrubRequest) -> Dict[str, Any]:
    """Fast timeline scrubbing using cached past_key_values"""
    try:
        if request.session_id not in generation_cache:
            raise HTTPException(status_code=404, detail="Session not found")
        
        cached_data = generation_cache[request.session_id]
        
        # Return data up to the requested token index
        filtered_trace = cached_data["trace_data"][:request.token_index]
        
        return {
            "session_id": request.session_id,
            "trace_data": filtered_trace,
            "prompt_tokens": cached_data["prompt_tokens"],
            "current_index": request.token_index
        }
    except Exception as e:
        print(f"🌐 SCRUB ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/embeddings/{session_id}")
async def get_embeddings(session_id: str):
    """Get token embeddings for visualization"""
    if session_id not in generation_cache:
        raise HTTPException(status_code=404, detail="Session not found")
    
    cached_data = generation_cache[session_id]
    prompt_token_count = len(cached_data["prompt_tokens"])
    
    return {
        "embeddings": cached_data["all_embeddings"],
        "tokens": [token["token"] for token in cached_data["prompt_tokens"]] + 
                 [trace["token"] for trace in cached_data["trace_data"]],
        "token_ids": [token["token_id"] for token in cached_data["prompt_tokens"]] + 
                    [trace["token_id"] for trace in cached_data["trace_data"]],
        "embedding_dim": len(cached_data["all_embeddings"][0]) if cached_data["all_embeddings"] else 0,
        "prompt_token_count": prompt_token_count
    }

@app.get("/")
async def root():
    return {"message": "Glassbox LLM Debugger - Real Transformer"}

@app.get("/api/health")
async def health():
    return {"status": "healthy", "mode": "real_transformer"}

@app.post("/api/logit-lens")
async def logit_lens(request: LogitLensRequest):
    """TransformerLens-inspired logit lens functionality"""
    try:
        if request.session_id not in generation_cache:
            raise HTTPException(status_code=404, detail="Session not found")
        
        cached_data = generation_cache[request.session_id]
        
        # Get current tokens up to the requested index
        prompt_tokens = cached_data["prompt_tokens"]
        trace_data = cached_data["trace_data"]
        
        if request.token_index >= len(prompt_tokens) + len(trace_data):
            raise HTTPException(status_code=400, detail="Token index out of range")
        
        # Reconstruct the input sequence up to this point
        if request.token_index < len(prompt_tokens):
            current_tokens = [token["token_id"] for token in prompt_tokens[:request.token_index + 1]]
        else:
            generated_idx = request.token_index - len(prompt_tokens)
            current_tokens = ([token["token_id"] for token in prompt_tokens] + 
                            [trace["token_id"] for trace in trace_data[:generated_idx + 1]])
        
        input_ids = torch.tensor([current_tokens])
        
        layer_predictions = []
        
        with torch.no_grad():
            # Hook into each requested layer
            for layer_idx in request.layers:
                if layer_idx >= len(model.transformer.h):
                    continue
                
                # Run forward pass up to this layer
                embeddings = model.transformer.wte(input_ids) + model.transformer.wpe(torch.arange(len(current_tokens)).unsqueeze(0))
                
                # Pass through layers up to the target layer
                hidden_states = embeddings
                for i in range(layer_idx + 1):
                    layer = model.transformer.h[i]
                    hidden_states = layer(hidden_states)[0]
                
                # Apply layer norm and get logits from this layer's representation
                if layer_idx < len(model.transformer.h) - 1:
                    # For intermediate layers, we need to approximate the final output
                    # by applying the remaining layer norms and the language model head
                    ln_f_output = model.transformer.ln_f(hidden_states)
                    logits = model.lm_head(ln_f_output)
                else:
                    # For the final layer, use the standard path
                    outputs = model(input_ids)
                    logits = outputs.logits
                
                # Get the logits for the last token
                last_token_logits = logits[0, -1, :]
                
                # Get top predictions
                probabilities = torch.softmax(last_token_logits, dim=-1)
                top_probs, top_indices = torch.topk(probabilities, 5)
                
                predictions = []
                for rank, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                    token_text = tokenizer.decode([idx.item()])
                    predictions.append({
                        "token": token_text,
                        "logit": last_token_logits[idx].item(),
                        "probability": prob.item(),
                        "rank": rank + 1
                    })
                
                layer_predictions.append({
                    "layer": layer_idx,
                    "predictions": predictions
                })
        
        return {
            "layer_predictions": layer_predictions,
            "token_index": request.token_index,
            "mode": request.mode
        }
        
    except Exception as e:
        print(f"🌐 LOGIT LENS ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/component-hooks")
async def component_hooks(request: ComponentHooksRequest):
    """TransformerLens-inspired component hooks functionality"""
    try:
        if request.session_id not in generation_cache:
            raise HTTPException(status_code=404, detail="Session not found")
        
        cached_data = generation_cache[request.session_id]
        
        # Get current tokens up to the requested index
        prompt_tokens = cached_data["prompt_tokens"]
        trace_data = cached_data["trace_data"]
        
        if request.token_index >= len(prompt_tokens) + len(trace_data):
            raise HTTPException(status_code=400, detail="Token index out of range")
        
        # Reconstruct the input sequence up to this point
        if request.token_index < len(prompt_tokens):
            current_tokens = [token["token_id"] for token in prompt_tokens[:request.token_index + 1]]
        else:
            generated_idx = request.token_index - len(prompt_tokens)
            current_tokens = ([token["token_id"] for token in prompt_tokens] + 
                            [trace["token_id"] for trace in trace_data[:generated_idx + 1]])
        
        input_ids = torch.tensor([current_tokens])
        hook_data = []
        
        with torch.no_grad():
            # Get the specific layer
            if request.layer >= len(model.transformer.h):
                raise HTTPException(status_code=400, detail="Layer index out of range")
            
            for component in request.components:
                if component == "embed":
                    # Token embeddings
                    activations = model.transformer.wte(input_ids)[0, -1, :].cpu().numpy()
                    
                elif component == "pos_embed":
                    # Positional embeddings
                    position_ids = torch.arange(len(current_tokens)).unsqueeze(0)
                    activations = model.transformer.wpe(position_ids)[0, -1, :].cpu().numpy()
                    
                elif component == "attn":
                    # Run forward pass to get attention output
                    embeddings = model.transformer.wte(input_ids) + model.transformer.wpe(torch.arange(len(current_tokens)).unsqueeze(0))
                    hidden_states = embeddings
                    
                    # Pass through layers up to the target layer
                    for i in range(request.layer):
                        hidden_states = model.transformer.h[i](hidden_states)[0]
                    
                    # Get attention layer output
                    layer = model.transformer.h[request.layer]
                    attn_output = layer.attn(layer.ln_1(hidden_states))[0]
                    activations = attn_output[0, -1, :].cpu().numpy()
                    
                elif component == "mlp":
                    # MLP component
                    embeddings = model.transformer.wte(input_ids) + model.transformer.wpe(torch.arange(len(current_tokens)).unsqueeze(0))
                    hidden_states = embeddings
                    
                    # Pass through layers up to the target layer
                    for i in range(request.layer):
                        hidden_states = model.transformer.h[i](hidden_states)[0]
                    
                    # Get MLP output
                    layer = model.transformer.h[request.layer]
                    mlp_input = layer.ln_2(hidden_states)
                    mlp_output = layer.mlp(mlp_input)
                    activations = mlp_output[0, -1, :].cpu().numpy()
                    
                elif component == "ln":
                    # Layer norm
                    embeddings = model.transformer.wte(input_ids) + model.transformer.wpe(torch.arange(len(current_tokens)).unsqueeze(0))
                    hidden_states = embeddings
                    
                    # Pass through layers up to the target layer
                    for i in range(request.layer):
                        hidden_states = model.transformer.h[i](hidden_states)[0]
                    
                    if request.layer < len(model.transformer.h):
                        layer = model.transformer.h[request.layer]
                        ln_output = layer.ln_1(hidden_states)
                        activations = ln_output[0, -1, :].cpu().numpy()
                    else:
                        ln_output = model.transformer.ln_f(hidden_states)
                        activations = ln_output[0, -1, :].cpu().numpy()
                    
                elif component == "residual":
                    # Residual stream
                    embeddings = model.transformer.wte(input_ids) + model.transformer.wpe(torch.arange(len(current_tokens)).unsqueeze(0))
                    hidden_states = embeddings
                    
                    # Pass through layers up to the target layer
                    for i in range(request.layer + 1):
                        hidden_states = model.transformer.h[i](hidden_states)[0]
                    
                    activations = hidden_states[0, -1, :].cpu().numpy()
                
                else:
                    continue
                
                # Calculate statistics
                magnitude = float(np.linalg.norm(activations))
                sparsity = float(np.mean(np.abs(activations) < 0.1))
                
                # Get top neurons
                top_indices = np.argsort(np.abs(activations))[-5:][::-1]
                top_neurons = [
                    {
                        "index": int(idx),
                        "activation": float(activations[idx]),
                        "description": f"Neuron {idx} in {component}"
                    }
                    for idx in top_indices
                ]
                
                hook_data.append({
                    "component": component,
                    "layer": request.layer,
                    "activations": activations[:20].tolist(),  # First 20 for display
                    "magnitude": magnitude,
                    "sparsity": sparsity,
                    "top_neurons": top_neurons
                })
        
        return {
            "hook_data": hook_data,
            "token_index": request.token_index,
            "layer": request.layer
        }
        
    except Exception as e:
        print(f"🌐 COMPONENT HOOKS ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 