from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import List, Dict, Any, Optional
import os
import numpy as np

# Import core mechanistic interpretability analyzers
from core.feature_analyzer import FeatureAnalyzer
from core.circuit_analyzer import CircuitAnalyzer
from core.intervention_tester import InterventionTester
from core.safety_analyzer import SafetyAnalyzer

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

# Load GPT-2 Large model for mechanistic interpretability analysis
print("Loading GPT-2 Large model for server initialization...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
model = GPT2LMHeadModel.from_pretrained('gpt2-large')
model.eval()
print("GPT-2 Large model successfully loaded in server")

# Initialize FeatureAnalyzer for neuron representation analysis
print("Initializing FeatureAnalyzer...")
feature_analyzer = FeatureAnalyzer(model, tokenizer)
print("FeatureAnalyzer successfully initialized")

# Initialize CircuitAnalyzer for information flow tracing
print("Initializing CircuitAnalyzer...")
circuit_analyzer = CircuitAnalyzer(model, tokenizer)
print("CircuitAnalyzer successfully initialized")

# Initialize InterventionTester for causal analysis
print("Initializing InterventionTester...")
intervention_tester = InterventionTester(model, tokenizer)
print("InterventionTester successfully initialized")

# Initialize SafetyAnalyzer for safety-critical behavior analysis
print("Initializing SafetyAnalyzer...")
safety_analyzer = SafetyAnalyzer(model, tokenizer)
print("SafetyAnalyzer successfully initialized")

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

# FeatureAnalyzer API Models
class FeatureAnalysisRequest(BaseModel):
    concept_tokens: List[str]
    layer: int
    top_k: int = 10

class ConceptAnalysisRequest(BaseModel):
    concept_pairs: List[List[str]]  # [["happy", "sad"], ["good", "bad"]]
    layer: int
    top_k: int = 20

class NeuronAnalysisRequest(BaseModel):
    neuron_idx: int
    layer: int
    test_tokens: List[str]

class LayerRepresentationRequest(BaseModel):
    tokens: List[str]
    layer: int

# CircuitAnalyzer API Models
class InformationFlowRequest(BaseModel):
    input_tokens: List[str]
    target_token: str
    max_depth: int = 3

class CircuitComponentsRequest(BaseModel):
    concept_tokens: List[str]
    layer: int
    min_activation_threshold: float = 0.5

# InterventionTester API Models
class AttentionHeadAblationRequest(BaseModel):
    layer_idx: int
    head_idx: int
    test_prompts: List[str]
    max_length: int = 50

class MLPNeuronAblationRequest(BaseModel):
    layer_idx: int
    neuron_idx: int
    test_prompts: List[str]
    max_length: int = 50

class HeadImportanceRequest(BaseModel):
    layer_idx: int
    test_prompts: List[str]
    max_length: int = 50

class NeuronImportanceRequest(BaseModel):
    layer_idx: int
    neuron_indices: List[int]
    test_prompts: List[str]
    max_length: int = 50

# SafetyAnalyzer API Models
class UncertaintyDetectionRequest(BaseModel):
    prompts: List[str]
    num_samples: int = 5
    max_length: int = 50

class ConsistencyAnalysisRequest(BaseModel):
    prompt_pairs: List[List[str]]  # [["prompt1", "prompt2"], ["prompt3", "prompt4"]]
    max_length: int = 50

class DeceptionDetectionRequest(BaseModel):
    prompts: List[str]
    max_length: int = 50

class SafetyAuditRequest(BaseModel):
    test_prompts: List[str]
    consistency_pairs: Optional[List[List[str]]] = None
    max_length: int = 50

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
    
    print(f"Server generating: '{prompt}' (session: {session_id})")
    
    # Check if we have cached data
    if session_id in generation_cache:
        print(f"Using cached generation for session {session_id}")
        return generation_cache[session_id]
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    print(f"Server tokens: {input_ids.tolist()[0]}")
    
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
            
            print(f"SERVER TOP {top_k_tokens}:")
            for i, token_data in enumerate(top_tokens_data[:5]):  # Show top 5 in logs
                print(f"  {i+1}. '{token_data['token']}' - {token_data['probability']:.4f}")
            
            # Greedy decoding
            next_token_id = torch.argmax(logits).item()
            next_token = tokenizer.decode([next_token_id])
            
            print(f"CHOSEN: '{next_token}' (ID: {next_token_id})")
            
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
    print(f"\nFINAL: '{prompt}' → '{generated_part}'")
    print(f"Cached session {session_id} for timeline scrubbing")
    
    return result

@app.post("/api/trace")
async def trace_generation(request: TraceRequest):
    """Generate tokens with attention tracing and caching"""
    try:
        print(f"API CALL: prompt='{request.prompt}', max_new_tokens={request.max_new_tokens}")
        
        # Generate tokens with caching
        result = generate_tokens_with_cache(
            request.prompt, 
            request.max_new_tokens, 
            request.top_k_attention, 
            request.top_k_tokens
        )
        
        print(f"API RESPONSE: {len(result['trace_data'])} tokens generated")
        return result
    except Exception as e:
        print(f"API ERROR: {e}")
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
        print(f"SCRUB ERROR: {e}")
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
    return {
        "status": "healthy", 
        "mode": "real_transformer",
        "feature_analyzer_ready": True,
        "circuit_analyzer_ready": True,
        "intervention_tester_ready": True,
        "safety_analyzer_ready": True
    }

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
        print(f"LOGIT LENS ERROR: {e}")
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
        print(f"COMPONENT HOOKS ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# FEATURE ANALYZER ENDPOINTS - MECHANISTIC INTERPRETABILITY
# ============================================================================

@app.post("/api/analyze/features")
async def analyze_features(request: FeatureAnalysisRequest):
    """
    Find neurons that activate strongly for specific concepts
    This is REAL mechanistic interpretability - finding what neurons represent
    """
    try:
        print(f"FEATURE ANALYSIS: Finding neurons for concept {request.concept_tokens} at layer {request.layer}")
        
        result = feature_analyzer.find_feature_neurons(
            concept_tokens=request.concept_tokens,
            layer=request.layer,
            top_k=request.top_k
        )
        
        print(f"Found {len(result.get('top_neurons', []))} feature neurons")
        return result
        
    except Exception as e:
        print(f"FEATURE ANALYSIS ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/concepts")
async def analyze_concepts(request: ConceptAnalysisRequest):
    """
    Find neurons that distinguish between concept pairs
    e.g., neurons that respond to "happy" vs "sad"
    """
    try:
        print(f"CONCEPT ANALYSIS: Finding concept-distinguishing neurons for {len(request.concept_pairs)} pairs at layer {request.layer}")
        
        # Convert list of lists to list of tuples for the analyzer
        concept_pairs = [(pair[0], pair[1]) for pair in request.concept_pairs]
        
        result = feature_analyzer.find_concept_neurons(
            concept_pairs=concept_pairs,
            layer=request.layer,
            top_k=request.top_k
        )
        
        print(f"Found {len(result.get('concept_distinguishing_neurons', []))} concept-distinguishing neurons")
        return result
        
    except Exception as e:
        print(f"CONCEPT ANALYSIS ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/neuron")
async def analyze_neuron(request: NeuronAnalysisRequest):
    """
    Analyze what a specific neuron responds to
    Deep dive into individual neuron behavior
    """
    try:
        print(f"NEURON ANALYSIS: Analyzing neuron {request.neuron_idx} at layer {request.layer}")
        
        result = feature_analyzer.analyze_single_neuron(
            neuron_idx=request.neuron_idx,
            layer=request.layer,
            test_tokens=request.test_tokens
        )
        
        print(f"Neuron {request.neuron_idx} analysis complete")
        return result
        
    except Exception as e:
        print(f"NEURON ANALYSIS ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/layer-representations")
async def analyze_layer_representations(request: LayerRepresentationRequest):
    """
    Get comprehensive layer representations including MLP, attention, and layer norm
    Full layer analysis for mechanistic interpretability
    """
    try:
        print(f"LAYER REPRESENTATIONS: Analyzing layer {request.layer} for {len(request.tokens)} tokens")
        
        result = feature_analyzer.get_layer_representations(
            tokens=request.tokens,
            layer=request.layer
        )
        
        print(f"Layer {request.layer} representation analysis complete")
        return result
        
    except Exception as e:
        print(f"LAYER REPRESENTATIONS ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analyze/model-info")
async def get_model_info():
    """
    Get information about the model architecture for the FeatureAnalyzer
    """
    try:
        return {
            "model_type": "GPT-2 Large",
            "layers": feature_analyzer.layer_count,
            "hidden_size": feature_analyzer.hidden_size,
            "feature_analyzer_ready": True
        }
    except Exception as e:
        print(f"MODEL INFO ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# CIRCUIT ANALYZER ENDPOINTS - MECHANISTIC INTERPRETABILITY
# ============================================================================

@app.post("/api/analyze/information-flow")
async def trace_information_flow(request: InformationFlowRequest):
    """
    Trace how information flows from input tokens to target token
    This is REAL mechanistic interpretability - understanding the computation
    """
    try:
        print(f"INFORMATION FLOW: Tracing flow from {request.input_tokens} to '{request.target_token}' (depth: {request.max_depth})")
        
        result = circuit_analyzer.trace_information_flow(
            input_tokens=request.input_tokens,
            target_token=request.target_token,
            max_depth=request.max_depth
        )
        
        if 'error' in result:
            print(f"INFORMATION FLOW ERROR: {result['error']}")
            raise HTTPException(status_code=500, detail=result['error'])
        
        print(f"Traced information flow through {len(result.get('layers_analyzed', []))} layers")
        return result
        
    except Exception as e:
        print(f"INFORMATION FLOW ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/circuit-components")
async def find_circuit_components(request: CircuitComponentsRequest):
    """
    Find which neurons and attention heads work together as a circuit
    This identifies functional units in the model
    """
    try:
        print(f"CIRCUIT COMPONENTS: Finding components for concept {request.concept_tokens} at layer {request.layer}")
        
        result = circuit_analyzer.find_circuit_components(
            concept_tokens=request.concept_tokens,
            layer=request.layer,
            min_activation_threshold=request.min_activation_threshold
        )
        
        if 'error' in result:
            print(f"CIRCUIT COMPONENTS ERROR: {result['error']}")
            raise HTTPException(status_code=500, detail=result['error'])
        
        active_neurons = result.get('circuit_stats', {}).get('active_neurons', 0)
        print(f"Found {active_neurons} active neurons in circuit")
        return result
        
    except Exception as e:
        print(f"CIRCUIT COMPONENTS ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analyze/circuit-info")
async def get_circuit_info():
    """
    Get information about the model architecture for the CircuitAnalyzer
    """
    try:
        return {
            "model_type": "GPT-2 Large",
            "layers": circuit_analyzer.layer_count,
            "hidden_size": circuit_analyzer.hidden_size,
            "num_heads": circuit_analyzer.num_heads,
            "circuit_analyzer_ready": True
        }
    except Exception as e:
        print(f"CIRCUIT INFO ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# INTERVENTION TESTER ENDPOINTS - MECHANISTIC INTERPRETABILITY
# ============================================================================

@app.post("/api/analyze/ablate-attention-head")
async def ablate_attention_head(request: AttentionHeadAblationRequest):
    """
    Ablate (zero out) a specific attention head and measure its impact
    This is REAL mechanistic interpretability - testing causal relationships
    """
    try:
        print(f"ATTENTION HEAD ABLATION: Ablating head {request.head_idx} at layer {request.layer_idx}")

        result = intervention_tester.ablate_attention_head(
            layer_idx=request.layer_idx,
            head_idx=request.head_idx,
            test_prompts=request.test_prompts,
            max_length=request.max_length
        )

        print(f"Attention head ablation complete")
        return result

    except Exception as e:
        print(f"ATTENTION HEAD ABLATION ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/ablate-mlp-neuron")
async def ablate_mlp_neuron(request: MLPNeuronAblationRequest):
    """
    Ablate (zero out) a specific MLP neuron and measure its impact
    This tests the importance of individual neurons in the model
    """
    try:
        print(f"MLP NEURON ABLATION: Ablating neuron {request.neuron_idx} at layer {request.layer_idx}")

        result = intervention_tester.ablate_mlp_neuron(
            layer_idx=request.layer_idx,
            neuron_idx=request.neuron_idx,
            test_prompts=request.test_prompts,
            max_length=request.max_length
        )

        print(f"MLP neuron ablation complete")
        return result

    except Exception as e:
        print(f"MLP NEURON ABLATION ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/test-head-importance")
async def test_head_importance(request: HeadImportanceRequest):
    """
    Test the importance of all attention heads in a layer
    This ranks heads by their impact on model outputs
    """
    try:
        print(f"HEAD IMPORTANCE TESTING: Testing layer {request.layer_idx}")

        result = intervention_tester.test_attention_head_importance(
            layer_idx=request.layer_idx,
            test_prompts=request.test_prompts,
            max_length=request.max_length
        )

        print(f"Head importance testing complete")
        return result

    except Exception as e:
        print(f"HEAD IMPORTANCE TESTING ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/test-neuron-importance")
async def test_neuron_importance(request: NeuronImportanceRequest):
    """
    Test the importance of specific neurons in a layer
    This ranks neurons by their impact on model outputs
    """
    try:
        print(f"NEURON IMPORTANCE TESTING: Testing {len(request.neuron_indices)} neurons at layer {request.layer_idx}")

        result = intervention_tester.test_neuron_importance(
            layer_idx=request.layer_idx,
            neuron_indices=request.neuron_indices,
            test_prompts=request.test_prompts,
            max_length=request.max_length
        )

        print(f"Neuron importance testing complete")
        return result

    except Exception as e:
        print(f"NEURON IMPORTANCE TESTING ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analyze/intervention-info")
async def get_intervention_info():
    """
    Get information about the model architecture for the InterventionTester
    """
    try:
        return {
            "model_type": "GPT-2 Large",
            "layers": intervention_tester.layer_count,
            "hidden_size": intervention_tester.hidden_size,
            "num_heads": intervention_tester.num_heads,
            "intervention_tester_ready": True
        }
    except Exception as e:
        print(f"INTERVENTION INFO ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# SafetyAnalyzer API Endpoints
@app.post("/api/analyze/detect-uncertainty")
async def detect_uncertainty(request: UncertaintyDetectionRequest):
    """
    Detect uncertainty in model responses by analyzing multiple samples
    This helps identify when the model is uncertain about its outputs
    """
    try:
        print(f"UNCERTAINTY DETECTION: Analyzing {len(request.prompts)} prompts with {request.num_samples} samples each")

        result = safety_analyzer.detect_uncertainty(
            prompts=request.prompts,
            num_samples=request.num_samples,
            max_length=request.max_length
        )

        print(f"Uncertainty detection complete")
        return result

    except Exception as e:
        print(f"UNCERTAINTY DETECTION ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/analyze-consistency")
async def analyze_consistency(request: ConsistencyAnalysisRequest):
    """
    Analyze consistency between related prompt pairs
    This helps identify inconsistencies that could indicate safety issues
    """
    try:
        print(f"CONSISTENCY ANALYSIS: Analyzing {len(request.prompt_pairs)} prompt pairs")

        result = safety_analyzer.analyze_consistency(
            prompt_pairs=request.prompt_pairs,
            max_length=request.max_length
        )

        print(f"Consistency analysis complete")
        return result

    except Exception as e:
        print(f"CONSISTENCY ANALYSIS ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/detect-deception")
async def detect_deception(request: DeceptionDetectionRequest):
    """
    Detect potential deception or misleading outputs
    This helps identify safety-critical failure modes
    """
    try:
        print(f"DECEPTION DETECTION: Analyzing {len(request.prompts)} prompts")

        result = safety_analyzer.detect_deception(
            prompts=request.prompts,
            max_length=request.max_length
        )

        print(f"Deception detection complete")
        return result

    except Exception as e:
        print(f"DECEPTION DETECTION ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/safety-audit")
async def run_safety_audit(request: SafetyAuditRequest):
    """
    Run a comprehensive safety audit combining all safety metrics
    This provides an overall safety assessment of the model
    """
    try:
        print(f"SAFETY AUDIT: Running comprehensive safety analysis")

        result = safety_analyzer.run_safety_audit(
            test_prompts=request.test_prompts,
            consistency_pairs=request.consistency_pairs,
            max_length=request.max_length
        )

        print(f"Safety audit complete")
        return result

    except Exception as e:
        print(f"SAFETY AUDIT ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analyze/safety-info")
async def get_safety_info():
    """
    Get information about the model architecture for the SafetyAnalyzer
    """
    try:
        return {
            "model_type": "GPT-2 Large",
            "layers": safety_analyzer.layer_count,
            "hidden_size": safety_analyzer.hidden_size,
            "num_heads": safety_analyzer.num_heads,
            "safety_analyzer_ready": True,
            "safety_features": [
                "Uncertainty Detection",
                "Consistency Analysis", 
                "Deception Detection",
                "Comprehensive Safety Audit"
            ]
        }
    except Exception as e:
        print(f"SAFETY INFO ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 