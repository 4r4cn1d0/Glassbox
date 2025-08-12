import torch
import numpy as np
from sklearn.decomposition import PCA
from typing import List, Dict, Tuple, Optional
import os
import json
from datetime import datetime

# Disable TensorFlow to avoid Keras conflicts
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

class FeatureAnalyzer:
    """
    Feature Analyzer - Identifies and analyzes neuron representations in transformer models.
    
    This class implements mechanistic interpretability techniques to understand what
    specific neurons represent within the model's internal representations.
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.activation_cache = {}
        
        # Initialize model architecture information
        self.layer_count = self._get_layer_count()
        self.hidden_size = self._get_hidden_size()
        
        print(f"FeatureAnalyzer initialized for {self.layer_count} layers, {self.hidden_size} hidden dimensions")
    
    def _get_layer_count(self) -> int:
        """Retrieve the total number of transformer layers in the model."""
        try:
            if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                return len(self.model.transformer.h)
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                return len(self.model.model.layers)
            else:
                return 12
        except:
            return 12
    
    def _get_hidden_size(self) -> int:
        """Retrieve the hidden dimension size of the model."""
        try:
            if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                return self.model.transformer.h[0].mlp.c_fc.out_features
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                return self.model.model.layers[0].mlp.dense_4h_to_h.out_features
            else:
                return 768
        except:
            return 768
    
    def get_neuron_activations(self, tokens: List[str], layer: int, cache_key: str = None) -> np.ndarray:
        """
        Retrieve neuron activations for specified tokens at a given layer.
        
        Args:
            tokens: List of tokens to analyze
            layer: Target transformer layer for analysis
            cache_key: Optional cache identifier for computational efficiency
            
        Returns:
            numpy array of shape (num_tokens, hidden_size) containing neuron activations
        """
        if cache_key and cache_key in self.activation_cache:
            return self.activation_cache[cache_key]
        
        # Ensure padding token configuration
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Process tokens individually and collect activations
        all_activations = []
        
        for token in tokens:
            # Tokenize input sequence
            token_ids = self.tokenizer.encode(token, return_tensors="pt")
            
            # Hook function to capture intermediate activations
            activations = []
            
            def hook_fn(module, input, output):
                activations.append(output.detach().cpu().numpy())
            
            # Register forward hook on target layer's MLP component
            try:
                if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                    target_layer = self.model.transformer.h[layer]
                    hook = target_layer.mlp.register_forward_hook(hook_fn)
                elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                    target_layer = self.model.model.layers[layer]
                    hook = target_layer.mlp.register_forward_hook(hook_fn)
                else:
                    raise ValueError("Unsupported model architecture")
                
                # Execute forward pass
                with torch.no_grad():
                    self.model(token_ids)
                
                # Clean up hook
                hook.remove()
                
                if not activations:
                    raise ValueError("No activations captured during forward pass")
                
                # Extract activations for current token
                token_activations = activations[0]
                all_activations.append(token_activations)
                
            except Exception as e:
                print(f"Error retrieving activations for token '{token}' at layer {layer}: {e}")
                # Provide zero activations as fallback
                all_activations.append(np.zeros((1, 1, self.hidden_size)))
        
        # Consolidate all activations with proper padding
        if all_activations:
            max_seq_len = max(act.shape[1] for act in all_activations)
            hidden_size = all_activations[0].shape[2]
            
            # Pad activations to uniform sequence length
            padded_activations = []
            for act in all_activations:
                if act.shape[1] < max_seq_len:
                    padding = np.zeros((1, max_seq_len - act.shape[1], hidden_size))
                    padded_act = np.concatenate([act, padding], axis=1)
                else:
                    padded_act = act
                padded_activations.append(padded_act)
            
            result = np.vstack(padded_activations)
        else:
            result = np.zeros((len(tokens), 1, self.hidden_size))
        
        # Cache results if requested
        if cache_key:
            self.activation_cache[cache_key] = result
        
        return result
    
    def find_feature_neurons(self, concept_tokens: List[str], layer: int, top_k: int = 10) -> Dict:
        """
        Identify neurons that exhibit strong activation patterns for specific conceptual categories.
        
        Args:
            concept_tokens: List of tokens representing the target concept
            layer: Target layer for analysis
            top_k: Number of top-ranking neurons to return
            
        Returns:
            Dictionary containing top neurons and their activation characteristics
        """
        print(f"Analyzing feature neurons for concept: {concept_tokens}")
        print(f"Target layer: {layer}, Top-K selection: {top_k}")
        
        try:
            # Retrieve activations for concept tokens
            cache_key = f"concept_{hash(tuple(concept_tokens))}_{layer}"
            activations = self.get_neuron_activations(concept_tokens, layer, cache_key)
            
            print(f"Activation tensor dimensions: {activations.shape}")
            
            # Compute mean activations across concept examples
            mean_activations = np.mean(activations, axis=(0, 1))
            print(f"Mean activation vector dimensions: {mean_activations.shape}")
            
            # Identify top-k most responsive neurons
            top_indices = np.argsort(mean_activations)[-top_k:][::-1]
            top_activations = mean_activations[top_indices]
            
            # Perform detailed analysis for top neurons
            top_neurons_analysis = []
            for i, (neuron_idx, activation) in enumerate(zip(top_indices, top_activations)):
                neuron_analysis = self.analyze_single_neuron(neuron_idx, layer, concept_tokens)
                top_neurons_analysis.append(neuron_analysis)
            
            result = {
                'concept_tokens': concept_tokens,
                'layer': layer,
                'top_neurons': top_indices.tolist(),
                'activation_strengths': top_activations.tolist(),
                'top_neurons_analysis': top_neurons_analysis,
                'concept_activation_pattern': {
                    'mean_activation': float(np.mean(mean_activations)),
                    'std_activation': float(np.std(mean_activations)),
                    'max_activation': float(np.max(mean_activations)),
                    'min_activation': float(np.min(mean_activations))
                }
            }
            
            print(f"Successfully identified {len(top_indices)} feature neurons")
            print(f"Primary neuron: {top_indices[0]} (activation strength: {top_activations[0]:.4f})")
            
            return result
            
        except Exception as e:
            print(f"Error finding feature neurons: {e}")
            return {
                'error': str(e),
                'concept_tokens': concept_tokens,
                'layer': layer
            }
    
    def analyze_single_neuron(self, neuron_idx: int, layer: int, test_tokens: List[str]) -> Dict:
        """
        Analyze the behavior of a specific neuron in the model.
        
        Args:
            neuron_idx: Index of the neuron to analyze
            layer: Target layer of the neuron
            test_tokens: Tokens to test the neuron's response to
            
        Returns:
            Dictionary containing detailed analysis of the neuron's response
        """
        try:
            # Retrieve activations for test tokens
            cache_key = f"test_{hash(tuple(test_tokens))}_{layer}"
            activations = self.get_neuron_activations(test_tokens, layer, cache_key)
            
            print(f"analyze_single_neuron - activations shape: {activations.shape}")
            
            # Extract activations for the specific neuron
            # activations shape is (num_tokens, seq_len, hidden_size)
            # We want to get the neuron activations across all positions
            neuron_activations = activations[0, :, neuron_idx]  # Shape: (seq_len,)
            print(f"neuron_activations shape: {neuron_activations.shape}")
            
            # Identify tokens that activate this neuron most strongly
            top_indices = np.argsort(neuron_activations)[-5:][::-1]  # Top 5
            
            # Calculate statistics
            mean_activation = float(np.mean(neuron_activations))
            std_activation = float(np.std(neuron_activations))
            max_activation = float(np.max(neuron_activations))
            min_activation = float(np.min(neuron_activations))
            
            # Determine neuron type based on activation pattern
            activation_range = max_activation - min_activation
            if activation_range > 2.0:
                neuron_type = "high_variance"
            elif activation_range > 0.5:
                neuron_type = "moderate_variance"
            else:
                neuron_type = "low_variance"
            
            return {
                'neuron_index': int(neuron_idx),
                'layer': layer,
                'top_activating_tokens': [test_tokens[i] for i in top_indices if i < len(test_tokens)],
                'top_activation_values': [float(neuron_activations[i]) for i in top_indices if i < len(test_tokens)],
                'activation_stats': {
                    'mean': mean_activation,
                    'std': std_activation,
                    'max': max_activation,
                    'min': min_activation,
                    'range': activation_range
                },
                'neuron_type': neuron_type,
                'responsiveness_score': float(max_activation / (std_activation + 1e-8))
            }
            
        except Exception as e:
            return {
                'neuron_index': int(neuron_idx),
                'layer': layer,
                'error': str(e)
            }
    
    def find_concept_neurons(self, concept_pairs: List[Tuple[str, str]], layer: int, top_k: int = 20) -> Dict:
        """
        Identify neurons that distinguish between concept pairs.
        
        Args:
            concept_pairs: List of (positive, negative) concept pairs
                          e.g., [("happy", "sad"), ("good", "bad")]
            layer: Target layer for analysis
            top_k: Number of top-ranking neurons to return
            
        Returns:
            Dictionary containing neurons that distinguish between concepts
        """
        print(f"Analyzing concept-distinguishing neurons for {len(concept_pairs)} pairs")
        print(f"Target layer: {layer}, Top-K selection: {top_k}")
        
        try:
            all_tokens = []
            for pos, neg in concept_pairs:
                all_tokens.extend([pos, neg])
            
            # Retrieve activations for all tokens
            cache_key = f"concept_pairs_{hash(tuple(all_tokens))}_{layer}"
            activations = self.get_neuron_activations(all_tokens, layer, cache_key)
            
            print(f"Concept pairs activations shape: {activations.shape}")
            
            # Calculate difference between positive and negative concepts
            concept_differences = []
            for i in range(0, activations.shape[0], 2):  # Step by 2 for pairs
                if i + 1 < activations.shape[0]:  # Ensure both tokens exist
                    pos_act = activations[i, :, :]      # Positive concept (seq_len, hidden_size)
                    neg_act = activations[i + 1, :, :]  # Negative concept (seq_len, hidden_size)
                    # Average over sequence dimension to get per-neuron differences
                    pos_mean = np.mean(pos_act, axis=0)  # (hidden_size,)
                    neg_mean = np.mean(neg_act, axis=0)  # (hidden_size,)
                    diff = pos_mean - neg_mean           # (hidden_size,)
                    concept_differences.append(diff)
            
            if not concept_differences:
                raise ValueError("No concept differences calculated")
            
            # Average differences across concept pairs
            mean_differences = np.mean(concept_differences, axis=0)
            print(f"Mean differences shape: {mean_differences.shape}")
            
            # Identify neurons with strongest concept distinction
            top_indices = np.argsort(np.abs(mean_differences))[-top_k:][::-1]
            distinction_scores = mean_differences[top_indices]
            
            # Perform detailed analysis for top distinguishing neurons
            top_neurons_analysis = []
            for neuron_idx, score in zip(top_indices, distinction_scores):
                neuron_analysis = self.analyze_single_neuron(neuron_idx, layer, all_tokens)
                neuron_analysis['concept_distinction_score'] = float(score)
                top_neurons_analysis.append(neuron_analysis)
            
            result = {
                'concept_pairs': concept_pairs,
                'layer': layer,
                'concept_distinguishing_neurons': top_indices.tolist(),
                'distinction_scores': distinction_scores.tolist(),
                'top_neurons_analysis': top_neurons_analysis,
                'concept_pairs_analyzed': len(concept_pairs),
                'analysis_summary': {
                    'strongest_distinction': float(np.max(np.abs(mean_differences))),
                    'mean_distinction': float(np.mean(np.abs(mean_differences))),
                    'distinction_std': float(np.std(np.abs(mean_differences)))
                }
            }
            
            print(f"Successfully identified {len(top_indices)} concept-distinguishing neurons")
            print(f"Strongest distinction: {result['analysis_summary']['strongest_distinction']:.4f}")
            
            return result
            
        except Exception as e:
            print(f"Error finding concept neurons: {e}")
            return {
                'error': str(e),
                'concept_pairs': concept_pairs,
                'layer': layer
            }
    
    def get_layer_representations(self, tokens: List[str], layer: int) -> Dict:
        """
        Retrieve comprehensive layer representations for a given set of tokens.
        
        Args:
            tokens: Tokens to analyze
            layer: Target layer for analysis
            
        Returns:
            Dictionary containing detailed analysis of the layer's representations
        """
        try:
            # Collect multiple types of activations
            layer_data = {}
            
            # Process each token individually like the working methods
            all_mlp_activations = []
            all_attn_activations = []
            all_ln_activations = []
            
            for token in tokens:
                # Tokenize input
                token_ids = self.tokenizer.encode(token, return_tensors="pt")
                
                # Hook functions for different components
                mlp_activations = []
                attn_activations = []
                ln_activations = []
                
                def mlp_hook(module, input, output):
                    # Handle different output types
                    if isinstance(output, tuple):
                        output = output[0]  # Take first element if tuple
                    mlp_activations.append(output.detach().cpu().numpy())
                
                def attn_hook(module, input, output):
                    # Handle different output types
                    if isinstance(output, tuple):
                        output = output[0]  # Take first element if tuple
                    attn_activations.append(output.detach().cpu().numpy())
                
                def ln_hook(module, input, output):
                    # Handle different output types
                    if isinstance(output, tuple):
                        output = output[0]  # Take first element if tuple
                    ln_activations.append(output.detach().cpu().numpy())
                
                # Register hooks for different layer components
                hooks = []
                try:
                    if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                        target_layer = self.model.transformer.h[layer]
                        hooks.extend([
                            target_layer.mlp.register_forward_hook(mlp_hook),
                            target_layer.attn.register_forward_hook(attn_hook),
                            target_layer.ln_1.register_forward_hook(ln_hook)
                        ])
                    elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                        target_layer = self.model.model.layers[layer]
                        hooks.extend([
                            target_layer.mlp.register_forward_hook(mlp_hook),
                            target_layer.self_attn.register_forward_hook(attn_hook),
                            target_layer.input_layernorm.register_forward_hook(ln_hook)
                        ])
                    else:
                        raise ValueError("Unsupported model architecture")
                    
                    # Execute forward pass
                    with torch.no_grad():
                        self.model(token_ids)
                    
                    # Extract captured activations
                    if mlp_activations:
                        all_mlp_activations.append(mlp_activations[0])
                    if attn_activations:
                        all_attn_activations.append(attn_activations[0])
                    if ln_activations:
                        all_ln_activations.append(ln_activations[0])
                    
                finally:
                    # Remove all registered hooks
                    for hook in hooks:
                        hook.remove()
            
            # Consolidate collected activations
            if all_mlp_activations:
                max_seq_len = max(act.shape[1] for act in all_mlp_activations)
                hidden_size = all_mlp_activations[0].shape[2]
                
                # Pad activations to uniform sequence length
                padded_mlp = []
                for act in all_mlp_activations:
                    if act.shape[1] < max_seq_len:
                        padding = np.zeros((1, max_seq_len - act.shape[1], hidden_size))
                        padded_act = np.concatenate([act, padding], axis=1)
                    else:
                        padded_act = act
                    padded_mlp.append(padded_act)
                
                # Stack and average MLP activations
                mlp_stacked = np.vstack(padded_mlp)
                layer_data['mlp'] = {
                    'activations': mlp_stacked.tolist(),
                    'shape': list(mlp_stacked.shape),
                    'stats': self._compute_activation_stats(mlp_stacked)
                }
            
            if all_attn_activations:
                max_seq_len = max(act.shape[1] for act in all_attn_activations)
                hidden_size = all_attn_activations[0].shape[2]
                
                # Pad activations to uniform sequence length
                padded_attn = []
                for act in all_attn_activations:
                    if act.shape[1] < max_seq_len:
                        padding = np.zeros((1, max_seq_len - act.shape[1], hidden_size))
                        padded_act = np.concatenate([act, padding], axis=1)
                    else:
                        padded_act = act
                    padded_attn.append(padded_act)
                
                # Stack and average attention activations
                attn_stacked = np.vstack(padded_attn)
                layer_data['attention'] = {
                    'activations': attn_stacked.tolist(),
                    'shape': list(attn_stacked.shape),
                    'stats': self._compute_activation_stats(attn_stacked)
                }
            
            if all_ln_activations:
                max_seq_len = max(act.shape[1] for act in all_ln_activations)
                hidden_size = all_ln_activations[0].shape[2]
                
                # Pad activations to uniform sequence length
                padded_ln = []
                for act in all_ln_activations:
                    if act.shape[1] < max_seq_len:
                        padding = np.zeros((1, max_seq_len - act.shape[1], hidden_size))
                        padded_act = np.concatenate([act, padding], axis=1)
                    else:
                        padded_act = act
                    padded_ln.append(padded_act)
                
                # Stack and average layer norm activations
                ln_stacked = np.vstack(padded_ln)
                layer_data['layer_norm'] = {
                    'activations': ln_stacked.tolist(),
                    'shape': list(ln_stacked.shape),
                    'stats': self._compute_activation_stats(ln_stacked)
                }
            
            return layer_data
            
        except Exception as e:
            return {
                'error': f"Layer representation analysis failed: {str(e)}",
                'layer': layer
            }
    
    def _compute_activation_stats(self, activations: np.ndarray) -> Dict:
        """Compute statistical measures for activation tensor."""
        return {
            'mean': float(np.mean(activations)),
            'std': float(np.std(activations)),
            'max': float(np.max(activations)),
            'min': float(np.min(activations)),
            'sparsity': float(np.mean(np.abs(activations) < 0.1)),
            'magnitude': float(np.linalg.norm(activations))
        }
    
    def export_analysis(self, analysis: Dict, filename: str = None) -> str:
        """Export analysis results to JSON file for external processing."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"feature_analysis_{timestamp}"
        
        filepath = f"{filename}.json"
        with open(filepath, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"Feature analysis exported to: {filepath}")
        return filepath

# Test function to verify the implementation works
def test_feature_analyzer():
    """Test the FeatureAnalyzer with a simple example."""
    print("Testing FeatureAnalyzer...")
    
    try:
        # Load a model (this would normally be passed in)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_name = "microsoft/DialoGPT-medium"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Initialize analyzer
        analyzer = FeatureAnalyzer(model, tokenizer)
        
        # Test 1: Find feature neurons for "happy" concept
        print("\nTest 1: Finding neurons for 'happy' concept")
        happy_neurons = analyzer.find_feature_neurons(["happy", "joy", "smile"], layer=6, top_k=5)
        print(f"Result: {len(happy_neurons.get('top_neurons', []))} neurons found")
        
        # Test 2: Find concept-distinguishing neurons
        print("\nTest 2: Finding concept-distinguishing neurons")
        concept_pairs = [("happy", "sad"), ("good", "bad"), ("love", "hate")]
        concept_neurons = analyzer.find_concept_neurons(concept_pairs, layer=6, top_k=5)
        print(f"Result: {len(concept_neurons.get('concept_distinguishing_neurons', []))} neurons found")
        
        # Test 3: Analyze single neuron
        if happy_neurons.get('top_neurons'):
            print("\nTest 3: Analyzing single neuron")
            neuron_analysis = analyzer.analyze_single_neuron(
                happy_neurons['top_neurons'][0], 
                layer=6, 
                test_tokens=["happy", "joy", "smile", "sad", "angry"]
            )
            print(f"Neuron {neuron_analysis['neuron_index']} type: {neuron_analysis.get('neuron_type', 'unknown')}")
        
        print("\nFeatureAnalyzer tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"FeatureAnalyzer test failed: {e}")
        return False

if __name__ == "__main__":
    # Run tests if this file is executed directly
    test_feature_analyzer()
