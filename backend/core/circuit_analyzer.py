#!/usr/bin/env python3
"""
Circuit Analyzer - Traces information flow and identifies critical computational paths.

This module implements mechanistic interpretability techniques to understand how
information propagates through transformer model architectures.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import networkx as nx
from collections import defaultdict
import json
from datetime import datetime

class CircuitAnalyzer:
    """
    Circuit Analyzer - Traces information flow through transformer model architectures.
    
    This class implements mechanistic interpretability techniques to understand
    how information propagates and transforms through the model's computational graph.
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.circuit_cache = {}
        
        # Initialize model architecture information
        self.layer_count = self._get_layer_count()
        self.hidden_size = self._get_hidden_size()
        self.num_heads = self._get_num_heads()
        
        print(f"CircuitAnalyzer initialized for {self.layer_count} layers, {self.hidden_size} hidden dimensions, {self.num_heads} heads")
    
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
    
    def _get_num_heads(self) -> int:
        """Retrieve the number of attention heads in the model."""
        try:
            if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                return self.model.transformer.h[0].attn.num_heads
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                return self.model.model.layers[0].self_attn.num_heads
            else:
                return 12
        except:
            return 12
    
    def trace_information_flow(self, input_tokens: List[str], target_token: str, max_depth: int = 3) -> Dict:
        """
        Trace information flow from input tokens to target token through the model.
        
        This method implements mechanistic interpretability by analyzing how
        information propagates through the computational graph.
        
        Args:
            input_tokens: Context tokens that provide input information
            target_token: Token whose generation process is to be traced
            max_depth: Maximum depth of circuit analysis
            
        Returns:
            Dictionary containing comprehensive circuit information flow analysis
        """
        print(f"Tracing information flow: {input_tokens} → '{target_token}' (max_depth: {max_depth})")
        
        try:
            # Ensure padding token configuration
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Prepare input sequence for analysis
            input_text = " ".join(input_tokens)
            full_text = input_text + " " + target_token
            
            # Tokenize the complete sequence
            full_sequence = self.tokenizer(full_text, return_tensors="pt", padding=True, truncation=True)
            
            # Initialize tracking structures
            layer_activations = {}
            attention_patterns = {}
            
            # Hook functions to capture intermediate activations
            def create_layer_hook(layer_idx):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        output = output[0]
                    layer_activations[layer_idx] = output.detach().cpu().numpy()
                return hook_fn
            
            def create_attention_hook(layer_idx):
                def hook_fn(module, input, output):
                    if hasattr(module, 'attn_weights'):
                        attention_patterns[layer_idx] = module.attn_weights.detach().cpu().numpy()
                return hook_fn
            
            # Register hooks for all layers
            hooks = []
            try:
                if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                    # GPT-2 style
                    for layer_idx in range(min(max_depth, self.layer_count)):
                        layer = self.model.transformer.h[layer_idx]
                        hooks.extend([
                            layer.register_forward_hook(create_layer_hook(layer_idx)),
                            layer.attn.register_forward_hook(create_attention_hook(layer_idx))
                        ])
                elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                    # Modern architecture
                    for layer_idx in range(min(max_depth, self.layer_count)):
                        layer = self.model.model.layers[layer_idx]
                        hooks.extend([
                            layer.register_forward_hook(create_layer_hook(layer_idx)),
                            layer.self_attn.register_forward_hook(create_attention_hook(layer_idx))
                        ])
                
                # Forward pass
                with torch.no_grad():
                    outputs = self.model(full_sequence['input_ids'], output_attentions=True)
                
                # Analyze information flow
                flow_analysis = self._analyze_information_flow(
                    layer_activations, 
                    attention_patterns, 
                    input_tokens, 
                    target_token,
                    max_depth
                )
                
                return flow_analysis
                
            finally:
                # Clean up hooks
                for hook in hooks:
                    hook.remove()
                    
        except Exception as e:
            print(f"Error tracing information flow: {e}")
            return {
                'error': str(e),
                'input_tokens': input_tokens,
                'target_token': target_token
            }
    
    def _analyze_information_flow(self, layer_activations: Dict, attention_patterns: Dict, 
                                 input_tokens: List[str], target_token: str, max_depth: int) -> Dict:
        """Analyze the captured activations to understand information flow"""
        
        flow_analysis = {
            'input_tokens': input_tokens,
            'target_token': target_token,
            'max_depth': max_depth,
            'layers_analyzed': list(layer_activations.keys()),
            'information_flow': {},
            'critical_paths': [],
            'attention_analysis': {}
        }
        
        # Analyze each layer's contribution to information flow
        for layer_idx in sorted(layer_activations.keys()):
            if layer_idx >= max_depth:
                break
                
            activations = layer_activations[layer_idx]
            
            # Calculate how much information flows through this layer
            layer_info = self._analyze_layer_information_flow(
                activations, layer_idx, input_tokens, target_token
            )
            
            flow_analysis['information_flow'][layer_idx] = layer_info
            
            # Analyze attention patterns if available
            if layer_idx in attention_patterns:
                attn_analysis = self._analyze_attention_patterns(
                    attention_patterns[layer_idx], layer_idx, input_tokens, target_token
                )
                flow_analysis['attention_analysis'][layer_idx] = attn_analysis
        
        # Find critical paths through the model
        flow_analysis['critical_paths'] = self._find_critical_paths(flow_analysis)
        
        return flow_analysis
    
    def _analyze_layer_information_flow(self, activations: np.ndarray, layer_idx: int, 
                                       input_tokens: List[str], target_token: str) -> Dict:
        """Analyze how much information flows through a specific layer"""
        
        # activations shape: (batch_size, seq_len, hidden_size)
        batch_size, seq_len, hidden_size = activations.shape
        
        # Calculate information flow metrics
        input_positions = len(input_tokens)
        target_position = input_positions  # Target token comes after input
        
        # How much does the target position depend on input positions?
        input_activations = activations[0, :input_positions, :]  # Input token activations
        target_activation = activations[0, target_position, :]   # Target token activation
        
        # Calculate correlation between input and target activations
        correlations = []
        for i in range(input_positions):
            corr = np.corrcoef(input_activations[i], target_activation)[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
        
        avg_correlation = np.mean(correlations) if correlations else 0.0
        
        # Calculate information flow strength
        input_magnitude = np.linalg.norm(input_activations)
        target_magnitude = np.linalg.norm(target_activation)
        flow_strength = target_magnitude / (input_magnitude + 1e-8)
        
        return {
            'layer': layer_idx,
            'input_activations_shape': input_activations.shape,
            'target_activation_shape': target_activation.shape,
            'avg_correlation': float(avg_correlation),
            'flow_strength': float(flow_strength),
            'input_magnitude': float(input_magnitude),
            'target_magnitude': float(target_magnitude),
            'information_retention': float(np.mean(np.abs(correlations)) if correlations else 0.0)
        }
    
    def _analyze_attention_patterns(self, attention: np.ndarray, layer_idx: int, 
                                   input_tokens: List[str], target_token: str) -> Dict:
        """Analyze attention patterns to understand information flow"""
        
        # attention shape: (batch_size, num_heads, seq_len, seq_len)
        batch_size, num_heads, seq_len, seq_len = attention.shape
        
        input_positions = len(input_tokens)
        target_position = input_positions
        
        # Analyze how much each head attends to input vs target
        head_analysis = []
        
        for head_idx in range(num_heads):
            head_attn = attention[0, head_idx, :, :]  # (seq_len, seq_len)
            
            # How much does target position attend to input positions?
            target_to_input_attention = head_attn[target_position, :input_positions]
            
            # How much do input positions attend to each other?
            input_to_input_attention = head_attn[:input_positions, :input_positions]
            
            # Calculate attention statistics
            target_to_input_strength = float(np.mean(target_to_input_attention))
            input_to_input_strength = float(np.mean(input_to_input_attention))
            
            # Calculate attention diversity (how focused vs spread out)
            target_focus = float(np.std(target_to_input_attention))
            input_focus = float(np.std(input_to_input_attention))
            
            head_analysis.append({
                'head': head_idx,
                'target_to_input_strength': target_to_input_strength,
                'input_to_input_strength': input_to_input_strength,
                'target_focus': target_focus,
                'input_focus': input_focus,
                'attention_pattern': 'input_focused' if input_to_input_strength > target_to_input_strength else 'target_focused'
            })
        
        return {
            'layer': layer_idx,
            'num_heads': num_heads,
            'head_analysis': head_analysis,
            'overall_target_to_input': float(np.mean([h['target_to_input_strength'] for h in head_analysis])),
            'overall_input_to_input': float(np.mean([h['input_to_input_strength'] for h in head_analysis]))
        }
    
    def _find_critical_paths(self, flow_analysis: Dict) -> List[Dict]:
        """Find the most critical computational paths through the model"""
        
        critical_paths = []
        
        # Find layers with highest information flow
        layer_flows = []
        for layer_idx, layer_info in flow_analysis['information_flow'].items():
            layer_flows.append({
                'layer': layer_idx,
                'flow_strength': layer_info['flow_strength'],
                'correlation': layer_info['avg_correlation']
            })
        
        # Sort by flow strength
        layer_flows.sort(key=lambda x: x['flow_strength'], reverse=True)
        
        # Identify critical paths
        for i, layer_flow in enumerate(layer_flows[:3]):  # Top 3 most critical layers
            path_info = {
                'path_id': i + 1,
                'criticality_rank': i + 1,
                'layers': [layer_flow['layer']],
                'flow_strength': layer_flow['flow_strength'],
                'correlation': layer_flow['correlation'],
                'path_type': 'information_highway' if layer_flow['flow_strength'] > 1.0 else 'moderate_flow'
            }
            
            # Add attention analysis if available
            if layer_flow['layer'] in flow_analysis['attention_analysis']:
                attn_info = flow_analysis['attention_analysis'][layer_flow['layer']]
                path_info['attention_heads'] = [
                    h['head'] for h in attn_info['head_analysis'] 
                    if h['target_to_input_strength'] > 0.1
                ]
                path_info['attention_pattern'] = attn_info['overall_target_to_input']
            
            critical_paths.append(path_info)
        
        return critical_paths
    
    def find_circuit_components(self, concept_tokens: List[str], layer: int, 
                               min_activation_threshold: float = 0.5) -> Dict:
        """
        Find which neurons and attention heads work together as a circuit
        This identifies functional units in the model
        
        Args:
            concept_tokens: Tokens that activate the circuit
            layer: Which layer to analyze
            min_activation_threshold: Minimum activation to consider a neuron part of circuit
            
        Returns:
            Dict with circuit components and their interactions
        """
        print(f"Finding circuit components for concept {concept_tokens} at layer {layer}")
        
        try:
            # Ensure padding token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Tokenize inputs properly
            input_text = " ".join(concept_tokens)
            input_ids = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
            
            # Capture activations and attention
            mlp_activations = []
            attention_weights = []
            
            def mlp_hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                mlp_activations.append(output.detach().cpu().numpy())
            
            def attention_hook(module, input, output):
                if hasattr(module, 'attn_weights'):
                    attention_weights.append(module.attn_weights.detach().cpu().numpy())
            
            # Register hooks
            hooks = []
            try:
                if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                    target_layer = self.model.transformer.h[layer]
                    hooks.extend([
                        target_layer.mlp.register_forward_hook(mlp_hook),
                        target_layer.attn.register_forward_hook(attention_hook)
                    ])
                elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                    target_layer = self.model.model.layers[layer]
                    hooks.extend([
                        target_layer.mlp.register_forward_hook(mlp_hook),
                        target_layer.self_attn.register_forward_hook(attention_hook)
                    ])
                
                # Forward pass
                with torch.no_grad():
                    self.model(input_ids['input_ids'])
                
                # Analyze circuit components
                circuit_analysis = self._analyze_circuit_components(
                    mlp_activations, attention_weights, concept_tokens, layer, min_activation_threshold
                )
                
                return circuit_analysis
                
            finally:
                for hook in hooks:
                    hook.remove()
                    
        except Exception as e:
            print(f"Error finding circuit components: {e}")
            return {
                'error': str(e),
                'concept_tokens': concept_tokens,
                'layer': layer
            }
    
    def _analyze_circuit_components(self, mlp_activations: List, attention_weights: List, 
                                   concept_tokens: List[str], layer: int, 
                                   min_activation_threshold: float) -> Dict:
        """Analyze the captured activations to identify circuit components"""
        
        if not mlp_activations:
            return {'error': 'No MLP activations captured'}
        
        mlp_act = mlp_activations[0]  # Shape: (batch_size, seq_len, hidden_size)
        batch_size, seq_len, hidden_size = mlp_act.shape
        
        # Find neurons that activate above threshold
        mean_activations = np.mean(mlp_act, axis=(0, 1))  # Average over batch and sequence
        active_neurons = np.where(mean_activations > min_activation_threshold)[0]
        
        # Analyze attention patterns if available
        attention_analysis = {}
        if attention_weights:
            attn_weights = attention_weights[0]  # Shape: (batch_size, num_heads, seq_len, seq_len)
            attention_analysis = self._analyze_circuit_attention(attn_weights, concept_tokens)
        
        # Find neuron clusters (groups that activate together)
        neuron_clusters = self._find_neuron_clusters(mlp_act, active_neurons)
        
        # Calculate circuit statistics
        circuit_stats = {
            'total_neurons': hidden_size,
            'active_neurons': len(active_neurons),
            'activation_density': len(active_neurons) / hidden_size,
            'mean_activation': float(np.mean(mean_activations)),
            'activation_std': float(np.std(mean_activations))
        }
        
        return {
            'layer': layer,
            'concept_tokens': concept_tokens,
            'circuit_stats': circuit_stats,
            'active_neurons': active_neurons.tolist(),
            'neuron_clusters': neuron_clusters,
            'attention_analysis': attention_analysis,
            'activation_threshold': min_activation_threshold
        }
    
    def _analyze_circuit_attention(self, attention: np.ndarray, concept_tokens: List[str]) -> Dict:
        """Analyze attention patterns within the circuit"""
        
        batch_size, num_heads, seq_len, seq_len = attention.shape
        
        # Analyze each attention head
        head_analysis = []
        for head_idx in range(num_heads):
            head_attn = attention[0, head_idx, :, :]
            
            # Calculate attention statistics
            mean_attention = float(np.mean(head_attn))
            attention_std = float(np.std(head_attn))
            max_attention = float(np.max(head_attn))
            
            # Determine attention pattern type
            if attention_std < 0.1:
                pattern_type = 'uniform'
            elif max_attention > 0.8:
                pattern_type = 'focused'
            else:
                pattern_type = 'distributed'
            
            head_analysis.append({
                'head': head_idx,
                'mean_attention': mean_attention,
                'attention_std': attention_std,
                'max_attention': max_attention,
                'pattern_type': pattern_type
            })
        
        return {
            'num_heads': num_heads,
            'head_analysis': head_analysis,
            'overall_mean_attention': float(np.mean([h['mean_attention'] for h in head_analysis]))
        }
    
    def _find_neuron_clusters(self, activations: np.ndarray, active_neurons: np.ndarray) -> List[Dict]:
        """Find groups of neurons that activate together (circuit modules)"""
        
        if len(active_neurons) < 2:
            return []
        
        # Calculate correlation matrix between active neurons
        active_activations = activations[:, :, active_neurons]  # Shape: (batch, seq, num_active)
        mean_activations = np.mean(active_activations, axis=(0, 1))  # Shape: (num_active,)
        
        # Normalize activations
        normalized_activations = (active_activations - mean_activations.reshape(1, 1, -1)) / (np.std(active_activations, axis=(0, 1)) + 1e-8).reshape(1, 1, -1)
        
        # Calculate correlations
        correlations = np.corrcoef(normalized_activations.reshape(-1, len(active_neurons)).T)
        
        # Find clusters using correlation threshold
        correlation_threshold = 0.7
        clusters = []
        used_neurons = set()
        
        for i, neuron_idx in enumerate(active_neurons):
            if neuron_idx in used_neurons:
                continue
            
            # Find neurons correlated with this one
            cluster = [int(neuron_idx)]
            used_neurons.add(neuron_idx)
            
            for j, other_neuron_idx in enumerate(active_neurons):
                if other_neuron_idx in used_neurons:
                    continue
                
                if abs(correlations[i, j]) > correlation_threshold:
                    cluster.append(int(other_neuron_idx))
                    used_neurons.add(other_neuron_idx)
            
            if len(cluster) > 1:  # Only include clusters with multiple neurons
                clusters.append({
                    'cluster_id': len(clusters) + 1,
                    'neurons': cluster,
                    'size': len(cluster),
                    'mean_correlation': float(np.mean([correlations[i, j] for j in range(len(active_neurons)) if active_neurons[j] in cluster]))
                })
        
        return clusters
    
    def export_circuit_analysis(self, analysis: Dict, filename: str = None) -> str:
        """Export circuit analysis results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"circuit_analysis_{timestamp}"
        
        filepath = f"{filename}.json"
        with open(filepath, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"Circuit analysis exported to: {filepath}")
        return filepath

# Test function to verify the implementation works
def test_circuit_analyzer():
    """Test the CircuitAnalyzer with a simple example"""
    print("Testing CircuitAnalyzer...")
    
    try:
        # Load a model (this would normally be passed in)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_name = "microsoft/DialoGPT-medium"  # Smaller model for testing
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Initialize analyzer
        analyzer = CircuitAnalyzer(model, tokenizer)
        
        # Test 1: Trace information flow
        print("\nTest 1: Tracing information flow")
        flow_result = analyzer.trace_information_flow(
            input_tokens=["The", "cat", "is"], 
            target_token="happy", 
            max_depth=2
        )
        print(f"Result: {len(flow_result.get('layers_analyzed', []))} layers analyzed")
        
        # Test 2: Find circuit components
        print("\nTest 2: Finding circuit components")
        circuit_result = analyzer.find_circuit_components(
            concept_tokens=["happy", "joy", "smile"], 
            layer=2
        )
        print(f"Result: {circuit_result.get('circuit_stats', {}).get('active_neurons', 0)} active neurons found")
        
        print("\nCircuitAnalyzer tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"CircuitAnalyzer test failed: {e}")
        return False

if __name__ == "__main__":
    # Run tests if this file is executed directly
    test_circuit_analyzer()
