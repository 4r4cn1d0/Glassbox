#!/usr/bin/env python3
"""
InterventionTester - Mechanistic Interpretability Feature #3

This module implements causal analysis techniques by modifying model components
through ablation studies and intervention testing.
"""

import torch
import torch.nn as nn
import numpy as np
import json
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime
import os


class InterventionTester:
    """
    Tests causal relationships in language models through systematic interventions.
    
    This class implements mechanistic interpretability techniques that enable:
    1. Attention head ablation to analyze their impact on model outputs
    2. Neuron importance testing through targeted zeroing operations
    3. Causal relationship analysis between model components
    4. Hypothesis validation about model behavior and architecture
    """
    
    def __init__(self, model: nn.Module, tokenizer):
        """
        Initialize the InterventionTester.
        
        Args:
            model: The language model to test
            tokenizer: The tokenizer for the model
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        
        # Initialize model architecture information
        self.layer_count = self._get_layer_count()
        self.hidden_size = self._get_hidden_size()
        self.num_heads = self._get_num_heads()
        
        # Store original model state for restoration
        self.original_state = {}
        self.intervention_active = False
        
        print(f"InterventionTester initialized for {self.layer_count} layers, {self.hidden_size} hidden size, {self.num_heads} heads")
    
    def _get_layer_count(self) -> int:
        """Retrieve the total number of transformer layers in the model."""
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return len(self.model.transformer.h)
        elif hasattr(self.model, 'layers'):
            return len(self.model.layers)
        else:
            raise ValueError("Cannot determine layer count - unsupported model architecture")
    
    def _get_hidden_size(self) -> int:
        """Retrieve the hidden size of the model."""
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'wte'):
            return self.model.transformer.wte.embedding_dim
        elif hasattr(self.model, 'embed_dim'):
            return self.model.embed_dim
        else:
            raise ValueError("Cannot determine hidden size - unsupported model architecture")
    
    def _get_num_heads(self) -> int:
        """Retrieve the number of attention heads in the model."""
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return self.model.transformer.h[0].attn.num_heads
        elif hasattr(self.model, 'layers'):
            return self.model.layers[0].self_attn.num_heads
        else:
            raise ValueError("Cannot determine number of attention heads - unsupported model architecture")
    
    def _save_original_state(self, layer_idx: int, head_idx: Optional[int] = None, neuron_idx: Optional[int] = None):
        """Save the original state of a component before intervention."""
        if head_idx is not None:
            # Save attention head state
            key = f"layer_{layer_idx}_head_{head_idx}"
            if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                layer = self.model.transformer.h[layer_idx]
                if hasattr(layer, 'attn'):
                    # Save attention weights and biases
                    self.original_state[key] = {
                        'attn_c_attn_weight': layer.attn.c_attn.weight.clone(),
                        'attn_c_attn_bias': layer.attn.c_attn.bias.clone() if layer.attn.c_attn.bias is not None else None,
                        'attn_c_proj_weight': layer.attn.c_proj.weight.clone(),
                        'attn_c_proj_bias': layer.attn.c_proj.bias.clone() if layer.attn.c_proj.bias is not None else None,
                    }
        
        elif neuron_idx is not None:
            # Save MLP neuron state
            key = f"layer_{layer_idx}_neuron_{neuron_idx}"
            if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                layer = self.model.transformer.h[layer_idx]
                if hasattr(layer, 'mlp'):
                    # Save MLP weights and biases
                    self.original_state[key] = {
                        'mlp_c_fc_weight': layer.mlp.c_fc.weight.clone(),
                        'mlp_c_fc_bias': layer.mlp.c_fc.bias.clone() if layer.mlp.c_fc.bias is not None else None,
                        'mlp_c_proj_weight': layer.mlp.c_proj.weight.clone(),
                        'mlp_c_proj_bias': layer.mlp.c_proj.bias.clone() if layer.mlp.c_proj.bias is not None else None,
                    }
    
    def _restore_original_state(self, layer_idx: int, head_idx: Optional[int] = None, neuron_idx: Optional[int] = None):
        """Restore the original state of a component after intervention."""
        if head_idx is not None:
            key = f"layer_{layer_idx}_head_{head_idx}"
            if key in self.original_state and hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                layer = self.model.transformer.h[layer_idx]
                if hasattr(layer, 'attn'):
                    state = self.original_state[key]
                    layer.attn.c_attn.weight.data = state['attn_c_attn_weight']
                    if state['attn_c_attn_bias'] is not None:
                        layer.attn.c_attn.bias.data = state['attn_c_attn_bias']
                    layer.attn.c_proj.weight.data = state['attn_c_proj_weight']
                    if state['attn_c_proj_bias'] is not None:
                        layer.attn.c_proj.bias.data = state['attn_c_proj_bias']
        
        elif neuron_idx is not None:
            key = f"layer_{layer_idx}_neuron_{neuron_idx}"
            if key in self.original_state and hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                layer = self.model.transformer.h[layer_idx]
                if hasattr(layer, 'mlp'):
                    state = self.original_state[key]
                    layer.mlp.c_fc.weight.data = state['mlp_c_fc_weight']
                    if state['mlp_c_fc_bias'] is not None:
                        layer.mlp.c_fc.bias.data = state['mlp_c_fc_bias']
                    layer.mlp.c_proj.weight.data = state['mlp_c_proj_weight']
                    if state['mlp_c_proj_bias'] is not None:
                        layer.mlp.c_proj.bias.data = state['mlp_c_proj_bias']
    
    def ablate_attention_head(self, layer_idx: int, head_idx: int, test_prompts: List[str], 
                             max_length: int = 50) -> Dict:
        """
        Ablate (zero out) a specific attention head and measure its impact.
        
        Args:
            layer_idx: Layer index (0-based)
            head_idx: Attention head index (0-based)
            test_prompts: List of prompts to test
            max_length: Maximum generation length
            
        Returns:
            Dictionary with ablation results and impact metrics
        """
        print(f"Ablating attention head {head_idx} at layer {layer_idx}")
        
        if not (0 <= layer_idx < self.layer_count):
            raise ValueError(f"Layer index {layer_idx} out of range [0, {self.layer_count})")
        
        if not (0 <= head_idx < self.num_heads):
            raise ValueError(f"Head index {head_idx} out of range [0, {self.num_heads})")
        
        # Save original state
        self._save_original_state(layer_idx, head_idx=head_idx)
        
        try:
            # Get baseline outputs
            baseline_outputs = self._generate_baseline_outputs(test_prompts, max_length)
            
            # Apply ablation
            self._apply_attention_head_ablation(layer_idx, head_idx)
            
            # Get ablated outputs
            ablated_outputs = self._generate_ablated_outputs(test_prompts, max_length)
            
            # Calculate impact metrics
            impact_metrics = self._calculate_ablation_impact(baseline_outputs, ablated_outputs)
            
            # Restore original state
            self._restore_original_state(layer_idx, head_idx=head_idx)
            
            return {
                'intervention_type': 'attention_head_ablation',
                'layer_idx': layer_idx,
                'head_idx': head_idx,
                'test_prompts': test_prompts,
                'baseline_outputs': baseline_outputs,
                'ablated_outputs': ablated_outputs,
                'impact_metrics': impact_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            # Restore original state on error
            self._restore_original_state(layer_idx, head_idx=head_idx)
            raise e
    
    def _apply_attention_head_ablation(self, layer_idx: int, head_idx: int):
        """Apply ablation to a specific attention head."""
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            layer = self.model.transformer.h[layer_idx]
            if hasattr(layer, 'attn'):
                # Zero out the attention head weights
                head_size = self.hidden_size // self.num_heads
                start_idx = head_idx * head_size
                end_idx = start_idx + head_size
                
                # Zero out the query, key, value projections for this head
                with torch.no_grad():
                    layer.attn.c_attn.weight.data[start_idx:end_idx, :] = 0.0
                    if layer.attn.c_attn.bias is not None:
                        layer.attn.c_attn.bias.data[start_idx:end_idx] = 0.0
                    
                    # Zero out the output projection for this head
                    layer.attn.c_proj.weight.data[:, start_idx:end_idx] = 0.0
                    if layer.attn.c_proj.bias is not None:
                        layer.attn.c_proj.bias.data[:] = 0.0  # Zero entire bias for this head
    
    def ablate_mlp_neuron(self, layer_idx: int, neuron_idx: int, test_prompts: List[str], 
                          max_length: int = 50) -> Dict:
        """
        Ablate (zero out) a specific MLP neuron and measure its impact.
        
        Args:
            layer_idx: Layer index (0-based)
            neuron_idx: Neuron index (0-based)
            test_prompts: List of prompts to test
            max_length: Maximum generation length
            
        Returns:
            Dictionary with ablation results and impact metrics
        """
        print(f"Ablating MLP neuron {neuron_idx} at layer {layer_idx}")
        
        if not (0 <= layer_idx < self.layer_count):
            raise ValueError(f"Layer index {layer_idx} out of range [0, {self.layer_count})")
        
        if not (0 <= neuron_idx < self.hidden_size):
            raise ValueError(f"Neuron index {neuron_idx} out of range [0, {self.hidden_size})")
        
        # Save original state
        self._save_original_state(layer_idx, neuron_idx=neuron_idx)
        
        try:
            # Get baseline outputs
            baseline_outputs = self._generate_baseline_outputs(test_prompts, max_length)
            
            # Apply ablation
            self._apply_mlp_neuron_ablation(layer_idx, neuron_idx)
            
            # Get ablated outputs
            ablated_outputs = self._generate_ablated_outputs(test_prompts, max_length)
            
            # Calculate impact metrics
            impact_metrics = self._calculate_ablation_impact(baseline_outputs, ablated_outputs)
            
            # Restore original state
            self._restore_original_state(layer_idx, neuron_idx=neuron_idx)
            
            return {
                'intervention_type': 'mlp_neuron_ablation',
                'layer_idx': layer_idx,
                'neuron_idx': neuron_idx,
                'test_prompts': test_prompts,
                'baseline_outputs': baseline_outputs,
                'ablated_outputs': ablated_outputs,
                'impact_metrics': impact_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            # Restore original state on error
            self._restore_original_state(layer_idx, neuron_idx=neuron_idx)
            raise e
    
    def _apply_mlp_neuron_ablation(self, layer_idx: int, neuron_idx: int):
        """Apply ablation to a specific MLP neuron."""
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            layer = self.model.transformer.h[layer_idx]
            if hasattr(layer, 'mlp'):
                # Zero out the neuron in the first MLP layer
                with torch.no_grad():
                    layer.mlp.c_fc.weight.data[neuron_idx, :] = 0.0
                    if layer.mlp.c_fc.bias is not None:
                        layer.mlp.c_fc.bias.data[neuron_idx] = 0.0
                    
                    # Zero out the neuron in the second MLP layer
                    layer.mlp.c_proj.weight.data[:, neuron_idx] = 0.0
                    if layer.mlp.c_proj.bias is not None:
                        layer.mlp.c_proj.bias.data[:] = 0.0  # Zero entire bias
    
    def _generate_baseline_outputs(self, test_prompts: List[str], max_length: int) -> List[str]:
        """Generate baseline outputs without any interventions."""
        outputs = []
        
        for prompt in test_prompts:
            try:
                # Tokenize input
                inputs = self.tokenizer.encode(prompt, return_tensors="pt")
                
                # Generate text
                with torch.no_grad():
                    generated = self.model.generate(
                        inputs,
                        max_length=max_length,
                        pad_token_id=self.tokenizer.eos_token_id,
                        do_sample=False
                    )
                
                # Decode output
                output_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                outputs.append(output_text)
                
            except Exception as e:
                print(f"Warning: Failed to generate baseline for prompt '{prompt}': {e}")
                outputs.append(f"[ERROR: {str(e)}]")
        
        return outputs
    
    def _generate_ablated_outputs(self, test_prompts: List[str], max_length: int) -> List[str]:
        """Generate outputs with ablation applied."""
        # This method is the same as baseline since ablation is already applied
        return self._generate_baseline_outputs(test_prompts, max_length)
    
    def _calculate_ablation_impact(self, baseline_outputs: List[str], ablated_outputs: List[str]) -> Dict:
        """Calculate metrics to measure the impact of ablation."""
        if len(baseline_outputs) != len(ablated_outputs):
            raise ValueError("Baseline and ablated outputs must have the same length")
        
        impact_metrics = {
            'total_prompts': len(baseline_outputs),
            'output_changes': [],
            'token_changes': [],
            'semantic_changes': []
        }
        
        for i, (baseline, ablated) in enumerate(zip(baseline_outputs, ablated_outputs)):
            # Check if output changed
            output_changed = baseline != ablated
            impact_metrics['output_changes'].append(output_changed)
            
            if output_changed:
                # Count token changes
                baseline_tokens = len(self.tokenizer.encode(baseline))
                ablated_tokens = len(self.tokenizer.encode(ablated))
                token_change = abs(baseline_tokens - ablated_tokens)
                impact_metrics['token_changes'].append(token_change)
                
                # Simple semantic change detection (output length difference)
                semantic_change = abs(len(baseline) - len(ablated))
                impact_metrics['semantic_changes'].append(semantic_change)
            else:
                impact_metrics['token_changes'].append(0)
                impact_metrics['semantic_changes'].append(0)
        
        # Calculate summary statistics
        impact_metrics['outputs_changed'] = sum(impact_metrics['output_changes'])
        impact_metrics['change_rate'] = impact_metrics['outputs_changed'] / len(baseline_outputs)
        
        if impact_metrics['token_changes']:
            impact_metrics['avg_token_change'] = np.mean(impact_metrics['token_changes'])
            impact_metrics['max_token_change'] = max(impact_metrics['token_changes'])
        
        if impact_metrics['semantic_changes']:
            impact_metrics['avg_semantic_change'] = np.mean(impact_metrics['semantic_changes'])
            impact_metrics['max_semantic_change'] = max(impact_metrics['semantic_changes'])
        
        return impact_metrics
    
    def test_attention_head_importance(self, layer_idx: int, test_prompts: List[str], 
                                     max_length: int = 50) -> Dict:
        """
        Test the importance of all attention heads in a layer.
        
        Args:
            layer_idx: Layer index to test
            test_prompts: List of prompts to test
            max_length: Maximum generation length
            
        Returns:
            Dictionary with importance scores for each head
        """
        print(f"Testing attention head importance for layer {layer_idx}")
        
        head_importance = {}
        
        for head_idx in range(self.num_heads):
            try:
                result = self.ablate_attention_head(layer_idx, head_idx, test_prompts, max_length)
                impact = result['impact_metrics']
                
                # Calculate importance score (higher change = more important)
                importance_score = impact['change_rate'] * 100  # Convert to percentage
                
                head_importance[f"head_{head_idx}"] = {
                    'importance_score': importance_score,
                    'outputs_changed': impact['outputs_changed'],
                    'change_rate': impact['change_rate'],
                    'avg_token_change': impact.get('avg_token_change', 0),
                    'avg_semantic_change': impact.get('avg_semantic_change', 0)
                }
                
                print(f"   Head {head_idx}: {importance_score:.1f}% importance")
                
            except Exception as e:
                print(f"   Warning: Failed to test head {head_idx}: {e}")
                head_importance[f"head_{head_idx}"] = {
                    'error': str(e),
                    'importance_score': 0.0
                }
        
        # Sort heads by importance
        sorted_heads = sorted(
            head_importance.items(),
            key=lambda x: x[1].get('importance_score', 0),
            reverse=True
        )
        
        return {
            'layer_idx': layer_idx,
            'total_heads': self.num_heads,
            'head_importance': dict(sorted_heads),
            'most_important_head': sorted_heads[0][0] if sorted_heads else None,
            'least_important_head': sorted_heads[-1][0] if sorted_heads else None,
            'timestamp': datetime.now().isoformat()
        }
    
    def test_neuron_importance(self, layer_idx: int, neuron_indices: List[int], 
                              test_prompts: List[str], max_length: int = 50) -> Dict:
        """
        Test the importance of specific neurons in a layer.
        
        Args:
            layer_idx: Layer index to test
            neuron_indices: List of neuron indices to test
            test_prompts: List of prompts to test
            max_length: Maximum generation length
            
        Returns:
            Dictionary with importance scores for each neuron
        """
        print(f"Testing neuron importance for layer {layer_idx}")
        
        neuron_importance = {}
        
        for neuron_idx in neuron_indices:
            try:
                result = self.ablate_mlp_neuron(layer_idx, neuron_idx, test_prompts, max_length)
                impact = result['impact_metrics']
                
                # Calculate importance score
                importance_score = impact['change_rate'] * 100
                
                neuron_importance[f"neuron_{neuron_idx}"] = {
                    'importance_score': importance_score,
                    'outputs_changed': impact['outputs_changed'],
                    'change_rate': impact['change_rate'],
                    'avg_token_change': impact.get('avg_token_change', 0),
                    'avg_semantic_change': impact.get('avg_semantic_change', 0)
                }
                
                print(f"   Neuron {neuron_idx}: {importance_score:.1f}% importance")
                
            except Exception as e:
                print(f"   Warning: Failed to test neuron {neuron_idx}: {e}")
                neuron_importance[f"neuron_{neuron_idx}"] = {
                    'error': str(e),
                    'importance_score': 0.0
                }
        
        # Sort neurons by importance
        sorted_neurons = sorted(
            neuron_importance.items(),
            key=lambda x: x[1].get('importance_score', 0),
            reverse=True
        )
        
        return {
            'layer_idx': layer_idx,
            'total_neurons_tested': len(neuron_indices),
            'neuron_importance': dict(sorted_neurons),
            'most_important_neuron': sorted_neurons[0][0] if sorted_neurons else None,
            'least_important_neuron': sorted_neurons[-1][0] if sorted_neurons else None,
            'timestamp': datetime.now().isoformat()
        }
    
    def export_intervention_results(self, results: Dict, filename: str = None) -> str:
        """
        Export intervention testing results to a JSON file.
        
        Args:
            results: Results dictionary from any intervention test
            filename: Optional custom filename
            
        Returns:
            Path to the exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            intervention_type = results.get('intervention_type', 'intervention')
            filename = f"intervention_{intervention_type}_{timestamp}.json"
        
        # Ensure exports directory exists
        exports_dir = "exports"
        if not os.path.exists(exports_dir):
            os.makedirs(exports_dir)
        
        filepath = os.path.join(exports_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"Intervention results exported to {filepath}")
            return filepath
            
        except Exception as e:
            print(f"Failed to export results: {e}")
            return None
    
    def get_model_info(self) -> Dict:
        """Get information about the model architecture."""
        return {
            'model_type': 'GPT-2 Large',
            'layers': self.layer_count,
            'hidden_size': self.hidden_size,
            'num_heads': self.num_heads,
            'intervention_tester_ready': True
        }
