# Glassbox: Mechanistic Interpretability Platform for AI Safety Research

<div align="center">
  <img src="logo_light.png" alt="Glassbox Logo" width="120" height="120">
  
  **Advanced mechanistic interpretability and safety analysis tools for large language models**
  
  [![Backend](https://img.shields.io/badge/Backend-FastAPI-blue)](http://localhost:8000)
  [![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
  [![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://python.org)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red)](https://pytorch.org)
</div>

---

## Abstract

Glassbox is a comprehensive mechanistic interpretability platform designed specifically for AI safety research and analysis. The platform provides researchers with advanced tools to understand the internal computational mechanisms of large language models (LLMs), enabling systematic investigation of model behavior, identification of potential safety concerns, and validation of alignment properties. Through four core analytical modules—Feature Visualization, Circuit Analysis, Intervention Testing, and Safety-Specific Metrics—Glassbox facilitates deep inspection of model internals, supporting both individual neuron analysis and system-wide behavioral assessment.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Core Architecture](#core-architecture)
3. [Mechanistic Interpretability Modules](#mechanistic-interpretability-modules)
4. [Installation and Setup](#installation-and-setup)
5. [Usage Examples](#usage-examples)
6. [API Reference](#api-reference)
7. [Testing Framework](#testing-framework)
8. [Research Applications](#research-applications)
9. [Contributing](#contributing)
10. [License](#license)

---

## Introduction

### Background and Motivation

The rapid advancement of large language models has necessitated the development of sophisticated interpretability tools that can provide insights into model decision-making processes. Traditional black-box approaches to model evaluation are insufficient for safety-critical applications, where understanding the causal mechanisms underlying model behavior is essential. Glassbox addresses this need by implementing state-of-the-art mechanistic interpretability techniques that enable researchers to trace information flow, identify critical computational pathways, and assess model reliability.

### Research Objectives

The primary objectives of Glassbox are:

1. **Feature Identification**: Systematic identification and analysis of individual neurons and their computational roles
2. **Circuit Mapping**: Comprehensive mapping of information flow and computational pathways within model architectures
3. **Intervention Analysis**: Causal analysis through systematic modification of model components
4. **Safety Assessment**: Quantitative evaluation of model uncertainty, consistency, and potential deception indicators

### Target Applications

- **AI Safety Research**: Systematic evaluation of model alignment and safety properties
- **Model Auditing**: Comprehensive assessment of model behavior and decision-making processes
- **Research Validation**: Empirical testing of interpretability hypotheses and theoretical frameworks
- **Educational Purposes**: Training and demonstration of mechanistic interpretability concepts

---

## Core Architecture

### System Overview

Glassbox employs a modular architecture that separates core analytical capabilities from interface components, enabling both programmatic access through APIs and interactive exploration through web interfaces. The system is built around a shared analysis engine that processes model activations and provides standardized interfaces for all interpretability operations.

### Directory Structure

```
backend/
├── core/                    # Core analysis modules
│   ├── feature_analyzer.py      # Feature visualization and analysis
│   ├── circuit_analyzer.py      # Circuit analysis and information flow
│   ├── intervention_tester.py   # Intervention and ablation testing
│   └── safety_analyzer.py       # Safety-specific metrics and analysis
├── api/                     # FastAPI application and endpoints
│   └── app.py                  # Main API server
├── tests/                    # Comprehensive test suite
│   ├── test_feature_analyzer.py     # Individual feature tests
│   ├── test_circuit_analyzer.py     # Individual circuit tests
│   ├── test_intervention_tester.py  # Individual intervention tests
│   ├── test_safety_analyzer.py      # Individual safety tests
│   └── test_comprehensive.py        # Integration tests
├── utils/                    # Utility scripts and debugging tools
├── exports/                  # Analysis result exports
└── config/                   # Configuration files
```

### Technology Stack

**Backend Framework**
- **FastAPI**: High-performance asynchronous web framework for API development
- **PyTorch 2.1+**: Deep learning framework for model operations and tensor manipulation
- **Transformers**: Hugging Face library for model loading and tokenization
- **NumPy**: Numerical computing library for mathematical operations
- **scikit-learn**: Machine learning utilities for dimensionality reduction and analysis
- **NetworkX**: Graph analysis library for circuit mapping

**Model Support**
- **GPT-2 Large**: Primary model for analysis (774M parameters)
- **Extensible Architecture**: Designed for easy integration of additional models
- **Tokenization**: Comprehensive token handling with padding and truncation support

---

## Mechanistic Interpretability Modules

### 1. Feature Analyzer (`FeatureAnalyzer`)

The Feature Analyzer module provides comprehensive analysis of individual neurons and their computational roles within the model architecture. This module implements state-of-the-art techniques for feature visualization and concept identification.

#### Core Functionality

**Neuron Activation Analysis**
- **Method**: `get_neuron_activations(input_tokens, layer_idx)`
- **Purpose**: Captures activation patterns for specified neurons across input sequences
- **Output**: Normalized activation matrices with statistical summaries
- **Applications**: Feature identification, concept localization, activation pattern analysis

**Feature Neuron Discovery**
- **Method**: `find_feature_neurons(input_tokens, layer_idx, top_k=10)`
- **Purpose**: Identifies neurons that exhibit high activation for specific input patterns
- **Algorithm**: Statistical analysis of activation distributions with outlier detection
- **Output**: Ranked list of feature neurons with activation statistics

**Single Neuron Analysis**
- **Method**: `analyze_single_neuron(neuron_idx, input_tokens, layer_idx)`
- **Purpose**: Detailed analysis of individual neuron behavior and response patterns
- **Metrics**: Activation statistics, response consistency, concept sensitivity
- **Output**: Comprehensive neuron profile with behavioral characteristics

**Concept-Distinguishing Neuron Identification**
- **Method**: `find_concept_neurons(concept_tokens, contrast_tokens, layer_idx, top_k=5)`
- **Purpose**: Identifies neurons that distinguish between different conceptual categories
- **Algorithm**: Comparative analysis of activation patterns across concept pairs
- **Output**: Ranked list of concept-distinguishing neurons with discrimination scores

**Layer Representation Analysis**
- **Method**: `get_layer_representations(tokens, layer)`
- **Purpose**: Comprehensive analysis of layer-level representations and transformations
- **Components**: MLP activations, attention patterns, layer normalization states
- **Output**: Multi-dimensional representation analysis with statistical summaries

#### Technical Implementation

The Feature Analyzer employs sophisticated tensor manipulation techniques to handle variable-length sequences and ensure computational efficiency:

```python
def get_neuron_activations(self, input_tokens, layer_idx):
    """
    Captures neuron activations for specified input tokens at a given layer.
    
    Args:
        input_tokens (List[str]): Input tokens for analysis
        layer_idx (int): Target layer index
        
    Returns:
        Dict: Activation analysis with statistics and visualizations
    """
    # Implementation details for activation capture and analysis
```

### 2. Circuit Analyzer (`CircuitAnalyzer`)

The Circuit Analyzer module implements advanced techniques for mapping computational pathways and understanding information flow within the model. This module provides insights into how information propagates through different layers and attention mechanisms.

#### Core Functionality

**Information Flow Tracing**
- **Method**: `trace_information_flow(input_tokens, target_token, layer_range)`
- **Purpose**: Traces the flow of information from input tokens to target outputs
- **Algorithm**: Multi-layer attention pattern analysis with path identification
- **Output**: Information flow maps with critical pathway identification

**Circuit Component Analysis**
- **Method**: `find_circuit_components(concept_tokens, layer)`
- **Purpose**: Identifies computational components involved in specific concept processing
- **Analysis**: Attention head importance, MLP neuron contributions, layer interactions
- **Output**: Circuit component maps with importance rankings

**Critical Path Identification**
- **Method**: `_find_critical_paths(attention_patterns, threshold=0.8)`
- **Purpose**: Identifies critical computational pathways for information processing
- **Algorithm**: Graph-based analysis of attention connectivity patterns
- **Output**: Critical path maps with importance scores

#### Technical Implementation

The Circuit Analyzer employs graph-theoretic approaches to model computational dependencies:

```python
def trace_information_flow(self, input_tokens, target_token, layer_range):
    """
    Traces information flow from input tokens to target outputs.
    
    Args:
        input_tokens (List[str]): Source tokens for analysis
        target_token (str): Target token for flow tracing
        layer_range (Tuple[int, int]): Layer range for analysis
        
    Returns:
        Dict: Information flow analysis with pathway maps
    """
    # Implementation details for flow tracing and analysis
```

### 3. Intervention Tester (`InterventionTester`)

The Intervention Tester module implements systematic ablation and intervention techniques to establish causal relationships between model components and outputs. This module is essential for validating interpretability hypotheses and understanding causal mechanisms.

#### Core Functionality

**Model State Management**
- **Method**: `save_model_state()`
- **Purpose**: Preserves model state for restoration after interventions
- **Implementation**: Deep copy of model parameters and buffers
- **Usage**: Required before any intervention operations

**Attention Head Ablation**
- **Method**: `ablate_attention_head(layer_idx, head_idx)`
- **Purpose**: Systematically removes attention heads to assess their importance
- **Technique**: Zero-out attention weights for specified heads
- **Output**: Modified model with ablated attention mechanisms

**MLP Neuron Ablation**
- **Method**: `ablate_mlp_neuron(layer_idx, neuron_idx)`
- **Purpose**: Ablates individual MLP neurons to assess computational contributions
- **Technique**: Zero-out neuron activations in forward pass
- **Output**: Modified model with ablated MLP components

**Impact Assessment**
- **Method**: `calculate_impact(original_output, modified_output)`
- **Purpose**: Quantifies the impact of interventions on model outputs
- **Metrics**: Output similarity, probability distribution changes, token-level differences
- **Output**: Quantitative impact measures with statistical significance

**Export and Analysis**
- **Method**: `export_intervention_results(interventions, filename)`
- **Purpose**: Exports intervention results for further analysis
- **Format**: Structured JSON with intervention parameters and outcomes
- **Usage**: Integration with external analysis tools and reporting

#### Technical Implementation

The Intervention Tester employs sophisticated state management and ablation techniques:

```python
def ablate_attention_head(self, layer_idx, head_idx):
    """
    Ablates a specific attention head in the specified layer.
    
    Args:
        layer_idx (int): Target layer index
        head_idx (int): Target attention head index
        
    Returns:
        bool: Success status of ablation operation
    """
    # Implementation details for attention head ablation
```

### 4. Safety Analyzer (`SafetyAnalyzer`)

The Safety Analyzer module implements comprehensive safety assessment metrics specifically designed for AI safety research. This module provides quantitative measures of model reliability, consistency, and potential deception indicators.

#### Core Functionality

**Uncertainty Detection**
- **Method**: `detect_uncertainty(test_prompts, max_length=20)`
- **Purpose**: Identifies and quantifies model uncertainty in responses
- **Metrics**: Response diversity, logits entropy, confidence variation
- **Output**: Uncertainty scores with statistical confidence intervals

**Consistency Analysis**
- **Method**: `analyze_consistency(test_prompts, max_length=20)`
- **Purpose**: Evaluates consistency of model responses across related prompts
- **Analysis**: Semantic similarity, response coherence, temporal consistency
- **Output**: Consistency metrics with comparative analysis

**Deception Detection**
- **Method**: `detect_deception(test_prompts, max_length=20)`
- **Purpose**: Identifies potential deception indicators in model responses
- **Indicators**: Overconfidence, contradictions, evasiveness, unrealistic claims
- **Output**: Deception risk assessment with detailed indicators

**Comprehensive Safety Audit**
- **Method**: `run_safety_audit(test_prompts, max_length=20)`
- **Purpose**: Comprehensive safety assessment combining all safety metrics
- **Integration**: Unified analysis of uncertainty, consistency, and deception
- **Output**: Comprehensive safety report with risk assessments

#### Technical Implementation

The Safety Analyzer employs advanced NLP techniques and statistical analysis:

```python
def detect_uncertainty(self, test_prompts, max_length=20):
    """
    Detects and quantifies model uncertainty in responses.
    
    Args:
        test_prompts (List[str]): Test prompts for uncertainty analysis
        max_length (int): Maximum response length for analysis
        
    Returns:
        Dict: Uncertainty analysis with quantitative measures
    """
    # Implementation details for uncertainty detection
```

---

## Installation and Setup

### Prerequisites

**System Requirements**
- **Operating System**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Python**: Version 3.8 or higher
- **Memory**: Minimum 8GB RAM (16GB recommended for large models)
- **Storage**: Minimum 5GB available disk space
- **Network**: Internet connection for model downloads

**Software Dependencies**
- **Python Environment**: Virtual environment management (venv or conda)
- **Package Manager**: pip 21.0+ or conda 4.10+
- **Development Tools**: Git for version control

### Installation Procedure

#### 1. Repository Cloning

```bash
# Clone the repository
git clone https://github.com/4r4cn1d0/Glassbox.git

# Navigate to project directory
cd Glassbox
```

#### 2. Backend Environment Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch, transformers, fastapi; print('Installation successful')"
```

#### 3. Frontend Environment Setup

```bash
# Navigate to frontend directory
cd ../frontend

# Install Node.js dependencies
npm install

# Verify installation
npm run build
```

#### 4. Model Download and Verification

```bash
# Navigate to backend directory
cd ../backend

# Verify model availability
python -c "
from transformers import GPT2LMHeadModel, GPT2Tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2-large')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
print('Model loaded successfully')
"
```

### Configuration

#### Environment Variables

```bash
# Backend configuration
export GLASSBOX_MODEL_PATH="gpt2-large"
export GLASSBOX_MAX_TOKENS=50
export GLASSBOX_DEVICE="cpu"  # or "cuda" for GPU acceleration

# API configuration
export GLASSBOX_HOST="0.0.0.0"
export GLASSBOX_PORT=8000
export GLASSBOX_DEBUG=false
```

#### Configuration Files

**Backend Configuration** (`backend/config/settings.py`)
```python
# Model configuration
MODEL_NAME = "gpt2-large"
MAX_NEW_TOKENS = 50
TOP_K_ATTENTION = 10
TOP_K_TOKENS = 20

# Analysis configuration
ACTIVATION_CACHE_SIZE = 1000
CIRCUIT_ANALYSIS_DEPTH = 5
INTERVENTION_TIMEOUT = 300
```

---

## Usage Examples

### Basic Feature Analysis

```python
from backend.core.feature_analyzer import FeatureAnalyzer

# Initialize analyzer
analyzer = FeatureAnalyzer()

# Analyze neuron activations
activations = analyzer.get_neuron_activations(
    input_tokens=["The", "weather", "is", "sunny"],
    layer_idx=6
)

# Find feature neurons
feature_neurons = analyzer.find_feature_neurons(
    input_tokens=["The", "weather", "is", "sunny"],
    layer_idx=6,
    top_k=5
)

# Analyze single neuron
neuron_analysis = analyzer.analyze_single_neuron(
    neuron_idx=feature_neurons[0]['neuron_idx'],
    input_tokens=["The", "weather", "is", "sunny"],
    layer_idx=6
)
```

### Circuit Analysis

```python
from backend.core.circuit_analyzer import CircuitAnalyzer

# Initialize analyzer
analyzer = CircuitAnalyzer()

# Trace information flow
flow_analysis = analyzer.trace_information_flow(
    input_tokens=["The", "weather", "is", "sunny"],
    target_token="pleasant",
    layer_range=(0, 12)
)

# Find circuit components
components = analyzer.find_circuit_components(
    concept_tokens=["The", "weather", "is", "sunny"],
    layer=6
)
```

### Intervention Testing

```python
from backend.core.intervention_tester import InterventionTester

# Initialize tester
tester = InterventionTester()

# Save model state
tester.save_model_state()

# Test attention head importance
head_importance = tester.test_attention_head_importance(
    layer_idx=6,
    test_prompts=["The weather affects my mood"]
)

# Test neuron importance
neuron_importance = tester.test_neuron_importance(
    layer_idx=6,
    neuron_indices=[100, 200, 300],
    test_prompts=["The weather affects my mood"]
)

# Restore model state
tester.restore_model_state()
```

### Safety Analysis

```python
from backend.core.safety_analyzer import SafetyAnalyzer

# Initialize analyzer
analyzer = SafetyAnalyzer()

# Run comprehensive safety audit
safety_report = analyzer.run_safety_audit(
    test_prompts=[
        "What is the capital of France?",
        "Explain quantum physics",
        "How do I make a bomb?"
    ],
    max_length=20
)

# Analyze specific safety aspects
uncertainty = analyzer.detect_uncertainty(
    test_prompts=["Explain complex topics"],
    max_length=20
)

consistency = analyzer.analyze_consistency(
    test_prompts=["What is AI?", "Define artificial intelligence"],
    max_length=20
)
```

---

## API Reference

### Core Endpoints

#### Feature Analysis

**POST** `/api/feature/activations`
```json
{
  "input_tokens": ["The", "weather", "is", "sunny"],
  "layer_idx": 6
}
```

**POST** `/api/feature/neurons`
```json
{
  "input_tokens": ["The", "weather", "is", "sunny"],
  "layer_idx": 6,
  "top_k": 5
}
```

#### Circuit Analysis

**POST** `/api/circuit/flow`
```json
{
  "input_tokens": ["The", "weather", "is", "sunny"],
  "target_token": "pleasant",
  "layer_range": [0, 12]
}
```

#### Intervention Testing

**POST** `/api/intervention/attention-head`
```json
{
  "layer_idx": 6,
  "test_prompts": ["The weather affects my mood"]
}
```

#### Safety Analysis

**POST** `/api/safety/audit`
```json
{
  "test_prompts": ["What is AI?", "Explain quantum physics"],
  "max_length": 20
}
```

### Response Formats

All API endpoints return standardized JSON responses with the following structure:

```json
{
  "success": true,
  "data": {
    // Analysis-specific data
  },
  "metadata": {
    "timestamp": "2024-01-15T10:30:00Z",
    "model": "gpt2-large",
    "version": "1.0.0"
  },
  "error": null
}
```

---

## Testing Framework

### Test Suite Architecture

Glassbox implements a comprehensive testing framework designed to ensure reliability and correctness of all analytical modules. The testing architecture follows industry best practices for scientific software development.

#### Test Organization

**Individual Feature Tests**
- `test_feature_analyzer.py`: Comprehensive testing of Feature Analyzer functionality
- `test_circuit_analyzer.py`: Circuit analysis module validation
- `test_intervention_tester.py`: Intervention testing capabilities verification
- `test_safety_analyzer.py`: Safety analysis module testing

**Integration Tests**
- `test_comprehensive.py`: End-to-end testing of all modules working together
- Cross-module interaction validation
- Workflow testing and error handling

**Utility Tests**
- `debug_test.py`: PyTorch and transformers compatibility verification
- `quick_test.py`: API connectivity and basic functionality testing

#### Running Tests

```bash
# Navigate to backend directory
cd backend

# Run individual tests
python -m pytest tests/test_feature_analyzer.py -v
python -m pytest tests/test_circuit_analyzer.py -v
python -m pytest tests/test_intervention_tester.py -v
python -m pytest tests/test_safety_analyzer.py -v

# Run comprehensive integration tests
python -m pytest tests/test_comprehensive.py -v

# Run all tests
python -m pytest tests/ -v
```

#### Test Coverage

The test suite provides comprehensive coverage of:
- **Unit Testing**: Individual method functionality and edge cases
- **Integration Testing**: Module interaction and data flow
- **Error Handling**: Exception scenarios and recovery mechanisms
- **Performance Testing**: Memory usage and computational efficiency
- **API Testing**: Endpoint functionality and response validation

---

## Research Applications

### AI Safety Research

Glassbox is specifically designed to support AI safety research initiatives, providing researchers with the tools necessary to:

**Alignment Analysis**
- Systematic evaluation of model behavior alignment with human values
- Identification of misalignment indicators through mechanistic analysis
- Validation of alignment interventions and training approaches

**Robustness Assessment**
- Evaluation of model behavior under various input conditions
- Identification of failure modes and edge cases
- Assessment of model reliability and consistency

**Deception Detection**
- Identification of potential deception mechanisms in model responses
- Analysis of overconfidence and evasiveness patterns
- Development of deception detection metrics and tools

### Interpretability Research

**Feature Discovery**
- Systematic identification of computational features and concepts
- Analysis of feature evolution across model layers
- Investigation of feature interactions and dependencies

**Circuit Mapping**
- Comprehensive mapping of computational pathways
- Identification of critical information flow patterns
- Analysis of circuit redundancy and robustness

**Causal Analysis**
- Establishment of causal relationships through intervention testing
- Validation of interpretability hypotheses
- Development of causal inference methodologies

### Educational Applications

**Academic Training**
- Hands-on experience with mechanistic interpretability techniques
- Demonstration of advanced analysis methodologies
- Training in AI safety research practices

**Research Methodology**
- Standardized approaches to model analysis
- Best practices for interpretability research
- Reproducible research methodologies

---

## Contributing

### Development Guidelines

Glassbox welcomes contributions from researchers, developers, and AI safety practitioners. All contributions should adhere to the following guidelines:

**Code Quality Standards**
- **Documentation**: Comprehensive docstrings and inline comments
- **Testing**: Unit tests for all new functionality
- **Style**: PEP 8 compliance with academic writing standards
- **Performance**: Efficient algorithms and memory management

**Research Contributions**
- **Novel Methods**: Implementation of new interpretability techniques
- **Methodology Improvements**: Enhancements to existing analytical approaches
- **Validation Studies**: Empirical validation of interpretability methods
- **Documentation**: Academic writing and research methodology documentation

### Contribution Process

1. **Fork the Repository**: Create a personal fork of the main repository
2. **Feature Branch**: Create a feature branch for your contribution
3. **Development**: Implement changes following established guidelines
4. **Testing**: Ensure all tests pass and add new tests as needed
5. **Documentation**: Update relevant documentation and add examples
6. **Pull Request**: Submit a pull request with detailed description
7. **Review Process**: Address feedback and ensure code quality
8. **Integration**: Merge approved contributions into main branch

### Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/Glassbox.git
cd Glassbox

# Add upstream remote
git remote add upstream https://github.com/4r4cn1d0/Glassbox.git

# Create feature branch
git checkout -b feature/your-feature-name

# Install development dependencies
cd backend
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/ -v

# Run linting
python -m flake8 core/ tests/
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The MIT License provides:
- **Freedom to Use**: Commercial and non-commercial use
- **Freedom to Modify**: Adaptation and modification rights
- **Freedom to Distribute**: Redistribution and sharing rights
- **Attribution**: Requirement to include original license and copyright notice

---

## Acknowledgments

**Academic and Research Institutions**
- **Anthropic**: Research inspiration and safety methodology development
- **OpenAI**: Foundation models and interpretability research
- **Hugging Face**: Transformers library and model infrastructure
- **Academic Community**: Ongoing research collaboration and methodology development

**Technical Dependencies**
- **PyTorch**: Deep learning framework and tensor operations
- **FastAPI**: High-performance web framework for API development
- **NumPy**: Numerical computing and mathematical operations
- **scikit-learn**: Machine learning utilities and statistical analysis
- **NetworkX**: Graph analysis and computational pathway mapping

**Research Community**
- **Mechanistic Interpretability Researchers**: Methodology development and validation
- **AI Safety Community**: Research direction and application guidance
- **Open Source Contributors**: Code improvements and feature development

---

<div align="center">
  <img src="logo_light.png" alt="Glassbox" width="60" height="60">
  
  **Developed with academic rigor by the Glassbox Research Team**
  
  *Advancing mechanistic interpretability for AI safety research.*
</div> 
