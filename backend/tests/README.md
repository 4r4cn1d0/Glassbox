# Glassbox Test Suite

This directory contains the organized test suite for Glassbox, a mechanistic interpretability platform for AI safety research.

## Test Structure

The test suite is organized into two categories:

### Individual Feature Tests

Each feature has its own dedicated test file that tests the feature in isolation:

- **`test_feature_analyzer.py`** - Tests FeatureAnalyzer functionality
  - Feature neuron identification
  - Concept-distinguishing neuron analysis
  - Single neuron analysis
  - Layer representation retrieval
  - Export functionality

- **`test_circuit_analyzer.py`** - Tests CircuitAnalyzer functionality
  - Information flow tracing
  - Circuit component identification
  - Attention pattern analysis
  - Export functionality

- **`test_intervention_tester.py`** - Tests InterventionTester functionality
  - Attention head ablation
  - MLP neuron ablation
  - Neuron importance testing
  - Export functionality

- **`test_safety_analyzer.py`** - Tests SafetyAnalyzer functionality
  - Uncertainty detection
  - Consistency analysis
  - Deception detection
  - Comprehensive safety audit
  - Export functionality

### Comprehensive Integration Test

- **`test_comprehensive.py`** - Tests all features working together
  - Cross-feature workflows
  - Coordinated analysis pipelines
  - End-to-end research scenarios
  - Combined result export

## Running Tests

### Individual Feature Tests

To test a specific feature:

```bash
cd backend/tests
python test_feature_analyzer.py
python test_circuit_analyzer.py
python test_intervention_tester.py
python test_safety_analyzer.py
```

### Comprehensive Test

To test all features together:

```bash
cd backend/tests
python test_comprehensive.py
```

## Test Requirements

All tests require:
- GPT-2 Large model (automatically downloaded)
- All core dependencies installed
- Proper directory structure with `core/` modules accessible

## Test Output

Tests provide detailed output including:
- Step-by-step progress indicators
- Success/failure status for each test
- Detailed error messages with tracebacks
- Export of results to JSON files

## Test Philosophy

The test suite follows these principles:
- **Isolation**: Individual tests focus on single features
- **Integration**: Comprehensive tests verify cross-feature workflows
- **Robustness**: Tests handle errors gracefully and provide clear feedback
- **Research-Ready**: Tests simulate real-world mechanistic interpretability research scenarios
