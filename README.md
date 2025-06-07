# Glassbox: Visual LLM Debugger

An open-source visual debugger for Large Language Models that provides real-time insights into token generation, attention patterns, and model behavior.

## Features

🔍 **Token-by-Token Analysis**: Trace the generation process step by step
🎯 **Attention Visualization**: Interactive heatmaps showing attention patterns
📊 **Probability Distribution**: Visual representation of token probabilities
🕸️ **Attention Networks**: Spider web visualizations of attention flows
⚡ **Real-Time Debugging**: Live analysis as models generate text

## Architecture

- **Backend**: Python/FastAPI server with HuggingFace Transformers integration
- **Frontend**: React/TypeScript with Material-UI for interactive visualizations
- **Models**: Supports GPT-2, GPT-3.5, and other transformer architectures

## Quick Start

### Backend Setup

1. Create a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

3. Run the FastAPI server:
```bash
python app.py
```

The server will start at http://localhost:8000

### Frontend Setup

Coming soon...

## API Endpoints

### POST /api/trace

Generates text and returns token-by-token trace data.

Request body:
```json
{
    "prompt": "Your prompt here",
    "max_new_tokens": 50
}
```

Response:
```json
[
    {
        "token": "generated",
        "token_id": 123,
        "logits": [...],
        "attention": [...]
    },
    ...
]
```

## Usage

1. Start both backend (port 8000) and frontend (port 3000)
2. Enter your prompt in the web interface
3. Click "TRACE GENERATION" to see real-time analysis
4. Explore attention patterns and token probabilities

## Research Applications

This tool is designed for:
- LLM interpretability research
- Debugging model behavior
- Educational demonstrations
- Academic papers on transformer analysis

## Development

This is an early-stage project. Contributions are welcome!

## License

MIT License - Open source for research and educational use. 