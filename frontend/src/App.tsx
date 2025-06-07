import React, { useState } from 'react';
import { 
  Box, 
  TextField, 
  Button, 
  Paper, 
  Typography, 
  Grid, 
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem
} from '@mui/material';
import AttentionHeatmap from './AttentionHeatmap';
import TokenProbabilityBars from './TokenProbabilityBars';
import AttentionSpiderWeb from './AttentionSpiderWeb';

interface TraceData {
  token: string;
  token_id: number;
  position: number;
  logits: number[];
  attention: number[][][];
  is_generated: boolean;
}

function App() {
  const [prompt, setPrompt] = useState('');
  const [traceData, setTraceData] = useState<TraceData[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedLayer, setSelectedLayer] = useState(0);
  const [selectedHead, setSelectedHead] = useState(0);

  const handleTrace = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/trace', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt, max_new_tokens: 20 }),
      });
      const data = await response.json();
      setTraceData(data);
    } catch (error) {
      console.error('Error:', error);
    }
    setLoading(false);
  };

  const generatedText = traceData.map(item => item.token).join('');
  const tokens = prompt.split(' ').concat(traceData.map(item => item.token));
  const allAttention = traceData.length > 0 ? traceData[0].attention : [];
  const allLogits = traceData.map(item => item.logits);

  const maxLayers = allAttention.length;
  const maxHeads = allAttention[0]?.length || 0;

  return (
    <Box 
      sx={{ 
        minHeight: '100vh',
        bgcolor: '#121212',
        color: '#fff',
        p: 3
      }}
    >
      <Typography 
        variant="h3" 
        component="h1" 
        sx={{ 
          mb: 4,
          fontWeight: 'bold',
          background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
        }}
      >
        Glassbox LLM Debugger
      </Typography>

      <Paper 
        elevation={3} 
        sx={{ 
          p: 3, 
          mb: 3, 
          bgcolor: '#1e1e1e',
          border: '1px solid #333'
        }}
      >
        <TextField
          fullWidth
          multiline
          rows={4}
          variant="outlined"
          label="Enter your prompt"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          sx={{
            mb: 2,
            '& .MuiOutlinedInput-root': {
              color: '#fff',
              '& fieldset': {
                borderColor: '#555',
              },
              '&:hover fieldset': {
                borderColor: '#777',
              },
              '&.Mui-focused fieldset': {
                borderColor: '#2196F3',
              },
            },
            '& .MuiInputLabel-root': {
              color: '#aaa',
            },
          }}
        />
        <Button
          variant="contained"
          onClick={handleTrace}
          disabled={loading || !prompt.trim()}
          sx={{
            bgcolor: '#2196F3',
            '&:hover': {
              bgcolor: '#1976D2',
            },
          }}
        >
          {loading ? 'GENERATING...' : 'TRACE GENERATION'}
        </Button>
      </Paper>

      {traceData.length > 0 && (
        <>
          {/* Controls for layer and head selection */}
          <Paper 
            elevation={2} 
            sx={{ 
              p: 2, 
              mb: 3, 
              bgcolor: '#1e1e1e',
              border: '1px solid #333'
            }}
          >
            <Typography variant="h6" sx={{ mb: 2 }}>
              Visualization Controls
            </Typography>
            <Grid container spacing={3}>
              <Grid item xs={6}>
                <Typography gutterBottom>
                  Layer: {selectedLayer + 1} / {maxLayers}
                </Typography>
                <Slider
                  value={selectedLayer}
                  onChange={(_, value) => setSelectedLayer(value as number)}
                  min={0}
                  max={Math.max(0, maxLayers - 1)}
                  step={1}
                  marks
                  valueLabelDisplay="auto"
                  sx={{ color: '#2196F3' }}
                />
              </Grid>
              <Grid item xs={6}>
                <Typography gutterBottom>
                  Attention Head: {selectedHead + 1} / {maxHeads}
                </Typography>
                <Slider
                  value={selectedHead}
                  onChange={(_, value) => setSelectedHead(value as number)}
                  min={0}
                  max={Math.max(0, maxHeads - 1)}
                  step={1}
                  marks
                  valueLabelDisplay="auto"
                  sx={{ color: '#2196F3' }}
                />
              </Grid>
            </Grid>
          </Paper>

          {/* Attention Heatmap */}
          {allAttention.length > 0 && (
            <AttentionHeatmap
              attention={allAttention}
              tokens={tokens}
              selectedLayer={selectedLayer}
              selectedHead={selectedHead}
            />
          )}

          {/* Attention Spider Web */}
          {allAttention.length > 0 && (
            <AttentionSpiderWeb
              attention={allAttention}
              tokens={tokens}
              selectedLayer={selectedLayer}
              selectedHead={selectedHead}
            />
          )}

          {/* Token Probability Bars */}
          <TokenProbabilityBars
            tokens={traceData.map(item => item.token)}
            logits={allLogits}
          />

          {/* Generated Text Display */}
          <Paper 
            elevation={2} 
            sx={{ 
              p: 3, 
              mb: 3, 
              bgcolor: '#1e1e1e',
              border: '1px solid #333'
            }}
          >
            <Typography variant="h6" sx={{ mb: 2 }}>
              Generated Text:
            </Typography>
            <Typography 
              variant="body1" 
              sx={{ 
                fontFamily: 'monospace',
                bgcolor: '#2a2a2a',
                p: 2,
                borderRadius: 1,
                wordBreak: 'break-all'
              }}
            >
              {prompt} <span style={{ color: '#4caf50' }}>{generatedText}</span>
            </Typography>
          </Paper>

          {/* Token-by-Token Analysis */}
          <Paper 
            elevation={2} 
            sx={{ 
              p: 3, 
              bgcolor: '#1e1e1e',
              border: '1px solid #333'
            }}
          >
            <Typography variant="h6" sx={{ mb: 2 }}>
              Token-by-Token Analysis:
            </Typography>
            <Box sx={{ maxHeight: 300, overflow: 'auto' }}>
              {traceData.map((item, index) => (
                <Typography 
                  key={index} 
                  variant="body2" 
                  sx={{ 
                    mb: 1,
                    fontFamily: 'monospace',
                    p: 1,
                    bgcolor: index % 2 === 0 ? '#2a2a2a' : '#1a1a1a',
                    borderRadius: 1
                  }}
                >
                  Token {index + 1}: <span style={{ color: '#4caf50' }}>"{item.token}"</span> 
                  {' '}(ID: {item.token_id}, Pos: {item.position})
                </Typography>
              ))}
            </Box>
          </Paper>
        </>
      )}
    </Box>
  );
}

export default App; 