import React from 'react';
import { Box, Typography, LinearProgress } from '@mui/material';

interface TokenProbabilityBarsProps {
  tokens: string[];
  logits: number[][];
}

const TokenProbabilityBars: React.FC<TokenProbabilityBarsProps> = ({ tokens, logits }) => {
  // Convert logits to probabilities using softmax approximation
  const getTopTokenProbabilities = (tokenLogits: number[]) => {
    const maxLogit = Math.max(...tokenLogits);
    const expLogits = tokenLogits.map(logit => Math.exp(logit - maxLogit));
    const sumExp = expLogits.reduce((sum, exp) => sum + exp, 0);
    return expLogits.map(exp => exp / sumExp);
  };

  return (
    <Box sx={{ p: 2, bgcolor: '#2a2a2a', borderRadius: 2, mb: 2 }}>
      <Typography variant="h6" sx={{ color: '#fff', mb: 2 }}>
        Token Generation Probabilities
      </Typography>
      <Box sx={{ maxHeight: 400, overflow: 'auto' }}>
        {tokens.slice(0, 10).map((token, index) => {
          const tokenLogits = logits[index] || [];
          const probabilities = getTopTokenProbabilities(tokenLogits);
          const maxProb = Math.max(...probabilities) * 100;
          const confidence = maxProb;

          return (
            <Box key={index} sx={{ mb: 2 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="body2" sx={{ color: '#fff', fontFamily: 'monospace' }}>
                  Token {index + 1}: "{token}"
                </Typography>
                <Typography variant="body2" sx={{ color: '#aaa' }}>
                  {confidence.toFixed(1)}%
                </Typography>
              </Box>
              <LinearProgress
                variant="determinate"
                value={confidence}
                sx={{
                  height: 8,
                  borderRadius: 4,
                  bgcolor: '#1a1a1a',
                  '& .MuiLinearProgress-bar': {
                    bgcolor: confidence > 80 ? '#4caf50' : 
                            confidence > 50 ? '#ff9800' : '#f44336',
                    borderRadius: 4,
                  },
                }}
              />
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 0.5 }}>
                <Typography variant="caption" sx={{ color: '#666' }}>
                  Low Confidence
                </Typography>
                <Typography variant="caption" sx={{ color: '#666' }}>
                  High Confidence
                </Typography>
              </Box>
            </Box>
          );
        })}
      </Box>
    </Box>
  );
};

export default TokenProbabilityBars; 