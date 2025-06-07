import React, { useEffect, useRef } from 'react';
import { Box, Typography } from '@mui/material';

interface AttentionHeatmapProps {
  attention: number[][][]; // [layer][head][token_positions]
  tokens: string[];
  selectedLayer: number;
  selectedHead: number;
}

const AttentionHeatmap: React.FC<AttentionHeatmapProps> = ({ 
  attention, 
  tokens, 
  selectedLayer, 
  selectedHead 
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!attention || !tokens || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const cellSize = 20;
    const padding = 100;
    const tokenCount = tokens.length;

    canvas.width = tokenCount * cellSize + padding * 2;
    canvas.height = tokenCount * cellSize + padding * 2;

    // Clear canvas
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Get attention weights for selected layer and head
    const attentionWeights = attention[selectedLayer]?.[selectedHead];
    if (!attentionWeights) return;

    // Draw heatmap
    for (let i = 0; i < tokenCount; i++) {
      for (let j = 0; j < Math.min(tokenCount, attentionWeights.length); j++) {
        const weight = attentionWeights[j] || 0;
        const intensity = Math.min(weight * 2, 1); // Scale for visibility
        
        // Color from blue (low) to red (high)
        const r = Math.floor(intensity * 255);
        const g = Math.floor((1 - intensity) * 100);
        const b = Math.floor((1 - intensity) * 255);
        
        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(
          padding + j * cellSize,
          padding + i * cellSize,
          cellSize - 1,
          cellSize - 1
        );
      }
    }

    // Draw token labels
    ctx.fillStyle = '#ffffff';
    ctx.font = '12px monospace';
    ctx.textAlign = 'center';

    // X-axis labels (source tokens)
    for (let i = 0; i < Math.min(tokens.length, 10); i++) {
      const token = tokens[i].substring(0, 8); // Truncate long tokens
      ctx.save();
      ctx.translate(padding + i * cellSize + cellSize/2, padding - 10);
      ctx.rotate(-Math.PI/4);
      ctx.fillText(token, 0, 0);
      ctx.restore();
    }

    // Y-axis labels (target tokens)
    ctx.textAlign = 'right';
    for (let i = 0; i < Math.min(tokens.length, 10); i++) {
      const token = tokens[i].substring(0, 8);
      ctx.fillText(token, padding - 10, padding + i * cellSize + cellSize/2);
    }

    // Draw grid lines
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= tokenCount; i++) {
      // Vertical lines
      ctx.beginPath();
      ctx.moveTo(padding + i * cellSize, padding);
      ctx.lineTo(padding + i * cellSize, padding + tokenCount * cellSize);
      ctx.stroke();
      
      // Horizontal lines
      ctx.beginPath();
      ctx.moveTo(padding, padding + i * cellSize);
      ctx.lineTo(padding + tokenCount * cellSize, padding + i * cellSize);
      ctx.stroke();
    }

  }, [attention, tokens, selectedLayer, selectedHead]);

  return (
    <Box sx={{ p: 2, bgcolor: '#2a2a2a', borderRadius: 2, mb: 2 }}>
      <Typography variant="h6" sx={{ color: '#fff', mb: 2 }}>
        Attention Heatmap - Layer {selectedLayer + 1}, Head {selectedHead + 1}
      </Typography>
      <Box sx={{ overflow: 'auto', maxWidth: '100%' }}>
        <canvas 
          ref={canvasRef} 
          style={{ 
            border: '1px solid #555',
            backgroundColor: '#1a1a1a'
          }}
        />
      </Box>
      <Typography variant="caption" sx={{ color: '#aaa', mt: 1, display: 'block' }}>
        Red = High Attention, Blue = Low Attention
      </Typography>
    </Box>
  );
};

export default AttentionHeatmap; 