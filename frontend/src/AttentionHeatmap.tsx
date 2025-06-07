import React, { useEffect, useRef } from 'react';
import { Box, Typography } from '@mui/material';

interface AttentionHeatmapProps {
  attention: number[][][]; // [layer][head][token_positions]
  tokens: string[];
  selectedLayer: number;
  selectedHead: number;
  currentTokenIndex?: number;
}

const AttentionHeatmap: React.FC<AttentionHeatmapProps> = ({ 
  attention, 
  tokens, 
  selectedLayer, 
  selectedHead,
  currentTokenIndex = 0
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!attention || !tokens || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const cellSize = 25;
    const padding = 120;
    const tokenCount = Math.min(tokens.length, 20); // Limit for visibility

    canvas.width = tokenCount * cellSize + padding * 2;
    canvas.height = tokenCount * cellSize + padding * 2;

    // Clear canvas with light background
    ctx.fillStyle = 'var(--bg)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Get attention weights for selected layer and head
    const attentionWeights = attention[selectedLayer]?.[selectedHead];
    if (!attentionWeights) return;

    // Draw heatmap cells
    for (let i = 0; i < tokenCount; i++) {
      for (let j = 0; j < Math.min(tokenCount, attentionWeights.length); j++) {
        const weight = Array.isArray(attentionWeights[i]) ? (attentionWeights[i] as number[])[j] : 0;
        const intensity = Math.min(weight * 2, 1); // Scale for visibility
        
        // Use iOS blue color gradient
        const opacity = 0.1 + intensity * 0.8;
        ctx.fillStyle = `rgba(10, 132, 255, ${opacity})`;
        
        // Highlight current token row/column
        if (i === currentTokenIndex || j === currentTokenIndex) {
          ctx.fillStyle = `rgba(10, 132, 255, ${Math.max(opacity, 0.3)})`;
        }
        
        ctx.fillRect(
          padding + j * cellSize,
          padding + i * cellSize,
          cellSize - 1,
          cellSize - 1
        );

        // Add subtle neumorphic shadow for depth
        if (intensity > 0.3) {
          ctx.shadowColor = 'rgba(0,0,0,0.1)';
          ctx.shadowBlur = 2;
          ctx.shadowOffsetX = 1;
          ctx.shadowOffsetY = 1;
          ctx.fillRect(
            padding + j * cellSize,
            padding + i * cellSize,
            cellSize - 1,
            cellSize - 1
          );
          ctx.shadowBlur = 0;
        }
      }
    }

    // Draw current token indicators with pulsing effect
    if (currentTokenIndex < tokenCount) {
      const pulseIntensity = 0.5 + 0.3 * Math.sin(Date.now() / 300);
      
      // Highlight current row
      ctx.strokeStyle = `rgba(10, 132, 255, ${pulseIntensity})`;
      ctx.lineWidth = 3;
      ctx.strokeRect(
        padding - 2,
        padding + currentTokenIndex * cellSize - 2,
        tokenCount * cellSize + 4,
        cellSize + 4
      );
      
      // Highlight current column
      ctx.strokeRect(
        padding + currentTokenIndex * cellSize - 2,
        padding - 2,
        cellSize + 4,
        tokenCount * cellSize + 4
      );
    }

    // Draw token labels with better styling
    ctx.font = 'bold 11px -apple-system, SF Pro Text, sans-serif';
    ctx.textAlign = 'center';

    // X-axis labels (source tokens)
    ctx.fillStyle = '#1C1C1E';
    for (let i = 0; i < tokenCount; i++) {
      const token = tokens[i]?.substring(0, 6) || ''; // Truncate long tokens
      
      // Highlight current token label
      if (i === currentTokenIndex) {
        ctx.fillStyle = '#0A84FF';
        ctx.font = 'bold 12px -apple-system, SF Pro Text, sans-serif';
      } else {
        ctx.fillStyle = '#8E8E93';
        ctx.font = 'bold 11px -apple-system, SF Pro Text, sans-serif';
      }
      
      ctx.save();
      ctx.translate(padding + i * cellSize + cellSize/2, padding - 15);
      ctx.rotate(-Math.PI/6);
      ctx.fillText(token, 0, 0);
      ctx.restore();
    }

    // Y-axis labels (target tokens)
    ctx.textAlign = 'right';
    for (let i = 0; i < tokenCount; i++) {
      const token = tokens[i]?.substring(0, 6) || '';
      
      // Highlight current token label
      if (i === currentTokenIndex) {
        ctx.fillStyle = '#0A84FF';
        ctx.font = 'bold 12px -apple-system, SF Pro Text, sans-serif';
      } else {
        ctx.fillStyle = '#8E8E93';
        ctx.font = 'bold 11px -apple-system, SF Pro Text, sans-serif';
      }
      
      ctx.fillText(token, padding - 15, padding + i * cellSize + cellSize/2 + 4);
    }

    // Draw grid lines with subtle styling
    ctx.strokeStyle = 'rgba(0,0,0,0.1)';
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

    // Add subtle outer border
    ctx.strokeStyle = 'rgba(0,0,0,0.15)';
    ctx.lineWidth = 2;
    ctx.strokeRect(padding - 1, padding - 1, tokenCount * cellSize + 2, tokenCount * cellSize + 2);

  }, [attention, tokens, selectedLayer, selectedHead, currentTokenIndex]);

  // Redraw periodically for pulsing effect
  useEffect(() => {
    const interval = setInterval(() => {
      // Trigger re-render for pulsing effect
      if (canvasRef.current) {
        canvasRef.current.style.filter = `drop-shadow(0 0 ${4 + 2 * Math.sin(Date.now() / 300)}px rgba(10, 132, 255, 0.3))`;
      }
    }, 50);
    
    return () => clearInterval(interval);
  }, []);

  return (
    <Box sx={{ p: 2, bgcolor: '#2a2a2a', borderRadius: 2, mb: 2 }}>
      <Typography variant="h6" sx={{ color: '#fff', mb: 2 }}>
        Attention Heatmap - Layer {selectedLayer + 1}, Head {selectedHead + 1}
      </Typography>
      <Box sx={{ overflow: 'auto', maxWidth: '100%' }}>
        <canvas 
          ref={canvasRef} 
          style={{ 
            borderRadius: '8px',
            backgroundColor: 'var(--card)',
            boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
            transition: 'filter 0.3s ease'
          }}
        />
      </Box>
      <Typography variant="caption" sx={{ color: '#aaa', mt: 1, display: 'block' }}>
        Red = High Attention, Blue = Low Attention
      </Typography>
      <div style={{
        fontSize: '11px',
        color: 'var(--secondary)',
        textAlign: 'center',
        marginTop: '8px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '16px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <div style={{
            width: '12px',
            height: '12px',
            background: 'rgba(10, 132, 255, 0.3)',
            borderRadius: '2px'
          }} />
          Low Attention
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <div style={{
            width: '12px',
            height: '12px',
            background: 'rgba(10, 132, 255, 0.9)',
            borderRadius: '2px'
          }} />
          High Attention
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <div style={{
            width: '12px',
            height: '12px',
            background: '#0A84FF',
            borderRadius: '2px',
            boxShadow: '0 0 6px rgba(10, 132, 255, 0.6)'
          }} />
          Current Token
        </div>
      </div>
    </Box>
  );
};

export default AttentionHeatmap; 