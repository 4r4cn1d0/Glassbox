import React, { useEffect, useRef, useState } from 'react';
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
  const [hoveredCell, setHoveredCell] = useState<{row: number, col: number, value: number} | null>(null);
  const [mousePos, setMousePos] = useState<{x: number, y: number}>({x: 0, y: 0});

  useEffect(() => {
    if (!attention || !tokens || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const cellSize = 35; // Increased from 25 for better visibility
    const padding = 150; // Increased for better label spacing
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
        const weight = Array.isArray(attentionWeights[i]) ? (attentionWeights[i] as unknown as number[])[j] : 0;
        const intensity = Math.min(weight * 2, 1); // Scale for visibility
        
        // Use iOS blue color gradient
        let opacity = 0.1 + intensity * 0.8;
        let fillColor = `rgba(10, 132, 255, ${opacity})`;
        
        // Enhanced highlighting for current token with much more prominent visibility
        if (i === currentTokenIndex || j === currentTokenIndex) {
          opacity = Math.max(opacity, 0.6);
          fillColor = `rgba(255, 193, 7, ${opacity})`; // Golden yellow for current token
          
          // Add extra glow for current token intersection
          if (i === currentTokenIndex && j === currentTokenIndex) {
            fillColor = `rgba(255, 87, 34, 0.9)`; // Orange for the intersection
          }
        }
        
        ctx.fillStyle = fillColor;
        ctx.fillRect(
          padding + j * cellSize,
          padding + i * cellSize,
          cellSize - 2,
          cellSize - 2
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
            cellSize - 2,
            cellSize - 2
          );
          ctx.shadowBlur = 0;
        }
      }
    }

    // Draw much more prominent current token indicators
    if (currentTokenIndex < tokenCount) {
      // Pulsing effect with larger amplitude
      const pulseIntensity = 0.7 + 0.4 * Math.sin(Date.now() / 200);
      
      // Thicker and more visible highlighting
      ctx.strokeStyle = `rgba(255, 193, 7, ${pulseIntensity})`;
      ctx.lineWidth = 6; // Increased from 3
      ctx.setLineDash([8, 4]); // Dashed line for better visibility
      
      // Highlight current row with larger border
      ctx.strokeRect(
        padding - 6,
        padding + currentTokenIndex * cellSize - 6,
        tokenCount * cellSize + 12,
        cellSize + 12
      );
      
      // Highlight current column with larger border
      ctx.strokeRect(
        padding + currentTokenIndex * cellSize - 6,
        padding - 6,
        cellSize + 12,
        tokenCount * cellSize + 12
      );
      
      ctx.setLineDash([]); // Reset dash pattern
    }

    // Draw token labels with better styling
    ctx.font = 'bold 12px -apple-system, SF Pro Text, sans-serif';
    ctx.textAlign = 'center';

    // X-axis labels (source tokens)
    for (let i = 0; i < tokenCount; i++) {
      const token = tokens[i]?.substring(0, 8) || ''; // Show more characters
      
      // Enhanced highlighting for current token label
      if (i === currentTokenIndex) {
        ctx.fillStyle = '#FF5722'; // Orange for current token
        ctx.font = 'bold 14px -apple-system, SF Pro Text, sans-serif';
        
        // Add background highlight
        const textWidth = ctx.measureText(token).width;
        ctx.fillStyle = 'rgba(255, 193, 7, 0.3)';
        ctx.fillRect(
          padding + i * cellSize + cellSize/2 - textWidth/2 - 4,
          padding - 40,
          textWidth + 8,
          20
        );
        ctx.fillStyle = '#FF5722';
      } else {
        ctx.fillStyle = '#8E8E93';
        ctx.font = 'bold 12px -apple-system, SF Pro Text, sans-serif';
      }
      
      ctx.save();
      ctx.translate(padding + i * cellSize + cellSize/2, padding - 20);
      ctx.rotate(-Math.PI/6);
      ctx.fillText(token, 0, 0);
      ctx.restore();
    }

    // Y-axis labels (target tokens)
    ctx.textAlign = 'right';
    for (let i = 0; i < tokenCount; i++) {
      const token = tokens[i]?.substring(0, 8) || '';
      
      // Enhanced highlighting for current token label
      if (i === currentTokenIndex) {
        ctx.fillStyle = '#FF5722'; // Orange for current token
        ctx.font = 'bold 14px -apple-system, SF Pro Text, sans-serif';
        
        // Add background highlight
        const textWidth = ctx.measureText(token).width;
        ctx.fillStyle = 'rgba(255, 193, 7, 0.3)';
        ctx.fillRect(
          padding - textWidth - 25,
          padding + i * cellSize + cellSize/2 - 10,
          textWidth + 8,
          20
        );
        ctx.fillStyle = '#FF5722';
      } else {
        ctx.fillStyle = '#8E8E93';
        ctx.font = 'bold 12px -apple-system, SF Pro Text, sans-serif';
      }
      
      ctx.fillText(token, padding - 20, padding + i * cellSize + cellSize/2 + 4);
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

  // Handle mouse events for hover functionality
  const handleMouseMove = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current || !attention[selectedLayer]?.[selectedHead]) return;
    
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    const cellSize = 35;
    const padding = 150;
    const tokenCount = Math.min(tokens.length, 20);
    
    // Calculate which cell is being hovered
    const col = Math.floor((x - padding) / cellSize);
    const row = Math.floor((y - padding) / cellSize);
    
    if (col >= 0 && col < tokenCount && row >= 0 && row < tokenCount) {
      const attentionWeights = attention[selectedLayer][selectedHead];
      const weight = Array.isArray(attentionWeights[row]) ? 
        (attentionWeights[row] as unknown as number[])[col] : 0;
      
      setHoveredCell({ row, col, value: weight });
      setMousePos({ x: event.clientX, y: event.clientY });
    } else {
      setHoveredCell(null);
    }
  };

  const handleMouseLeave = () => {
    setHoveredCell(null);
  };

  // Redraw periodically for pulsing effect
  useEffect(() => {
    const interval = setInterval(() => {
      // Trigger re-render for pulsing effect
      if (canvasRef.current) {
        canvasRef.current.style.filter = `drop-shadow(0 0 ${6 + 3 * Math.sin(Date.now() / 200)}px rgba(255, 193, 7, 0.5))`;
      }
    }, 50);
    
    return () => clearInterval(interval);
  }, []);

  return (
    <Box sx={{ p: 2, bgcolor: '#2a2a2a', borderRadius: 2, mb: 2, position: 'relative' }}>
      <Typography variant="h6" sx={{ color: '#fff', mb: 2 }}>
        Attention Heatmap - Layer {selectedLayer + 1}, Head {selectedHead + 1}
      </Typography>
      <Box sx={{ overflow: 'auto', maxWidth: '100%' }}>
        <canvas 
          ref={canvasRef} 
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          style={{ 
            borderRadius: '8px',
            backgroundColor: 'var(--card)',
            boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
            transition: 'filter 0.3s ease',
            cursor: 'crosshair'
          }}
        />
      </Box>
      
      {/* Hover tooltip */}
      {hoveredCell && (
        <div
          style={{
            position: 'fixed',
            left: mousePos.x + 10,
            top: mousePos.y - 60,
            background: 'rgba(0, 0, 0, 0.9)',
            color: 'white',
            padding: '8px 12px',
            borderRadius: '6px',
            fontSize: '12px',
            pointerEvents: 'none',
            zIndex: 1000,
            boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
            border: '1px solid rgba(255,255,255,0.1)'
          }}
        >
          <div><strong>From:</strong> "{tokens[hoveredCell.row]?.substring(0, 10) || 'N/A'}"</div>
          <div><strong>To:</strong> "{tokens[hoveredCell.col]?.substring(0, 10) || 'N/A'}"</div>
          <div><strong>Attention:</strong> {hoveredCell.value.toFixed(4)}</div>
          <div><strong>Position:</strong> [{hoveredCell.row}, {hoveredCell.col}]</div>
        </div>
      )}
      
      <Typography variant="caption" sx={{ color: '#aaa', mt: 1, display: 'block' }}>
        Hover over squares for detailed attention values
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
            background: '#FFC107',
            borderRadius: '2px',
            boxShadow: '0 0 8px rgba(255, 193, 7, 0.6)'
          }} />
          Current Token
        </div>
      </div>
    </Box>
  );
};

export default AttentionHeatmap; 