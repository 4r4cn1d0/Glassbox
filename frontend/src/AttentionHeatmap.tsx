import React, { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';

interface TopKAttentionWeight {
  from_token: number;
  to_token: number;
  weight: number;
}

interface TopKAttentionData {
  top_k_weights: TopKAttentionWeight[];
  shape: number[];
  total_weights: number;
}

interface AttentionHeatmapProps {
  attention: TopKAttentionData[][]; // [layer][head]
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

    const cellSize = 40;
    const padding = 200;
    const tokenCount = Math.min(tokens.length, 15); // Limit for visibility

    canvas.width = tokenCount * cellSize + padding * 2;
    canvas.height = tokenCount * cellSize + padding * 2;

    // Clear canvas with clean light background
    ctx.fillStyle = '#F8F9FA';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Get attention weights for selected layer and head
    const attentionData = attention[selectedLayer]?.[selectedHead];
    if (!attentionData || !attentionData.top_k_weights) {
      // Draw empty grid if no data
      drawEmptyGrid(ctx, tokenCount, cellSize, padding);
      return;
    }

    // Create attention matrix from top-K data
    const attentionMatrix: number[][] = Array(tokenCount).fill(0).map(() => Array(tokenCount).fill(0));
    let maxWeight = 0;

    attentionData.top_k_weights.forEach(({ from_token, to_token, weight }) => {
      if (from_token < tokenCount && to_token < tokenCount) {
        attentionMatrix[from_token][to_token] = weight;
        maxWeight = Math.max(maxWeight, weight);
      }
    });

    // Draw heatmap cells with vibrant colors
    for (let i = 0; i < tokenCount; i++) {
      for (let j = 0; j < tokenCount; j++) {
        const weight = attentionMatrix[i][j];
        const normalizedWeight = maxWeight > 0 ? weight / maxWeight : 0;
        
        let fillColor = '#F8F9FA'; // Default light background
        
        if (normalizedWeight > 0) {
          // Use a vibrant blue to red gradient for attention weights
          const intensity = normalizedWeight;
          if (intensity > 0.7) {
            // High attention - red/orange
            fillColor = `rgba(255, 69, 0, ${0.3 + intensity * 0.7})`;
          } else if (intensity > 0.4) {
            // Medium attention - yellow/orange
            fillColor = `rgba(255, 165, 0, ${0.2 + intensity * 0.6})`;
          } else {
            // Low attention - blue
            fillColor = `rgba(30, 144, 255, ${0.1 + intensity * 0.5})`;
          }
        }
        
        // Special highlighting for current token
        if (i === currentTokenIndex || j === currentTokenIndex) {
          if (normalizedWeight > 0) {
            fillColor = `rgba(50, 205, 50, ${0.4 + normalizedWeight * 0.6})`; // Green highlight
          } else {
            fillColor = 'rgba(50, 205, 50, 0.2)'; // Light green for current token row/col
          }
        }
        
        ctx.fillStyle = fillColor;
        ctx.fillRect(
          padding + j * cellSize + 2,
          padding + i * cellSize + 2,
          cellSize - 4,
          cellSize - 4
        );

        // Add cell border
        ctx.strokeStyle = '#E0E0E0';
        ctx.lineWidth = 1;
        ctx.strokeRect(
          padding + j * cellSize + 2,
          padding + i * cellSize + 2,
          cellSize - 4,
          cellSize - 4
        );

        // Display attention value in cell if significant
        if (normalizedWeight > 0.1) {
          ctx.fillStyle = normalizedWeight > 0.5 ? '#FFFFFF' : '#000000';
          ctx.font = '10px SF Pro Text, -apple-system, sans-serif';
          ctx.textAlign = 'center';
          ctx.fillText(
            weight.toFixed(3),
            padding + j * cellSize + cellSize/2,
            padding + i * cellSize + cellSize/2 + 3
          );
        }
      }
    }

    // Draw current token indicators
    if (currentTokenIndex < tokenCount) {
      const pulseIntensity = 0.6 + 0.4 * Math.sin(Date.now() / 300);
      
      ctx.strokeStyle = `rgba(50, 205, 50, ${pulseIntensity})`;
      ctx.lineWidth = 4;
      ctx.setLineDash([10, 5]);
      
      // Highlight current row
      ctx.strokeRect(
        padding,
        padding + currentTokenIndex * cellSize,
        tokenCount * cellSize,
        cellSize
      );
      
      // Highlight current column
      ctx.strokeRect(
        padding + currentTokenIndex * cellSize,
        padding,
        cellSize,
        tokenCount * cellSize
      );
      
      ctx.setLineDash([]);
    }

    // Draw token labels with better styling
    ctx.font = 'bold 14px SF Pro Text, -apple-system, sans-serif';
    ctx.textAlign = 'center';

    // X-axis labels (target tokens)
    for (let i = 0; i < tokenCount; i++) {
      const token = tokens[i] || '';
      const displayToken = token.length > 10 ? token.substring(0, 10) + '...' : token;
      
      ctx.fillStyle = i === currentTokenIndex ? '#32CD32' : '#495057';
      ctx.font = i === currentTokenIndex ? 'bold 16px SF Pro Text' : 'bold 14px SF Pro Text';
      
      ctx.save();
      ctx.translate(padding + i * cellSize + cellSize/2, padding - 40);
      ctx.rotate(-Math.PI/4);
      ctx.fillText(displayToken, 0, 0);
      ctx.restore();
    }

    // Y-axis labels (source tokens)
    ctx.textAlign = 'right';
    for (let i = 0; i < tokenCount; i++) {
      const token = tokens[i] || '';
      const displayToken = token.length > 10 ? token.substring(0, 10) + '...' : token;
      
      ctx.fillStyle = i === currentTokenIndex ? '#32CD32' : '#495057';
      ctx.font = i === currentTokenIndex ? 'bold 16px SF Pro Text' : 'bold 14px SF Pro Text';
      
      ctx.fillText(displayToken, padding - 15, padding + i * cellSize + cellSize/2 + 5);
    }

    // Add axis labels
    ctx.textAlign = 'center';
    ctx.fillStyle = '#495057';
    ctx.font = 'bold 16px SF Pro Text, -apple-system, sans-serif';
    ctx.fillText('Target Tokens', canvas.width/2, 30);
    
    ctx.save();
    ctx.translate(30, canvas.height/2);
    ctx.rotate(-Math.PI/2);
    ctx.fillText('Source Tokens', 0, 0);
    ctx.restore();

  }, [attention, tokens, selectedLayer, selectedHead, currentTokenIndex]);

  const drawEmptyGrid = (ctx: CanvasRenderingContext2D, tokenCount: number, cellSize: number, padding: number) => {
    ctx.fillStyle = '#F8F9FA';
    ctx.font = '18px SF Pro Text, -apple-system, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillStyle = '#6C757D';
    ctx.fillText('No attention data available', ctx.canvas.width/2, ctx.canvas.height/2);
    ctx.fillText(`Layer ${selectedLayer + 1}, Head ${selectedHead + 1}`, ctx.canvas.width/2, ctx.canvas.height/2 + 30);
  };

  const handleMouseMove = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    setMousePos({ x: event.clientX, y: event.clientY });

    const padding = 200;
    const cellSize = 40;
    const tokenCount = Math.min(tokens.length, 15);
    
    const col = Math.floor((x - padding) / cellSize);
    const row = Math.floor((y - padding) / cellSize);
    
    if (row >= 0 && row < tokenCount && col >= 0 && col < tokenCount && attention[selectedLayer]?.[selectedHead]) {
      const attentionData = attention[selectedLayer][selectedHead];
      const weight = attentionData.top_k_weights.find(w => w.from_token === row && w.to_token === col)?.weight || 0;
      setHoveredCell({ row, col, value: weight });
    } else {
      setHoveredCell(null);
    }
  };

  const handleMouseLeave = () => {
    setHoveredCell(null);
  };

  return (
    <motion.div
      style={{
        background: '#FFFFFF',
        borderRadius: '16px',
        padding: '20px',
        border: '1px solid #E9ECEF',
        overflow: 'auto'
      }}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        marginBottom: '20px'
      }}>
        <h3 style={{ 
          color: '#495057', 
          margin: 0,
          fontSize: '20px',
          fontWeight: 'bold'
        }}>
          🔥 Attention Heatmap - Layer {selectedLayer + 1}, Head {selectedHead + 1}
        </h3>
        
        <div style={{
          display: 'flex',
          gap: '20px',
          fontSize: '14px',
          color: '#6C757D'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{ 
              width: '16px', 
              height: '16px', 
              background: 'linear-gradient(90deg, rgba(30, 144, 255, 0.3), rgba(255, 69, 0, 0.8))',
              borderRadius: '3px'
            }} />
            Low → High Attention
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{ 
              width: '16px', 
              height: '16px', 
              background: 'rgba(50, 205, 50, 0.6)',
              borderRadius: '3px'
            }} />
            Current Token
          </div>
        </div>
      </div>

      <canvas
        ref={canvasRef}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        style={{ 
          cursor: 'crosshair',
          maxWidth: '100%',
          height: 'auto'
        }}
      />

      {/* Tooltip */}
      {hoveredCell && (
        <div
          style={{
            position: 'fixed',
            left: mousePos.x + 10,
            top: mousePos.y - 40,
            background: '#212529',
            color: '#FFFFFF',
            padding: '8px 12px',
            borderRadius: '8px',
            fontSize: '14px',
            fontWeight: 'bold',
            zIndex: 1000,
            pointerEvents: 'none',
            boxShadow: '0 4px 12px rgba(0,0,0,0.3)'
          }}
        >
          <div>From: "{tokens[hoveredCell.row]}"</div>
          <div>To: "{tokens[hoveredCell.col]}"</div>
          <div>Weight: {hoveredCell.value.toFixed(4)}</div>
        </div>
      )}
    </motion.div>
  );
};

export default AttentionHeatmap; 