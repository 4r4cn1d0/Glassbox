import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface HeadThumbnailsProps {
  attention: number[][][];
  tokens: string[];
  selectedLayer: number;
  selectedHead: number;
  onHeadSelect: (headIndex: number) => void;
}

const HeadThumbnails: React.FC<HeadThumbnailsProps> = ({
  attention,
  tokens,
  selectedLayer,
  selectedHead,
  onHeadSelect
}) => {
  const [hoveredHead, setHoveredHead] = useState<number | null>(null);

  if (!attention || attention.length === 0 || !attention[selectedLayer]) {
    return null;
  }

  const currentLayerHeads = attention[selectedLayer];
  const maxHeads = Math.min(6, currentLayerHeads.length); // Show up to 6 heads

  // Generate mini heatmap for a head
  const generateMiniHeatmap = (headAttention: number[][] | number[], size = 40) => {
    const tokenCount = Math.min(tokens.length, 20); // Limit for thumbnail
    
    // Handle case where headAttention might be 1D array
    let attention2D: number[][];
    if (Array.isArray(headAttention) && headAttention.length > 0) {
      if (Array.isArray(headAttention[0])) {
        attention2D = headAttention as number[][];
      } else {
        // Convert 1D to 2D by creating identity-like matrix
        const size = Math.min(headAttention.length, tokenCount);
        attention2D = Array(size).fill(0).map(() => Array(size).fill(0));
        // Fill with some sample pattern for visualization
        for (let i = 0; i < size; i++) {
          for (let j = 0; j < size; j++) {
            attention2D[i][j] = (headAttention as number[])[Math.min(i, headAttention.length - 1)] || 0;
          }
        }
      }
    } else {
      // Fallback to empty matrix
      attention2D = Array(tokenCount).fill(0).map(() => Array(tokenCount).fill(0));
    }

    return (
      <div 
        style={{ 
          width: size, 
          height: size, 
          display: 'grid',
          gridTemplateColumns: `repeat(${tokenCount}, 1fr)`,
          gap: '1px',
          background: 'var(--bg)',
          borderRadius: '4px',
          overflow: 'hidden'
        }}
      >
        {Array.from({ length: tokenCount }, (_, i) => 
          Array.from({ length: tokenCount }, (_, j) => {
            const attention_val = attention2D[i]?.[j] || 0;
            const opacity = Math.min(attention_val * 2, 1);
            
            return (
              <div
                key={`${i}-${j}`}
                style={{
                  background: `rgba(10, 132, 255, ${opacity})`,
                  minHeight: '2px',
                  minWidth: '2px'
                }}
              />
            );
          })
        ).flat()}
      </div>
    );
  };

  return (
    <div style={{ marginTop: '16px' }}>
      <h4 className="text-small" style={{ 
        marginBottom: '12px', 
        color: 'var(--secondary)',
        fontWeight: 500
      }}>
        Layer {selectedLayer + 1} Heads
      </h4>
      
      <div style={{ 
        display: 'flex', 
        gap: '8px', 
        overflowX: 'auto',
        paddingBottom: '8px'
      }}>
        {Array.from({ length: maxHeads }, (_, headIndex) => {
          const isSelected = headIndex === selectedHead;
          const isHovered = headIndex === hoveredHead;
          
          return (
            <motion.div
              key={headIndex}
              className="head-thumbnail"
              style={{
                minWidth: '60px',
                padding: '8px',
                background: isSelected ? 'var(--accent)' : 'var(--card)',
                borderRadius: '12px',
                cursor: 'pointer',
                border: isSelected ? '2px solid var(--accent)' : '2px solid transparent',
                boxShadow: isSelected 
                  ? '0 4px 12px rgba(10, 132, 255, 0.3)' 
                  : '0 2px 8px rgba(0,0,0,0.1)',
                transition: 'all 0.2s ease'
              }}
              whileHover={{ 
                scale: 1.05,
                boxShadow: '0 6px 16px rgba(10, 132, 255, 0.2)'
              }}
              whileTap={{ scale: 0.95 }}
              onClick={() => onHeadSelect(headIndex)}
              onMouseEnter={() => setHoveredHead(headIndex)}
              onMouseLeave={() => setHoveredHead(null)}
            >
              <div style={{ 
                display: 'flex', 
                flexDirection: 'column', 
                alignItems: 'center',
                gap: '6px'
              }}>
                <div style={{
                  fontSize: '10px',
                  fontWeight: 600,
                  color: isSelected ? 'white' : 'var(--accent)',
                  marginBottom: '2px'
                }}>
                  H{headIndex + 1}
                </div>
                
                {currentLayerHeads[headIndex] && (
                  <div style={{ transform: 'scale(0.8)' }}>
                    {generateMiniHeatmap(currentLayerHeads[headIndex])}
                  </div>
                )}
              </div>
            </motion.div>
          );
        })}
      </div>
      
      {/* Preview tooltip for hovered head */}
      <AnimatePresence>
        {hoveredHead !== null && hoveredHead !== selectedHead && (
          <motion.div
            initial={{ opacity: 0, y: 10, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 10, scale: 0.9 }}
            transition={{ duration: 0.2 }}
            style={{
              position: 'absolute',
              zIndex: 1000,
              background: 'var(--card)',
              padding: '12px',
              borderRadius: '12px',
              boxShadow: '0 8px 24px rgba(0,0,0,0.15)',
              marginTop: '8px',
              left: `${hoveredHead * 68}px`,
              maxWidth: '200px'
            }}
          >
            <div style={{ fontSize: '12px', fontWeight: 600, marginBottom: '6px' }}>
              Head {hoveredHead + 1} Preview
            </div>
            {currentLayerHeads[hoveredHead] && (
              <div style={{ display: 'flex', justifyContent: 'center' }}>
                {generateMiniHeatmap(currentLayerHeads[hoveredHead], 60)}
              </div>
            )}
            <div style={{ 
              fontSize: '10px', 
              color: 'var(--secondary)', 
              marginTop: '6px',
              textAlign: 'center'
            }}>
              Click to select
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default HeadThumbnails; 