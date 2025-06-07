import React from 'react';
import { motion } from 'framer-motion';

interface TokenMiniMapProps {
  tokens: string[];
  attentionData: number[][][];
  currentTokenIndex: number;
  onTokenClick: (index: number) => void;
}

const TokenMiniMap: React.FC<TokenMiniMapProps> = ({
  tokens,
  attentionData,
  currentTokenIndex,
  onTokenClick
}) => {
  // Calculate attention intensity for each token
  const getTokenIntensity = (tokenIndex: number): number => {
    if (!attentionData || attentionData.length === 0) return 0;
    
    // Average attention across all layers and heads for this token
    let totalAttention = 0;
    let count = 0;
    
    attentionData.forEach(layer => {
      layer.forEach(head => {
        if (Array.isArray(head) && head[tokenIndex] && Array.isArray(head[tokenIndex])) {
          const tokenAttentions = head[tokenIndex] as unknown as number[];
          totalAttention += tokenAttentions.reduce((sum: number, val: number) => sum + val, 0);
          count++;
        } else if (Array.isArray(head) && typeof head[tokenIndex] === 'number') {
          // Handle case where head[tokenIndex] is a single number
          totalAttention += head[tokenIndex] as number;
          count++;
        }
      });
    });
    
    return count > 0 ? totalAttention / count : 0;
  };

  const maxIntensity = Math.max(...tokens.map((_, i) => getTokenIntensity(i)));

  return (
    <motion.div
      className="mini-map"
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      style={{
        position: 'fixed',
        top: '80px',
        left: '50%',
        transform: 'translateX(-50%)',
        zIndex: 1000,
        background: 'var(--card)',
        borderRadius: '24px',
        padding: '8px 16px',
        boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
        display: 'flex',
        gap: '2px',
        maxWidth: '80vw',
        overflow: 'hidden'
      }}
    >
      {tokens.map((token, index) => {
        const intensity = getTokenIntensity(index);
        const normalizedIntensity = maxIntensity > 0 ? intensity / maxIntensity : 0;
        const isCurrent = index === currentTokenIndex;
        
        return (
          <motion.div
            key={index}
            className="mini-map-token"
            onClick={() => onTokenClick(index)}
            style={{
              width: '4px',
              height: '20px',
              borderRadius: '2px',
              background: isCurrent 
                ? 'var(--accent)' 
                : `rgba(10, 132, 255, ${0.2 + normalizedIntensity * 0.8})`,
              cursor: 'pointer',
              transition: 'all 0.2s ease',
              position: 'relative'
            }}
            whileHover={{ 
              scale: 1.5, 
              height: '24px',
              background: 'var(--accent)'
            }}
            animate={isCurrent ? {
              boxShadow: [
                '0 0 0 rgba(10, 132, 255, 0)',
                '0 0 8px rgba(10, 132, 255, 0.6)',
                '0 0 0 rgba(10, 132, 255, 0)'
              ]
            } : {}}
            transition={{ 
              boxShadow: { duration: 1.5, repeat: Infinity }
            }}
          />
        );
      })}
      
      {/* Progress indicator */}
      <div 
        className="progress-line"
        style={{
          position: 'absolute',
          bottom: '0',
          left: '16px',
          right: '16px',
          height: '2px',
          background: 'rgba(10, 132, 255, 0.1)',
          borderRadius: '1px'
        }}
      >
        <motion.div
          style={{
            height: '100%',
            background: 'var(--accent)',
            borderRadius: '1px',
            width: `${((currentTokenIndex + 1) / tokens.length) * 100}%`
          }}
          transition={{ duration: 0.3 }}
        />
      </div>
    </motion.div>
  );
};

export default TokenMiniMap; 