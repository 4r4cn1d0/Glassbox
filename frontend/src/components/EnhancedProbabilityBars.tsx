import React, { useRef, useEffect } from 'react';
import { motion } from 'framer-motion';

interface EnhancedProbabilityBarsProps {
  tokens: string[];
  logits: number[][];
  currentTokenIndex?: number;
}

const EnhancedProbabilityBars: React.FC<EnhancedProbabilityBarsProps> = ({
  tokens,
  logits,
  currentTokenIndex = 0
}) => {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Convert logits to probabilities using softmax
  const softmax = (arr: number[]): number[] => {
    const maxLogit = Math.max(...arr);
    const exps = arr.map(x => Math.exp(x - maxLogit));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    return exps.map(x => x / sumExps);
  };

  // Get top tokens for current position
  const getCurrentTokenData = () => {
    if (!logits || logits.length === 0 || currentTokenIndex >= logits.length) {
      return [];
    }

    const currentLogits = logits[currentTokenIndex];
    const probabilities = softmax(currentLogits);
    
    // Get top 20 tokens
    const tokenProbPairs = probabilities.map((prob, index) => ({
      token: `token_${index}`, // In real implementation, you'd have token decoder
      probability: prob,
      logit: currentLogits[index],
      index
    })).sort((a, b) => b.probability - a.probability).slice(0, 20);

    return tokenProbPairs;
  };

  // Generate sparkline data for last 5 tokens
  const getSparklineData = (tokenIndex: number): number[] => {
    const sparklineLength = 5;
    const data: number[] = [];
    
    for (let i = Math.max(0, currentTokenIndex - sparklineLength + 1); i <= currentTokenIndex; i++) {
      if (i < logits.length && logits[i]) {
        const probs = softmax(logits[i]);
        data.push(probs[tokenIndex] || 0);
      }
    }
    
    return data;
  };

  // Generate mini sparkline SVG
  const generateSparkline = (data: number[], width = 40, height = 16) => {
    if (data.length < 2) return null;
    
    const max = Math.max(...data);
    const min = Math.min(...data);
    const range = max - min;
    
    const points = data.map((value, index) => {
      const x = (index / (data.length - 1)) * width;
      const y = range > 0 ? height - ((value - min) / range) * height : height / 2;
      return `${x},${y}`;
    }).join(' ');
    
    return (
      <svg width={width} height={height} style={{ overflow: 'visible' }}>
        <polyline
          points={points}
          fill="none"
          stroke="var(--accent)"
          strokeWidth="1.5"
          opacity={0.7}
        />
        <circle
          cx={width}
          cy={range > 0 ? height - ((data[data.length - 1] - min) / range) * height : height / 2}
          r="2"
          fill="var(--accent)"
        />
      </svg>
    );
  };

  const tokenData = getCurrentTokenData();
  const maxProbability = Math.max(...tokenData.map(t => t.probability));

  // Auto-scroll to current token
  useEffect(() => {
    if (scrollRef.current && currentTokenIndex < tokens.length) {
      const scrollPosition = (currentTokenIndex / tokens.length) * scrollRef.current.scrollWidth;
      scrollRef.current.scrollTo({ left: scrollPosition, behavior: 'smooth' });
    }
  }, [currentTokenIndex, tokens.length]);

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Horizontal scrollable probability bars */}
      <div
        ref={scrollRef}
        style={{
          display: 'flex',
          gap: '12px',
          overflowX: 'auto',
          paddingBottom: '12px',
          scrollBehavior: 'smooth',
          WebkitOverflowScrolling: 'touch', // Smooth scrolling on iOS
          scrollSnapType: 'x mandatory'
        }}
      >
        {tokenData.map((tokenInfo, index) => {
          const barHeight = (tokenInfo.probability / maxProbability) * 120;
          const sparklineData = getSparklineData(tokenInfo.index);
          
          return (
            <motion.div
              key={tokenInfo.index}
              className="probability-item"
              style={{
                minWidth: '80px',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                scrollSnapAlign: 'center'
              }}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: index * 0.02 }}
            >
              {/* Probability percentage */}
              <div style={{
                fontSize: '11px',
                fontWeight: 600,
                color: 'var(--accent)',
                marginBottom: '4px',
                textAlign: 'center'
              }}>
                {(tokenInfo.probability * 100).toFixed(1)}%
              </div>
              
              {/* Probability bar */}
              <motion.div
                style={{
                  width: '8px',
                  height: '120px',
                  background: 'var(--bg)',
                  borderRadius: '4px',
                  position: 'relative',
                  overflow: 'hidden',
                  marginBottom: '8px'
                }}
                initial={{ height: 0 }}
                animate={{ height: '120px' }}
                transition={{ duration: 0.5, delay: index * 0.02 }}
              >
                <motion.div
                  style={{
                    position: 'absolute',
                    bottom: 0,
                    width: '100%',
                    background: `linear-gradient(to top, var(--accent), rgba(10, 132, 255, 0.6))`,
                    borderRadius: '4px',
                    height: `${barHeight}px`
                  }}
                  initial={{ height: 0 }}
                  animate={{ height: `${barHeight}px` }}
                  transition={{ duration: 0.5, delay: index * 0.02 }}
                />
              </motion.div>
              
              {/* Token label */}
              <div style={{
                fontSize: '10px',
                color: 'var(--fg)',
                fontWeight: 500,
                textAlign: 'center',
                marginBottom: '6px',
                fontFamily: 'Monaco, monospace',
                maxWidth: '70px',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap'
              }}>
                {tokenInfo.token}
              </div>
              
              {/* Sparkline */}
              <div style={{
                display: 'flex',
                justifyContent: 'center',
                marginBottom: '4px'
              }}>
                {generateSparkline(sparklineData)}
              </div>
              
              {/* Variance indicator */}
              <div style={{
                fontSize: '9px',
                color: 'var(--secondary)',
                textAlign: 'center'
              }}>
                {sparklineData.length > 1 && (
                  <>±{(Math.max(...sparklineData) - Math.min(...sparklineData) * 100).toFixed(1)}</>
                )}
              </div>
            </motion.div>
          );
        })}
      </div>
      
      {/* Gradient legend */}
      <div style={{
        marginTop: 'auto',
        padding: '12px 0',
        borderTop: '1px solid rgba(0,0,0,0.05)'
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
          fontSize: '11px',
          color: 'var(--secondary)'
        }}>
          <span>Low confidence</span>
          <div style={{
            flex: 1,
            height: '6px',
            background: 'linear-gradient(to right, rgba(10, 132, 255, 0.2), var(--accent))',
            borderRadius: '3px'
          }} />
          <span>High confidence</span>
        </div>
        <div style={{
          fontSize: '10px',
          color: 'var(--secondary)',
          textAlign: 'center',
          marginTop: '6px'
        }}>
          Scroll horizontally • Sparklines show last 5 steps
        </div>
      </div>
    </div>
  );
};

export default EnhancedProbabilityBars; 