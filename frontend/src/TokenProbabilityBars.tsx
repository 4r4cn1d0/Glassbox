import React from 'react';
import { motion } from 'framer-motion';

interface TokenData {
  token: string;
  token_id: number;
  probability: number;
  logit: number;
}

interface TokenProbabilityBarsProps {
  topTokensData: TokenData[][];
  currentTokenIndex: number;
  onTokenSelect?: (tokenIndex: number) => void;
}

const TokenProbabilityBars: React.FC<TokenProbabilityBarsProps> = ({ 
  topTokensData, 
  currentTokenIndex,
  onTokenSelect 
}) => {
  if (!topTokensData || topTokensData.length === 0) {
    return (
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        height: '300px',
        color: 'var(--secondary)'
      }}>
        No probability data available
      </div>
    );
  }

  const currentTokenData = topTokensData[currentTokenIndex] || [];
  const maxProbability = Math.max(...currentTokenData.map(t => t.probability));

  return (
    <div style={{ 
      display: 'flex', 
      flexDirection: 'column',
      height: '100%',
      padding: '20px'
    }}>
      {/* Header */}
      <div style={{ 
        marginBottom: '20px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <h3 style={{ 
          color: 'var(--accent)', 
          margin: 0,
          display: 'flex',
          alignItems: 'center',
          gap: '8px'
        }}>
          📊 Real-time Token Probabilities
        </h3>
        <div style={{ 
          fontSize: '14px',
          color: 'var(--secondary)'
        }}>
          Position: {currentTokenIndex + 1} / {topTokensData.length}
        </div>
      </div>

      {/* Token selector */}
      <div style={{
        display: 'flex',
        gap: '8px',
        marginBottom: '20px',
        overflowX: 'auto',
        paddingBottom: '8px'
      }}>
        {topTokensData.map((_, index) => (
          <button
            key={index}
            className={`btn-small ${index === currentTokenIndex ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => onTokenSelect?.(index)}
            style={{ minWidth: '40px' }}
          >
            {index + 1}
          </button>
        ))}
      </div>

      {/* Probability bars */}
      <div style={{ 
        flex: 1,
        overflowY: 'auto',
        display: 'flex',
        flexDirection: 'column',
        gap: '12px'
      }}>
        {currentTokenData.map((tokenData, index) => {
          const barWidth = (tokenData.probability / maxProbability) * 100;
          const isTopChoice = index === 0;
          
          return (
            <motion.div
              key={`${tokenData.token_id}-${index}`}
              style={{
                background: 'rgba(255, 255, 255, 0.05)',
                borderRadius: '12px',
                padding: '16px',
                border: isTopChoice ? '2px solid var(--accent)' : '1px solid rgba(255, 255, 255, 0.1)',
                position: 'relative',
                overflow: 'hidden'
              }}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3, delay: index * 0.05 }}
            >
              {/* Background gradient */}
              <motion.div
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  bottom: 0,
                  background: `linear-gradient(90deg, 
                    ${isTopChoice ? 'rgba(10, 132, 255, 0.2)' : 'rgba(10, 132, 255, 0.1)'} 0%, 
                    transparent ${barWidth}%)`,
                  zIndex: 0,
                  borderRadius: '12px'
                }}
                initial={{ width: 0 }}
                animate={{ width: '100%' }}
                transition={{ duration: 0.6, delay: index * 0.05 }}
              />

              {/* Content */}
              <div style={{ 
                position: 'relative', 
                zIndex: 1,
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center'
              }}>
                <div style={{ flex: 1 }}>
                  <div style={{ 
                    display: 'flex', 
                    alignItems: 'center',
                    gap: '12px',
                    marginBottom: '8px'
                  }}>
                    <span style={{ 
                      fontSize: '14px',
                      color: 'var(--secondary)',
                      minWidth: '20px'
                    }}>
                      #{index + 1}
                    </span>
                    <span style={{ 
                      fontFamily: 'Monaco, Consolas, monospace',
                      fontSize: '16px',
                      fontWeight: isTopChoice ? 'bold' : 'normal',
                      color: isTopChoice ? 'var(--accent)' : 'var(--fg)',
                      background: 'rgba(255, 255, 255, 0.1)',
                      padding: '4px 8px',
                      borderRadius: '6px',
                      maxWidth: '200px',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap'
                    }}>
                      "{tokenData.token}"
                    </span>
                    {isTopChoice && (
                      <span style={{ 
                        fontSize: '16px',
                        filter: 'drop-shadow(0 0 4px var(--accent))'
                      }}>
                        👑
                      </span>
                    )}
                  </div>
                  
                  <div style={{ 
                    display: 'flex',
                    gap: '16px',
                    fontSize: '12px',
                    color: 'var(--secondary)'
                  }}>
                    <span>ID: {tokenData.token_id}</span>
                    <span>Logit: {tokenData.logit.toFixed(3)}</span>
                  </div>
                </div>

                <div style={{ 
                  textAlign: 'right',
                  minWidth: '80px'
                }}>
                  <div style={{ 
                    fontSize: '20px',
                    fontWeight: 'bold',
                    color: isTopChoice ? 'var(--accent)' : 'var(--fg)',
                    marginBottom: '4px'
                  }}>
                    {(tokenData.probability * 100).toFixed(2)}%
                  </div>
                  <div style={{ 
                    fontSize: '11px',
                    color: 'var(--secondary)'
                  }}>
                    {tokenData.probability.toExponential(3)}
                  </div>
                </div>
              </div>

              {/* Probability bar */}
              <motion.div
                style={{
                  position: 'absolute',
                  bottom: 0,
                  left: 0,
                  height: '4px',
                  background: isTopChoice 
                    ? 'linear-gradient(90deg, var(--accent), rgba(10, 132, 255, 0.6))'
                    : 'linear-gradient(90deg, rgba(10, 132, 255, 0.6), transparent)',
                  borderRadius: '0 0 12px 12px'
                }}
                initial={{ width: 0 }}
                animate={{ width: `${barWidth}%` }}
                transition={{ duration: 0.8, delay: index * 0.05 }}
              />
            </motion.div>
          );
        })}
      </div>

      {/* Statistics footer */}
      <div style={{
        marginTop: '20px',
        padding: '16px',
        background: 'rgba(255, 255, 255, 0.05)',
        borderRadius: '12px',
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
        gap: '16px',
        fontSize: '12px'
      }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ 
            fontSize: '16px', 
            fontWeight: 'bold', 
            color: 'var(--accent)',
            marginBottom: '4px'
          }}>
            {currentTokenData.length}
          </div>
          <div style={{ color: 'var(--secondary)' }}>Top Tokens</div>
        </div>
        
        <div style={{ textAlign: 'center' }}>
          <div style={{ 
            fontSize: '16px', 
            fontWeight: 'bold', 
            color: 'var(--success)',
            marginBottom: '4px'
          }}>
            {currentTokenData.length > 0 ? (currentTokenData[0].probability * 100).toFixed(1) : 0}%
          </div>
          <div style={{ color: 'var(--secondary)' }}>Top Confidence</div>
        </div>

        <div style={{ textAlign: 'center' }}>
          <div style={{ 
            fontSize: '16px', 
            fontWeight: 'bold', 
            color: 'var(--warning)',
            marginBottom: '4px'
          }}>
            {currentTokenData.length > 0 ? 
              (-Math.log2(currentTokenData[0].probability)).toFixed(1) : 
              0
            }
          </div>
          <div style={{ color: 'var(--secondary)' }}>Bits Surprised</div>
        </div>

        <div style={{ textAlign: 'center' }}>
          <div style={{ 
            fontSize: '16px', 
            fontWeight: 'bold', 
            color: 'var(--info)',
            marginBottom: '4px'
          }}>
            {currentTokenData.length > 1 ? 
              (currentTokenData[0].probability / currentTokenData[1].probability).toFixed(1) : 
              '∞'
            }×
          </div>
          <div style={{ color: 'var(--secondary)' }}>Top Ratio</div>
        </div>
      </div>
    </div>
  );
};

export default TokenProbabilityBars; 