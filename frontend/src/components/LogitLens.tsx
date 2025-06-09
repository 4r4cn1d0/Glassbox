import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Eye, EyeOff, BarChart3, Layers } from 'lucide-react';

interface LogitLensProps {
  sessionId: string | null;
  currentTokenIndex: number;
  onLayerSelect?: (layer: number) => void;
}

interface LayerPrediction {
  layer: number;
  predictions: Array<{
    token: string;
    logit: number;
    probability: number;
    rank: number;
  }>;
}

const LogitLens: React.FC<LogitLensProps> = ({ 
  sessionId, 
  currentTokenIndex, 
  onLayerSelect 
}) => {
  const [layerPredictions, setLayerPredictions] = useState<LayerPrediction[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedLayers, setSelectedLayers] = useState<Set<number>>(new Set([0, 11, 23, 35])); // Default to key layers
  const [showMode, setShowMode] = useState<'top5' | 'diff' | 'entropy'>('top5');

  useEffect(() => {
    if (sessionId && currentTokenIndex >= 0) {
      fetchLogitLens();
    }
  }, [sessionId, currentTokenIndex]);

  const fetchLogitLens = async () => {
    if (!sessionId) return;
    
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/logit-lens', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          token_index: currentTokenIndex,
          layers: Array.from(selectedLayers),
          mode: showMode
        }),
      });
      
      if (response.ok) {
        const data = await response.json();
        setLayerPredictions(data.layer_predictions || []);
      }
    } catch (error) {
      console.error('Logit lens fetch error:', error);
      // Generate mock data for demo
      generateMockLogitLens();
    }
    setLoading(false);
  };

  const generateMockLogitLens = () => {
    const mockTokens = [' the', ' a', ' and', ' to', ' of', ' in', ' that', ' is', ' for', ' on'];
    const predictions = Array.from(selectedLayers).map(layer => ({
      layer,
      predictions: mockTokens.slice(0, 5).map((token, idx) => ({
        token,
        logit: 10 - idx * 2 + Math.random() * 0.5,
        probability: Math.exp(10 - idx * 2) / 100,
        rank: idx + 1
      }))
    }));
    setLayerPredictions(predictions);
  };

  const toggleLayer = (layer: number) => {
    const newSelection = new Set(selectedLayers);
    if (newSelection.has(layer)) {
      newSelection.delete(layer);
    } else {
      newSelection.add(layer);
    }
    setSelectedLayers(newSelection);
  };

  const getColorForLayer = (layer: number) => {
    const hue = (layer * 10) % 360;
    return `hsl(${hue}, 70%, 60%)`;
  };

  const getIntensityColor = (probability: number, maxProb: number) => {
    const intensity = Math.min(probability / maxProb, 1);
    return `rgba(0, 123, 255, ${0.2 + intensity * 0.6})`;
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      style={{
        background: '#FFFFFF',
        borderRadius: '16px',
        padding: '24px',
        border: '1px solid #E9ECEF',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        gap: '20px'
      }}
    >
      {/* Header */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <Eye size={24} style={{ color: '#007BFF' }} />
          <h3 style={{
            fontSize: '20px',
            fontWeight: 'bold',
            color: '#495057',
            margin: 0
          }}>
            Neural Lens
          </h3>
        </div>

        <div style={{ display: 'flex', gap: '8px' }}>
          {['top5', 'diff', 'entropy'].map(mode => (
            <button
              key={mode}
              onClick={() => setShowMode(mode as any)}
              style={{
                padding: '6px 12px',
                border: showMode === mode ? '2px solid #007BFF' : '1px solid #DEE2E6',
                borderRadius: '6px',
                background: showMode === mode ? '#E7F3FF' : '#F8F9FA',
                color: showMode === mode ? '#007BFF' : '#495057',
                fontSize: '12px',
                fontWeight: '600',
                cursor: 'pointer',
                textTransform: 'uppercase'
              }}
            >
              {mode}
            </button>
          ))}
        </div>
      </div>

      {/* Layer Selection */}
      <div style={{
        background: '#F8F9FA',
        borderRadius: '12px',
        padding: '16px'
      }}>
        <h4 style={{
          fontSize: '14px',
          fontWeight: '600',
          color: '#495057',
          marginBottom: '12px',
          display: 'flex',
          alignItems: 'center',
          gap: '8px'
        }}>
          <Layers size={16} />
          Layer Selection
        </h4>
        
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(60px, 1fr))',
          gap: '6px'
        }}>
          {Array.from({ length: 36 }, (_, i) => (
            <button
              key={i}
              onClick={() => toggleLayer(i)}
              style={{
                padding: '8px 4px',
                border: selectedLayers.has(i) ? `2px solid ${getColorForLayer(i)}` : '1px solid #DEE2E6',
                borderRadius: '6px',
                background: selectedLayers.has(i) ? getColorForLayer(i) : '#FFFFFF',
                color: selectedLayers.has(i) ? '#FFFFFF' : '#495057',
                fontSize: '11px',
                fontWeight: '600',
                cursor: 'pointer',
                transition: 'all 0.2s ease'
              }}
            >
              L{i}
            </button>
          ))}
        </div>
      </div>

      {/* Predictions Grid */}
      <div style={{
        flex: 1,
        overflow: 'auto'
      }}>
        {loading ? (
          <div style={{
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            height: '200px',
            color: '#6C757D'
          }}>
            Loading logit lens...
          </div>
        ) : layerPredictions.length === 0 ? (
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center',
            height: '200px',
            color: '#6C757D'
          }}>
            <Eye size={48} style={{ opacity: 0.3, marginBottom: '16px' }} />
            <p>Select layers to view predictions</p>
          </div>
        ) : (
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
            gap: '16px'
          }}>
            {layerPredictions.map(({ layer, predictions }) => {
              const maxProb = Math.max(...predictions.map(p => p.probability));
              
              return (
                <motion.div
                  key={layer}
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.3 }}
                  style={{
                    background: '#FFFFFF',
                    border: `2px solid ${getColorForLayer(layer)}`,
                    borderRadius: '12px',
                    padding: '16px',
                    cursor: 'pointer'
                  }}
                  onClick={() => onLayerSelect?.(layer)}
                >
                  <h5 style={{
                    fontSize: '16px',
                    fontWeight: 'bold',
                    color: getColorForLayer(layer),
                    marginBottom: '12px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px'
                  }}>
                    <BarChart3 size={16} />
                    Layer {layer}
                  </h5>
                  
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    {predictions.map(({ token, probability, rank }, idx) => (
                      <div
                        key={idx}
                        style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                          padding: '8px 12px',
                          background: getIntensityColor(probability, maxProb),
                          borderRadius: '8px',
                          border: '1px solid rgba(0, 123, 255, 0.2)'
                        }}
                      >
                        <div style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: '8px'
                        }}>
                          <span style={{
                            fontSize: '12px',
                            fontWeight: 'bold',
                            color: '#6C757D',
                            minWidth: '20px'
                          }}>
                            #{rank}
                          </span>
                          <span style={{
                            fontFamily: 'Monaco, Consolas, monospace',
                            fontSize: '14px',
                            fontWeight: 'bold',
                            color: '#495057'
                          }}>
                            "{token}"
                          </span>
                        </div>
                        
                        <span style={{
                          fontSize: '12px',
                          fontWeight: '600',
                          color: '#007BFF'
                        }}>
                          {(probability * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </motion.div>
              );
            })}
          </div>
        )}
      </div>
    </motion.div>
  );
};

export default LogitLens; 