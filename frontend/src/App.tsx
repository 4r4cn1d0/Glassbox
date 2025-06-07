import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Box, 
  TextField, 
  Button, 
  Paper, 
  Typography, 
  Grid, 
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem
} from '@mui/material';
import { 
  Grid3X3, 
  Network, 
  BarChart3, 
  FileText, 
  Settings, 
  Moon, 
  Sun, 
  ChevronLeft, 
  ChevronRight,
  Play,
  Loader2
} from 'lucide-react';
import AttentionHeatmap from './AttentionHeatmap';
import TokenProbabilityBars from './TokenProbabilityBars';
import AttentionSpiderWeb from './AttentionSpiderWeb';
import SkeletonLoader from './components/SkeletonLoader';
import './theme.css';

interface TraceData {
  token: string;
  token_id: number;
  position: number;
  logits: number[];
  attention: number[][][];
  is_generated: boolean;
}

type ViewTab = 'heatmap' | 'spiderweb' | 'probabilities' | 'text';

function App() {
  const [prompt, setPrompt] = useState('');
  const [traceData, setTraceData] = useState<TraceData[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedLayer, setSelectedLayer] = useState(0);
  const [selectedHead, setSelectedHead] = useState(0);
  const [activeTab, setActiveTab] = useState<ViewTab>('heatmap');
  const [darkMode, setDarkMode] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Initialize dark mode from system preference
  useEffect(() => {
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    setDarkMode(prefersDark);
  }, []);

  // Apply dark mode to document
  useEffect(() => {
    if (darkMode) {
      document.documentElement.setAttribute('data-theme', 'dark');
    } else {
      document.documentElement.removeAttribute('data-theme');
    }
  }, [darkMode]);

  const handleTrace = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/trace', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt, max_new_tokens: 20 }),
      });
      const data = await response.json();
      setTraceData(data);
    } catch (error) {
      console.error('Error:', error);
    }
    setLoading(false);
  };

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
  };

  const generatedText = traceData.map(item => item.token).join('');
  const tokens = prompt.split(' ').concat(traceData.map(item => item.token));
  const allAttention = traceData.length > 0 ? traceData[0].attention : [];
  const allLogits = traceData.map(item => item.logits);

  const maxLayers = allAttention.length;
  const maxHeads = allAttention[0]?.length || 0;

  const tabs = [
    { id: 'heatmap' as ViewTab, label: 'Heatmap', icon: Grid3X3 },
    { id: 'spiderweb' as ViewTab, label: 'Network', icon: Network },
    { id: 'probabilities' as ViewTab, label: 'Probabilities', icon: BarChart3 },
    { id: 'text' as ViewTab, label: 'Analysis', icon: FileText },
  ];

  const renderActiveView = () => {
    if (loading) {
      return <SkeletonLoader type="visualization" />;
    }

    if (traceData.length === 0) {
      return (
        <motion.div 
          className="viz-panel"
          style={{ 
            textAlign: 'center',
            padding: '60px 40px',
            background: 'linear-gradient(135deg, var(--card) 0%, rgba(10, 132, 255, 0.02) 100%)'
          }}
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
        >
          <div style={{ fontSize: '48px', marginBottom: '16px' }}>🧠</div>
          <h3 className="text-large" style={{ marginBottom: '8px' }}>
            Welcome to Glassbox
          </h3>
          <p className="text-small" style={{ color: 'var(--secondary)', maxWidth: '400px', margin: '0 auto' }}>
            Enter a prompt to begin exploring LLM attention patterns and token generation in real-time.
          </p>
        </motion.div>
      );
    }

    switch (activeTab) {
      case 'heatmap':
        return (
          <motion.div 
            className="viz-panel"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            <div className="viz-header">
              <h3 className="text-medium">Attention Heatmap</h3>
            </div>
            <div className="viz-content">
              <AttentionHeatmap
                attention={allAttention}
                tokens={tokens}
                selectedLayer={selectedLayer}
                selectedHead={selectedHead}
              />
            </div>
          </motion.div>
        );

      case 'spiderweb':
        return (
          <motion.div 
            className="viz-panel"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            <div className="viz-header">
              <h3 className="text-medium">Attention Network</h3>
            </div>
            <div className="viz-content">
              <AttentionSpiderWeb
                attention={allAttention}
                tokens={tokens}
                selectedLayer={selectedLayer}
                selectedHead={selectedHead}
              />
            </div>
          </motion.div>
        );

      case 'probabilities':
        return (
          <motion.div 
            className="viz-panel"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            <div className="viz-header">
              <h3 className="text-medium">Token Probabilities</h3>
            </div>
            <div className="viz-content">
              <TokenProbabilityBars
                tokens={traceData.map(item => item.token)}
                logits={allLogits}
              />
            </div>
          </motion.div>
        );

      case 'text':
        return (
          <motion.div 
            className="viz-panel"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            <div className="viz-header">
              <h3 className="text-medium">Generated Text & Analysis</h3>
            </div>
            <div className="viz-content">
              {/* Generated Text Display */}
              <div style={{ marginBottom: '24px' }}>
                <h4 className="text-medium" style={{ marginBottom: '12px' }}>Output</h4>
                <div style={{ 
                  fontFamily: 'Monaco, Consolas, monospace',
                  background: 'var(--bg)',
                  padding: '16px',
                  borderRadius: '12px',
                  fontSize: '16px',
                  lineHeight: '1.6',
                  wordBreak: 'break-word'
                }}>
                  {prompt} 
                  <span style={{ 
                    color: 'var(--accent)', 
                    fontWeight: '600',
                    background: 'rgba(10, 132, 255, 0.1)',
                    padding: '2px 4px',
                    borderRadius: '4px',
                    marginLeft: '4px'
                  }}>
                    {generatedText}
                  </span>
                </div>
              </div>

              {/* Token Analysis */}
              <div>
                <h4 className="text-medium" style={{ marginBottom: '12px' }}>Token Details</h4>
                <div style={{ maxHeight: '300px', overflow: 'auto' }}>
                  {traceData.map((item, index) => (
                    <motion.div 
                      key={index} 
                      className="token-hover"
                      style={{ 
                        marginBottom: '8px',
                        padding: '12px',
                        background: index % 2 === 0 ? 'var(--bg)' : 'transparent',
                        borderRadius: '8px',
                        fontFamily: 'Monaco, monospace',
                        fontSize: '14px'
                      }}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.2, delay: index * 0.05 }}
                    >
                      <strong>Token {index + 1}:</strong> 
                      <span style={{ color: 'var(--accent)', fontWeight: '600' }}>
                        "{item.token}"
                      </span>
                      {' '}(ID: {item.token_id}, Pos: {item.position})
                    </motion.div>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>
        );

      default:
        return null;
    }
  };

  return (
    <div style={{ background: 'var(--bg)', minHeight: '100vh', display: 'flex' }}>
      {/* Collapsible Sidebar */}
      <motion.div 
        className="card"
        style={{ 
          width: sidebarCollapsed ? '60px' : '320px',
          height: '100vh',
          margin: '0',
          borderRadius: '0',
          borderTopRightRadius: '16px',
          borderBottomRightRadius: '16px',
          position: 'relative',
          overflow: 'hidden',
          transition: 'width 0.3s ease'
        }}
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.5 }}
      >
        {/* Sidebar Toggle */}
        <button
          className="btn-icon"
          onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
          style={{
            position: 'absolute',
            top: '16px',
            right: '16px',
            zIndex: 10
          }}
        >
          {sidebarCollapsed ? <ChevronRight size={24} /> : <ChevronLeft size={24} />}
        </button>

        <div style={{ padding: sidebarCollapsed ? '16px 8px' : '24px', paddingTop: '60px' }}>
          {!sidebarCollapsed && (
            <>
              {/* Header with Dark Mode Toggle */}
              <div className="header-actions" style={{ marginBottom: '24px', justifyContent: 'space-between' }}>
                <h1 className="text-large" style={{ 
                  background: 'linear-gradient(45deg, var(--accent), #21CBF3)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent'
                }}>
                  Glassbox
                </h1>
                <button
                  className="btn-icon"
                  onClick={toggleDarkMode}
                  title={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
                >
                  {darkMode ? <Sun size={24} /> : <Moon size={24} />}
                </button>
              </div>

              {/* Input Section */}
              <motion.div 
                style={{ marginBottom: '24px' }}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
              >
                <h2 className="text-medium" style={{ marginBottom: '16px' }}>
                  Input Prompt
                </h2>
                <textarea
                  className="input"
                  placeholder="Enter your prompt to analyze..."
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  rows={4}
                  style={{ marginBottom: '16px' }}
                />
                <button
                  className={`btn ${loading || !prompt.trim() ? 'btn-secondary' : 'btn-primary'}`}
                  onClick={handleTrace}
                  disabled={loading || !prompt.trim()}
                  style={{ width: '100%' }}
                >
                  {loading ? (
                    <>
                      <Loader2 size={20} className="animate-spin" />
                      GENERATING...
                    </>
                  ) : (
                    <>
                      <Play size={20} />
                      TRACE GENERATION
                    </>
                  )}
                </button>
              </motion.div>

              {/* Advanced Controls */}
              {traceData.length > 0 && (
                <motion.div 
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  transition={{ duration: 0.4 }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', marginBottom: '16px' }}>
                    <h3 className="text-medium" style={{ flex: 1 }}>Controls</h3>
                    <button
                      className="btn-ghost"
                      onClick={() => setShowAdvanced(!showAdvanced)}
                      style={{ padding: '4px 8px' }}
                    >
                      <Settings size={20} />
                    </button>
                  </div>
                  
                  <AnimatePresence>
                    {showAdvanced && (
                      <motion.div
                        className="collapsible"
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        transition={{ duration: 0.3 }}
                      >
                        <div className="slider-container">
                          <div className="slider-label">
                            <span>Layer</span>
                            <span className="slider-value">{selectedLayer + 1} / {maxLayers}</span>
                          </div>
                          <Slider
                            value={selectedLayer}
                            onChange={(_, value) => setSelectedLayer(value as number)}
                            min={0}
                            max={Math.max(0, maxLayers - 1)}
                            step={1}
                            sx={{ 
                              color: 'var(--accent)',
                              height: 8,
                              '& .MuiSlider-thumb': {
                                width: 20,
                                height: 20,
                                backgroundColor: 'var(--accent)',
                                border: 'none',
                                boxShadow: '0 2px 6px rgba(0,0,0,0.2)',
                                '&:hover': {
                                  boxShadow: '0 4px 8px rgba(10, 132, 255, 0.4)',
                                },
                              },
                              '& .MuiSlider-track': {
                                backgroundColor: 'var(--accent)',
                                border: 'none',
                                height: 8,
                              },
                              '& .MuiSlider-rail': {
                                backgroundColor: '#DDD',
                                height: 8,
                              }
                            }}
                          />
                        </div>

                        <div className="slider-container">
                          <div className="slider-label">
                            <span>Attention Head</span>
                            <span className="slider-value">{selectedHead + 1} / {maxHeads}</span>
                          </div>
                          <Slider
                            value={selectedHead}
                            onChange={(_, value) => setSelectedHead(value as number)}
                            min={0}
                            max={Math.max(0, maxHeads - 1)}
                            step={1}
                            sx={{ 
                              color: 'var(--accent)',
                              height: 8,
                              '& .MuiSlider-thumb': {
                                width: 20,
                                height: 20,
                                backgroundColor: 'var(--accent)',
                                border: 'none',
                                boxShadow: '0 2px 6px rgba(0,0,0,0.2)',
                                '&:hover': {
                                  boxShadow: '0 4px 8px rgba(10, 132, 255, 0.4)',
                                },
                              },
                              '& .MuiSlider-track': {
                                backgroundColor: 'var(--accent)',
                                border: 'none',
                                height: 8,
                              },
                              '& .MuiSlider-rail': {
                                backgroundColor: '#DDD',
                                height: 8,
                              }
                            }}
                          />
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.div>
              )}
            </>
          )}
        </div>
      </motion.div>

      {/* Main Content Area */}
      <div style={{ 
        flex: 1, 
        padding: '24px',
        overflow: 'auto'
      }}>
        {/* Tab Navigation */}
        {(traceData.length > 0 || loading) && (
          <motion.div 
            className="tab-container"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            {tabs.map(tab => {
              const IconComponent = tab.icon;
              return (
                <button
                  key={tab.id}
                  className={`tab ${activeTab === tab.id ? 'active' : ''}`}
                  onClick={() => setActiveTab(tab.id)}
                >
                  <IconComponent size={20} />
                  {tab.label}
                </button>
              );
            })}
          </motion.div>
        )}

        {/* Content Area */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          {renderActiveView()}
        </motion.div>
      </div>
    </div>
  );
}

export default App; 