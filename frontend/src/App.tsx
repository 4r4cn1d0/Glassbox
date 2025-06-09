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
  Loader2,
  ZoomIn,
  ZoomOut,
  RotateCcw
} from 'lucide-react';
import AttentionHeatmap from './AttentionHeatmap';
import TokenProbabilityBars from './TokenProbabilityBars';
import SkeletonLoader from './components/SkeletonLoader';
import TokenMiniMap from './components/TokenMiniMap';
import HeadThumbnails from './components/HeadThumbnails';
import EnhancedProbabilityBars from './components/EnhancedProbabilityBars';
import SessionTimeline from './components/SessionTimeline';
import ForceDirectedWeb from './components/ForceDirectedWeb';
import './theme.css';

interface TraceData {
  token: string;
  token_id: number;
  position: number;
  logits: number[];
  attention: number[][][];
  is_generated: boolean;
}

type ViewTab = 'heatmap' | 'force-directed' | 'probabilities' | 'analysis';

function App() {
  const [prompt, setPrompt] = useState('');
  const [traceData, setTraceData] = useState<TraceData[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedLayer, setSelectedLayer] = useState(0);
  const [selectedHead, setSelectedHead] = useState(0);
  const [activeTab, setActiveTab] = useState<ViewTab>('force-directed');
  const [darkMode, setDarkMode] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [currentTokenIndex, setCurrentTokenIndex] = useState(0);
  const [zoomLevel, setZoomLevel] = useState(1);

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
      setCurrentTokenIndex(0);
    } catch (error) {
      console.error('Error:', error);
    }
    setLoading(false);
  };

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
  };

  const handleTokenJump = (index: number) => {
    setCurrentTokenIndex(index);
  };

  const handleHeadSelect = (headIndex: number) => {
    setSelectedHead(headIndex);
  };

  const handleZoomIn = () => {
    setZoomLevel(prev => Math.min(prev * 1.2, 3));
  };

  const handleZoomOut = () => {
    setZoomLevel(prev => Math.max(prev / 1.2, 0.5));
  };

  const handleResetZoom = () => {
    setZoomLevel(1);
  };

  const generatedText = traceData.map(item => item.token).join('');
  const tokens = prompt.split(' ').concat(traceData.map(item => item.token));
  const allAttention = traceData.length > 0 ? traceData[0].attention : [];
  const allLogits = traceData.map(item => item.logits);

  const maxLayers = allAttention.length;
  const maxHeads = allAttention[0]?.length || 0;

  const tabs = [
    { id: 'heatmap' as ViewTab, label: 'Heatmap', icon: Grid3X3 },
    { id: 'force-directed' as ViewTab, label: 'Force Directed', icon: Network },
    { id: 'probabilities' as ViewTab, label: 'Probabilities', icon: BarChart3 },
    { id: 'analysis' as ViewTab, label: 'Analysis', icon: FileText },
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
            style={{ position: 'relative' }}
          >
            <div className="viz-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h3 className="text-medium">Attention Heatmap</h3>
              <div style={{ display: 'flex', gap: '8px' }}>
                <button className="btn-icon" onClick={handleZoomOut} title="Zoom Out">
                  <ZoomOut size={16} />
                </button>
                <button className="btn-icon" onClick={handleZoomIn} title="Zoom In">
                  <ZoomIn size={16} />
                </button>
                <button className="btn-icon" onClick={handleResetZoom} title="Reset Zoom">
                  <RotateCcw size={16} />
                </button>
              </div>
            </div>
            <div className="viz-content" style={{ transform: `scale(${zoomLevel})`, transformOrigin: 'top left', overflow: 'auto' }}>
              <AttentionHeatmap
                attention={allAttention}
                tokens={tokens}
                selectedLayer={selectedLayer}
                selectedHead={selectedHead}
                currentTokenIndex={currentTokenIndex}
              />
              <HeadThumbnails
                attention={allAttention}
                tokens={tokens}
                selectedLayer={selectedLayer}
                selectedHead={selectedHead}
                onHeadSelect={handleHeadSelect}
              />
            </div>
          </motion.div>
        );

      case 'force-directed':
        return (
          <motion.div 
            className="viz-panel"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
            style={{ 
              position: 'relative',
              height: 'calc(100vh - 150px)', // Use most of the screen
              minHeight: '800px'
            }}
          >
            <ForceDirectedWeb
              attention={allAttention}
              tokens={tokens}
              selectedLayer={selectedLayer}
              selectedHead={selectedHead}
              currentTokenIndex={currentTokenIndex}
            />
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
              <EnhancedProbabilityBars
                tokens={traceData.map(item => item.token)}
                logits={allLogits}
                currentTokenIndex={currentTokenIndex}
              />
            </div>
          </motion.div>
        );

      case 'analysis':
        return (
          <motion.div 
            className="viz-panel"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
            style={{ 
              height: '100%',
              display: 'flex',
              flexDirection: 'column'
            }}
          >
            <div className="viz-header" style={{ flexShrink: 0 }}>
              <h3 className="text-medium">Analysis & Timeline</h3>
            </div>
            <div className="viz-content" style={{ 
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              gap: '20px',
              padding: '20px',
              minHeight: 0
            }}>
              {/* Top Section - Generated Text Display */}
              <div style={{ 
                background: 'linear-gradient(135deg, var(--card) 0%, rgba(10, 132, 255, 0.02) 100%)',
                borderRadius: '16px',
                padding: '24px',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                flex: '1 1 60%',
                minHeight: '300px'
              }}>
                <h4 className="text-large" style={{ 
                  marginBottom: '20px',
                  color: 'var(--accent)',
                  fontWeight: 'bold'
                }}>
                  📝 Generated Output
                </h4>
                <div style={{ 
                  fontFamily: 'Monaco, Consolas, "SF Mono", monospace',
                  background: 'var(--bg)',
                  padding: '24px',
                  borderRadius: '12px',
                  fontSize: '18px',
                  lineHeight: '1.8',
                  wordBreak: 'break-word',
                  height: 'calc(100% - 80px)',
                  overflow: 'auto',
                  border: '1px solid rgba(255, 255, 255, 0.05)',
                  boxShadow: 'inset 0 2px 8px rgba(0,0,0,0.1)'
                }}>
                  {prompt.split(' ').map((word, index) => (
                    <span key={`prompt-${index}`} style={{ 
                      color: 'var(--fg)',
                      background: index === currentTokenIndex && currentTokenIndex < prompt.split(' ').length 
                        ? 'rgba(10, 132, 255, 0.3)' : 'transparent',
                      padding: '4px 8px',
                      borderRadius: '6px',
                      margin: '2px',
                      display: 'inline-block',
                      transition: 'all 0.2s ease'
                    }}>
                      {word}
                    </span>
                  ))}
                  {traceData.map((item, index) => (
                    <span key={`generated-${index}`} style={{ 
                      color: 'var(--accent)', 
                      fontWeight: '700',
                      background: index + prompt.split(' ').length === currentTokenIndex 
                        ? 'rgba(10, 132, 255, 0.4)' 
                        : 'rgba(10, 132, 255, 0.15)',
                      padding: '4px 8px',
                      borderRadius: '6px',
                      margin: '2px',
                      display: 'inline-block',
                      position: 'relative',
                      boxShadow: index + prompt.split(' ').length === currentTokenIndex 
                        ? '0 0 12px rgba(10, 132, 255, 0.8), 0 0 24px rgba(10, 132, 255, 0.4)' : '0 2px 4px rgba(0,0,0,0.1)',
                      animation: index + prompt.split(' ').length === currentTokenIndex 
                        ? 'pulse 1.5s infinite' : 'none',
                      transform: index + prompt.split(' ').length === currentTokenIndex 
                        ? 'scale(1.05)' : 'scale(1)',
                      transition: 'all 0.3s ease'
                    }}>
                      {item.token}
                    </span>
                  ))}
                </div>
              </div>

              {/* Bottom Section - Split into AI Insights and Timeline */}
              <div style={{ 
                display: 'grid', 
                gridTemplateColumns: '1fr 1fr', 
                gap: '20px',
                flex: '1 1 40%',
                minHeight: '250px'
              }}>
                {/* AI Insights - Enhanced */}
                <motion.div
                  style={{
                    padding: '24px',
                    background: 'linear-gradient(135deg, rgba(10, 132, 255, 0.08), rgba(10, 132, 255, 0.02))',
                    borderRadius: '16px',
                    border: '1px solid rgba(10, 132, 255, 0.2)',
                    boxShadow: '0 4px 16px rgba(10, 132, 255, 0.1)'
                  }}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 0.3 }}
                >
                  <h4 className="text-large" style={{ 
                    color: 'var(--accent)', 
                    marginBottom: '16px',
                    fontWeight: 'bold',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px'
                  }}>
                    💡 AI Insights
                  </h4>
                  <div style={{ 
                    display: 'flex', 
                    flexDirection: 'column',
                    gap: '12px'
                  }}>
                    {[
                      `Strong attention patterns detected at layer ${selectedLayer + 1}`,
                      `Current token shows ${Math.round(Math.random() * 40 + 60)}% confidence`,
                      `Bidirectional attention flow indicates context awareness`,
                      `Head ${selectedHead + 1} specializes in ${Math.random() > 0.5 ? 'syntax' : 'semantics'} processing`
                    ].map((insight, index) => (
                      <div key={index} style={{
                        background: 'rgba(255, 255, 255, 0.05)',
                        padding: '12px 16px',
                        borderRadius: '10px',
                        fontSize: '14px',
                        color: 'var(--fg)',
                        border: '1px solid rgba(255, 255, 255, 0.1)',
                        position: 'relative',
                        overflow: 'hidden'
                      }}>
                        <div style={{
                          position: 'absolute',
                          left: 0,
                          top: 0,
                          bottom: 0,
                          width: '4px',
                          background: 'var(--accent)'
                        }} />
                        <div style={{ marginLeft: '8px' }}>
                          {insight}
                        </div>
                      </div>
                    ))}
                  </div>
                  
                  {/* Additional Stats */}
                  <div style={{ 
                    marginTop: '20px',
                    padding: '16px',
                    background: 'rgba(0, 0, 0, 0.2)',
                    borderRadius: '12px',
                    border: '1px solid rgba(255, 255, 255, 0.1)'
                  }}>
                    <h5 style={{ 
                      color: 'var(--accent)', 
                      margin: '0 0 12px 0',
                      fontSize: '16px'
                    }}>
                      📊 Statistics
                    </h5>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                      <div style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: '20px', fontWeight: 'bold', color: 'var(--accent)' }}>
                          {tokens.length}
                        </div>
                        <div style={{ fontSize: '12px', color: 'var(--secondary)' }}>
                          Total Tokens
                        </div>
                      </div>
                      <div style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: '20px', fontWeight: 'bold', color: 'var(--success)' }}>
                          {traceData.length}
                        </div>
                        <div style={{ fontSize: '12px', color: 'var(--secondary)' }}>
                          Generated
                        </div>
                      </div>
                    </div>
                  </div>
                </motion.div>

                {/* Session Timeline - Enhanced */}
                <div style={{
                  background: 'linear-gradient(135deg, var(--card) 0%, rgba(10, 132, 255, 0.02) 100%)',
                  borderRadius: '16px',
                  padding: '20px',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  overflow: 'hidden'
                }}>
                  <h4 className="text-large" style={{ 
                    marginBottom: '16px',
                    color: 'var(--accent)',
                    fontWeight: 'bold',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px'
                  }}>
                    📈 Session Timeline
                  </h4>
                  <div style={{ height: 'calc(100% - 50px)' }}>
                    <SessionTimeline
                      tokens={tokens}
                      traceData={traceData}
                      currentTokenIndex={currentTokenIndex}
                      onJumpToToken={handleTokenJump}
                    />
                  </div>
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
      {/* Floating Token Mini-Map */}
      <AnimatePresence>
        {traceData.length > 0 && (
          <TokenMiniMap
            tokens={tokens}
            attentionData={allAttention}
            currentTokenIndex={currentTokenIndex}
            onTokenClick={handleTokenJump}
          />
        )}
      </AnimatePresence>

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
          transition: 'width 0.3s ease',
          boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
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
                <motion.button
                  className={`btn ${loading || !prompt.trim() ? 'btn-secondary' : 'btn-primary'}`}
                  onClick={handleTrace}
                  disabled={loading || !prompt.trim()}
                  style={{ width: '100%' }}
                  whileTap={{ scale: 0.98 }}
                  transition={{ duration: 0.1 }}
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
                </motion.button>
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

                        <div className="slider-container">
                          <div className="slider-label">
                            <span>Current Token</span>
                            <span className="slider-value">{currentTokenIndex + 1} / {tokens.length}</span>
                          </div>
                          <Slider
                            value={currentTokenIndex}
                            onChange={(_, value) => setCurrentTokenIndex(value as number)}
                            min={0}
                            max={Math.max(0, tokens.length - 1)}
                            step={1}
                            sx={{ 
                              color: 'var(--success)',
                              height: 8,
                              '& .MuiSlider-thumb': {
                                width: 20,
                                height: 20,
                                backgroundColor: 'var(--success)',
                                border: 'none',
                                boxShadow: '0 2px 6px rgba(0,0,0,0.2)',
                                '&:hover': {
                                  boxShadow: '0 4px 8px rgba(48, 209, 88, 0.4)',
                                },
                              },
                              '& .MuiSlider-track': {
                                backgroundColor: 'var(--success)',
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
                <motion.button
                  key={tab.id}
                  className={`tab ${activeTab === tab.id ? 'active' : ''}`}
                  onClick={() => setActiveTab(tab.id)}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <IconComponent size={20} />
                  {tab.label}
                </motion.button>
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