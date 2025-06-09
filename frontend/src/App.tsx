import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Slider } from '@mui/material';
import { 
  Grid3X3, 
  BarChart3, 
  FileText, 
  Play, 
  Settings, 
  ZoomIn, 
  ZoomOut, 
  RotateCcw,
  ChevronLeft,
  ChevronRight,
  Loader2,
  Map,
  Eye,
  Zap
} from 'lucide-react';
import AttentionHeatmap from './AttentionHeatmap';
import TokenProbabilityBars from './TokenProbabilityBars';
import SkeletonLoader from './components/SkeletonLoader';
import TokenMiniMap from './components/TokenMiniMap';
import HeadThumbnails from './components/HeadThumbnails';
import SessionTimeline from './components/SessionTimeline';
import EmbeddingProjection from './components/EmbeddingProjection';
import LogitLens from './components/LogitLens';
import ComponentHooks from './components/ComponentHooks';
import LandingPage from './components/LandingPage';
import './theme.css';

interface TokenData {
  token: string;
  token_id: number;
  probability: number;
  logit: number;
}

interface TraceData {
  token: string;
  token_id: number;
  position: number;
  logits: number[];
  probabilities: number[];
  attention: any[];
  top_tokens: TokenData[];
  embedding: number[];
  is_generated: boolean;
}

interface PromptToken {
  token: string;
  token_id: number;
  embedding: number[];
}

interface ApiResponse {
  session_id: string;
  prompt: string;
  trace_data: TraceData[];
  prompt_tokens: PromptToken[];
  all_embeddings: number[][];
  vocabulary_size: number;
}

type ViewTab = 'heatmap' | 'probabilities' | 'analysis' | 'embeddings' | 'logit_lens' | 'component_hooks';

function App() {
  const [showLandingPage, setShowLandingPage] = useState(true);
  const [prompt, setPrompt] = useState('The secret to happiness is');
  const [apiResponse, setApiResponse] = useState<ApiResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedLayer, setSelectedLayer] = useState(0);
  const [selectedHead, setSelectedHead] = useState(0);
  const [activeTab, setActiveTab] = useState<ViewTab>('heatmap');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [currentTokenIndex, setCurrentTokenIndex] = useState(0);
  const [zoomLevel, setZoomLevel] = useState(1);

  const handleLaunchApp = () => {
    setShowLandingPage(false);
  };

  // Force light mode - no dark mode toggle
  useEffect(() => {
    document.documentElement.removeAttribute('data-theme');
  }, []);

  // Early return for landing page
  if (showLandingPage) {
    return <LandingPage onLaunchApp={handleLaunchApp} />;
  }

  const handleTrace = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/trace', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          prompt, 
          max_new_tokens: 20,
          top_k_attention: 50,
          top_k_tokens: 20
        }),
      });
      const data: ApiResponse = await response.json();
      setApiResponse(data);
      setCurrentTokenIndex(0);
    } catch (error) {
      console.error('Error:', error);
    }
    setLoading(false);
  };

  const handleTimelineScrub = async (tokenIndex: number) => {
    if (!apiResponse?.session_id) return;
    
    try {
      const response = await fetch('http://localhost:8000/api/scrub', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: apiResponse.session_id,
          token_index: tokenIndex
        }),
      });
      
      if (response.ok) {
        setCurrentTokenIndex(tokenIndex);
      }
    } catch (error) {
      console.error('Scrubbing error:', error);
      // Fallback to local index change
      setCurrentTokenIndex(tokenIndex);
    }
  };

  const handleTokenJump = (index: number) => {
    handleTimelineScrub(index);
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

  // Derived data
  const traceData = apiResponse?.trace_data || [];
  const promptTokens = apiResponse?.prompt_tokens || [];
  const allTokens = [...promptTokens.map(t => t.token), ...traceData.map(t => t.token)];
  const allAttention = traceData.length > 0 ? traceData[0].attention : [];
  const topTokensData = traceData.map(item => item.top_tokens);

  const maxLayers = allAttention.length;
  const maxHeads = allAttention[0]?.length || 0;

  const tabs = [
    { id: 'heatmap' as ViewTab, label: 'Heatmap', icon: Grid3X3 },
    { id: 'probabilities' as ViewTab, label: 'Probabilities', icon: BarChart3 },
    { id: 'embeddings' as ViewTab, label: 'Embeddings', icon: Map },
    { id: 'logit_lens' as ViewTab, label: 'Neural Lens', icon: Eye },
    { id: 'component_hooks' as ViewTab, label: 'Circuit Analysis', icon: Zap },
    { id: 'analysis' as ViewTab, label: 'Analysis', icon: FileText },
  ];

  const renderActiveView = () => {
    if (loading) {
      return <SkeletonLoader type="visualization" />;
    }

    if (!apiResponse || traceData.length === 0) {
      return (
        <motion.div 
          style={{ 
            textAlign: 'center',
            padding: '60px 40px',
            background: '#FFFFFF',
            borderRadius: '16px',
            border: '1px solid #E9ECEF'
          }}
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
        >
          <div style={{ fontSize: '48px', marginBottom: '16px' }}>🧠</div>
          <h3 style={{ 
            marginBottom: '8px',
            fontSize: '24px',
            fontWeight: 'bold',
            color: '#495057'
          }}>
            Welcome to Glassbox
          </h3>
          <p style={{ 
            color: '#6C757D', 
            maxWidth: '400px', 
            margin: '0 auto',
            fontSize: '16px'
          }}>
            Enter a prompt to begin exploring LLM attention patterns and token generation in real-time.
          </p>
        </motion.div>
      );
    }

    switch (activeTab) {
      case 'heatmap':
        return (
          <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
            style={{ position: 'relative' }}
          >
            <div style={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              alignItems: 'center',
              marginBottom: '20px',
              background: '#FFFFFF',
              padding: '16px',
              borderRadius: '12px',
              border: '1px solid #E9ECEF'
            }}>
              <h3 style={{
                fontSize: '20px',
                fontWeight: 'bold',
                color: '#495057',
                margin: 0
              }}>
                Attention Heatmap
              </h3>
              <div style={{ display: 'flex', gap: '8px' }}>
                <button 
                  onClick={handleZoomOut} 
                  title="Zoom Out"
                  style={{
                    background: '#F8F9FA',
                    border: '1px solid #DEE2E6',
                    borderRadius: '6px',
                    padding: '8px',
                    cursor: 'pointer',
                    color: '#495057'
                  }}
                >
                  <ZoomOut size={16} />
                </button>
                <button 
                  onClick={handleZoomIn} 
                  title="Zoom In"
                  style={{
                    background: '#F8F9FA',
                    border: '1px solid #DEE2E6',
                    borderRadius: '6px',
                    padding: '8px',
                    cursor: 'pointer',
                    color: '#495057'
                  }}
                >
                  <ZoomIn size={16} />
                </button>
                <button 
                  onClick={handleResetZoom} 
                  title="Reset Zoom"
                  style={{
                    background: '#F8F9FA',
                    border: '1px solid #DEE2E6',
                    borderRadius: '6px',
                    padding: '8px',
                    cursor: 'pointer',
                    color: '#495057'
                  }}
                >
                  <RotateCcw size={16} />
                </button>
              </div>
            </div>
            <div style={{ 
              transform: `scale(${zoomLevel})`, 
              transformOrigin: 'top left', 
              overflow: 'auto' 
            }}>
              <AttentionHeatmap
                attention={allAttention}
                tokens={allTokens}
                selectedLayer={selectedLayer}
                selectedHead={selectedHead}
                currentTokenIndex={currentTokenIndex}
              />
              <HeadThumbnails
                attention={allAttention}
                tokens={allTokens}
                selectedLayer={selectedLayer}
                selectedHead={selectedHead}
                onHeadSelect={handleHeadSelect}
              />
            </div>
          </motion.div>
        );

      case 'probabilities':
        return (
          <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
            style={{ height: '100%' }}
          >
            <TokenProbabilityBars
              topTokensData={topTokensData}
              currentTokenIndex={Math.max(0, currentTokenIndex - promptTokens.length)}
              onTokenSelect={(index: number) => handleTokenJump(index + promptTokens.length)}
            />
          </motion.div>
        );

      case 'embeddings':
        return (
          <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
            style={{ height: '100%' }}
          >
            <EmbeddingProjection
              sessionId={apiResponse?.session_id || null}
              currentTokenIndex={currentTokenIndex}
              onTokenClick={handleTokenJump}
            />
          </motion.div>
        );

      case 'analysis':
        return (
          <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
            style={{ 
              height: '100%',
              display: 'flex',
              flexDirection: 'column'
            }}
          >
            <div style={{ 
              flexShrink: 0,
              marginBottom: '20px',
              background: '#FFFFFF',
              padding: '16px',
              borderRadius: '12px',
              border: '1px solid #E9ECEF'
            }}>
              <h3 style={{
                fontSize: '20px',
                fontWeight: 'bold',
                color: '#495057',
                margin: 0
              }}>
                Analysis & Timeline
              </h3>
            </div>
            <div style={{ 
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              gap: '20px',
              padding: '20px',
              minHeight: 0
            }}>
              {/* Top Section - Generated Text Display */}
              <div style={{ 
                background: 'linear-gradient(135deg, #FFFFFF 0%, #F8F9FA 100%)',
                borderRadius: '16px',
                padding: '24px',
                border: '1px solid #E9ECEF',
                flex: '1 1 60%',
                minHeight: '300px'
              }}>
                <h4 style={{ 
                  marginBottom: '20px',
                  color: '#007BFF',
                  fontWeight: 'bold',
                  fontSize: '20px'
                }}>
                  📝 Generated Output
                </h4>
                <div style={{ 
                  fontFamily: 'Monaco, Consolas, "SF Mono", monospace',
                  background: '#F8F9FA',
                  padding: '24px',
                  borderRadius: '12px',
                  fontSize: '18px',
                  lineHeight: '1.8',
                  wordBreak: 'break-word',
                  height: 'calc(100% - 80px)',
                  overflow: 'auto',
                  border: '1px solid #E9ECEF',
                  boxShadow: 'inset 0 2px 8px rgba(0,0,0,0.05)'
                }}>
                  {promptTokens.map((token, index) => (
                    <span key={`prompt-${index}`} style={{ 
                      color: '#495057',
                      background: index === currentTokenIndex 
                        ? 'rgba(0, 123, 255, 0.3)' : 'transparent',
                      padding: '4px 8px',
                      borderRadius: '6px',
                      margin: '2px',
                      display: 'inline-block',
                      transition: 'all 0.2s ease',
                      cursor: 'pointer'
                    }}
                    onClick={() => handleTokenJump(index)}
                    >
                      {token.token}
                    </span>
                  ))}
                  {traceData.map((item, index) => (
                    <span key={`generated-${index}`} style={{ 
                      color: '#007BFF', 
                      fontWeight: '700',
                      background: index + promptTokens.length === currentTokenIndex 
                        ? 'rgba(0, 123, 255, 0.4)' 
                        : 'rgba(0, 123, 255, 0.15)',
                      padding: '4px 8px',
                      borderRadius: '6px',
                      margin: '2px',
                      display: 'inline-block',
                      position: 'relative',
                      boxShadow: index + promptTokens.length === currentTokenIndex 
                        ? '0 0 12px rgba(0, 123, 255, 0.8), 0 0 24px rgba(0, 123, 255, 0.4)' : '0 2px 4px rgba(0,0,0,0.1)',
                      animation: index + promptTokens.length === currentTokenIndex 
                        ? 'pulse 1.5s infinite' : 'none',
                      transform: index + promptTokens.length === currentTokenIndex 
                        ? 'scale(1.05)' : 'scale(1)',
                      transition: 'all 0.3s ease',
                      cursor: 'pointer'
                    }}
                    onClick={() => handleTokenJump(index + promptTokens.length)}
                    >
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
                    background: 'linear-gradient(135deg, rgba(0, 123, 255, 0.08), rgba(0, 123, 255, 0.02))',
                    borderRadius: '16px',
                    border: '1px solid rgba(0, 123, 255, 0.2)',
                    boxShadow: '0 4px 16px rgba(0, 123, 255, 0.1)'
                  }}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 0.3 }}
                >
                  <h4 style={{ 
                    color: '#007BFF', 
                    marginBottom: '16px',
                    fontWeight: 'bold',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    fontSize: '18px'
                  }}>
                    💡 AI Insights
                  </h4>
                  <div style={{ 
                    display: 'flex', 
                    flexDirection: 'column',
                    gap: '12px'
                  }}>
                    {[
                      `Cached session: ${apiResponse?.session_id?.slice(-8) || 'N/A'}`,
                      `Vocabulary size: ${apiResponse?.vocabulary_size?.toLocaleString() || 'Unknown'}`,
                      `Current position: ${currentTokenIndex + 1}/${allTokens.length}`,
                      `Layer ${selectedLayer + 1}, Head ${selectedHead + 1} selected`
                    ].map((insight, index) => (
                      <div key={index} style={{
                        background: 'rgba(255, 255, 255, 0.8)',
                        padding: '12px 16px',
                        borderRadius: '10px',
                        fontSize: '14px',
                        color: '#495057',
                        border: '1px solid rgba(0, 123, 255, 0.1)',
                        position: 'relative',
                        overflow: 'hidden'
                      }}>
                        <div style={{
                          position: 'absolute',
                          left: 0,
                          top: 0,
                          bottom: 0,
                          width: '4px',
                          background: '#007BFF'
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
                    background: 'rgba(248, 249, 250, 0.8)',
                    borderRadius: '12px',
                    border: '1px solid #E9ECEF'
                  }}>
                    <h5 style={{ 
                      color: '#007BFF', 
                      margin: '0 0 12px 0',
                      fontSize: '16px'
                    }}>
                      📊 Statistics
                    </h5>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                      <div style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#007BFF' }}>
                          {allTokens.length}
                        </div>
                        <div style={{ fontSize: '12px', color: '#6C757D' }}>
                          Total Tokens
                        </div>
                      </div>
                      <div style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#28A745' }}>
                          {traceData.length}
                        </div>
                        <div style={{ fontSize: '12px', color: '#6C757D' }}>
                          Generated
                        </div>
                      </div>
                    </div>
                  </div>
                </motion.div>

                {/* Session Timeline - Enhanced */}
                <div style={{
                  background: 'linear-gradient(135deg, #FFFFFF 0%, #F8F9FA 100%)',
                  borderRadius: '16px',
                  padding: '20px',
                  border: '1px solid #E9ECEF',
                  overflow: 'hidden'
                }}>
                  <h4 style={{ 
                    marginBottom: '16px',
                    color: '#007BFF',
                    fontWeight: 'bold',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    fontSize: '18px'
                  }}>
                    📈 Session Timeline
                  </h4>
                  <div style={{ height: 'calc(100% - 50px)' }}>
                    <SessionTimeline
                      tokens={allTokens}
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

      case 'logit_lens':
        return (
          <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
            style={{ height: '100%' }}
          >
            <LogitLens
              sessionId={apiResponse?.session_id || null}
              currentTokenIndex={currentTokenIndex}
              onLayerSelect={(layer: number) => setSelectedLayer(layer)}
            />
          </motion.div>
        );

      case 'component_hooks':
        return (
          <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
            style={{ height: '100%' }}
          >
            <ComponentHooks
              sessionId={apiResponse?.session_id || null}
              currentTokenIndex={currentTokenIndex}
              selectedLayer={selectedLayer}
              onComponentSelect={(component: string, layer: number) => {
                setSelectedLayer(layer);
                console.log(`Selected component: ${component} at layer ${layer}`);
              }}
            />
          </motion.div>
        );

      default:
        return null;
    }
  };

  return (
    <div style={{ background: '#F8F9FA', minHeight: '100vh', display: 'flex' }}>
      {/* Header with Logo and Back to Home */}
      <div style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        height: '60px',
        background: 'rgba(255, 255, 255, 0.95)',
        backdropFilter: 'blur(10px)',
        borderBottom: '1px solid #E9ECEF',
        zIndex: 1000,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 24px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <img 
            src="/logo_light.png" 
            alt="Glassbox Logo" 
            style={{ 
              height: '32px', 
              width: 'auto'
            }} 
          />
          <span style={{
            fontSize: '20px',
            fontWeight: '700',
            color: '#007aff'
          }}>
            Glassbox
          </span>
        </div>
        
        <button
          onClick={() => setShowLandingPage(true)}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            background: 'rgba(0, 122, 255, 0.1)',
            color: '#007aff',
            border: '1px solid rgba(0, 122, 255, 0.2)',
            padding: '8px 16px',
            borderRadius: '8px',
            fontWeight: '600',
            fontSize: '14px',
            cursor: 'pointer',
            transition: 'all 0.2s ease'
          }}
          onMouseOver={(e) => {
            e.currentTarget.style.background = 'rgba(0, 122, 255, 0.2)';
            e.currentTarget.style.borderColor = 'rgba(0, 122, 255, 0.4)';
          }}
          onMouseOut={(e) => {
            e.currentTarget.style.background = 'rgba(0, 122, 255, 0.1)';
            e.currentTarget.style.borderColor = 'rgba(0, 122, 255, 0.2)';
          }}
        >
          ← Back to Home
        </button>
      </div>

      {/* Floating Token Mini-Map */}
      <AnimatePresence>
        {traceData.length > 0 && (
          <TokenMiniMap
            tokens={allTokens}
            attentionData={allAttention}
            currentTokenIndex={currentTokenIndex}
            onTokenClick={handleTokenJump}
          />
        )}
      </AnimatePresence>

      {/* Collapsible Sidebar */}
      <motion.div 
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
          boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
          background: '#FFFFFF',
          border: '1px solid #E9ECEF',
          marginTop: '60px'
        }}
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.5 }}
      >
        {/* Sidebar Toggle */}
        <button
          style={{
            position: 'absolute',
            top: '16px',
            right: '16px',
            zIndex: 10,
            background: '#F8F9FA',
            border: '1px solid #DEE2E6',
            borderRadius: '8px',
            padding: '8px',
            cursor: 'pointer',
            color: '#495057'
          }}
          onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
        >
          {sidebarCollapsed ? <ChevronRight size={24} /> : <ChevronLeft size={24} />}
        </button>

        <div style={{ padding: sidebarCollapsed ? '16px 8px' : '24px', paddingTop: '60px' }}>
          {!sidebarCollapsed && (
            <>
              {/* Header - No Dark Mode Toggle */}
              <div style={{ marginBottom: '24px', justifyContent: 'flex-start' }}>
                <h1 style={{ 
                  background: 'linear-gradient(45deg, #007BFF, #0056B3)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  fontSize: '28px',
                  fontWeight: 'bold',
                  margin: 0
                }}>
                  Glassbox
                </h1>
              </div>

              {/* Input Section */}
              <motion.div 
                style={{ marginBottom: '24px' }}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
              >
                <h2 style={{ 
                  marginBottom: '16px',
                  fontSize: '18px',
                  fontWeight: 'bold',
                  color: '#495057'
                }}>
                  Input Prompt
                </h2>
                <textarea
                  placeholder="Enter your prompt to analyze..."
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  rows={4}
                  style={{ 
                    width: '100%',
                    marginBottom: '16px',
                    padding: '12px',
                    border: '2px solid #DEE2E6',
                    borderRadius: '8px',
                    fontSize: '14px',
                    fontFamily: 'SF Pro Text, -apple-system, sans-serif',
                    resize: 'vertical',
                    background: '#FFFFFF',
                    color: '#495057'
                  }}
                />
                <motion.button
                  onClick={handleTrace}
                  disabled={loading || !prompt.trim()}
                  style={{ 
                    width: '100%',
                    padding: '12px',
                    border: 'none',
                    borderRadius: '8px',
                    fontSize: '16px',
                    fontWeight: 'bold',
                    cursor: loading || !prompt.trim() ? 'not-allowed' : 'pointer',
                    background: loading || !prompt.trim() ? '#6C757D' : '#007BFF',
                    color: '#FFFFFF',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '8px'
                  }}
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
                    <h3 style={{ 
                      flex: 1,
                      fontSize: '18px',
                      fontWeight: 'bold',
                      color: '#495057',
                      margin: 0
                    }}>
                      Controls
                    </h3>
                    <button
                      onClick={() => setShowAdvanced(!showAdvanced)}
                      style={{ 
                        padding: '4px 8px',
                        background: 'transparent',
                        border: '1px solid #DEE2E6',
                        borderRadius: '6px',
                        cursor: 'pointer',
                        color: '#495057'
                      }}
                    >
                      <Settings size={20} />
                    </button>
                  </div>
                  
                  <AnimatePresence>
                    {showAdvanced && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        transition={{ duration: 0.3 }}
                      >
                        <div style={{ marginBottom: '20px' }}>
                          <div style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center',
                            marginBottom: '8px'
                          }}>
                            <span style={{ color: '#495057', fontWeight: 'bold' }}>Layer</span>
                            <span style={{ 
                              color: '#007BFF', 
                              fontWeight: 'bold',
                              background: '#E7F3FF',
                              padding: '4px 8px',
                              borderRadius: '4px'
                            }}>
                              {selectedLayer + 1} / {maxLayers}
                            </span>
                          </div>
                          <Slider
                            value={selectedLayer}
                            onChange={(_, value) => setSelectedLayer(value as number)}
                            min={0}
                            max={Math.max(0, maxLayers - 1)}
                            step={1}
                            sx={{ 
                              color: '#007BFF',
                              height: 8,
                              '& .MuiSlider-thumb': {
                                width: 20,
                                height: 20,
                                backgroundColor: '#007BFF',
                                border: 'none',
                                boxShadow: '0 2px 6px rgba(0,0,0,0.2)',
                                '&:hover': {
                                  boxShadow: '0 4px 8px rgba(0, 123, 255, 0.4)',
                                },
                              },
                              '& .MuiSlider-track': {
                                backgroundColor: '#007BFF',
                                border: 'none',
                                height: 8,
                              },
                              '& .MuiSlider-rail': {
                                backgroundColor: '#DEE2E6',
                                height: 8,
                              }
                            }}
                          />
                        </div>

                        <div style={{ marginBottom: '20px' }}>
                          <div style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center',
                            marginBottom: '8px'
                          }}>
                            <span style={{ color: '#495057', fontWeight: 'bold' }}>Attention Head</span>
                            <span style={{ 
                              color: '#007BFF', 
                              fontWeight: 'bold',
                              background: '#E7F3FF',
                              padding: '4px 8px',
                              borderRadius: '4px'
                            }}>
                              {selectedHead + 1} / {maxHeads}
                            </span>
                          </div>
                          <Slider
                            value={selectedHead}
                            onChange={(_, value) => setSelectedHead(value as number)}
                            min={0}
                            max={Math.max(0, maxHeads - 1)}
                            step={1}
                            sx={{ 
                              color: '#007BFF',
                              height: 8,
                              '& .MuiSlider-thumb': {
                                width: 20,
                                height: 20,
                                backgroundColor: '#007BFF',
                                border: 'none',
                                boxShadow: '0 2px 6px rgba(0,0,0,0.2)',
                                '&:hover': {
                                  boxShadow: '0 4px 8px rgba(0, 123, 255, 0.4)',
                                },
                              },
                              '& .MuiSlider-track': {
                                backgroundColor: '#007BFF',
                                border: 'none',
                                height: 8,
                              },
                              '& .MuiSlider-rail': {
                                backgroundColor: '#DEE2E6',
                                height: 8,
                              }
                            }}
                          />
                        </div>

                        <div style={{ marginBottom: '20px' }}>
                          <div style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center',
                            marginBottom: '8px'
                          }}>
                            <span style={{ color: '#495057', fontWeight: 'bold' }}>Current Token</span>
                            <span style={{ 
                              color: '#28A745', 
                              fontWeight: 'bold',
                              background: '#E8F5E8',
                              padding: '4px 8px',
                              borderRadius: '4px'
                            }}>
                              {currentTokenIndex + 1} / {allTokens.length}
                            </span>
                          </div>
                          <Slider
                            value={currentTokenIndex}
                            onChange={(_, value) => setCurrentTokenIndex(value as number)}
                            min={0}
                            max={Math.max(0, allTokens.length - 1)}
                            step={1}
                            sx={{ 
                              color: '#28A745',
                              height: 8,
                              '& .MuiSlider-thumb': {
                                width: 20,
                                height: 20,
                                backgroundColor: '#28A745',
                                border: 'none',
                                boxShadow: '0 2px 6px rgba(0,0,0,0.2)',
                                '&:hover': {
                                  boxShadow: '0 4px 8px rgba(40, 167, 69, 0.4)',
                                },
                              },
                              '& .MuiSlider-track': {
                                backgroundColor: '#28A745',
                                border: 'none',
                                height: 8,
                              },
                              '& .MuiSlider-rail': {
                                backgroundColor: '#DEE2E6',
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
        paddingTop: '84px', // Account for fixed header + padding
        overflow: 'auto'
      }}>
        {/* Tab Navigation */}
        {(traceData.length > 0 || loading) && (
          <motion.div 
            style={{
              display: 'flex',
              gap: '8px',
              marginBottom: '24px',
              background: '#FFFFFF',
              padding: '8px',
              borderRadius: '12px',
              border: '1px solid #E9ECEF',
              boxShadow: '0 2px 8px rgba(0,0,0,0.05)'
            }}
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            {tabs.map(tab => {
              const IconComponent = tab.icon;
              return (
                <motion.button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    padding: '12px 20px',
                    border: 'none',
                    borderRadius: '8px',
                    fontSize: '14px',
                    fontWeight: 'bold',
                    cursor: 'pointer',
                    background: activeTab === tab.id ? '#007BFF' : 'transparent',
                    color: activeTab === tab.id ? '#FFFFFF' : '#495057',
                    transition: 'all 0.2s ease'
                  }}
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