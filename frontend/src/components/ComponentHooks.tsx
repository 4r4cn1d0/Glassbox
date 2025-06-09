import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  Zap, 
  Layers3, 
  Brain, 
  GitBranch, 
  Activity, 
  Target,
  ChevronDown,
  ChevronRight
} from 'lucide-react';

interface ComponentHooksProps {
  sessionId: string | null;
  currentTokenIndex: number;
  selectedLayer: number;
  onComponentSelect?: (component: string, layer: number) => void;
}

interface HookData {
  component: string;
  layer: number;
  activations: number[];
  magnitude: number;
  sparsity: number;
  top_neurons: Array<{
    index: number;
    activation: number;
    description?: string;
  }>;
}

const ComponentHooks: React.FC<ComponentHooksProps> = ({
  sessionId,
  currentTokenIndex,
  selectedLayer,
  onComponentSelect
}) => {
  const [hookData, setHookData] = useState<HookData[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedComponents, setSelectedComponents] = useState<Set<string>>(
    new Set(['embed', 'attn', 'mlp', 'ln'])
  );
  const [expandedComponents, setExpandedComponents] = useState<Set<string>>(new Set(['attn']));

  const componentTypes = [
    { id: 'embed', name: 'Token Embeddings', icon: Target, color: '#FF6B6B' },
    { id: 'pos_embed', name: 'Positional Embeddings', icon: GitBranch, color: '#4ECDC4' },
    { id: 'attn', name: 'Attention', icon: Brain, color: '#45B7D1' },
    { id: 'mlp', name: 'MLP', icon: Layers3, color: '#96CEB4' },
    { id: 'ln', name: 'Layer Norm', icon: Activity, color: '#FFEAA7' },
    { id: 'residual', name: 'Residual Stream', icon: Zap, color: '#DDA0DD' }
  ];

  useEffect(() => {
    if (sessionId && currentTokenIndex >= 0) {
      fetchComponentHooks();
    }
  }, [sessionId, currentTokenIndex, selectedLayer, selectedComponents]);

  const fetchComponentHooks = async () => {
    if (!sessionId) return;

    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/component-hooks', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          token_index: currentTokenIndex,
          layer: selectedLayer,
          components: Array.from(selectedComponents)
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setHookData(data.hook_data || []);
      }
    } catch (error) {
      console.error('Component hooks fetch error:', error);
      generateMockHookData();
    }
    setLoading(false);
  };

  const generateMockHookData = () => {
    const mockData = Array.from(selectedComponents).map(component => {
      const size = component === 'embed' ? 1600 : component === 'attn' ? 1600 : 6400; // GPT-2 Large dimensions
      const activations = Array.from({ length: 20 }, () => Math.random() * 2 - 1);
      
      return {
        component,
        layer: selectedLayer,
        activations,
        magnitude: Math.sqrt(activations.reduce((sum, val) => sum + val * val, 0) / activations.length),
        sparsity: activations.filter(val => Math.abs(val) < 0.1).length / activations.length,
        top_neurons: activations
          .map((activation, index) => ({ index, activation }))
          .sort((a, b) => Math.abs(b.activation) - Math.abs(a.activation))
          .slice(0, 5)
          .map(({ index, activation }) => ({
            index,
            activation,
            description: `Neuron ${index} in ${component}`
          }))
      };
    });
    setHookData(mockData);
  };

  const toggleComponent = (componentId: string) => {
    const newSelection = new Set(selectedComponents);
    if (newSelection.has(componentId)) {
      newSelection.delete(componentId);
    } else {
      newSelection.add(componentId);
    }
    setSelectedComponents(newSelection);
  };

  const toggleExpanded = (componentId: string) => {
    const newExpanded = new Set(expandedComponents);
    if (newExpanded.has(componentId)) {
      newExpanded.delete(componentId);
    } else {
      newExpanded.add(componentId);
    }
    setExpandedComponents(newExpanded);
  };

  const getComponentColor = (componentId: string) => {
    return componentTypes.find(c => c.id === componentId)?.color || '#6C757D';
  };

  const getComponentIcon = (componentId: string) => {
    return componentTypes.find(c => c.id === componentId)?.icon || Activity;
  };

  const renderActivationBar = (activation: number, maxAbs: number) => {
    const width = Math.abs(activation) / maxAbs * 100;
    const isPositive = activation > 0;
    
    return (
      <div style={{
        width: '100px',
        height: '8px',
        background: '#E9ECEF',
        borderRadius: '4px',
        position: 'relative',
        overflow: 'hidden'
      }}>
        <div style={{
          width: `${width}%`,
          height: '100%',
          background: isPositive ? '#28A745' : '#DC3545',
          borderRadius: '4px',
          transition: 'width 0.3s ease'
        }} />
      </div>
    );
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
          <Zap size={24} style={{ color: '#007BFF' }} />
          <h3 style={{
            fontSize: '20px',
            fontWeight: 'bold',
            color: '#495057',
            margin: 0
          }}>
            Circuit Analysis
          </h3>
        </div>

        <div style={{
          padding: '8px 12px',
          background: '#E7F3FF',
          borderRadius: '8px',
          fontSize: '14px',
          fontWeight: '600',
          color: '#007BFF'
        }}>
          Layer {selectedLayer} • Token {currentTokenIndex}
        </div>
      </div>

      {/* Component Selection */}
      <div style={{
        background: '#F8F9FA',
        borderRadius: '12px',
        padding: '16px'
      }}>
        <h4 style={{
          fontSize: '14px',
          fontWeight: '600',
          color: '#495057',
          marginBottom: '12px'
        }}>
          Component Selection
        </h4>
        
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
          gap: '8px'
        }}>
          {componentTypes.map(({ id, name, icon: Icon, color }) => (
            <button
              key={id}
              onClick={() => toggleComponent(id)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                padding: '10px 12px',
                border: selectedComponents.has(id) ? `2px solid ${color}` : '1px solid #DEE2E6',
                borderRadius: '8px',
                background: selectedComponents.has(id) ? `${color}20` : '#FFFFFF',
                color: selectedComponents.has(id) ? color : '#495057',
                fontSize: '12px',
                fontWeight: '600',
                cursor: 'pointer',
                transition: 'all 0.2s ease'
              }}
            >
              <Icon size={16} />
              {name}
            </button>
          ))}
        </div>
      </div>

      {/* Hook Data */}
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
            Loading component hooks...
          </div>
        ) : hookData.length === 0 ? (
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center',
            height: '200px',
            color: '#6C757D'
          }}>
            <Zap size={48} style={{ opacity: 0.3, marginBottom: '16px' }} />
            <p>Select components to view activations</p>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {hookData.map((data) => {
              const Icon = getComponentIcon(data.component);
              const isExpanded = expandedComponents.has(data.component);
              const maxAbs = Math.max(...data.activations.map(Math.abs));
              
              return (
                <motion.div
                  key={data.component}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3 }}
                  style={{
                    border: `2px solid ${getComponentColor(data.component)}`,
                    borderRadius: '12px',
                    overflow: 'hidden'
                  }}
                >
                  {/* Component Header */}
                  <div
                    style={{
                      background: `${getComponentColor(data.component)}20`,
                      padding: '16px',
                      cursor: 'pointer',
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center'
                    }}
                    onClick={() => toggleExpanded(data.component)}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                      <Icon size={20} style={{ color: getComponentColor(data.component) }} />
                      <h5 style={{
                        fontSize: '16px',
                        fontWeight: 'bold',
                        color: getComponentColor(data.component),
                        margin: 0
                      }}>
                        {componentTypes.find(c => c.id === data.component)?.name}
                      </h5>
                    </div>
                    
                    <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                      <div style={{ display: 'flex', gap: '16px', fontSize: '12px', color: '#6C757D' }}>
                        <span>Mag: {data.magnitude.toFixed(3)}</span>
                        <span>Sparsity: {(data.sparsity * 100).toFixed(1)}%</span>
                      </div>
                      {isExpanded ? <ChevronDown size={20} /> : <ChevronRight size={20} />}
                    </div>
                  </div>

                  {/* Expanded Content */}
                  {isExpanded && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.3 }}
                      style={{
                        background: '#FFFFFF',
                        padding: '16px'
                      }}
                    >
                      {/* Top Neurons */}
                      <h6 style={{
                        fontSize: '14px',
                        fontWeight: '600',
                        color: '#495057',
                        marginBottom: '12px'
                      }}>
                        Top Active Neurons
                      </h6>
                      
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                        {data.top_neurons.map(({ index, activation }, idx) => (
                          <div
                            key={idx}
                            style={{
                              display: 'flex',
                              justifyContent: 'space-between',
                              alignItems: 'center',
                              padding: '8px 12px',
                              background: '#F8F9FA',
                              borderRadius: '6px'
                            }}
                          >
                            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                              <span style={{
                                fontSize: '12px',
                                fontWeight: 'bold',
                                color: '#6C757D',
                                minWidth: '60px'
                              }}>
                                #{index}
                              </span>
                              {renderActivationBar(activation, maxAbs)}
                            </div>
                            
                            <span style={{
                              fontSize: '12px',
                              fontWeight: '600',
                              color: activation > 0 ? '#28A745' : '#DC3545'
                            }}>
                              {activation.toFixed(3)}
                            </span>
                          </div>
                        ))}
                      </div>

                      {/* Action Buttons */}
                      <div style={{
                        marginTop: '16px',
                        display: 'flex',
                        gap: '8px'
                      }}>
                        <button
                          onClick={() => onComponentSelect?.(data.component, data.layer)}
                          style={{
                            padding: '8px 16px',
                            background: getComponentColor(data.component),
                            color: '#FFFFFF',
                            border: 'none',
                            borderRadius: '6px',
                            fontSize: '12px',
                            fontWeight: '600',
                            cursor: 'pointer'
                          }}
                        >
                          Explore Component
                        </button>
                        <button
                          style={{
                            padding: '8px 16px',
                            background: '#F8F9FA',
                            color: '#495057',
                            border: '1px solid #DEE2E6',
                            borderRadius: '6px',
                            fontSize: '12px',
                            fontWeight: '600',
                            cursor: 'pointer'
                          }}
                        >
                          Ablate
                        </button>
                      </div>
                    </motion.div>
                  )}
                </motion.div>
              );
            })}
          </div>
        )}
      </div>
    </motion.div>
  );
};

export default ComponentHooks; 