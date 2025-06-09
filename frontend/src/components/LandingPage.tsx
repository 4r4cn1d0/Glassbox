import React from 'react';
import { motion } from 'framer-motion';

interface LandingPageProps {
  onLaunchApp: () => void;
}

const LandingPage: React.FC<LandingPageProps> = ({ onLaunchApp }) => {
  const scrollToSection = (sectionId: string) => {
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <div style={{
      fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      lineHeight: '1.6',
      color: '#1a1a1a',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      minHeight: '100vh'
    }}>
      {/* Header */}
      <header style={{
        padding: '20px 0',
        background: 'rgba(255, 255, 255, 0.95)',
        backdropFilter: 'blur(10px)',
        borderBottom: '1px solid rgba(0, 0, 0, 0.1)',
        position: 'sticky',
        top: 0,
        zIndex: 100
      }}>
        <div style={{
          maxWidth: '1200px',
          margin: '0 auto',
          padding: '0 24px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <img 
              src="/logo_light.png" 
              alt="Glassbox" 
              style={{ height: '32px', width: 'auto' }}
            />
            <span style={{
              fontSize: '24px',
              fontWeight: '700',
              color: '#007aff'
            }}>
              Glassbox
            </span>
          </div>
          
          <nav style={{ display: 'flex', gap: '32px', alignItems: 'center' }}>
            <button
              onClick={() => scrollToSection('features')}
              style={{ 
                background: 'none',
                border: 'none',
                color: '#666', 
                fontWeight: '500',
                fontSize: '16px',
                cursor: 'pointer',
                transition: 'color 0.2s ease'
              }}
              onMouseOver={(e) => e.currentTarget.style.color = '#007aff'}
              onMouseOut={(e) => e.currentTarget.style.color = '#666'}
            >
              Features
            </button>
            <button
              onClick={() => scrollToSection('demo')}
              style={{ 
                background: 'none',
                border: 'none',
                color: '#666', 
                fontWeight: '500',
                fontSize: '16px',
                cursor: 'pointer',
                transition: 'color 0.2s ease'
              }}
              onMouseOver={(e) => e.currentTarget.style.color = '#007aff'}
              onMouseOut={(e) => e.currentTarget.style.color = '#666'}
            >
              Demo
            </button>
            <button
              onClick={onLaunchApp}
              style={{
                background: '#007aff',
                color: 'white',
                border: 'none',
                padding: '12px 24px',
                borderRadius: '12px',
                fontWeight: '600',
                fontSize: '16px',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                boxShadow: '0 4px 12px rgba(0, 122, 255, 0.3)'
              }}
              onMouseOver={(e) => {
                e.currentTarget.style.background = '#0056b3';
                e.currentTarget.style.transform = 'translateY(-1px)';
              }}
              onMouseOut={(e) => {
                e.currentTarget.style.background = '#007aff';
                e.currentTarget.style.transform = 'translateY(0)';
              }}
            >
              Launch App
            </button>
          </nav>
        </div>
      </header>

      {/* Hero Section */}
      <section style={{
        padding: '120px 24px',
        textAlign: 'center',
        maxWidth: '1200px',
        margin: '0 auto'
      }}>
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <h1 style={{
            fontSize: 'clamp(48px, 8vw, 72px)',
            fontWeight: '800',
            color: 'white',
            margin: '0 0 24px 0',
            letterSpacing: '-0.02em',
            textShadow: '0 4px 12px rgba(0, 0, 0, 0.3)'
          }}>
            Your beautiful LLM toolkit.
          </h1>
          
          <p style={{
            fontSize: 'clamp(20px, 3vw, 28px)',
            color: 'rgba(255, 255, 255, 0.9)',
            maxWidth: '800px',
            margin: '0 auto 48px auto',
            fontWeight: '500',
            textShadow: '0 2px 8px rgba(0, 0, 0, 0.2)'
          }}>
            From token generation to attention visualization
          </p>

          <div style={{
            display: 'flex',
            gap: '20px',
            justifyContent: 'center',
            flexWrap: 'wrap',
            marginBottom: '80px'
          }}>
            <button
              onClick={onLaunchApp}
              style={{
                background: 'white',
                color: '#007aff',
                border: 'none',
                padding: '16px 32px',
                borderRadius: '16px',
                fontWeight: '700',
                fontSize: '18px',
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                boxShadow: '0 8px 24px rgba(0, 0, 0, 0.2)',
                minWidth: '180px'
              }}
              onMouseOver={(e) => {
                e.currentTarget.style.transform = 'translateY(-2px)';
                e.currentTarget.style.boxShadow = '0 12px 32px rgba(0, 0, 0, 0.3)';
              }}
              onMouseOut={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = '0 8px 24px rgba(0, 0, 0, 0.2)';
              }}
            >
              Launch Debugger
            </button>
            
            <button
              style={{
                background: 'rgba(255, 255, 255, 0.1)',
                color: 'white',
                border: '2px solid rgba(255, 255, 255, 0.3)',
                padding: '14px 32px',
                borderRadius: '16px',
                fontWeight: '600',
                fontSize: '18px',
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                minWidth: '180px',
                backdropFilter: 'blur(10px)'
              }}
              onClick={() => scrollToSection('demo')}
              onMouseOver={(e) => {
                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.2)';
                e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.5)';
              }}
              onMouseOut={(e) => {
                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';
                e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.3)';
              }}
            >
              View Demo
            </button>
          </div>
        </motion.div>

        {/* Feature Preview */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          style={{
            background: 'rgba(255, 255, 255, 0.95)',
            borderRadius: '24px',
            padding: '40px',
            boxShadow: '0 20px 60px rgba(0, 0, 0, 0.3)',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255, 255, 255, 0.2)'
          }}
        >
          <h3 style={{
            fontSize: '24px',
            fontWeight: '700',
            color: '#333',
            marginBottom: '24px'
          }}>
            Real-time LLM introspection at your fingertips
          </h3>
          
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
            gap: '24px',
            marginTop: '32px'
          }}>
            <div style={{
              padding: '24px',
              background: 'linear-gradient(135deg, #667eea, #764ba2)',
              borderRadius: '16px',
              color: 'white',
              textAlign: 'center'
            }}>
              <div style={{ fontSize: '32px', marginBottom: '12px' }}>🔥</div>
              <h4 style={{ fontSize: '18px', fontWeight: '600', margin: '0 0 8px 0' }}>
                Attention Heatmaps
              </h4>
              <p style={{ fontSize: '14px', opacity: 0.9, margin: 0 }}>
                Visualize token-to-token attention patterns across all heads and layers
              </p>
            </div>
            
            <div style={{
              padding: '24px',
              background: 'linear-gradient(135deg, #f093fb, #f5576c)',
              borderRadius: '16px',
              color: 'white',
              textAlign: 'center'
            }}>
              <div style={{ fontSize: '32px', marginBottom: '12px' }}>📊</div>
              <h4 style={{ fontSize: '18px', fontWeight: '600', margin: '0 0 8px 0' }}>
                Token Probabilities
              </h4>
              <p style={{ fontSize: '14px', opacity: 0.9, margin: 0 }}>
                Track probability distributions and entropy at each generation step
              </p>
            </div>
            
            <div style={{
              padding: '24px',
              background: 'linear-gradient(135deg, #4facfe, #00f2fe)',
              borderRadius: '16px',
              color: 'white',
              textAlign: 'center'
            }}>
              <div style={{ fontSize: '32px', marginBottom: '12px' }}>🗺️</div>
              <h4 style={{ fontSize: '18px', fontWeight: '600', margin: '0 0 8px 0' }}>
                Embedding Space
              </h4>
              <p style={{ fontSize: '14px', opacity: 0.9, margin: 0 }}>
                Explore 2D projections of high-dimensional token embeddings
              </p>
            </div>
            
            <div style={{
              padding: '24px',
              background: 'linear-gradient(135deg, #a8e6cf, #dcedc8)',
              borderRadius: '16px',
              color: '#2E7D32',
              textAlign: 'center'
            }}>
              <div style={{ fontSize: '32px', marginBottom: '12px' }}>👁️</div>
              <h4 style={{ fontSize: '18px', fontWeight: '600', margin: '0 0 8px 0' }}>
                Neural Lens
              </h4>
              <p style={{ fontSize: '14px', opacity: 0.9, margin: 0 }}>
                See what the model "thinks" at each layer with TransformerLens-style predictions
              </p>
            </div>
            
            <div style={{
              padding: '24px',
              background: 'linear-gradient(135deg, #ffecd2, #fcb69f)',
              borderRadius: '16px',
              color: '#E65100',
              textAlign: 'center'
            }}>
              <div style={{ fontSize: '32px', marginBottom: '12px' }}>⚡</div>
              <h4 style={{ fontSize: '18px', fontWeight: '600', margin: '0 0 8px 0' }}>
                Circuit Analysis
              </h4>
              <p style={{ fontSize: '14px', opacity: 0.9, margin: 0 }}>
                Hook into transformer internals: embeddings, attention, MLP, and residual streams
              </p>
            </div>
          </div>
        </motion.div>
      </section>

      {/* Features Section */}
      <section id="features" style={{
        padding: '120px 24px',
        background: 'rgba(255, 255, 255, 0.95)',
        backdropFilter: 'blur(10px)'
      }}>
        <div style={{
          maxWidth: '1200px',
          margin: '0 auto',
          textAlign: 'center'
        }}>
          <h2 style={{
            fontSize: 'clamp(36px, 6vw, 48px)',
            fontWeight: '800',
            color: '#333',
            marginBottom: '24px'
          }}>
            Debug LLMs with style.
          </h2>
          
          <p style={{
            fontSize: '20px',
            color: '#666',
            maxWidth: '600px',
            margin: '0 auto 80px auto'
          }}>
            Create stunning visualizations that reveal how language models think, 
            generate, and attend to information.
          </p>

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))',
            gap: '40px'
          }}>
            <div style={{
              padding: '40px',
              background: 'white',
              borderRadius: '20px',
              boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
              border: '1px solid rgba(0, 0, 0, 0.05)',
              textAlign: 'left'
            }}>
              <div style={{
                width: '60px',
                height: '60px',
                background: 'linear-gradient(135deg, #667eea, #764ba2)',
                borderRadius: '16px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '24px',
                marginBottom: '24px'
              }}>
                🎯
              </div>
              <h3 style={{
                fontSize: '24px',
                fontWeight: '700',
                color: '#333',
                marginBottom: '16px'
              }}>
                Real-time Generation
              </h3>
              <p style={{
                fontSize: '16px',
                color: '#666',
                lineHeight: '1.6'
              }}>
                Watch tokens generate in real-time with cached session replay, 
                letting you scrub through the generation timeline like a video.
              </p>
            </div>

            <div style={{
              padding: '40px',
              background: 'white',
              borderRadius: '20px',
              boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
              border: '1px solid rgba(0, 0, 0, 0.05)',
              textAlign: 'left'
            }}>
              <div style={{
                width: '60px',
                height: '60px',
                background: 'linear-gradient(135deg, #f093fb, #f5576c)',
                borderRadius: '16px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '24px',
                marginBottom: '24px'
              }}>
                🔍
              </div>
              <h3 style={{
                fontSize: '24px',
                fontWeight: '700',
                color: '#333',
                marginBottom: '16px'
              }}>
                Mechanistic Interpretability
              </h3>
              <p style={{
                fontSize: '16px',
                color: '#666',
                lineHeight: '1.6'
              }}>
                Hook into transformer components with TransformerLens-inspired tools.
                Explore logit lens predictions, component activations, and circuit-level 
                analysis across all 36 layers and 20 heads of GPT-2 Large.
              </p>
            </div>

            <div style={{
              padding: '40px',
              background: 'white',
              borderRadius: '20px',
              boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
              border: '1px solid rgba(0, 0, 0, 0.05)',
              textAlign: 'left'
            }}>
              <div style={{
                width: '60px',
                height: '60px',
                background: 'linear-gradient(135deg, #4facfe, #00f2fe)',
                borderRadius: '16px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '24px',
                marginBottom: '24px'
              }}>
                💎
              </div>
              <h3 style={{
                fontSize: '24px',
                fontWeight: '700',
                color: '#333',
                marginBottom: '16px'
              }}>
                Information Theory
              </h3>
              <p style={{
                fontSize: '16px',
                color: '#666',
                lineHeight: '1.6'
              }}>
                Deep insights with entropy analysis, top-K probability tracking, 
                and statistical measures of model confidence.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Demo Section */}
      <section id="demo" style={{
        padding: '120px 24px',
        background: 'rgba(255, 255, 255, 0.98)',
        backdropFilter: 'blur(10px)'
      }}>
        <div style={{
          maxWidth: '1200px',
          margin: '0 auto',
          textAlign: 'center'
        }}>
          <h2 style={{
            fontSize: 'clamp(36px, 6vw, 48px)',
            fontWeight: '800',
            color: '#333',
            marginBottom: '24px'
          }}>
            See it in action.
          </h2>
          
          <p style={{
            fontSize: '20px',
            color: '#666',
            marginBottom: '60px',
            maxWidth: '600px',
            margin: '0 auto 60px auto'
          }}>
            Watch how Glassbox reveals the inner workings of language models 
            through intuitive, real-time visualizations.
          </p>

          <div style={{
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            borderRadius: '20px',
            padding: '60px 40px',
            color: 'white',
            textAlign: 'center'
          }}>
            <div style={{ fontSize: '64px', marginBottom: '24px' }}>🎬</div>
            <h3 style={{
              fontSize: '28px',
              fontWeight: '700',
              marginBottom: '16px'
            }}>
              Interactive Demo
            </h3>
            <p style={{
              fontSize: '18px',
              opacity: 0.9,
              marginBottom: '32px',
              maxWidth: '500px',
              margin: '0 auto 32px auto'
            }}>
              Ready to explore? Launch the debugger and see how GPT-2 Large 
              processes text in real-time with attention patterns, token probabilities, 
              and embedding visualizations.
            </p>
            <button
              onClick={onLaunchApp}
              style={{
                background: 'white',
                color: '#667eea',
                border: 'none',
                padding: '16px 32px',
                borderRadius: '12px',
                fontWeight: '700',
                fontSize: '18px',
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                boxShadow: '0 4px 16px rgba(0, 0, 0, 0.2)'
              }}
              onMouseOver={(e) => {
                e.currentTarget.style.transform = 'translateY(-2px)';
                e.currentTarget.style.boxShadow = '0 8px 24px rgba(0, 0, 0, 0.3)';
              }}
              onMouseOut={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = '0 4px 16px rgba(0, 0, 0, 0.2)';
              }}
            >
              Try It Now →
            </button>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section style={{
        padding: '120px 24px',
        textAlign: 'center',
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
      }}>
        <div style={{
          maxWidth: '800px',
          margin: '0 auto'
        }}>
          <h2 style={{
            fontSize: 'clamp(36px, 6vw, 48px)',
            fontWeight: '800',
            color: 'white',
            marginBottom: '24px',
            textShadow: '0 4px 12px rgba(0, 0, 0, 0.3)'
          }}>
            Ready to see inside your models?
          </h2>
          
          <p style={{
            fontSize: '20px',
            color: 'rgba(255, 255, 255, 0.9)',
            marginBottom: '48px',
            textShadow: '0 2px 8px rgba(0, 0, 0, 0.2)'
          }}>
            Launch Glassbox and start exploring LLM internals in a whole new way.
          </p>

          <button
            onClick={onLaunchApp}
            style={{
              background: 'white',
              color: '#007aff',
              border: 'none',
              padding: '20px 40px',
              borderRadius: '16px',
              fontWeight: '700',
              fontSize: '20px',
              cursor: 'pointer',
              transition: 'all 0.3s ease',
              boxShadow: '0 8px 24px rgba(0, 0, 0, 0.2)'
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.transform = 'translateY(-2px)';
              e.currentTarget.style.boxShadow = '0 12px 32px rgba(0, 0, 0, 0.3)';
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.transform = 'translateY(0)';
              e.currentTarget.style.boxShadow = '0 8px 24px rgba(0, 0, 0, 0.2)';
            }}
          >
            Launch Glassbox
          </button>
        </div>
      </section>

      {/* Footer */}
      <footer style={{
        padding: '60px 24px 40px',
        background: 'rgba(255, 255, 255, 0.95)',
        backdropFilter: 'blur(10px)',
        borderTop: '1px solid rgba(0, 0, 0, 0.1)'
      }}>
        <div style={{
          maxWidth: '1200px',
          margin: '0 auto',
          textAlign: 'center'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '12px', marginBottom: '24px' }}>
            <img 
              src="/logo_light.png" 
              alt="Glassbox" 
              style={{ height: '24px', width: 'auto', opacity: 0.7 }}
            />
            <span style={{
              fontSize: '18px',
              fontWeight: '600',
              color: '#666'
            }}>
              Glassbox
            </span>
          </div>
          
          <p style={{
            color: '#999',
            fontSize: '14px',
            margin: 0
          }}>
            Built with transparency in mind. Made for AI researchers and enthusiasts.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage; 