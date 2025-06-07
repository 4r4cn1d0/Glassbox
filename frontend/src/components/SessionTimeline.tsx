import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Clock, Zap, Eye, Target } from 'lucide-react';

interface TimelineEvent {
  id: string;
  type: 'token' | 'intervention' | 'analysis' | 'attention_peak';
  timestamp: number;
  tokenIndex: number;
  data: {
    token?: string;
    confidence?: number;
    attention_intensity?: number;
    intervention_type?: string;
    description?: string;
  };
}

interface SessionTimelineProps {
  tokens: string[];
  traceData: any[];
  currentTokenIndex: number;
  onJumpToToken: (index: number) => void;
}

const SessionTimeline: React.FC<SessionTimelineProps> = ({
  tokens,
  traceData,
  currentTokenIndex,
  onJumpToToken
}) => {
  const [selectedEvent, setSelectedEvent] = useState<TimelineEvent | null>(null);

  // Generate timeline events from trace data
  const generateTimelineEvents = (): TimelineEvent[] => {
    const events: TimelineEvent[] = [];
    
    traceData.forEach((trace, index) => {
      // Token generation event
      events.push({
        id: `token-${index}`,
        type: 'token',
        timestamp: Date.now() - (traceData.length - index) * 1000,
        tokenIndex: index,
        data: {
          token: trace.token,
          confidence: Math.random() * 0.4 + 0.6 // Mock confidence
        }
      });

      // Add attention peaks (mock)
      if (Math.random() > 0.7) {
        events.push({
          id: `attention-${index}`,
          type: 'attention_peak',
          timestamp: Date.now() - (traceData.length - index) * 1000 + 500,
          tokenIndex: index,
          data: {
            attention_intensity: Math.random() * 0.5 + 0.5,
            description: `High attention to token "${trace.token}"`
          }
        });
      }

      // Add interventions (mock)
      if (Math.random() > 0.8) {
        events.push({
          id: `intervention-${index}`,
          type: 'intervention',
          timestamp: Date.now() - (traceData.length - index) * 1000 + 200,
          tokenIndex: index,
          data: {
            intervention_type: 'confidence_boost',
            description: `Confidence adjusted for "${trace.token}"`
          }
        });
      }
    });

    return events.sort((a, b) => a.timestamp - b.timestamp);
  };

  const events = generateTimelineEvents();

  const getEventIcon = (type: string) => {
    switch (type) {
      case 'token': return <Target size={12} />;
      case 'intervention': return <Zap size={12} />;
      case 'analysis': return <Eye size={12} />;
      case 'attention_peak': return <Clock size={12} />;
      default: return <Target size={12} />;
    }
  };

  const getEventColor = (type: string) => {
    switch (type) {
      case 'token': return 'var(--accent)';
      case 'intervention': return 'var(--warning)';
      case 'analysis': return 'var(--success)';
      case 'attention_peak': return '#FF6B35';
      default: return 'var(--secondary)';
    }
  };

  const formatTimestamp = (timestamp: number) => {
    const now = Date.now();
    const diff = now - timestamp;
    const seconds = Math.floor(diff / 1000);
    
    if (seconds < 60) return `${seconds}s ago`;
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    return `${hours}h ago`;
  };

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div style={{ 
        display: 'flex', 
        alignItems: 'center', 
        marginBottom: '16px',
        gap: '8px'
      }}>
        <Clock size={16} style={{ color: 'var(--accent)' }} />
        <h4 className="text-medium" style={{ margin: 0 }}>
          Session Timeline
        </h4>
        <div style={{
          fontSize: '11px',
          color: 'var(--secondary)',
          background: 'var(--bg)',
          padding: '2px 8px',
          borderRadius: '10px'
        }}>
          {events.length} events
        </div>
      </div>

      {/* Timeline visualization */}
      <div style={{
        position: 'relative',
        height: '80px',
        background: 'var(--bg)',
        borderRadius: '12px',
        padding: '16px',
        marginBottom: '16px',
        overflow: 'hidden'
      }}>
        {/* Timeline base line */}
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '16px',
          right: '16px',
          height: '2px',
          background: 'var(--secondary)',
          opacity: 0.3,
          borderRadius: '1px'
        }} />

        {/* Current progress */}
        <motion.div
          style={{
            position: 'absolute',
            top: '50%',
            left: '16px',
            height: '2px',
            background: 'var(--accent)',
            borderRadius: '1px',
            width: `${(currentTokenIndex / Math.max(tokens.length - 1, 1)) * 100}%`
          }}
          transition={{ duration: 0.3 }}
        />

        {/* Timeline events */}
        {events.map((event, index) => {
          const position = (event.tokenIndex / Math.max(tokens.length - 1, 1)) * 100;
          const isCurrent = event.tokenIndex === currentTokenIndex;
          
          return (
            <motion.div
              key={event.id}
              style={{
                position: 'absolute',
                left: `${Math.min(position, 95)}%`,
                top: '50%',
                transform: 'translate(-50%, -50%)',
                cursor: 'pointer',
                zIndex: isCurrent ? 10 : 5
              }}
              whileHover={{ scale: 1.2 }}
              whileTap={{ scale: 0.9 }}
              onClick={() => {
                onJumpToToken(event.tokenIndex);
                setSelectedEvent(event);
              }}
            >
              <div style={{
                width: '12px',
                height: '12px',
                borderRadius: '50%',
                background: getEventColor(event.type),
                border: isCurrent ? '2px solid white' : 'none',
                boxShadow: isCurrent 
                  ? '0 0 0 2px var(--accent), 0 0 8px rgba(10, 132, 255, 0.4)'
                  : '0 2px 4px rgba(0,0,0,0.1)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'white',
                fontSize: '8px'
              }}>
                {getEventIcon(event.type)}
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Event details list */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        maxHeight: '300px'
      }}>
        {events.slice().reverse().map((event, index) => {
          const isCurrent = event.tokenIndex === currentTokenIndex;
          const isSelected = selectedEvent?.id === event.id;
          
          return (
            <motion.div
              key={event.id}
              className="timeline-event"
              style={{
                padding: '8px 12px',
                marginBottom: '4px',
                borderRadius: '8px',
                background: isCurrent 
                  ? 'rgba(10, 132, 255, 0.1)' 
                  : isSelected 
                    ? 'rgba(10, 132, 255, 0.05)'
                    : 'transparent',
                border: isCurrent ? '1px solid var(--accent)' : '1px solid transparent',
                cursor: 'pointer',
                transition: 'all 0.2s ease'
              }}
              whileHover={{ 
                background: 'rgba(10, 132, 255, 0.05)',
                scale: 1.02 
              }}
              onClick={() => onJumpToToken(event.tokenIndex)}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.2, delay: index * 0.02 }}
            >
              <div style={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: '8px',
                marginBottom: '2px'
              }}>
                <div style={{ 
                  color: getEventColor(event.type),
                  display: 'flex',
                  alignItems: 'center'
                }}>
                  {getEventIcon(event.type)}
                </div>
                
                <div style={{
                  fontSize: '12px',
                  fontWeight: 500,
                  color: 'var(--fg)'
                }}>
                  {event.type === 'token' && `Token: "${event.data.token}"`}
                  {event.type === 'intervention' && 'Intervention'}
                  {event.type === 'attention_peak' && 'Attention Peak'}
                  {event.type === 'analysis' && 'Analysis'}
                </div>
                
                <div style={{
                  marginLeft: 'auto',
                  fontSize: '10px',
                  color: 'var(--secondary)'
                }}>
                  {formatTimestamp(event.timestamp)}
                </div>
              </div>
              
              {event.data.description && (
                <div style={{
                  fontSize: '10px',
                  color: 'var(--secondary)',
                  marginLeft: '20px'
                }}>
                  {event.data.description}
                </div>
              )}
              
              {event.data.confidence && (
                <div style={{
                  marginLeft: '20px',
                  marginTop: '4px'
                }}>
                  <div style={{
                    fontSize: '9px',
                    color: 'var(--secondary)',
                    marginBottom: '2px'
                  }}>
                    Confidence: {(event.data.confidence * 100).toFixed(1)}%
                  </div>
                  <div style={{
                    width: '60px',
                    height: '3px',
                    background: 'var(--bg)',
                    borderRadius: '2px',
                    overflow: 'hidden'
                  }}>
                    <div style={{
                      width: `${event.data.confidence * 100}%`,
                      height: '100%',
                      background: 'var(--accent)',
                      borderRadius: '2px'
                    }} />
                  </div>
                </div>
              )}
            </motion.div>
          );
        })}
      </div>
    </div>
  );
};

export default SessionTimeline; 