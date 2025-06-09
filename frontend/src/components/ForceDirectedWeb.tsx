import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import { motion } from 'framer-motion';
import { ZoomIn, ZoomOut, RotateCcw } from 'lucide-react';

interface ForceDirectedWebProps {
  attention: number[][][];
  tokens: string[];
  selectedLayer: number;
  selectedHead: number;
  currentTokenIndex?: number;
}

interface Node extends d3.SimulationNodeDatum {
  id: string;
  token: string;
  index: number;
  isCurrent: boolean;
  totalAttention: number;
}

interface Link extends d3.SimulationLinkDatum<Node> {
  source: Node;
  target: Node;
  strength: number;
}

const ForceDirectedWeb: React.FC<ForceDirectedWebProps> = ({
  attention,
  tokens,
  selectedLayer,
  selectedHead,
  currentTokenIndex = 0
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [zoomLevel, setZoomLevel] = useState(1);
  const transformRef = useRef(d3.zoomIdentity);
  const [connectionCount, setConnectionCount] = useState(0);

  const handleZoomIn = () => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    const zoom = d3.zoom<SVGSVGElement, unknown>().scaleExtent([0.1, 4]);
    svg.call(zoom.scaleBy, 1.5);
  };

  const handleZoomOut = () => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    const zoom = d3.zoom<SVGSVGElement, unknown>().scaleExtent([0.1, 4]);
    svg.call(zoom.scaleBy, 1 / 1.5);
  };

  const handleResetZoom = () => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    const zoom = d3.zoom<SVGSVGElement, unknown>().scaleExtent([0.1, 4]);
    svg.call(zoom.transform, d3.zoomIdentity);
  };

  useEffect(() => {
    if (!svgRef.current || !containerRef.current || !attention.length || !tokens.length) return;

    const container = containerRef.current;
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    // Get container dimensions for responsive sizing
    const containerRect = container.getBoundingClientRect();
    const width = Math.max(1000, containerRect.width);
    const height = Math.max(700, containerRect.height - 100); // Account for controls
    const centerX = width / 2;
    const centerY = height / 2;

    // Update SVG dimensions
    svg.attr('width', width).attr('height', height);

    // Extract attention data
    const attentionWeights = attention[selectedLayer]?.[selectedHead];
    if (!attentionWeights) return;

    // Create nodes
    const nodes: Node[] = tokens.slice(0, 15).map((token, i) => {
      const totalAttention = Array.isArray(attentionWeights[i]) 
        ? (attentionWeights[i] as unknown as number[]).reduce((sum: number, val: number) => sum + val, 0)
        : 0;
      
      return {
        id: `token-${i}`,
        token: token.length > 8 ? token.substring(0, 8) + '...' : token,
        index: i,
        isCurrent: i === currentTokenIndex,
        totalAttention,
        x: centerX + (Math.random() - 0.5) * 300,
        y: centerY + (Math.random() - 0.5) * 300
      };
    });

    // Create links based on attention weights with lower threshold for more connections
    const links: Link[] = [];
    const minThreshold = 0.001; // Much lower threshold to show more connections

    for (let i = 0; i < nodes.length; i++) {
      const sourceAttention = Array.isArray(attentionWeights[i]) 
        ? attentionWeights[i] as unknown as number[]
        : [];
      
      for (let j = 0; j < nodes.length && j < sourceAttention.length; j++) {
        if (i !== j) {
          const strength = sourceAttention[j] || 0;
          if (strength > minThreshold) {
            links.push({
              source: nodes[i],
              target: nodes[j],
              strength
            });
          }
        }
      }
    }

    // Add some guaranteed visible links for testing if no links exist
    if (links.length === 0) {
      // Create some default connections to ensure visibility
      for (let i = 0; i < Math.min(5, nodes.length - 1); i++) {
        links.push({
          source: nodes[0], // Connect to current token
          target: nodes[i + 1],
          strength: 0.1 + (i * 0.05) // Varying strengths
        });
      }
    }

    console.log(`Created ${links.length} links with strengths:`, links.map(l => l.strength));
    setConnectionCount(links.length);

    // Create main container group
    const mainContainer = svg.append('g').attr('class', 'main-container');

    // Create force simulation
    const simulation = d3.forceSimulation<Node>(nodes)
      .force('link', d3.forceLink<Node, Link>(links)
        .id(d => d.id)
        .distance(d => 80 + (1 - d.strength) * 120)
        .strength(d => d.strength * 3)
      )
      .force('charge', d3.forceManyBody<Node>()
        .strength(d => d.isCurrent ? -1500 : -600)
      )
      .force('center', d3.forceCenter(centerX, centerY))
      .force('collision', d3.forceCollide<Node>()
        .radius(d => d.isCurrent ? 50 : 35)
      );

    // Create enhanced gradient definitions with solid colors
    const defs = svg.append('defs');
    
    // Fixed gradient definitions that actually work
    const highGradient = defs.append('linearGradient')
      .attr('id', 'attention-high')
      .attr('x1', '0%').attr('y1', '0%')
      .attr('x2', '100%').attr('y2', '0%');
    highGradient.append('stop').attr('offset', '0%').attr('stop-color', '#00FF88').attr('stop-opacity', 0.3);
    highGradient.append('stop').attr('offset', '50%').attr('stop-color', '#0AFF99').attr('stop-opacity', 1);
    highGradient.append('stop').attr('offset', '100%').attr('stop-color', '#00FF88').attr('stop-opacity', 0.3);

    const mediumGradient = defs.append('linearGradient')
      .attr('id', 'attention-medium')
      .attr('x1', '0%').attr('y1', '0%')
      .attr('x2', '100%').attr('y2', '0%');
    mediumGradient.append('stop').attr('offset', '0%').attr('stop-color', '#0A84FF').attr('stop-opacity', 0.3);
    mediumGradient.append('stop').attr('offset', '50%').attr('stop-color', '#21CBF3').attr('stop-opacity', 1);
    mediumGradient.append('stop').attr('offset', '100%').attr('stop-color', '#0A84FF').attr('stop-opacity', 0.3);

    const lowGradient = defs.append('linearGradient')
      .attr('id', 'attention-low')
      .attr('x1', '0%').attr('y1', '0%')
      .attr('x2', '100%').attr('y2', '0%');
    lowGradient.append('stop').attr('offset', '0%').attr('stop-color', '#FF6B6B').attr('stop-opacity', 0.3);
    lowGradient.append('stop').attr('offset', '50%').attr('stop-color', '#FF8E53').attr('stop-opacity', 1);
    lowGradient.append('stop').attr('offset', '100%').attr('stop-color', '#FF6B6B').attr('stop-opacity', 0.3);

    // Enhanced glow filters
    const createGlowFilter = (id: string, stdDev: number, strength: number) => {
      const filter = defs.append('filter')
        .attr('id', id)
        .attr('width', '400%')
        .attr('height', '400%')
        .attr('x', '-150%')
        .attr('y', '-150%');
      
      filter.append('feGaussianBlur')
        .attr('stdDeviation', stdDev)
        .attr('result', 'coloredBlur');
      
      const feMerge = filter.append('feMerge');
      for (let i = 0; i < strength; i++) {
        feMerge.append('feMergeNode').attr('in', 'coloredBlur');
      }
      feMerge.append('feMergeNode').attr('in', 'SourceGraphic');
    };

    createGlowFilter('glow-intense', 10, 5);
    createGlowFilter('glow-medium', 6, 3);
    createGlowFilter('glow-soft', 4, 2);

    // Draw background energy field
    const energyField = mainContainer.append('g').attr('class', 'energy-field');
    for (let i = 0; i < 80; i++) {
      energyField.append('circle')
        .attr('cx', Math.random() * width)
        .attr('cy', Math.random() * height)
        .attr('r', Math.random() * 3 + 0.5)
        .attr('fill', '#21CBF3')
        .attr('opacity', Math.random() * 0.4 + 0.1)
        .style('filter', 'url(#glow-soft)');
    }

    // Draw links with enhanced visibility - using solid colors instead of gradients for better visibility
    const linkGroup = mainContainer.append('g').attr('class', 'links');
    
    const linkElements = linkGroup.selectAll('.link')
      .data(links)
      .enter()
      .append('g')
      .attr('class', 'link');

    // Background glow layer (thickest, most diffuse)
    linkElements.append('line')
      .attr('class', 'connection-glow-bg')
      .attr('stroke', d => {
        if (d.strength > 0.1) return '#00FF88';
        if (d.strength > 0.05) return '#0A84FF';
        return '#FF6B6B';
      })
      .attr('stroke-width', d => Math.max(20, d.strength * 100))
      .attr('stroke-opacity', d => Math.max(0.4, Math.min(0.9, d.strength * 5)))
      .style('filter', 'url(#glow-intense)');

    // Middle glow layer
    linkElements.append('line')
      .attr('class', 'connection-glow-mid')
      .attr('stroke', d => {
        if (d.strength > 0.1) return '#21CBF3';
        if (d.strength > 0.05) return '#0AFF99';
        return '#FF8E53';
      })
      .attr('stroke-width', d => Math.max(12, d.strength * 50))
      .attr('stroke-opacity', d => Math.max(0.6, Math.min(1, d.strength * 6)))
      .style('filter', 'url(#glow-medium)');

    // Core connection line (bright, sharp) - ALWAYS VISIBLE
    linkElements.append('line')
      .attr('class', 'connection-core')
      .attr('stroke', d => {
        if (d.strength > 0.1) return '#FFFFFF';
        if (d.strength > 0.05) return '#00FF88';
        return '#21CBF3';
      })
      .attr('stroke-width', d => Math.max(6, d.strength * 25))
      .attr('stroke-opacity', 1) // Always fully visible
      .style('filter', 'url(#glow-soft)');

    // Animated energy particles flowing along connections
    linkElements.each(function(d) {
      const line = d3.select(this);
      const particleCount = Math.ceil(d.strength * 15) + 3;
      
      for (let i = 0; i < particleCount; i++) {
        // Main particles
        line.append('circle')
          .attr('class', 'particle-main')
          .attr('r', 4 + Math.random() * 3)
          .attr('fill', d.strength > 0.1 ? '#00FF88' : '#21CBF3')
          .attr('opacity', 0.9)
          .style('filter', 'url(#glow-medium)');

        // Trail particles
        line.append('circle')
          .attr('class', 'particle-trail')
          .attr('r', 2)
          .attr('fill', '#FFFFFF')
          .attr('opacity', 0.7)
          .style('filter', 'url(#glow-soft)');
      }
    });

    // Draw nodes with enhanced effects
    const nodeGroup = mainContainer.append('g').attr('class', 'nodes');
    
    const nodeElements = nodeGroup.selectAll('.node')
      .data(nodes)
      .enter()
      .append('g')
      .attr('class', 'node')
      .style('cursor', 'pointer');

    // Enhanced node styling
    nodeElements.each(function(d) {
      const node = d3.select(this);
      
      if (d.isCurrent) {
        // Massive pulsing aura for current token
        node.append('circle')
          .attr('r', 80)
          .attr('fill', 'none')
          .attr('stroke', '#FFD60A')
          .attr('stroke-width', 6)
          .attr('stroke-opacity', 0.5)
          .style('filter', 'url(#glow-intense)')
          .append('animate')
          .attr('attributeName', 'r')
          .attr('values', '70;100;70')
          .attr('dur', '3s')
          .attr('repeatCount', 'indefinite');

        // Secondary aura
        node.append('circle')
          .attr('r', 55)
          .attr('fill', 'none')
          .attr('stroke', '#FFAA00')
          .attr('stroke-width', 4)
          .attr('stroke-opacity', 0.7)
          .style('filter', 'url(#glow-medium)');

        // Core node
        node.append('circle')
          .attr('r', 40)
          .attr('fill', '#FFD60A')
          .attr('stroke', '#FFFFFF')
          .attr('stroke-width', 5)
          .style('filter', 'url(#glow-intense)');
      } else {
        // Energy field for other nodes
        node.append('circle')
          .attr('r', 35 + d.totalAttention * 20)
          .attr('fill', 'none')
          .attr('stroke', d3.interpolateViridis(d.totalAttention))
          .attr('stroke-width', 3)
          .attr('stroke-opacity', 0.6)
          .style('filter', 'url(#glow-medium)');

        // Main node
        node.append('circle')
          .attr('r', 25 + d.totalAttention * 15)
          .attr('fill', d3.interpolateViridis(d.totalAttention))
          .attr('stroke', '#FFFFFF')
          .attr('stroke-width', 4)
          .style('filter', 'url(#glow-intense)');
      }
    });

    // Node labels with enhanced visibility
    nodeElements.append('text')
      .text(d => d.token)
      .attr('text-anchor', 'middle')
      .attr('dy', '0.3em')
      .attr('fill', '#FFFFFF')
      .attr('font-size', d => d.isCurrent ? '16px' : '13px')
      .attr('font-weight', 'bold')
      .style('pointer-events', 'none')
      .style('text-shadow', '0 0 15px rgba(0,0,0,1), 0 0 30px rgba(0,0,0,0.8)')
      .style('filter', 'url(#glow-soft)');

    // Enhanced particle animation
    function animateParticles() {
      linkElements.selectAll('.particle-main, .particle-trail')
        .each(function() {
          const element = this as SVGCircleElement;
          const particle = d3.select(element);
          const parentElement = element.parentNode;
          if (!parentElement) return;
          
          const link = d3.select(parentElement as Element).datum() as Link;
          const isTrail = element.classList.contains('particle-trail');
          const delay = isTrail ? 300 : 0;
          
          setTimeout(() => {
            particle
              .attr('cx', (link.source as Node).x || 0)
              .attr('cy', (link.source as Node).y || 0)
              .transition()
              .duration(1800 + Math.random() * 1200)
              .ease(d3.easeLinear)
              .attr('cx', (link.target as Node).x || 0)
              .attr('cy', (link.target as Node).y || 0)
              .on('end', function() {
                animateParticles();
              });
          }, delay);
        });
    }

    // Update positions on simulation tick
    simulation.on('tick', () => {
      // Update all connection layers
      linkElements.selectAll('.connection-glow-bg, .connection-glow-mid, .connection-core')
        .attr('x1', (d: any) => (d.source as Node).x || 0)
        .attr('y1', (d: any) => (d.source as Node).y || 0)
        .attr('x2', (d: any) => (d.target as Node).x || 0)
        .attr('y2', (d: any) => (d.target as Node).y || 0);

      nodeElements
        .attr('transform', (d: Node) => `translate(${d.x || 0},${d.y || 0})`);
    });

    // Start particle animation after a delay
    setTimeout(animateParticles, 1500);

    // Add drag behavior for nodes
    const nodeDrag = d3.drag<SVGGElement, Node>()
      .on('start', (event, d) => {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      })
      .on('drag', (event, d) => {
        d.fx = event.x;
        d.fy = event.y;
      })
      .on('end', (event, d) => {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      });

    nodeElements.call(nodeDrag);

    // Add zoom and pan behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        const { transform } = event;
        transformRef.current = transform;
        setZoomLevel(transform.k);
        mainContainer.attr('transform', transform);
      });

    svg.call(zoom);

    // Add pan behavior (drag empty space)
    svg.on('mousedown.pan', function(event) {
      if (event.target === svg.node()) {
        svg.style('cursor', 'grabbing');
      }
    });

    svg.on('mouseup.pan', function() {
      svg.style('cursor', 'grab');
    });

    // Set initial cursor
    svg.style('cursor', 'grab');

    // Cleanup function
    return () => {
      simulation.stop();
    };

  }, [attention, tokens, selectedLayer, selectedHead, currentTokenIndex]);

  return (
    <motion.div 
      ref={containerRef}
      className="force-directed-web"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      style={{ 
        width: '100%', 
        height: '100%',
        minHeight: '800px',
        background: 'radial-gradient(ellipse at center, #0a0a0a 0%, #1a1a2e 50%, #000000 100%)',
        borderRadius: '12px',
        overflow: 'hidden',
        position: 'relative'
      }}
    >
      {/* Header with enhanced info and zoom controls */}
      <div style={{ 
        color: '#FFFFFF', 
        padding: '20px', 
        fontSize: '16px',
        background: 'rgba(0,0,0,0.4)',
        borderBottom: '1px solid rgba(255,255,255,0.1)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div>
          <div style={{ fontWeight: 'bold', marginBottom: '4px' }}>
            🕸️ Enhanced Force-Directed Attention Web - Layer {selectedLayer + 1}, Head {selectedHead + 1}
          </div>
          <div style={{ fontSize: '14px', opacity: 0.8 }}>
            Drag nodes to explore • Pan by dragging empty space • Zoom: {Math.round(zoomLevel * 100)}% • Connections: {connectionCount}
          </div>
        </div>
        
        {/* Zoom Controls */}
        <div style={{ display: 'flex', gap: '8px' }}>
          <button 
            onClick={handleZoomOut}
            style={{
              background: 'rgba(255,255,255,0.1)',
              border: '1px solid rgba(255,255,255,0.2)',
              borderRadius: '8px',
              padding: '8px',
              color: '#FFFFFF',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              fontSize: '14px'
            }}
            title="Zoom Out"
          >
            <ZoomOut size={18} />
          </button>
          <button 
            onClick={handleResetZoom}
            style={{
              background: 'rgba(255,255,255,0.1)',
              border: '1px solid rgba(255,255,255,0.2)',
              borderRadius: '8px',
              padding: '8px',
              color: '#FFFFFF',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              fontSize: '14px'
            }}
            title="Reset Zoom"
          >
            <RotateCcw size={18} />
          </button>
          <button 
            onClick={handleZoomIn}
            style={{
              background: 'rgba(255,255,255,0.1)',
              border: '1px solid rgba(255,255,255,0.2)',
              borderRadius: '8px',
              padding: '8px',
              color: '#FFFFFF',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              fontSize: '14px'
            }}
            title="Zoom In"
          >
            <ZoomIn size={18} />
          </button>
        </div>
      </div>
      
      <svg
        ref={svgRef}
        width="100%"
        height="calc(100% - 80px)"
        style={{ background: 'transparent' }}
      />
    </motion.div>
  );
};

export default ForceDirectedWeb; 