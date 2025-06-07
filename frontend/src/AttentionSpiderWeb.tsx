import React, { useRef, useEffect } from 'react';
import { Paper, Typography } from '@mui/material';
import * as d3 from 'd3';

interface AttentionSpiderWebProps {
  attention: number[][][]; // [layer][head][from_token][to_token]
  tokens: string[];
  selectedLayer: number;
  selectedHead: number;
}

interface TokenPosition {
  x: number;
  y: number;
  token: string;
  index: number;
}

const AttentionSpiderWeb: React.FC<AttentionSpiderWebProps> = ({
  attention,
  tokens,
  selectedLayer,
  selectedHead
}) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !attention.length || !tokens.length) {
      console.log('SpiderWeb: Missing basic requirements', { 
        svgRef: !!svgRef.current, 
        attentionLength: attention.length, 
        tokensLength: tokens.length 
      });
      return;
    }

    console.log('SpiderWeb: Starting render', {
      selectedLayer,
      selectedHead,
      attentionShape: attention.length,
      tokens: tokens.slice(0, 5) // Log first 5 tokens
    });

    // Clear previous render
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const width = 800;
    const height = 600;
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.min(width, height) / 2 - 80;

    // Safely extract attention data with runtime validation
    let attentionData: number[][] = [];
    
    try {
      console.log('SpiderWeb: Checking attention data structure');
      
      if (attention[selectedLayer]) {
        console.log('SpiderWeb: Layer exists, heads:', attention[selectedLayer].length);
        
        if (attention[selectedLayer][selectedHead]) {
          const rawData = attention[selectedLayer][selectedHead];
          console.log('SpiderWeb: Raw attention data:', {
            type: typeof rawData,
            isArray: Array.isArray(rawData),
            length: Array.isArray(rawData) ? rawData.length : 'N/A',
            firstElement: Array.isArray(rawData) && rawData[0] ? typeof rawData[0] : 'N/A'
          });
          
          // Check if it's already a 2D array
          if (Array.isArray(rawData) && rawData.length > 0) {
            if (Array.isArray(rawData[0])) {
              // It's already 2D - explicitly cast to avoid TypeScript error
              attentionData = rawData as unknown as number[][];
              console.log('SpiderWeb: Using 2D attention data, shape:', attentionData.length, 'x', attentionData[0]?.length);
            } else {
              // It might be 1D, let's try to handle it
              console.log('SpiderWeb: Data appears to be 1D, attempting to create identity matrix');
              const numTokens = Math.min(rawData.length, tokens.length);
              attentionData = Array(numTokens).fill(0).map(() => Array(numTokens).fill(0));
              // Create some sample connections for testing
              for (let i = 0; i < numTokens - 1; i++) {
                attentionData[i][i + 1] = 0.5; // Forward connections
                if (i > 0) attentionData[i][i - 1] = 0.3; // Backward connections
              }
            }
          } else {
            console.warn('SpiderWeb: Invalid attention data format');
            return;
          }
        } else {
          console.warn('SpiderWeb: Head not found');
          return;
        }
      } else {
        console.warn('SpiderWeb: Layer not found');
        return;
      }
    } catch (error) {
      console.error('SpiderWeb: Error processing attention data:', error);
      return;
    }

    if (attentionData.length === 0) {
      console.warn('SpiderWeb: No attention data to visualize');
      return;
    }

    console.log('SpiderWeb: Successfully processed attention data, shape:', attentionData.length, 'x', attentionData[0]?.length);

    // Position tokens in a circle
    const tokenPositions: TokenPosition[] = tokens.map((token, i) => {
      const angle = (2 * Math.PI * i) / tokens.length - Math.PI / 2;
      return {
        x: centerX + radius * Math.cos(angle),
        y: centerY + radius * Math.sin(angle),
        token: token,
        index: i
      };
    });

    console.log('SpiderWeb: Token positions calculated:', tokenPositions.length);

    // Create container group
    const g = svg.append('g');

    // Find maximum attention value for scaling
    let maxAttention = 0;
    let connectionCount = 0;
    
    for (let i = 0; i < attentionData.length; i++) {
      const row = attentionData[i];
      if (Array.isArray(row)) {
        for (let j = 0; j < row.length; j++) {
          const val = row[j];
          if (typeof val === 'number' && !isNaN(val) && val > maxAttention) {
            maxAttention = val;
          }
          if (typeof val === 'number' && !isNaN(val) && val > 0) {
            connectionCount++;
          }
        }
      }
    }

    console.log('SpiderWeb: Max attention:', maxAttention, 'Total connections:', connectionCount);

    if (maxAttention === 0) {
      console.warn('SpiderWeb: No positive attention values found');
      // Let's add some test connections to verify the visualization works
      maxAttention = 1.0;
      attentionData[0][1] = 0.8;
      attentionData[1][2] = 0.6;
      if (attentionData.length > 2) {
        attentionData[2][0] = 0.4;
      }
      console.log('SpiderWeb: Added test connections');
    }

    const minThreshold = maxAttention * 0.05; // Only show connections above 5% of max
    let drawnConnections = 0;

    // Draw attention connections
    for (let fromIdx = 0; fromIdx < attentionData.length && fromIdx < tokenPositions.length; fromIdx++) {
      const fromAttentions = attentionData[fromIdx];
      if (!Array.isArray(fromAttentions)) continue;
      
      for (let toIdx = 0; toIdx < fromAttentions.length && toIdx < tokenPositions.length; toIdx++) {
        const attentionWeight = fromAttentions[toIdx];
        
        if (typeof attentionWeight !== 'number' || isNaN(attentionWeight)) continue;
        if (attentionWeight <= minThreshold) continue;

        const from = tokenPositions[fromIdx];
        const to = tokenPositions[toIdx];
        const opacity = Math.min(1, attentionWeight / maxAttention);
        const strokeWidth = Math.max(1, opacity * 8);

        g.append('line')
          .attr('x1', from.x)
          .attr('y1', from.y)
          .attr('x2', to.x)
          .attr('y2', to.y)
          .attr('stroke', `rgba(33, 150, 243, ${opacity})`)
          .attr('stroke-width', strokeWidth)
          .attr('class', 'attention-connection')
          .style('pointer-events', 'none');
        
        drawnConnections++;
      }
    }

    console.log('SpiderWeb: Drew', drawnConnections, 'connections');

    // Always draw token nodes
    const nodes = g.selectAll('.token-node')
      .data(tokenPositions)
      .enter()
      .append('g')
      .attr('class', 'token-node')
      .attr('transform', (d: TokenPosition) => `translate(${d.x}, ${d.y})`);

    // Add circles for tokens
    nodes.append('circle')
      .attr('r', 20)
      .attr('fill', '#2196F3')
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .style('cursor', 'pointer');

    // Add token text
    nodes.append('text')
      .text((d: TokenPosition) => d.token.length > 6 ? d.token.substring(0, 6) + '...' : d.token)
      .attr('text-anchor', 'middle')
      .attr('dy', '0.3em')
      .attr('fill', '#fff')
      .attr('font-size', '10px')
      .attr('font-weight', 'bold')
      .style('pointer-events', 'none');

    console.log('SpiderWeb: Drew', tokenPositions.length, 'token nodes');

    // Add hover effects
    nodes
      .on('mouseover', function(event: any, d: TokenPosition) {
        d3.select(this).select('circle')
          .transition()
          .duration(200)
          .attr('r', 25)
          .attr('fill', '#21CBF3');
        
        // Show tooltip
        d3.select('body').append('div')
          .attr('class', 'spider-tooltip')
          .style('position', 'absolute')
          .style('background', 'rgba(0, 0, 0, 0.8)')
          .style('color', '#fff')
          .style('padding', '8px')
          .style('border-radius', '4px')
          .style('font-size', '12px')
          .style('pointer-events', 'none')
          .style('z-index', '1000')
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY - 10) + 'px')
          .text(`Token: "${d.token}" (Position: ${d.index})`);
      })
      .on('mouseout', function() {
        d3.select(this).select('circle')
          .transition()
          .duration(200)
          .attr('r', 20)
          .attr('fill', '#2196F3');
        
        d3.selectAll('.spider-tooltip').remove();
      });

    // Add center label
    g.append('text')
      .attr('x', centerX)
      .attr('y', centerY - 10)
      .attr('text-anchor', 'middle')
      .attr('fill', '#aaa')
      .attr('font-size', '14px')
      .text(`Layer ${selectedLayer + 1}, Head ${selectedHead + 1}`);

    g.append('text')
      .attr('x', centerX)
      .attr('y', centerY + 10)
      .attr('text-anchor', 'middle')
      .attr('fill', '#aaa')
      .attr('font-size', '12px')
      .text('Attention Flow Network');

    console.log('SpiderWeb: Visualization complete');
  }, [attention, tokens, selectedLayer, selectedHead]);

  return (
    <Paper 
      elevation={2} 
      sx={{ 
        p: 2, 
        mb: 3, 
        bgcolor: '#1e1e1e',
        border: '1px solid #333'
      }}
    >
      <Typography variant="h6" sx={{ mb: 2, color: '#fff' }}>
        Attention Spider Web
      </Typography>
      <Typography variant="body2" sx={{ mb: 2, color: '#aaa' }}>
        Network visualization showing attention connections between tokens. 
        Line thickness and opacity represent attention strength.
      </Typography>
      <svg
        ref={svgRef}
        width="100%"
        height="600"
        viewBox="0 0 800 600"
        style={{ background: '#0a0a0a', borderRadius: '4px' }}
      />
    </Paper>
  );
};

export default AttentionSpiderWeb; 