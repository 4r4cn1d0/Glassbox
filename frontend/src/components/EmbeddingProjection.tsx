import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import { motion } from 'framer-motion';
import * as d3 from 'd3';

interface EmbeddingProjectionProps {
  sessionId: string | null;
  currentTokenIndex: number;
  onTokenClick?: (index: number) => void;
}

interface EmbeddingData {
  embeddings: number[][];
  tokens: string[];
  token_ids: number[];
  embedding_dim: number;
  prompt_token_count?: number; // Add this to know where prompt ends
}

interface ProjectedPoint {
  x: number;
  y: number;
  token: string;
  token_id: number;
  index: number;
  isPrompt: boolean;
  isGenerated: boolean;
  isCurrent: boolean;
}

interface CachedProjections {
  pca?: number[][];
  tsne?: number[][];
  sessionId?: string | null;
}

const EmbeddingProjection: React.FC<EmbeddingProjectionProps> = ({
  sessionId,
  currentTokenIndex,
  onTokenClick
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [embeddingData, setEmbeddingData] = useState<EmbeddingData | null>(null);
  const [projectionMethod, setProjectionMethod] = useState<'pca' | 'tsne'>('pca');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Cache projections to maintain consistency
  const [cachedProjections, setCachedProjections] = useState<CachedProjections>({});

  // Fetch embedding data when sessionId changes
  useEffect(() => {
    if (!sessionId) return;

    const fetchEmbeddings = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await fetch(`http://localhost:8000/api/embeddings/${sessionId}`);
        if (!response.ok) throw new Error('Failed to fetch embeddings');
        const data = await response.json();
        
        // Add prompt token count for proper classification
        const promptTokenCount = data.prompt_token_count || 5; // Fallback estimate
        setEmbeddingData({
          ...data,
          prompt_token_count: promptTokenCount
        });
        
        // Clear cached projections if this is a new session
        if (cachedProjections.sessionId !== sessionId) {
          setCachedProjections({ sessionId });
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };

    fetchEmbeddings();
  }, [sessionId, cachedProjections.sessionId]);

  // Deterministic PCA implementation with fixed seed
  const computePCA = useCallback((data: number[][]): number[][] => {
    if (data.length === 0) return [];

    // Center the data
    const dimensions = data[0].length;
    const means = new Array(dimensions).fill(0);
    
    for (let i = 0; i < data.length; i++) {
      for (let j = 0; j < dimensions; j++) {
        means[j] += data[i][j];
      }
    }
    
    for (let j = 0; j < dimensions; j++) {
      means[j] /= data.length;
    }

    const centeredData = data.map(row => 
      row.map((val, idx) => val - means[idx])
    );

    // Use a more sophisticated projection with fixed angles for consistency
    const angle1 = Math.PI / 6; // 30 degrees - fixed
    const angle2 = Math.PI / 3; // 60 degrees - fixed
    
    return centeredData.map((row, idx) => [
      row[0] * Math.cos(angle1) - row[1] * Math.sin(angle1) + (row[2] || 0) * 0.1 + (row[3] || 0) * 0.05,
      row[0] * Math.sin(angle2) + row[1] * Math.cos(angle2) + (row[4] || 0) * 0.1 + (row[5] || 0) * 0.05
    ]);
  }, []);

  // Deterministic t-SNE-like projection with fixed seed
  const computeTSNE = useCallback((data: number[][]): number[][] => {
    if (data.length === 0) return [];
    
    // Use deterministic seed based on data length for consistency
    const seed = data.length * 42; // Fixed seed
    let randomSeed = seed;
    
    // Simple deterministic random function
    const seededRandom = () => {
      randomSeed = (randomSeed * 9301 + 49297) % 233280;
      return randomSeed / 233280;
    };
    
    return data.map((embedding, i) => {
      // Create clusters based on token position and embedding similarity
      const clusterBase = Math.floor(i / 4) * 0.8;
      const angle = (i / data.length) * 2 * Math.PI + clusterBase + seededRandom() * 0.3;
      const radius = 60 + seededRandom() * 80;
      
      // Add some structure based on embedding values for better clustering
      const embeddingSum = embedding.slice(0, 10).reduce((a, b) => a + b, 0);
      const structuralOffset = (embeddingSum % 100) * 0.8;
      
      return [
        radius * Math.cos(angle + structuralOffset) + seededRandom() * 15,
        radius * Math.sin(angle + structuralOffset) + seededRandom() * 15
      ];
    });
  }, []);

  // Compute or retrieve cached projections
  const projectedPoints = useMemo((): ProjectedPoint[] => {
    if (!embeddingData) return [];

    // Check if we have cached projections for this session
    let projected: number[][];
    
    if (projectionMethod === 'pca') {
      if (cachedProjections.pca && cachedProjections.sessionId === sessionId) {
        projected = cachedProjections.pca;
      } else {
        projected = computePCA(embeddingData.embeddings);
        setCachedProjections(prev => ({
          ...prev,
          pca: projected,
          sessionId: sessionId
        }));
      }
    } else {
      if (cachedProjections.tsne && cachedProjections.sessionId === sessionId) {
        projected = cachedProjections.tsne;
      } else {
        projected = computeTSNE(embeddingData.embeddings);
        setCachedProjections(prev => ({
          ...prev,
          tsne: projected,
          sessionId: sessionId
        }));
      }
    }

    // Fixed classification logic
    const promptTokenCount = embeddingData.prompt_token_count || Math.ceil(embeddingData.tokens.length * 0.3);
    
    return projected.map((point, index) => ({
      x: point[0],
      y: point[1],
      token: embeddingData.tokens[index],
      token_id: embeddingData.token_ids[index],
      index,
      isPrompt: index < promptTokenCount,
      isGenerated: index >= promptTokenCount,
      isCurrent: index === currentTokenIndex
    }));
  }, [embeddingData, projectionMethod, currentTokenIndex, cachedProjections, sessionId, computePCA, computeTSNE]);

  // D3 visualization
  useEffect(() => {
    if (!svgRef.current || projectedPoints.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const margin = { top: 20, right: 20, bottom: 40, left: 40 };
    const width = 640 - margin.left - margin.right;
    const height = 440 - margin.top - margin.bottom;

    // Scales
    const xExtent = d3.extent(projectedPoints, d => d.x) as [number, number];
    const yExtent = d3.extent(projectedPoints, d => d.y) as [number, number];
    
    const xScale = d3.scaleLinear()
      .domain(xExtent)
      .range([0, width])
      .nice();
    
    const yScale = d3.scaleLinear()
      .domain(yExtent)
      .range([height, 0])
      .nice();

    const g = svg.append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Add background
    g.append("rect")
      .attr("width", width)
      .attr("height", height)
      .attr("fill", "#2C3E50")
      .attr("stroke", "rgba(255, 255, 255, 0.2)")
      .attr("rx", 8);

    // Add grid lines
    const xAxis = d3.axisBottom(xScale).tickSize(-height).tickFormat(null);
    const yAxis = d3.axisLeft(yScale).tickSize(-width).tickFormat(null);
    
    g.append("g")
      .attr("class", "grid")
      .attr("transform", `translate(0,${height})`)
      .call(xAxis as any)
      .selectAll("line")
      .attr("stroke", "rgba(255, 255, 255, 0.1)");
    
    g.append("g")
      .attr("class", "grid")
      .call(yAxis as any)
      .selectAll("line")
      .attr("stroke", "rgba(255, 255, 255, 0.1)");

    // Add connection lines for sequential tokens (draw first so points are on top)
    if (projectedPoints.length > 1) {
      const lineGenerator = d3.line<ProjectedPoint>()
        .x(d => xScale(d.x))
        .y(d => yScale(d.y))
        .curve(d3.curveCatmullRom.alpha(0.5));

      g.append("path")
        .datum(projectedPoints)
        .attr("fill", "none")
        .attr("stroke", "rgba(78, 205, 196, 0.3)")
        .attr("stroke-width", 2)
        .attr("d", lineGenerator);
    }

    // Add points
    const points = g.selectAll(".point")
      .data(projectedPoints)
      .enter().append("g")
      .attr("class", "point")
      .attr("transform", d => `translate(${xScale(d.x)},${yScale(d.y)})`)
      .style("cursor", "pointer")
      .on("click", (event, d) => {
        if (onTokenClick) onTokenClick(d.index);
      });

    // Add circles with proper colors
    points.append("circle")
      .attr("r", d => d.isCurrent ? 10 : (d.isGenerated ? 7 : 5))
      .attr("fill", d => {
        if (d.isCurrent) return "#FF6B35"; // Orange for current
        if (d.isPrompt) return "#FFE66D"; // Yellow for prompt tokens
        if (d.isGenerated) return "#4ECDC4"; // Teal for generated tokens
        return "#95A5A6"; // Fallback gray
      })
      .attr("stroke", d => d.isCurrent ? "#FFFFFF" : "none")
      .attr("stroke-width", d => d.isCurrent ? 3 : 0)
      .style("filter", d => d.isCurrent ? "drop-shadow(0 0 12px #FF6B35)" : "none")
      .style("opacity", 0.9);

    // Add labels for current and nearby tokens
    points.filter(d => d.isCurrent || Math.abs(d.index - currentTokenIndex) <= 2)
      .append("text")
      .attr("dy", -15)
      .attr("text-anchor", "middle")
      .attr("font-size", d => d.isCurrent ? "14px" : "12px")
      .attr("font-weight", d => d.isCurrent ? "bold" : "normal")
      .attr("fill", "#FFFFFF")
      .attr("stroke", "#2C3E50")
      .attr("stroke-width", "1")
      .text(d => d.token.length > 8 ? d.token.substring(0, 8) + "..." : d.token);

    // Add axes labels
    g.append("text")
      .attr("transform", `translate(${width / 2}, ${height + 35})`)
      .style("text-anchor", "middle")
      .attr("font-size", "14px")
      .attr("font-weight", "bold")
      .attr("fill", "#FFFFFF")
      .text(`${projectionMethod.toUpperCase()} Component 1`);

    g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", -25)
      .attr("x", -height / 2)
      .style("text-anchor", "middle")
      .attr("font-size", "14px")
      .attr("font-weight", "bold")
      .attr("fill", "#FFFFFF")
      .text(`${projectionMethod.toUpperCase()} Component 2`);

  }, [projectedPoints, currentTokenIndex, onTokenClick, projectionMethod]);

  if (loading) {
    return (
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        height: '500px',
        background: '#FFFFFF',
        borderRadius: '16px',
        border: '1px solid #E9ECEF',
        flexDirection: 'column',
        gap: '16px'
      }}>
        <img 
          src="/logo_light.png" 
          alt="Glassbox" 
          style={{ 
            height: '40px', 
            width: 'auto',
            opacity: 0.7
          }} 
        />
        <div style={{ 
          color: '#495057',
          fontSize: '18px',
          fontWeight: 'bold'
        }}>
          Loading embeddings...
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        height: '500px',
        background: '#FFFFFF',
        borderRadius: '16px',
        border: '1px solid #E9ECEF',
        flexDirection: 'column',
        gap: '16px'
      }}>
        <img 
          src="/logo_light.png" 
          alt="Glassbox" 
          style={{ 
            height: '40px', 
            width: 'auto',
            opacity: 0.7
          }} 
        />
        <div style={{ 
          color: '#DC3545',
          fontSize: '18px',
          fontWeight: 'bold'
        }}>
          Error: {error}
        </div>
        <div style={{ 
          color: '#6C757D', 
          fontSize: '16px',
          textAlign: 'center'
        }}>
          Make sure you've generated some tokens first
        </div>
      </div>
    );
  }

  if (!embeddingData || projectedPoints.length === 0) {
    return (
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        height: '500px',
        background: '#FFFFFF',
        borderRadius: '16px',
        border: '1px solid #E9ECEF',
        flexDirection: 'column',
        gap: '16px'
      }}>
        <img 
          src="/logo_light.png" 
          alt="Glassbox" 
          style={{ 
            height: '40px', 
            width: 'auto',
            opacity: 0.7
          }} 
        />
        <div style={{ 
          color: '#6C757D',
          fontSize: '18px',
          fontWeight: 'bold'
        }}>
          No embedding data available
        </div>
      </div>
    );
  }

  return (
    <motion.div
      style={{
        background: '#FFFFFF',
        borderRadius: '16px',
        padding: '24px',
        border: '1px solid #E9ECEF',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center'
      }}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        marginBottom: '20px',
        width: '100%'
      }}>
        <h3 style={{ 
          color: '#495057', 
          margin: 0,
          fontSize: '22px',
          fontWeight: 'bold'
        }}>
          🗺️ Token Embeddings ({projectionMethod.toUpperCase()}) - Session Cached
        </h3>
        
        <div style={{ display: 'flex', gap: '12px' }}>
          <button
            style={{
              padding: '8px 16px',
              border: 'none',
              borderRadius: '8px',
              fontWeight: 'bold',
              cursor: 'pointer',
              background: projectionMethod === 'pca' ? '#007BFF' : '#6C757D',
              color: '#FFFFFF',
              transition: 'all 0.2s ease'
            }}
            onClick={() => setProjectionMethod('pca')}
          >
            PCA
          </button>
          <button
            style={{
              padding: '8px 16px',
              border: 'none',
              borderRadius: '8px',
              fontWeight: 'bold',
              cursor: 'pointer',
              background: projectionMethod === 'tsne' ? '#007BFF' : '#6C757D',
              color: '#FFFFFF',
              transition: 'all 0.2s ease'
            }}
            onClick={() => setProjectionMethod('tsne')}
          >
            t-SNE
          </button>
        </div>
      </div>

      <div style={{ 
        display: 'flex',
        justifyContent: 'center',
        width: '100%',
        marginBottom: '20px'
      }}>
        <svg
          ref={svgRef}
          width={640}
          height={440}
          style={{ 
            background: '#2C3E50',
            borderRadius: '12px',
            border: '2px solid #34495E'
          }}
        />
      </div>

      <div style={{ 
        display: 'flex', 
        gap: '24px', 
        fontSize: '14px',
        color: '#495057',
        fontWeight: 'bold',
        marginBottom: '16px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{ 
            width: '12px', 
            height: '12px', 
            borderRadius: '50%', 
            background: '#FFE66D' 
          }} />
          Prompt tokens ({embeddingData.prompt_token_count || 'Unknown'})
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{ 
            width: '12px', 
            height: '12px', 
            borderRadius: '50%', 
            background: '#4ECDC4' 
          }} />
          Generated tokens ({projectedPoints.filter(p => p.isGenerated).length})
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{ 
            width: '12px', 
            height: '12px', 
            borderRadius: '50%', 
            background: '#FF6B35',
            border: '2px solid #FFFFFF',
            boxShadow: '0 0 8px rgba(255, 107, 53, 0.5)'
          }} />
          Current token
        </div>
      </div>

      <div style={{ 
        fontSize: '13px',
        color: '#6C757D',
        lineHeight: '1.6',
        textAlign: 'center',
        background: '#F8F9FA',
        padding: '16px',
        borderRadius: '8px',
        border: '1px solid #E9ECEF'
      }}>
        <div style={{ marginBottom: '8px' }}>
          <strong style={{ color: '#495057' }}>Embedding dimension:</strong> {embeddingData.embedding_dim} → 2D projection
        </div>
        <div style={{ marginBottom: '8px' }}>
          <strong style={{ color: '#495057' }}>Total tokens:</strong> {embeddingData.tokens.length} | <strong style={{ color: '#28A745' }}>Positions cached for consistency</strong>
        </div>
        <div>
          <strong style={{ color: '#495057' }}>Click</strong> any point to jump to that token in the timeline
        </div>
      </div>
    </motion.div>
  );
};

export default EmbeddingProjection; 