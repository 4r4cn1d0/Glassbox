import React from 'react';

interface SkeletonLoaderProps {
  type?: 'text' | 'title' | 'button' | 'card' | 'visualization';
  count?: number;
  className?: string;
}

const SkeletonLoader: React.FC<SkeletonLoaderProps> = ({ 
  type = 'text', 
  count = 1, 
  className = '' 
}) => {
  const renderSkeleton = () => {
    switch (type) {
      case 'title':
        return <div className={`skeleton skeleton-title ${className}`} />;
      case 'button':
        return <div className={`skeleton skeleton-button ${className}`} />;
      case 'card':
        return <div className={`skeleton skeleton-card ${className}`} />;
      case 'visualization':
        return (
          <div className="viz-panel">
            <div className="viz-header">
              <div className="skeleton skeleton-title" style={{ width: '40%' }} />
            </div>
            <div className="viz-content">
              <div className="skeleton skeleton-card" />
            </div>
          </div>
        );
      default:
        return <div className={`skeleton skeleton-text ${className}`} />;
    }
  };

  return (
    <>
      {Array.from({ length: count }, (_, index) => (
        <div key={index}>
          {renderSkeleton()}
        </div>
      ))}
    </>
  );
};

export default SkeletonLoader; 