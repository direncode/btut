'use client';

import { useEffect, useRef, useState } from 'react';
import { Maximize2, Minimize2, RotateCw, ZoomIn, ZoomOut } from 'lucide-react';

interface Node {
  id: number;
  x: number;
  y: number;
  z: number;
  degree: number;
  strategy: 'A' | 'B';
}

interface Edge {
  source: number;
  target: number;
}

interface NetworkVisualization3DProps {
  nodeCount?: number;
  edges?: Edge[];
  strategies?: ('A' | 'B')[];
  autoRotate?: boolean;
}

export default function NetworkVisualization3D({
  nodeCount = 100,
  edges = [],
  strategies = [],
  autoRotate = true,
}: NetworkVisualization3DProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [rotation, setRotation] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1.0);
  const [nodes, setNodes] = useState<Node[]>([]);
  const animationFrameRef = useRef<number>();
  const isDragging = useRef(false);
  const lastMousePos = useRef({ x: 0, y: 0 });

  // Generate network layout using force-directed algorithm
  useEffect(() => {
    const generateNodes = (): Node[] => {
      const newNodes: Node[] = [];
      const radius = 200;

      for (let i = 0; i < nodeCount; i++) {
        // Spherical distribution
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos(2 * Math.random() - 1);
        const r = radius * Math.cbrt(Math.random());

        newNodes.push({
          id: i,
          x: r * Math.sin(phi) * Math.cos(theta),
          y: r * Math.sin(phi) * Math.sin(theta),
          z: r * Math.cos(phi),
          degree: 0,
          strategy: strategies[i] || (Math.random() > 0.5 ? 'A' : 'B'),
        });
      }

      // Calculate node degrees
      edges.forEach(edge => {
        if (newNodes[edge.source]) newNodes[edge.source].degree++;
        if (newNodes[edge.target]) newNodes[edge.target].degree++;
      });

      return newNodes;
    };

    setNodes(generateNodes());
  }, [nodeCount, edges, strategies]);

  // Render 3D visualization
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2;
    const centerY = height / 2;

    const render = () => {
      ctx.clearRect(0, 0, width, height);

      // Sort nodes by z-coordinate for proper depth rendering
      const sortedNodes = [...nodes].sort((a, b) => b.z - a.z);

      // Apply rotation and zoom
      const rotateX = (node: Node) => {
        const y = node.y * Math.cos(rotation.x) - node.z * Math.sin(rotation.x);
        const z = node.y * Math.sin(rotation.x) + node.z * Math.cos(rotation.x);
        return { ...node, y, z };
      };

      const rotateY = (node: Node) => {
        const x = node.x * Math.cos(rotation.y) + node.z * Math.sin(rotation.y);
        const z = -node.x * Math.sin(rotation.y) + node.z * Math.cos(rotation.y);
        return { ...node, x, z };
      };

      const project = (node: Node) => {
        const perspective = 600;
        const scale = perspective / (perspective + node.z);
        return {
          x: centerX + node.x * scale * zoom,
          y: centerY + node.y * scale * zoom,
          scale,
        };
      };

      // Render edges
      ctx.strokeStyle = 'rgba(100, 100, 100, 0.2)';
      ctx.lineWidth = 1;

      edges.forEach(edge => {
        const sourceNode = nodes[edge.source];
        const targetNode = nodes[edge.target];
        if (!sourceNode || !targetNode) return;

        const rotatedSource = rotateY(rotateX(sourceNode));
        const rotatedTarget = rotateY(rotateX(targetNode));

        const projectedSource = project(rotatedSource);
        const projectedTarget = project(rotatedTarget);

        ctx.beginPath();
        ctx.moveTo(projectedSource.x, projectedSource.y);
        ctx.lineTo(projectedTarget.x, projectedTarget.y);
        ctx.stroke();
      });

      // Render nodes
      sortedNodes.forEach(node => {
        const rotated = rotateY(rotateX(node));
        const projected = project(rotated);

        // Node size based on degree (hub detection)
        const baseSize = 3;
        const hubSize = Math.min(baseSize + node.degree * 0.5, 12);
        const size = hubSize * projected.scale;

        // Color based on strategy
        const colorA = 'rgba(59, 130, 246, ';  // Blue for strategy A
        const colorB = 'rgba(239, 68, 68, ';   // Red for strategy B
        const alpha = Math.max(0.3, projected.scale);
        ctx.fillStyle = node.strategy === 'A'
          ? colorA + alpha + ')'
          : colorB + alpha + ')';

        // Draw node
        ctx.beginPath();
        ctx.arc(projected.x, projected.y, size, 0, Math.PI * 2);
        ctx.fill();

        // Highlight hubs
        if (node.degree > 5) {
          ctx.strokeStyle = 'rgba(255, 255, 255, ' + alpha + ')';
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      });

      // Auto-rotate
      if (autoRotate) {
        setRotation(prev => ({
          x: prev.x + 0.002,
          y: prev.y + 0.005,
        }));
      }

      animationFrameRef.current = requestAnimationFrame(render);
    };

    render();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [nodes, rotation, zoom, autoRotate, edges]);

  // Mouse controls
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const handleMouseDown = (e: MouseEvent) => {
      isDragging.current = true;
      lastMousePos.current = { x: e.clientX, y: e.clientY };
    };

    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging.current) return;

      const deltaX = e.clientX - lastMousePos.current.x;
      const deltaY = e.clientY - lastMousePos.current.y;

      setRotation(prev => ({
        x: prev.x + deltaY * 0.01,
        y: prev.y + deltaX * 0.01,
      }));

      lastMousePos.current = { x: e.clientX, y: e.clientY };
    };

    const handleMouseUp = () => {
      isDragging.current = false;
    };

    canvas.addEventListener('mousedown', handleMouseDown);
    canvas.addEventListener('mousemove', handleMouseMove);
    canvas.addEventListener('mouseup', handleMouseUp);
    canvas.addEventListener('mouseleave', handleMouseUp);

    return () => {
      canvas.removeEventListener('mousedown', handleMouseDown);
      canvas.removeEventListener('mousemove', handleMouseMove);
      canvas.removeEventListener('mouseup', handleMouseUp);
      canvas.removeEventListener('mouseleave', handleMouseUp);
    };
  }, []);

  const toggleFullscreen = () => {
    setIsFullscreen(!isFullscreen);
  };

  const resetRotation = () => {
    setRotation({ x: 0, y: 0 });
    setZoom(1.0);
  };

  const zoomIn = () => {
    setZoom(prev => Math.min(prev + 0.2, 3.0));
  };

  const zoomOut = () => {
    setZoom(prev => Math.max(prev - 0.2, 0.3));
  };

  return (
    <div className={`relative ${isFullscreen ? 'fixed inset-0 z-50 bg-gray-900' : ''}`}>
      <div className="relative bg-gray-100 dark:bg-gray-900 rounded-lg overflow-hidden">
        <canvas
          ref={canvasRef}
          width={isFullscreen ? window.innerWidth : 800}
          height={isFullscreen ? window.innerHeight : 600}
          className="w-full h-full cursor-grab active:cursor-grabbing"
          role="img"
          aria-label="3D network visualization showing agent connections and strategies"
        />

        {/* Controls Overlay */}
        <div className="absolute top-4 right-4 flex flex-col gap-2">
          <button
            onClick={zoomIn}
            className="p-2 bg-white dark:bg-gray-800 rounded-lg shadow-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            aria-label="Zoom in"
          >
            <ZoomIn className="h-5 w-5 text-gray-700 dark:text-gray-300" />
          </button>

          <button
            onClick={zoomOut}
            className="p-2 bg-white dark:bg-gray-800 rounded-lg shadow-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            aria-label="Zoom out"
          >
            <ZoomOut className="h-5 w-5 text-gray-700 dark:text-gray-300" />
          </button>

          <button
            onClick={resetRotation}
            className="p-2 bg-white dark:bg-gray-800 rounded-lg shadow-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            aria-label="Reset rotation"
          >
            <RotateCw className="h-5 w-5 text-gray-700 dark:text-gray-300" />
          </button>

          <button
            onClick={toggleFullscreen}
            className="p-2 bg-white dark:bg-gray-800 rounded-lg shadow-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            aria-label={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
          >
            {isFullscreen ? (
              <Minimize2 className="h-5 w-5 text-gray-700 dark:text-gray-300" />
            ) : (
              <Maximize2 className="h-5 w-5 text-gray-700 dark:text-gray-300" />
            )}
          </button>
        </div>

        {/* Legend */}
        <div className="absolute bottom-4 left-4 bg-white dark:bg-gray-800 rounded-lg shadow-lg p-3">
          <div className="text-xs font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Strategy
          </div>
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-blue-500"></div>
              <span className="text-xs text-gray-600 dark:text-gray-400">
                Strategy A (Cooperate)
              </span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-red-500"></div>
              <span className="text-xs text-gray-600 dark:text-gray-400">
                Strategy B (Defect)
              </span>
            </div>
          </div>
          <div className="text-xs text-gray-500 dark:text-gray-500 mt-2">
            Size indicates hub centrality
          </div>
        </div>

        {/* Instructions */}
        <div className="absolute top-4 left-4 bg-white dark:bg-gray-800 rounded-lg shadow-lg p-3">
          <div className="text-xs text-gray-600 dark:text-gray-400">
            Drag to rotate • Scroll to zoom
          </div>
        </div>
      </div>
    </div>
  );
}
