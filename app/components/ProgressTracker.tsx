'use client';

import { useEffect, useState } from 'react';
import { Activity, CheckCircle, Clock, Zap } from 'lucide-react';

interface ProgressTrackerProps {
  simulationId?: string;
  totalIterations?: number;
  onComplete?: (results: any) => void;
}

interface ProgressState {
  currentIteration: number;
  totalIterations: number;
  fractionA: number;
  converged: boolean;
  status: 'idle' | 'running' | 'completed' | 'error';
  elapsedTime: number;
  estimatedTimeRemaining: number;
}

export default function ProgressTracker({
  simulationId,
  totalIterations = 100,
  onComplete
}: ProgressTrackerProps) {
  const [progress, setProgress] = useState<ProgressState>({
    currentIteration: 0,
    totalIterations,
    fractionA: 0.5,
    converged: false,
    status: 'idle',
    elapsedTime: 0,
    estimatedTimeRemaining: 0,
  });

  useEffect(() => {
    if (!simulationId) return;

    // WebSocket connection for real-time updates
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = process.env.NEXT_PUBLIC_WS_URL ||
                  `${protocol}//${window.location.host}/ws/simulation/${simulationId}`;

    let ws: WebSocket;
    let reconnectAttempts = 0;
    const maxReconnectAttempts = 5;

    const connect = () => {
      try {
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
          console.log('WebSocket connected');
          reconnectAttempts = 0;
          setProgress(prev => ({ ...prev, status: 'running' }));
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);

            if (data.type === 'progress') {
              setProgress(prev => ({
                ...prev,
                currentIteration: data.iteration,
                fractionA: data.fractionA,
                converged: data.converged || false,
                elapsedTime: data.elapsedTime || prev.elapsedTime,
                estimatedTimeRemaining: data.estimatedTimeRemaining || 0,
              }));
            } else if (data.type === 'complete') {
              setProgress(prev => ({
                ...prev,
                status: 'completed',
                converged: data.results?.converged || true,
              }));

              if (onComplete) {
                onComplete(data.results);
              }
            } else if (data.type === 'error') {
              setProgress(prev => ({ ...prev, status: 'error' }));
              console.error('Simulation error:', data.message);
            }
          } catch (err) {
            console.error('Failed to parse WebSocket message:', err);
          }
        };

        ws.onerror = (error) => {
          console.error('WebSocket error:', error);
        };

        ws.onclose = () => {
          console.log('WebSocket disconnected');

          // Attempt reconnection if not completed
          if (progress.status === 'running' && reconnectAttempts < maxReconnectAttempts) {
            reconnectAttempts++;
            setTimeout(connect, 2000 * reconnectAttempts);
          }
        };
      } catch (err) {
        console.error('Failed to create WebSocket:', err);
      }
    };

    connect();

    return () => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    };
  }, [simulationId, onComplete, progress.status]);

  const progressPercentage = progress.totalIterations > 0
    ? (progress.currentIteration / progress.totalIterations) * 100
    : 0;

  const formatTime = (seconds: number): string => {
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`;
    return `${(seconds / 3600).toFixed(1)}h`;
  };

  return (
    <div className="w-full max-w-2xl mx-auto p-6 bg-white dark:bg-gray-800 rounded-lg shadow-lg">
      {/* Status Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          {progress.status === 'running' && (
            <>
              <Activity className="h-5 w-5 text-blue-500 animate-pulse" />
              <span className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                Running Simulation
              </span>
            </>
          )}
          {progress.status === 'completed' && (
            <>
              <CheckCircle className="h-5 w-5 text-green-500" />
              <span className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                Completed
              </span>
            </>
          )}
          {progress.status === 'idle' && (
            <>
              <Clock className="h-5 w-5 text-gray-400" />
              <span className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                Waiting to Start
              </span>
            </>
          )}
          {progress.status === 'error' && (
            <>
              <span className="h-5 w-5 text-red-500">⚠</span>
              <span className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                Error
              </span>
            </>
          )}
        </div>

        {progress.converged && (
          <div className="flex items-center gap-1 px-3 py-1 bg-green-100 dark:bg-green-900 rounded-full">
            <Zap className="h-4 w-4 text-green-600 dark:text-green-400" />
            <span className="text-sm font-medium text-green-700 dark:text-green-300">
              Converged
            </span>
          </div>
        )}
      </div>

      {/* Progress Bar */}
      <div className="mb-4">
        <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400 mb-2">
          <span>Iteration {progress.currentIteration} / {progress.totalIterations}</span>
          <span>{progressPercentage.toFixed(1)}%</span>
        </div>
        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3 overflow-hidden">
          <div
            className={`h-full transition-all duration-300 ${
              progress.converged
                ? 'bg-green-500'
                : progress.status === 'running'
                ? 'bg-blue-500'
                : 'bg-gray-400'
            }`}
            style={{ width: `${Math.min(progressPercentage, 100)}%` }}
            role="progressbar"
            aria-valuenow={progressPercentage}
            aria-valuemin={0}
            aria-valuemax={100}
          />
        </div>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
        <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
          <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">
            Cooperation Rate
          </div>
          <div className="text-xl font-bold text-gray-900 dark:text-gray-100">
            {(progress.fractionA * 100).toFixed(1)}%
          </div>
        </div>

        <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
          <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">
            Elapsed Time
          </div>
          <div className="text-xl font-bold text-gray-900 dark:text-gray-100">
            {formatTime(progress.elapsedTime)}
          </div>
        </div>

        <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
          <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">
            Time Remaining
          </div>
          <div className="text-xl font-bold text-gray-900 dark:text-gray-100">
            {progress.status === 'running'
              ? formatTime(progress.estimatedTimeRemaining)
              : '-'
            }
          </div>
        </div>

        <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
          <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">
            Status
          </div>
          <div className="text-xl font-bold text-gray-900 dark:text-gray-100 capitalize">
            {progress.status}
          </div>
        </div>
      </div>

      {/* Convergence Indicator */}
      {progress.status === 'running' && (
        <div className="text-sm text-gray-600 dark:text-gray-400 text-center">
          {progress.converged
            ? 'System has reached equilibrium'
            : 'Waiting for convergence...'
          }
        </div>
      )}
    </div>
  );
}
