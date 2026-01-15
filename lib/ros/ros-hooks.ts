/**
 * React hooks for ROS integration
 */

import { useState, useEffect, useCallback, useRef } from 'react'
import ROSLIB from 'roslib'
import {
  connectToROS,
  subscribeToAgentStates,
  subscribeToCoordinationResults,
  callRunSimulation,
  callGetStrategy,
  publishParameterUpdate,
  getROSNodes,
  getROSTopics,
  getROSServices,
  isROSConnected,
  disconnectROS,
  type AgentState,
  type SimulationConfig,
  type CoordinationResult
} from './rosbridge-client'

export interface UseROSConnectionOptions {
  url?: string
  autoConnect?: boolean
  reconnectInterval?: number
}

export interface ROSConnectionState {
  connected: boolean
  connecting: boolean
  error: string | null
  ros: ROSLIB.Ros | null
}

/**
 * Hook for managing ROS connection
 */
export function useROSConnection(options: UseROSConnectionOptions = {}) {
  const {
    url = 'ws://localhost:9090',
    autoConnect = false,
    reconnectInterval = 5000
  } = options

  const [state, setState] = useState<ROSConnectionState>({
    connected: false,
    connecting: false,
    error: null,
    ros: null
  })

  const reconnectTimerRef = useRef<NodeJS.Timeout | null>(null)
  const rosRef = useRef<ROSLIB.Ros | null>(null)

  const connect = useCallback(() => {
    if (state.connecting || state.connected) return

    setState(prev => ({ ...prev, connecting: true, error: null }))

    try {
      const ros = connectToROS(url)
      rosRef.current = ros

      ros.on('connection', () => {
        setState({ connected: true, connecting: false, error: null, ros })
        if (reconnectTimerRef.current) {
          clearInterval(reconnectTimerRef.current)
          reconnectTimerRef.current = null
        }
      })

      ros.on('error', (error: any) => {
        const errorMsg = error?.message || 'Connection error'
        setState(prev => ({ ...prev, error: errorMsg, connecting: false }))
      })

      ros.on('close', () => {
        setState(prev => ({ ...prev, connected: false, connecting: false }))

        // Auto-reconnect
        if (!reconnectTimerRef.current && reconnectInterval > 0) {
          reconnectTimerRef.current = setInterval(() => {
            if (!isROSConnected(ros)) {
              connect()
            }
          }, reconnectInterval)
        }
      })
    } catch (error: any) {
      setState({
        connected: false,
        connecting: false,
        error: error.message,
        ros: null
      })
    }
  }, [url, reconnectInterval, state.connecting, state.connected])

  const disconnect = useCallback(() => {
    if (rosRef.current) {
      disconnectROS(rosRef.current)
      rosRef.current = null
    }
    if (reconnectTimerRef.current) {
      clearInterval(reconnectTimerRef.current)
      reconnectTimerRef.current = null
    }
    setState({ connected: false, connecting: false, error: null, ros: null })
  }, [])

  useEffect(() => {
    if (autoConnect) {
      connect()
    }

    return () => {
      if (reconnectTimerRef.current) {
        clearInterval(reconnectTimerRef.current)
      }
      if (rosRef.current) {
        disconnectROS(rosRef.current)
      }
    }
  }, [autoConnect, connect])

  return {
    ...state,
    connect,
    disconnect
  }
}

/**
 * Hook for subscribing to agent states
 */
export function useAgentStates(ros: ROSLIB.Ros | null, namespace: string = '/btut') {
  const [agents, setAgents] = useState<AgentState[]>([])
  const [lastUpdate, setLastUpdate] = useState<number>(0)

  useEffect(() => {
    if (!ros || !isROSConnected(ros)) return

    const unsubscribe = subscribeToAgentStates(
      ros,
      (newAgents) => {
        setAgents(newAgents)
        setLastUpdate(Date.now())
      },
      namespace
    )

    return unsubscribe
  }, [ros, namespace])

  return { agents, lastUpdate }
}

/**
 * Hook for subscribing to coordination results
 */
export function useCoordinationResults(ros: ROSLIB.Ros | null, namespace: string = '/btut') {
  const [result, setResult] = useState<CoordinationResult | null>(null)
  const [lastUpdate, setLastUpdate] = useState<number>(0)

  useEffect(() => {
    if (!ros || !isROSConnected(ros)) return

    const unsubscribe = subscribeToCoordinationResults(
      ros,
      (newResult) => {
        setResult(newResult)
        setLastUpdate(Date.now())
      },
      namespace
    )

    return unsubscribe
  }, [ros, namespace])

  return { result, lastUpdate }
}

/**
 * Hook for running simulations
 */
export function useRunSimulation(ros: ROSLIB.Ros | null, namespace: string = '/btut') {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<CoordinationResult | null>(null)

  const runSimulation = useCallback(async (config: SimulationConfig) => {
    if (!ros || !isROSConnected(ros)) {
      setError('Not connected to ROS')
      return
    }

    setLoading(true)
    setError(null)

    try {
      const simResult = await callRunSimulation(ros, config, namespace)
      setResult(simResult)
      setLoading(false)
    } catch (err: any) {
      setError(err.message)
      setLoading(false)
    }
  }, [ros, namespace])

  return { runSimulation, loading, error, result }
}

/**
 * Hook for getting agent strategy
 */
export function useGetStrategy(ros: ROSLIB.Ros | null, namespace: string = '/btut') {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const getStrategy = useCallback(async (agentId: string): Promise<'A' | 'B' | null> => {
    if (!ros || !isROSConnected(ros)) {
      setError('Not connected to ROS')
      return null
    }

    setLoading(true)
    setError(null)

    try {
      const strategy = await callGetStrategy(ros, agentId, namespace)
      setLoading(false)
      return strategy
    } catch (err: any) {
      setError(err.message)
      setLoading(false)
      return null
    }
  }, [ros, namespace])

  return { getStrategy, loading, error }
}

/**
 * Hook for updating parameters
 */
export function useParameterUpdate(ros: ROSLIB.Ros | null, namespace: string = '/btut') {
  const updateParameters = useCallback((params: Partial<SimulationConfig>) => {
    if (!ros || !isROSConnected(ros)) {
      console.warn('Cannot update parameters: not connected to ROS')
      return
    }

    publishParameterUpdate(ros, params, namespace)
  }, [ros, namespace])

  return { updateParameters }
}

/**
 * Hook for getting ROS system info
 */
export function useROSSystemInfo(ros: ROSLIB.Ros | null) {
  const [nodes, setNodes] = useState<string[]>([])
  const [topics, setTopics] = useState<string[]>([])
  const [services, setServices] = useState<string[]>([])
  const [loading, setLoading] = useState(false)

  const refresh = useCallback(async () => {
    if (!ros || !isROSConnected(ros)) return

    setLoading(true)
    try {
      const [nodesList, topicsList, servicesList] = await Promise.all([
        getROSNodes(ros),
        getROSTopics(ros),
        getROSServices(ros)
      ])
      setNodes(nodesList)
      setTopics(topicsList)
      setServices(servicesList)
    } catch (error) {
      console.error('Failed to fetch ROS system info:', error)
    }
    setLoading(false)
  }, [ros])

  useEffect(() => {
    refresh()
  }, [refresh])

  return { nodes, topics, services, loading, refresh }
}
