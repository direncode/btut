/**
 * React hooks for SUMO integration
 */

import { useState, useEffect, useCallback, useRef } from 'react'
import { createTraCIClient, TraCIClient, type VehicleState, type TraCIMetrics, type SimulationStep, type NetworkInfo } from './traci-client'

export interface UseSUMOConnectionOptions {
  host?: string
  port?: number
  autoConnect?: boolean
}

export interface SUMOConnectionState {
  connected: boolean
  connecting: boolean
  error: string | null
  client: TraCIClient | null
}

/**
 * Hook for managing SUMO connection
 */
export function useSUMOConnection(options: UseSUMOConnectionOptions = {}) {
  const {
    host = 'localhost',
    port = 8813,
    autoConnect = false
  } = options

  const [state, setState] = useState<SUMOConnectionState>({
    connected: false,
    connecting: false,
    error: null,
    client: null
  })

  const clientRef = useRef<TraCIClient | null>(null)

  const connect = useCallback(async () => {
    if (state.connecting || state.connected) return

    setState(prev => ({ ...prev, connecting: true, error: null }))

    try {
      const client = createTraCIClient()
      clientRef.current = client

      client.on('connect', () => {
        setState({ connected: true, connecting: false, error: null, client })
      })

      client.on('disconnect', () => {
        setState(prev => ({ ...prev, connected: false }))
      })

      client.on('error', (error: any) => {
        setState(prev => ({
          ...prev,
          error: error?.message || 'Connection error',
          connecting: false
        }))
      })

      await client.connect(host, port)
    } catch (error: any) {
      setState({
        connected: false,
        connecting: false,
        error: error.message,
        client: null
      })
    }
  }, [host, port, state.connecting, state.connected])

  const disconnect = useCallback(() => {
    if (clientRef.current) {
      clientRef.current.disconnect()
      clientRef.current = null
    }
    setState({ connected: false, connecting: false, error: null, client: null })
  }, [])

  useEffect(() => {
    if (autoConnect) {
      connect()
    }

    return () => {
      if (clientRef.current) {
        clientRef.current.disconnect()
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
 * Hook for simulation control
 */
export function useSimulationControl(client: TraCIClient | null) {
  const [isRunning, setIsRunning] = useState(false)
  const [isPaused, setIsPaused] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)

  const start = useCallback(async (networkFile: string, config?: any) => {
    if (!client) throw new Error('Not connected to SUMO')
    await client.startSimulation(networkFile, config)
    setIsRunning(true)
    setIsPaused(false)
  }, [client])

  const stop = useCallback(async () => {
    if (!client) throw new Error('Not connected to SUMO')
    await client.stop()
    setIsRunning(false)
    setIsPaused(false)
    setCurrentTime(0)
  }, [client])

  const pause = useCallback(async () => {
    if (!client) throw new Error('Not connected to SUMO')
    await client.pause()
    setIsPaused(true)
  }, [client])

  const resume = useCallback(async () => {
    if (!client) throw new Error('Not connected to SUMO')
    await client.resume()
    setIsPaused(false)
  }, [client])

  const step = useCallback(async (numSteps: number = 1) => {
    if (!client) throw new Error('Not connected to SUMO')
    const result = await client.step(numSteps)
    setCurrentTime(result.currentTime)
    return result
  }, [client])

  return {
    isRunning,
    isPaused,
    currentTime,
    start,
    stop,
    pause,
    resume,
    step
  }
}

/**
 * Hook for real-time vehicle data
 */
export function useVehicleData(client: TraCIClient | null) {
  const [vehicles, setVehicles] = useState<VehicleState[]>([])
  const [vehicleCount, setVehicleCount] = useState(0)
  const [lastUpdate, setLastUpdate] = useState(0)

  useEffect(() => {
    if (!client) return

    const handleStep = (data: SimulationStep) => {
      setVehicles(data.vehicles)
      setVehicleCount(data.vehicles.length)
      setLastUpdate(Date.now())
    }

    client.on('step', handleStep)

    return () => {
      client.off('step', handleStep)
    }
  }, [client])

  return { vehicles, vehicleCount, lastUpdate }
}

/**
 * Hook for traffic metrics
 */
export function useTraCIMetrics(client: TraCIClient | null) {
  const [metrics, setMetrics] = useState<TraCIMetrics | null>(null)
  const [lastUpdate, setLastUpdate] = useState(0)

  useEffect(() => {
    if (!client) return

    const handleMetrics = (data: TraCIMetrics) => {
      setMetrics(data)
      setLastUpdate(Date.now())
    }

    const handleStep = (data: SimulationStep) => {
      setMetrics(data.metrics)
      setLastUpdate(Date.now())
    }

    client.on('metrics', handleMetrics)
    client.on('step', handleStep)

    return () => {
      client.off('metrics', handleMetrics)
      client.off('step', handleStep)
    }
  }, [client])

  const refreshMetrics = useCallback(async () => {
    if (!client) return
    const newMetrics = await client.getMetrics()
    setMetrics(newMetrics)
    setLastUpdate(Date.now())
  }, [client])

  return { metrics, lastUpdate, refreshMetrics }
}

/**
 * Hook for network information
 */
export function useNetworkInfo(client: TraCIClient | null) {
  const [networkInfo, setNetworkInfo] = useState<NetworkInfo | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadNetwork = useCallback(async (networkFile?: string) => {
    if (!client) {
      setError('Not connected to SUMO')
      return
    }

    setLoading(true)
    setError(null)

    try {
      if (networkFile) {
        await client.loadNetwork(networkFile)
      }
      const info = await client.getNetworkInfo()
      setNetworkInfo(info)
    } catch (err: any) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [client])

  return { networkInfo, loading, error, loadNetwork }
}

/**
 * Hook for BTUT coordination
 */
export function useBTUTCoordination(client: TraCIClient | null) {
  const [enabled, setEnabled] = useState(false)
  const [gamma, setGamma] = useState(1.5)
  const [tau, setTau] = useState(0.3)

  const applyCoordination = useCallback(async (newGamma?: number, newTau?: number) => {
    if (!client) throw new Error('Not connected to SUMO')

    const g = newGamma ?? gamma
    const t = newTau ?? tau

    await client.applyBTUTCoordination(g, t)
    setEnabled(true)
    setGamma(g)
    setTau(t)
  }, [client, gamma, tau])

  const disableCoordination = useCallback(() => {
    setEnabled(false)
  }, [])

  return {
    enabled,
    gamma,
    tau,
    setGamma,
    setTau,
    applyCoordination,
    disableCoordination
  }
}

/**
 * Hook for exporting results
 */
export function useExportResults(client: TraCIClient | null) {
  const [exporting, setExporting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const exportResults = useCallback(async (format: 'csv' | 'json' | 'xml' = 'json') => {
    if (!client) {
      setError('Not connected to SUMO')
      return null
    }

    setExporting(true)
    setError(null)

    try {
      const data = await client.exportResults(format)
      setExporting(false)

      // Download file
      const blob = new Blob([data], { type: getContentType(format) })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `sumo-results-${Date.now()}.${format}`
      a.click()
      URL.revokeObjectURL(url)

      return data
    } catch (err: any) {
      setError(err.message)
      setExporting(false)
      return null
    }
  }, [client])

  return { exportResults, exporting, error }
}

function getContentType(format: string): string {
  switch (format) {
    case 'csv': return 'text/csv'
    case 'json': return 'application/json'
    case 'xml': return 'application/xml'
    default: return 'text/plain'
  }
}

/**
 * Hook for comparison mode (baseline vs BTUT)
 */
export function useComparisonMode(client: TraCIClient | null) {
  const [baselineMetrics, setBaselineMetrics] = useState<TraCIMetrics | null>(null)
  const [btutMetrics, setBtutMetrics] = useState<TraCIMetrics | null>(null)
  const [mode, setMode] = useState<'baseline' | 'btut' | 'both'>('baseline')

  const runBaseline = useCallback(async (networkFile: string) => {
    if (!client) return

    await client.startSimulation(networkFile)
    // Run simulation and collect metrics
    const metrics = await client.getMetrics()
    setBaselineMetrics(metrics)
  }, [client])

  const runWithBTUT = useCallback(async (networkFile: string, gamma: number, tau: number) => {
    if (!client) return

    await client.startSimulation(networkFile)
    await client.applyBTUTCoordination(gamma, tau)
    // Run simulation and collect metrics
    const metrics = await client.getMetrics()
    setBtutMetrics(metrics)
  }, [client])

  const getImprovement = useCallback(() => {
    if (!baselineMetrics || !btutMetrics) return null

    return {
      speedIncrease: ((btutMetrics.averageSpeed - baselineMetrics.averageSpeed) / baselineMetrics.averageSpeed) * 100,
      waitTimeReduction: ((baselineMetrics.totalWaitingTime - btutMetrics.totalWaitingTime) / baselineMetrics.totalWaitingTime) * 100,
      throughputIncrease: ((btutMetrics.throughput - baselineMetrics.throughput) / baselineMetrics.throughput) * 100
    }
  }, [baselineMetrics, btutMetrics])

  return {
    baselineMetrics,
    btutMetrics,
    mode,
    setMode,
    runBaseline,
    runWithBTUT,
    getImprovement
  }
}
