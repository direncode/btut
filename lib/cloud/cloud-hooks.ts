/**
 * React hooks for cloud deployment
 */

import { useState, useCallback } from 'react'
import { createAWSClient, AWSLambdaClient, type AWSCredentials, type LambdaFunction, type SimulationRequest } from './aws-client'
import { deployBTUTLambda, type DeploymentConfig, type DeploymentResult } from './lambda-deploy'

export interface UseAWSConnectionOptions {
  credentials?: AWSCredentials
  autoConnect?: boolean
}

export interface AWSConnectionState {
  connected: boolean
  connecting: boolean
  error: string | null
  client: AWSLambdaClient | null
}

/**
 * Hook for managing AWS connection
 */
export function useAWSConnection(options: UseAWSConnectionOptions = {}) {
  const [state, setState] = useState<AWSConnectionState>({
    connected: false,
    connecting: false,
    error: null,
    client: null
  })

  const connect = useCallback((credentials: AWSCredentials) => {
    setState(prev => ({ ...prev, connecting: true, error: null }))

    try {
      const client = createAWSClient(credentials)
      setState({
        connected: true,
        connecting: false,
        error: null,
        client
      })
    } catch (error: any) {
      setState({
        connected: false,
        connecting: false,
        error: error.message,
        client: null
      })
    }
  }, [])

  const disconnect = useCallback(() => {
    setState({ connected: false, connecting: false, error: null, client: null })
  }, [])

  return {
    ...state,
    connect,
    disconnect
  }
}

/**
 * Hook for Lambda functions management
 */
export function useLambdaFunctions(client: AWSLambdaClient | null) {
  const [functions, setFunctions] = useState<LambdaFunction[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadFunctions = useCallback(async () => {
    if (!client) {
      setError('Not connected to AWS')
      return
    }

    setLoading(true)
    setError(null)

    try {
      const funcs = await client.listFunctions()
      setFunctions(funcs)
    } catch (err: any) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [client])

  const getFunction = useCallback(async (functionName: string) => {
    if (!client) {
      setError('Not connected to AWS')
      return null
    }

    setLoading(true)
    setError(null)

    try {
      const func = await client.getFunction(functionName)
      setLoading(false)
      return func
    } catch (err: any) {
      setError(err.message)
      setLoading(false)
      return null
    }
  }, [client])

  const deleteFunction = useCallback(async (functionName: string) => {
    if (!client) {
      setError('Not connected to AWS')
      return false
    }

    setLoading(true)
    setError(null)

    try {
      await client.deleteFunction(functionName)
      await loadFunctions() // Refresh list
      setLoading(false)
      return true
    } catch (err: any) {
      setError(err.message)
      setLoading(false)
      return false
    }
  }, [client, loadFunctions])

  return {
    functions,
    loading,
    error,
    loadFunctions,
    getFunction,
    deleteFunction
  }
}

/**
 * Hook for deploying Lambda functions
 */
export function useLambdaDeployment(client: AWSLambdaClient | null) {
  const [deploying, setDeploying] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<DeploymentResult | null>(null)

  const deploy = useCallback(async (config: DeploymentConfig) => {
    if (!client) {
      setError('Not connected to AWS')
      return null
    }

    setDeploying(true)
    setError(null)
    setResult(null)

    try {
      const deployResult = await deployBTUTLambda(client, config)
      setResult(deployResult)
      setDeploying(false)
      return deployResult
    } catch (err: any) {
      setError(err.message)
      setDeploying(false)
      return null
    }
  }, [client])

  return {
    deploying,
    error,
    result,
    deploy
  }
}

/**
 * Hook for invoking Lambda functions
 */
export function useLambdaInvoke(client: AWSLambdaClient | null) {
  const [invoking, setInvoking] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const invoke = useCallback(async (functionName: string, payload: any) => {
    if (!client) {
      setError('Not connected to AWS')
      return null
    }

    setInvoking(true)
    setError(null)

    try {
      const result = await client.invoke(functionName, payload)
      setInvoking(false)

      if (result.error) {
        setError(result.error)
        return null
      }

      return result
    } catch (err: any) {
      setError(err.message)
      setInvoking(false)
      return null
    }
  }, [client])

  const runSimulation = useCallback(async (functionName: string, request: SimulationRequest) => {
    if (!client) {
      setError('Not connected to AWS')
      return null
    }

    setInvoking(true)
    setError(null)

    try {
      const result = await client.runSimulation(functionName, request)
      setInvoking(false)
      return result
    } catch (err: any) {
      setError(err.message)
      setInvoking(false)
      return null
    }
  }, [client])

  const runBatch = useCallback(async (functionName: string, requests: SimulationRequest[]) => {
    if (!client) {
      setError('Not connected to AWS')
      return null
    }

    setInvoking(true)
    setError(null)

    try {
      const results = await client.runBatch(functionName, requests)
      setInvoking(false)
      return results
    } catch (err: any) {
      setError(err.message)
      setInvoking(false)
      return null
    }
  }, [client])

  return {
    invoking,
    error,
    invoke,
    runSimulation,
    runBatch
  }
}

/**
 * Hook for cost tracking
 */
export function useCostTracking(client: AWSLambdaClient | null) {
  const [costs, setCosts] = useState({
    today: 0,
    week: 0,
    month: 0
  })
  const [invocations, setInvocations] = useState({
    today: 0,
    week: 0,
    month: 0
  })
  const [loading, setLoading] = useState(false)

  const refresh = useCallback(async (functionName: string) => {
    if (!client) return

    setLoading(true)

    try {
      const [todayCost, weekCost, monthCost] = await Promise.all([
        client.getTotalCost(functionName, 'today'),
        client.getTotalCost(functionName, 'week'),
        client.getTotalCost(functionName, 'month')
      ])

      const [todayInvocations, weekInvocations, monthInvocations] = await Promise.all([
        client.getInvocationCount(functionName, 'today'),
        client.getInvocationCount(functionName, 'week'),
        client.getInvocationCount(functionName, 'month')
      ])

      setCosts({ today: todayCost, week: weekCost, month: monthCost })
      setInvocations({ today: todayInvocations, week: weekInvocations, month: monthInvocations })
    } catch (error) {
      console.error('Failed to fetch cost data:', error)
    } finally {
      setLoading(false)
    }
  }, [client])

  const estimateCost = useCallback((agents: number, iterations: number = 20) => {
    if (!client) return 0
    return client.estimateCost(agents, iterations)
  }, [client])

  return {
    costs,
    invocations,
    loading,
    refresh,
    estimateCost
  }
}

/**
 * Hook for CloudWatch logs
 */
export function useCloudWatchLogs(client: AWSLambdaClient | null) {
  const [logs, setLogs] = useState<string[]>([])
  const [loading, setLoading] = useState(false)

  const fetchLogs = useCallback(async (functionName: string, limit: number = 50) => {
    if (!client) return

    setLoading(true)

    // This would require CloudWatch Logs API integration
    // Simplified version for now
    setLogs([
      `[${new Date().toISOString()}] Function invoked`,
      `[${new Date().toISOString()}] Processing simulation request`,
      `[${new Date().toISOString()}] Simulation completed successfully`
    ])

    setLoading(false)
  }, [client])

  return {
    logs,
    loading,
    fetchLogs
  }
}
