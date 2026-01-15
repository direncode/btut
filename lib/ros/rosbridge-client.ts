/**
 * ROS Bridge Client
 * Connects to ROS via rosbridge_suite WebSocket server
 */

import * as ROSLIB from 'roslib'

export interface ROSConnection {
  ros: ROSLIB.Ros
  connected: boolean
  error?: string
}

export interface AgentState {
  id: string
  position: { x: number; y: number; z: number }
  strategy: 'A' | 'B'
  utility: number
  timestamp: number
}

export interface SimulationConfig {
  N: number
  gamma: number
  tau: number
  cA_SH: number
  cB_SH: number
  cA_PD: number
  cB_PD: number
  alpha: number
  m: number
}

export interface CoordinationResult {
  finalFractionA: number
  convergenceTime: number
  totalIterations: number
  averageUtility: number
}

/**
 * Connect to ROS master via rosbridge
 */
export function connectToROS(url: string = 'ws://localhost:9090'): ROSLIB.Ros {
  const ros = new ROSLIB.Ros({ url })

  ros.on('connection', () => {
    console.log('✅ Connected to ROS master at', url)
  })

  ros.on('error', (error) => {
    console.error('❌ ROS connection error:', error)
  })

  ros.on('close', () => {
    console.log('🔌 ROS connection closed')
  })

  return ros
}

/**
 * Subscribe to agent states topic
 */
export function subscribeToAgentStates(
  ros: ROSLIB.Ros,
  callback: (agents: AgentState[]) => void,
  namespace: string = '/btut'
): () => void {
  const topic = new ROSLIB.Topic({
    ros,
    name: `${namespace}/agent_states`,
    messageType: 'btut_msgs/AgentStateArray'
  })

  topic.subscribe((message: any) => {
    const agents: AgentState[] = message.agents.map((agent: any) => ({
      id: agent.id,
      position: agent.position,
      strategy: agent.strategy,
      utility: agent.utility,
      timestamp: agent.header.stamp.secs + agent.header.stamp.nsecs / 1e9
    }))
    callback(agents)
  })

  return () => topic.unsubscribe()
}

/**
 * Subscribe to coordination results
 */
export function subscribeToCoordinationResults(
  ros: ROSLIB.Ros,
  callback: (result: CoordinationResult) => void,
  namespace: string = '/btut'
): () => void {
  const topic = new ROSLIB.Topic({
    ros,
    name: `${namespace}/coordination_result`,
    messageType: 'btut_msgs/CoordinationResult'
  })

  topic.subscribe((message: any) => {
    const result: CoordinationResult = {
      finalFractionA: message.final_fraction_a,
      convergenceTime: message.convergence_time,
      totalIterations: message.total_iterations,
      averageUtility: message.average_utility
    }
    callback(result)
  })

  return () => topic.unsubscribe()
}

/**
 * Call simulation service
 */
export function callRunSimulation(
  ros: ROSLIB.Ros,
  config: SimulationConfig,
  namespace: string = '/btut'
): Promise<CoordinationResult> {
  return new Promise((resolve, reject) => {
    const service = new ROSLIB.Service({
      ros,
      name: `${namespace}/run_simulation`,
      serviceType: 'btut_msgs/RunSimulation'
    })

    const request = new ROSLIB.ServiceRequest({
      config: {
        n: config.N,
        gamma: config.gamma,
        tau: config.tau,
        c_a_sh: config.cA_SH,
        c_b_sh: config.cB_SH,
        c_a_pd: config.cA_PD,
        c_b_pd: config.cB_PD,
        alpha: config.alpha,
        m: config.m
      }
    })

    service.callService(request, (response: any) => {
      if (response.success) {
        resolve({
          finalFractionA: response.result.final_fraction_a,
          convergenceTime: response.result.convergence_time,
          totalIterations: response.result.total_iterations,
          averageUtility: response.result.average_utility
        })
      } else {
        reject(new Error(response.message || 'Simulation failed'))
      }
    }, (error) => {
      reject(error)
    })
  })
}

/**
 * Get strategy for specific agent
 */
export function callGetStrategy(
  ros: ROSLIB.Ros,
  agentId: string,
  namespace: string = '/btut'
): Promise<'A' | 'B'> {
  return new Promise((resolve, reject) => {
    const service = new ROSLIB.Service({
      ros,
      name: `${namespace}/get_strategy`,
      serviceType: 'btut_msgs/GetStrategy'
    })

    const request = new ROSLIB.ServiceRequest({
      agent_id: agentId
    })

    service.callService(request, (response: any) => {
      if (response.success) {
        resolve(response.strategy)
      } else {
        reject(new Error(response.message || 'Failed to get strategy'))
      }
    }, reject)
  })
}

/**
 * Publish parameter update
 */
export function publishParameterUpdate(
  ros: ROSLIB.Ros,
  params: Partial<SimulationConfig>,
  namespace: string = '/btut'
): void {
  const topic = new ROSLIB.Topic({
    ros,
    name: `${namespace}/params`,
    messageType: 'btut_msgs/SimulationConfig'
  })

  const message = new ROSLIB.Message({
    n: params.N,
    gamma: params.gamma,
    tau: params.tau,
    c_a_sh: params.cA_SH,
    c_b_sh: params.cB_SH,
    c_a_pd: params.cA_PD,
    c_b_pd: params.cB_PD,
    alpha: params.alpha,
    m: params.m
  })

  topic.publish(message)
}

/**
 * Get list of active ROS nodes
 */
export function getROSNodes(ros: ROSLIB.Ros): Promise<string[]> {
  return new Promise((resolve, reject) => {
    ros.getNodes((nodes) => {
      resolve(nodes)
    }, reject)
  })
}

/**
 * Get list of active ROS topics
 */
export function getROSTopics(ros: ROSLIB.Ros): Promise<string[]> {
  return new Promise((resolve, reject) => {
    ros.getTopics((topics) => {
      resolve(topics.topics)
    }, reject)
  })
}

/**
 * Get list of available ROS services
 */
export function getROSServices(ros: ROSLIB.Ros): Promise<string[]> {
  return new Promise((resolve, reject) => {
    ros.getServices((services) => {
      resolve(services)
    }, reject)
  })
}

/**
 * Check if ROS connection is alive
 */
export function isROSConnected(ros: ROSLIB.Ros): boolean {
  return ros.isConnected
}

/**
 * Disconnect from ROS
 */
export function disconnectROS(ros: ROSLIB.Ros): void {
  ros.close()
}
