/**
 * TypeScript type definitions for ROS messages
 */

export interface ROSHeader {
  seq: number
  stamp: {
    secs: number
    nsecs: number
  }
  frame_id: string
}

export interface ROSPoint {
  x: number
  y: number
  z: number
}

export interface ROSQuaternion {
  x: number
  y: number
  z: number
  w: number
}

export interface ROSPose {
  position: ROSPoint
  orientation: ROSQuaternion
}

export interface ROSTwist {
  linear: ROSPoint
  angular: ROSPoint
}

export interface AgentStateMsg {
  header: ROSHeader
  id: string
  position: ROSPoint
  velocity: ROSTwist
  strategy: string
  utility: number
  neighbors: string[]
  hub_agent: boolean
}

export interface AgentStateArrayMsg {
  header: ROSHeader
  agents: AgentStateMsg[]
}

export interface SimulationConfigMsg {
  n: number
  gamma: number
  tau: number
  c_a_sh: number
  c_b_sh: number
  c_a_pd: number
  c_b_pd: number
  alpha: number
  m: number
}

export interface CoordinationResultMsg {
  header: ROSHeader
  success: boolean
  final_fraction_a: number
  convergence_time: number
  total_iterations: number
  average_utility: number
  convergence_history: number[]
}

export interface RunSimulationRequest {
  config: SimulationConfigMsg
}

export interface RunSimulationResponse {
  success: boolean
  result: CoordinationResultMsg
  message: string
}

export interface GetStrategyRequest {
  agent_id: string
}

export interface GetStrategyResponse {
  success: boolean
  strategy: string
  utility: number
  message: string
}

export interface ROSNodeInfo {
  name: string
  publications: string[]
  subscriptions: string[]
  services: string[]
}

export interface ROSTopicInfo {
  name: string
  type: string
  publishers: string[]
  subscribers: string[]
}

export interface ROSServiceInfo {
  name: string
  type: string
  node: string
}

export interface ROSSystemInfo {
  nodes: ROSNodeInfo[]
  topics: ROSTopicInfo[]
  services: ROSServiceInfo[]
  rosVersion: string
  rosMaster: string
}
