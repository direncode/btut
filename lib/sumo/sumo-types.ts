/**
 * TypeScript type definitions for SUMO integration
 */

export interface SUMOConnection {
  host: string
  port: number
  connected: boolean
  error?: string
}

export interface VehicleInfo {
  id: string
  type: string
  routeId: string
  edgeId: string
  laneId: string
  laneIndex: number
  position: VehiclePosition
  speed: number
  angle: number
  slope: number
  waitingTime: number
  accumulatedWaitingTime: number
  distance: number
  co2Emission: number
  fuelConsumption: number
  strategy: 'A' | 'B' | null
}

export interface VehiclePosition {
  x: number
  y: number
  z?: number
  lanePosition: number
}

export interface EdgeInfo {
  id: string
  fromNode: string
  toNode: string
  type?: string
  numLanes: number
  length: number
  speedLimit: number
  shape: Position[]
}

export interface Position {
  x: number
  y: number
}

export interface LaneInfo {
  id: string
  edgeId: string
  index: number
  length: number
  speedLimit: number
  width: number
  shape: Position[]
}

export interface JunctionInfo {
  id: string
  position: Position
  type: string
  incLanes: string[]
  intLanes: string[]
}

export interface TrafficLightInfo {
  id: string
  programId: string
  phase: number
  phaseDuration: number
  nextSwitch: number
  state: string
}

export interface DetectorInfo {
  id: string
  laneId: string
  position: number
  frequency: number
  vehicleCount: number
  speed: number
  occupancy: number
}

export interface RouteInfo {
  id: string
  edges: string[]
  color?: string
}

export interface SUMONetworkBounds {
  xMin: number
  yMin: number
  xMax: number
  yMax: number
  convBoundary: {
    xMin: number
    yMin: number
    xMax: number
    yMax: number
  }
  origBoundary: {
    xMin: number
    yMin: number
    xMax: number
    yMax: number
  }
}

export interface SUMONetworkInfo {
  edges: EdgeInfo[]
  junctions: JunctionInfo[]
  bounds: SUMONetworkBounds
  version: string
}

export interface SimulationState {
  currentTime: number
  loaded: number
  departed: number
  arrived: number
  running: number
  waiting: number
  ended: number
  colliding: number
  emergencyStopping: number
  meanWaitingTime: number
  meanSpeed: number
}

export interface ComparisonMetrics {
  baseline: TrafficMetrics
  withBTUT: TrafficMetrics
  improvement: {
    speedIncrease: number
    throughputIncrease: number
    waitTimeReduction: number
    fuelSavings: number
    co2Reduction: number
  }
}

export interface TrafficMetrics {
  timestamp: number
  totalVehicles: number
  activeVehicles: number
  waitingVehicles: number
  meanSpeed: number
  meanWaitingTime: number
  totalDistance: number
  totalFuel: number
  totalCO2: number
  throughput: number
  averageTravelTime: number
  congestionLevel: number
}

export interface SUMOConfig {
  networkFile: string
  routeFile?: string
  additionalFiles?: string[]
  beginTime?: number
  endTime?: number
  stepLength?: number
  gui?: boolean
  seed?: number
}

export interface BTUTTrafficConfig {
  gamma: number
  tau: number
  updateInterval: number
  hubVehicleRatio: number
  coordinationRadius: number
}

export interface ExportOptions {
  format: 'csv' | 'json' | 'xml' | 'fcd' | 'emissions'
  includeTimestamps: boolean
  includePositions: boolean
  includeStrategies: boolean
  includeEmissions: boolean
  filename?: string
}

export interface SUMOCommand {
  type: 'start' | 'stop' | 'pause' | 'resume' | 'step' | 'load' | 'setStrategy' | 'getMetrics'
  params?: any
}

export interface SUMOResponse {
  success: boolean
  data?: any
  error?: string
  timestamp: number
}

export interface CongestionZone {
  edgeId: string
  severity: 'low' | 'medium' | 'high' | 'critical'
  averageSpeed: number
  density: number
  position: Position
}

export interface TrafficIncident {
  id: string
  type: 'accident' | 'breakdown' | 'congestion' | 'roadwork'
  position: Position
  edgeId: string
  severity: number
  startTime: number
  duration: number
  affectedVehicles: string[]
}
