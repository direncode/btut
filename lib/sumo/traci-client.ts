/**
 * SUMO TraCI Client
 * Connects to SUMO traffic simulation via TraCI protocol over WebSocket
 */

import { io, Socket } from 'socket.io-client'

export interface VehicleState {
  id: string
  position: { x: number; y: number }
  speed: number
  angle: number
  roadId: string
  laneId: string
  lanePosition: number
  strategy: 'A' | 'B'
  waitingTime: number
}

export interface TraCIMetrics {
  totalVehicles: number
  averageSpeed: number
  totalWaitingTime: number
  throughput: number
  meanTravelTime: number
  fuelConsumption: number
  co2Emissions: number
}

export interface NetworkInfo {
  edges: string[]
  nodes: string[]
  bounds: {
    xMin: number
    yMin: number
    xMax: number
    yMax: number
  }
}

export interface SimulationStep {
  currentTime: number
  vehicles: VehicleState[]
  metrics: TraCIMetrics
}

export class TraCIClient {
  private socket: Socket | null = null
  private connected: boolean = false
  private listeners: Map<string, Set<Function>> = new Map()

  /**
   * Connect to SUMO TraCI WebSocket server
   */
  async connect(host: string = 'localhost', port: number = 8813): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.socket = io(`http://${host}:${port}`, {
          transports: ['websocket'],
          reconnection: true,
          reconnectionDelay: 1000,
          reconnectionAttempts: 5
        })

        this.socket.on('connect', () => {
          console.log('✅ Connected to SUMO TraCI server')
          this.connected = true
          this.emit('connect', null)
          resolve()
        })

        this.socket.on('disconnect', () => {
          console.log('🔌 Disconnected from SUMO')
          this.connected = false
          this.emit('disconnect', null)
        })

        this.socket.on('error', (error) => {
          console.error('❌ SUMO connection error:', error)
          this.emit('error', error)
          reject(error)
        })

        this.socket.on('step', (data: SimulationStep) => {
          this.emit('step', data)
        })

        this.socket.on('metrics', (data: TraCIMetrics) => {
          this.emit('metrics', data)
        })

      } catch (error) {
        reject(error)
      }
    })
  }

  /**
   * Disconnect from SUMO
   */
  disconnect(): void {
    if (this.socket) {
      this.socket.disconnect()
      this.socket = null
      this.connected = false
    }
  }

  /**
   * Check if connected
   */
  isConnected(): boolean {
    return this.connected && this.socket !== null
  }

  /**
   * Start SUMO simulation
   */
  async startSimulation(networkFile: string, config?: any): Promise<void> {
    if (!this.socket) throw new Error('Not connected to SUMO')

    return new Promise((resolve, reject) => {
      this.socket!.emit('start', { networkFile, config }, (response: any) => {
        if (response.success) {
          resolve()
        } else {
          reject(new Error(response.error || 'Failed to start simulation'))
        }
      })
    })
  }

  /**
   * Step simulation forward
   */
  async step(numSteps: number = 1): Promise<SimulationStep> {
    if (!this.socket) throw new Error('Not connected to SUMO')

    return new Promise((resolve, reject) => {
      this.socket!.emit('step', { numSteps }, (response: any) => {
        if (response.success) {
          resolve(response.data)
        } else {
          reject(new Error(response.error || 'Simulation step failed'))
        }
      })
    })
  }

  /**
   * Pause simulation
   */
  async pause(): Promise<void> {
    if (!this.socket) throw new Error('Not connected to SUMO')

    return new Promise((resolve, reject) => {
      this.socket!.emit('pause', {}, (response: any) => {
        if (response.success) {
          resolve()
        } else {
          reject(new Error(response.error || 'Failed to pause'))
        }
      })
    })
  }

  /**
   * Resume simulation
   */
  async resume(): Promise<void> {
    if (!this.socket) throw new Error('Not connected to SUMO')

    return new Promise((resolve, reject) => {
      this.socket!.emit('resume', {}, (response: any) => {
        if (response.success) {
          resolve()
        } else {
          reject(new Error(response.error || 'Failed to resume'))
        }
      })
    })
  }

  /**
   * Stop simulation
   */
  async stop(): Promise<void> {
    if (!this.socket) throw new Error('Not connected to SUMO')

    return new Promise((resolve, reject) => {
      this.socket!.emit('stop', {}, (response: any) => {
        if (response.success) {
          resolve()
        } else {
          reject(new Error(response.error || 'Failed to stop'))
        }
      })
    })
  }

  /**
   * Get all vehicle IDs
   */
  async getVehicleIds(): Promise<string[]> {
    if (!this.socket) throw new Error('Not connected to SUMO')

    return new Promise((resolve, reject) => {
      this.socket!.emit('getVehicleIds', {}, (response: any) => {
        if (response.success) {
          resolve(response.data)
        } else {
          reject(new Error(response.error || 'Failed to get vehicle IDs'))
        }
      })
    })
  }

  /**
   * Get vehicle positions
   */
  async getVehiclePositions(): Promise<Map<string, { x: number; y: number }>> {
    if (!this.socket) throw new Error('Not connected to SUMO')

    return new Promise((resolve, reject) => {
      this.socket!.emit('getVehiclePositions', {}, (response: any) => {
        if (response.success) {
          resolve(new Map(Object.entries(response.data)))
        } else {
          reject(new Error(response.error || 'Failed to get positions'))
        }
      })
    })
  }

  /**
   * Set vehicle strategy
   */
  async setVehicleStrategy(vehicleId: string, strategy: 'A' | 'B'): Promise<void> {
    if (!this.socket) throw new Error('Not connected to SUMO')

    return new Promise((resolve, reject) => {
      this.socket!.emit('setStrategy', { vehicleId, strategy }, (response: any) => {
        if (response.success) {
          resolve()
        } else {
          reject(new Error(response.error || 'Failed to set strategy'))
        }
      })
    })
  }

  /**
   * Apply BTUT coordination
   */
  async applyBTUTCoordination(gamma: number = 1.5, tau: number = 0.3): Promise<void> {
    if (!this.socket) throw new Error('Not connected to SUMO')

    return new Promise((resolve, reject) => {
      this.socket!.emit('applyBTUT', { gamma, tau }, (response: any) => {
        if (response.success) {
          resolve()
        } else {
          reject(new Error(response.error || 'Failed to apply BTUT'))
        }
      })
    })
  }

  /**
   * Get traffic metrics
   */
  async getMetrics(): Promise<TraCIMetrics> {
    if (!this.socket) throw new Error('Not connected to SUMO')

    return new Promise((resolve, reject) => {
      this.socket!.emit('getMetrics', {}, (response: any) => {
        if (response.success) {
          resolve(response.data)
        } else {
          reject(new Error(response.error || 'Failed to get metrics'))
        }
      })
    })
  }

  /**
   * Get network information
   */
  async getNetworkInfo(): Promise<NetworkInfo> {
    if (!this.socket) throw new Error('Not connected to SUMO')

    return new Promise((resolve, reject) => {
      this.socket!.emit('getNetworkInfo', {}, (response: any) => {
        if (response.success) {
          resolve(response.data)
        } else {
          reject(new Error(response.error || 'Failed to get network info'))
        }
      })
    })
  }

  /**
   * Load network file
   */
  async loadNetwork(networkFile: string): Promise<void> {
    if (!this.socket) throw new Error('Not connected to SUMO')

    return new Promise((resolve, reject) => {
      this.socket!.emit('loadNetwork', { networkFile }, (response: any) => {
        if (response.success) {
          resolve()
        } else {
          reject(new Error(response.error || 'Failed to load network'))
        }
      })
    })
  }

  /**
   * Export simulation results
   */
  async exportResults(format: 'csv' | 'json' | 'xml' = 'json'): Promise<string> {
    if (!this.socket) throw new Error('Not connected to SUMO')

    return new Promise((resolve, reject) => {
      this.socket!.emit('exportResults', { format }, (response: any) => {
        if (response.success) {
          resolve(response.data)
        } else {
          reject(new Error(response.error || 'Failed to export results'))
        }
      })
    })
  }

  /**
   * Subscribe to events
   */
  on(event: string, callback: Function): void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set())
    }
    this.listeners.get(event)!.add(callback)
  }

  /**
   * Unsubscribe from events
   */
  off(event: string, callback: Function): void {
    if (this.listeners.has(event)) {
      this.listeners.get(event)!.delete(callback)
    }
  }

  /**
   * Emit event to listeners
   */
  private emit(event: string, data: any): void {
    if (this.listeners.has(event)) {
      this.listeners.get(event)!.forEach(callback => callback(data))
    }
  }
}

/**
 * Create a new TraCI client instance
 */
export function createTraCIClient(): TraCIClient {
  return new TraCIClient()
}
