/**
 * API Client for BTUT Backend
 * TypeScript client for interacting with FastAPI backend
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface SimulationConfig {
  N: number;
  gamma: number;
  tau: number;
  cA_SH: number;
  cB_SH: number;
  cA_PD: number;
  cB_PD: number;
  alpha: number;
  iterations: number;
  m: number;
  seed?: number;
}

export interface SimulationResult {
  simulation_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  config: SimulationConfig;
  results?: {
    final_fraction_a: number;
    convergence_history: number[];
    iterations_completed: number;
    converged: boolean;
    agent_count: number;
    final_strategy_a_count: number;
    final_strategy_b_count: number;
  };
  created_at: string;
  completed_at?: string;
  error?: string;
  runtime_seconds?: number;
}

export interface BenchmarkResult {
  agents: number;
  iterations: number;
  runtime_seconds: number;
  throughput: number;
}

class BTUTAPIClient {
  private baseURL: string;

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(error.detail || `API Error: ${response.status}`);
    }

    return response.json();
  }

  // ==================== Simulation Endpoints ====================

  async runSimulation(
    config: SimulationConfig,
    asyncMode: boolean = false
  ): Promise<SimulationResult> {
    return this.request<SimulationResult>('/api/simulate', {
      method: 'POST',
      body: JSON.stringify({ config, async_mode: asyncMode }),
    });
  }

  async getSimulation(simulationId: string): Promise<SimulationResult> {
    return this.request<SimulationResult>(`/api/simulate/${simulationId}`);
  }

  async deleteSimulation(simulationId: string): Promise<{ message: string }> {
    return this.request(`/api/simulate/${simulationId}`, {
      method: 'DELETE',
    });
  }

  async batchSimulate(
    configs: SimulationConfig[],
    parallel: boolean = true
  ): Promise<{
    batch_id: string;
    total: number;
    completed: number;
    failed: number;
    simulation_ids: string[];
    results: SimulationResult[];
  }> {
    return this.request('/api/simulate/batch', {
      method: 'POST',
      body: JSON.stringify({ configs, parallel }),
    });
  }

  async parameterSweep(
    baseConfig: SimulationConfig,
    parameter: 'gamma' | 'tau' | 'N' | 'alpha',
    values: number[],
    parallel: boolean = true
  ): Promise<{
    sweep_id: string;
    parameter: string;
    values: number[];
    total: number;
    results: Array<{
      parameter_value: number;
      simulation_id: string;
      result: SimulationResult;
    }>;
  }> {
    return this.request('/api/simulate/sweep', {
      method: 'POST',
      body: JSON.stringify({
        base_config: baseConfig,
        parameter,
        values,
        parallel,
      }),
    });
  }

  // ==================== Benchmark Endpoints ====================

  async runBenchmark(
    agentCounts: number[] = [1000, 10000, 100000],
    iterations: number = 20
  ): Promise<{
    benchmark_id: string;
    timestamp: string;
    results: BenchmarkResult[];
    comparison: Record<string, string>;
  }> {
    return this.request('/api/benchmark', {
      method: 'POST',
      body: JSON.stringify({ agent_counts: agentCounts, iterations }),
    });
  }

  // ==================== Preset Endpoints ====================

  async getPresets(): Promise<{
    presets: Record<string, SimulationConfig>;
    descriptions: Record<string, string>;
  }> {
    return this.request('/api/presets');
  }

  // ==================== Stats Endpoints ====================

  async getStats(): Promise<{
    total_simulations: number;
    by_status: Record<string, number>;
    total_agents_simulated: number;
    average_runtime_seconds: number;
    timestamp: string;
  }> {
    return this.request('/api/stats');
  }

  async getHealth(): Promise<{
    status: string;
    timestamp: string;
    active_simulations: number;
    total_simulations: number;
    uptime_seconds: number;
  }> {
    return this.request('/health');
  }

  // ==================== Project Endpoints ====================

  async createProject(
    name: string,
    description?: string,
    teamMembers: string[] = []
  ): Promise<{
    project_id: string;
    name: string;
    description?: string;
    owner: string;
    team_members: string[];
    simulations: string[];
    created_at: string;
  }> {
    return this.request('/api/projects', {
      method: 'POST',
      body: JSON.stringify({
        name,
        description,
        team_members: teamMembers,
      }),
    });
  }

  async getProject(projectId: string): Promise<{
    project_id: string;
    name: string;
    description?: string;
    owner: string;
    team_members: string[];
    simulations: string[];
    created_at: string;
  }> {
    return this.request(`/api/projects/${projectId}`);
  }

  async addSimulationToProject(
    projectId: string,
    simulationId: string
  ): Promise<{ message: string }> {
    return this.request(
      `/api/projects/${projectId}/simulations/${simulationId}`,
      {
        method: 'POST',
      }
    );
  }

  // ==================== WebSocket Connection ====================

  connectWebSocket(
    onMessage: (data: any) => void,
    onError?: (error: Event) => void
  ): WebSocket {
    const wsURL = this.baseURL.replace('http', 'ws') + '/ws';
    const ws = new WebSocket(wsURL);

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      if (onError) onError(error);
    };

    return ws;
  }
}

// Export singleton instance
export const apiClient = new BTUTAPIClient();

// Export class for custom instances
export default BTUTAPIClient;
