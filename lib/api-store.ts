/**
 * Server-side in-memory store for API routes
 * Shared across all Next.js API route handlers
 */

export interface StoredSimulation {
  simulation_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  config: Record<string, any>;
  results: Record<string, any> | null;
  created_at: string;
  completed_at: string | null;
  error: string | null;
  runtime_seconds: number | null;
}

export interface StoredProject {
  project_id: string;
  name: string;
  description: string | null;
  owner: string;
  team_members: string[];
  simulations: string[];
  created_at: string;
}

// Global in-memory stores (persist across requests in the same server process)
const globalStore = globalThis as any;

if (!globalStore.__btut_simulations) {
  globalStore.__btut_simulations = new Map<string, StoredSimulation>();
}
if (!globalStore.__btut_projects) {
  globalStore.__btut_projects = new Map<string, StoredProject>();
}
if (!globalStore.__btut_start_time) {
  globalStore.__btut_start_time = Date.now();
}

export const simulations: Map<string, StoredSimulation> = globalStore.__btut_simulations;
export const projects: Map<string, StoredProject> = globalStore.__btut_projects;
export const startTime: number = globalStore.__btut_start_time;
