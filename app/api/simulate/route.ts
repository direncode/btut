import { NextRequest, NextResponse } from 'next/server';
import { BTUTSimulator } from '@/lib/simulation/btut-engine';
import { simulations, type StoredSimulation } from '@/lib/api-store';

function generateId(): string {
  return crypto.randomUUID();
}

function runSimulation(config: Record<string, any>): StoredSimulation['results'] {
  const simulator = new BTUTSimulator({
    N: config.N ?? 10000,
    gamma: config.gamma ?? 1.45,
    tau: config.tau ?? 0.30,
    cA_SH: config.cA_SH ?? 0.40,
    cB_SH: config.cB_SH ?? 0.10,
    cA_PD: config.cA_PD ?? 0.20,
    cB_PD: config.cB_PD ?? 0.08,
    alpha: config.alpha ?? 0.60,
    iterations: config.iterations ?? 20,
    m: config.m ?? 3,
    seed: config.seed ?? 42,
  });

  const states = simulator.run();
  const finalState = states[states.length - 1];

  return {
    final_fraction_a: finalState?.fractionA ?? 0,
    convergence_history: states.map(s => s.fractionA),
    iterations_completed: states.length,
    converged: finalState?.isComplete ?? false,
    agent_count: config.N ?? 10000,
    final_strategy_a_count: finalState?.strategyACount ?? 0,
    final_strategy_b_count: (config.N ?? 10000) - (finalState?.strategyACount ?? 0),
  };
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const config = body.config ?? body;
    const asyncMode = body.async_mode ?? false;

    const simId = generateId();
    const now = new Date().toISOString();

    const record: StoredSimulation = {
      simulation_id: simId,
      status: 'pending',
      config,
      results: null,
      created_at: now,
      completed_at: null,
      error: null,
      runtime_seconds: null,
    };

    simulations.set(simId, record);

    if (asyncMode) {
      // Mark as pending and return immediately; run in background
      record.status = 'running';
      // Fire-and-forget simulation
      Promise.resolve().then(() => {
        try {
          const startTime = performance.now();
          const results = runSimulation(config);
          const runtime = (performance.now() - startTime) / 1000;
          record.status = 'completed';
          record.results = results;
          record.completed_at = new Date().toISOString();
          record.runtime_seconds = runtime;
        } catch (e: any) {
          record.status = 'failed';
          record.error = e.message;
          record.completed_at = new Date().toISOString();
        }
      });

      return NextResponse.json(record);
    }

    // Synchronous mode
    record.status = 'running';
    const startTime = performance.now();
    const results = runSimulation(config);
    const runtime = (performance.now() - startTime) / 1000;

    record.status = 'completed';
    record.results = results;
    record.completed_at = new Date().toISOString();
    record.runtime_seconds = runtime;

    return NextResponse.json(record);
  } catch (e: any) {
    return NextResponse.json(
      { detail: e.message || 'Simulation failed' },
      { status: 500 }
    );
  }
}
