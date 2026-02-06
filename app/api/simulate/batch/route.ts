import { NextRequest, NextResponse } from 'next/server';
import { BTUTSimulator } from '@/lib/simulation/btut-engine';
import { simulations, type StoredSimulation } from '@/lib/api-store';

function generateId(): string {
  return crypto.randomUUID();
}

function runSingle(config: Record<string, any>): StoredSimulation['results'] {
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
    const configs: Record<string, any>[] = body.configs ?? [];

    if (configs.length === 0) {
      return NextResponse.json(
        { detail: 'No configs provided' },
        { status: 400 }
      );
    }

    if (configs.length > 100) {
      return NextResponse.json(
        { detail: 'Maximum 100 configurations per batch' },
        { status: 400 }
      );
    }

    const batchId = generateId();
    const simIds: string[] = [];
    const results: StoredSimulation[] = [];

    for (const config of configs) {
      const simId = generateId();
      const now = new Date().toISOString();
      simIds.push(simId);

      const record: StoredSimulation = {
        simulation_id: simId,
        status: 'running',
        config,
        results: null,
        created_at: now,
        completed_at: null,
        error: null,
        runtime_seconds: null,
      };

      simulations.set(simId, record);

      try {
        const startTime = performance.now();
        const simResults = runSingle(config);
        const runtime = (performance.now() - startTime) / 1000;

        record.status = 'completed';
        record.results = simResults;
        record.completed_at = new Date().toISOString();
        record.runtime_seconds = runtime;
      } catch (e: any) {
        record.status = 'failed';
        record.error = e.message;
        record.completed_at = new Date().toISOString();
      }

      results.push(record);
    }

    return NextResponse.json({
      batch_id: batchId,
      total: configs.length,
      completed: results.filter(r => r.status === 'completed').length,
      failed: results.filter(r => r.status === 'failed').length,
      simulation_ids: simIds,
      results,
    });
  } catch (e: any) {
    return NextResponse.json(
      { detail: e.message || 'Batch simulation failed' },
      { status: 500 }
    );
  }
}
