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
    const baseConfig = body.base_config ?? {};
    const parameter = body.parameter;
    const values: number[] = body.values ?? [];

    if (!parameter || !['gamma', 'tau', 'N', 'alpha'].includes(parameter)) {
      return NextResponse.json(
        { detail: 'Invalid parameter. Must be one of: gamma, tau, N, alpha' },
        { status: 400 }
      );
    }

    if (values.length < 2 || values.length > 50) {
      return NextResponse.json(
        { detail: 'Values must contain 2-50 entries' },
        { status: 400 }
      );
    }

    const sweepId = generateId();
    const results: Array<{
      parameter_value: number;
      simulation_id: string;
      result: StoredSimulation;
    }> = [];

    for (const value of values) {
      const config = { ...baseConfig, [parameter]: value };
      const simId = generateId();
      const now = new Date().toISOString();

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

      results.push({
        parameter_value: value,
        simulation_id: simId,
        result: record,
      });
    }

    return NextResponse.json({
      sweep_id: sweepId,
      parameter,
      values,
      total: results.length,
      results,
    });
  } catch (e: any) {
    return NextResponse.json(
      { detail: e.message || 'Parameter sweep failed' },
      { status: 500 }
    );
  }
}
