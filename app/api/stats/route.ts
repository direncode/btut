import { NextResponse } from 'next/server';
import { simulations } from '@/lib/api-store';

export async function GET() {
  const total = simulations.size;
  const byStatus: Record<string, number> = {};
  let totalAgents = 0;
  let totalRuntime = 0;
  let completedCount = 0;

  for (const sim of simulations.values()) {
    byStatus[sim.status] = (byStatus[sim.status] ?? 0) + 1;
    totalAgents += sim.config?.N ?? 0;

    if (sim.status === 'completed' && sim.runtime_seconds != null) {
      totalRuntime += sim.runtime_seconds;
      completedCount++;
    }
  }

  return NextResponse.json({
    total_simulations: total,
    by_status: byStatus,
    total_agents_simulated: totalAgents,
    average_runtime_seconds: completedCount > 0 ? totalRuntime / completedCount : 0,
    timestamp: new Date().toISOString(),
  });
}
