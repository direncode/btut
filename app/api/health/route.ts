import { NextResponse } from 'next/server';
import { simulations, startTime } from '@/lib/api-store';

export async function GET() {
  let activeCount = 0;
  for (const sim of simulations.values()) {
    if (sim.status === 'running') activeCount++;
  }

  return NextResponse.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    active_simulations: activeCount,
    total_simulations: simulations.size,
    uptime_seconds: (Date.now() - startTime) / 1000,
  });
}
