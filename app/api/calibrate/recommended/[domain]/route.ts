import { NextRequest, NextResponse } from 'next/server';
import { BTUTSimulator } from '@/lib/simulation/btut-engine';

function findRecommendedGamma(domain: string, tau: number, N: number): number {
  const domainConfigs = {
    abstract: { cA_SH: 0.40, cB_SH: 0.10, cA_PD: 0.20, cB_PD: 0.08, alpha: 0.60 },
    traffic: { cA_SH: 0.35, cB_SH: 0.15, cA_PD: 0.25, cB_PD: 0.10, alpha: 0.50 },
    drone: { cA_SH: 0.30, cB_SH: 0.12, cA_PD: 0.18, cB_PD: 0.06, alpha: 0.70 },
    robot: { cA_SH: 0.45, cB_SH: 0.08, cA_PD: 0.22, cB_PD: 0.09, alpha: 0.55 },
  } as const;

  const costs = domainConfigs[domain as keyof typeof domainConfigs] ?? domainConfigs.abstract;

  let lo = 0.1;
  let hi = 5.0;
  const threshold = 0.5;

  for (let i = 0; i < 20; i++) {
    const mid = (lo + hi) / 2;
    const sim = new BTUTSimulator({
      N,
      gamma: mid,
      tau,
      ...costs,
      iterations: 50,
      m: 3,
      seed: 42,
    });
    const states = sim.run();
    const fraction = states[states.length - 1]?.fractionA ?? 0;

    if (fraction > threshold) {
      hi = mid;
    } else {
      lo = mid;
    }
  }

  return (lo + hi) / 2 + 0.15; // Add margin
}

export async function GET(
  request: NextRequest,
  { params }: { params: { domain: string } }
) {
  try {
    const { domain } = params;
    const { searchParams } = new URL(request.url);
    const tau = parseFloat(searchParams.get('tau') ?? '0.3');
    const N = parseInt(searchParams.get('N') ?? '1000', 10);

    if (!['abstract', 'traffic', 'drone', 'robot'].includes(domain)) {
      return NextResponse.json(
        { detail: 'Invalid domain' },
        { status: 400 }
      );
    }

    const gamma = findRecommendedGamma(domain, tau, N);

    return NextResponse.json({
      domain,
      tau,
      N,
      recommended_gamma: gamma,
      description: `Optimal gamma for ${domain} domain with tau=${tau}`,
    });
  } catch (e: any) {
    return NextResponse.json(
      { detail: e.message || 'Failed to get recommended gamma' },
      { status: 500 }
    );
  }
}
