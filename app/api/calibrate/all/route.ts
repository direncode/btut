import { NextRequest, NextResponse } from 'next/server';
import { BTUTSimulator } from '@/lib/simulation/btut-engine';

function calibrateDomain(
  domain: string,
  tau: number
): {
  gamma_critical: number;
  recommended_gamma: number;
  transition_sharpness: number;
  confidence: number;
} {
  const domainConfigs = {
    abstract: { cA_SH: 0.40, cB_SH: 0.10, cA_PD: 0.20, cB_PD: 0.08, alpha: 0.60 },
    traffic: { cA_SH: 0.35, cB_SH: 0.15, cA_PD: 0.25, cB_PD: 0.10, alpha: 0.50 },
    drone: { cA_SH: 0.30, cB_SH: 0.12, cA_PD: 0.18, cB_PD: 0.06, alpha: 0.70 },
    robot: { cA_SH: 0.45, cB_SH: 0.08, cA_PD: 0.22, cB_PD: 0.09, alpha: 0.55 },
  } as const;

  const costs = domainConfigs[domain as keyof typeof domainConfigs] ?? domainConfigs.abstract;
  const N = 1000;

  let lo = 0.1;
  let hi = 5.0;
  const threshold = 0.5;

  function testGamma(gamma: number): number {
    const sim = new BTUTSimulator({
      N,
      gamma,
      tau,
      ...costs,
      iterations: 50,
      m: 3,
      seed: 42,
    });
    const states = sim.run();
    return states[states.length - 1]?.fractionA ?? 0;
  }

  for (let i = 0; i < 20; i++) {
    const mid = (lo + hi) / 2;
    if (testGamma(mid) > threshold) {
      hi = mid;
    } else {
      lo = mid;
    }
  }

  const gammaCritical = (lo + hi) / 2;
  const fractionBelow = testGamma(gammaCritical - 0.1);
  const fractionAbove = testGamma(gammaCritical + 0.1);
  const sharpness = Math.abs(fractionAbove - fractionBelow) / 0.2;

  return {
    gamma_critical: gammaCritical,
    recommended_gamma: gammaCritical + 0.15,
    transition_sharpness: sharpness,
    confidence: Math.min(0.99, 1 - (hi - lo) / 5),
  };
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json().catch(() => ({}));
    const tau: number = body.tau ?? 0.3;

    const domains = ['abstract', 'traffic', 'drone', 'robot'];
    const domainResults: Record<string, ReturnType<typeof calibrateDomain>> = {};
    const summary: Record<string, number | null> = {};

    for (const domain of domains) {
      const result = calibrateDomain(domain, tau);
      domainResults[domain] = result;
      summary[domain] = result.recommended_gamma;
    }

    return NextResponse.json({
      calibration_id: crypto.randomUUID(),
      tau,
      timestamp: new Date().toISOString(),
      domains: domainResults,
      summary,
    });
  } catch (e: any) {
    return NextResponse.json(
      { detail: e.message || 'Calibration failed' },
      { status: 500 }
    );
  }
}
