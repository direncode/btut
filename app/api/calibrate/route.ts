import { NextRequest, NextResponse } from 'next/server';
import { BTUTSimulator } from '@/lib/simulation/btut-engine';

/**
 * Find the critical gamma value where cooperation emerges.
 * Uses binary search over gamma to find the phase transition point.
 */
function findCriticalGamma(
  tau: number,
  N: number,
  domain: string
): {
  gamma_critical: number;
  gamma_lower: number;
  gamma_upper: number;
  transition_sharpness: number;
  recommended_gamma: number;
  recommended_margin: number;
  search_iterations: number;
  confidence: number;
} {
  // Domain-specific cost adjustments
  const domainConfigs = {
    abstract: { cA_SH: 0.40, cB_SH: 0.10, cA_PD: 0.20, cB_PD: 0.08, alpha: 0.60 },
    traffic: { cA_SH: 0.35, cB_SH: 0.15, cA_PD: 0.25, cB_PD: 0.10, alpha: 0.50 },
    drone: { cA_SH: 0.30, cB_SH: 0.12, cA_PD: 0.18, cB_PD: 0.06, alpha: 0.70 },
    robot: { cA_SH: 0.45, cB_SH: 0.08, cA_PD: 0.22, cB_PD: 0.09, alpha: 0.55 },
  } as const;

  const costs = domainConfigs[domain as keyof typeof domainConfigs] ?? domainConfigs.abstract;

  let lo = 0.1;
  let hi = 5.0;
  let iterations = 0;
  const maxIter = 30;
  const threshold = 0.5; // cooperation fraction threshold

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

  while (hi - lo > 0.01 && iterations < maxIter) {
    const mid = (lo + hi) / 2;
    const fraction = testGamma(mid);

    if (fraction > threshold) {
      hi = mid;
    } else {
      lo = mid;
    }
    iterations++;
  }

  const gammaCritical = (lo + hi) / 2;
  const margin = 0.15;

  // Test sharpness by checking fraction at gamma +/- 0.1
  const fractionBelow = testGamma(gammaCritical - 0.1);
  const fractionAbove = testGamma(gammaCritical + 0.1);
  const sharpness = Math.abs(fractionAbove - fractionBelow) / 0.2;

  return {
    gamma_critical: gammaCritical,
    gamma_lower: lo,
    gamma_upper: hi,
    transition_sharpness: sharpness,
    recommended_gamma: gammaCritical + margin,
    recommended_margin: margin,
    search_iterations: iterations,
    confidence: Math.min(0.99, 1 - (hi - lo) / 5),
  };
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const domain: string = body.domain ?? 'abstract';
    const tau: number = body.tau ?? 0.3;
    const N: number = body.N ?? 1000;

    if (!['abstract', 'traffic', 'drone', 'robot'].includes(domain)) {
      return NextResponse.json(
        { detail: 'Invalid domain. Must be one of: abstract, traffic, drone, robot' },
        { status: 400 }
      );
    }

    const result = findCriticalGamma(tau, N, domain);

    return NextResponse.json({
      ...result,
      domain,
      config_used: { tau, N, domain },
      timestamp: new Date().toISOString(),
      cached: false,
    });
  } catch (e: any) {
    return NextResponse.json(
      { detail: e.message || 'Calibration failed' },
      { status: 500 }
    );
  }
}
