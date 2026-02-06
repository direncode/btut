import { NextResponse } from 'next/server';
import { PRESETS } from '@/lib/simulation/btut-engine';

export async function GET() {
  return NextResponse.json({
    presets: PRESETS,
    descriptions: {
      quick: 'Fast 1K agent test',
      default: 'Standard 1K simulation',
      small: 'Small 100 agent simulation',
      large: 'Large-scale 10K agents',
      highCooperation: 'High cooperation scenario',
      lowCooperation: 'Low cooperation scenario',
    },
  });
}
