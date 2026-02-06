import { NextRequest, NextResponse } from 'next/server';
import { simulations } from '@/lib/api-store';

export async function GET(
  _request: NextRequest,
  { params }: { params: { id: string } }
) {
  const { id } = params;
  const sim = simulations.get(id);

  if (!sim) {
    return NextResponse.json(
      { detail: 'Simulation not found' },
      { status: 404 }
    );
  }

  return NextResponse.json(sim);
}

export async function DELETE(
  _request: NextRequest,
  { params }: { params: { id: string } }
) {
  const { id } = params;

  if (!simulations.has(id)) {
    return NextResponse.json(
      { detail: 'Simulation not found' },
      { status: 404 }
    );
  }

  simulations.delete(id);
  return NextResponse.json({ message: 'Simulation deleted', simulation_id: id });
}
