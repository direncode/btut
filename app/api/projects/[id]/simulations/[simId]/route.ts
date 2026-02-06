import { NextRequest, NextResponse } from 'next/server';
import { projects, simulations } from '@/lib/api-store';

export async function POST(
  _request: NextRequest,
  { params }: { params: { id: string; simId: string } }
) {
  const { id, simId } = params;

  const project = projects.get(id);
  if (!project) {
    return NextResponse.json(
      { detail: 'Project not found' },
      { status: 404 }
    );
  }

  if (!simulations.has(simId)) {
    return NextResponse.json(
      { detail: 'Simulation not found' },
      { status: 404 }
    );
  }

  if (!project.simulations.includes(simId)) {
    project.simulations.push(simId);
  }

  return NextResponse.json({ message: 'Simulation added to project' });
}
