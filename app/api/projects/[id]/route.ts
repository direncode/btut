import { NextRequest, NextResponse } from 'next/server';
import { projects } from '@/lib/api-store';

export async function GET(
  _request: NextRequest,
  { params }: { params: { id: string } }
) {
  const { id } = params;
  const project = projects.get(id);

  if (!project) {
    return NextResponse.json(
      { detail: 'Project not found' },
      { status: 404 }
    );
  }

  return NextResponse.json(project);
}
