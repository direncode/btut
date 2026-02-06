import { NextRequest, NextResponse } from 'next/server';
import { projects, type StoredProject } from '@/lib/api-store';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const name: string = body.name;
    const description: string | null = body.description ?? null;
    const teamMembers: string[] = body.team_members ?? [];

    if (!name || name.length === 0) {
      return NextResponse.json(
        { detail: 'Project name is required' },
        { status: 400 }
      );
    }

    const projectId = crypto.randomUUID();
    const project: StoredProject = {
      project_id: projectId,
      name,
      description,
      owner: 'anonymous',
      team_members: teamMembers,
      simulations: [],
      created_at: new Date().toISOString(),
    };

    projects.set(projectId, project);

    return NextResponse.json(project);
  } catch (e: any) {
    return NextResponse.json(
      { detail: e.message || 'Failed to create project' },
      { status: 500 }
    );
  }
}
