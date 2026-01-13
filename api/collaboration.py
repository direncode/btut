"""
BTUT Collaboration System
==========================
Team collaboration features for sharing simulations and results.

Features:
- Projects (collections of simulations)
- Team workspaces
- Simulation sharing
- Forking
- Comments
- Version history
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import uuid

router = APIRouter(prefix="/api/collab", tags=["collaboration"])

# Mock database (use PostgreSQL/MongoDB in production)
projects_db = {}
teams_db = {}
shares_db = {}


# Models
class User(BaseModel):
    user_id: str
    username: str
    email: str


class Project(BaseModel):
    project_id: str
    name: str
    description: Optional[str] = None
    owner_id: str
    team_id: Optional[str] = None
    simulations: List[str] = []
    created_at: str
    updated_at: str
    visibility: str = "private"  # "private", "team", "public"
    tags: List[str] = []


class Team(BaseModel):
    team_id: str
    name: str
    description: Optional[str] = None
    owner_id: str
    members: List[str] = []
    projects: List[str] = []
    created_at: str


class Share(BaseModel):
    share_id: str
    resource_type: str  # "project" or "simulation"
    resource_id: str
    shared_by: str
    shared_with: List[str] = []
    permission: str = "view"  # "view", "comment", "edit"
    created_at: str
    expires_at: Optional[str] = None


class Comment(BaseModel):
    comment_id: str
    user_id: str
    resource_type: str
    resource_id: str
    text: str
    created_at: str


# Endpoints

@router.post("/projects", response_model=Project)
async def create_project(
    name: str,
    description: Optional[str] = None,
    team_id: Optional[str] = None,
    user_id: str = "default_user"
):
    """Create a new project"""
    
    project_id = str(uuid.uuid4())
    
    project = Project(
        project_id=project_id,
        name=name,
        description=description,
        owner_id=user_id,
        team_id=team_id,
        simulations=[],
        created_at=datetime.utcnow().isoformat(),
        updated_at=datetime.utcnow().isoformat(),
        visibility="private" if not team_id else "team"
    )
    
    projects_db[project_id] = project.dict()
    
    return project


@router.get("/projects", response_model=List[Project])
async def list_projects(
    user_id: str = "default_user",
    team_id: Optional[str] = None
):
    """List accessible projects"""
    
    projects = []
    
    for project in projects_db.values():
        # Check access
        if (project['owner_id'] == user_id or 
            project['visibility'] == 'public' or
            (team_id and project['team_id'] == team_id)):
            projects.append(Project(**project))
    
    return projects


@router.get("/projects/{project_id}", response_model=Project)
async def get_project(project_id: str, user_id: str = "default_user"):
    """Get project details"""
    
    if project_id not in projects_db:
        raise HTTPException(status_code=404, detail="Project not found")
    
    project = projects_db[project_id]
    
    # Check access
    if (project['owner_id'] != user_id and 
        project['visibility'] == 'private'):
        raise HTTPException(status_code=403, detail="Access denied")
    
    return Project(**project)


@router.post("/projects/{project_id}/simulations")
async def add_simulation_to_project(
    project_id: str,
    simulation_id: str,
    user_id: str = "default_user"
):
    """Add a simulation to a project"""
    
    if project_id not in projects_db:
        raise HTTPException(status_code=404, detail="Project not found")
    
    project = projects_db[project_id]
    
    # Check permission
    if project['owner_id'] != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    if simulation_id not in project['simulations']:
        project['simulations'].append(simulation_id)
        project['updated_at'] = datetime.utcnow().isoformat()
    
    return {"message": "Simulation added to project"}


@router.post("/projects/{project_id}/fork")
async def fork_project(
    project_id: str,
    new_name: Optional[str] = None,
    user_id: str = "default_user"
):
    """Fork a project (create a copy)"""
    
    if project_id not in projects_db:
        raise HTTPException(status_code=404, detail="Project not found")
    
    original = projects_db[project_id]
    
    # Create fork
    fork_id = str(uuid.uuid4())
    fork = {
        **original,
        'project_id': fork_id,
        'name': new_name or f"{original['name']} (Fork)",
        'owner_id': user_id,
        'team_id': None,
        'visibility': 'private',
        'created_at': datetime.utcnow().isoformat(),
        'updated_at': datetime.utcnow().isoformat(),
        'forked_from': project_id
    }
    
    projects_db[fork_id] = fork
    
    return Project(**fork)


@router.post("/teams", response_model=Team)
async def create_team(
    name: str,
    description: Optional[str] = None,
    user_id: str = "default_user"
):
    """Create a team workspace"""
    
    team_id = str(uuid.uuid4())
    
    team = Team(
        team_id=team_id,
        name=name,
        description=description,
        owner_id=user_id,
        members=[user_id],
        projects=[],
        created_at=datetime.utcnow().isoformat()
    )
    
    teams_db[team_id] = team.dict()
    
    return team


@router.post("/teams/{team_id}/members")
async def add_team_member(
    team_id: str,
    member_id: str,
    user_id: str = "default_user"
):
    """Add member to team"""
    
    if team_id not in teams_db:
        raise HTTPException(status_code=404, detail="Team not found")
    
    team = teams_db[team_id]
    
    if team['owner_id'] != user_id:
        raise HTTPException(status_code=403, detail="Only team owner can add members")
    
    if member_id not in team['members']:
        team['members'].append(member_id)
    
    return {"message": "Member added"}


@router.post("/share", response_model=Share)
async def share_resource(
    resource_type: str,
    resource_id: str,
    shared_with: List[str],
    permission: str = "view",
    user_id: str = "default_user"
):
    """Share a project or simulation"""
    
    share_id = str(uuid.uuid4())
    
    share = Share(
        share_id=share_id,
        resource_type=resource_type,
        resource_id=resource_id,
        shared_by=user_id,
        shared_with=shared_with,
        permission=permission,
        created_at=datetime.utcnow().isoformat()
    )
    
    shares_db[share_id] = share.dict()
    
    return share


@router.post("/comments", response_model=Comment)
async def add_comment(
    resource_type: str,
    resource_id: str,
    text: str,
    user_id: str = "default_user"
):
    """Add comment to a project or simulation"""
    
    comment_id = str(uuid.uuid4())
    
    comment = Comment(
        comment_id=comment_id,
        user_id=user_id,
        resource_type=resource_type,
        resource_id=resource_id,
        text=text,
        created_at=datetime.utcnow().isoformat()
    )
    
    return comment


@router.get("/projects/{project_id}/export")
async def export_project(
    project_id: str,
    format: str = "json",
    user_id: str = "default_user"
):
    """Export project (JSON, PDF, or Python script)"""
    
    if project_id not in projects_db:
        raise HTTPException(status_code=404, detail="Project not found")
    
    project = projects_db[project_id]
    
    if format == "json":
        return project
    
    elif format == "python":
        # Generate Python script
        script = f'''
"""
Project: {project['name']}
Generated from BTUT Platform
"""

from btut import Simulator

# Project simulations
simulations = {project['simulations']}

# Run each simulation
for sim_id in simulations:
    # Load simulation config
    # sim = Simulator(**config)
    # results = sim.run()
    pass
'''
        return {"script": script}
    
    else:
        raise HTTPException(status_code=400, detail="Unsupported format")
