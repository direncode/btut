"""
BTUT REST API
=============
Headless simulation API for researchers and external integrations.

Endpoints:
- POST /api/simulate - Run a simulation
- GET /api/simulate/:id - Get simulation results
- POST /api/simulate/batch - Batch simulations
- GET /api/presets - List available presets
- POST /api/benchmark - Run performance benchmark
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uuid
import json
from datetime import datetime
import asyncio

# Import BTUT engine
import sys
sys.path.append('../lib/simulation')
from btut_engine import BTUTSimulator, PRESETS


app = FastAPI(
    title="BTUT API",
    description="Scalable multi-agent simulation API",
    version="1.0.0"
)

# CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage (use Redis/MongoDB in production)
simulations_db = {}


# Request/Response Models
class SimulationConfig(BaseModel):
    N: int = Field(10000, description="Number of agents")
    gamma: float = Field(1.45, description="Cooperation bonus")
    tau: float = Field(0.30, description="Hub influence")
    cA_SH: float = Field(0.40, description="Cost for A in Stag Hunt")
    cB_SH: float = Field(0.10, description="Cost for B in Stag Hunt")
    cA_PD: float = Field(0.20, description="Cost for A in Prisoner's Dilemma")
    cB_PD: float = Field(0.08, description="Cost for B in Prisoner's Dilemma")
    alpha: float = Field(0.60, description="Hawk-Dove aggression")
    iterations: int = Field(20, description="Number of iterations")
    m: int = Field(3, description="Minimum degree")
    seed: Optional[int] = Field(None, description="Random seed")


class SimulationRequest(BaseModel):
    config: SimulationConfig
    async_mode: bool = Field(False, description="Run asynchronously")


class SimulationResponse(BaseModel):
    simulation_id: str
    status: str  # "pending", "running", "completed", "failed"
    config: SimulationConfig
    results: Optional[Dict[str, Any]] = None
    created_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None


class BatchSimulationRequest(BaseModel):
    configs: List[SimulationConfig]
    parallel: bool = Field(True, description="Run in parallel")


# Endpoints

@app.get("/")
def root():
    return {
        "service": "BTUT API",
        "version": "1.0.0",
        "endpoints": {
            "simulate": "/api/simulate",
            "get_result": "/api/simulate/{id}",
            "batch": "/api/simulate/batch",
            "presets": "/api/presets",
            "benchmark": "/api/benchmark"
        }
    }


@app.get("/api/presets")
def get_presets():
    """Get available preset configurations"""
    return {
        "presets": {
            name: {k: v for k, v in config.items()}
            for name, config in PRESETS.items()
        }
    }


@app.post("/api/simulate", response_model=SimulationResponse)
async def run_simulation(
    request: SimulationRequest,
    background_tasks: BackgroundTasks
):
    """
    Run a simulation
    
    - **config**: Simulation parameters
    - **async_mode**: If True, returns immediately with pending status
    """
    
    sim_id = str(uuid.uuid4())
    
    # Create simulation record
    sim_record = {
        "simulation_id": sim_id,
        "status": "pending",
        "config": request.config.dict(),
        "results": None,
        "created_at": datetime.utcnow().isoformat(),
        "completed_at": None,
        "error": None
    }
    
    simulations_db[sim_id] = sim_record
    
    if request.async_mode:
        # Run in background
        background_tasks.add_task(execute_simulation, sim_id, request.config)
        return SimulationResponse(**sim_record)
    else:
        # Run synchronously
        await execute_simulation(sim_id, request.config)
        return SimulationResponse(**simulations_db[sim_id])


async def execute_simulation(sim_id: str, config: SimulationConfig):
    """Execute simulation and store results"""
    
    simulations_db[sim_id]["status"] = "running"
    
    try:
        # Run simulation
        config_dict = config.dict()
        simulator = BTUTSimulator(config_dict)
        states = simulator.run()
        
        # Extract results
        final_state = states[-1]
        convergence_history = [s['fractionA'] for s in states]
        
        results = {
            "final_fraction_a": final_state['fractionA'],
            "convergence_history": convergence_history,
            "iterations_completed": len(states),
            "converged": final_state['fractionA'] > 0.99,
            "agent_count": config_dict['N']
        }
        
        # Update record
        simulations_db[sim_id].update({
            "status": "completed",
            "results": results,
            "completed_at": datetime.utcnow().isoformat()
        })
    
    except Exception as e:
        simulations_db[sim_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.utcnow().isoformat()
        })


@app.get("/api/simulate/{simulation_id}", response_model=SimulationResponse)
def get_simulation(simulation_id: str):
    """Get simulation results by ID"""
    
    if simulation_id not in simulations_db:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    return SimulationResponse(**simulations_db[simulation_id])


@app.post("/api/simulate/batch")
async def batch_simulate(request: BatchSimulationRequest):
    """
    Run multiple simulations in batch
    
    - **configs**: List of simulation configurations
    - **parallel**: Run in parallel (faster) or sequential
    """
    
    results = []
    
    if request.parallel:
        # Run in parallel using asyncio
        tasks = [
            execute_simulation(str(uuid.uuid4()), config)
            for config in request.configs
        ]
        await asyncio.gather(*tasks)
    else:
        # Run sequentially
        for config in request.configs:
            sim_id = str(uuid.uuid4())
            await execute_simulation(sim_id, config)
            results.append(simulations_db[sim_id])
    
    return {
        "batch_id": str(uuid.uuid4()),
        "total": len(request.configs),
        "completed": len(results),
        "results": results
    }


@app.post("/api/benchmark")
async def run_benchmark(
    agent_counts: List[int] = [1000, 10000, 100000],
    iterations: int = 20
):
    """
    Run performance benchmark
    
    - **agent_counts**: List of agent counts to test
    - **iterations**: Iterations per simulation
    """
    
    import time
    
    results = []
    
    for N in agent_counts:
        config = {
            **PRESETS['standard'],
            'N': N,
            'iterations': iterations
        }
        
        start = time.time()
        simulator = BTUTSimulator(config)
        simulator.run()
        elapsed = time.time() - start
        
        results.append({
            "agents": N,
            "iterations": iterations,
            "runtime_seconds": elapsed,
            "throughput": (N * iterations) / elapsed
        })
    
    return {
        "benchmark_results": results,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.delete("/api/simulate/{simulation_id}")
def delete_simulation(simulation_id: str):
    """Delete a simulation record"""
    
    if simulation_id not in simulations_db:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    del simulations_db[simulation_id]
    return {"message": "Simulation deleted"}


@app.get("/api/stats")
def get_stats():
    """Get API usage statistics"""
    
    total = len(simulations_db)
    by_status = {}
    
    for sim in simulations_db.values():
        status = sim["status"]
        by_status[status] = by_status.get(status, 0) + 1
    
    return {
        "total_simulations": total,
        "by_status": by_status,
        "timestamp": datetime.utcnow().isoformat()
    }


# Startup message
@app.on_event("startup")
def startup_event():
    print("="*60)
    print("BTUT API Server Started")
    print("="*60)
    print("Documentation: http://localhost:8000/docs")
    print("API Endpoints: http://localhost:8000/")
    print("="*60)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
