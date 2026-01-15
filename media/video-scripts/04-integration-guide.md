# BTUT Integration Guide (8 minutes)

**Target Audience:** Developers integrating BTUT into existing systems
**Goal:** Show how to use the API, deploy at scale, and integrate with other tools

---

## SCENE 1: Introduction (0:00-0:30)

**Visual:** Split screen showing BTUT connecting to various systems

**Narration:**
> "BTUT isn't just a standalone tool—it's designed to integrate seamlessly with your existing infrastructure. In this guide, we'll cover the REST API, Python SDK integration patterns, deployment options, and connecting BTUT to popular frameworks like ROS and SUMO."

**On-screen text:**
- Integration Options:
  - 🌐 REST API
  - 🐍 Python SDK embedding
  - ☁️ Cloud deployment
  - 🤖 ROS/SUMO integration

---

## SCENE 2: REST API Basics (0:30-2:00)

**Visual:** API documentation page, then Postman/cURL

**Narration:**
> "The BTUT REST API lets any application run simulations via HTTP. No Python required."

### API Overview

**Terminal:**
```bash
# Health check
curl https://btut-api.fly.dev/health

# Response:
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": "24h 15m"
}
```

### Running a Simulation

**Code:**
```bash
# Run simulation via POST
curl -X POST https://btut-api.fly.dev/api/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "N": 10000,
      "gamma": 1.5,
      "tau": 0.3,
      "alpha": 0.1
    },
    "async_mode": false
  }'
```

**Response:**
```json
{
  "simulation_id": "sim_abc123",
  "config": {...},
  "results": {
    "final_cooperation": 0.6000,
    "converged": true,
    "iterations_completed": 19,
    "runtime_seconds": 0.124
  },
  "status": "completed"
}
```

**Narration:**
> "The API returns complete results including convergence status and runtime metrics."

### Async Mode

**Code:**
```javascript
// JavaScript example: Async simulation
const response = await fetch('https://btut-api.fly.dev/api/simulate', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    config: {N: 100000, gamma: 1.5},
    async_mode: true
  })
});

const {simulation_id} = await response.json();

// Poll for results
const checkStatus = async () => {
  const status = await fetch(
    `https://btut-api.fly.dev/api/simulate/${simulation_id}`
  );
  return await status.json();
};

// Wait for completion
let result;
while (true) {
  result = await checkStatus();
  if (result.status === 'completed') break;
  await new Promise(r => setTimeout(r, 1000));
}

console.log('Final cooperation:', result.results.final_cooperation);
```

**Narration:**
> "For large simulations, use async mode. Submit the job, get a simulation ID, then poll for results. Perfect for web applications."

---

## SCENE 3: Python SDK Integration (2:00-3:30)

**Visual:** Python application code

**Narration:**
> "Embed BTUT directly into your Python applications."

### Flask Web App

**Code:**
```python
from flask import Flask, request, jsonify
from btut import Simulator

app = Flask(__name__)

@app.route('/api/coordination', methods=['POST'])
def analyze_coordination():
    """Endpoint to analyze coordination dynamics"""
    data = request.json

    # Extract parameters
    agent_count = data.get('agents', 10000)
    bonus = data.get('cooperation_bonus', 1.5)

    # Run simulation
    sim = Simulator(agents=agent_count, gamma=bonus)
    result = sim.run()

    # Return analysis
    return jsonify({
        'equilibrium_cooperation': result.final_cooperation,
        'is_stable': result.converged,
        'convergence_iterations': result.iterations_completed,
        'recommendation': 'cooperative' if result.final_cooperation > 0.55 else 'competitive'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**Terminal:**
```bash
# Test the endpoint
curl -X POST http://localhost:5000/api/coordination \
  -H "Content-Type: application/json" \
  -d '{"agents": 50000, "cooperation_bonus": 2.0}'

# Response:
{
  "equilibrium_cooperation": 0.6667,
  "is_stable": true,
  "convergence_iterations": 18,
  "recommendation": "cooperative"
}
```

**Narration:**
> "This Flask app exposes BTUT as a microservice. Great for integrating coordination analysis into larger systems."

### Background Processing with Celery

**Code:**
```python
from celery import Celery
from btut import Simulator

app = Celery('tasks', broker='redis://localhost:6379')

@app.task
def run_large_simulation(agent_count, gamma):
    """Async task for large simulations"""
    sim = Simulator(agents=agent_count, gamma=gamma)
    result = sim.run()

    return {
        'final_cooperation': result.final_cooperation,
        'iterations': result.iterations_completed,
        'converged': result.converged
    }

# Usage:
# result = run_large_simulation.delay(1000000, 1.5)
# result.get()  # Block until complete
```

**Narration:**
> "For very large simulations, use Celery for background processing. Submit jobs, continue working, and retrieve results when ready."

---

## SCENE 4: Cloud Deployment (3:30-5:00)

**Visual:** Deployment architecture diagram

**Narration:**
> "Let's deploy BTUT to production on AWS, Google Cloud, or your own infrastructure."

### Docker Deployment

**Code:**
```dockerfile
# Dockerfile.btut
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ ./app/
COPY btut/ ./btut/

# Expose port
EXPOSE 8000

# Run with gunicorn
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "-b", "0.0.0.0:8000"]
```

**Terminal:**
```bash
# Build image
docker build -t btut-api:latest -f Dockerfile.btut .

# Run locally
docker run -p 8000:8000 btut-api:latest

# Push to registry
docker tag btut-api:latest myregistry.azurecr.io/btut-api:latest
docker push myregistry.azurecr.io/btut-api:latest
```

**Narration:**
> "Containerize BTUT for consistent deployment across environments."

### Kubernetes Deployment

**Code:**
```yaml
# btut-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: btut-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: btut-api
  template:
    metadata:
      labels:
        app: btut-api
    spec:
      containers:
      - name: btut-api
        image: myregistry.azurecr.io/btut-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        env:
        - name: MAX_AGENTS
          value: "1000000"
---
apiVersion: v1
kind: Service
metadata:
  name: btut-api-service
spec:
  selector:
    app: btut-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

**Terminal:**
```bash
# Deploy to Kubernetes
kubectl apply -f btut-deployment.yaml

# Check status
kubectl get pods -l app=btut-api

# Get service URL
kubectl get service btut-api-service
```

**Narration:**
> "Kubernetes provides auto-scaling, load balancing, and high availability for production workloads."

### AWS Lambda (Serverless)

**Code:**
```python
# lambda_handler.py
import json
from btut import Simulator

def lambda_handler(event, context):
    """AWS Lambda handler for BTUT simulations"""

    # Parse request
    body = json.loads(event.get('body', '{}'))
    agent_count = body.get('agents', 10000)
    gamma = body.get('gamma', 1.5)

    # Run simulation
    sim = Simulator(agents=agent_count, gamma=gamma)
    result = sim.run()

    # Return response
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps({
            'final_cooperation': result.final_cooperation,
            'converged': result.converged,
            'iterations': result.iterations_completed
        })
    }
```

**serverless.yml:**
```yaml
service: btut-lambda

provider:
  name: aws
  runtime: python3.11
  region: us-east-1
  memorySize: 3008
  timeout: 300

functions:
  simulate:
    handler: lambda_handler.lambda_handler
    events:
      - http:
          path: simulate
          method: post
          cors: true

package:
  exclude:
    - node_modules/**
    - .git/**
```

**Terminal:**
```bash
# Deploy with Serverless Framework
serverless deploy

# Test
curl -X POST https://abc123.execute-api.us-east-1.amazonaws.com/dev/simulate \
  -d '{"agents": 50000, "gamma": 1.8}'
```

**Narration:**
> "Lambda provides serverless execution—pay only for actual simulation time, with automatic scaling."

---

## SCENE 5: ROS Integration (5:00-6:15)

**Visual:** ROS node graph, robot simulation

**Narration:**
> "Integrate BTUT with ROS for robot swarm coordination."

### ROS Node

**Code:**
```python
#!/usr/bin/env python3
import rospy
from btut import Simulator
from btut_ros.msg import AgentState, SimulationConfig
from btut_ros.srv import RunSimulation, RunSimulationResponse

class BTUTNode:
    def __init__(self):
        rospy.init_node('btut_coordination')

        # Publisher for agent states
        self.state_pub = rospy.Publisher(
            '/btut/agent_states',
            AgentState,
            queue_size=10
        )

        # Service for running simulations
        self.sim_service = rospy.Service(
            '/btut/simulate',
            RunSimulation,
            self.handle_simulation
        )

        rospy.loginfo("BTUT node ready")

    def handle_simulation(self, req):
        """Handle simulation request"""
        rospy.loginfo(f"Running simulation with {req.num_agents} agents")

        # Run BTUT simulation
        sim = Simulator(
            agents=req.num_agents,
            gamma=req.gamma,
            tau=req.tau
        )
        result = sim.run()

        # Publish result
        msg = AgentState()
        msg.cooperation_rate = result.final_cooperation
        msg.converged = result.converged
        self.state_pub.publish(msg)

        # Return response
        return RunSimulationResponse(
            success=True,
            final_cooperation=result.final_cooperation,
            iterations=result.iterations_completed
        )

    def spin(self):
        rospy.spin()

if __name__ == '__main__':
    node = BTUTNode()
    node.spin()
```

**Terminal:**
```bash
# Launch ROS node
rosrun btut_ros btut_node.py

# Call service from another node
rosservice call /btut/simulate "num_agents: 1000
gamma: 1.5
tau: 0.3"

# Response:
success: True
final_cooperation: 0.6
iterations: 19
```

**Narration:**
> "The ROS node exposes BTUT as a service. Robot controllers can query coordination strategies in real-time."

---

## SCENE 6: SUMO Traffic Integration (6:15-7:15)

**Visual:** SUMO traffic simulation with BTUT coordination

**Narration:**
> "Coordinate traffic using BTUT and SUMO."

**Code:**
```python
import traci
from btut import Simulator

class TrafficCoordinator:
    def __init__(self, sumo_config):
        # Start SUMO
        traci.start(["sumo-gui", "-c", sumo_config])

        # Initialize BTUT
        self.sim = Simulator(agents=1000, gamma=1.5)

    def coordinate_vehicles(self):
        """Update vehicle behaviors based on BTUT coordination"""

        # Get all vehicles
        vehicle_ids = traci.vehicle.getIDList()
        num_vehicles = len(vehicle_ids)

        if num_vehicles == 0:
            return

        # Run coordination simulation
        self.sim = Simulator(agents=num_vehicles, gamma=1.5)
        result = self.sim.run()

        # Apply coordination strategy
        cooperative_fraction = result.final_cooperation

        for i, vid in enumerate(vehicle_ids):
            # Assign strategy based on cooperation rate
            is_cooperative = (i / num_vehicles) < cooperative_fraction

            if is_cooperative:
                # Cooperative: maintain safe speed
                traci.vehicle.setSpeed(vid, 13.89)  # 50 km/h
            else:
                # Competitive: more aggressive
                traci.vehicle.setSpeed(vid, 16.67)  # 60 km/h

    def run(self, steps=1000):
        """Run coordinated traffic simulation"""
        for step in range(steps):
            traci.simulationStep()

            # Update coordination every 10 steps
            if step % 10 == 0:
                self.coordinate_vehicles()

        traci.close()

# Usage
coordinator = TrafficCoordinator("traffic.sumocfg")
coordinator.run(steps=3600)  # 1 hour of traffic
```

**Visual:** SUMO visualization showing coordinated vs uncoordinated traffic

**Narration:**
> "BTUT determines optimal coordination levels, which SUMO applies to vehicle behaviors. The result: smoother traffic flow and fewer collisions."

---

## SCENE 7: Best Practices (7:15-7:45)

**Visual:** Checklist with code snippets

**Narration:**
> "Five integration best practices:"

**1. Handle Errors Gracefully**
```python
from btut import Simulator, SimulationError

try:
    sim = Simulator(agents=N, gamma=gamma)
    result = sim.run()
except SimulationError as e:
    log.error(f"Simulation failed: {e}")
    return fallback_strategy()
```

**2. Cache Results**
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_equilibrium(agents, gamma):
    sim = Simulator(agents=agents, gamma=gamma)
    return sim.run().final_cooperation
```

**3. Monitor Performance**
```python
import time

start = time.time()
result = sim.run()
duration = time.time() - start

metrics.record('btut_runtime', duration)
metrics.record('btut_iterations', result.iterations_completed)
```

**4. Use Environment Variables**
```python
import os

API_URL = os.getenv('BTUT_API_URL', 'http://localhost:8000')
MAX_AGENTS = int(os.getenv('BTUT_MAX_AGENTS', 1000000))
```

**5. Version Your Configurations**
```json
{
  "btut_version": "1.0.0",
  "simulation_config": {
    "agents": 100000,
    "gamma": 1.5,
    "tau": 0.3
  },
  "created_at": "2025-01-14T12:00:00Z"
}
```

---

## SCENE 8: Wrap-Up (7:45-8:00)

**Visual:** Instructor on camera with integration diagram

**Narration:**
> "You now know how to integrate BTUT into any system—via REST API, embedded Python, cloud deployment, or robotics frameworks. Check the documentation for more integration examples and templates. Happy integrating!"

**End card:**
- 📖 Integration docs: btut.ai/docs/integrations
- 🔧 Example repos: github.com/direncode/btut/examples
- 💬 Get help: github.com/direncode/btut/discussions

---

## Production Notes

### Code Examples
- Tested on multiple platforms
- Include error handling
- Realistic parameters
- Clear comments

### Visuals
- Architecture diagrams
- API request/response flows
- Deployment pipelines
- Live integrations

### Demonstrations
- Real ROS robot coordination
- SUMO traffic simulation
- Cloud deployment walkthrough
- API testing with Postman

### Tools
- Postman for API demos
- Docker Desktop
- Kubernetes (minikube OK)
- ROS (melodic/noetic)
- SUMO traffic simulator

### Accessibility
- Code narration
- Visual descriptions
- Alternative examples
- Transcripts
