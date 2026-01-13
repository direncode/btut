# BTUT Platform - Complete Deployment Guide

## Quick Links
- [API Server](#api-server-deployment)
- [AWS Lambda](#aws-lambda-deployment)
- [Docker](#docker-deployment)
- [ROS Integration](#ros-integration-setup)
- [Python SDK](#python-sdk-usage)
- [Benchmarking](#running-benchmarks)

---

## API Server Deployment

### Requirements
```bash
pip install fastapi uvicorn pydantic boto3 psutil
```

### Start Server
```bash
cd api
python server.py

# Or with production WSGI server
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4
```

### API Endpoints
- POST `/api/simulate` - Run simulation
- GET `/api/simulate/{id}` - Get results
- POST `/api/simulate/batch` - Batch simulations
- GET `/api/presets` - List presets
- POST `/api/benchmark` - Run benchmark

### Test API
```bash
curl -X POST http://localhost:8000/api/simulate \
  -H "Content-Type: application/json" \
  -d '{"config": {"N": 10000, "gamma": 1.45}}'
```

---

## AWS Lambda Deployment

### 1. Package Function
```bash
cd cloud/lambda
pip install -r requirements.txt -t package/
cp lambda_function.py package/
cd package && zip -r ../btut-lambda.zip .
```

### 2. Deploy
```bash
aws lambda create-function \
  --function-name btut-simulator \
  --runtime python3.11 \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://btut-lambda.zip \
  --memory-size 3008 \
  --timeout 300
```

### 3. Test
```bash
aws lambda invoke \
  --function-name btut-simulator \
  --payload '{"action":"simulate","preset":"quick"}' \
  response.json
```

---

## Docker Deployment

### Web App
```bash
docker build -t btut-platform .
docker run -p 3000:3000 btut-platform
```

### API Server
```bash
docker build -f Dockerfile.api -t btut-api .
docker run -p 8000:8000 btut-api
```

### Full Stack (docker-compose)
```bash
docker-compose up -d
```

---

## ROS Integration Setup

### Install
```bash
cd ~/catkin_ws/src
git clone [repo]
catkin_make
source devel/setup.bash
```

### Launch
```bash
roslaunch btut_ros coordinator.launch num_robots:=10
```

---

## Python SDK Usage

### Install
```bash
pip install btut-sdk
```

### Basic Usage
```python
from btut import Simulator

sim = Simulator(agents=10000, gamma=1.45)
results = sim.run()
print(f"Cooperation: {results.final_cooperation}")
```

---

## Running Benchmarks

```bash
cd benchmarks
python benchmark_suite.py
```

Generates:
- `benchmark_results.csv`
- `benchmark_results.png`

---

## Environment Variables

### .env.local (Web)
```
NEXT_PUBLIC_API_URL=https://api.btut.ai
```

### .env (API)
```
DATABASE_URL=postgresql://...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

---

## Support
- Docs: https://docs.btut.ai
- Issues: https://github.com/btut/platform/issues
