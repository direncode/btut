# BTUT AWS Lambda Deployment

Serverless deployment of BTUT simulations on AWS Lambda.

## Overview

Run BTUT simulations serverlessly on AWS Lambda with automatic scaling, pay-per-execution pricing, and zero server management.

## Features

- **Serverless**: No servers to manage
- **Auto-scaling**: Handles 1 to 10,000+ concurrent requests
- **Cost-effective**: Pay only for execution time
- **Low latency**: Global deployment via CloudFront
- **Batch processing**: Run multiple simulations in parallel

## Architecture

```
Client → API Gateway → Lambda Function → BTUT Engine → Response
```

**Benefits:**
- No cold start optimization needed (BTUT is fast)
- Linear scaling with concurrent requests
- Built-in monitoring via CloudWatch
- Automatic retries and error handling

## Prerequisites

- AWS Account
- AWS CLI configured
- Node.js 16+ (for Serverless Framework)
- Python 3.11+
- Docker (for packaging dependencies)

## Installation

### 1. Install Serverless Framework

```bash
npm install -g serverless
```

### 2. Install Plugins

```bash
cd lambda
npm install --save-dev serverless-python-requirements serverless-offline
```

### 3. Configure AWS Credentials

```bash
aws configure
# Enter your AWS Access Key ID, Secret Access Key, and region
```

## Deployment

### Deploy to AWS

```bash
# Deploy to development
serverless deploy --stage dev

# Deploy to production
serverless deploy --stage prod --region us-east-1

# Deploy to specific region
serverless deploy --stage prod --region eu-west-1
```

**Output:**
```
Service deployed to stack btut-lambda-dev

endpoints:
  POST - https://abc123.execute-api.us-east-1.amazonaws.com/simulate
  GET - https://abc123.execute-api.us-east-1.amazonaws.com/health
  POST - https://abc123.execute-api.us-east-1.amazonaws.com/batch

functions:
  simulate: btut-lambda-dev-simulate
  health: btut-lambda-dev-health
  batch: btut-lambda-dev-batch
```

### Test Deployment

```bash
# Health check
curl https://YOUR_API_ENDPOINT/health

# Run simulation
curl -X POST https://YOUR_API_ENDPOINT/simulate \
  -H "Content-Type: application/json" \
  -d '{"agents": 10000, "gamma": 1.5}'
```

## API Endpoints

### POST /simulate

Run a single simulation.

**Request:**
```json
{
  "agents": 10000,
  "gamma": 1.5,
  "tau": 0.3,
  "alpha": 0.1,
  "iterations": 100
}
```

**Response:**
```json
{
  "success": true,
  "config": {
    "agents": 10000,
    "gamma": 1.5,
    "tau": 0.3,
    "alpha": 0.1,
    "iterations": 100
  },
  "results": {
    "final_cooperation": 0.6000,
    "converged": true,
    "iterations_completed": 19,
    "runtime_seconds": 0.124,
    "convergence_history": [0.5, 0.55, ..., 0.6]
  },
  "request_id": "abc-123",
  "function_version": "$LATEST"
}
```

**Example (curl):**
```bash
curl -X POST https://YOUR_API_ENDPOINT/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "agents": 50000,
    "gamma": 2.0,
    "tau": 0.4
  }'
```

**Example (Python):**
```python
import requests

response = requests.post(
    'https://YOUR_API_ENDPOINT/simulate',
    json={
        'agents': 50000,
        'gamma': 2.0,
        'tau': 0.4
    }
)

result = response.json()
print(f"Cooperation: {result['results']['final_cooperation']:.2%}")
```

**Example (JavaScript):**
```javascript
const response = await fetch('https://YOUR_API_ENDPOINT/simulate', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    agents: 50000,
    gamma: 2.0,
    tau: 0.4
  })
});

const result = await response.json();
console.log(`Cooperation: ${result.results.final_cooperation}`);
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "service": "btut-lambda"
}
```

### POST /batch

Run multiple simulations in batch.

**Request:**
```json
{
  "simulations": [
    {"agents": 10000, "gamma": 1.2},
    {"agents": 10000, "gamma": 1.5},
    {"agents": 10000, "gamma": 2.0}
  ]
}
```

**Response:**
```json
{
  "success": true,
  "num_simulations": 3,
  "results": [
    {
      "index": 0,
      "success": true,
      "config": {"agents": 10000, "gamma": 1.2},
      "final_cooperation": 0.5455,
      "converged": true,
      "iterations": 18
    },
    ...
  ]
}
```

## Configuration

### Memory and Timeout

Edit `serverless.yml`:

```yaml
provider:
  memorySize: 3008  # Max memory (MB)
  timeout: 300      # Max execution time (seconds)
```

**Memory vs Performance:**
| Memory | vCPUs | Cost | Performance |
|--------|-------|------|-------------|
| 1024 MB | 0.6 | $ | Baseline |
| 2048 MB | 1.0 | $$ | 1.7x faster |
| 3008 MB | 1.8 | $$$ | 2.5x faster |

**Recommendation:** 3008 MB for best price/performance

### Region Selection

Deploy to region closest to users:

```bash
# US East (N. Virginia)
serverless deploy --region us-east-1

# US West (Oregon)
serverless deploy --region us-west-2

# Europe (Ireland)
serverless deploy --region eu-west-1

# Asia Pacific (Tokyo)
serverless deploy --region ap-northeast-1
```

## Cost Estimation

### Pricing (as of 2025)

- **Request cost**: $0.20 per 1M requests
- **Compute cost**: $0.0000166667 per GB-second (arm64)

### Example Scenarios

**Scenario 1: 10,000 agents, ~0.1s runtime**
- Memory: 3008 MB = 2.94 GB
- Compute: 2.94 GB × 0.1s = 0.294 GB-s
- Cost: $0.0000166667 × 0.294 = $0.0000049
- **~$0.005 per simulation**

**Scenario 2: 1M requests/month, 100K agents each**
- Requests: 1M × $0.20/1M = $0.20
- Compute: 1M × 1s × 2.94 GB × $0.0000166667 = $49
- **Total: ~$49.20/month**

**Free Tier:**
- 1M requests/month (forever)
- 400,000 GB-seconds/month (first 12 months)

## Performance

### Benchmarks (on Lambda 3008MB)

| Agents | Runtime | Iterations | Cost per Run |
|--------|---------|-----------|--------------|
| 1,000 | 15ms | 18 | $0.0001 |
| 10,000 | 120ms | 19 | $0.0005 |
| 100,000 | 1.2s | 20 | $0.005 |
| 1,000,000 | 12s | 21 | $0.05 |

### Cold Start

- **First invocation**: ~500ms (includes import time)
- **Warm invocations**: <5ms overhead
- **Mitigation**: Use provisioned concurrency if needed

## Monitoring

### CloudWatch Logs

```bash
# View logs
serverless logs --function simulate --tail

# View logs for specific time
serverless logs --function simulate --startTime 1h
```

### CloudWatch Metrics

Navigate to AWS Console → Lambda → Functions → btut-lambda-prod-simulate → Monitoring

**Key Metrics:**
- Invocations
- Duration
- Errors
- Throttles
- Concurrent executions

### Custom Metrics

Add to `handler.py`:

```python
import boto3
cloudwatch = boto3.client('cloudwatch')

cloudwatch.put_metric_data(
    Namespace='BTUT',
    MetricData=[
        {
            'MetricName': 'AgentsSimulated',
            'Value': agents,
            'Unit': 'Count'
        }
    ]
)
```

## Local Development

### Test Locally

```bash
# Install serverless-offline
npm install --save-dev serverless-offline

# Start local server
serverless offline

# Test endpoint
curl -X POST http://localhost:3000/dev/simulate \
  -H "Content-Type: application/json" \
  -d '{"agents": 1000, "gamma": 1.5}'
```

### Unit Testing

```python
# test_handler.py
import json
from handler import lambda_handler

class MockContext:
    request_id = 'test-123'
    function_version = '$LATEST'

def test_simulation():
    event = {
        'body': json.dumps({
            'agents': 1000,
            'gamma': 1.5
        })
    }

    response = lambda_handler(event, MockContext())

    assert response['statusCode'] == 200
    body = json.loads(response['body'])
    assert body['success'] is True
    assert 'final_cooperation' in body['results']
```

## Security

### API Key Protection

Add API key requirement in `serverless.yml`:

```yaml
provider:
  apiGateway:
    apiKeys:
      - btut-api-key
    usagePlan:
      quota:
        limit: 10000
        period: MONTH
      throttle:
        rateLimit: 100
        burstLimit: 200
```

### Rate Limiting

```yaml
functions:
  simulate:
    events:
      - httpApi:
          path: /simulate
          method: post
          throttling:
            maxRequestsPerSecond: 100
            maxConcurrentRequests: 50
```

### VPC Integration

For secure deployments:

```yaml
provider:
  vpc:
    securityGroupIds:
      - sg-abc123
    subnetIds:
      - subnet-123
      - subnet-456
```

## Troubleshooting

### Deployment Fails

```bash
# Check AWS credentials
aws sts get-caller-identity

# Verbose deployment
serverless deploy --verbose

# Check CloudFormation
aws cloudformation describe-stacks --stack-name btut-lambda-dev
```

### Function Timeout

Increase timeout in `serverless.yml`:

```yaml
functions:
  simulate:
    timeout: 600  # 10 minutes
```

### Memory Issues

Increase memory allocation:

```yaml
provider:
  memorySize: 10240  # Max Lambda memory
```

### Import Errors

```bash
# Rebuild dependencies
rm -rf node_modules .serverless
serverless deploy
```

## Advanced Usage

### Step Functions Integration

Orchestrate complex workflows:

```yaml
# serverless.yml
stepFunctions:
  stateMachines:
    simulationPipeline:
      definition:
        StartAt: RunSimulation
        States:
          RunSimulation:
            Type: Task
            Resource: !GetAtt SimulateLambdaFunction.Arn
            End: true
```

### EventBridge Triggers

Schedule simulations:

```yaml
functions:
  simulate:
    events:
      - eventBridge:
          schedule: rate(1 hour)
          input:
            agents: 100000
            gamma: 1.5
```

### SQS Queue Processing

Process simulation requests from queue:

```yaml
functions:
  simulate:
    events:
      - sqs:
          arn: !GetAtt SimulationQueue.Arn
          batchSize: 10
```

## Cleanup

```bash
# Remove deployment
serverless remove --stage dev

# Or via AWS CLI
aws cloudformation delete-stack --stack-name btut-lambda-dev
```

## Best Practices

1. **Use arm64**: Better price/performance than x86
2. **Optimize memory**: Test different sizes
3. **Enable X-Ray**: For request tracing
4. **Set alarms**: Monitor errors and throttles
5. **Use layers**: Share dependencies across functions
6. **Implement caching**: Store results in DynamoDB/S3
7. **Batch when possible**: Reduce request overhead

## Support

- Documentation: https://btut.ai/docs/integrations/aws-lambda
- Issues: https://github.com/direncode/btut/issues
- Serverless Forum: https://forum.serverless.com

## License

Apache License 2.0 - see LICENSE file for details.
