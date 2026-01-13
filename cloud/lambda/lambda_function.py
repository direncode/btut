"""
BTUT AWS Lambda Deployment
===========================
Run BTUT simulations serverlessly at massive scale using AWS Lambda.

Features:
- Auto-scaling to 1000+ concurrent simulations
- Pay-per-use pricing
- No server management
- Global deployment

Architecture:
    API Gateway -> Lambda -> S3 (results storage)
"""

import json
import boto3
import time
from typing import Dict, Any
import os

# Import BTUT (packaged in Lambda layer)
from btut_engine import BTUTSimulator, PRESETS

# AWS clients
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

# Configuration
RESULTS_BUCKET = os.environ.get('RESULTS_BUCKET', 'btut-results')
SIMULATIONS_TABLE = os.environ.get('SIMULATIONS_TABLE', 'btut-simulations')


def lambda_handler(event, context):
    """
    Main Lambda handler for BTUT simulations
    
    Event format:
    {
        "action": "simulate" | "get_result" | "benchmark",
        "config": {...},  # For simulate action
        "simulation_id": "..."  # For get_result action
    }
    """
    
    print(f"Received event: {json.dumps(event)}")
    
    action = event.get('action', 'simulate')
    
    if action == 'simulate':
        return run_simulation(event)
    elif action == 'get_result':
        return get_simulation_result(event)
    elif action == 'benchmark':
        return run_benchmark(event)
    else:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': f'Unknown action: {action}'})
        }


def run_simulation(event: Dict) -> Dict:
    """Run a BTUT simulation"""
    
    # Extract config
    config = event.get('config', {})
    
    # Apply defaults from preset if specified
    preset = event.get('preset', None)
    if preset and preset in PRESETS:
        config = {**PRESETS[preset], **config}
    
    # Generate simulation ID
    simulation_id = f"sim-{int(time.time() * 1000)}"
    
    try:
        # Run simulation
        start_time = time.time()
        
        simulator = BTUTSimulator(config)
        states = simulator.run()
        
        runtime = time.time() - start_time
        
        # Extract results
        final_state = states[-1]
        results = {
            'simulation_id': simulation_id,
            'config': config,
            'final_fraction_a': final_state['fractionA'],
            'convergence_history': final_state['convergenceHistory'],
            'iterations_completed': len(states),
            'converged': final_state['fractionA'] > 0.99,
            'agent_count': config.get('N', 10000),
            'runtime_seconds': runtime,
            'timestamp': time.time()
        }
        
        # Store in S3
        s3_key = f"simulations/{simulation_id}.json"
        s3_client.put_object(
            Bucket=RESULTS_BUCKET,
            Key=s3_key,
            Body=json.dumps(results),
            ContentType='application/json'
        )
        
        # Store metadata in DynamoDB
        table = dynamodb.Table(SIMULATIONS_TABLE)
        table.put_item(Item={
            'simulation_id': simulation_id,
            'timestamp': int(time.time()),
            'agent_count': config.get('N', 10000),
            'final_cooperation': float(final_state['fractionA']),
            'runtime_seconds': float(runtime),
            's3_key': s3_key
        })
        
        # Return results
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'simulation_id': simulation_id,
                'results': results,
                's3_url': f"s3://{RESULTS_BUCKET}/{s3_key}"
            })
        }
    
    except Exception as e:
        print(f"Simulation error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'simulation_id': simulation_id
            })
        }


def get_simulation_result(event: Dict) -> Dict:
    """Retrieve simulation results from S3"""
    
    simulation_id = event.get('simulation_id')
    
    if not simulation_id:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'simulation_id required'})
        }
    
    try:
        # Get from S3
        s3_key = f"simulations/{simulation_id}.json"
        response = s3_client.get_object(
            Bucket=RESULTS_BUCKET,
            Key=s3_key
        )
        
        results = json.loads(response['Body'].read())
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(results)
        }
    
    except s3_client.exceptions.NoSuchKey:
        return {
            'statusCode': 404,
            'body': json.dumps({'error': 'Simulation not found'})
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }


def run_benchmark(event: Dict) -> Dict:
    """Run performance benchmark"""
    
    agent_counts = event.get('agent_counts', [1000, 10000, 100000])
    iterations = event.get('iterations', 20)
    
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
            'agents': N,
            'iterations': iterations,
            'runtime_seconds': elapsed,
            'throughput': (N * iterations) / elapsed
        })
    
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps({
            'benchmark_results': results,
            'lambda_memory_mb': int(context.memory_limit_in_mb),
            'lambda_timeout_seconds': int(context.get_remaining_time_in_millis() / 1000)
        })
    }


# For local testing
if __name__ == '__main__':
    test_event = {
        'action': 'simulate',
        'preset': 'quick',
        'config': {
            'gamma': 1.5
        }
    }
    
    class MockContext:
        memory_limit_in_mb = 3008
        def get_remaining_time_in_millis(self):
            return 300000
    
    result = lambda_handler(test_event, MockContext())
    print(json.dumps(result, indent=2))
