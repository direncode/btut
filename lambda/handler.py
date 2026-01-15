"""
AWS Lambda Handler for BTUT Simulations

Serverless execution of BTUT simulations via AWS Lambda.
"""

import json
import sys
import os

# Add BTUT to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python-sdk'))

from btut import Simulator
import traceback


def lambda_handler(event, context):
    """
    AWS Lambda handler function

    Event format:
    {
        "body": "{\"agents\": 10000, \"gamma\": 1.5, \"tau\": 0.3, \"alpha\": 0.1}"
    }

    Response format:
    {
        "statusCode": 200,
        "headers": {...},
        "body": "{\"final_cooperation\": 0.6, ...}"
    }
    """

    try:
        # Parse request
        if 'body' in event:
            if isinstance(event['body'], str):
                body = json.loads(event['body'])
            else:
                body = event['body']
        else:
            body = event

        # Extract parameters with defaults
        agents = body.get('agents', 10000)
        gamma = body.get('gamma', 1.5)
        tau = body.get('tau', 0.3)
        alpha = body.get('alpha', 0.1)
        iterations = body.get('iterations', 100)

        # Validate parameters
        if agents < 1 or agents > 1000000:
            return error_response(400, "agents must be between 1 and 1,000,000")

        if gamma < 1.0 or gamma > 10.0:
            return error_response(400, "gamma must be between 1.0 and 10.0")

        if tau < 0.0 or tau >= 1.0:
            return error_response(400, "tau must be between 0.0 and 1.0")

        if alpha <= 0.0 or alpha > 10.0:
            return error_response(400, "alpha must be between 0.0 and 10.0")

        # Create simulator
        sim = Simulator(
            agents=agents,
            gamma=gamma,
            tau=tau,
            alpha=alpha,
            iterations=iterations
        )

        # Run simulation
        result = sim.run()

        # Prepare response
        response_body = {
            'success': True,
            'config': {
                'agents': agents,
                'gamma': gamma,
                'tau': tau,
                'alpha': alpha,
                'iterations': iterations
            },
            'results': {
                'final_cooperation': result.final_cooperation,
                'converged': result.converged,
                'iterations_completed': result.iterations_completed,
                'runtime_seconds': result.runtime_seconds,
                'convergence_history': result.convergence_history
            },
            'request_id': context.request_id if context else 'local',
            'function_version': context.function_version if context else '1.0'
        }

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            'body': json.dumps(response_body)
        }

    except json.JSONDecodeError as e:
        return error_response(400, f"Invalid JSON: {str(e)}")

    except Exception as e:
        print(f"Error in lambda_handler: {str(e)}")
        print(traceback.format_exc())
        return error_response(500, f"Internal error: {str(e)}")


def error_response(status_code: int, message: str):
    """Create error response"""
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps({
            'success': False,
            'error': message
        })
    }


def health_check(event, context):
    """Health check endpoint"""
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps({
            'status': 'healthy',
            'version': '1.0.0',
            'service': 'btut-lambda'
        })
    }


def batch_handler(event, context):
    """
    Handle batch simulations

    Event format:
    {
        "simulations": [
            {"agents": 10000, "gamma": 1.5},
            {"agents": 50000, "gamma": 2.0},
            ...
        ]
    }
    """

    try:
        if 'body' in event:
            body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
        else:
            body = event

        simulations = body.get('simulations', [])

        if not simulations:
            return error_response(400, "No simulations provided")

        if len(simulations) > 100:
            return error_response(400, "Maximum 100 simulations per batch")

        results = []

        for i, sim_config in enumerate(simulations):
            try:
                # Run individual simulation
                sim = Simulator(
                    agents=sim_config.get('agents', 10000),
                    gamma=sim_config.get('gamma', 1.5),
                    tau=sim_config.get('tau', 0.3),
                    alpha=sim_config.get('alpha', 0.1)
                )

                result = sim.run()

                results.append({
                    'index': i,
                    'success': True,
                    'config': sim_config,
                    'final_cooperation': result.final_cooperation,
                    'converged': result.converged,
                    'iterations': result.iterations_completed
                })

            except Exception as e:
                results.append({
                    'index': i,
                    'success': False,
                    'config': sim_config,
                    'error': str(e)
                })

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'success': True,
                'num_simulations': len(simulations),
                'results': results
            })
        }

    except Exception as e:
        print(f"Error in batch_handler: {str(e)}")
        print(traceback.format_exc())
        return error_response(500, f"Batch processing error: {str(e)}")


# For local testing
if __name__ == '__main__':
    # Test event
    test_event = {
        'body': json.dumps({
            'agents': 10000,
            'gamma': 1.5,
            'tau': 0.3,
            'alpha': 0.1
        })
    }

    # Mock context
    class MockContext:
        request_id = 'test-request-123'
        function_version = '$LATEST'

    result = lambda_handler(test_event, MockContext())
    print(json.dumps(json.loads(result['body']), indent=2))
