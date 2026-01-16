/**
 * Lambda deployment utilities
 */

import { AWSLambdaClient, type DeploymentOptions } from './aws-client'
import JSZip from 'jszip'

export interface DeploymentConfig {
  functionName: string
  region: string
  runtime: 'python3.11' | 'python3.10' | 'python3.9'
  handler: string
  timeout?: number
  memorySize?: number
  role?: string
  environment?: Record<string, string>
}

export interface DeploymentResult {
  success: boolean
  functionArn?: string
  error?: string
  deploymentTime: number
}

/**
 * Package Lambda function code
 */
export async function packageLambdaCode(
  handlerCode: string,
  dependencies: Record<string, string> = {}
): Promise<Buffer> {
  const zip = new JSZip()

  // Add handler code
  zip.file('handler.py', handlerCode)

  // Add requirements.txt
  const requirements = Object.entries(dependencies)
    .map(([pkg, version]) => `${pkg}==${version}`)
    .join('\n')

  if (requirements) {
    zip.file('requirements.txt', requirements)
  }

  // Generate zip
  const buffer = await zip.generateAsync({
    type: 'nodebuffer',
    compression: 'DEFLATE',
    compressionOptions: { level: 9 }
  })

  return buffer
}

/**
 * Deploy BTUT Lambda function
 */
export async function deployBTUTLambda(
  client: AWSLambdaClient,
  config: DeploymentConfig
): Promise<DeploymentResult> {
  const startTime = Date.now()

  try {
    // Read handler code from integration file
    const handlerCode = await fetch('/api/lambda-handler').then(r => r.text())

    // Package code
    const codeZip = await packageLambdaCode(handlerCode, {
      'numpy': '1.24.0',
      'btut-sdk': '0.1.0'
    })

    // Deploy to Lambda
    const options: DeploymentOptions = {
      functionName: config.functionName,
      runtime: config.runtime,
      handler: config.handler || 'handler.lambda_handler',
      role: config.role || 'arn:aws:iam::123456789012:role/lambda-execution-role',
      code: codeZip,
      timeout: config.timeout || 300,
      memorySize: config.memorySize || 512,
      environment: config.environment || {
        'BTUT_VERSION': '1.0.0'
      }
    }

    const result = await client.deploy(options)

    return {
      success: true,
      functionArn: result.arn,
      deploymentTime: Date.now() - startTime
    }
  } catch (error: any) {
    return {
      success: false,
      error: error.message,
      deploymentTime: Date.now() - startTime
    }
  }
}

/**
 * Get Lambda handler code template
 */
export function getLambdaHandlerTemplate(): string {
  return `"""
BTUT Lambda Handler
Handles simulation requests via AWS Lambda
"""

import json
import numpy as np
from btut import Simulator

def lambda_handler(event, context):
    """
    Lambda function handler for BTUT simulations

    Event payload:
    {
        "type": "simulation" | "batch",
        "agents": int,
        "gamma": float,
        "tau": float,
        ...other params
    }
    """

    try:
        # Parse request
        body = json.loads(event.get('body', '{}')) if isinstance(event.get('body'), str) else event

        request_type = body.get('type', 'simulation')

        if request_type == 'batch':
            return handle_batch(body, context)
        else:
            return handle_simulation(body, context)

    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'success': False,
                'error': str(e)
            })
        }

def handle_simulation(body, context):
    """Handle single simulation request"""

    # Extract parameters
    agents = body.get('agents', 10000)
    gamma = body.get('gamma', 1.5)
    tau = body.get('tau', 0.3)
    iterations = body.get('iterations', 20)

    # Create simulator
    sim = Simulator(
        agents=agents,
        gamma=gamma,
        tau=tau,
        iterations=iterations
    )

    # Run simulation
    result = sim.run()

    # Return results
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps({
            'success': True,
            'results': {
                'finalFractionA': float(result['fraction_a']),
                'convergenceTime': float(result['convergence_time']),
                'iterations': int(result['iterations']),
                'executionTime': context.get_remaining_time_in_millis() / 1000
            }
        }, cls=NumpyEncoder)
    }

def handle_batch(body, context):
    """Handle batch simulation requests"""

    requests = body.get('requests', [])
    results = []

    for req in requests:
        try:
            agents = req.get('agents', 10000)
            gamma = req.get('gamma', 1.5)
            tau = req.get('tau', 0.3)

            sim = Simulator(agents=agents, gamma=gamma, tau=tau)
            result = sim.run()

            results.append({
                'success': True,
                'params': {'agents': agents, 'gamma': gamma, 'tau': tau},
                'results': result
            })
        except Exception as e:
            results.append({
                'success': False,
                'params': req,
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
            'results': results
        }, cls=NumpyEncoder)
    }

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for NumPy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
`
}

/**
 * Validate deployment configuration
 */
export function validateDeploymentConfig(config: DeploymentConfig): string[] {
  const errors: string[] = []

  if (!config.functionName) {
    errors.push('Function name is required')
  }

  if (!config.region) {
    errors.push('AWS region is required')
  }

  if (!config.runtime) {
    errors.push('Runtime is required')
  }

  if (!config.role) {
    errors.push('IAM role ARN is required')
  }

  return errors
}

/**
 * Estimate deployment cost
 */
export function estimateDeploymentCost(
  invocationsPerMonth: number,
  avgDurationMs: number,
  memoryMB: number
): number {
  // Lambda pricing
  const requestCost = (invocationsPerMonth / 1000000) * 0.20 // $0.20 per 1M requests
  const computeCost = (invocationsPerMonth * (avgDurationMs / 1000) * (memoryMB / 1024)) * 0.0000166667

  return requestCost + computeCost
}
