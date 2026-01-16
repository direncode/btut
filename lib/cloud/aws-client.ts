/**
 * AWS Client for Lambda Deployment
 */

import { LambdaClient, InvokeCommand, ListFunctionsCommand, GetFunctionCommand, UpdateFunctionCodeCommand, CreateFunctionCommand, DeleteFunctionCommand } from '@aws-sdk/client-lambda'
import { S3Client, PutObjectCommand, GetObjectCommand } from '@aws-sdk/client-s3'

export interface AWSCredentials {
  accessKeyId: string
  secretAccessKey: string
  region: string
}

export interface LambdaFunction {
  name: string
  arn: string
  runtime: string
  handler: string
  codeSize: number
  lastModified: string
  timeout: number
  memorySize: number
  environment?: Record<string, string>
}

export interface InvocationResult {
  statusCode: number
  payload: any
  executionTime: number
  billedDuration: number
  memoryUsed: number
  error?: string
}

export interface DeploymentOptions {
  functionName: string
  runtime: string
  handler: string
  role: string
  code: Buffer | string
  timeout?: number
  memorySize?: number
  environment?: Record<string, string>
  layers?: string[]
}

export interface SimulationRequest {
  agents: number
  gamma?: number
  tau?: number
  cA_SH?: number
  cB_SH?: number
  cA_PD?: number
  cB_PD?: number
  alpha?: number
  iterations?: number
  m?: number
  seed?: number
}

export class AWSLambdaClient {
  private lambda: LambdaClient
  private s3: S3Client
  private credentials: AWSCredentials

  constructor(credentials: AWSCredentials) {
    this.credentials = credentials
    this.lambda = new LambdaClient({
      region: credentials.region,
      credentials: {
        accessKeyId: credentials.accessKeyId,
        secretAccessKey: credentials.secretAccessKey
      }
    })
    this.s3 = new S3Client({
      region: credentials.region,
      credentials: {
        accessKeyId: credentials.accessKeyId,
        secretAccessKey: credentials.secretAccessKey
      }
    })
  }

  /**
   * List all Lambda functions
   */
  async listFunctions(): Promise<LambdaFunction[]> {
    const command = new ListFunctionsCommand({})
    const response = await this.lambda.send(command)

    return (response.Functions || []).map(func => ({
      name: func.FunctionName || '',
      arn: func.FunctionArn || '',
      runtime: func.Runtime || '',
      handler: func.Handler || '',
      codeSize: func.CodeSize || 0,
      lastModified: func.LastModified || '',
      timeout: func.Timeout || 0,
      memorySize: func.MemorySize || 0,
      environment: func.Environment?.Variables
    }))
  }

  /**
   * Get function details
   */
  async getFunction(functionName: string): Promise<LambdaFunction | null> {
    try {
      const command = new GetFunctionCommand({ FunctionName: functionName })
      const response = await this.lambda.send(command)

      const config = response.Configuration
      if (!config) return null

      return {
        name: config.FunctionName || '',
        arn: config.FunctionArn || '',
        runtime: config.Runtime || '',
        handler: config.Handler || '',
        codeSize: config.CodeSize || 0,
        lastModified: config.LastModified || '',
        timeout: config.Timeout || 0,
        memorySize: config.MemorySize || 0,
        environment: config.Environment?.Variables
      }
    } catch (error) {
      console.error('Failed to get function:', error)
      return null
    }
  }

  /**
   * Invoke Lambda function
   */
  async invoke(functionName: string, payload: any): Promise<InvocationResult> {
    const startTime = Date.now()

    try {
      const command = new InvokeCommand({
        FunctionName: functionName,
        Payload: JSON.stringify(payload)
      })

      const response = await this.lambda.send(command)
      const executionTime = Date.now() - startTime

      let resultPayload: any = {}
      if (response.Payload) {
        const payloadString = new TextDecoder().decode(response.Payload)
        resultPayload = JSON.parse(payloadString)
      }

      return {
        statusCode: response.StatusCode || 200,
        payload: resultPayload,
        executionTime,
        billedDuration: parseInt(response.LogResult || '0'),
        memoryUsed: 0, // Parse from logs if needed
        error: response.FunctionError
      }
    } catch (error: any) {
      return {
        statusCode: 500,
        payload: null,
        executionTime: Date.now() - startTime,
        billedDuration: 0,
        memoryUsed: 0,
        error: error.message
      }
    }
  }

  /**
   * Run BTUT simulation on Lambda
   */
  async runSimulation(functionName: string, request: SimulationRequest): Promise<any> {
    const result = await this.invoke(functionName, {
      type: 'simulation',
      ...request
    })

    if (result.error) {
      throw new Error(result.error)
    }

    return result.payload
  }

  /**
   * Run batch simulations
   */
  async runBatch(functionName: string, requests: SimulationRequest[]): Promise<any[]> {
    const promises = requests.map(req => this.runSimulation(functionName, req))
    return Promise.all(promises)
  }

  /**
   * Deploy Lambda function
   */
  async deploy(options: DeploymentOptions): Promise<LambdaFunction> {
    const code = typeof options.code === 'string'
      ? Buffer.from(options.code)
      : options.code

    try {
      // Try to create new function
      const command = new CreateFunctionCommand({
        FunctionName: options.functionName,
        Runtime: options.runtime as any,
        Role: options.role,
        Handler: options.handler,
        Code: {
          ZipFile: code
        },
        Timeout: options.timeout || 300,
        MemorySize: options.memorySize || 512,
        Environment: options.environment ? {
          Variables: options.environment
        } : undefined,
        Layers: options.layers
      })

      const response = await this.lambda.send(command)
      const config = response

      return {
        name: config.FunctionName || '',
        arn: config.FunctionArn || '',
        runtime: config.Runtime || '',
        handler: config.Handler || '',
        codeSize: config.CodeSize || 0,
        lastModified: config.LastModified || '',
        timeout: config.Timeout || 0,
        memorySize: config.MemorySize || 0,
        environment: config.Environment?.Variables
      }
    } catch (error: any) {
      if (error.name === 'ResourceConflictException') {
        // Function exists, update it
        return this.updateFunctionCode(options.functionName, code)
      }
      throw error
    }
  }

  /**
   * Update function code
   */
  async updateFunctionCode(functionName: string, code: Buffer): Promise<LambdaFunction> {
    const command = new UpdateFunctionCodeCommand({
      FunctionName: functionName,
      ZipFile: code
    })

    const response = await this.lambda.send(command)

    return {
      name: response.FunctionName || '',
      arn: response.FunctionArn || '',
      runtime: response.Runtime || '',
      handler: response.Handler || '',
      codeSize: response.CodeSize || 0,
      lastModified: response.LastModified || '',
      timeout: response.Timeout || 0,
      memorySize: response.MemorySize || 0
    }
  }

  /**
   * Delete function
   */
  async deleteFunction(functionName: string): Promise<void> {
    const command = new DeleteFunctionCommand({
      FunctionName: functionName
    })
    await this.lambda.send(command)
  }

  /**
   * Upload file to S3
   */
  async uploadToS3(bucket: string, key: string, data: Buffer | string): Promise<string> {
    const command = new PutObjectCommand({
      Bucket: bucket,
      Key: key,
      Body: typeof data === 'string' ? Buffer.from(data) : data
    })

    await this.s3.send(command)
    return `s3://${bucket}/${key}`
  }

  /**
   * Download from S3
   */
  async downloadFromS3(bucket: string, key: string): Promise<Buffer> {
    const command = new GetObjectCommand({
      Bucket: bucket,
      Key: key
    })

    const response = await this.s3.send(command)
    const stream = response.Body as any

    return new Promise((resolve, reject) => {
      const chunks: any[] = []
      stream.on('data', (chunk: any) => chunks.push(chunk))
      stream.on('error', reject)
      stream.on('end', () => resolve(Buffer.concat(chunks)))
    })
  }

  /**
   * Estimate cost for simulation
   */
  estimateCost(agents: number, iterations: number = 20): number {
    // AWS Lambda pricing: $0.0000166667 per GB-second
    // Assume 512MB memory, ~100ms per 1000 agents
    const estimatedTimeSeconds = (agents / 1000) * 0.1
    const memoryGB = 0.5
    const gbSeconds = estimatedTimeSeconds * memoryGB

    // Add request cost: $0.20 per 1M requests
    const requestCost = 0.0000002
    const computeCost = gbSeconds * 0.0000166667

    return requestCost + computeCost
  }

  /**
   * Get invocation count
   */
  async getInvocationCount(functionName: string, period: 'today' | 'week' | 'month'): Promise<number> {
    // This would require CloudWatch integration
    // Simplified version - return mock data
    return 0
  }

  /**
   * Get total cost
   */
  async getTotalCost(functionName: string, period: 'today' | 'week' | 'month'): Promise<number> {
    // This would require Cost Explorer API
    // Simplified version - return mock data
    return 0
  }
}

/**
 * Create AWS Lambda client
 */
export function createAWSClient(credentials: AWSCredentials): AWSLambdaClient {
  return new AWSLambdaClient(credentials)
}
