/**
 * WASM Loader for BTUT Engine
 * Loads and initializes the Rust WASM module
 */

let wasmModule: any = null;
let isInitialized = false;

export interface WASMSimulationConfig {
  n: number;
  gamma: number;
  tau: number;
  c_a_sh: number;
  c_b_sh: number;
  c_a_pd: number;
  c_b_pd: number;
  alpha: number;
  iterations: number;
  m: number;
  seed: number;
}

export interface SimulationState {
  iteration: number;
  fraction_a: number;
  convergence_history: number[];
  is_complete: boolean;
  agent_count: number;
  strategy_a_count: number;
}

export async function initWASM(): Promise<void> {
  if (isInitialized) return;

  try {
    // Dynamic import of WASM module
    // @ts-ignore - WASM module is generated at build time
    const module = await import('../rust-engine/pkg/btut_wasm');
    await module.default();
    wasmModule = module;
    isInitialized = true;
    console.log('✅ BTUT WASM module initialized');
  } catch (error) {
    console.error('Failed to load WASM module:', error);
    throw new Error('WASM initialization failed. Make sure to run build-wasm.sh first.');
  }
}

export function isWASMReady(): boolean {
  return isInitialized;
}

export function createSimulator(config: WASMSimulationConfig): any {
  if (!isInitialized || !wasmModule) {
    throw new Error('WASM module not initialized. Call initWASM() first.');
  }

  const wasmConfig = new wasmModule.SimulationConfig(
    config.n,
    config.gamma,
    config.tau,
    config.c_a_sh,
    config.c_b_sh,
    config.c_a_pd,
    config.c_b_pd,
    config.alpha,
    config.iterations,
    config.m,
    config.seed
  );

  return new wasmModule.BTUTSimulator(wasmConfig);
}

export function getPreset(name: 'quick' | 'standard' | 'massive'): WASMSimulationConfig {
  if (!isInitialized || !wasmModule) {
    throw new Error('WASM module not initialized');
  }

  let config;
  switch (name) {
    case 'quick':
      config = wasmModule.SimulationConfig.quick_preset();
      break;
    case 'standard':
      config = wasmModule.SimulationConfig.standard_preset();
      break;
    case 'massive':
      config = wasmModule.SimulationConfig.massive_preset();
      break;
    default:
      config = wasmModule.SimulationConfig.standard_preset();
  }

  return {
    n: config.n,
    gamma: config.gamma,
    tau: config.tau,
    c_a_sh: config.c_a_sh,
    c_b_sh: config.c_b_sh,
    c_a_pd: config.c_a_pd,
    c_b_pd: config.c_b_pd,
    alpha: config.alpha,
    iterations: config.iterations,
    m: config.m,
    seed: config.seed
  };
}

export async function runBenchmark(maxN: number, step: number): Promise<any> {
  if (!isInitialized || !wasmModule) {
    throw new Error('WASM module not initialized');
  }

  return wasmModule.benchmark_scaling(maxN, step);
}

export { wasmModule };
