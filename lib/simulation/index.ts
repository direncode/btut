/**
 * Unified BTUT simulation interface
 * Automatically selects WASM or TypeScript implementation
 */

import { BTUTSimulator as TypeScriptSimulator, PRESETS } from './btut-engine';
import { BTUTWasmSimulator, initWasm as initWasmInternal, isWasmAvailable } from './btut-wasm-wrapper';
import type { SimulationConfig, SimulationState, Agent } from './btut-engine';

export type { SimulationConfig, SimulationState, Agent };
export { PRESETS };

let wasmInitialized = false;

/**
 * Initialize WASM module
 * Call this once at app startup
 */
export async function initWasm(): Promise<boolean> {
  if (wasmInitialized) {
    return true;
  }

  try {
    const success = await initWasmInternal();
    wasmInitialized = success;
    return success;
  } catch (error) {
    console.warn('WASM initialization failed:', error);
    return false;
  }
}

/**
 * Check if WASM is available
 */
export { isWasmAvailable };

/**
 * Unified simulator that automatically uses WASM if available,
 * falls back to TypeScript implementation otherwise
 */
export class BTUTUnifiedSimulator {
  private simulator: TypeScriptSimulator | BTUTWasmSimulator;
  private usingWasm: boolean;

  constructor(config: SimulationConfig) {
    this.usingWasm = isWasmAvailable();

    if (this.usingWasm) {
      console.log('Using WASM implementation');
      this.simulator = new BTUTWasmSimulator(config);
    } else {
      console.log('Using TypeScript implementation');
      this.simulator = new TypeScriptSimulator(config);
    }
  }

  public step(): SimulationState {
    return this.simulator.step();
  }

  public run(): any {
    return this.simulator.run();
  }

  public getState(): SimulationState {
    return this.simulator.getState();
  }

  public reset(): void {
    this.simulator.reset();
  }

  public isUsingWasm(): boolean {
    return this.usingWasm;
  }

  public getEngineType(): 'wasm' | 'typescript' | 'loading' {
    return this.usingWasm ? 'wasm' : 'typescript';
  }
}

// Re-export individual simulators for explicit use
export { BTUTSimulator as TypeScriptSimulator } from './btut-engine';
export { BTUTWasmSimulator } from './btut-wasm-wrapper';
