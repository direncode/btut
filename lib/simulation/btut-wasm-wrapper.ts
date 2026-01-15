/**
 * WASM wrapper for BTUT simulation
 * Provides unified interface between TypeScript and Rust WASM implementations
 */

import type { SimulationConfig, SimulationState } from './btut-engine';

let wasmModule: any = null;
let isInitialized = false;

export async function initWasm(): Promise<boolean> {
  if (isInitialized) {
    return true;
  }

  try {
    // Dynamically import WASM module
    // @ts-ignore - WASM module is generated at build time
    const module = await import('../../rust-engine/pkg/btut_wasm');
    await module.default();
    wasmModule = module;
    isInitialized = true;
    return true;
  } catch (error) {
    console.warn('WASM initialization failed, falling back to TypeScript:', error);
    return false;
  }
}

export function isWasmAvailable(): boolean {
  return isInitialized && wasmModule !== null;
}

export class BTUTWasmSimulator {
  private wasmSim: any;
  private config: SimulationConfig;

  constructor(config: SimulationConfig) {
    if (!isWasmAvailable()) {
      throw new Error('WASM not initialized. Call initWasm() first.');
    }

    this.config = config;

    // Create WASM config object
    const wasmConfig = new wasmModule.SimulationConfig(
      config.N,
      config.gamma,
      config.tau,
      config.cA_SH,
      config.cB_SH,
      config.cA_PD,
      config.cB_PD,
      config.alpha,
      config.iterations,
      config.m,
      config.seed
    );

    this.wasmSim = new wasmModule.BTUTSimulator(wasmConfig);
  }

  public step(): SimulationState {
    const wasmState = this.wasmSim.step();
    return this.convertWasmState(wasmState);
  }

  public run(): SimulationState {
    const wasmState = this.wasmSim.run();
    return this.convertWasmState(wasmState);
  }

  public runComplete(): SimulationState {
    const wasmState = this.wasmSim.run_complete();
    return this.convertWasmState(wasmState);
  }

  public getState(): SimulationState {
    const wasmState = this.wasmSim.get_state();
    return this.convertWasmState(wasmState);
  }

  public reset(): void {
    this.wasmSim.reset();
  }

  private convertWasmState(wasmState: any): SimulationState {
    // Convert WASM state to TypeScript format
    return {
      iteration: wasmState.iteration || 0,
      fractionA: wasmState.fraction_a || 0,
      convergenceHistory: wasmState.convergence_history || [],
      agents: wasmState.agents || [],
      isComplete: wasmState.is_complete || false,
      agentCount: wasmState.agent_count || 0,
      strategyACount: wasmState.strategy_a_count || 0,
    };
  }
}
