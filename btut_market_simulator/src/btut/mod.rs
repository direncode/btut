//! BTUT (Bivariate Trajectory-Undercurrent Theory) Kernel Module
//!
//! This module implements the core BTUT algorithm for multi-agent coordination.
//!
//! ## Theory Overview
//!
//! BTUT achieves O(N) complexity by using kernel-weighted mean-field dynamics instead
//! of explicit pairwise agent interactions. Each agent's strategy is influenced by:
//!
//! 1. **Global Mean Field**: Weighted average of all agent strategies
//! 2. **Local Kernel**: Influence decay based on agent "distance" (size, type, book position)
//! 3. **Momentum**: Persistence of past strategies
//!
//! ## Algorithm
//!
//! For each BTUT iteration:
//! ```text
//! 1. Compute global cooperation level: C = Σᵢ wᵢ · sᵢ / Σᵢ wᵢ
//! 2. For each agent i:
//!    a. Compute kernel weight: Kᵢ = exp(-λ · dᵢ)
//!    b. Compute target: tᵢ = C + bias + noise
//!    c. Update strategy: sᵢ' = α · sᵢ + (1-α) · Kᵢ · tᵢ
//! 3. Check convergence: max|sᵢ' - sᵢ| < threshold
//! ```
//!
//! ## Strategies
//!
//! - **Cooperate** (s → 1): Provide liquidity, tighten spreads, stabilize market
//! - **Defect** (s → 0): Front-run, widen spreads, spoof orders, destabilize

mod kernel;

pub use kernel::{BTUTKernel, CooperationByType, KernelConfig, ShockType, Strategy, StrategyUpdate};
