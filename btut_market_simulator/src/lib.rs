//! # BTUT Market Simulator
//!
//! A planetary-scale multi-agent financial market simulator using
//! **Bivariate Trajectory-Undercurrent Theory (BTUT)** as the core coordination engine.
//!
//! ## Key Properties
//!
//! - **O(N) Complexity**: No explicit graph storage; uses kernel-weighted mean-field dynamics
//! - **Kernel-Weighted Dynamics**: Influence decays with "distance" in order-book depth/size
//! - **Momentum-Based Strategies**: Cooperate (provide liquidity) vs Defect (front-run/spoof)
//! - **Fast Convergence**: 20-30 iterations even at 1M+ agents
//! - **Heterogeneous Agents**: Retail, HFT, Institutions, Market Makers
//! - **Emergent Behaviors**: Flash crashes, liquidity spirals, regeneration, drag
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                        Simulation Loop                          │
//! │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐│
//! │  │  Agents  │──│   BTUT   │──│  Market  │──│     Metrics      ││
//! │  │ (1M+)    │  │  Kernel  │  │OrderBook │  │    Collector     ││
//! │  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘│
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use btut_market_simulator::{Simulation, SimulationConfig};
//!
//! let config = SimulationConfig::default();
//! let mut sim = Simulation::new(config);
//! let results = sim.run();
//! ```
//!
//! ## Scaling
//!
//! | Agents | Single Machine | Cluster (Future) |
//! |--------|----------------|------------------|
//! | 1M     | < 60s          | N/A              |
//! | 10M    | ~10min         | < 60s            |
//! | 1B     | N/A            | < 10min          |

#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]
#![deny(unsafe_code)]

pub mod agent;
pub mod btut;
pub mod config;
pub mod market;
pub mod metrics;
pub mod simulation;
pub mod utils;

// Re-exports for convenient access
pub use agent::{Agent, AgentId, AgentState, AgentType};
pub use btut::{BTUTKernel, KernelConfig, Strategy};
pub use config::SimulationConfig;
pub use market::{Order, OrderBook, OrderId, OrderSide, OrderType, Price, Quantity};
pub use metrics::{CrashEvent, MarketMetrics, MetricsCollector};
pub use simulation::{Simulation, SimulationResults, SimulationState};
pub use utils::{LogLevel, SyntheticGenerator};

/// Crate version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default number of BTUT convergence iterations
pub const DEFAULT_BTUT_ITERATIONS: usize = 25;

/// Default kernel decay parameter (controls influence falloff)
pub const DEFAULT_KERNEL_DECAY: f64 = 0.1;

/// Default momentum strength for strategy updates
pub const DEFAULT_MOMENTUM_STRENGTH: f64 = 0.7;

/// Prelude module for common imports
pub mod prelude {
    //! Convenient re-exports for common usage patterns.
    //!
    //! ```rust
    //! use btut_market_simulator::prelude::*;
    //! ```

    pub use crate::agent::{Agent, AgentId, AgentState, AgentType};
    pub use crate::btut::{BTUTKernel, KernelConfig, Strategy};
    pub use crate::config::SimulationConfig;
    pub use crate::market::{Order, OrderBook, OrderId, OrderSide, OrderType, Price, Quantity};
    pub use crate::metrics::{CrashEvent, MarketMetrics, MetricsCollector};
    pub use crate::simulation::{Simulation, SimulationResults};
}
