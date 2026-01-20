//! BTUT Market Simulator - Main Entry Point
//!
//! A planetary-scale multi-agent financial market simulator using
//! Bivariate Trajectory-Undercurrent Theory (BTUT) as the core coordination engine.
//!
//! # Usage
//!
//! ```bash
//! # Run with default configuration
//! btut_market_simulator
//!
//! # Run with custom config file
//! btut_market_simulator --config config.toml
//!
//! # Run preset configurations
//! btut_market_simulator --preset million     # 1M agents
//! btut_market_simulator --preset shock       # Liquidity shock experiment
//! btut_market_simulator --preset test        # Quick test
//!
//! # Run baseline comparison
//! btut_market_simulator --compare
//!
//! # Generate default config file
//! btut_market_simulator --generate-config config.toml
//! ```

use anyhow::{Context, Result};
use btut_market_simulator::{
    config::SimulationConfig,
    simulation::{run_baseline_comparison, Simulation},
    utils::{init_logging, PlotHelper},
};
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
use tracing::info;

/// BTUT Market Simulator - Planetary-scale multi-agent financial simulation
#[derive(Parser, Debug)]
#[command(name = "btut_market_simulator")]
#[command(author = "BTUT Research Team")]
#[command(version = btut_market_simulator::VERSION)]
#[command(about = "Planetary-scale multi-agent financial market simulator using BTUT")]
#[command(long_about = r#"
BTUT Market Simulator

A high-performance multi-agent financial market simulator using Bivariate
Trajectory-Undercurrent Theory (BTUT) as the core coordination engine.

Key Features:
  • O(N) complexity via kernel-weighted mean-field dynamics
  • Scalable to 1M+ agents on single machine
  • Emergent flash crashes and liquidity spirals
  • Heterogeneous agents (Retail, HFT, Institution, MarketMaker)

Examples:
  # Run default simulation
  btut_market_simulator run

  # Run 1M agent preset
  btut_market_simulator run --preset million

  # Run with custom config
  btut_market_simulator run --config my_config.toml

  # Run liquidity shock experiment with comparison
  btut_market_simulator run --preset shock --compare

  # Generate config file
  btut_market_simulator generate-config output.toml

  # Show preset configurations
  btut_market_simulator list-presets
"#)]
struct Cli {
    /// Subcommand to execute
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run a simulation
    Run {
        /// Path to configuration file
        #[arg(short, long)]
        config: Option<PathBuf>,

        /// Use a preset configuration
        #[arg(short, long, value_enum)]
        preset: Option<Preset>,

        /// Number of agents (overrides config)
        #[arg(short = 'n', long)]
        agents: Option<usize>,

        /// Number of timesteps (overrides config)
        #[arg(short, long)]
        timesteps: Option<usize>,

        /// Random seed
        #[arg(short, long)]
        seed: Option<u64>,

        /// Output directory
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Run baseline comparison (BTUT vs zero-intelligence)
        #[arg(long)]
        compare: bool,

        /// Inject liquidity shock at timestep
        #[arg(long)]
        shock_at: Option<u64>,

        /// Shock severity (0.0 to 1.0)
        #[arg(long, default_value = "0.5")]
        shock_severity: f64,

        /// Disable progress bar
        #[arg(long)]
        no_progress: bool,

        /// Enable verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Generate a configuration file
    GenerateConfig {
        /// Output file path
        output: PathBuf,

        /// Preset to use as base
        #[arg(short, long, value_enum)]
        preset: Option<Preset>,
    },

    /// List available presets
    ListPresets,

    /// Show simulation info
    Info {
        /// Config file to analyze
        #[arg(short, long)]
        config: Option<PathBuf>,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Preset {
    /// Default configuration (100k agents, 1000 timesteps)
    Default,
    /// Million agent simulation
    Million,
    /// Quick test configuration
    Test,
    /// Liquidity shock experiment
    Shock,
}

impl Preset {
    fn to_config(self) -> SimulationConfig {
        match self {
            Preset::Default => SimulationConfig::default(),
            Preset::Million => SimulationConfig::million_agent_preset(),
            Preset::Test => SimulationConfig::test_preset(),
            Preset::Shock => SimulationConfig::liquidity_shock_preset(),
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Run {
            config,
            preset,
            agents,
            timesteps,
            seed,
            output,
            compare,
            shock_at,
            shock_severity,
            no_progress,
            verbose,
        } => {
            run_simulation(
                config,
                preset,
                agents,
                timesteps,
                seed,
                output,
                compare,
                shock_at,
                shock_severity,
                no_progress,
                verbose,
            )
        }
        Commands::GenerateConfig { output, preset } => generate_config(output, preset),
        Commands::ListPresets => list_presets(),
        Commands::Info { config } => show_info(config),
    }
}

fn run_simulation(
    config_path: Option<PathBuf>,
    preset: Option<Preset>,
    agents: Option<usize>,
    timesteps: Option<usize>,
    seed: Option<u64>,
    output_dir: Option<PathBuf>,
    compare: bool,
    shock_at: Option<u64>,
    shock_severity: f64,
    no_progress: bool,
    verbose: bool,
) -> Result<()> {
    // Load or create configuration
    let mut config = if let Some(path) = config_path {
        SimulationConfig::from_file(&path)
            .with_context(|| format!("Failed to load config from {:?}", path))?
    } else if let Some(p) = preset {
        p.to_config()
    } else {
        SimulationConfig::default()
    };

    // Apply overrides
    if let Some(n) = agents {
        config.simulation.num_agents = n;
    }
    if let Some(t) = timesteps {
        config.simulation.num_timesteps = t;
    }
    if let Some(s) = seed {
        config.simulation.seed = Some(s);
    }
    if let Some(ref dir) = output_dir {
        config.output.output_dir = dir.to_string_lossy().to_string();
    }
    if no_progress {
        config.output.show_progress = false;
    }
    if verbose {
        config.output.log_level = btut_market_simulator::config::LogLevelConfig::Debug;
    }

    // Validate config
    config.validate().context("Invalid configuration")?;

    // Initialize logging
    init_logging(&config);

    // Create output directory
    std::fs::create_dir_all(&config.output.output_dir)
        .context("Failed to create output directory")?;

    info!(
        agents = config.simulation.num_agents,
        timesteps = config.simulation.num_timesteps,
        "Starting BTUT Market Simulator"
    );

    if compare {
        // Run baseline comparison
        let shock_time = shock_at.unwrap_or(config.simulation.num_timesteps as u64 / 2);
        let (btut_results, zi_results) = run_baseline_comparison(&config, shock_time);

        // Print comparison results
        println!("\n{}", "=".repeat(60));
        println!("BASELINE COMPARISON: BTUT vs Zero-Intelligence");
        println!("{}", "=".repeat(60));

        println!("\n--- BTUT Agents ---");
        println!("{}", btut_results.summary);

        println!("\n--- Zero-Intelligence Agents ---");
        println!("{}", zi_results.summary);

        println!("\n--- Analysis ---");
        let btut_recovery = btut_results
            .crashes
            .iter()
            .filter(|c| c.recovered)
            .count();
        let zi_recovery = zi_results.crashes.iter().filter(|c| c.recovered).count();

        println!(
            "BTUT Recovery Rate: {:.1}%",
            if btut_results.crashes.is_empty() {
                100.0
            } else {
                btut_recovery as f64 / btut_results.crashes.len() as f64 * 100.0
            }
        );
        println!(
            "ZI Recovery Rate: {:.1}%",
            if zi_results.crashes.is_empty() {
                100.0
            } else {
                zi_recovery as f64 / zi_results.crashes.len() as f64 * 100.0
            }
        );

        if btut_results.summary.mean_regeneration > zi_results.summary.mean_regeneration {
            println!(
                "\nBTUT shows {:.1}% higher regeneration score",
                (btut_results.summary.mean_regeneration / zi_results.summary.mean_regeneration
                    - 1.0)
                    * 100.0
            );
        }

        // Save results
        let btut_path = format!("{}/btut_results.json", config.output.output_dir);
        let zi_path = format!("{}/zi_results.json", config.output.output_dir);

        btut_results.save_json(&btut_path)?;
        zi_results.save_json(&zi_path)?;

        btut_results.save_metrics_csv(&format!("{}/btut_metrics.csv", config.output.output_dir))?;
        zi_results.save_metrics_csv(&format!("{}/zi_metrics.csv", config.output.output_dir))?;

        println!("\nResults saved to {}/ directory", config.output.output_dir);

        // ASCII plots for comparison
        let btut_prices: Vec<f64> = btut_results
            .metrics
            .iter()
            .filter_map(|m| m.mid_price)
            .collect();
        let zi_prices: Vec<f64> = zi_results
            .metrics
            .iter()
            .filter_map(|m| m.mid_price)
            .collect();

        println!("\n{}", PlotHelper::plot_comparison(
            &btut_prices,
            &zi_prices,
            "Price Comparison",
            "BTUT",
            "ZI",
            60,
            15
        ));

        let btut_coop: Vec<f64> = btut_results.metrics.iter().map(|m| m.mean_cooperation).collect();
        let zi_coop: Vec<f64> = zi_results.metrics.iter().map(|m| m.mean_cooperation).collect();

        println!("{}", PlotHelper::plot_comparison(
            &btut_coop,
            &zi_coop,
            "Cooperation Level",
            "BTUT",
            "ZI",
            60,
            10
        ));
    } else {
        // Run single simulation
        let mut sim = Simulation::new(config.clone());

        // Schedule shock if requested
        if let Some(t) = shock_at {
            sim.schedule_liquidity_crisis(t, shock_severity);
        }

        let results = sim.run();

        // Print summary
        println!("\n{}", "=".repeat(60));
        println!("SIMULATION COMPLETE");
        println!("{}", "=".repeat(60));
        println!("{}", results.summary);

        // Print crash info
        if !results.crashes.is_empty() {
            println!("\n--- Detected Flash Crashes ---");
            for (i, crash) in results.crashes.iter().enumerate() {
                println!(
                    "Crash {}: t={}-{:?}, drawdown={:.2}%, spread={:.1}bps, recovered={}",
                    i + 1,
                    crash.start_timestep,
                    crash.end_timestep,
                    crash.max_drawdown_pct,
                    crash.peak_spread_bps,
                    crash.recovered
                );
            }
        }

        // Save results
        let results_path = format!("{}/results.json", config.output.output_dir);
        let metrics_path = format!("{}/metrics.csv", config.output.output_dir);

        results.save_json(&results_path)?;
        results.save_metrics_csv(&metrics_path)?;

        println!("\nResults saved to {}/ directory", config.output.output_dir);

        // ASCII plots
        let prices: Vec<f64> = results.metrics.iter().filter_map(|m| m.mid_price).collect();
        if !prices.is_empty() {
            println!("\n{}", PlotHelper::plot_ascii(&prices, "Price", 60, 15));
        }

        let coop: Vec<f64> = results.metrics.iter().map(|m| m.mean_cooperation).collect();
        println!("{}", PlotHelper::plot_ascii(&coop, "Cooperation Level", 60, 10));

        let regen: Vec<f64> = results.metrics.iter().map(|m| m.regeneration_score).collect();
        println!("{}", PlotHelper::plot_ascii(&regen, "Regeneration Score", 60, 10));
    }

    Ok(())
}

fn generate_config(output: PathBuf, preset: Option<Preset>) -> Result<()> {
    let config = preset.map_or_else(SimulationConfig::default, |p| p.to_config());

    config
        .save(&output)
        .with_context(|| format!("Failed to save config to {:?}", output))?;

    println!("Configuration saved to {:?}", output);
    println!("\nYou can edit this file and run with:");
    println!("  btut_market_simulator run --config {:?}", output);

    Ok(())
}

fn list_presets() -> Result<()> {
    println!("Available Presets:");
    println!("==================\n");

    println!("default");
    println!("  Agents: 100,000");
    println!("  Timesteps: 1,000");
    println!("  Description: Balanced configuration for general use\n");

    println!("million");
    println!("  Agents: 1,000,000");
    println!("  Timesteps: 1,000");
    println!("  Description: Large-scale simulation, optimized for performance\n");

    println!("test");
    println!("  Agents: 1,000");
    println!("  Timesteps: 100");
    println!("  Description: Quick test configuration for development\n");

    println!("shock");
    println!("  Agents: 100,000");
    println!("  Timesteps: 500");
    println!("  Description: Liquidity shock experiment with tuned parameters\n");

    println!("Usage: btut_market_simulator run --preset <PRESET>");

    Ok(())
}

fn show_info(config_path: Option<PathBuf>) -> Result<()> {
    println!("BTUT Market Simulator v{}", btut_market_simulator::VERSION);
    println!("================================\n");

    if let Some(path) = config_path {
        let config = SimulationConfig::from_file(&path)
            .with_context(|| format!("Failed to load config from {:?}", path))?;

        println!("Configuration: {:?}\n", path);
        println!("Simulation:");
        println!("  Agents: {}", config.simulation.num_agents);
        println!("  Timesteps: {}", config.simulation.num_timesteps);
        println!("  Seed: {:?}", config.simulation.seed);
        println!("  Parallel: {}", config.simulation.parallel);

        println!("\nBTUT Kernel:");
        println!("  Kernel Decay: {}", config.btut.kernel_decay);
        println!("  Momentum: {}", config.btut.momentum_strength);
        println!("  Iterations: {}", config.btut.convergence_iterations);
        println!("  Temperature: {}", config.btut.temperature);

        println!("\nMarket:");
        println!("  Initial Price: {}", config.market.initial_price);
        println!("  Tick Size: {}", config.market.tick_size);
        println!("  Max Book Depth: {}", config.market.max_book_depth);

        println!("\nAgents:");
        println!("  Retail: {:.1}%", config.agents.retail_fraction * 100.0);
        println!("  HFT: {:.1}%", config.agents.hft_fraction * 100.0);
        println!(
            "  Institution: {:.1}%",
            config.agents.institution_fraction * 100.0
        );
        println!(
            "  Market Maker: {:.1}%",
            config.agents.market_maker_fraction * 100.0
        );

        // Estimate memory usage
        let mem_per_agent = 128; // bytes, approximate
        let total_mem_mb = config.simulation.num_agents * mem_per_agent / 1_000_000;
        println!("\nEstimated Memory: ~{}MB", total_mem_mb);

        // Estimate runtime
        let base_time_per_step_us = 50; // microseconds per timestep per 1000 agents
        let estimated_time_s = (config.simulation.num_agents as f64 / 1000.0)
            * (config.simulation.num_timesteps as f64)
            * (base_time_per_step_us as f64)
            / 1_000_000.0;
        println!("Estimated Runtime: ~{:.1}s", estimated_time_s);
    } else {
        println!("No config file specified. Showing defaults.\n");

        let config = SimulationConfig::default();
        println!("Default Configuration:");
        println!("  Agents: {}", config.simulation.num_agents);
        println!("  Timesteps: {}", config.simulation.num_timesteps);
        println!("  Kernel Decay: {}", config.btut.kernel_decay);
        println!("  Momentum: {}", config.btut.momentum_strength);

        println!("\nUse 'btut_market_simulator list-presets' to see available presets");
        println!("Use 'btut_market_simulator generate-config <file>' to create a config file");
    }

    Ok(())
}
