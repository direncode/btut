//! Agent Module
//!
//! This module defines the various agent types and their behaviors in the market.
//!
//! ## Agent Types
//!
//! - **Retail**: Uninformed, random timing, small size, high latency
//! - **HFT**: Informed, continuous trading, small size, low latency
//! - **Institution**: Informed, patient execution, large size, medium latency
//! - **MarketMaker**: Liquidity provider, two-sided quotes, medium size
//!
//! ## Strategy Influence
//!
//! Agent behavior is influenced by their BTUT cooperation level:
//! - High cooperation (→1): Provide liquidity, tighten spreads, stabilize
//! - Low cooperation (→0): Front-run, widen spreads, spoof, destabilize

use crate::config::AgentConfig;
use crate::market::{Order, OrderBook, OrderSide};
use crate::utils::SimRng;
use serde::{Deserialize, Serialize};

/// Unique agent identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentId(pub u64);

impl AgentId {
    /// Create a new agent ID
    pub fn new(id: u64) -> Self {
        Self(id)
    }
}

/// Agent type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgentType {
    /// Retail trader: uninformed, random, small
    Retail,
    /// High-frequency trader: informed, fast, small
    HFT,
    /// Institutional investor: informed, patient, large
    Institution,
    /// Market maker: liquidity provider, two-sided
    MarketMaker,
}

impl AgentType {
    /// Get base activity probability (how often agent participates)
    pub fn activity_probability(&self) -> f64 {
        match self {
            AgentType::Retail => 0.01, // 1% chance per tick
            AgentType::HFT => 0.50,    // 50% chance per tick
            AgentType::Institution => 0.05, // 5% chance per tick
            AgentType::MarketMaker => 0.80, // 80% chance per tick
        }
    }

    /// Get latency factor (lower = faster reaction to market)
    pub fn latency_factor(&self) -> f64 {
        match self {
            AgentType::Retail => 10.0,
            AgentType::HFT => 0.1,
            AgentType::Institution => 2.0,
            AgentType::MarketMaker => 0.5,
        }
    }

    /// Get base order size multiplier
    pub fn size_multiplier(&self) -> f64 {
        match self {
            AgentType::Retail => 1.0,
            AgentType::HFT => 5.0,
            AgentType::Institution => 100.0,
            AgentType::MarketMaker => 20.0,
        }
    }

    /// Get information quality (0 = noise, 1 = perfect)
    pub fn information_quality(&self) -> f64 {
        match self {
            AgentType::Retail => 0.1,
            AgentType::HFT => 0.7,
            AgentType::Institution => 0.6,
            AgentType::MarketMaker => 0.5,
        }
    }
}

/// Agent state (mutable during simulation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    /// Unique identifier
    pub id: AgentId,
    /// Agent type
    pub agent_type: AgentType,
    /// Available capital
    pub capital: f64,
    /// Current cooperation level (BTUT strategy)
    pub cooperation: f64,
    /// Current position (positive = long, negative = short)
    pub position: f64,
    /// Cumulative P&L
    pub pnl: f64,
}

impl AgentState {
    /// Create a new agent state
    pub fn new(id: AgentId, agent_type: AgentType, capital: f64, cooperation: f64) -> Self {
        Self {
            id,
            agent_type,
            capital,
            cooperation,
            position: 0.0,
            pnl: 0.0,
        }
    }

    /// Update P&L based on price change
    pub fn mark_to_market(&mut self, price_change: f64) {
        self.pnl += self.position * price_change;
    }

    /// Execute a trade (update position and capital)
    pub fn execute_trade(&mut self, quantity: f64, price: f64, is_buy: bool) {
        let signed_qty = if is_buy { quantity } else { -quantity };
        let cost = quantity * price;

        self.position += signed_qty;

        if is_buy {
            self.capital -= cost;
        } else {
            self.capital += cost;
        }
    }

    /// Check if agent has capacity to trade
    pub fn can_trade(&self, price: f64, quantity: f64) -> bool {
        // Can always sell if have position
        if quantity < 0.0 && self.position >= quantity.abs() {
            return true;
        }
        // Need capital to buy
        self.capital >= price * quantity.abs() * 1.1 // 10% margin
    }
}

/// Agent behavior trait
pub trait Agent: Send + Sync {
    /// Get agent state
    fn state(&self) -> &AgentState;

    /// Get mutable agent state
    fn state_mut(&mut self) -> &mut AgentState;

    /// Generate orders for this timestep
    fn generate_orders(
        &mut self,
        book: &mut OrderBook,
        rng: &mut SimRng,
        timestep: u64,
    ) -> Vec<Order>;

    /// Called when one of this agent's orders is filled
    fn on_fill(&mut self, price: f64, quantity: f64, is_buy: bool);

    /// Called at the end of each timestep
    fn on_timestep_end(&mut self, mid_price: Option<f64>);
}

/// Retail agent implementation
#[derive(Debug, Clone)]
pub struct RetailAgent {
    state: AgentState,
    last_order_time: u64,
    order_cooldown: u64,
}

impl RetailAgent {
    /// Create a new retail agent
    pub fn new(id: AgentId, capital: f64, cooperation: f64) -> Self {
        Self {
            state: AgentState::new(id, AgentType::Retail, capital, cooperation),
            last_order_time: 0,
            order_cooldown: 10, // Minimum ticks between orders
        }
    }
}

impl Agent for RetailAgent {
    fn state(&self) -> &AgentState {
        &self.state
    }

    fn state_mut(&mut self) -> &mut AgentState {
        &mut self.state
    }

    fn generate_orders(
        &mut self,
        book: &mut OrderBook,
        rng: &mut SimRng,
        timestep: u64,
    ) -> Vec<Order> {
        // Check cooldown
        if timestep < self.last_order_time + self.order_cooldown {
            return vec![];
        }

        // Activity check
        if !rng.bernoulli(AgentType::Retail.activity_probability()) {
            return vec![];
        }

        let mid_price = book.mid_price().unwrap_or(100.0);
        let spread = book.spread().unwrap_or(0.02);

        // Retail behavior influenced by cooperation:
        // High cooperation: limit orders inside spread (provide liquidity)
        // Low cooperation: aggressive market orders (take liquidity)

        let is_buy = rng.bernoulli(0.5 + self.state.cooperation * 0.1);
        let quantity = rng.log_normal(2.0, 0.5).min(100.0);

        let order = if self.state.cooperation > 0.6 {
            // Cooperative: limit order
            let offset = spread * rng.uniform(0.2, 0.5);
            let price = if is_buy {
                mid_price - offset
            } else {
                mid_price + offset
            };

            Order::limit(
                book.next_order_id(),
                self.state.id,
                if is_buy { OrderSide::Bid } else { OrderSide::Ask },
                price,
                quantity,
                timestep,
            )
        } else {
            // Aggressive: market order
            Order::market(
                book.next_order_id(),
                self.state.id,
                if is_buy { OrderSide::Bid } else { OrderSide::Ask },
                quantity,
                timestep,
            )
        };

        self.last_order_time = timestep;
        vec![order]
    }

    fn on_fill(&mut self, price: f64, quantity: f64, is_buy: bool) {
        self.state.execute_trade(quantity, price, is_buy);
    }

    fn on_timestep_end(&mut self, mid_price: Option<f64>) {
        if let Some(price) = mid_price {
            // Simple mark-to-market
            self.state.pnl = self.state.position * price - self.state.capital;
        }
    }
}

/// HFT agent implementation
#[derive(Debug, Clone)]
pub struct HFTAgent {
    state: AgentState,
    signal: f64,
    inventory_target: f64,
}

impl HFTAgent {
    /// Create a new HFT agent
    pub fn new(id: AgentId, capital: f64, cooperation: f64) -> Self {
        Self {
            state: AgentState::new(id, AgentType::HFT, capital, cooperation),
            signal: 0.0,
            inventory_target: 0.0,
        }
    }

    /// Update signal based on market conditions
    fn update_signal(&mut self, book: &mut OrderBook, rng: &mut SimRng) {
        let imbalance = book.imbalance();
        let noise = rng.normal(0.0, 0.1);
        // Mean-reverting signal with noise
        self.signal = 0.8 * self.signal + 0.2 * (imbalance + noise);
    }
}

impl Agent for HFTAgent {
    fn state(&self) -> &AgentState {
        &self.state
    }

    fn state_mut(&mut self) -> &mut AgentState {
        &mut self.state
    }

    fn generate_orders(
        &mut self,
        book: &mut OrderBook,
        rng: &mut SimRng,
        timestep: u64,
    ) -> Vec<Order> {
        // HFT is highly active
        if !rng.bernoulli(AgentType::HFT.activity_probability()) {
            return vec![];
        }

        self.update_signal(book, rng);

        let mid_price = book.mid_price().unwrap_or(100.0);
        let spread = book.spread().unwrap_or(0.02);

        let mut orders = vec![];

        // HFT behavior influenced by cooperation:
        // High cooperation: provide liquidity, narrow spreads
        // Low cooperation: front-run, take liquidity, widen spreads

        if self.state.cooperation > 0.5 {
            // Market making mode: provide two-sided liquidity
            let half_spread = spread * 0.3 * (2.0 - self.state.cooperation);
            let base_qty = rng.uniform(5.0, 20.0);

            // Bid
            orders.push(Order::limit(
                book.next_order_id(),
                self.state.id,
                OrderSide::Bid,
                mid_price - half_spread,
                base_qty * (1.0 + self.signal.max(0.0)),
                timestep,
            ));

            // Ask
            orders.push(Order::limit(
                book.next_order_id(),
                self.state.id,
                OrderSide::Ask,
                mid_price + half_spread,
                base_qty * (1.0 - self.signal.min(0.0)),
                timestep,
            ));
        } else {
            // Aggressive mode: directional trading
            let direction = if self.signal > 0.1 {
                OrderSide::Bid
            } else if self.signal < -0.1 {
                OrderSide::Ask
            } else {
                return vec![];
            };

            let aggressiveness = 1.0 - self.state.cooperation;
            let quantity = rng.uniform(10.0, 50.0) * aggressiveness;

            orders.push(Order::market(
                book.next_order_id(),
                self.state.id,
                direction,
                quantity,
                timestep,
            ));
        }

        orders
    }

    fn on_fill(&mut self, price: f64, quantity: f64, is_buy: bool) {
        self.state.execute_trade(quantity, price, is_buy);
    }

    fn on_timestep_end(&mut self, _mid_price: Option<f64>) {
        // HFT targets zero inventory
        self.inventory_target = -self.state.position * 0.1;
    }
}

/// Institutional agent implementation
#[derive(Debug, Clone)]
pub struct InstitutionAgent {
    state: AgentState,
    target_position: f64,
    execution_progress: f64,
    parent_order_size: f64,
}

impl InstitutionAgent {
    /// Create a new institutional agent
    pub fn new(id: AgentId, capital: f64, cooperation: f64) -> Self {
        Self {
            state: AgentState::new(id, AgentType::Institution, capital, cooperation),
            target_position: 0.0,
            execution_progress: 1.0,
            parent_order_size: 0.0,
        }
    }

    /// Start a new parent order
    fn initiate_order(&mut self, rng: &mut SimRng) {
        // Large order with random direction
        self.parent_order_size = rng.pareto_truncated(100.0, 1.5, 10000.0);
        self.target_position = if rng.bernoulli(0.5) {
            self.parent_order_size
        } else {
            -self.parent_order_size
        };
        self.execution_progress = 0.0;
    }
}

impl Agent for InstitutionAgent {
    fn state(&self) -> &AgentState {
        &self.state
    }

    fn state_mut(&mut self) -> &mut AgentState {
        &mut self.state
    }

    fn generate_orders(
        &mut self,
        book: &mut OrderBook,
        rng: &mut SimRng,
        timestep: u64,
    ) -> Vec<Order> {
        // Check if we need a new parent order
        if self.execution_progress >= 1.0 {
            if rng.bernoulli(0.01) {
                // 1% chance to start new order
                self.initiate_order(rng);
            } else {
                return vec![];
            }
        }

        // Execute slice of parent order
        let remaining = (self.target_position - self.state.position).abs();
        if remaining < 1.0 {
            self.execution_progress = 1.0;
            return vec![];
        }

        let mid_price = book.mid_price().unwrap_or(100.0);
        let spread = book.spread().unwrap_or(0.02);

        // Slice size based on cooperation (higher = more patient)
        let slice_fraction = 0.02 + 0.08 * (1.0 - self.state.cooperation);
        let slice_size = (remaining * slice_fraction).min(remaining);

        let is_buy = self.target_position > self.state.position;

        // Institution behavior influenced by cooperation:
        // High cooperation: patient limit orders, reduce market impact
        // Low cooperation: aggressive execution, larger slices

        let order = if self.state.cooperation > 0.4 {
            // Patient: limit order behind best price
            let offset = spread * rng.uniform(0.5, 1.5);
            let price = if is_buy {
                mid_price - offset
            } else {
                mid_price + offset
            };

            Order::limit(
                book.next_order_id(),
                self.state.id,
                if is_buy { OrderSide::Bid } else { OrderSide::Ask },
                price,
                slice_size,
                timestep,
            )
        } else {
            // Aggressive: market order
            Order::market(
                book.next_order_id(),
                self.state.id,
                if is_buy { OrderSide::Bid } else { OrderSide::Ask },
                slice_size,
                timestep,
            )
        };

        vec![order]
    }

    fn on_fill(&mut self, price: f64, quantity: f64, is_buy: bool) {
        self.state.execute_trade(quantity, price, is_buy);
        if self.parent_order_size > 0.0 {
            self.execution_progress =
                (self.state.position - self.target_position).abs() / self.parent_order_size;
        }
    }

    fn on_timestep_end(&mut self, _mid_price: Option<f64>) {
        // Nothing special
    }
}

/// Market maker agent implementation
#[derive(Debug, Clone)]
pub struct MarketMakerAgent {
    state: AgentState,
    inventory_limit: f64,
    base_spread: f64,
    quote_size: f64,
}

impl MarketMakerAgent {
    /// Create a new market maker agent
    pub fn new(id: AgentId, capital: f64, cooperation: f64) -> Self {
        Self {
            state: AgentState::new(id, AgentType::MarketMaker, capital, cooperation),
            inventory_limit: capital / 10.0, // 10% of capital max position
            base_spread: 0.02,               // 2 ticks base spread
            quote_size: 100.0,
        }
    }
}

impl Agent for MarketMakerAgent {
    fn state(&self) -> &AgentState {
        &self.state
    }

    fn state_mut(&mut self) -> &mut AgentState {
        &mut self.state
    }

    fn generate_orders(
        &mut self,
        book: &mut OrderBook,
        rng: &mut SimRng,
        timestep: u64,
    ) -> Vec<Order> {
        // Market makers are highly active
        if !rng.bernoulli(AgentType::MarketMaker.activity_probability()) {
            return vec![];
        }

        let mid_price = book.mid_price().unwrap_or(100.0);

        // MM behavior influenced by cooperation:
        // High cooperation: tight spreads, large size, symmetric quotes
        // Low cooperation: wide spreads, small size, biased quotes (spoofing)

        // Spread widens with low cooperation
        let spread_multiplier = 1.0 + 3.0 * (1.0 - self.state.cooperation);
        let half_spread = self.base_spread * spread_multiplier / 2.0;

        // Size increases with cooperation
        let size_multiplier = 0.2 + 0.8 * self.state.cooperation;
        let base_size = self.quote_size * size_multiplier;

        // Inventory skew
        let inventory_ratio = self.state.position / self.inventory_limit.max(1.0);
        let skew = -inventory_ratio * 0.5; // Skew quotes to reduce inventory

        // Low cooperation: asymmetric quotes (potential spoofing)
        let bid_size_mult = if self.state.cooperation < 0.3 {
            rng.uniform(0.1, 0.5) // Small real bid
        } else {
            1.0 + skew.max(0.0)
        };

        let ask_size_mult = if self.state.cooperation < 0.3 {
            rng.uniform(0.1, 0.5) // Small real ask
        } else {
            1.0 - skew.min(0.0)
        };

        let mut orders = vec![];

        // Bid quote
        if self.state.position < self.inventory_limit {
            orders.push(Order::limit(
                book.next_order_id(),
                self.state.id,
                OrderSide::Bid,
                mid_price - half_spread,
                base_size * bid_size_mult,
                timestep,
            ));
        }

        // Ask quote
        if self.state.position > -self.inventory_limit {
            orders.push(Order::limit(
                book.next_order_id(),
                self.state.id,
                OrderSide::Ask,
                mid_price + half_spread,
                base_size * ask_size_mult,
                timestep,
            ));
        }

        orders
    }

    fn on_fill(&mut self, price: f64, quantity: f64, is_buy: bool) {
        self.state.execute_trade(quantity, price, is_buy);
    }

    fn on_timestep_end(&mut self, _mid_price: Option<f64>) {
        // Adjust base spread based on recent volatility (simplified)
        self.base_spread = 0.02; // Could be adaptive
    }
}

/// Factory for creating agents based on configuration
pub struct AgentFactory;

impl AgentFactory {
    /// Create a population of agents based on configuration
    pub fn create_population(config: &AgentConfig, num_agents: usize, mut rng: SimRng) -> Vec<Box<dyn Agent>> {
        let num_retail = (num_agents as f64 * config.retail_fraction) as usize;
        let num_hft = (num_agents as f64 * config.hft_fraction) as usize;
        let num_inst = (num_agents as f64 * config.institution_fraction) as usize;
        let num_mm = num_agents - num_retail - num_hft - num_inst;

        let mut agents: Vec<Box<dyn Agent>> = Vec::with_capacity(num_agents);
        let mut id_counter: u64 = 1;

        // Create retail agents
        for _ in 0..num_retail {
            let cooperation = (config.initial_cooperation
                + rng.normal(0.0, config.cooperation_noise))
            .clamp(0.0, 1.0);
            agents.push(Box::new(RetailAgent::new(
                AgentId(id_counter),
                config.retail_capital,
                cooperation,
            )));
            id_counter += 1;
        }

        // Create HFT agents
        for _ in 0..num_hft {
            let cooperation = (config.initial_cooperation
                + rng.normal(0.0, config.cooperation_noise))
            .clamp(0.0, 1.0);
            agents.push(Box::new(HFTAgent::new(
                AgentId(id_counter),
                config.retail_capital * 10.0,
                cooperation,
            )));
            id_counter += 1;
        }

        // Create institutional agents
        for _ in 0..num_inst {
            let cooperation = (config.initial_cooperation
                + rng.normal(0.0, config.cooperation_noise))
            .clamp(0.0, 1.0);
            agents.push(Box::new(InstitutionAgent::new(
                AgentId(id_counter),
                config.retail_capital * config.institution_capital_multiplier,
                cooperation,
            )));
            id_counter += 1;
        }

        // Create market maker agents
        for _ in 0..num_mm {
            let cooperation = (config.initial_cooperation
                + rng.normal(0.0, config.cooperation_noise * 0.5)) // MMs are more stable
            .clamp(0.0, 1.0);
            agents.push(Box::new(MarketMakerAgent::new(
                AgentId(id_counter),
                config.retail_capital * 50.0,
                cooperation,
            )));
            id_counter += 1;
        }

        agents
    }

    /// Create agent states for BTUT kernel initialization
    pub fn create_agent_states(
        config: &AgentConfig,
        num_agents: usize,
        mut rng: SimRng,
    ) -> Vec<AgentState> {
        let num_retail = (num_agents as f64 * config.retail_fraction) as usize;
        let num_hft = (num_agents as f64 * config.hft_fraction) as usize;
        let num_inst = (num_agents as f64 * config.institution_fraction) as usize;
        let num_mm = num_agents - num_retail - num_hft - num_inst;

        let mut states = Vec::with_capacity(num_agents);
        let mut id_counter: u64 = 1;

        // Retail
        for _ in 0..num_retail {
            let cooperation = (config.initial_cooperation
                + rng.normal(0.0, config.cooperation_noise))
            .clamp(0.0, 1.0);
            states.push(AgentState::new(
                AgentId(id_counter),
                AgentType::Retail,
                config.retail_capital,
                cooperation,
            ));
            id_counter += 1;
        }

        // HFT
        for _ in 0..num_hft {
            let cooperation = (config.initial_cooperation
                + rng.normal(0.0, config.cooperation_noise))
            .clamp(0.0, 1.0);
            states.push(AgentState::new(
                AgentId(id_counter),
                AgentType::HFT,
                config.retail_capital * 10.0,
                cooperation,
            ));
            id_counter += 1;
        }

        // Institution
        for _ in 0..num_inst {
            let cooperation = (config.initial_cooperation
                + rng.normal(0.0, config.cooperation_noise))
            .clamp(0.0, 1.0);
            states.push(AgentState::new(
                AgentId(id_counter),
                AgentType::Institution,
                config.retail_capital * config.institution_capital_multiplier,
                cooperation,
            ));
            id_counter += 1;
        }

        // Market Maker
        for _ in 0..num_mm {
            let cooperation = (config.initial_cooperation
                + rng.normal(0.0, config.cooperation_noise * 0.5))
            .clamp(0.0, 1.0);
            states.push(AgentState::new(
                AgentId(id_counter),
                AgentType::MarketMaker,
                config.retail_capital * 50.0,
                cooperation,
            ));
            id_counter += 1;
        }

        states
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_state() {
        let mut state = AgentState::new(AgentId(1), AgentType::Retail, 10000.0, 0.5);

        assert_eq!(state.position, 0.0);
        assert_eq!(state.pnl, 0.0);

        state.execute_trade(100.0, 10.0, true);
        assert_eq!(state.position, 100.0);
        assert_eq!(state.capital, 9000.0);

        state.execute_trade(50.0, 11.0, false);
        assert_eq!(state.position, 50.0);
        assert_eq!(state.capital, 9550.0);
    }

    #[test]
    fn test_agent_factory() {
        let config = AgentConfig::default();
        let rng = SimRng::from_seed(42);
        let states = AgentFactory::create_agent_states(&config, 100, rng);

        assert_eq!(states.len(), 100);

        let retail_count = states.iter().filter(|s| s.agent_type == AgentType::Retail).count();
        let expected_retail = (100.0 * config.retail_fraction) as usize;
        assert_eq!(retail_count, expected_retail);
    }

    #[test]
    fn test_retail_agent() {
        let mut agent = RetailAgent::new(AgentId(1), 10000.0, 0.5);
        let mut book = OrderBook::new(0.01);
        book.initialize_liquidity(100.0, 10, 100.0);

        let mut rng = SimRng::from_seed(42);

        // Generate orders over multiple timesteps
        let mut total_orders = 0;
        for t in 0..100 {
            let orders = agent.generate_orders(&mut book, &mut rng, t);
            total_orders += orders.len();
        }

        // Should have some orders (not zero due to activity probability)
        assert!(total_orders > 0);
    }
}
