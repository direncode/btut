//! Order Book Implementation
//!
//! A high-performance limit order book with price-time priority matching.

use crate::agent::AgentId;
use ordered_float::OrderedFloat;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::collections::BTreeMap;
use tracing::{debug, trace};

/// Price type (uses OrderedFloat for proper ordering of f64)
pub type Price = OrderedFloat<f64>;

/// Quantity type
pub type Quantity = f64;

/// Unique order identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OrderId(pub u64);

impl OrderId {
    /// Create a new order ID
    pub fn new(id: u64) -> Self {
        Self(id)
    }
}

/// Order side (bid or ask)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderSide {
    /// Buy order (bid)
    Bid,
    /// Sell order (ask)
    Ask,
}

impl OrderSide {
    /// Get the opposite side
    pub fn opposite(&self) -> Self {
        match self {
            OrderSide::Bid => OrderSide::Ask,
            OrderSide::Ask => OrderSide::Bid,
        }
    }
}

/// Order type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderType {
    /// Limit order at specified price
    Limit,
    /// Market order (execute immediately at best price)
    Market,
    /// Immediate-or-cancel (fill what you can, cancel rest)
    IOC,
}

/// A single order in the book
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    /// Unique order identifier
    pub id: OrderId,
    /// Agent that placed the order
    pub agent_id: AgentId,
    /// Order side
    pub side: OrderSide,
    /// Order type
    pub order_type: OrderType,
    /// Limit price (ignored for market orders)
    pub price: Price,
    /// Original quantity
    pub quantity: Quantity,
    /// Remaining quantity
    pub remaining: Quantity,
    /// Timestamp (simulation tick)
    pub timestamp: u64,
}

impl Order {
    /// Create a new limit order
    pub fn limit(
        id: OrderId,
        agent_id: AgentId,
        side: OrderSide,
        price: f64,
        quantity: Quantity,
        timestamp: u64,
    ) -> Self {
        Self {
            id,
            agent_id,
            side,
            order_type: OrderType::Limit,
            price: OrderedFloat(price),
            quantity,
            remaining: quantity,
            timestamp,
        }
    }

    /// Create a new market order
    pub fn market(
        id: OrderId,
        agent_id: AgentId,
        side: OrderSide,
        quantity: Quantity,
        timestamp: u64,
    ) -> Self {
        Self {
            id,
            agent_id,
            side,
            order_type: OrderType::Market,
            price: OrderedFloat(0.0), // Price not used for market orders
            quantity,
            remaining: quantity,
            timestamp,
        }
    }

    /// Check if order is fully filled
    pub fn is_filled(&self) -> bool {
        self.remaining <= 0.0
    }
}

/// A single price level in the order book
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PriceLevel {
    /// Total quantity at this level
    pub total_quantity: Quantity,
    /// Orders at this level (FIFO queue)
    pub orders: SmallVec<[Order; 8]>,
}

impl PriceLevel {
    /// Create a new empty price level
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an order to this level
    pub fn add_order(&mut self, order: Order) {
        self.total_quantity += order.remaining;
        self.orders.push(order);
    }

    /// Remove an order by ID, returns the removed order if found
    pub fn remove_order(&mut self, order_id: OrderId) -> Option<Order> {
        if let Some(pos) = self.orders.iter().position(|o| o.id == order_id) {
            let order = self.orders.remove(pos);
            self.total_quantity -= order.remaining;
            Some(order)
        } else {
            None
        }
    }

    /// Check if level is empty
    pub fn is_empty(&self) -> bool {
        self.orders.is_empty()
    }

    /// Get number of orders at this level
    pub fn num_orders(&self) -> usize {
        self.orders.len()
    }
}

/// Executed trade
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Aggressor order ID
    pub aggressor_id: OrderId,
    /// Passive order ID
    pub passive_id: OrderId,
    /// Aggressor agent
    pub aggressor_agent: AgentId,
    /// Passive agent
    pub passive_agent: AgentId,
    /// Trade price
    pub price: f64,
    /// Trade quantity
    pub quantity: Quantity,
    /// Timestamp
    pub timestamp: u64,
    /// Side of aggressor
    pub aggressor_side: OrderSide,
}

/// Result of order execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Execution {
    /// The order that was submitted
    pub order: Order,
    /// Trades that occurred
    pub trades: Vec<Trade>,
    /// Remaining quantity (0 if fully filled or cancelled)
    pub remaining: Quantity,
    /// Was the order added to the book?
    pub added_to_book: bool,
}

/// Order book snapshot for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BookSnapshot {
    /// Bid levels (price, total quantity)
    pub bids: Vec<(f64, Quantity)>,
    /// Ask levels (price, total quantity)
    pub asks: Vec<(f64, Quantity)>,
    /// Best bid price
    pub best_bid: Option<f64>,
    /// Best ask price
    pub best_ask: Option<f64>,
    /// Mid price
    pub mid_price: Option<f64>,
    /// Spread
    pub spread: Option<f64>,
    /// Total bid volume
    pub total_bid_volume: Quantity,
    /// Total ask volume
    pub total_ask_volume: Quantity,
    /// Timestamp
    pub timestamp: u64,
}

/// Central Limit Order Book
///
/// Implements a price-time priority order book with efficient matching.
pub struct OrderBook {
    /// Bid side (sorted by price descending - best bid first)
    bids: BTreeMap<Price, PriceLevel>,
    /// Ask side (sorted by price ascending - best ask first)
    asks: BTreeMap<Price, PriceLevel>,
    /// Order lookup by ID
    orders: FxHashMap<OrderId, (OrderSide, Price)>,
    /// Tick size (minimum price increment)
    tick_size: f64,
    /// Current timestamp
    current_timestamp: u64,
    /// Next order ID
    next_order_id: u64,
    /// Trade history (recent trades)
    recent_trades: Vec<Trade>,
    /// Maximum trades to keep in history
    max_trade_history: usize,
    /// Total volume traded
    total_volume: Quantity,
    /// Number of trades
    num_trades: u64,
}

impl OrderBook {
    /// Create a new order book
    pub fn new(tick_size: f64) -> Self {
        Self {
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            orders: FxHashMap::default(),
            tick_size,
            current_timestamp: 0,
            next_order_id: 1,
            recent_trades: Vec::with_capacity(1000),
            max_trade_history: 1000,
            total_volume: 0.0,
            num_trades: 0,
        }
    }

    /// Generate next order ID
    pub fn next_order_id(&mut self) -> OrderId {
        let id = OrderId(self.next_order_id);
        self.next_order_id += 1;
        id
    }

    /// Set current timestamp
    pub fn set_timestamp(&mut self, timestamp: u64) {
        self.current_timestamp = timestamp;
    }

    /// Get current timestamp
    pub fn timestamp(&self) -> u64 {
        self.current_timestamp
    }

    /// Round price to tick size
    pub fn round_price(&self, price: f64) -> f64 {
        (price / self.tick_size).round() * self.tick_size
    }

    /// Submit an order for execution
    pub fn submit_order(&mut self, mut order: Order) -> Execution {
        order.price = OrderedFloat(self.round_price(order.price.0));
        order.timestamp = self.current_timestamp;

        trace!(
            order_id = ?order.id,
            side = ?order.side,
            price = %order.price,
            quantity = order.quantity,
            "Order submitted"
        );

        let mut trades = Vec::new();

        // Try to match against opposite side
        match order.side {
            OrderSide::Bid => {
                self.match_bid(&mut order, &mut trades);
            }
            OrderSide::Ask => {
                self.match_ask(&mut order, &mut trades);
            }
        }

        let remaining = order.remaining;
        let added_to_book = if order.remaining > 0.0 && order.order_type == OrderType::Limit {
            self.add_to_book(order.clone());
            true
        } else {
            false
        };

        Execution {
            order,
            trades,
            remaining,
            added_to_book,
        }
    }

    /// Match a bid order against asks
    fn match_bid(&mut self, order: &mut Order, trades: &mut Vec<Trade>) {
        // Collect prices to process (ascending order - best ask first)
        let prices_to_check: Vec<Price> = self.asks.keys().cloned().collect();

        for ask_price in prices_to_check {
            // For limit orders, stop if we've passed the limit price
            if order.order_type == OrderType::Limit && ask_price > order.price {
                break;
            }

            if order.remaining <= 0.0 {
                break;
            }

            // Process matching at this price level
            let trade_price = ask_price.0;
            let timestamp = self.current_timestamp;

            if let Some(level) = self.asks.get_mut(&ask_price) {
                let mut i = 0;
                while i < level.orders.len() && order.remaining > 0.0 {
                    let passive = &mut level.orders[i];
                    let fill_qty = order.remaining.min(passive.remaining);

                    if fill_qty > 0.0 {
                        let trade = Trade {
                            aggressor_id: order.id,
                            passive_id: passive.id,
                            aggressor_agent: order.agent_id,
                            passive_agent: passive.agent_id,
                            price: trade_price,
                            quantity: fill_qty,
                            timestamp,
                            aggressor_side: order.side,
                        };

                        order.remaining -= fill_qty;
                        passive.remaining -= fill_qty;
                        level.total_quantity -= fill_qty;
                        trades.push(trade);
                    }

                    if passive.remaining <= 0.0 {
                        level.orders.remove(i);
                    } else {
                        i += 1;
                    }
                }
            }
        }

        // Update stats and cleanup
        for trade in trades.iter() {
            self.total_volume += trade.quantity;
            self.num_trades += 1;
            if self.recent_trades.len() >= self.max_trade_history {
                self.recent_trades.remove(0);
            }
            self.recent_trades.push(trade.clone());
        }

        // Remove filled orders from lookup and empty price levels
        self.asks.retain(|_, level| !level.is_empty());
    }

    /// Match an ask order against bids
    fn match_ask(&mut self, order: &mut Order, trades: &mut Vec<Trade>) {
        // Collect prices to process (descending order - best bid first)
        let prices_to_check: Vec<Price> = self.bids.keys().rev().cloned().collect();

        for bid_price in prices_to_check {
            // For limit orders, stop if we've passed the limit price
            if order.order_type == OrderType::Limit && bid_price < order.price {
                break;
            }

            if order.remaining <= 0.0 {
                break;
            }

            // Process matching at this price level
            let trade_price = bid_price.0;
            let timestamp = self.current_timestamp;

            if let Some(level) = self.bids.get_mut(&bid_price) {
                let mut i = 0;
                while i < level.orders.len() && order.remaining > 0.0 {
                    let passive = &mut level.orders[i];
                    let fill_qty = order.remaining.min(passive.remaining);

                    if fill_qty > 0.0 {
                        let trade = Trade {
                            aggressor_id: order.id,
                            passive_id: passive.id,
                            aggressor_agent: order.agent_id,
                            passive_agent: passive.agent_id,
                            price: trade_price,
                            quantity: fill_qty,
                            timestamp,
                            aggressor_side: order.side,
                        };

                        order.remaining -= fill_qty;
                        passive.remaining -= fill_qty;
                        level.total_quantity -= fill_qty;
                        trades.push(trade);
                    }

                    if passive.remaining <= 0.0 {
                        level.orders.remove(i);
                    } else {
                        i += 1;
                    }
                }
            }
        }

        // Update stats and cleanup
        for trade in trades.iter() {
            self.total_volume += trade.quantity;
            self.num_trades += 1;
            if self.recent_trades.len() >= self.max_trade_history {
                self.recent_trades.remove(0);
            }
            self.recent_trades.push(trade.clone());
        }

        // Remove empty price levels
        self.bids.retain(|_, level| !level.is_empty());
    }

    /// Add an order to the book (no matching)
    fn add_to_book(&mut self, order: Order) {
        let price = order.price;
        let side = order.side;
        let id = order.id;

        self.orders.insert(id, (side, price));

        match side {
            OrderSide::Bid => {
                self.bids.entry(price).or_default().add_order(order);
            }
            OrderSide::Ask => {
                self.asks.entry(price).or_default().add_order(order);
            }
        }

        trace!(order_id = ?id, "Order added to book");
    }

    /// Cancel an order by ID
    pub fn cancel_order(&mut self, order_id: OrderId) -> Option<Order> {
        if let Some((side, price)) = self.orders.remove(&order_id) {
            let book = match side {
                OrderSide::Bid => &mut self.bids,
                OrderSide::Ask => &mut self.asks,
            };

            if let Some(level) = book.get_mut(&price) {
                let order = level.remove_order(order_id);
                if level.is_empty() {
                    book.remove(&price);
                }
                debug!(order_id = ?order_id, "Order cancelled");
                return order;
            }
        }
        None
    }

    /// Get best bid price
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.keys().next_back().map(|p| p.0)
    }

    /// Get best ask price
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.keys().next().map(|p| p.0)
    }

    /// Get mid price
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid + ask) / 2.0),
            (Some(bid), None) => Some(bid),
            (None, Some(ask)) => Some(ask),
            (None, None) => None,
        }
    }

    /// Get bid-ask spread
    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Get spread as percentage of mid price
    pub fn spread_bps(&self) -> Option<f64> {
        match (self.spread(), self.mid_price()) {
            (Some(spread), Some(mid)) if mid > 0.0 => Some(spread / mid * 10000.0),
            _ => None,
        }
    }

    /// Get total bid volume
    pub fn total_bid_volume(&self) -> Quantity {
        self.bids.values().map(|l| l.total_quantity).sum()
    }

    /// Get total ask volume
    pub fn total_ask_volume(&self) -> Quantity {
        self.asks.values().map(|l| l.total_quantity).sum()
    }

    /// Get book imbalance (-1 = all asks, +1 = all bids)
    pub fn imbalance(&self) -> f64 {
        let bid_vol = self.total_bid_volume();
        let ask_vol = self.total_ask_volume();
        let total = bid_vol + ask_vol;
        if total > 0.0 {
            (bid_vol - ask_vol) / total
        } else {
            0.0
        }
    }

    /// Get depth at N levels
    pub fn depth(&self, levels: usize) -> (Vec<(f64, Quantity)>, Vec<(f64, Quantity)>) {
        let bids: Vec<(f64, Quantity)> = self
            .bids
            .iter()
            .rev()
            .take(levels)
            .map(|(p, l)| (p.0, l.total_quantity))
            .collect();

        let asks: Vec<(f64, Quantity)> = self
            .asks
            .iter()
            .take(levels)
            .map(|(p, l)| (p.0, l.total_quantity))
            .collect();

        (bids, asks)
    }

    /// Get a snapshot of the current book state
    pub fn snapshot(&self, depth: usize) -> BookSnapshot {
        let (bids, asks) = self.depth(depth);

        BookSnapshot {
            bids,
            asks,
            best_bid: self.best_bid(),
            best_ask: self.best_ask(),
            mid_price: self.mid_price(),
            spread: self.spread(),
            total_bid_volume: self.total_bid_volume(),
            total_ask_volume: self.total_ask_volume(),
            timestamp: self.current_timestamp,
        }
    }

    /// Get recent trades
    pub fn recent_trades(&self) -> &[Trade] {
        &self.recent_trades
    }

    /// Get total traded volume
    pub fn total_volume(&self) -> Quantity {
        self.total_volume
    }

    /// Get total number of trades
    pub fn num_trades(&self) -> u64 {
        self.num_trades
    }

    /// Get number of orders in the book
    pub fn num_orders(&self) -> usize {
        self.orders.len()
    }

    /// Get number of bid levels
    pub fn num_bid_levels(&self) -> usize {
        self.bids.len()
    }

    /// Get number of ask levels
    pub fn num_ask_levels(&self) -> usize {
        self.asks.len()
    }

    /// Clear all orders (reset book)
    pub fn clear(&mut self) {
        self.bids.clear();
        self.asks.clear();
        self.orders.clear();
        self.recent_trades.clear();
    }

    /// Initialize book with synthetic liquidity
    pub fn initialize_liquidity(&mut self, mid_price: f64, depth: usize, base_quantity: Quantity) {
        for i in 1..=depth {
            let spread = i as f64 * self.tick_size;
            let qty = base_quantity * (-0.1 * i as f64).exp();

            // Add bid
            let bid_order = Order::limit(
                self.next_order_id(),
                AgentId(0), // System agent
                OrderSide::Bid,
                mid_price - spread,
                qty,
                self.current_timestamp,
            );
            self.add_to_book(bid_order);

            // Add ask
            let ask_order = Order::limit(
                self.next_order_id(),
                AgentId(0),
                OrderSide::Ask,
                mid_price + spread,
                qty,
                self.current_timestamp,
            );
            self.add_to_book(ask_order);
        }

        debug!(
            mid_price = mid_price,
            depth = depth,
            "Order book initialized"
        );
    }

    /// Get volume-weighted average price (VWAP) for recent trades
    pub fn vwap(&self, num_trades: usize) -> Option<f64> {
        let trades = if num_trades >= self.recent_trades.len() {
            &self.recent_trades[..]
        } else {
            &self.recent_trades[self.recent_trades.len() - num_trades..]
        };

        if trades.is_empty() {
            return None;
        }

        let (volume_sum, value_sum) =
            trades
                .iter()
                .fold((0.0, 0.0), |(vol, val), t| {
                    (vol + t.quantity, val + t.price * t.quantity)
                });

        if volume_sum > 0.0 {
            Some(value_sum / volume_sum)
        } else {
            None
        }
    }

    /// Get last trade price
    pub fn last_price(&self) -> Option<f64> {
        self.recent_trades.last().map(|t| t.price)
    }
}

impl std::fmt::Debug for OrderBook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrderBook")
            .field("best_bid", &self.best_bid())
            .field("best_ask", &self.best_ask())
            .field("spread", &self.spread())
            .field("num_orders", &self.num_orders())
            .field("num_trades", &self.num_trades)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_book_basic() {
        let mut book = OrderBook::new(0.01);

        // Add bid
        let bid = Order::limit(
            OrderId(1),
            AgentId(1),
            OrderSide::Bid,
            100.0,
            10.0,
            0,
        );
        let exec = book.submit_order(bid);
        assert!(exec.added_to_book);
        assert!(exec.trades.is_empty());

        assert_eq!(book.best_bid(), Some(100.0));
        assert_eq!(book.best_ask(), None);
    }

    #[test]
    fn test_order_matching() {
        let mut book = OrderBook::new(0.01);

        // Add bid
        let bid = Order::limit(
            OrderId(1),
            AgentId(1),
            OrderSide::Bid,
            100.0,
            10.0,
            0,
        );
        book.submit_order(bid);

        // Add crossing ask (should match)
        let ask = Order::limit(
            OrderId(2),
            AgentId(2),
            OrderSide::Ask,
            100.0,
            5.0,
            1,
        );
        let exec = book.submit_order(ask);

        assert_eq!(exec.trades.len(), 1);
        assert_eq!(exec.trades[0].quantity, 5.0);
        assert_eq!(exec.trades[0].price, 100.0);
        assert_eq!(exec.remaining, 0.0);

        // Remaining bid should be 5.0
        assert_eq!(book.total_bid_volume(), 5.0);
    }

    #[test]
    fn test_market_order() {
        let mut book = OrderBook::new(0.01);

        // Add asks at different prices
        book.submit_order(Order::limit(
            OrderId(1),
            AgentId(1),
            OrderSide::Ask,
            100.0,
            10.0,
            0,
        ));
        book.submit_order(Order::limit(
            OrderId(2),
            AgentId(2),
            OrderSide::Ask,
            101.0,
            10.0,
            0,
        ));

        // Submit market buy
        let market = Order::market(
            OrderId(3),
            AgentId(3),
            OrderSide::Bid,
            15.0,
            1,
        );
        let exec = book.submit_order(market);

        assert_eq!(exec.trades.len(), 2);
        assert_eq!(exec.trades[0].quantity, 10.0);
        assert_eq!(exec.trades[0].price, 100.0);
        assert_eq!(exec.trades[1].quantity, 5.0);
        assert_eq!(exec.trades[1].price, 101.0);
    }

    #[test]
    fn test_cancel_order() {
        let mut book = OrderBook::new(0.01);

        let order = Order::limit(
            OrderId(1),
            AgentId(1),
            OrderSide::Bid,
            100.0,
            10.0,
            0,
        );
        book.submit_order(order);
        assert_eq!(book.num_orders(), 1);

        let cancelled = book.cancel_order(OrderId(1));
        assert!(cancelled.is_some());
        assert_eq!(book.num_orders(), 0);
    }

    #[test]
    fn test_spread_and_mid() {
        let mut book = OrderBook::new(0.01);

        book.submit_order(Order::limit(
            OrderId(1),
            AgentId(1),
            OrderSide::Bid,
            99.95,
            10.0,
            0,
        ));
        book.submit_order(Order::limit(
            OrderId(2),
            AgentId(2),
            OrderSide::Ask,
            100.05,
            10.0,
            0,
        ));

        assert!((book.spread().unwrap() - 0.10).abs() < 1e-10);
        assert!((book.mid_price().unwrap() - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_initialize_liquidity() {
        let mut book = OrderBook::new(0.01);
        book.initialize_liquidity(100.0, 10, 1000.0);

        assert!(book.num_bid_levels() > 0);
        assert!(book.num_ask_levels() > 0);
        assert!(book.best_bid().is_some());
        assert!(book.best_ask().is_some());
    }

    #[test]
    fn test_snapshot() {
        let mut book = OrderBook::new(0.01);
        book.initialize_liquidity(100.0, 5, 100.0);

        let snapshot = book.snapshot(10);
        assert!(snapshot.best_bid.is_some());
        assert!(snapshot.best_ask.is_some());
        assert!(!snapshot.bids.is_empty());
        assert!(!snapshot.asks.is_empty());
    }
}
